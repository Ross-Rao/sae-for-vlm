import numpy as np
import tqdm
import os
import glob
import torch
import torch.nn.functional as F
from models.llava import Llava
from utils import IdentitySAE, get_text_model
import argparse
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE
from PIL import Image
import random
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="Compute CLIP-based score for steering accuracy")
    parser.add_argument('--hai_indices_path', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--sae_path', type=str, default=None)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pre_zero", action=argparse.BooleanOptionalAction)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--neuron_prefix', type=int, default=None)
    parser.add_argument("--steer", action=argparse.BooleanOptionalAction)
    parser.add_argument("--vlm_backend", type=str, default="llava", choices=["llava", "llava_med", "med_flamingo", "chexagent"])
    parser.add_argument("--max_images", type=int, default=200, help="Max images to sample for steering eval")
    return parser.parse_args()


def load_env_key(env_path):
    """Read DEEPSEEK_API_KEY from a .env file."""
    if not os.path.exists(env_path):
        return None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def extract_keywords_batch(sentences, api_key, batch_size=100):
    """
    Call DeepSeek API to extract the single most important visual/medical
    keyword from each sentence.  Returns a list of keywords in the same order.
    Uses a cache to avoid duplicate API calls.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Cache: sentence -> keyword
    cache = {}
    unique_sentences = list(dict.fromkeys(sentences))  # preserve order, deduplicate

    print(f"Extracting keywords for {len(unique_sentences)} unique sentences "
          f"({len(sentences)} total) via DeepSeek API ...")

    for i in tqdm.tqdm(range(0, len(unique_sentences), batch_size), desc="DeepSeek API batches"):
        batch = unique_sentences[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch))
        user_prompt = (
            "Below are descriptions of medical images. "
            "For each numbered sentence, extract the single most important "
            "medical or visual keyword (e.g. 'X-ray', 'tumor', 'CT', 'lesion', 'MRI'). "
            "Reply with exactly one keyword per line, numbered to match, nothing else.\n\n"
            + numbered
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a medical image analysis assistant. "
                 "Extract the single most important keyword from each description."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=batch_size * 10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().split("\n")
        for idx, line in enumerate(raw):
            if idx >= len(batch):
                break
            kw = line.strip()
            # Strip leading "1. " numbering if present
            if kw and kw[0].isdigit():
                kw = kw.split(".", 1)[-1].strip()
            cache[batch[idx]] = kw if kw else "unknown"

    # Fill any sentences that somehow didn't get a keyword
    for s in unique_sentences:
        if s not in cache:
            cache[s] = "unknown"

    return [cache[s] for s in sentences]


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = 1
    args.num_workers = 0

    # Load DeepSeek API key from .env in project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    deepseek_api_key = load_env_key(os.path.join(project_dir, ".env"))
    if deepseek_api_key is None:
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key is None:
        raise RuntimeError("DEEPSEEK_API_KEY not found in .env or environment")
    print("DeepSeek API key loaded.")

    # Get HAI indices
    _raw_hai = np.load(args.hai_indices_path, allow_pickle=True)
    _k = max((len(x) for x in _raw_hai if len(x) > 0), default=16)
    def _pad(x, k):
        if len(x) == 0: return np.zeros(k, dtype=np.int64)
        return np.pad(x, (0, max(0, k - len(x))), mode='edge')[:k]
    hai_indices = torch.from_numpy(np.stack([_pad(x, _k) for x in _raw_hai])).to(args.device)
    print(f"Loaded HAI indices found at {args.hai_indices_path}")
    print(f"hai_indices shape: {hai_indices.shape}")

    # Get image embeddings
    embeddings = torch.load(args.embeddings_path).to(args.device)
    print(f"Loaded embeddings found at {args.embeddings_path}")
    print(f"embeddings shape: {embeddings.shape}")

    # Compute mean image embedding of HAI per neuron
    hai_embeddings = embeddings[hai_indices]
    print(f"hai_embeddings shape: {hai_embeddings.shape}")
    hai_embeddings = hai_embeddings.mean(dim=1)
    print(f"hai_embeddings shape: {hai_embeddings.shape}")

    # Load VLM model
    if args.vlm_backend == 'llava_med':
        from models.llava_med import LlavaMed
        llava = LlavaMed(args.device)
    elif args.vlm_backend == "med_flamingo":
        from models.med_flamingo import MedFlamingo
        llava = MedFlamingo(args.device)
    elif args.vlm_backend == "chexagent":
        from models.chexagent import CheXAgent
        llava = CheXAgent(args.device)
    else:
        llava = Llava(args.device)
    if args.sae_path:
        sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path).to(args.device)
        print(f"Attached SAE from {args.sae_path}")
    else:
        sae = IdentitySAE()
        print(f"Attached Identity SAE (scoring original neurons)")

    # Filter neurons
    num_neurons = hai_embeddings.shape[0]
    if args.neuron_prefix:
        neuron_indices = list(range(args.neuron_prefix))
    else:
        neuron_indices = list(range(num_neurons))
    print(f"Evaluating on {len(neuron_indices)} neurons")

    # Find images recursively
    all_image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPEG'):
        all_image_paths.extend(glob.glob(os.path.join(args.images_path, '**', ext), recursive=True))
    if args.max_images and len(all_image_paths) > args.max_images:
        random.seed(42)
        all_image_paths = random.sample(all_image_paths, args.max_images)
    image_files = all_image_paths
    print(f"Found {len(image_files)} images in {args.images_path} (max_images={args.max_images})")

    # VLM inference — save full sentences
    text = "What is shown in this image? Describe in one short phrase."
    raw_responses = {neuron: [] for neuron in neuron_indices}  # (image_file, full_sentence)
    if args.steer:
        print("Steering")
    else:
        print("Not steering")

    for neuron in tqdm.tqdm(neuron_indices, desc="Processing neurons"):
        if args.steer:
            llava.attach_and_fix(sae=sae, neurons_to_fix={neuron: 100}, pre_zero=args.pre_zero)
        for image_path in image_files:
            image = Image.open(image_path)
            response = llava.prompt(text, image, max_tokens=30)[0].strip()
            raw_responses[neuron].append((os.path.basename(image_path), response))

    # Save raw sentences to labels_raw.txt
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)
    raw_labels_path = os.path.join(out_dir, "labels_raw.txt")
    with open(raw_labels_path, "w") as f:
        for neuron, items in raw_responses.items():
            for image_file, sentence in items:
                f.write(f"{neuron},{image_file},{sentence}\n")
    print(f"Raw sentences saved to {raw_labels_path}")

    # Extract keywords via DeepSeek API
    all_sentences = [sentence for neuron in neuron_indices for _, sentence in raw_responses[neuron]]
    all_keywords = extract_keywords_batch(all_sentences, deepseek_api_key, batch_size=100)

    # Rebuild labels dict with keywords
    labels = {neuron: [] for neuron in neuron_indices}
    idx = 0
    for neuron in neuron_indices:
        for image_file, _ in raw_responses[neuron]:
            labels[neuron].append((image_file, all_keywords[idx]))
            idx += 1

    # Save keyword labels to labels.txt
    labels_path = os.path.join(out_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for neuron, image_labels in labels.items():
            for image_file, keyword in image_labels:
                f.write(f"{neuron},{image_file},{keyword}\n")
    print(f"Keyword labels saved to {labels_path}")

    # Compute text embeddings for keywords
    text_encoder, tokenizer = get_text_model(args)
    label_embeddings = torch.zeros(len(neuron_indices), len(image_files), hai_embeddings.shape[1]).to(args.device)
    for i, (neuron, image_labels) in tqdm.tqdm(enumerate(labels.items()), desc="Computing text embeddings"):
        for j, label in enumerate(image_labels):
            with torch.no_grad():
                inputs = tokenizer([label[1]], padding=True, return_tensors="pt").to(args.device)
                outputs = text_encoder(**inputs)
                label_embeddings[i, j] = outputs.text_embeds

    # Compute cosine similarities
    cosine_similarities = []
    for i in range(label_embeddings.shape[1]):
        cosine_similarities.append(F.cosine_similarity(hai_embeddings[neuron_indices], label_embeddings[:, i], dim=1))
    cosine_similarities = torch.cat(cosine_similarities)
    print(cosine_similarities.shape)
    torch.save(cosine_similarities, os.path.join(out_dir, "scores_per_neuron"))
    mean_cosine_similarity = cosine_similarities.mean().item()
    std_cosine_similarity = cosine_similarities.std().item()

    print("Mean Cosine Similarity:", mean_cosine_similarity)
    print("Standard Deviation Cosine Similarity:", std_cosine_similarity)

    with open(os.path.join(out_dir, "metric.txt"), "w") as f:
        f.write(f"Mean Cosine Similarity: {mean_cosine_similarity}\n")
        f.write(f"Standard Deviation Cosine Similarity: {std_cosine_similarity}\n")

print("Done")
