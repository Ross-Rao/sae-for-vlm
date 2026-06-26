import torch
import os.path
import argparse
from datasets.activations import ActivationsDataset
import os

from torch.utils.data import DataLoader, Subset
import tqdm
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser("Measure monosemanticity via weighted pairwise cosine similarity", add_help=False)
    parser.add_argument("--embeddings_path")
    parser.add_argument("--activations_dir")
    parser.add_argument("--output_subdir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--take_every", type=int, default=1, help="Subsample every N-th image to reduce O(n^2) cost")
    parser.add_argument("--min_support", type=int, default=10,
                        help="Minimum number of non-zero activating images required; neurons below this are set to NaN")
    return parser

def print_and_save_stats(monosemanticity, num_neurons, output_path, min_support):
    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum().item()
    valid = monosemanticity[~is_nan]
    mean = torch.mean(valid).item()
    std  = torch.std(valid).item()

    print(f"Monosemanticity: {mean} +- {std}")
    print(f"NaN neurons (dead + low-support): {nan_count}")
    print(f"Total neurons: {num_neurons}")

    valid_mask = ~is_nan
    valid_indices = torch.nonzero(valid_mask).squeeze()
    valid_scores  = monosemanticity[valid_mask]

    top_vals, top_local  = torch.topk(valid_scores, 10)
    bot_vals, bot_local  = torch.topk(valid_scores, 10, largest=False)
    top_idx = valid_indices[top_local]
    bot_idx = valid_indices[bot_local]

    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_idx, top_vals)):
        print(f"{i+1}. Neuron {idx.item()} - {val.item()}")
    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bot_idx, bot_vals)):
        print(f"{i+1}. Neuron {idx.item()} - {val.item()}")

    with open(output_path, "w") as f:
        f.write(f"Monosemanticity: {mean} +- {std}\n")
        f.write(f"NaN neurons (dead + low-support, min_support={min_support}): {nan_count}\n")
        f.write(f"Total neurons: {num_neurons}\n\n")
        f.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_idx, top_vals):
            f.write(f"Neuron {idx.item()} - {val.item()}\n")
        f.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bot_idx, bot_vals):
            f.write(f"Neuron {idx.item()} - {val.item()}\n")

def main(args):
    all_embeddings = torch.load(args.embeddings_path, map_location=torch.device("cpu"))
    print(f"Loaded embeddings found at {args.embeddings_path}")
    print(f"Embeddings shape: {all_embeddings.shape}")

    activations_dataset = ActivationsDataset(args.activations_dir, device=torch.device("cpu"), take_every=args.take_every)
    activations_dataloader = DataLoader(activations_dataset, batch_size=len(activations_dataset), shuffle=False)
    activations = next(iter(activations_dataloader))
    print(f"Loaded activations found at {args.activations_dir}")
    print(f"Activations shape: {activations.shape}")

    selected_indices = list(range(0, all_embeddings.shape[0], args.take_every))[:activations.shape[0]]
    embeddings = all_embeddings[selected_indices].to(args.device)
    print(f"Subsampled embeddings shape: {embeddings.shape}")

    # Count non-zero activating images per neuron BEFORE normalization
    nonzero_count = (activations > 0).sum(dim=0)  # [num_neurons]

    # Scale to 0-1 per neuron
    min_values = activations.min(dim=0, keepdim=True)[0]
    max_values = activations.max(dim=0, keepdim=True)[0]
    activations = (activations - min_values) / (max_values - min_values)

    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]

    weighted_cosine_similarity_sum = torch.zeros(num_neurons)
    weight_sum = torch.zeros(num_neurons)
    batch_size = 100

    for i in tqdm.tqdm(range(num_images), desc="Processing image pairs"):
        for j_start in range(i + 1, num_images, batch_size):
            j_end = min(j_start + batch_size, num_images)

            embeddings_i = embeddings[i].cuda()
            embeddings_j = embeddings[j_start:j_end].cuda()
            activations_i = activations[i].cuda()
            activations_j = activations[j_start:j_end].cuda()

            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),
                embeddings_j,
                dim=1
            )

            weights = activations_i.unsqueeze(0) * activations_j
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)

            weighted_cosine_similarity_sum += torch.sum(weighted_cosine_similarities, dim=0).cpu()
            weight_sum += torch.sum(weights, dim=0).cpu()

    # Base score: NaN for dead neurons (weight_sum == 0)
    monosemanticity = torch.where(
        weight_sum != 0,
        weighted_cosine_similarity_sum / weight_sum,
        torch.tensor(float("nan"))
    )

    # Apply min_support filter: neurons with too few activating images -> NaN
    low_support_mask = nonzero_count < args.min_support
    low_support_count = low_support_mask.sum().item()
    monosemanticity[low_support_mask] = float("nan")
    print(f"Neurons filtered by min_support={args.min_support}: {low_support_count}")

    os.makedirs(os.path.join(args.activations_dir, args.output_subdir), exist_ok=True)
    torch.save(monosemanticity, os.path.join(args.activations_dir, args.output_subdir, "all_neurons_scores.pth"))

    output_path = os.path.join(args.activations_dir, args.output_subdir, "metric_stats_new.txt")
    print_and_save_stats(monosemanticity, num_neurons, output_path, args.min_support)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
