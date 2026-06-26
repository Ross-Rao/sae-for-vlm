import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import math
from torchvision import transforms
from utils import get_dataset
import argparse
from math import isclose
import torch


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols, "Number of images must match rows * cols."
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def make_grid(images, cell_size=224):
    """Build a grid image from a variable-length list. Pads with gray if needed."""
    n = len(images)
    if n == 0:
        return None
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    # Pad to rows*cols with gray images
    gray = Image.new("RGB", (cell_size, cell_size), (128, 128, 128))
    while len(images) < rows * cols:
        images.append(gray)
    return image_grid(images, rows=rows, cols=cols)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize top-k activating images for neurons.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--visualization_size", type=int, default=224)
    parser.add_argument("--group_fractions", type=float, nargs="+")
    parser.add_argument("--hai_indices_path", type=str)
    parser.add_argument("--ms_scores_path", type=str, default=None,
                        help="Path to all_neurons_scores.pth from metric.py.")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of neurons to visualize per selection when ms_scores_path is set.")
    return parser.parse_args()


def get_group_info(absolute_id, group_sizes):
    start = 0
    for g, size in enumerate(group_sizes):
        if absolute_id < start + size:
            return g, absolute_id - start
        start += size
    return len(group_sizes) - 1, absolute_id - start


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = 1
    args.num_workers = 0

    # Support both fixed-size arrays and object arrays (nonzero-filtered)
    importants = np.load(args.hai_indices_path, allow_pickle=True)
    is_object_array = importants.dtype == object
    print(f"Loaded HAI indices from {args.hai_indices_path} (object_array={is_object_array})", flush=True)
    num_neurons = importants.shape[0]

    def _convert_to_rgb(image):
        return image.convert("RGB")

    visualization_preprocess = transforms.Compose([
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ])

    ds, dl = get_dataset(args, preprocess=visualization_preprocess, processor=None, split=args.split, subset=1)

    os.makedirs(os.path.join(args.output_dir, "hai"), exist_ok=True)

    assert isclose(sum(args.group_fractions), 1.0), "group_fractions must sum to 1.0"
    group_sizes = [int(f * num_neurons) for f in args.group_fractions[:-1]]
    group_sizes.append(num_neurons - sum(group_sizes))

    if args.ms_scores_path is not None:
        scores = torch.load(args.ms_scores_path, map_location="cpu")
        valid_mask = ~torch.isnan(scores)
        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        valid_scores = scores[valid_mask]

        n = args.top_n
        sorted_by_score = torch.argsort(valid_scores, descending=True)
        top_local = sorted_by_score[:n]
        bottom_local = sorted_by_score[-n:]
        mean_score = valid_scores.mean()
        dist_from_mean = (valid_scores - mean_score).abs()
        mid_local = torch.argsort(dist_from_mean)[:n]

        selected = []
        seen = set()
        for tag, local_ids in [("top", top_local), ("bottom", bottom_local), ("mid", mid_local)]:
            for rank, li in enumerate(local_ids):
                abs_id = valid_indices[li].item()
                if abs_id in seen:
                    continue
                seen.add(abs_id)
                selected.append((tag, rank + 1, abs_id, valid_scores[li].item()))

        output_subdir = os.path.join(args.output_dir, "ms_selected")
        os.makedirs(output_subdir, exist_ok=True)

        for tag, rank, abs_id, score in selected:
            g, local_id = get_group_info(abs_id, group_sizes)
            important = importants[abs_id]
            n_imgs = len(important)
            label = f"[{tag}#{rank}] neuron {abs_id} (group {g}, local {local_id}) MS={score:.4f} n_hai={n_imgs}"
            print(label, flush=True)

            if n_imgs == 0:
                print("  -> skip: no non-zero activating images", flush=True)
                continue

            images = [ds[i][0] for i in important]
            grid_image = make_grid(list(reversed(images)))

            plt.imshow(grid_image)
            plt.axis("off")
            plt.title(f"MS={score:.4f}  n={n_imgs}", fontsize=8)
            filename = f"{tag}{rank}_neuron{abs_id}_ms{score:.4f}_n{n_imgs}_group{g}_gneuron{local_id}.png"
            plt.savefig(os.path.join(output_subdir, filename), bbox_inches="tight", pad_inches=0.05)
            plt.close()

    else:
        start_idx = 0
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            group_neurons = range(start_idx, end_idx)
            for neuron_id, absolute_id in enumerate(group_neurons[:5000]):
                print(f"Visualizing neuron {neuron_id} (absolute {absolute_id}) in group {group_idx}", flush=True)
                important = importants[absolute_id]
                n_imgs = len(important)
                if n_imgs == 0:
                    continue
                images = [ds[i][0] for i in important]
                grid_image = make_grid(list(reversed(images)))
                plt.imshow(grid_image)
                plt.axis("off")
                filename = f"group_{group_idx}_neuron_{neuron_id}_absolute_{absolute_id}.png"
                output_path = os.path.join(args.output_dir, "tree", filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
                plt.close()
            start_idx = end_idx
