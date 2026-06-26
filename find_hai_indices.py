import numpy as np
import tqdm
import os
import torch
from torchvision import transforms
from utils import get_dataset
import argparse
from datasets.activations import ActivationsDataset, ChunkedActivationsDataset
from torch.utils.data import DataLoader, Subset
from itertools import combinations
from collections import Counter
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Find indices of highest activating images from activations")
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--data_path", default="/shared-network/inat2021", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--chunk_size", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.batch_size = 1
    args.num_workers = 0

    hai_indices_path = os.path.join(args.activations_dir, f"hai_indices_{args.k}_nonzero")
    if os.path.exists(hai_indices_path + ".npy"):
        print(f"HAI indices already saved at {hai_indices_path}.npy")
    else:
        print("Computing HAI indices (non-zero filtered)", flush=True)
        print(f"Loading activations from {args.activations_dir}", flush=True)
        activations_dataset = ActivationsDataset(args.activations_dir, device=torch.device("cpu"))
        print(f"Dataset loaded. Total samples: {len(activations_dataset)}", flush=True)

        activations_dataloader = DataLoader(activations_dataset, batch_size=args.chunk_size, shuffle=False, num_workers=16)
        num_samples = len(activations_dataset)

        first_batch = next(iter(activations_dataloader))
        num_neurons = first_batch.shape[1]
        print(f"Number of neurons detected: {num_neurons}")
        num_chunks = math.ceil(num_neurons / args.chunk_size)
        print(f"Processing {num_chunks} chunks of {args.chunk_size} neurons each...", flush=True)

        importants = []
        worst_hais = []
        zero_count = 0
        sparse_count = 0
        pbar = tqdm.tqdm(list(range(num_chunks)))
        for i in pbar:
            neuron_start = i * args.chunk_size
            neuron_end = min((i + 1) * args.chunk_size, num_neurons)
            activations_chunks = np.zeros((num_samples, neuron_end - neuron_start))
            for j, activations_chunk in enumerate(activations_dataloader):
                sample_start = j * args.chunk_size
                sample_end = min((j + 1) * args.chunk_size, num_samples)
                activations_chunk = activations_chunk.numpy()
                activations_chunks[sample_start:sample_end, :] = activations_chunk[:, neuron_start:neuron_end]
            for neuron in range(neuron_end - neuron_start):
                neuron_activations = activations_chunks[:, neuron]
                nonzero_mask = neuron_activations > 0
                nonzero_count = int(nonzero_mask.sum())
                if nonzero_count == 0:
                    # Dead neuron
                    important = np.array([], dtype=np.int64)
                    worst_hai = 0.0
                    zero_count += 1
                elif nonzero_count <= args.k:
                    # Fewer than k non-zero images: take all, sorted ascending by activation
                    nonzero_idx = np.where(nonzero_mask)[0]
                    important = nonzero_idx[np.argsort(neuron_activations[nonzero_idx])]
                    worst_hai = neuron_activations[important[0]]
                    sparse_count += 1
                else:
                    # Enough non-zero images: take top k (all will be > 0)
                    important = np.argsort(neuron_activations)[-args.k:]
                    worst_hai = neuron_activations[important[0]]
                importants.append(important)
                worst_hais.append(worst_hai)

        print(f"Dead neurons (0 activations): {zero_count}")
        print(f"Sparse neurons (<{args.k} non-zero): {sparse_count}")
        print(f"Normal neurons (>={args.k} non-zero): {num_neurons - zero_count - sparse_count}")

        # Save as object array to support variable-length rows
        hai_indices = np.empty(len(importants), dtype=object)
        for i, imp in enumerate(importants):
            hai_indices[i] = imp
        np.save(hai_indices_path, hai_indices)
        print(f"Saved HAI indices to: {hai_indices_path}.npy")

        worst_hai_indices_path = os.path.join(args.activations_dir, f"hai_indices_{args.k}_nonzero_worst")
        worst_hais = np.array(worst_hais)
        np.save(worst_hai_indices_path, worst_hais)
        print(f"Saved worst HAI to: {worst_hai_indices_path}.npy")
