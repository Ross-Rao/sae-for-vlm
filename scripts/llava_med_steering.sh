#!/bin/bash
# SAE steering experiment for LLaVA-Med v1.5 on OmniMedVQA
# Vision encoder: CLIP ViT-L/14-336 (loaded from LLaVA-Med checkpoint)
#
# Pipeline:
#   1. save_activations  — collect raw vision tower activations
#   2. sae_train         — train Matryoshka SAE offline
#   3. save_activations  — re-run with SAE to get mean-pooled neuron activations
#   4. find_hai_indices  — find top-16 activating images per neuron
#   5. visualize_neurons — render neuron visualizations
#   6. encode_images     — encode OmniMedVQA images with CLIP-base for scoring
#   7. steering_score    — measure steering accuracy (with / without SAE)

set -e  # exit on error

PROJECT_DIR="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm"
DATASET_PATH="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/datasets/OmniMedVQA/OmniMedVQA"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

LAYER=23
MODEL=llava_med
DATASET=omnimed
SAE_MODEL=matroyshka_batch_top_k
K=20
EXPANSION=64

RAW_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/raw/random_k_2/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
CKPT_DIR="${PROJECT_DIR}/checkpoints_dir/${MODEL}/${SAE_MODEL}_${K}_x${EXPANSION}/random_k_2"
SAE_PATH="${CKPT_DIR}/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual_${SAE_MODEL}_${K}_x${EXPANSION}/trainer_0/checkpoints/ae_100000.pt"
MEAN_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/${SAE_MODEL}_${K}_x${EXPANSION}/mean_pool/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
EMBED_DIR="${PROJECT_DIR}/embeddings_dir"
RESULTS_DIR="${PROJECT_DIR}/llava_med_results_dir/${DATASET}"

cd "$PROJECT_DIR"

# ── Step 1: Save raw activations ────────────────────────────────────────────
echo "=== Step 1: save_activations (raw, random_k=2) ==="
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name "$MODEL" \
  --attachment_point post_mlp_residual \
  --layer "$LAYER" \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 8 \
  --output_dir "$RAW_ACT_DIR" \
  --random_k 2 \
  --save_every 50000

# ── Step 2: Train SAE ────────────────────────────────────────────────────────
echo "=== Step 2: sae_train ==="
$PYTHON sae_train.py \
  --sae_model "$SAE_MODEL" \
  --activations_dir "$RAW_ACT_DIR" \
  --val_activations_dir "$RAW_ACT_DIR" \
  --checkpoints_dir "$CKPT_DIR" \
  --expansion_factor "$EXPANSION" \
  --steps 110000 \
  --save_steps 20000 \
  --log_steps 1000 \
  --batch_size 4096 \
  --k "$K" \
  --auxk_alpha 0.03 \
  --decay_start 109999 \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --no_wandb

# ── Step 3: Save mean-pooled SAE activations ─────────────────────────────────
echo "=== Step 3: save_activations (SAE, mean-pooled) ==="
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name "$MODEL" \
  --attachment_point post_mlp_residual \
  --layer "$LAYER" \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 8 \
  --output_dir "$MEAN_ACT_DIR" \
  --mean_pool \
  --save_every 50000 \
  --sae_model "$SAE_MODEL" \
  --sae_path "$SAE_PATH"

# ── Step 4: Find top-16 activating images per neuron ─────────────────────────
echo "=== Step 4: find_hai_indices ==="
$PYTHON find_hai_indices.py \
  --activations_dir "$MEAN_ACT_DIR" \
  --dataset_name "$DATASET" \
  --data_path "$DATASET_PATH" \
  --split train \
  --k 16 \
  --chunk_size 1000

# ── Step 5: Visualize neurons ─────────────────────────────────────────────────
echo "=== Step 5: visualize_neurons ==="
$PYTHON visualize_neurons.py \
  --output_dir "$MEAN_ACT_DIR" \
  --top_k 16 \
  --dataset_name "$DATASET" \
  --data_path "$DATASET_PATH" \
  --split train \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --hai_indices_path "${MEAN_ACT_DIR}/hai_indices_16.npy"

# ── Step 6: Encode OmniMedVQA images with CLIP-base ──────────────────────────
echo "=== Step 6: encode_images (clip-vit-base-patch32) ==="
$PYTHON encode_images.py \
  --embeddings_path "${EMBED_DIR}/${DATASET}_train_embeddings_clip-vit-base-patch32.pt" \
  --model_name clip-vit-base-patch32 \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --batch_size 128

# ── Step 7a: Steering score — no steering (baseline) ─────────────────────────
echo "=== Step 7a: steering_score (no steering) ==="
$PYTHON steering_score.py \
  --hai_indices_path "${MEAN_ACT_DIR}/hai_indices_16.npy" \
  --embeddings_path "${EMBED_DIR}/${DATASET}_train_embeddings_clip-vit-base-patch32.pt" \
  --sae_path "$SAE_PATH" \
  --images_path "${DATASET_PATH}/Images/" \
  --no-pre_zero \
  --model_name clip-vit-base-patch32 \
  --vlm_backend llava_med \
  --neuron_prefix 10 \
  --no-steer \
  --output_path "${RESULTS_DIR}/no_steering/"

# ── Step 7b: Steering score — with steering ───────────────────────────────────
echo "=== Step 7b: steering_score (with steering) ==="
$PYTHON steering_score.py \
  --hai_indices_path "${MEAN_ACT_DIR}/hai_indices_16.npy" \
  --embeddings_path "${EMBED_DIR}/${DATASET}_train_embeddings_clip-vit-base-patch32.pt" \
  --sae_path "$SAE_PATH" \
  --images_path "${DATASET_PATH}/Images/" \
  --no-pre_zero \
  --model_name clip-vit-base-patch32 \
  --vlm_backend llava_med \
  --neuron_prefix 10 \
  --steer \
  --output_path "${RESULTS_DIR}/steering/"

echo "=== All done ==="
