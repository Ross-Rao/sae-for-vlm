#!/bin/bash
# Monosemanticity Score experiment for LLaVA-Med on OmniMedVQA
#
# 前置条件：已完整运行 llava_med_steering.sh
#   已有产出（直接复用，无需重跑）：
#     - activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/...  (SAE 激活)
#     - embeddings_dir/omnimed_train_embeddings_clip-vit-base-patch32.pt       (图片嵌入)
#     - checkpoints_dir/llava_med/.../ae_100000.pt                             (SAE 权重)
#
# 本脚本新增：
#   Step 1  保存原始神经元激活（mean-pooled，不挂 SAE），用于 baseline 对比
#   Step 2  用 metric.py 计算 SAE 神经元的单义性分数
#   Step 3  用 metric.py 计算原始神经元的单义性分数（baseline）

set -e

PROJECT_DIR="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm"
DATASET_PATH="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/datasets/OmniMedVQA/OmniMedVQA"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

LAYER=23
MODEL=llava_med
DATASET=omnimed
SAE_MODEL=matroyshka_batch_top_k
K=20
EXPANSION=64

ORIG_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/raw/mean_pool/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
MEAN_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/${SAE_MODEL}_${K}_x${EXPANSION}/mean_pool/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
EMBED_PATH="${PROJECT_DIR}/embeddings_dir/${DATASET}_train_embeddings_clip-vit-base-patch32.pt"
MS_SUBDIR="ms_clip-vit-base-patch32"

cd "$PROJECT_DIR"

# ── Step 1: 保存原始神经元激活（mean-pooled，无 SAE）───────────────────────
# steering 脚本存的是 random_k=2 用于 SAE 训练，这里需要 mean-pooled 用于 metric
echo "=== Step 1: save_activations (原始神经元, mean-pooled) ==="
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name "$MODEL" \
  --attachment_point post_mlp_residual \
  --layer "$LAYER" \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 8 \
  --output_dir "$ORIG_ACT_DIR" \
  --mean_pool \
  --save_every 50000

# ── Step 2: 计算 SAE 神经元单义性分数 ────────────────────────────────────────
# 直接复用 llava_med_steering.sh Step 3 的产出
echo "=== Step 2: metric.py (SAE 神经元) ==="
$PYTHON metric.py \
  --activations_dir "$MEAN_ACT_DIR" \
  --embeddings_path "$EMBED_PATH" \
  --output_subdir "$MS_SUBDIR" \
  --device cuda

echo "SAE 神经元结果保存在: ${MEAN_ACT_DIR}/${MS_SUBDIR}/metric_stats_new.txt"

# ── Step 3: 计算原始神经元单义性分数（baseline）─────────────────────────────
echo "=== Step 3: metric.py (原始神经元 baseline) ==="
$PYTHON metric.py \
  --activations_dir "$ORIG_ACT_DIR" \
  --embeddings_path "$EMBED_PATH" \
  --output_subdir "$MS_SUBDIR" \
  --device cuda

echo "原始神经元结果保存在: ${ORIG_ACT_DIR}/${MS_SUBDIR}/metric_stats_new.txt"

echo "=== All done ==="
echo ""
echo "对比结果："
echo "  SAE 神经元 MS:  cat ${MEAN_ACT_DIR}/${MS_SUBDIR}/metric_stats_new.txt"
echo "  原始神经元 MS:  cat ${ORIG_ACT_DIR}/${MS_SUBDIR}/metric_stats_new.txt"
