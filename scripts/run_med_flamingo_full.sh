#!/bin/bash
# Med-Flamingo SAE 完整流水线（Steering + Monosemanticity Score）
# 与 run_llava_med_full.sh 配置相同，仅模型换为 med_flamingo
# Step 顺序：1(raw acts)→2(SAE train)→3(SAE acts)→4(HAI)→5(embed)→6(MS score)→7(steering)→8(visualize)

set -e

PROJECT_DIR="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm"
DATASET_PATH="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/datasets/OmniMedVQA/OmniMedVQA"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
RESULTS_DIR="${PROJECT_DIR}/results/med_flamingo_omnimed"

LAYER=18
MODEL=med_flamingo
DATASET=omnimed
SAE_MODEL=matroyshka_batch_top_k
K=20
EXPANSION=64

RAW_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/raw/random_k_2/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
ORIG_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/raw/mean_pool/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
CKPT_DIR="${PROJECT_DIR}/checkpoints_dir/${MODEL}/${SAE_MODEL}_${K}_x${EXPANSION}/random_k_2"
SAE_PATH="${CKPT_DIR}/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual_${SAE_MODEL}_${K}_x${EXPANSION}/trainer_0/checkpoints/ae_100000.pt"
MEAN_ACT_DIR="${PROJECT_DIR}/activations_dir/${MODEL}/${SAE_MODEL}_${K}_x${EXPANSION}/mean_pool/${DATASET}_train_activations_${MODEL}_${LAYER}_post_mlp_residual"
EMBED_PATH="${PROJECT_DIR}/embeddings_dir/${DATASET}_train_embeddings_biomedclip.pt"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"

echo "========================================"
echo " Med-Flamingo SAE Pipeline"
echo " 结果目录: $RESULTS_DIR"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# ── Step 1: 采集原始激活（random_k=2，用于 SAE 训练）───────────────────────
echo ""
echo "[Step 1/8] save_activations (raw, random_k=2)  $(date '+%H:%M:%S')"
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name "$MODEL" \
  --attachment_point post_mlp_residual \
  --layer "$LAYER" \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 4 \
  --output_dir "$RAW_ACT_DIR" \
  --random_k 2 \
  --save_every 5000

# ── Step 2: 训练 SAE ─────────────────────────────────────────────────────────
echo ""
echo "[Step 2/8] sae_train  $(date '+%H:%M:%S')"
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

# ── Step 3: 采集 SAE 激活（mean-pooled）─────────────────────────────────────
echo ""
echo "[Step 3/8] save_activations (SAE, mean-pooled)  $(date '+%H:%M:%S')"
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name "$MODEL" \
  --attachment_point post_mlp_residual \
  --layer "$LAYER" \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 4 \
  --output_dir "$MEAN_ACT_DIR" \
  --mean_pool \
  --save_every 5000 \
  --sae_model "$SAE_MODEL" \
  --sae_path "$SAE_PATH"

# ── Step 4: 查找 Top-16 激活图片（只保留非零激活）──────────────────────────
echo ""
echo "[Step 4/8] find_hai_indices  $(date '+%H:%M:%S')"
$PYTHON find_hai_indices.py \
  --activations_dir "$MEAN_ACT_DIR" \
  --dataset_name "$DATASET" \
  --data_path "$DATASET_PATH" \
  --split train \
  --k 16 \
  --chunk_size 1000

# ── Step 5: 编码图片嵌入（BiomedCLIP，可复用 LLaVA-Med 已有结果）────────────
echo ""
echo "[Step 5/8] encode_images  $(date '+%H:%M:%S')"
if [ -f "$EMBED_PATH" ]; then
  echo "  嵌入文件已存在，跳过: $EMBED_PATH"
else
  $PYTHON encode_images.py \
    --embeddings_path "$EMBED_PATH" \
    --model_name biomedclip \
    --dataset_name "$DATASET" \
    --split train \
    --data_path "$DATASET_PATH" \
    --batch_size 128
fi

# ── Step 6: 单义性评分（需要 embed，早于 visualize）─────────────────────────
echo ""
echo "[Step 6/8] monosemanticity score  $(date '+%H:%M:%S')"

echo "  6a. 采集原始神经元激活（mean-pooled）..."
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name "$MODEL" \
  --attachment_point post_mlp_residual \
  --layer "$LAYER" \
  --dataset_name "$DATASET" \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 4 \
  --output_dir "$ORIG_ACT_DIR" \
  --mean_pool \
  --save_every 50000

echo "  6b. metric.py (SAE 神经元)..."
$PYTHON metric.py \
  --activations_dir "$MEAN_ACT_DIR" \
  --embeddings_path "$EMBED_PATH" \
  --output_subdir ms_biomedclip \
  --take_every 50 \
  --min_support 10 \
  --device cuda

echo "  6c. metric.py (原始神经元 baseline)..."
$PYTHON metric.py \
  --activations_dir "$ORIG_ACT_DIR" \
  --embeddings_path "$EMBED_PATH" \
  --output_subdir ms_biomedclip \
  --take_every 50 \
  --min_support 10 \
  --device cuda

# ── Step 7: Steering 评分 ────────────────────────────────────────────────────
echo ""
echo "[Step 7/8] steering_score  $(date '+%H:%M:%S')"

echo "  7a. no steering (baseline)..."
$PYTHON steering_score.py \
  --hai_indices_path "${MEAN_ACT_DIR}/hai_indices_16_nonzero.npy" \
  --embeddings_path "$EMBED_PATH" \
  --sae_path "$SAE_PATH" \
  --images_path "${DATASET_PATH}/Images/" \
  --no-pre_zero \
  --model_name biomedclip \
  --vlm_backend med_flamingo \
  --neuron_prefix 10 \
  --no-steer \
  --output_path "${RESULTS_DIR}/steering/no_steering/"

echo "  7b. with steering..."
$PYTHON steering_score.py \
  --hai_indices_path "${MEAN_ACT_DIR}/hai_indices_16_nonzero.npy" \
  --embeddings_path "$EMBED_PATH" \
  --sae_path "$SAE_PATH" \
  --images_path "${DATASET_PATH}/Images/" \
  --no-pre_zero \
  --model_name biomedclip \
  --vlm_backend med_flamingo \
  --neuron_prefix 10 \
  --steer \
  --output_path "${RESULTS_DIR}/steering/with_steering/"

# ── Step 8: 可视化神经元（按 MS score 选取）─────────────────────────────────
echo ""
echo "[Step 8/8] visualize_neurons (MS-score selected)  $(date '+%H:%M:%S')"
$PYTHON visualize_neurons.py \
  --output_dir "$MEAN_ACT_DIR/visualizations_minsup10" \
  --top_k 16 \
  --dataset_name "$DATASET" \
  --data_path "$DATASET_PATH" \
  --split train \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --hai_indices_path "${MEAN_ACT_DIR}/hai_indices_16_nonzero.npy" \
  --ms_scores_path "${MEAN_ACT_DIR}/ms_biomedclip/all_neurons_scores.pth" \
  --top_n 20

# ── 汇总结果到 results/ ──────────────────────────────────────────────────────
echo ""
echo "收集结果文件..."
MS_SAE="${MEAN_ACT_DIR}/ms_biomedclip/metric_stats_new.txt"
MS_ORIG="${ORIG_ACT_DIR}/ms_biomedclip/metric_stats_new.txt"

mkdir -p "${RESULTS_DIR}/monosemanticity"
[ -f "$MS_SAE" ]  && cp "$MS_SAE"  "${RESULTS_DIR}/monosemanticity/sae_neurons.txt"
[ -f "$MS_ORIG" ] && cp "$MS_ORIG" "${RESULTS_DIR}/monosemanticity/original_neurons.txt"

echo ""
echo "========================================"
echo " 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果目录结构:"
find "$RESULTS_DIR" -type f | sort
echo "========================================"
