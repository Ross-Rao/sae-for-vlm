# SAE 分析框架运行指南

在 OmniMedVQA 数据集上，以 LLaVA-Med 为 VLM 后端，运行两个实验：
- **实验一：SAE Steering** — 通过干预 SAE 神经元控制 LLaVA-Med 输出
- **实验二：Monosemanticity Score** — 量化 SAE 神经元的单义性，与原始神经元对比

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 项目根目录 | `/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm` |
| 数据集根目录 | `/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/datasets/OmniMedVQA/OmniMedVQA` |
| Python | `<项目根目录>/.venv/bin/python` |
| GPU | RTX 3090（`cuda:0`，主力）/ GTX 1080（`cuda:1`，不支持 bfloat16 conv，LLaVA-Med 须强制用 cuda:0） |

---

## 步骤依赖总览

两个实验共用 Steps 1–6，各自追加一个实验专属步骤：

```
Step 1  save_activations (raw, random_k=2)       ← SAE 训练数据
    │
Step 2  sae_train                                ← 训练 Matryoshka SAE
    │
Step 3  save_activations (SAE, mean-pooled)      ← 每张图一个 SAE 激活向量
    │
Step 4  find_hai_indices                         ← 每个神经元最强激活的 Top-16 图
    │
Step 5  visualize_neurons                        ← 可视化神经元
    │
Step 6  encode_images (CLIP-base)                ← 全库图片嵌入
    │
    ├──── Step 7  steering_score ×2              【实验一：Steering】
    │
    └──── Step MS  save_activations (原始 mean-pooled)
               └── metric.py ×2                 【实验二：Monosemanticity Score】
```

---

## 快速启动

```bash
PROJECT_DIR="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm"
cd "$PROJECT_DIR"
```

**实验一（Steering，含 Steps 1–7）：**
```bash
tmux new-session -s llava_sae \
  'bash scripts/llava_med_steering.sh 2>&1 | tee llava_med_steering.log'
```

**实验二（Monosemanticity，在实验一完成后运行）：**
```bash
tmux new-session -s llava_ms \
  'bash scripts/llava_med_monosemanticity.sh 2>&1 | tee llava_med_monosemanticity.log'
```

查看进度：
```bash
tail -f llava_med_steering.log
tail -f llava_med_monosemanticity.log
```

---

## 分步说明

以下命令中使用的公共变量：

```bash
PROJECT_DIR="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm"
DATASET_PATH="/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/datasets/OmniMedVQA/OmniMedVQA"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
SAE_PATH="checkpoints_dir/llava_med/matroyshka_batch_top_k_20_x64/random_k_2/omnimed_train_activations_llava_med_23_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_100000.pt"
cd "$PROJECT_DIR"
```

---

### Step 1 — 采集原始激活（~14 分钟）

从 LLaVA-Med 视觉编码器第 23 层采集激活，每张图随机采 2 个 token（用于 SAE 训练）：

```bash
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name llava_med \
  --attachment_point post_mlp_residual \
  --layer 23 \
  --dataset_name omnimed \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 8 \
  --output_dir activations_dir/llava_med/raw/random_k_2/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --random_k 2 \
  --save_every 50000
```

---

### Step 2 — 训练 SAE

用 Matryoshka BatchTopK SAE（expansion=64，k=20）训练 11 万步：

```bash
$PYTHON sae_train.py \
  --sae_model matroyshka_batch_top_k \
  --activations_dir activations_dir/llava_med/raw/random_k_2/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --val_activations_dir activations_dir/llava_med/raw/random_k_2/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --checkpoints_dir checkpoints_dir/llava_med/matroyshka_batch_top_k_20_x64/random_k_2 \
  --expansion_factor 64 \
  --steps 110000 \
  --save_steps 20000 \
  --log_steps 1000 \
  --batch_size 4096 \
  --k 20 \
  --auxk_alpha 0.03 \
  --decay_start 109999 \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --no_wandb
```

checkpoint 写入：`$SAE_PATH`（见上方变量定义）

---

### Step 3 — 采集 SAE 激活（mean-pooled，~14 分钟）

挂载训练好的 SAE，对全量数据集做 mean pooling，每张图产出一个激活向量：

```bash
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name llava_med \
  --attachment_point post_mlp_residual \
  --layer 23 \
  --dataset_name omnimed \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 8 \
  --output_dir activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --mean_pool \
  --save_every 50000 \
  --sae_model matroyshka_batch_top_k \
  --sae_path "$SAE_PATH"
```

---

### Step 4 — 查找每个神经元的 Top-16 激活图片

```bash
$PYTHON find_hai_indices.py \
  --activations_dir activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --dataset_name omnimed \
  --data_path "$DATASET_PATH" \
  --split train \
  --k 16 \
  --chunk_size 1000
```

---

### Step 5 — 可视化神经元

```bash
$PYTHON visualize_neurons.py \
  --output_dir activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --top_k 16 \
  --dataset_name omnimed \
  --data_path "$DATASET_PATH" \
  --split train \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --hai_indices_path activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual/hai_indices_16.npy
```

---

### Step 6 — 编码图片嵌入（~5 分钟）

用轻量 CLIP-base 对全库图片编码，供 steering 评分和单义性计算共用：

```bash
$PYTHON encode_images.py \
  --embeddings_path embeddings_dir/omnimed_train_embeddings_clip-vit-base-patch32.pt \
  --model_name clip-vit-base-patch32 \
  --dataset_name omnimed \
  --split train \
  --data_path "$DATASET_PATH" \
  --batch_size 128
```

---

### Step 7 — Steering 评分【实验一专属】

**7a. 无 steering（baseline）：**

```bash
$PYTHON steering_score.py \
  --hai_indices_path activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual/hai_indices_16.npy \
  --embeddings_path embeddings_dir/omnimed_train_embeddings_clip-vit-base-patch32.pt \
  --sae_path "$SAE_PATH" \
  --images_path "${DATASET_PATH}/Images/" \
  --no-pre_zero \
  --model_name clip-vit-base-patch32 \
  --vlm_backend llava_med \
  --neuron_prefix 10 \
  --no-steer \
  --output_path llava_med_results_dir/omnimed/no_steering/
```

**7b. 有 steering（实验组）：**

```bash
$PYTHON steering_score.py \
  --hai_indices_path activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual/hai_indices_16.npy \
  --embeddings_path embeddings_dir/omnimed_train_embeddings_clip-vit-base-patch32.pt \
  --sae_path "$SAE_PATH" \
  --images_path "${DATASET_PATH}/Images/" \
  --no-pre_zero \
  --model_name clip-vit-base-patch32 \
  --vlm_backend llava_med \
  --neuron_prefix 10 \
  --steer \
  --output_path llava_med_results_dir/omnimed/steering/
```

---

### Step MS — 单义性评分【实验二专属，在 Steps 1–6 完成后运行】

**MS-1. 采集原始神经元激活（mean-pooled，无 SAE，~14 分钟）：**

```bash
$PYTHON save_activations.py \
  --batch_size 32 \
  --model_name llava_med \
  --attachment_point post_mlp_residual \
  --layer 23 \
  --dataset_name omnimed \
  --split train \
  --data_path "$DATASET_PATH" \
  --num_workers 8 \
  --output_dir activations_dir/llava_med/raw/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --mean_pool \
  --save_every 50000
```

**MS-2. 计算 SAE 神经元单义性分数：**

```bash
$PYTHON metric.py \
  --activations_dir activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --embeddings_path embeddings_dir/omnimed_train_embeddings_clip-vit-base-patch32.pt \
  --output_subdir ms_clip-vit-base-patch32 \
  --device cuda
```

**MS-3. 计算原始神经元单义性分数（baseline）：**

```bash
$PYTHON metric.py \
  --activations_dir activations_dir/llava_med/raw/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual \
  --embeddings_path embeddings_dir/omnimed_train_embeddings_clip-vit-base-patch32.pt \
  --output_subdir ms_clip-vit-base-patch32 \
  --device cuda
```

查看对比结果：

```bash
# SAE 神经元
cat activations_dir/llava_med/matroyshka_batch_top_k_20_x64/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual/ms_clip-vit-base-patch32/metric_stats_new.txt

# 原始神经元（baseline）
cat activations_dir/llava_med/raw/mean_pool/omnimed_train_activations_llava_med_23_post_mlp_residual/ms_clip-vit-base-patch32/metric_stats_new.txt
```

MS 值越高表示神经元越单义。

---

## 预估运行时间

| 步骤 | 耗时 | 备注 |
|------|------|------|
| Step 1 采集原始激活 | ~14 分钟 | 实测 104 img/s，88,996 张图 |
| Step 2 训练 SAE | 数小时 | 纯矩阵运算，无需推理模型在显存 |
| Step 3 采集 SAE 激活 | ~14 分钟 | 同 Step 1 |
| Step 4 find_hai_indices | ~数分钟 | — |
| Step 5 visualize_neurons | ~数分钟 | — |
| Step 6 encode_images | ~5 分钟 | CLIP-base 轻量 |
| Step 7 steering_score ×2 | 取决于 neuron_prefix | — |
| Step MS-1 原始激活 | ~14 分钟 | 与 Step 1 相同量级 |
| Step MS-2/3 metric.py ×2 | 数十分钟 | 88k 图两两对比，O(n²) |

---

## 其他模型适配说明

Med-Flamingo / CheXAgent 支持 Steps 1–5（激活采集、SAE 训练、神经元可视化）及实验二（单义性评分），替换以下变量即可：

| 变量 | Med-Flamingo | CheXAgent |
|------|-------------|-----------|
| `MODEL` | `med_flamingo` | `chexagent` |
| `LAYER` | `18` | `18` |

Step 7 Steering 目前仅支持 `llava_med` 后端，Med-Flamingo / CheXAgent 需额外实现 `--vlm_backend` 对应逻辑。
