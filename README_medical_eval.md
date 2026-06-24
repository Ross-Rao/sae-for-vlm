# Medical VQA Evaluation

本模块在 OmniMedVQA 数据集上对医学视觉语言模型进行多选题评测，使用 DeepSeek 作为 LLM 裁判打分。

## 评测结果（OmniMedVQA，50 样本子集）

| 模型 | 正确 / 总计 | 准确率 |
|------|------------|--------|
| LLaVA-Med | 36 / 50 | **72%** |
| Med-Flamingo | 23 / 50 | **46%** |

评测时间：2026-06

## 环境准备

项目根目录：`/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm`

```bash
cd /media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm
source .venv/bin/activate
```

将 DeepSeek API Key 写入 `.env`（已在 `.gitignore` 中排除）：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

## 运行命令

### LLaVA-Med

LLaVA-Med 的 tokenizer 依赖旧版 protobuf 接口，需设置环境变量绕过版本冲突：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python eval_medical_vqa.py \
  --dataset omnimed \
  --vlm_backend llava_med \
  --max_samples 50 \
  --max_new_tokens 10 \
  --deepseek_api_key <YOUR_KEY> \
  --output_dir ./results/llava_med_omnimed_n50
```

tmux 后台运行：

```bash
cat > /tmp/run_llava.sh << 'EOF'
#!/bin/bash
cd /media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm
source .venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python eval_medical_vqa.py \
  --dataset omnimed --vlm_backend llava_med \
  --max_samples 50 --max_new_tokens 10 \
  --deepseek_api_key <YOUR_KEY> \
  --output_dir ./results/llava_med_omnimed_n50 \
  2>&1 | tee ./results/llava_med_n50.log
EOF
tmux new-session -d -s llava "bash /tmp/run_llava.sh"
```

### Med-Flamingo

```bash
python eval_medical_vqa.py \
  --dataset omnimed \
  --vlm_backend med_flamingo \
  --max_samples 50 \
  --max_new_tokens 10 \
  --deepseek_api_key <YOUR_KEY> \
  --output_dir ./results/med_flamingo_omnimed_n50
```

tmux 后台运行：

```bash
tmux new-session -d -s flamingo \
  "cd /media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm && \
   source .venv/bin/activate && \
   python eval_medical_vqa.py \
     --dataset omnimed --vlm_backend med_flamingo \
     --max_samples 50 --max_new_tokens 10 \
     --deepseek_api_key <YOUR_KEY> \
     --output_dir ./results/med_flamingo_omnimed_n50 \
     2>&1 | tee ./results/med_flamingo_n50.log"
```

## 结果文件结构

```
results/
├── llava_med_omnimed_n50/
│   ├── summary.json        # 整体统计（accuracy 等）
│   └── predictions.jsonl   # 每条样本的预测与裁判结果
├── llava_med_n50.log
├── med_flamingo_omnimed_n50/
│   ├── summary.json
│   └── predictions.jsonl
└── med_flamingo_n50.log
```

`results/` 已加入 `.gitignore`，不会被提交。

## 模型权重位置

| 模型 | 路径 |
|------|------|
| LLaVA-Med | `downloads/llava-med` |
| Med-Flamingo checkpoint | `downloads/med-flamingo/model.pt` |
| LLaMA-7B | `downloads/llama-7b-hf` |

（路径均相对于 `/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/`）

## 已知问题

- **protobuf 冲突**：LLaVA-Med 运行时须设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`。
- **显存**：Med-Flamingo fp16 约需 16 GB，LLaVA-Med 约需 14 GB。

## 与官方配置的对比分析

### Med-Flamingo

参考官方：`Med-Flamingo/scripts/demo.py`

| 配置项 | 官方 demo | 当前实现 | 说明 |
|--------|-----------|---------|------|
| 模型初始化参数 | `ViT-L-14`, openai, `cross_attn_every_n_layers=4` | 相同 | ✓ 一致 |
| 模型精度 | `accelerator.prepare(model)`（不显式 fp16） | `.half().to(device)` | 为适配显存限制主动转 fp16 |
| **推理模式** | **7-shot**（7 张示例图 + QA 对） | **Zero-shot** | ✗ 不一致，影响最大 |
| 提示格式 | `<image>Question: X Answer: Y.<\|endofchunk\|>` × 7 + `<image>Question: X Answer:` | `<image>{text}\nAnswer:` | ✗ 缺少 few-shot 示例 |
| max_new_tokens | 10 | 10（eval 传入） | ✓ 一致 |
| do_sample | 默认 False（greedy） | False | ✓ 一致 |
| clean_generation | `replace('<unk> ', '').strip()` | 截取 `Answer:` 后、`<\|endofchunk\|>` 前 | ✗ 逻辑不同 |

**关键差异**：Med-Flamingo 是为 few-shot 设计的多模态模型，官方 demo 使用 7 张带标注的示例图作为上下文。当前实现采用 zero-shot，偏离了模型的预期使用方式，当前 46% 的准确率偏保守。

---

### LLaVA-Med

参考官方：`LLaVA-Med/llava/eval/model_vqa.py`，使用权重 `llava-med-v1.5-mistral-7b`（Mistral-7B 骨干）

| 配置项 | 官方脚本默认值 | 当前实现 | 说明 |
|--------|--------------|---------|------|
| Conv template | `"vicuna_v1"` | `"mistral_instruct"` | ✓ 我们使用 Mistral 权重，`mistral_instruct` 正确 |
| `mm_use_im_start_end` | 读 config（config=False） | 直接用 `DEFAULT_IMAGE_TOKEN` | ✓ 与 config 一致 |
| **do_sample** | `True, temperature=0.2` | `False`（greedy） | ✗ 不一致，但 greedy 对 MCQ 更合适 |
| max_new_tokens | 1024 | 10（eval 传入） | MCQ 只需 1 个字母，10 足够 |
| 输出后处理 | `.strip()` | `split('[/INST]')[-1].strip()` | ✓ Mistral 格式需要清除 `[/INST]` 前缀 |

**关键差异**：官方脚本默认对 Vicuna 权重使用 `do_sample=True`；我们对 Mistral 权重使用 greedy decoding，对 MCQ 评测更合适、可复现。Conv template 和输出后处理均已针对 Mistral 权重正确适配。

---

## CheXAgent (CheXagent-2-3b) 评测结果

### 准确率（OmniMedVQA，50 样本）

| 模型 | 样本数 | 正确 | 准确率 |
|------|--------|------|--------|
| Med-Flamingo | 50 | 23 | 46% |
| LLaVA-Med | 50 | 36 | 72% |
| **CheXAgent** | **50** | **29** | **58%** |

### 运行命令

```bash
cd /media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/sae-for-vlm
source .env
.venv/bin/python eval_medical_vqa.py \
  --dataset omnimed \
  --vlm_backend chexagent \
  --max_samples 50 \
  --output_dir results/chexagent_n50 \
  --deepseek_api_key $DEEPSEEK_API_KEY
```

### 模型配置

| 配置项 | 值 |
|--------|----|
| 模型权重 | `downloads/CheXagent-2-3b` |
| 视觉编码器 | XraySigLIP（SigLIP ViT-L/16, 384px）|
| LLM 骨干 | Phi-2 |
| 精度 | bfloat16 |
| 解码策略 | Greedy（`do_sample=False`, `num_beams=1`）|
| 图像输入 | 保存为临时文件，由 tokenizer `from_list_format` 读取 |

### 适配说明

- `transformers==4.40.0` 版本 assert 已替换为 warning，兼容项目 venv 中的 4.46.3
- `config.json` 的 `vision_model_name_or_path` 已指向本地 XraySigLIP 路径
- 降级 `protobuf<=3.20.3` 和 `numpy<2` 以兼容 sentencepiece 和 torch 扩展
- PIL 图像通过 `tempfile.NamedTemporaryFile` 转为文件路径后传入模型

### 分析

CheXAgent 专为胸部 X 光设计，在泛化的 OmniMedVQA 数据集上得到 58%，介于 Med-Flamingo（46%）和 LLaVA-Med（72%）之间。在 X 光相关子集上预期表现更佳。

---

## CheXAgent 配置对比（官方 vs 我们）

| 配置项 | 官方 README | 我们的实现 | 一致？ |
|--------|------------|-----------|--------|
| 加载方式 | `from_pretrained(..., device_map="auto")` + `.to(bfloat16)` | 同 | ✓ |
| `do_sample` | `False` | `False` | ✓ |
| `num_beams` | `1` | `1` | ✓ |
| `temperature` | `1.` | `1.0` | ✓ |
| `top_p` | `1.` | `1.0` | ✓ |
| `use_cache` | `True` | `True` | ✓ |
| `max_new_tokens` | `512` | 默认 `512`（可覆盖） | ✓ |
| system prompt | `"You are a helpful assistant."` | 同 | ✓ |
| tokenizer 格式 | `from_list_format([{'image': path}, {'text': prompt}])` | 同 | ✓ |
| 输出截取 | `output[input_ids.size(1):-1]` | 同 | ✓ |

**结论：全部配置与官方推荐一致。**

注：`config.json` 中 `"torch_dtype": "float32"`，但官方示例本身也是先 `from_pretrained` 再 `.to(bfloat16)`，我们做法相同。
