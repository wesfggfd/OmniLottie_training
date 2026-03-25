# OmniLottie-Qwen3.5-9B

> A Qwen3.5-9B based reimplementation of the OmniLottie training pipeline for multimodal Lottie animation generation.

本项目是一个基于 **OmniLottie** 论文思路的个人重构版本，面向 **多模态到 Lottie 动画生成** 任务。与原始项目不同，这个仓库将主干模型替换为 **Qwen3.5-9B**，并围绕当前仓库中的 `train.py` 重新整理了训练流程、数据读取方式和模型导出逻辑。

## Highlights

- 基于 **OmniLottie** 的参数化 Lottie token 建模思路
- 将 backbone 替换为 **Qwen3.5-9B**
- 支持 **文本 / 图像 / 视频** 作为条件输入
- 使用 **LoRA + 扩展 vocab 行训练** 进行参数高效微调
- 支持 **Parquet 数据集**、断点续训、early stopping、best/final 模型导出
- README 内容以当前仓库实现为准，尤其是 `train.py`

---

## Overview

Lottie 是一种结构化、可编辑的矢量动画表示。相比直接生成像素级视频，Lottie 生成更强调：

- 输出 JSON 的合法性
- 图形结构的可编辑性
- 运动参数的可控性
- 动画表达与条件输入的一致性

本项目延续 OmniLottie 的核心思想：

1. 将 Lottie 动画编码为可学习的离散 token；
2. 使用多模态条件输入进行建模；
3. 采用自回归方式生成目标 Lottie 序列；
4. 将生成结果恢复为结构化 Lottie 表示。

当前版本的主要改动包括：

- **Backbone 从原方案替换为 Qwen3.5-9B**
- 训练入口统一为 `train.py`
- 使用 `HybridLottieTokenizer` 编码 Lottie target
- 使用 `PEFT LoRA` 对 Qwen 主干进行高效微调
- 保留官方式 `resize_token_embeddings(...)` 架构，并将新增 Lottie token id 整体平移到 Qwen3.5 词表之后

---

## Repository Structure

```text
OmniLottie/
├── train.py
├── data/
│   ├── collator.py
│   └── lottie_dataset.py
├── models/
│   └── lottie_qwen35.py
├── tokenizer/
│   └── hybrid_lottie_tokenizer.py
├── configs/
│   └── ds_zero2.json
├── inference.py
├── inference_hf.py
├── app.py
├── app_hf.py
├── requirements.txt
└── requirements_train.txt
```

核心训练相关文件：

- `train.py`：训练主入口
- `decoder.py`：基于官方 OmniLottie 思路的 Qwen3.5 decoder
- `tokenizer/hybrid_lottie_tokenizer.py`：Lottie token / JSON 编码器
- `data/lottie_dataset.py`：Parquet 数据加载与多模态样本构造
- `data/collator.py`：batch padding 逻辑

---

## Model

当前训练脚本默认使用如下 backbone：

```bash
Qwen/Qwen3.5-9B
```

对应 `train.py` 中的默认参数：

```python
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-9B")
```

因此，这个仓库的训练目标是一个基于 **Qwen3.5-9B** 的多模态 Lottie 生成模型，而不是原始 OmniLottie 官方 release 的原始配置。

---

## Installation

建议使用 Python 3.10+。

### 1. Clone

```bash
git clone https://github.com/wesfggfd/OmniLottie_training.git
cd OmniLottie_training/OmniLottie
```

### 2. Create Environment

```bash
conda create -n omnilottie_qwen35 python=3.10 -y
conda activate omnilottie_qwen35
```

### 3. Install Training Dependencies

```bash
pip install -r requirements_train.txt
```

训练依赖主要包括：

- `torch==2.7.0`
- `torchvision==0.22.0`
- `torchaudio==2.7.0`
- `transformers` (main)
- `accelerate>=1.3.0`
- `peft>=0.14.0`
- `deepspeed>=0.16.0`
- `datasets>=3.5.0`
- `pyarrow>=18.0.0`
- `Pillow>=10.1.0,<12.0`
- `qwen-vl-utils>=0.0.11`

如需兼容仓库中的推理脚本，也可额外安装：

```bash
pip install -r requirements.txt
```

---

## Dataset Format

训练脚本会递归读取 `--data_path` 下的所有 `*.parquet` 文件：

```python
parquet_files = sorted(str(path) for path in dataset_root.rglob("*.parquet"))
```

示例目录结构：

```text
data/
├── part-00000.parquet
├── part-00001.parquet
└── subdir/
    └── part-00002.parquet
```

### Supported Input Fields

`data/lottie_dataset.py` 使用字段候选映射自动适配不同数据格式。

**Text fields**

- `desc_en`
- `motion_caption`
- `detail`
- `keywords_en`
- `short_desc`
- `medium_desc`
- `long_desc`
- `caption`
- `text`
- `prompt`
- `instruction`

**Image fields**

- `image`
- `image_path`
- `keyframe`
- `keyframe_path`

**Video fields**

- `video`
- `video_path`
- `rendered_video`
- `rendered_video_path`

**Target sequence fields**

优先使用：

- `sequence_text`
- `lottie_sequence`
- `token_sequence`

若没有 sequence 字段，则回退到：

- `lottie_json`
- `json`
- `animation_json`
- `lottie`

### Minimum Sample Requirement

每条样本至少需要：

- 一个条件输入：文本 / 图像 / 视频中的一种或多种
- 一个目标字段：`sequence_text` 或 Lottie JSON

### Train / Eval Split

当前 `train.py` 默认按 **parquet 文件数** 划分训练集和验证集：

- 前 98% 文件：训练集
- 后 2% 文件：验证集

```python
split_index = max(1, int(len(parquet_files) * 0.98))
```

> 注意：这里不是按样本随机切分，而是按 parquet 文件切分。

---

## Training

### Quick Start

最简单的训练命令：

```bash
python train.py \
  --data_path /path/to/parquet_data \
  --output_dir /path/to/output
```

### Recommended: Accelerate Launch

由于训练脚本基于 `Accelerator` 实现，推荐使用：

```bash
accelerate launch train.py \
  --model_path Qwen/Qwen3.5-9B \
  --data_path /path/to/parquet_data \
  --output_dir /path/to/output \
  --max_seq_len 4096 \
  --num_epochs 5 \
  --per_device_batch 1 \
  --grad_accum 8 \
  --save_steps 2000 \
  --eval_steps 1000 \
  --logging_steps 10
```

如果你已经配置了 deepspeed / accelerate，也可以结合自定义配置文件使用。

---

## Training Arguments

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | `Qwen/Qwen3.5-9B` | Base model path or HF repo name |
| `--data_path` | required | Parquet dataset root |
| `--output_dir` | required | Output directory |
| `--max_seq_len` | `4096` | Maximum sequence length |
| `--num_epochs` | `5` | Number of epochs |
| `--per_device_batch` | `1` | Per-device batch size |
| `--grad_accum` | `8` | Gradient accumulation steps |
| `--save_steps` | `2000` | Checkpoint save interval |
| `--eval_steps` | `1000` | Evaluation interval |
| `--logging_steps` | `10` | Logging interval |
| `--warmup_ratio` | `0.03` | Warmup ratio |
| `--lora_rank` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--lora_lr` | `2e-5` | LoRA learning rate |
| `--lottie_lr` | `5e-4` | Learning rate for appended Lottie vocab rows |
| `--num_workers` | `2` | Dataloader workers |
| `--resume_from` | `None` | Resume from checkpoint |
| `--seed` | `42` | Random seed |
| `--early_stopping_patience` | `5` | Early stopping patience |
| `--early_stopping_min_delta` | `0.0` | Minimum delta for improvement |

---

## Optimization Strategy

当前实现采用 **LoRA + 扩展后的 embedding / lm_head 行训练** 的策略，而不是全参数微调。

优化参数主要分为两组：

1. `embed_tokens.weight` / `lm_head.weight` 中新增的 Lottie token 行
2. `lora_*`

并分别使用不同学习率：

- `lottie_lr = 5e-4`
- `lora_lr = 2e-5`

这种设计的动机是：

- 使用 LoRA 适配 Qwen3.5-9B 的语言与多模态表示能力
- 使用更高学习率训练新增 Lottie token 对应的 embedding / lm head 行
- 在有限训练资源下提高适配效率

---

## Input Construction and Supervision

每条样本会被构造成多模态 chat 输入。当前默认 system prompt 为：

```text
Generate a valid Lottie animation.
```

输入可包含：

- 文本描述
- 文本-图像
- 视频

目标可来自：

- `sequence_text`
- 或由 Lottie JSON 编码得到的 token 序列

训练时：

- 条件输入与目标 token 被拼接为同一条序列
- 仅目标 Lottie token 部分参与 loss 计算
- 条件输入对应的 label 被置为 `-100`

这符合标准的 conditional autoregressive training 设置。

---

## Output Structure

训练输出目录通常包含：

```text
output_dir/
├── training_args.json
├── train_log.jsonl
├── best/
├── final/
├── checkpoint-xxxx/
└── latest_checkpoint.txt

> 推荐将训练输出写到仓库外部目录，或确保这些目录被 `.gitignore` 忽略，避免将 checkpoint 和日志提交到代码仓库。
```

说明：

- `training_args.json`：保存训练参数
- `train_log.jsonl`：训练 / 验证 / checkpoint / early stop 日志
- `best/`：验证集最佳模型
- `final/`：训练结束后从 `best/` 复制得到的最终模型
- `checkpoint-xxxx/`：阶段性保存的训练状态
- `latest_checkpoint.txt`：最近一次 checkpoint 的路径

### Checkpoint Policy

当前实现会在保存新 checkpoint 后删除旧 checkpoint，只保留最新一个。

---

## Resume Training

### Resume from a Specific Checkpoint

```bash
accelerate launch train.py \
  --model_path Qwen/Qwen3.5-9B \
  --data_path /path/to/parquet_data \
  --output_dir /path/to/output \
  --resume_from /path/to/output/checkpoint-2000
```

### Resume from Latest Checkpoint

```bash
accelerate launch train.py \
  --model_path Qwen/Qwen3.5-9B \
  --data_path /path/to/parquet_data \
  --output_dir /path/to/output \
  --resume_from latest
```

当 `--resume_from latest` 时，脚本会读取：

```text
output_dir/latest_checkpoint.txt
```

并恢复到最近一次保存的位置。

---

## Early Stopping

项目内置 early stopping 机制。

相关参数：

```bash
--early_stopping_patience
--early_stopping_min_delta
```

默认设置：

- `patience = 5`
- `min_delta = 0.0`

即验证损失连续 5 次评估没有改善时，训练提前停止。

---

## Full Training Example

```bash
accelerate launch train.py \
  --model_path Qwen/Qwen3.5-9B \
  --data_path ./datasets/mmlottie_parquet \
  --output_dir ./outputs/qwen35_omnilottie \
  --max_seq_len 4096 \
  --num_epochs 5 \
  --per_device_batch 1 \
  --grad_accum 8 \
  --save_steps 2000 \
  --eval_steps 1000 \
  --logging_steps 10 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_lr 2e-5 \
  --lottie_lr 5e-4 \
  --num_workers 2 \
  --seed 42
```

---

## Relation to the Original OmniLottie

本仓库不是原始官方 README 的镜像，而是基于 OmniLottie 论文思路做的个人实现与训练重构。

主要区别如下：

- 方法来源：参考 OmniLottie 的参数化 Lottie token 思路
- 模型主干：替换为 **Qwen3.5-9B**
- 训练流程：以当前仓库的 `train.py` 为准
- 数据适配：使用 parquet + 字段候选映射支持更灵活的数据组织
- 微调方式：采用 LoRA，而不是默认沿用原始官方训练配置

说明：

1. 原始方法灵感来自 OmniLottie；
2. 当前实现对 backbone 与训练流程进行了重新设计。

---

## Inference Scripts

仓库中仍然保留了部分原有推理相关脚本：

- `inference.py`
- `inference_hf.py`
- `app.py`
- `app_hf.py`

本 README 主要针对 **Qwen3.5-9B 训练版本**。
---

## Training Notes

- 建议不要将 `datasets/`、`models/`、`outputs/`、checkpoint 文件和日志直接提交到仓库。
- 若需复现实验，请优先记录数据来源、训练参数、模型版本和输出路径。
- 如果训练输出目录放在仓库内部，请确认根目录 `.gitignore` 已生效。

---

## TODO

- [ ] 补充训练前的数据预处理脚本
- [ ] 补充 `best/` 和 `final/` 模型的推理说明
- [ ] 补充评测脚本与可视化对比
- [ ] 补充不同 Lottie token 化策略的实验记录
- [ ] 补充多机多卡训练配置示例

---

## Acknowledgements

本项目基于原始 **OmniLottie** 论文思路进行改写与实现，感谢原作者在 Lottie 动画生成方向上的探索。

同时感谢以下开源生态：

- OmniLottie
- Qwen / Qwen-VL
- HuggingFace Transformers
- Accelerate
- PEFT

---

## Citation

如果你需要引用原始方法，请引用 OmniLottie 论文：

```bibtex
@article{yang2026omnilottie,
  title={OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens},
  author={Yiying Yang and Wei Cheng and Sijin Chen and Honghao Fu and Xianfang Zeng and Yujun Cai and Gang Yu and Xinjun Ma},
  journal={arXiv preprint arXiv:2603.02138},
  year={2026}
}
```


