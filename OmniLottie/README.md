# OmniLottie-Qwen3.5-9B

> Qwen3.5-9B based OmniLottie-style training code.

本仓库保留 **OmniLottie 论文 / 官方 repo 的核心思路**：
- 多模态条件输入
- 自回归生成 Lottie token
- 只对 target Lottie 段计算 loss
- 将 Lottie token 空间追加到基础模型词表之后

当前实现做的主要事情只有两类：
- 将 backbone 从原始 `Qwen2.5-VL` 路线替换为 `Qwen/Qwen3.5-9B`
- 按新的 base vocab 大小，整体平移官方 Lottie 规则中的 token id / offset

## Current Code Path

当前保留并实际使用的主链路只有这一套：

```text
OmniLottie/
├── train.py
├── decoder.py
├── data/
│   ├── collator.py
│   └── lottie_dataset.py
├── lottie/
│   └── objects/
│       ├── lottie_tokenize.py
│       └── lottie_rule_tokenizer.py
├── inference.py
├── app.py
├── requirements.txt
└── requirements_train.txt
```

核心文件说明：
- `train.py`：训练主入口
- `decoder.py`：Qwen3.5 decoder，保留官方式 `resize_token_embeddings(...)` 路线
- `data/lottie_dataset.py`：多模态条件构造与 label mask
- `lottie/objects/lottie_tokenize.py`：原始官方规则主体
- `lottie/objects/lottie_rule_tokenizer.py`：在官方规则基础上做 Qwen3.5 词表平移
- `inference.py`：主推理脚本
- `app.py`：主 demo 脚本

已删除的旧包装代码不再属于当前主实现。

## Training Design

训练逻辑是：
1. 文本 / 图像 / 视频通过 Qwen processor 编码成条件输入
2. `lottie_json` 或 `sequence_text` 通过 Lottie 规则代码编码成 target token ids
3. 输入序列为 `[condition tokens] + [bos + lottie target + eos]`
4. condition 段 labels 置为 `-100`
5. 只对 Lottie target 段计算自回归 CE loss

这对应标准的 conditional autoregressive training。

## Qwen3.5 Migration Rule

本项目不是重新设计一套新 tokenizer 架构，而是直接沿用官方 Lottie 规则，再做 Qwen3.5 迁移：

- 原始官方 base vocab 视为 `151643`
- 新 base vocab 取自当前 `Qwen3.5` config
- `shift = new_base_vocab_size - 151643`
- 所有 Lottie command token / number token / special token / parameter offset 都按这个 `shift` 整体后移

因此迁移重点是：
- 扩大 `vocab_size`
- 扩大 embedding / LM head
- 平移官方规则中的 Lottie token id 区间

而不是引入一套额外的 hybrid head / hybrid tokenizer 框架。

## Default Backbone

默认 backbone：

```python
Qwen/Qwen3.5-9B
```

对应 `train.py`：

```python
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-9B")
```

## Install

建议 Python `3.10+`。

```bash
git clone https://github.com/wesfggfd/OmniLottie_training.git
cd OmniLottie_training/OmniLottie

conda create -n omnilottie_qwen35 python=3.10 -y
conda activate omnilottie_qwen35

pip install -r requirements_train.txt
```

如需运行推理 / demo，可额外安装：

```bash
pip install -r requirements.txt
```

## Dataset

`train.py` 会递归读取 `--data_path` 下所有 `*.parquet` 文件。

支持的条件字段：
- 文本：`desc_en` `motion_caption` `detail` `keywords_en` `short_desc` `medium_desc` `long_desc` `caption` `text` `prompt` `instruction`
- 图像：`image` `image_path` `keyframe` `keyframe_path`
- 视频：`video` `video_path` `rendered_video` `rendered_video_path`

支持的 target 字段：
- 优先：`sequence_text` `lottie_sequence` `token_sequence`
- 回退：`lottie_json` `json` `animation_json` `lottie`

最小样本要求：
- 至少一个条件输入
- 至少一个 target 字段

默认切分方式：
- 前 `98%` parquet 文件作为训练集
- 后 `2%` parquet 文件作为验证集

## Train

最小命令：

```bash
python train.py \
  --data_path /path/to/parquet_data \
  --output_dir /path/to/output
```

推荐：

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

主要参数：

| Argument | Default |
|---|---:|
| `--model_path` | `Qwen/Qwen3.5-9B` |
| `--max_seq_len` | `4096` |
| `--num_epochs` | `5` |
| `--per_device_batch` | `1` |
| `--grad_accum` | `8` |
| `--lora_lr` | `2e-5` |
| `--lottie_lr` | `5e-4` |
| `--resume_from` | `None` |

## Optimization

当前训练策略是：
- LoRA 训练主干适配参数
- 对扩展后 `embed_tokens.weight` / `lm_head.weight` 中新增的 Lottie token 行使用更高学习率

不是全参数微调，也不是自定义独立 `lottie_head` 模块。

## Inference

当前主推理入口：
- `inference.py`
- `app.py`

## Notes

- README 以当前仓库代码为准
- 当前仓库是 **Qwen3.5-9B 训练版本**
- 如果后续继续改结构，应优先保持与 `train.py` / `decoder.py` / `lottie/objects` 主链一致

## Citation

如果引用原始方法，请引用 OmniLottie 论文：

```bibtex
@article{yang2026omnilottie,
  title={OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens},
  author={Yiying Yang and Wei Cheng and Sijin Chen and Honghao Fu and Xianfang Zeng and Yujun Cai and Gang Yu and Xinjun Ma},
  journal={arXiv preprint arXiv:2603.02138},
  year={2026}
}
```
