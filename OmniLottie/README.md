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
└── requirements.txt
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
  --per_device_batch 2 \
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
| `--per_device_batch` | `2` |
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

当前推理策略仍然保持 OmniLottie 的主线定义：
- 多模态条件仍由 Qwen processor 编码
- target 仍然是自回归生成的 Lottie token 序列
- 生成终止仍以 Lottie `eos_token_id` 为边界

但当前仓库在解码约束上仍然是 **轻量级版本**，不是完整 grammar-constrained decoding：
- 首个 target token 强制为 `LOTTIE_BOS`
- target 段内禁止再次生成 `LOTTIE_BOS`
- 禁止生成 `PAD`
- 将生成 token 约束在“base tokenizer 可解释区 + Lottie 扩展区”内，避免明显越界 token
- 支持 best-of-N 候选后处理选择

这意味着当前版本已经比“完全裸生成”更接近官方 token 边界设定，也比只做 BOS/PAD 边界控制更稳一些，但**仍未完全等价于论文/官方 repo 未来可能扩展的更强结构化约束解码**。

## Current Scope and Limitations

当前实现的目标是：
- **只替换 backbone 到 `Qwen3.5-9B`**
- 保留原 OmniLottie 的多模态条件理解能力迁移路径
- 将能力集中迁移到高效生成 Lottie token 上

因此它更准确地说是：
- **official OmniLottie method 的 Qwen3.5 migration 版**
- 而不是一套重新设计的新框架

同时也需要明确：
- 当前 `lottie_rule_tokenizer.py` 是在官方规则基础上做词表平移和工程化校验
- 当前训练主线已经覆盖：target-only loss、扩表、LoRA、embedding/lm_head 新增 token 行训练
- 当前数据链路已经覆盖：lazy parquet 读取、condition/target budget 截断、同 task-type batch 约束
- 当前推理链路已经补上：greedy decoding 修正、best-of-N 选择、最小边界约束
- **当前版本仍不等于“完整官方 reproduction”**，尤其在更强的结构合法性约束、系统化 fidelity/eval、全 schema 覆盖上仍有继续补齐空间

## Validation and Inference Device Policy

当前仓库从这一版开始，验证与推理支持显式指定 GPU，也支持自动选择空闲卡：
- 新增 `--gpu_id`
- 如果不显式指定，会优先自动选择 **非 gpu0** 的空闲 GPU
- 这用于避免占用训练主卡，符合“验证和后续推理尽量放到除 gpu0 外的空闲卡”这一使用策略

同时新增了一个轻量 roundtrip 验证入口：

```bash
python inference.py \
  --tokenizer_name Qwen/Qwen3.5-9B \
  --output_dir /path/to/output \
  --roundtrip_validate \
  --roundtrip_split real \
  --max_samples 64
```

它主要验证：
- `lottie_json -> token_ids -> lottie_json` 规则链路能否跑通
- 是否存在明显的 tokenization / detokenization 崩溃

此外，batch 推理和 benchmark 推理会额外输出 validity-aware metrics，例如：
- `decode_success_rate`
- `valid_json_rate`
- `eos_rate`
- `bos_rate`
- `avg_generated_tokens`
- `avg_layers_on_success`

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
