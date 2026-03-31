

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

当前主推理入口是 `inference.py`；`app.py` 主要用于本地 Gradio demo。

### CLI 推理

`inference.py` 目前支持：
- 单条文本 / 图片 / 视频推理
- `mmlottie_bench` benchmark 推理
- 批量文本推理
- tokenizer roundtrip 验证
- `--gpu_id` 指定 GPU，或自动选择空闲卡
- `--num_candidates` best-of-N 候选选择

最小示例：

```bash
python inference.py \
  --sketch_weight /path/to/output/final \
  --tokenizer_name Qwen/Qwen3.5-9B \
  --single_text "a blue bird appearing, pulsing while sliding downward" \
  --output_dir /path/to/infer_outputs
```

如果要跑 roundtrip 验证：

```bash
python inference.py \
  --sketch_weight /path/to/output/final \
  --tokenizer_name Qwen/Qwen3.5-9B \
  --output_dir /path/to/output \
  --roundtrip_validate \
  --roundtrip_split real \
  --max_samples 64
```

当前推理策略仍然保持 OmniLottie 的主线定义：
- 多模态条件仍由 Qwen processor 编码
- target 仍然是自回归生成的 Lottie token 序列
- 生成终止仍以 Lottie `eos_token_id` 为边界

当前仓库在解码约束上是轻量级版本，而不是完整 grammar-constrained decoding：
- 首个 target token 强制为 `LOTTIE_BOS`
- target 段内禁止再次生成 `LOTTIE_BOS`
- 禁止生成 `PAD`
- 将生成 token 约束在“base tokenizer 可解释区 + Lottie 扩展区”内，避免明显越界 token
- 支持 best-of-N 候选后处理选择

### Demo

`app.py` 可以启动 Gradio 页面，但当前代码里默认 checkpoint 路径仍是占位值：

```python
checkpoint_path = "/PATH/TO/OmniLottie"
```

因此在直接运行前，需要先把它改成实际训练导出的目录，或自行改造成从命令行参数 / 环境变量读取。

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

## Maintainers

- wesfggfd
- Claude

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
