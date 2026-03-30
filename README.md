# OmniLottie Training — Qwen3.5-9B

This repository contains the **training, inference, validation, and utility code** for a Qwen3.5-based OmniLottie migration implementation.

The code keeps the core OmniLottie formulation:
- multimodal conditioning
- autoregressive generation of Lottie tokens
- target-only loss on the Lottie segment
- extending the base model vocabulary with the Lottie token space

At the same time, it adapts the original method to **`Qwen/Qwen3.5-9B`** by shifting the official Lottie token rules to the new base vocabulary and resizing the embedding / LM head accordingly.

This repository is therefore best understood as a **Qwen3.5 migration / reimplementation path for OmniLottie**, rather than a strict paper-backbone reproduction of the original Qwen2.5-VL setup.

> Paper reference: **OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens**
> arXiv:2603.02138 (2026)

---

## Scope of this repository

This repository is intended to store **source code and scripts only**:
- model code
- data pipeline code
- training launchers
- auditing and utility scripts
- inference code
- documentation

It is **not** intended to version large local artifacts such as:
- training outputs
- logs
- checkpoints
- downloaded datasets
- temporary experiment files

These artifacts are excluded through `.gitignore` before pushing.

---

## Repository layout

```text
OmniLottie_training/
├── OmniLottie/
│   ├── train.py
│   ├── decoder.py
│   ├── inference.py
│   ├── app.py
│   ├── data/
│   │   ├── collator.py
│   │   ├── lottie_dataset.py
│   │   ├── task_constants.py
│   │   └── task_sampler.py
│   └── lottie/objects/
│       ├── lottie_tokenize.py
│       └── lottie_rule_tokenizer.py
├── run_train_stages.sh
├── run_train_all_stages.sh
├── valid_auditing.sh
├── gpu_watcher.sh
└── README.md
```

Important files:
- `OmniLottie/train.py`: training entrypoint
- `OmniLottie/decoder.py`: Qwen3.5-based decoder with resized embeddings
- `OmniLottie/data/lottie_dataset.py`: multimodal sample building and label masking
- `OmniLottie/data/collator.py`: variable-length batching logic
- `OmniLottie/data/task_sampler.py`: task-aware sampling helpers
- `OmniLottie/lottie/objects/lottie_tokenize.py`: official Lottie rule tokenizer path
- `OmniLottie/lottie/objects/lottie_rule_tokenizer.py`: Qwen3.5-shifted Lottie token layout
- `OmniLottie/inference.py`: inference / benchmark / validation entrypoint
- `run_train_stages.sh`: staged launcher used for sequential training
- `valid_auditing.sh`: audit-only dataset validation script
- `gpu_watcher.sh`: utility for immediately claiming newly freed non-gpu0 idle GPUs

---

## Training design

Training follows conditional autoregressive language modeling:

1. text / image / video conditions are encoded into prefix tokens
2. Lottie target data is converted into token ids with the rule-based tokenizer
3. the final sequence is `[condition tokens] + [bos + lottie target + eos]`
4. condition labels are masked with `-100`
5. loss is computed only on the Lottie target segment

This preserves the target-only OmniLottie training formulation.

---

## Qwen3.5 migration rule

This project does **not** redesign the tokenizer from scratch.
Instead, it keeps the official Lottie rule space and shifts it to the Qwen3.5 vocabulary.

Core idea:
- original base vocab: `151643`
- new base vocab: taken from Qwen3.5 config
- `shift = new_base_vocab_size - 151643`
- all Lottie command tokens / number tokens / special tokens / offsets are shifted by `shift`

So the migration mainly consists of:
- expanding `vocab_size`
- resizing embeddings and LM head
- shifting the official Lottie token id ranges

---

## Backbone

Default backbone:

```python
Qwen/Qwen3.5-9B
```

Defined in `OmniLottie/train.py`.

---

## Installation

Recommended: Python `3.10+`.

```bash
git clone https://github.com/wesfggfd/OmniLottie_training.git
cd OmniLottie_training/OmniLottie

conda create -n omnilottie_qwen35 python=3.10 -y
conda activate omnilottie_qwen35

pip install -r requirements_train.txt
```

If you also want demo / inference dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset expectations

`train.py` recursively reads all `*.parquet` files under `--data_path`.

Supported condition fields include:
- text: `desc_en`, `motion_caption`, `detail`, `keywords_en`, `short_desc`, `medium_desc`, `long_desc`, `caption`, `text`, `prompt`, `instruction`
- image: `image`, `image_path`, `keyframe`, `keyframe_path`
- video: `video`, `video_path`, `rendered_video`, `rendered_video_path`

Supported target fields include:
- preferred: `sequence_text`, `lottie_sequence`, `token_sequence`
- fallback: `lottie_json`, `json`, `animation_json`, `lottie`

Minimum sample requirement:
- at least one condition input
- at least one target field

Default split rule:
- first `98%` of parquet files for training
- last `2%` for evaluation

Recommended training corpus for this Qwen3.5 migration path:
- `/opt/liblibai-models/user-workspace2/dataset/MMLottie-2M/data/Lottie_merged_70_30`
- this local merged directory encodes the paper-aligned **70/30 file-level mixture** used in this project
- local verification: `34` native `Lottie` parquet files + `15` `Lottie_SVG` parquet files

---

## Training scripts

### Single staged launcher

```bash
bash run_train_stages.sh 1
```

The current staged setup is organized as progressive task modes:
- stage 1: `text`
- stage 2: `image`
- stage 3: `video`
- stage 4: `mixed`

For stage 4, this project no longer uses a fixed `1:1:1` task ratio. Instead it supports:
- explicit ratios via `--mixed_task_ratios text=...,image=...,video=...`
- an adaptive default strategy (`adaptive_stage_loss`) that increases sampling weight for task branches whose earlier stage still has higher validation loss

This adaptive strategy is intended to rebalance optimization pressure across text / image / video rather than assuming equal difficulty.

### Full pipeline launcher

```bash
bash run_train_all_stages.sh
```

### Direct training entrypoint

```bash
cd OmniLottie
accelerate launch train.py \
  --model_path Qwen/Qwen3.5-9B \
  --data_path /path/to/parquet_data \
  --output_dir /path/to/output
```

---

## GPU occupation policy

This repo uses the following policy for non-training utility jobs such as validation / inference helpers:
- prefer **non-gpu0** devices
- only treat a GPU as idle when **`utilization.gpu == 0` and `memory.used == 0`**
- once a GPU becomes idle, claim it immediately

Relevant scripts / code:
- `valid_auditing.sh`
- `gpu_watcher.sh`
- `OmniLottie/inference.py`

This is intended to avoid disturbing the main training GPU while still grabbing newly freed devices quickly.

---

## Inference and validation

Main inference entrypoints:
- `OmniLottie/inference.py`
- `OmniLottie/app.py`

Roundtrip validation, benchmark inference, and device-aware inference are all handled from `inference.py`.

The inference path currently focuses on:
- single-sample and benchmark inference entrypoints
- validity-aware generation diagnostics such as decode success, EOS / BOS rate, and generated token statistics
- benchmark-compatible artifact export for downstream evaluation

---

## Notes

- this repo tracks the **Qwen3.5-9B training implementation**
- the README should match the code in this repository
- if the architecture is extended further, keep it aligned with the main path in `train.py`, `decoder.py`, and `lottie/objects`

---

## Citation

```bibtex
@article{yang2026omnilottie,
  title={OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens},
  author={Yiying Yang and Wei Cheng and Sijin Chen and Honghao Fu and Xianfang Zeng and Yujun Cai and Gang Yu and Xinjun Ma},
  journal={arXiv preprint arXiv:2603.02138},
  year={2026}
}
```
