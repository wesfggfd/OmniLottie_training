from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import random

import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from data.collator import LottieDataCollator
from data.lottie_dataset import (
    LottieFieldMap,
    MMLottieAutoregressiveDataset,
    TASK_IMAGE,
    TASK_MIXED,
    TASK_TEXT,
    TASK_VIDEO,
)
from decoder import LottieDecoder
from lottie.objects.lottie_rule_tokenizer import LottieRuleTokenizer


def build_optimizer(
    model: torch.nn.Module,
    lottie_lr: float,
    lora_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    lottie_params: List[torch.nn.Parameter] = []
    lora_params: List[torch.nn.Parameter] = []
    unexpected_trainable: List[str] = []
    lottie_param_ids = {
        id(model.transformer.get_input_embeddings().weight),
        id(model.transformer.get_output_embeddings().weight),
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in lottie_param_ids:
            param._optimizer_tensor_numel = param.numel()
            if getattr(param, "_is_lottie_row_masked", False):
                base_vocab_size = int(getattr(param, "_lottie_base_vocab_size"))
                param._effective_trainable_numel = max(0, param.shape[0] - base_vocab_size) * param.shape[1]
            else:
                param._effective_trainable_numel = param.numel()
            lottie_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        else:
            unexpected_trainable.append(name)

    if unexpected_trainable:
        raise ValueError(
            "Unexpected trainable parameters outside Lottie embeddings/LM head and LoRA modules: "
            + ", ".join(unexpected_trainable)
        )

    param_groups = []
    if lottie_params:
        param_groups.append({"params": lottie_params, "lr": lottie_lr, "weight_decay": 0.0})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
    )


def collect_parquet_files(dataset_root: Path) -> List[str]:
    parquet_files = sorted(str(path) for path in dataset_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}")
    return parquet_files


def split_parquet_files(parquet_files: List[str], seed: int) -> tuple[List[str], List[str]]:
    shuffled = list(parquet_files)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    split_index = max(1, int(len(shuffled) * 0.98))
    split_index = min(split_index, len(shuffled) - 1)
    return shuffled[:split_index], shuffled[split_index:]


def run_dataset_audit(
    parquet_files: List[str],
    *,
    processor,
    lottie_tokenizer: LottieRuleTokenizer,
    field_map: LottieFieldMap,
    max_seq_len: int,
    task_mode: str,
    task_ratios: Dict[str, float] | None = None,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    audits = MMLottieAutoregressiveDataset.audit_parquet_files(
        parquet_files,
        processor=processor,
        lottie_tokenizer=lottie_tokenizer,
        field_map=field_map,
        max_seq_len=max_seq_len,
        task_mode=task_mode,
        task_ratios=task_ratios,
        max_samples=max_samples,
    )
    total = len(audits)
    ok = sum(1 for item in audits if item.get("ok"))
    invalid = total - ok
    task_hist: Dict[str, int] = {}
    task_hist_labels: Dict[str, int] = {}
    errors: List[str] = []
    valid_indices: List[int] = []
    for item in audits:
        if item.get("ok"):
            task = str(item.get("task_type") or "unknown")
            task_hist[task] = task_hist.get(task, 0) + 1
            task_hist_labels[task] = task_hist_labels.get(task, 0) + 1
            valid_indices.append(int(item["sample_index"]))
        else:
            errors.append(str(item.get("error", "unknown error")))
    return {
        "total": total,
        "ok": ok,
        "invalid": invalid,
        "valid_ratio": ok / max(1, total),
        "task_histogram": task_hist,
        "errors": errors,
        "valid_indices": valid_indices,
        "records": audits,
    }


def report_trainable_parameters(model: torch.nn.Module, accelerator: Accelerator) -> None:
    total_params = 0
    trainable_params = 0
    effective_trainable_params = 0
    trainable_lines: List[str] = []
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
            effective_numel = numel
            if getattr(param, "_is_lottie_row_masked", False):
                base_vocab_size = int(getattr(param, "_lottie_base_vocab_size"))
                rows = max(0, param.shape[0] - base_vocab_size)
                effective_numel = rows * param.shape[1]
                trainable_lines.append(
                    f"  - {name}: shape={tuple(param.shape)} params={numel} effective_lottie_rows={effective_numel}"
                )
            else:
                trainable_lines.append(f"  - {name}: shape={tuple(param.shape)} params={numel}")
            effective_trainable_params += effective_numel
    accelerator.print(f"Total parameters: {total_params}")
    accelerator.print(f"Trainable parameters (optimizer tensors): {trainable_params}")
    accelerator.print(f"Effective trainable parameters (active rows/modules): {effective_trainable_params}")
    accelerator.print(f"Trainable ratio: {trainable_params / max(1, total_params):.6f}")
    accelerator.print(f"Effective trainable ratio: {effective_trainable_params / max(1, total_params):.6f}")
    for line in trainable_lines:
        accelerator.print(line)


def append_log(log_path: Path, event: Dict[str, Any]) -> None:
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    checkpoint_dir: Path,
    *,
    epoch: int,
    global_step: int,
    completed_batches_in_epoch: int,
    best_eval: float,
    best_step: int,
    no_improve_count: int,
    log_path: Path,
) -> None:
    state_dir = checkpoint_dir / "state"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(state_dir))
    if accelerator.is_main_process:
        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "completed_batches_in_epoch": completed_batches_in_epoch,
            "best_eval": best_eval,
            "best_step": best_step,
            "no_improve_count": no_improve_count,
        }
        write_json(checkpoint_dir / "trainer_state.json", trainer_state)
        latest_path = checkpoint_dir.parent / "latest_checkpoint.txt"
        latest_path.write_text(str(checkpoint_dir), encoding="utf-8")
        append_log(
            log_path,
            {
                "event": "checkpoint_saved",
                "path": str(checkpoint_dir),
                "epoch": epoch,
                "global_step": global_step,
                "completed_batches_in_epoch": completed_batches_in_epoch,
                "best_eval": best_eval,
                "best_step": best_step,
                "no_improve_count": no_improve_count,
            },
        )
        for old_checkpoint in sorted(checkpoint_dir.parent.glob("checkpoint-*")):
            if old_checkpoint != checkpoint_dir:
                shutil.rmtree(old_checkpoint, ignore_errors=True)
                append_log(
                    log_path,
                    {
                        "event": "checkpoint_deleted",
                        "path": str(old_checkpoint),
                    },
                )
    accelerator.wait_for_everyone()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_valid_indices(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(item) for item in payload]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported valid indices payload type at {path}: {type(payload)!r}")
    valid_indices = payload.get("valid_indices")
    if valid_indices is None:
        raise KeyError(f"Missing valid_indices in {path}")
    if not isinstance(valid_indices, list):
        raise TypeError(f"valid_indices must be a list in {path}")
    return [int(item) for item in valid_indices]


def resolve_resume_dir(resume_from: str, output_dir: Path) -> Path:
    if resume_from == "latest":
        latest_path = output_dir / "latest_checkpoint.txt"
        if not latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint marker found at {latest_path}")
        return Path(latest_path.read_text(encoding="utf-8").strip())
    return Path(resume_from)


def _mean_or_zero(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def resolve_task_mode(task_mode: str, task_entrypoint: str | None) -> str:
    if task_entrypoint is None:
        return task_mode
    if task_mode != "mixed" and task_mode != task_entrypoint:
        raise ValueError(
            f"task_mode={task_mode} conflicts with task_entrypoint={task_entrypoint}; use one or the other"
        )
    return task_entrypoint


def parse_task_ratios(raw: str | None) -> Dict[str, float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    ratios: Dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid task ratio entry: {part}. Expected key=value format.")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = float(value.strip())
        if value < 0:
            raise ValueError(f"Task ratio must be non-negative: {part}")
        alias_map = {"text": TASK_TEXT, "image": TASK_IMAGE, "video": TASK_VIDEO}
        normalized_key = alias_map.get(key, key)
        if normalized_key not in {TASK_TEXT, TASK_IMAGE, TASK_VIDEO}:
            raise ValueError(f"Unsupported task ratio key: {key}")
        ratios[normalized_key] = value
    return ratios


def _load_best_eval_from_stage(stage_dir: Path) -> float | None:
    log_path = stage_dir / "train_log.jsonl"
    if not log_path.exists():
        return None
    best_value = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("event") in {"best_updated", "run_finished", "eval_step"} and event.get("eval_loss") is not None:
            value = float(event["eval_loss"])
            if best_value is None or value < best_value:
                best_value = value
        elif event.get("event") == "best_updated" and event.get("best_eval") is not None:
            value = float(event["best_eval"])
            if best_value is None or value < best_value:
                best_value = value
    return best_value


def resolve_mixed_task_ratios(args, output_dir: Path, accelerator: Accelerator) -> Dict[str, float]:
    explicit = parse_task_ratios(args.mixed_task_ratios)
    if explicit is not None:
        resolved = {TASK_TEXT: 0.0, TASK_IMAGE: 0.0, TASK_VIDEO: 0.0}
        resolved.update(explicit)
        if sum(resolved.values()) <= 0:
            raise ValueError(f"Explicit mixed_task_ratios must contain a positive value: {args.mixed_task_ratios}")
        accelerator.print(f"Using explicit mixed task ratios: {resolved}")
        return resolved

    strategy = str(args.mixed_ratio_strategy or "adaptive_stage_loss").strip().lower()
    if strategy == "equal":
        resolved = {TASK_TEXT: 1.0, TASK_IMAGE: 1.0, TASK_VIDEO: 1.0}
        accelerator.print(f"Using equal mixed task ratios: {resolved}")
        return resolved

    if strategy != "adaptive_stage_loss":
        raise ValueError(f"Unsupported mixed_ratio_strategy: {args.mixed_ratio_strategy}")

    stage_root = Path(args.mixed_ratio_stage_root) if args.mixed_ratio_stage_root else output_dir.parent
    stage_dirs = {
        TASK_TEXT: stage_root / "stage1_text_to_lottie",
        TASK_IMAGE: stage_root / "stage2_text_image_to_lottie",
        TASK_VIDEO: stage_root / "stage3_video_to_lottie",
    }
    stage_losses = {task: _load_best_eval_from_stage(path) for task, path in stage_dirs.items()}
    if any(value is None for value in stage_losses.values()):
        fallback = {TASK_TEXT: 1.0, TASK_IMAGE: 1.2, TASK_VIDEO: 1.4}
        accelerator.print(
            "Falling back to heuristic mixed task ratios because stage eval logs were incomplete: "
            f"{stage_losses}; fallback={fallback}"
        )
        return fallback

    exponent = float(args.mixed_ratio_temperature)
    floor = float(args.mixed_ratio_floor)
    resolved = {}
    for task, loss in stage_losses.items():
        assert loss is not None
        resolved[task] = floor + max(loss, 1e-6) ** exponent
    accelerator.print(f"Using adaptive mixed task ratios from stage losses: losses={stage_losses}, ratios={resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B OmniLottie LoRA training")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_lr", type=float, default=2e-5)
    parser.add_argument("--lottie_lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help="Path to a previous stage's best/ directory (containing pytorch_model.bin). "
             "Loads model weights only — optimizer and scheduler start fresh. "
             "Use this for multi-stage training (e.g. text→image→video→mixed).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--sanity_eval_batches", type=int, default=8)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--audit_max_samples", type=int, default=None)
    parser.add_argument("--task_mode", type=str, default="mixed", choices=["mixed", "text", "image", "video"])
    parser.add_argument("--task_entrypoint", type=str, default=None, choices=["text", "image", "video"])
    parser.add_argument("--valid_indices_path", type=str, default=None)
    parser.add_argument("--skip_audit", action="store_true")
    parser.add_argument("--audit_fail_on_empty", action="store_true", default=True)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="Cap the number of gradient-update steps per epoch. "
                             "Useful to shorten very large datasets.")
    parser.add_argument("--mixed_task_ratios", type=str, default=None,
                        help="Explicit mixed task ratios, e.g. text=1,image=1.2,video=1.4")
    parser.add_argument("--mixed_ratio_strategy", type=str, default="adaptive_stage_loss",
                        choices=["adaptive_stage_loss", "equal"],
                        help="How to derive mixed-stage task ratios when --mixed_task_ratios is not provided.")
    parser.add_argument("--mixed_ratio_stage_root", type=str, default=None,
                        help="Directory containing stage1/2/3 outputs used by adaptive mixed ratio resolution.")
    parser.add_argument("--mixed_ratio_temperature", type=float, default=1.0,
                        help="Exponent applied to stage losses when deriving adaptive mixed ratios.")
    parser.add_argument("--mixed_ratio_floor", type=float, default=0.25,
                        help="Minimum additive floor for each adaptive mixed task ratio.")
    args = parser.parse_args()

    resolved_task_mode = resolve_task_mode(args.task_mode, args.task_entrypoint)

    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    mixed_task_ratios = resolve_mixed_task_ratios(args, output_dir, accelerator) if resolved_task_mode == TASK_MIXED else None
    if accelerator.is_main_process:
        training_args = dict(vars(args))
        training_args["resolved_task_mode"] = resolved_task_mode
        training_args["resolved_mixed_task_ratios"] = mixed_task_ratios
        write_json(output_dir / "training_args.json", training_args)
    accelerator.wait_for_everyone()
    log_path = output_dir / "train_log.jsonl"

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    lottie_tokenizer = LottieRuleTokenizer(args.model_path)
    model = LottieDecoder(
        pix_len=4560,
        text_len=1500,
        model_path=args.model_path,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LottieDecoder.default_lora_target_modules(),
    )
    model.transformer = get_peft_model(model.transformer, lora_config)
    model.transformer.get_input_embeddings().weight.requires_grad_(True)
    model.transformer.get_output_embeddings().weight.requires_grad_(True)
    model.transformer.gradient_checkpointing_enable()

    if args.init_weights is not None:
        init_path = Path(args.init_weights)
        weight_file = init_path / "pytorch_model.bin"
        if not weight_file.exists():
            available = sorted(p.name for p in init_path.iterdir()) if init_path.exists() else []
            raise FileNotFoundError(
                f"init_weights file not found: {weight_file}. "
                f"Directory exists={init_path.exists()} contents={available[:20]}"
            )
        accelerator.print(f"Loading stage-init weights from {weight_file} ...")
        state_dict = torch.load(str(weight_file), map_location="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        accelerator.print(f"  Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            accelerator.print(f"  Missing preview: {missing[:8]}")
        if unexpected:
            accelerator.print(f"  Unexpected preview: {unexpected[:8]}")
        del state_dict

    field_map = LottieFieldMap()
    parquet_files = collect_parquet_files(Path(args.data_path))
    train_files, eval_files = split_parquet_files(parquet_files, seed=args.seed)
    accelerator.print(f"Collected {len(parquet_files)} parquet files")
    accelerator.print(f"Train parquet files: {len(train_files)}")
    accelerator.print(f"Eval parquet files: {len(eval_files)}")

    audit_summary: Dict[str, Any] | None = None
    valid_indices: List[int] | None = None
    if not args.skip_audit:
        audit_summary = run_dataset_audit(
            train_files,
            processor=processor,
            lottie_tokenizer=lottie_tokenizer,
            field_map=field_map,
            max_seq_len=args.max_seq_len,
            task_mode=resolved_task_mode,
            task_ratios=mixed_task_ratios,
            max_samples=args.audit_max_samples,
        )
        valid_indices = list(audit_summary["valid_indices"])
        if accelerator.is_main_process:
            append_log(
                log_path,
                {
                    "event": "dataset_audit",
                    "split": "train",
                    "task_mode": resolved_task_mode,
                    "total": audit_summary["total"],
                    "ok": audit_summary["ok"],
                    "invalid": audit_summary["invalid"],
                    "valid_ratio": audit_summary["valid_ratio"],
                    "task_histogram": audit_summary["task_histogram"],
                    "sample_errors": audit_summary["errors"][:20],
                },
            )
            if args.valid_indices_path is not None:
                valid_indices_path = Path(args.valid_indices_path)
                write_json(
                    valid_indices_path,
                    {
                        "task_mode": resolved_task_mode,
                        "source_parquet_files": train_files,
                        "total": audit_summary["total"],
                        "ok": audit_summary["ok"],
                        "invalid": audit_summary["invalid"],
                        "valid_ratio": audit_summary["valid_ratio"],
                        "valid_indices": audit_summary["valid_indices"],
                    },
                )
        accelerator.print(
            f"Dataset audit: total={audit_summary['total']} ok={audit_summary['ok']} invalid={audit_summary['invalid']} valid_ratio={audit_summary['valid_ratio']:.3f}"
        )
        if audit_summary["ok"] <= 0 and args.audit_fail_on_empty:
            raise RuntimeError(
                f"Dataset audit found no valid training samples for task_mode={resolved_task_mode}; check source parquet files and field mapping"
            )
        if args.audit_only:
            return
    else:
        if args.valid_indices_path is not None:
            valid_indices = load_valid_indices(Path(args.valid_indices_path))
        accelerator.print(
            f"Skipping dataset audit; using {'provided' if valid_indices is not None else 'all'} indices for task_mode={resolved_task_mode}"
        )

    train_dataset = MMLottieAutoregressiveDataset.from_parquet_files(
        train_files,
        processor=processor,
        lottie_tokenizer=lottie_tokenizer,
        field_map=field_map,
        max_seq_len=args.max_seq_len,
        task_mode=resolved_task_mode,
        task_ratios=mixed_task_ratios,
        row_indices=valid_indices,
    )
    if len(train_dataset) <= 0:
        raise RuntimeError(f"Train dataset is empty for task_mode={resolved_task_mode}")
    accelerator.print(
        f"Train dataset built: task_mode={resolved_task_mode} samples={len(train_dataset)} valid_indices={'yes' if valid_indices is not None else 'no'}"
    )
    train_task_hist = train_dataset.task_view_histogram()
    train_source_hist_sample_limit = 4096
    train_source_hist = train_dataset.source_histogram(max_unique_samples=train_source_hist_sample_limit)
    accelerator.print(f"Train task-view histogram: {train_task_hist}")
    accelerator.print(
        f"Train source histogram (first {train_source_hist_sample_limit} unique samples): {train_source_hist}"
    )
    if accelerator.is_main_process:
        append_log(
            log_path,
            {
                "event": "train_dataset_summary",
                "task_mode": resolved_task_mode,
                "mixed_task_ratios": mixed_task_ratios,
                "task_view_histogram": train_task_hist,
                "source_histogram": train_source_hist,
                "source_histogram_sample_limit": train_source_hist_sample_limit,
                "sample_count": len(train_dataset),
            },
        )
    eval_dataset = (
        MMLottieAutoregressiveDataset.from_parquet_files(
            eval_files,
            processor=processor,
            lottie_tokenizer=lottie_tokenizer,
            field_map=field_map,
            max_seq_len=args.max_seq_len,
            task_mode=resolved_task_mode,
            task_ratios=mixed_task_ratios,
        )
        if eval_files
        else None
    )
    if eval_dataset is not None:
        eval_task_hist = eval_dataset.task_view_histogram()
        eval_source_hist_sample_limit = 2048
        eval_source_hist = eval_dataset.source_histogram(max_unique_samples=eval_source_hist_sample_limit)
        accelerator.print(f"Eval task-view histogram: {eval_task_hist}")
        accelerator.print(
            f"Eval source histogram (first {eval_source_hist_sample_limit} unique samples): {eval_source_hist}"
        )
        if accelerator.is_main_process:
            append_log(
                log_path,
                {
                    "event": "eval_dataset_summary",
                    "task_mode": resolved_task_mode,
                    "task_view_histogram": eval_task_hist,
                    "source_histogram": eval_source_hist,
                    "source_histogram_sample_limit": eval_source_hist_sample_limit,
                    "sample_count": len(eval_dataset),
                },
            )

    sample_probe = None
    probe_limit = min(64, len(train_dataset))
    for probe_idx in range(probe_limit):
        try:
            sample_probe = train_dataset[probe_idx]
            break
        except Exception as exc:
            if accelerator.is_main_process:
                append_log(
                    log_path,
                    {
                        "event": "sample_probe_retry",
                        "probe_idx": probe_idx,
                        "error": str(exc),
                    },
                )
            continue
    if sample_probe is None:
        if args.skip_audit:
            raise RuntimeError(
                "Failed to find a valid sample probe in the first 64 dataset items while skip_audit=True; provide --valid_indices_path from an audit run"
            )
        raise RuntimeError("Failed to find a valid sample probe in the first 64 dataset items; check dataset audit results")
    accelerator.print(
        "Sample probe passed: "
        f"input_len={sample_probe['input_ids'].shape[0]} "
        f"mm_token_type_ids={'mm_token_type_ids' in sample_probe} "
        f"task_type={sample_probe['task_type']}"
    )

    collator = LottieDataCollator(pad_token_id=model.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.per_device_batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        if eval_dataset is not None
        else None
    )

    optimizer = build_optimizer(model, args.lottie_lr, args.lora_lr, weight_decay=0.01)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    if args.max_steps_per_epoch is not None:
        updates_per_epoch = min(updates_per_epoch, args.max_steps_per_epoch)
    total_update_steps = max(1, updates_per_epoch * args.num_epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    if eval_loader is not None:
        model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    report_trainable_parameters(model, accelerator)

    global_step = 0
    best_eval = float("inf")
    best_step = 0
    no_improve_count = 0
    start_epoch = 0
    completed_batches_in_epoch = 0
    should_stop = False

    if accelerator.is_main_process:
        append_log(
            log_path,
            {
                "event": "run_started",
                "output_dir": str(output_dir),
                "resume_from": args.resume_from,
                "num_epochs": args.num_epochs,
                "grad_accum": args.grad_accum,
                "train_parquet_files": len(train_files),
                "eval_parquet_files": len(eval_files),
            },
        )
    accelerator.wait_for_everyone()

    if args.resume_from:
        resume_dir = resolve_resume_dir(args.resume_from, output_dir)
        trainer_state_path = resume_dir / "trainer_state.json"
        if not trainer_state_path.exists():
            raise FileNotFoundError(f"Missing trainer state at {trainer_state_path}")
        accelerator.print(f"Resuming from checkpoint: {resume_dir}")
        accelerator.load_state(str(resume_dir / "state"))
        trainer_state = load_json(trainer_state_path)
        global_step = int(trainer_state["global_step"])
        best_eval = float(trainer_state["best_eval"])
        best_step = int(trainer_state.get("best_step", 0))
        start_epoch = int(trainer_state["epoch"])
        completed_batches_in_epoch = int(trainer_state["completed_batches_in_epoch"])
        no_improve_count = int(trainer_state.get("no_improve_count", 0))
        if accelerator.is_main_process:
            append_log(
                log_path,
                {
                    "event": "run_resumed",
                    "path": str(resume_dir),
                    "epoch": start_epoch,
                    "global_step": global_step,
                    "completed_batches_in_epoch": completed_batches_in_epoch,
                    "best_eval": best_eval,
                    "best_step": best_step,
                    "no_improve_count": no_improve_count,
                },
            )
        accelerator.wait_for_everyone()

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_train_loader = train_loader
        if epoch == start_epoch and completed_batches_in_epoch > 0:
            epoch_train_loader = skip_first_batches(train_loader, completed_batches_in_epoch)

        current_batch_index = completed_batches_in_epoch
        epoch_step = 0
        for batch in epoch_train_loader:
            current_batch_index += 1
            epoch_step += 1
            if args.max_steps_per_epoch is not None and epoch_step > args.max_steps_per_epoch:
                break
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    loss_value = loss.detach().float().item()
                    accelerator.print(
                        f"epoch={epoch} step={global_step} loss={loss_value:.4f}"
                    )
                    if accelerator.is_main_process:
                        append_log(
                            log_path,
                            {
                                "event": "train_step",
                                "epoch": epoch,
                                "global_step": global_step,
                                "loss": loss_value,
                            },
                        )

                if eval_loader is not None and global_step % args.eval_steps == 0:
                    eval_metrics = run_eval(
                        accelerator,
                        model,
                        eval_loader,
                        max_batches=args.sanity_eval_batches,
                    )
                    eval_loss = eval_metrics["eval_loss"]
                    accelerator.print(
                        f"epoch={epoch} step={global_step} eval_loss={eval_loss:.4f} "
                        f"decodeable_rate={eval_metrics['decodeable_rate']:.3f} "
                        f"eos_label_rate={eval_metrics['eos_label_rate']:.3f} "
                        f"mean_target_length={eval_metrics['mean_target_length']:.1f}"
                    )
                    if accelerator.is_main_process:
                        append_log(
                            log_path,
                            {
                                "event": "eval_step",
                                "epoch": epoch,
                                "global_step": global_step,
                                **eval_metrics,
                            },
                        )
                    improved = eval_loss < (best_eval - args.early_stopping_min_delta)
                    if improved:
                        best_eval = eval_loss
                        best_step = global_step
                        no_improve_count = 0
                        save_model(accelerator, model, output_dir / "best", processor=processor)
                        if accelerator.is_main_process:
                            append_log(
                                log_path,
                                {
                                    "event": "best_updated",
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "best_eval": best_eval,
                                },
                            )
                    else:
                        no_improve_count += 1
                        if accelerator.is_main_process:
                            append_log(
                                log_path,
                                {
                                    "event": "early_stop_wait",
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "best_eval": best_eval,
                                    "eval_loss": eval_loss,
                                    "no_improve_count": no_improve_count,
                                    "patience": args.early_stopping_patience,
                                },
                            )
                        if no_improve_count >= args.early_stopping_patience:
                            should_stop = True
                            accelerator.print(
                                f"Early stopping at epoch={epoch} step={global_step} best_eval={best_eval:.4f}"
                            )
                            if accelerator.is_main_process:
                                append_log(
                                    log_path,
                                    {
                                        "event": "early_stopped",
                                        "epoch": epoch,
                                        "global_step": global_step,
                                        "best_eval": best_eval,
                                        "best_step": best_step,
                                    },
                                )

                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        output_dir / f"checkpoint-{global_step}",
                        epoch=epoch,
                        global_step=global_step,
                        completed_batches_in_epoch=current_batch_index,
                        best_eval=best_eval,
                        log_path=log_path,
                        best_step=best_step,
                        no_improve_count=no_improve_count,
                    )
                if should_stop:
                    break
        completed_batches_in_epoch = 0
        if should_stop:
            break

    final_path = output_dir / "final"
    best_path = output_dir / "best"
    if eval_loader is None and not best_path.exists():
        save_model(accelerator, model, best_path, processor=processor)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if final_path.exists():
            shutil.rmtree(final_path, ignore_errors=True)
        if best_path.exists():
            shutil.copytree(best_path, final_path)
        else:
            raise FileNotFoundError(f"Best model path missing at {best_path}")
    accelerator.wait_for_everyone()
    if eval_loader is not None:
        final_eval_metrics = run_eval(
            accelerator,
            model,
            eval_loader,
            max_batches=args.sanity_eval_batches,
        )
        accelerator.print(
            "final_eval "
            f"loss={final_eval_metrics['eval_loss']:.4f} "
            f"decodeable_rate={final_eval_metrics['decodeable_rate']:.3f} "
            f"eos_label_rate={final_eval_metrics['eos_label_rate']:.3f}"
        )
    else:
        final_eval_metrics = {
            "eval_loss": best_eval,
            "decodeable_rate": 0.0,
            "eos_label_rate": 0.0,
            "mean_target_length": 0.0,
            "truncated_condition_rate": 0.0,
            "truncated_target_rate": 0.0,
        }

    if accelerator.is_main_process:
        append_log(
            log_path,
            {
                "event": "run_finished",
                "global_step": global_step,
                "best_eval": best_eval,
                "best_step": best_step,
                "final_model_path": str(output_dir / "final"),
                **final_eval_metrics,
            },
        )


@torch.no_grad()
def run_eval(
    accelerator: Accelerator,
    model: torch.nn.Module,
    eval_loader: DataLoader,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    decodeable_flags: List[float] = []
    eos_label_flags: List[float] = []
    target_lengths: List[float] = []
    truncated_condition_flags: List[float] = []
    truncated_target_flags: List[float] = []

    for batch_idx, batch in enumerate(eval_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        outputs = model(**batch)
        loss = outputs.loss.detach().float()
        losses.append(accelerator.gather_for_metrics(loss.unsqueeze(0)).mean().item())

        labels = batch["labels"]
        valid_target = labels.ne(-100)
        per_sample_decodeable = valid_target.any(dim=1).float()
        _unwrapped = accelerator.unwrap_model(model)
        per_sample_has_eos = labels.eq(_unwrapped.eos_token_id).any(dim=1).float()
        per_sample_target_length = valid_target.sum(dim=1).float()

        decodeable_flags.extend(accelerator.gather_for_metrics(per_sample_decodeable).cpu().tolist())
        eos_label_flags.extend(accelerator.gather_for_metrics(per_sample_has_eos).cpu().tolist())
        target_lengths.extend(accelerator.gather_for_metrics(per_sample_target_length).cpu().tolist())

        if "truncated_condition" in batch:
            truncated_condition = torch.tensor(batch["truncated_condition"], device=labels.device, dtype=torch.float32)
            truncated_condition_flags.extend(accelerator.gather_for_metrics(truncated_condition).cpu().tolist())
        if "truncated_target" in batch:
            truncated_target = torch.tensor(batch["truncated_target"], device=labels.device, dtype=torch.float32)
            truncated_target_flags.extend(accelerator.gather_for_metrics(truncated_target).cpu().tolist())

    model.train()
    return {
        "eval_loss": _mean_or_zero(losses),
        "decodeable_rate": _mean_or_zero(decodeable_flags),
        "eos_label_rate": _mean_or_zero(eos_label_flags),
        "mean_target_length": _mean_or_zero(target_lengths),
        "truncated_condition_rate": _mean_or_zero(truncated_condition_flags),
        "truncated_target_rate": _mean_or_zero(truncated_target_flags),
    }


def save_model(accelerator: Accelerator, model: torch.nn.Module, path: Path, processor=None) -> None:
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        replace_dir(path)
        unwrapped.save_pretrained(path)
        if processor is not None:
            processor.save_pretrained(path)


if __name__ == "__main__":
    main()

