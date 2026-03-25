from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from data.collator import LottieDataCollator
from data.lottie_dataset import LottieFieldMap, MMLottieAutoregressiveDataset
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
    other_params: List[torch.nn.Parameter] = []
    lottie_param_ids = {
        id(model.transformer.get_input_embeddings().weight),
        id(model.transformer.get_output_embeddings().weight),
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in lottie_param_ids:
            lottie_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if lottie_params:
        param_groups.append({"params": lottie_params, "lr": lottie_lr, "weight_decay": weight_decay})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "lr": lora_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
    )


def collect_parquet_files(dataset_root: Path) -> List[str]:
    parquet_files = sorted(str(path) for path in dataset_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}")
    return parquet_files


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


def resolve_resume_dir(resume_from: str, output_dir: Path) -> Path:
    if resume_from == "latest":
        latest_path = output_dir / "latest_checkpoint.txt"
        if not latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint marker found at {latest_path}")
        return Path(latest_path.read_text(encoding="utf-8").strip())
    return Path(resume_from)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "training_args.json", vars(args))
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

    field_map = LottieFieldMap()
    parquet_files = collect_parquet_files(Path(args.data_path))
    split_index = max(1, int(len(parquet_files) * 0.98))
    train_dataset = MMLottieAutoregressiveDataset.from_parquet_files(
        parquet_files[:split_index],
        processor=processor,
        lottie_tokenizer=lottie_tokenizer,
        field_map=field_map,
        max_seq_len=args.max_seq_len,
    )
    eval_dataset = MMLottieAutoregressiveDataset.from_parquet_files(
        parquet_files[split_index:] or parquet_files[:1],
        processor=processor,
        lottie_tokenizer=lottie_tokenizer,
        field_map=field_map,
        max_seq_len=args.max_seq_len,
    )

    collator = LottieDataCollator(pad_token_id=model.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = build_optimizer(model, args.lottie_lr, args.lora_lr, weight_decay=0.01)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_update_steps = max(1, updates_per_epoch * args.num_epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

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
        for batch in epoch_train_loader:
            current_batch_index += 1
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

                if global_step % args.eval_steps == 0:
                    eval_loss = run_eval(accelerator, model, eval_loader)
                    accelerator.print(f"epoch={epoch} step={global_step} eval_loss={eval_loss:.4f}")
                    if accelerator.is_main_process:
                        append_log(
                            log_path,
                            {
                                "event": "eval_step",
                                "epoch": epoch,
                                "global_step": global_step,
                                "eval_loss": eval_loss,
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

    if accelerator.is_main_process:
        best_path = output_dir / "best"
        final_path = output_dir / "final"
        if not best_path.exists():
            raise FileNotFoundError(f"Best model path missing at {best_path}")
        if final_path.exists():
            shutil.rmtree(final_path, ignore_errors=True)
        shutil.copytree(best_path, final_path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        append_log(
            log_path,
            {
                "event": "run_finished",
                "global_step": global_step,
                "best_eval": best_eval,
                "best_step": best_step,
                "final_model_path": str(output_dir / "final"),
            },
        )


@torch.no_grad()
def run_eval(accelerator: Accelerator, model: torch.nn.Module, eval_loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in eval_loader:
        outputs = model(**batch)
        loss = outputs.loss.detach().float()
        losses.append(accelerator.gather_for_metrics(loss.unsqueeze(0)).mean().item())
    model.train()
    return sum(losses) / max(1, len(losses))


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

