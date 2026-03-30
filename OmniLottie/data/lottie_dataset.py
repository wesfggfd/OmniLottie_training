from __future__ import annotations

import bisect
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TaskViewIndex = Tuple[int, str]

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from .task_sampler import MixedTaskIndexSampler
from lottie.objects.lottie_rule_tokenizer import LottieRuleTokenizer


from .task_constants import TASK_TEXT, TASK_IMAGE, TASK_VIDEO, TASK_MIXED, TASK_ORDER
TASK_ALIASES = {
    "text": TASK_TEXT,
    "text2lottie": TASK_TEXT,
    "text_to_lottie": TASK_TEXT,
    TASK_TEXT: TASK_TEXT,
    "image": TASK_IMAGE,
    "text_image2lottie": TASK_IMAGE,
    "text_image_to_lottie": TASK_IMAGE,
    TASK_IMAGE: TASK_IMAGE,
    "video": TASK_VIDEO,
    "video2lottie": TASK_VIDEO,
    "video_to_lottie": TASK_VIDEO,
    TASK_VIDEO: TASK_VIDEO,
    TASK_MIXED: TASK_MIXED,
}


def _normalize_task_label(value: Any) -> str:
    if value is None:
        return TASK_TEXT
    normalized = str(value).strip().lower().replace("-", "_")
    return TASK_ALIASES.get(normalized, TASK_ALIASES.get(normalized.replace("_", "-"), normalized.replace("_", "-")))


def _first_present(sample: Dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


@dataclass
class LottieFieldMap:
    text_candidates: tuple[str, ...] = (
        "desc_en",
        "motion_caption",
        "detail",
        "keywords_en",
        "short_desc",
        "medium_desc",
        "long_desc",
        "caption",
        "text",
        "prompt",
        "instruction",
    )
    image_candidates: tuple[str, ...] = ("image", "image_path", "keyframe", "keyframe_path")
    video_candidates: tuple[str, ...] = ("video", "video_path", "rendered_video", "rendered_video_path")
    sequence_candidates: tuple[str, ...] = ("sequence_text", "lottie_sequence", "token_sequence")
    json_candidates: tuple[str, ...] = ("lottie_json", "json", "animation_json", "lottie")


class MMLottieAutoregressiveDataset(Dataset):
    def __init__(
        self,
        processor,
        lottie_tokenizer: LottieRuleTokenizer,
        field_map: Optional[LottieFieldMap] = None,
        system_prompt: str = "Generate a valid Lottie animation.",
        max_seq_len: int = 4096,
        rows: Optional[List[Dict[str, Any]]] = None,
        parquet_files: Optional[List[str]] = None,
        row_indices: Optional[List[int]] = None,
        task_mode: str = "mixed",
        task_ratios: Optional[Dict[str, float]] = None,
        max_sample_retries: int = 16,
    ):
        task_mode = _normalize_task_label(task_mode)
        self.processor = processor
        self.lottie_tokenizer = lottie_tokenizer
        self.field_map = field_map or LottieFieldMap()
        self.system_prompt = system_prompt
        self.max_seq_len = max_seq_len
        if task_mode not in {"mixed", "text", "image", "video", TASK_TEXT, TASK_IMAGE, TASK_VIDEO}:
            raise ValueError(f"Unsupported task_mode: {task_mode}")
        self.rows = rows
        self.parquet_files = parquet_files or []
        self.row_indices = row_indices
        self.task_mode = task_mode
        if self.task_mode not in {TASK_TEXT, TASK_IMAGE, TASK_VIDEO, TASK_MIXED}:
            raise ValueError(f"Unsupported task_mode: {task_mode}")
        self.is_mixed = self.task_mode == TASK_MIXED
        self.task_ratios = self._normalize_task_ratios(task_ratios if self.is_mixed else None)
        self.max_sample_retries = max_sample_retries
        self._parquet_handles: Dict[str, pq.ParquetFile] = {}
        self._row_offsets: List[int] = []
        self._row_group_offsets: Dict[str, List[int]] = {}
        self._sample_indices: List[int] = []
        self._task_view_indices: List[TaskViewIndex] = []
        self._num_rows = 0

        if self.rows is not None and self.parquet_files:
            raise ValueError("Use either in-memory rows or parquet_files, not both.")
        if self.rows is None and not self.parquet_files:
            raise ValueError("Either rows or parquet_files must be provided.")
        if self.rows is None:
            running = 0
            for parquet_file in self.parquet_files:
                parquet_handle = pq.ParquetFile(parquet_file)
                self._parquet_handles[parquet_file] = parquet_handle
                self._row_offsets.append(running)
                row_group_offsets: List[int] = []
                group_running = 0
                for group_idx in range(parquet_handle.metadata.num_row_groups):
                    row_group_offsets.append(group_running)
                    group_running += parquet_handle.metadata.row_group(group_idx).num_rows
                self._row_group_offsets[parquet_file] = row_group_offsets
                running += parquet_handle.metadata.num_rows
            self._num_rows = running
        else:
            self._num_rows = len(self.rows)

        if self.row_indices is None:
            self._sample_indices = list(range(self._num_rows))
        else:
            self._sample_indices = [int(idx) for idx in self.row_indices]
            if not self._sample_indices:
                raise ValueError("row_indices must not be empty")
            for sample_idx in self._sample_indices:
                if sample_idx < 0 or sample_idx >= self._num_rows:
                    raise IndexError(f"row_indices contains out-of-range index: {sample_idx}")
        if self.is_mixed:
            self._mixed_sampler = MixedTaskIndexSampler(
                dataset=self,
                base_indices=self._sample_indices,
                task_ratios=self.task_ratios,
            )
            self._sample_indices = self._mixed_sampler.build()
            self._task_view_indices = self._build_mixed_task_view_indices(self._sample_indices)
        else:
            self._mixed_sampler = None
            self._task_view_indices = [(sample_idx, self.task_mode) for sample_idx in self._sample_indices]
        self._num_rows = len(self._task_view_indices)

    @classmethod
    def from_hf_dataset(cls, hf_dataset, **kwargs) -> "MMLottieAutoregressiveDataset":
        return cls(rows=[hf_dataset[i] for i in range(len(hf_dataset))], **kwargs)

    @classmethod
    def from_parquet_files(cls, parquet_files: List[str], **kwargs) -> "MMLottieAutoregressiveDataset":
        return cls(parquet_files=parquet_files, **kwargs)

    def _normalize_task_ratios(self, task_ratios: Optional[Dict[str, float]]) -> Dict[str, float]:
        if task_ratios is None:
            return {TASK_TEXT: 1.0, TASK_IMAGE: 1.0, TASK_VIDEO: 1.0}
        normalized: Dict[str, float] = {TASK_TEXT: 0.0, TASK_IMAGE: 0.0, TASK_VIDEO: 0.0}
        for key, value in task_ratios.items():
            task_key = _normalize_task_label(key)
            if task_key == TASK_MIXED:
                continue
            if task_key not in normalized:
                raise ValueError(f"Unsupported task ratio key: {key}")
            normalized[task_key] = float(value)
        if sum(normalized.values()) <= 0:
            raise ValueError(f"task_ratios must contain at least one positive value: {task_ratios}")
        return normalized

    @classmethod
    def audit_parquet_files(
        cls,
        parquet_files: List[str],
        *,
        processor,
        lottie_tokenizer: LottieRuleTokenizer,
        field_map: Optional[LottieFieldMap] = None,
        system_prompt: str = "Generate a valid Lottie animation.",
        max_seq_len: int = 4096,
        task_mode: str = "mixed",
        task_ratios: Optional[Dict[str, float]] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        dataset = cls(
            processor=processor,
            lottie_tokenizer=lottie_tokenizer,
            field_map=field_map,
            system_prompt=system_prompt,
            max_seq_len=max_seq_len,
            parquet_files=parquet_files,
            task_mode=task_mode,
            task_ratios=task_ratios,
        )
        audits: List[Dict[str, Any]] = []
        sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for idx in range(sample_count):
            sample_idx = dataset._sample_indices[idx]
            try:
                sample = dataset[idx]
                audits.append(
                    {
                        "index": idx,
                        "sample_index": sample_idx,
                        "ok": True,
                        "task_type": sample.get("task_type"),
                        "condition_length": sample.get("condition_length"),
                        "target_length": sample.get("target_length"),
                        "truncated_condition": sample.get("truncated_condition"),
                        "truncated_target": sample.get("truncated_target"),
                    }
                )
            except Exception as exc:
                audits.append(
                    {
                        "index": idx,
                        "sample_index": sample_idx,
                        "ok": False,
                        "error": str(exc),
                    }
                )
        return audits

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_sample_retries):
            try:
                return self._build_item(idx)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_sample_retries:
                    break
                idx = (idx + 1) % self._num_rows
        assert last_error is not None
        raise RuntimeError(f"Failed to build sample after {self.max_sample_retries} attempts") from last_error

    def _build_item(self, idx: int) -> Dict[str, torch.Tensor]:
        sample, sample_task_mode = self._get_sample_with_task_view(idx)
        messages = self._build_messages(sample, sample_task_mode)
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        processor_outputs = self.processor(
            text=[text_input],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        self._validate_processor_outputs(processor_outputs, image_inputs=image_inputs, video_inputs=video_inputs)

        lottie_target = self._encode_target(sample)
        input_ids = processor_outputs["input_ids"][0]
        attention_mask = processor_outputs["attention_mask"][0]
        target_ids = torch.tensor(self.lottie_tokenizer.wrap_target(lottie_target), dtype=torch.long)

        is_multimodal = bool(image_inputs or video_inputs)
        original_condition_length = input_ids.shape[0]
        min_target_reserve = min(1024, max(256, self.max_seq_len // 2))
        if is_multimodal:
            target_budget = max(1, self.max_seq_len - original_condition_length)
        else:
            target_budget = min(target_ids.shape[0], min_target_reserve)

        truncated_target = target_ids.shape[0] > target_budget
        if truncated_target:
            target_ids = torch.cat(
                [
                    target_ids[: max(1, target_budget - 1)],
                    torch.tensor([self.lottie_tokenizer.eos_token_id], dtype=torch.long),
                ],
                dim=0,
            )

        condition_budget = max(1, self.max_seq_len - target_ids.shape[0])
        truncated_condition = input_ids.shape[0] > condition_budget
        if is_multimodal and truncated_condition:
            raise ValueError(
                "Multimodal condition exceeds max_seq_len after target budgeting; "
                f"condition_length={input_ids.shape[0]} target_length={target_ids.shape[0]} max_seq_len={self.max_seq_len}"
            )
        input_ids = input_ids[:condition_budget]
        attention_mask = attention_mask[:condition_budget]
        mm_token_type_ids = None
        if "mm_token_type_ids" in processor_outputs:
            condition_mm_token_type_ids = processor_outputs["mm_token_type_ids"][0][:condition_budget]
            target_mm_token_type_ids = torch.zeros_like(target_ids)
            mm_token_type_ids = torch.cat([condition_mm_token_type_ids, target_mm_token_type_ids], dim=0)

        merged_input_ids = torch.cat([input_ids, target_ids], dim=0)
        merged_attention_mask = torch.cat([attention_mask, torch.ones_like(target_ids)], dim=0)
        labels = torch.full((merged_input_ids.shape[0],), -100, dtype=torch.long)
        labels[input_ids.shape[0] :] = target_ids

        batch = {
            "input_ids": merged_input_ids,
            "attention_mask": merged_attention_mask,
            "labels": labels,
            "task_type": sample_task_mode,
            "condition_length": int(input_ids.shape[0]),
            "target_length": int(target_ids.shape[0]),
            "truncated_condition": truncated_condition,
            "truncated_target": truncated_target,
        }

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in processor_outputs:
                value = processor_outputs[key]
                if value is None:
                    continue
                batch[key] = value
        if mm_token_type_ids is not None:
            batch["mm_token_type_ids"] = mm_token_type_ids

        if not self._task_mode_matches_sample(image_inputs, video_inputs):
            raise ValueError(
                f"Sample task mismatch for task_mode={self.task_mode}: image_inputs={bool(image_inputs)} video_inputs={bool(video_inputs)}"
            )
        self._validate_batch_shapes(batch, is_multimodal=is_multimodal)
        return batch

    def _get_raw_sample(self, sample_idx: int) -> Dict[str, Any]:
        if self.rows is not None:
            return self.rows[sample_idx]

        file_index = bisect.bisect_right(self._row_offsets, sample_idx) - 1
        file_index = max(0, file_index)
        parquet_file = self.parquet_files[file_index]
        row_in_file = sample_idx - self._row_offsets[file_index]
        parquet_handle = self._parquet_handles[parquet_file]
        row_group_offsets = self._row_group_offsets[parquet_file]
        row_group_index = bisect.bisect_right(row_group_offsets, row_in_file) - 1
        row_group_index = max(0, row_group_index)
        row_in_group = row_in_file - row_group_offsets[row_group_index]
        table = parquet_handle.read_row_group(row_group_index)
        return table.slice(row_in_group, 1).to_pylist()[0]

    def _get_sample(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(idx)
        sample_idx, _ = self._task_view_indices[idx]
        return self._get_raw_sample(sample_idx)

    def _get_sample_with_task_view(self, idx: int) -> Tuple[Dict[str, Any], str]:
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(idx)
        sample_idx, task_view = self._task_view_indices[idx]
        return self._get_raw_sample(sample_idx), task_view

    def _available_task_views(self, sample: Dict[str, Any]) -> List[str]:
        sample_task = _normalize_task_label(sample.get("task_type"))
        if sample_task in {TASK_TEXT, TASK_IMAGE, TASK_VIDEO}:
            return [sample_task]

        available: List[str] = []
        text_value = _first_present(sample, self.field_map.text_candidates)
        image_value = self._normalize_media_value(_first_present(sample, self.field_map.image_candidates))
        video_value = self._normalize_media_value(_first_present(sample, self.field_map.video_candidates))

        if text_value not in (None, ""):
            available.append(TASK_TEXT)
        if image_value is not None:
            available.append(TASK_IMAGE)
        if video_value is not None:
            available.append(TASK_VIDEO)
        return available

    def _build_mixed_task_view_indices(self, sample_indices: List[int]) -> List[TaskViewIndex]:
        task_view_indices: List[TaskViewIndex] = []
        for sample_idx in sample_indices:
            sample = self._get_raw_sample(sample_idx)
            task_views = self._available_task_views(sample)
            if not task_views:
                continue
            if len(task_views) == 1:
                task_view_indices.append((sample_idx, task_views[0]))
                continue
            for task_view in (TASK_TEXT, TASK_IMAGE, TASK_VIDEO):
                if task_view in task_views:
                    task_view_indices.append((sample_idx, task_view))
        if not task_view_indices:
            raise RuntimeError("mixed mode produced no task-view samples")
        return task_view_indices

    def _resolve_sample_task_mode(self, sample: Dict[str, Any]) -> str:
        if self.task_mode != TASK_MIXED:
            return self.task_mode
        task_views = self._available_task_views(sample)
        if TASK_TEXT in task_views:
            return TASK_TEXT
        if TASK_IMAGE in task_views:
            return TASK_IMAGE
        if TASK_VIDEO in task_views:
            return TASK_VIDEO
        return TASK_TEXT

    def _build_messages(self, sample: Dict[str, Any], task_mode: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        image_value = self._normalize_media_value(_first_present(sample, self.field_map.image_candidates))
        video_value = self._normalize_media_value(_first_present(sample, self.field_map.video_candidates))
        text_value = _first_present(sample, self.field_map.text_candidates)

        if task_mode == TASK_TEXT:
            if text_value:
                content.append({"type": "text", "text": str(text_value)})
        elif task_mode == TASK_IMAGE:
            if image_value is not None:
                content.append({"type": "image", "image": image_value})
            if text_value:
                content.append({"type": "text", "text": str(text_value)})
        elif task_mode == TASK_VIDEO:
            if video_value is not None:
                content.append({"type": "video", "video": video_value})
        else:
            raise ValueError(f"Unsupported task_mode for build_messages: {task_mode}")

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if not content:
            raise ValueError(
                f"No usable condition found for task_mode={task_mode}; sample keys={sorted(sample.keys())}"
            )
        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _infer_task_type(image_inputs: Any, video_inputs: Any) -> str:
        if video_inputs:
            return TASK_VIDEO
        if image_inputs:
            return TASK_IMAGE
        return TASK_TEXT

    def _task_mode_matches_sample(self, image_inputs: Any, video_inputs: Any) -> bool:
        if self.task_mode == TASK_MIXED:
            return True
        if self.task_mode == TASK_TEXT:
            return not image_inputs and not video_inputs
        if self.task_mode == TASK_IMAGE:
            return bool(image_inputs) and not video_inputs
        if self.task_mode == TASK_VIDEO:
            return bool(video_inputs) and not image_inputs
        return False

    @staticmethod
    def _normalize_media_value(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                media_bytes = value["bytes"]
                media_path = value.get("path") or ""
                suffix = Path(media_path).suffix
                cache_dir = Path(tempfile.gettempdir()) / "omnilottie_media_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_name = hashlib.sha1(media_bytes).hexdigest() + suffix
                cache_path = cache_dir / cache_name
                if not cache_path.exists():
                    cache_path.write_bytes(media_bytes)
                return str(cache_path)
            if value.get("path"):
                return value["path"]
        return value

    def _validate_processor_outputs(self, processor_outputs: Dict[str, Any], *, image_inputs: Any, video_inputs: Any) -> None:
        if image_inputs:
            image_grid_thw = processor_outputs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError("Processor returned image inputs but missing image_grid_thw")
            if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
                raise ValueError(f"Invalid image_grid_thw shape: {tuple(image_grid_thw.shape)}")
            merge_size = self._get_spatial_merge_size()
            image_token_count = int(image_grid_thw[0].prod().item()) // max(1, merge_size * merge_size)
            image_token_count = max(1, image_token_count)
            tokenized_image_count = int(processor_outputs["input_ids"][0].eq(getattr(self.processor.tokenizer, "image_token_id", -1)).sum().item())
            if tokenized_image_count <= 0:
                raise ValueError("Processor returned image inputs but no image tokens were found in input_ids")
            if tokenized_image_count != image_token_count:
                raise ValueError(
                    f"Image features and image tokens do not match, tokens: {tokenized_image_count}, features: {image_token_count}"
                )
        if video_inputs:
            video_grid_thw = processor_outputs.get("video_grid_thw")
            if video_grid_thw is None:
                raise ValueError("Processor returned video inputs but missing video_grid_thw")
            if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
                raise ValueError(f"Invalid video_grid_thw shape: {tuple(video_grid_thw.shape)}")

    def _get_spatial_merge_size(self) -> int:
        visual = getattr(getattr(self.processor, "model", None), "visual", None)
        if visual is not None and hasattr(visual, "spatial_merge_size"):
            return int(visual.spatial_merge_size)
        model = getattr(self.processor, "image_processor", None)
        if model is not None and hasattr(model, "spatial_merge_size"):
            return int(model.spatial_merge_size)
        return 2

    def _validate_batch_shapes(self, batch: Dict[str, Any], *, is_multimodal: bool) -> None:
        input_len = batch["input_ids"].shape[0]
        attention_len = batch["attention_mask"].shape[0]
        labels_len = batch["labels"].shape[0]
        if not (input_len == attention_len == labels_len):
            raise ValueError(
                f"Batch length mismatch: input_ids={input_len}, attention_mask={attention_len}, labels={labels_len}"
            )
        if "mm_token_type_ids" in batch and batch["mm_token_type_ids"].shape[0] != input_len:
            raise ValueError(
                f"mm_token_type_ids length mismatch: {batch['mm_token_type_ids'].shape[0]} vs input_ids={input_len}"
            )
        if is_multimodal:
            for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                if key in batch and batch[key] is None:
                    raise ValueError(f"Multimodal batch contains empty {key}")

    def task_view_histogram(self) -> Dict[str, int]:
        histogram: Dict[str, int] = {TASK_TEXT: 0, TASK_IMAGE: 0, TASK_VIDEO: 0}
        for _, task_view in self._task_view_indices:
            if task_view in histogram:
                histogram[task_view] += 1
        return histogram

    def source_histogram(self) -> Dict[str, int]:
        histogram: Dict[str, int] = {}
        seen_indices = set()
        for sample_idx, _ in self._task_view_indices:
            if sample_idx in seen_indices:
                continue
            seen_indices.add(sample_idx)
            sample = self._get_raw_sample(sample_idx)
            source = str(sample.get("source") or "unknown")
            histogram[source] = histogram.get(source, 0) + 1
        return histogram

    def _encode_target(self, sample: Dict[str, Any]) -> List[int]:
        sequence_value = _first_present(sample, self.field_map.sequence_candidates)
        if sequence_value:
            token_ids = self.lottie_tokenizer.encode_sequence_text(str(sequence_value), max_length=self.max_seq_len)
            return self._validate_token_ids(token_ids)

        json_value = _first_present(sample, self.field_map.json_candidates)
        if json_value is None:
            raise KeyError(f"No Lottie target field found in sample keys: {sorted(sample.keys())}")

        if isinstance(json_value, str):
            try:
                parsed = json.loads(json_value)
            except json.JSONDecodeError:
                parsed = json_value
        else:
            parsed = json_value

        token_ids = self.lottie_tokenizer.encode_lottie_json(parsed, max_length=self.max_seq_len)
        return self._validate_token_ids(token_ids)

    def _validate_token_ids(self, token_ids: List[int]) -> List[int]:
        """Reject samples whose Lottie token IDs exceed the vocabulary boundary."""
        vocab_size = self.lottie_tokenizer.vocab.vocab_size
        max_id = max(token_ids) if token_ids else 0
        min_id = min(token_ids) if token_ids else 0
        if max_id >= vocab_size or min_id < 0:
            raise ValueError(
                f"Out-of-range Lottie token ID detected: min={min_id}, max={max_id}, "
                f"vocab_size={vocab_size}. Sample will be skipped."
            )
        return token_ids

