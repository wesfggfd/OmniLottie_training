from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from lottie.objects.lottie_rule_tokenizer import LottieRuleTokenizer


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
        rows: List[Dict[str, Any]],
        processor,
        lottie_tokenizer: LottieRuleTokenizer,
        field_map: Optional[LottieFieldMap] = None,
        system_prompt: str = "Generate a valid Lottie animation.",
        max_seq_len: int = 4096,
    ):
        self.rows = rows
        self.processor = processor
        self.lottie_tokenizer = lottie_tokenizer
        self.field_map = field_map or LottieFieldMap()
        self.system_prompt = system_prompt
        self.max_seq_len = max_seq_len

    @classmethod
    def from_hf_dataset(cls, hf_dataset, **kwargs) -> "MMLottieAutoregressiveDataset":
        return cls(rows=[hf_dataset[i] for i in range(len(hf_dataset))], **kwargs)

    @classmethod
    def from_parquet_files(cls, parquet_files: List[str], **kwargs) -> "MMLottieAutoregressiveDataset":
        rows: List[Dict[str, Any]] = []
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            rows.extend(table.to_pylist())
        return cls(rows=rows, **kwargs)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.rows[idx]
        messages = self._build_messages(sample)
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

        lottie_target = self._encode_target(sample)
        input_ids = processor_outputs["input_ids"][0]
        attention_mask = processor_outputs["attention_mask"][0]

        target_ids = torch.tensor(self.lottie_tokenizer.wrap_target(lottie_target), dtype=torch.long)

        labels = torch.full((input_ids.shape[0] + target_ids.shape[0],), -100, dtype=torch.long)
        labels[input_ids.shape[0] :] = target_ids

        merged_input_ids = torch.cat([input_ids, target_ids], dim=0)[: self.max_seq_len]
        merged_attention_mask = torch.ones_like(merged_input_ids)
        labels = labels[: self.max_seq_len]

        batch = {
            "input_ids": merged_input_ids,
            "attention_mask": merged_attention_mask,
            "labels": labels,
        }

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in processor_outputs:
                batch[key] = processor_outputs[key][0]

        return batch

    def _build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        image_value = self._normalize_media_value(_first_present(sample, self.field_map.image_candidates))
        video_value = self._normalize_media_value(_first_present(sample, self.field_map.video_candidates))
        text_value = _first_present(sample, self.field_map.text_candidates)

        if image_value:
            content.append({"type": "image", "image": image_value})
        if video_value:
            content.append({"type": "video", "video": video_value})
        if text_value:
            content.append({"type": "text", "text": str(text_value)})

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if not content:
            content.append({"type": "text", "text": "Generate a valid Lottie animation."})
        messages.append({"role": "user", "content": content})
        return messages

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

    def _encode_target(self, sample: Dict[str, Any]) -> List[int]:
        sequence_value = _first_present(sample, self.field_map.sequence_candidates)
        if sequence_value:
            return self.lottie_tokenizer.encode_sequence_text(str(sequence_value), max_length=self.max_seq_len)

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

        return self.lottie_tokenizer.encode_lottie_json(parsed, max_length=self.max_seq_len)

