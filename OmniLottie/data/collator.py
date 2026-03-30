from __future__ import annotations

from typing import Any, Dict, List

import torch


class LottieDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(feature["input_ids"].shape[0] for feature in features)
        batch: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_length = max_length - feature["input_ids"].shape[0]
            batch["input_ids"].append(self._pad_1d(feature["input_ids"], pad_length, self.pad_token_id))
            batch["attention_mask"].append(self._pad_1d(feature["attention_mask"], pad_length, 0))
            batch["labels"].append(self._pad_1d(feature["labels"], pad_length, -100))

        merged: Dict[str, Any] = {key: torch.stack(value, dim=0) for key, value in batch.items()}
        merged["task_type"] = [feature.get("task_type", "text") for feature in features]

        for meta_key in ("condition_length", "target_length", "truncated_condition", "truncated_target"):
            if any(meta_key in feature for feature in features):
                merged[meta_key] = [feature.get(meta_key) for feature in features]

        # mm_token_type_ids is seq-len aligned — pad like input_ids
        if all("mm_token_type_ids" in f for f in features):
            max_length_local = merged["input_ids"].shape[1]
            padded = []
            for f in features:
                t = f["mm_token_type_ids"]
                pad_len = max_length_local - t.shape[0]
                padded.append(self._pad_1d(t, pad_len, 0))
            merged["mm_token_type_ids"] = torch.stack(padded, dim=0)

        # pixel_values / grid_thw vary in first dim — cat along dim 0
        for cat_key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [feature[cat_key] for feature in features if cat_key in feature]
            if values and len(values) == len(features):
                merged[cat_key] = torch.cat(values, dim=0)

        return merged

    @staticmethod
    def _pad_1d(tensor: torch.Tensor, pad_length: int, pad_value: int) -> torch.Tensor:
        if pad_length <= 0:
            return tensor
        padding = torch.full((pad_length,), pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)

