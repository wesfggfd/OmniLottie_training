from __future__ import annotations

from typing import Dict, List

import torch


class LottieDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].shape[0] for feature in features)
        batch: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_length = max_length - feature["input_ids"].shape[0]
            batch["input_ids"].append(self._pad_1d(feature["input_ids"], pad_length, self.pad_token_id))
            batch["attention_mask"].append(self._pad_1d(feature["attention_mask"], pad_length, 0))
            batch["labels"].append(self._pad_1d(feature["labels"], pad_length, -100))

        merged = {key: torch.stack(value, dim=0) for key, value in batch.items()}

        for optional_key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [feature[optional_key] for feature in features if optional_key in feature]
            if values and len(values) == len(features):
                merged[optional_key] = torch.stack(values, dim=0)

        return merged

    @staticmethod
    def _pad_1d(tensor: torch.Tensor, pad_length: int, pad_value: int) -> torch.Tensor:
        if pad_length <= 0:
            return tensor
        padding = torch.full((pad_length,), pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)

