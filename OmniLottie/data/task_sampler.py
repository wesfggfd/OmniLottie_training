from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

from torch.utils.data import Sampler

from .task_constants import TASK_IMAGE, TASK_TEXT, TASK_VIDEO

if TYPE_CHECKING:
    from .lottie_dataset import MMLottieAutoregressiveDataset


@dataclass
class MixedTaskIndexSampler(Sampler[int]):
    dataset: "MMLottieAutoregressiveDataset"
    base_indices: List[int]
    task_ratios: Dict[str, float]
    seed: int = 42
    _cached_indices: Optional[List[int]] = None

    def __iter__(self) -> Iterator[int]:
        return iter(self.build())

    def __len__(self) -> int:
        return len(self.build())

    def build(self) -> List[int]:
        if self._cached_indices is not None:
            return list(self._cached_indices)

        grouped: Dict[str, List[int]] = {TASK_TEXT: [], TASK_IMAGE: [], TASK_VIDEO: []}
        for sample_idx in self.base_indices:
            sample = self.dataset._get_raw_sample(sample_idx)
            sample_task = self.dataset._resolve_sample_task_mode(sample)
            if sample_task not in grouped:
                continue
            grouped[sample_task].append(sample_idx)
        if not any(grouped.values()):
            raise RuntimeError('mixed mode could not find any task-specific sample indices')

        ordered_tasks = [TASK_TEXT, TASK_IMAGE, TASK_VIDEO]
        total_weight = sum(self.task_ratios.get(task, 0.0) for task in ordered_tasks)
        if total_weight <= 0:
            raise ValueError('task_ratios must be positive')

        target_total = sum(len(grouped[task]) for task in ordered_tasks)
        mixed_indices: List[int] = []
        for task in ordered_tasks:
            pool = grouped[task]
            if not pool:
                continue
            weight = self.task_ratios.get(task, 0.0)
            if weight <= 0:
                continue
            needed = max(1, int(round(target_total * weight / total_weight)))
            repeats = (needed + len(pool) - 1) // len(pool)
            mixed_indices.extend((pool * repeats)[:needed])

        if not mixed_indices:
            raise RuntimeError('mixed mode produced no sample indices')

        rng = random.Random(self.seed)
        rng.shuffle(mixed_indices)
        self._cached_indices = mixed_indices
        return list(mixed_indices)
