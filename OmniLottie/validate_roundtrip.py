from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow.parquet as pq

from data.lottie_dataset import LottieFieldMap
from lottie.objects.lottie_rule_tokenizer import LottieRuleTokenizer


def collect_parquet_files(dataset_root: Path) -> List[Path]:
    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root}")
    return parquet_files


def iter_rows(parquet_files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        yield from table.to_pylist()


def first_present(sample: Dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Lottie JSON/sequence/token roundtrip.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--dump_sequence_path", type=str, default=None)
    args = parser.parse_args()

    field_map = LottieFieldMap()
    parquet_files = collect_parquet_files(Path(args.data_path))

    chosen_sample = None
    valid_idx = -1
    for sample in iter_rows(parquet_files):
        target = first_present(sample, field_map.sequence_candidates) or first_present(sample, field_map.json_candidates)
        if target is None:
            continue
        valid_idx += 1
        if valid_idx == args.sample_index:
            chosen_sample = sample
            break

    if chosen_sample is None:
        raise IndexError(f"No eligible sample found at valid sample index {args.sample_index}")

    tokenizer = LottieRuleTokenizer(args.model_path)

    sequence_value = first_present(chosen_sample, field_map.sequence_candidates)
    if sequence_value:
        source_mode = "sequence"
        sequence_text = str(sequence_value)
        token_ids = tokenizer.encode_sequence_text(sequence_text, max_length=args.max_length)
    else:
        source_mode = "json"
        json_value = first_present(chosen_sample, field_map.json_candidates)
        parsed = json.loads(json_value) if isinstance(json_value, str) else json_value
        token_ids = tokenizer.encode_lottie_json(parsed, max_length=args.max_length)
        sequence_text = tokenizer.token_ids_to_sequence(token_ids)

    recovered_sequence = tokenizer.token_ids_to_sequence(token_ids)
    recovered_animation = tokenizer.token_ids_to_lottie_json(token_ids)

    if args.dump_sequence_path:
        Path(args.dump_sequence_path).write_text(sequence_text, encoding="utf-8")

    print(f"source_mode={source_mode}")
    print(f"token_count={len(token_ids)}")
    print("sequence_preview_start")
    print("\n".join(sequence_text.splitlines()[:40]))
    print("sequence_preview_end")
    print("recovered_sequence_preview_start")
    print("\n".join(recovered_sequence.splitlines()[:40]))
    print("recovered_sequence_preview_end")
    print(f"recovered_layer_count={len(recovered_animation.get('layers', []))}")
    print(f"recovered_asset_count={len(recovered_animation.get('assets', []))}")


if __name__ == "__main__":
    main()
