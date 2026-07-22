from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from formtsr_exp.io_utils import read_json, read_jsonl, write_json, write_jsonl

from .core import LANGUAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a balanced 400-page evaluation subset from untouched SRFUND locked data."
    )
    parser.add_argument(
        "--locked-index",
        default="outputs/srfund_transfer_exploratory/splits/locked/index.jsonl",
    )
    parser.add_argument(
        "--dev-index",
        default="outputs/srfund_transfer_exploratory/splits/dev/index.jsonl",
    )
    parser.add_argument(
        "--diagnostic-index",
        default="outputs/srfund_formstruct_aligned_dataset/index.jsonl",
    )
    parser.add_argument(
        "--selection",
        default="outputs/srfund_transfer_exploratory/checkpoint_sweep/selection.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/srfund_transfer_locked400/split",
    )
    parser.add_argument("--per-language", type=int, default=50)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_order(row: Mapping[str, Any], seed: int) -> str:
    sample_id = str(row["sample_id"])
    return hashlib.sha256(
        f"{seed}:locked-eval:{sample_id}".encode("utf-8")
    ).hexdigest()


def select_balanced_locked_subset(
    rows: Sequence[Mapping[str, Any]], *, per_language: int, seed: int
) -> list[dict[str, Any]]:
    if per_language < 1:
        raise ValueError("per_language must be positive")
    selected: list[dict[str, Any]] = []
    for language in LANGUAGES:
        candidates = sorted(
            (row for row in rows if str(row.get("language")) == language),
            key=lambda row: _hash_order(row, seed),
        )
        if len(candidates) < per_language:
            raise ValueError(
                f"not enough locked {language} pages: requested={per_language}, "
                f"available={len(candidates)}"
            )
        selected.extend(dict(row) for row in candidates[:per_language])

    sample_ids = [str(row["sample_id"]) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("locked evaluation subset contains duplicate sample IDs")
    return selected


def main() -> None:
    args = parse_args()
    locked_index = Path(args.locked_index)
    dev_index = Path(args.dev_index)
    diagnostic_index = Path(args.diagnostic_index)
    selection_path = Path(args.selection)
    output_dir = Path(args.output_dir)
    output_index = output_dir / "index.jsonl"

    locked_rows = read_jsonl(locked_index)
    selected = select_balanced_locked_subset(
        locked_rows, per_language=args.per_language, seed=args.seed
    )
    selected_ids = {str(row["sample_id"]) for row in selected}
    locked_ids = {str(row["sample_id"]) for row in locked_rows}
    dev_ids = {str(row["sample_id"]) for row in read_jsonl(dev_index)}
    diagnostic_ids = {
        str(row["sample_id"]) for row in read_jsonl(diagnostic_index)
    }
    if not selected_ids <= locked_ids:
        raise ValueError("locked evaluation subset contains pages outside the locked pool")
    if selected_ids & dev_ids:
        raise ValueError("locked evaluation subset overlaps checkpoint-selection dev")
    if selected_ids & diagnostic_ids:
        raise ValueError("locked evaluation subset overlaps the inspected diagnostic slice")

    selection = read_json(selection_path)
    if int(selection.get("selected_step", -1)) != 100:
        raise ValueError(
            f"expected frozen dev selection step 100, got {selection.get('selected_step')}"
        )

    write_jsonl(output_index, selected)
    protocol = {
        "status": "frozen_after_dev_selection_before_locked_inference",
        "purpose": "single final Pre-SFT versus selected FormStruct-SFT evaluation",
        "selection_source": str(selection_path),
        "selection_source_sha256": _sha256(selection_path),
        "selected_checkpoint": selection["selected_checkpoint"],
        "selected_step": selection["selected_step"],
        "locked_parent_index": str(locked_index),
        "locked_parent_index_sha256": _sha256(locked_index),
        "dev_index": str(dev_index),
        "dev_index_sha256": _sha256(dev_index),
        "diagnostic_index": str(diagnostic_index),
        "diagnostic_index_sha256": _sha256(diagnostic_index),
        "seed": args.seed,
        "hash_order": "SHA256(f'{seed}:locked-eval:{sample_id}') within each language",
        "per_language": args.per_language,
        "n_pages": len(selected),
        "by_language": dict(sorted(Counter(str(row["language"]) for row in selected).items())),
        "dev_overlap": len(selected_ids & dev_ids),
        "diagnostic_overlap": len(selected_ids & diagnostic_ids),
        "real_labels_used_for_training": False,
        "locked_labels_used_for_checkpoint_selection": False,
        "invalid_policy": "invalid_or_missing_prediction_scores_zero",
        "index": str(output_index),
        "index_sha256": _sha256(output_index),
    }
    write_json(output_dir / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
