from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config
from .io_utils import read_jsonl
from .results import DEFAULT_DIFFICULTY_CSV, write_difficulty_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L1-L4 difficulty-stratified FormTSR-Bench reports.")
    parser.add_argument("--metrics", default="outputs/main_exp/per_sample_metrics.jsonl")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--out", default="outputs/main_exp")
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    parser.add_argument("--difficulty-csv", default="")
    parser.add_argument("--layout-root", default="")
    parser.add_argument("--models", default="", help="Optional comma-separated run ids to include.")
    return parser.parse_args()


def _path_from_value(value: Any) -> Path | None:
    return Path(value) if value else None


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config)) if Path(args.config).exists() else {}

    rows = read_jsonl(Path(args.metrics))
    requested_models = {item.strip() for item in args.models.split(",") if item.strip()}
    if requested_models:
        rows = [row for row in rows if str(row.get("model")) in requested_models]

    index_path = Path(args.index)
    index_rows = read_jsonl(index_path) if index_path.exists() else []
    difficulty_csv = _path_from_value(args.difficulty_csv or cfg.get("difficulty_csv") or DEFAULT_DIFFICULTY_CSV)
    layout_root = _path_from_value(args.layout_root or cfg.get("layout_root"))
    present_models = {str(row.get("model")) for row in rows if row.get("model")}
    model_configs = [
        model
        for model in cfg.get("models", [])
        if isinstance(model, dict) and str(model.get("name")) in present_models
    ]
    if requested_models:
        model_configs = [model for model in model_configs if str(model.get("name")) in requested_models]

    result = write_difficulty_results(
        Path(args.out),
        rows,
        index_rows=index_rows,
        difficulty_csv=difficulty_csv,
        layout_root=layout_root,
        model_configs=model_configs,
    )
    metadata = result["metadata"]
    print(f"wrote difficulty summary -> {Path(args.out) / 'difficulty_results.csv'}")
    print(f"rows with difficulty: {metadata['n_rows_with_difficulty']}/{metadata['n_rows_input']}")
    print(f"template counts by level: {metadata['template_counts_by_level']}")


if __name__ == "__main__":
    main()
