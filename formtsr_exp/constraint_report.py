from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config
from .constraint_slices import write_constraint_slice_results
from .io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build constraint-sliced FormTSR-Bench Delta(c) reports.")
    parser.add_argument("--metrics", default="outputs/main_exp/per_sample_metrics.jsonl")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--out", default="outputs/main_exp")
    parser.add_argument("--config", default="configs/main_experiment.yaml")
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
    layout_root = _path_from_value(args.layout_root or cfg.get("layout_root"))
    present_models = {str(row.get("model")) for row in rows if row.get("model")}
    model_configs = [
        model
        for model in cfg.get("models", [])
        if isinstance(model, dict) and str(model.get("name")) in present_models
    ]
    if requested_models:
        model_configs = [model for model in model_configs if str(model.get("name")) in requested_models]

    result = write_constraint_slice_results(
        Path(args.out),
        rows,
        index_rows=index_rows,
        layout_root=layout_root,
        model_configs=model_configs,
    )
    metadata = result["metadata"]
    print(f"wrote constraint slices -> {Path(args.out) / 'constraint_slice_results.csv'}")
    print(f"rows with constraint metadata: {metadata['n_rows_with_constraint_metadata']}/{metadata['n_rows_input']}")
    print(f"template counts by constraint: {metadata['template_counts_by_constraint']}")
    print(f"ignored constraints: {metadata['ignored_constraints']}")


if __name__ == "__main__":
    main()
