from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config
from .io_utils import read_jsonl
from .structure_ablation import write_structure_ablation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structural evaluation ablation reports.")
    parser.add_argument("--metrics", default="outputs/main_exp/per_sample_metrics.jsonl")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
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

    index_rows = read_jsonl(Path(args.index))
    layout_root = _path_from_value(args.layout_root or cfg.get("layout_root"))
    present_models = {str(row.get("model")) for row in rows if row.get("model")}
    model_configs = [
        model
        for model in cfg.get("models", [])
        if isinstance(model, dict) and str(model.get("name")) in present_models
    ]
    if requested_models:
        model_configs = [model for model in model_configs if str(model.get("name")) in requested_models]

    result = write_structure_ablation_results(
        Path(args.out),
        rows,
        index_rows=index_rows,
        pred_root=Path(args.pred_root),
        layout_root=layout_root,
        model_configs=model_configs,
    )
    metadata = result["metadata"]
    print(f"wrote structure ablation variants -> {Path(args.out) / 'structure_ablation_variants.csv'}")
    print(f"wrote structure ablation deltas -> {Path(args.out) / 'structure_ablation_deltas.csv'}")
    print(f"rows with ablation metrics: {metadata['n_rows_with_ablation_metrics']}/{metadata['n_rows_input']}")


if __name__ == "__main__":
    main()
