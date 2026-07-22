from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config
from .constraint_slices import write_constraint_slice_results
from .io_utils import read_json, read_jsonl, write_jsonl
from .metrics import compute_cds, evaluate_sample
from .results import DEFAULT_DIFFICULTY_CSV, summarize_models, write_difficulty_results, write_main_results
from .structure_ablation import write_structure_ablation_results


def _load_existing_model_metrics(metrics_dir: Path, active_models: set[str]) -> list[dict]:
    rows: list[dict] = []
    if not metrics_dir.exists():
        return rows
    for path in sorted(metrics_dir.glob("*.jsonl")):
        if path.stem in active_models:
            continue
        rows.extend(read_jsonl(path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing FormTSR-Bench predictions.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    parser.add_argument("--models", default="", help="Optional comma-separated model directory names.")
    parser.add_argument(
        "--skip-extra-reports",
        action="store_true",
        help="Only write per-sample metrics and main results; skip difficulty, constraint, and structure-ablation reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config)) if Path(args.config).exists() else {}
    weights = {str(k): float(v) for k, v in dict(cfg.get("metric_weights", {})).items()} or {
        "TSR-path": 0.20,
        "VAcc": 0.20,
        "R-F1": 0.20,
        "LIG-F1": 0.20,
        "WAcc": 0.20,
    }
    layout_root_value = cfg.get("layout_root")
    layout_root = Path(layout_root_value) if layout_root_value else None
    difficulty_csv_value = cfg.get("difficulty_csv", str(DEFAULT_DIFFICULTY_CSV))
    difficulty_csv = Path(difficulty_csv_value) if difficulty_csv_value else None
    index_rows = read_jsonl(Path(args.index))
    cfg_models = {str(model.get("name")): model for model in cfg.get("models", []) if isinstance(model, dict)}
    pred_root = Path(args.pred_root)
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    model_dirs = sorted(p for p in pred_root.iterdir() if p.is_dir())
    if requested:
        model_dirs = [p for p in model_dirs if p.name in requested]

    rows = []
    by_model_rows: dict[str, list[dict]] = {}
    for model_dir in model_dirs:
        model_cfg = cfg_models.get(model_dir.name, {})
        provider = str(model_cfg.get("provider") or ("traditional_tsr" if "traditional" in model_dir.name.lower() else "existing_predictions"))
        group = str(model_cfg.get("group") or provider)
        model_id = str(model_cfg.get("model") or model_dir.name)
        for sample in index_rows:
            pred_path = model_dir / f"{sample['sample_id']}.json"
            if not pred_path.exists():
                metric_row = evaluate_sample(sample, None, valid_json=False, layout_root=layout_root, group=group)
                metric_row.update({"model": model_dir.name, "model_id": model_id, "provider": provider, "status": "missing_prediction"})
            else:
                try:
                    pred = read_json(pred_path)
                    metric_row = evaluate_sample(sample, pred, valid_json=True, layout_root=layout_root, group=group)
                    metric_row.update({"model": model_dir.name, "model_id": model_id, "provider": provider, "status": "ok"})
                except Exception as exc:
                    metric_row = evaluate_sample(sample, None, valid_json=False, layout_root=layout_root, group=group)
                    metric_row.update({"model": model_dir.name, "model_id": model_id, "provider": provider, "status": "invalid_json", "error": str(exc)})
            metric_row["CDS"] = compute_cds(metric_row, weights, traditional=(provider == "traditional_tsr"))
            rows.append(metric_row)
            by_model_rows.setdefault(model_dir.name, []).append(metric_row)

    out_dir = Path(args.out)
    metrics_dir = out_dir / "per_model_metrics"
    for model_name, model_rows in by_model_rows.items():
        write_jsonl(metrics_dir / f"{model_name}.jsonl", model_rows)
    rows = _load_existing_model_metrics(metrics_dir, set(by_model_rows)) + rows
    write_jsonl(out_dir / "per_sample_metrics.jsonl", rows)
    summary_model_configs = [cfg_models[name] for name in requested if name in cfg_models] if requested else list(cfg_models.values())
    summaries = summarize_models(rows, model_configs=summary_model_configs)
    write_main_results(
        out_dir,
        summaries,
        {
            "index_path": args.index,
            "pred_root": args.pred_root,
            "metric_weights": weights,
            "cds_rule": "Weighted average over numeric TSR-path, VAcc, R-F1, LIG-F1, and WAcc using fixed configured weights. NA metrics are excluded from that sample's weighted denominator. Traditional TSR CDS is NA.",
            "layout_root": str(layout_root) if layout_root else None,
            "difficulty_csv": str(difficulty_csv) if difficulty_csv else None,
        },
    )
    skip_extra_reports = args.skip_extra_reports or str(os.environ.get("FORMTSR_SKIP_EXTRA_REPORTS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if skip_extra_reports:
        print(f"evaluated {len(rows)} prediction/sample pairs -> {out_dir / 'main_results.csv'}")
        print("skipped extra reports")
        return
    write_difficulty_results(
        out_dir,
        rows,
        index_rows=index_rows,
        difficulty_csv=difficulty_csv,
        layout_root=layout_root,
        model_configs=summary_model_configs,
    )
    write_constraint_slice_results(
        out_dir,
        rows,
        index_rows=index_rows,
        layout_root=layout_root,
        model_configs=summary_model_configs,
    )
    write_structure_ablation_results(
        out_dir,
        rows,
        index_rows=index_rows,
        pred_root=Path(args.pred_root),
        layout_root=layout_root,
        model_configs=summary_model_configs,
    )
    print(f"evaluated {len(rows)} prediction/sample pairs -> {out_dir / 'main_results.csv'}")
    print(f"wrote difficulty summary -> {out_dir / 'difficulty_results.csv'}")
    print(f"wrote constraint slices -> {out_dir / 'constraint_slice_results.csv'}")
    print(f"wrote structure ablation -> {out_dir / 'structure_ablation_deltas.csv'}")


if __name__ == "__main__":
    main()
