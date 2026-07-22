from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from .adapters import AdapterResult, make_adapter
from .build_index import main as build_index_main  # noqa: F401
from .config import enabled_models, load_config
from .constraint_slices import write_constraint_slice_results
from .dataset import scan_dataset, select_samples
from .io_utils import append_jsonl, ensure_dir, read_jsonl, write_json, write_jsonl
from .io_utils import read_json
from .json_parser import parse_json_response
from .metrics import compute_cds, evaluate_sample
from .model_parsers import parse_deepseek_ocr_markdown, parse_paddleocr_vl_pipeline, parse_unlimited_ocr_markdown
from .prompt import build_model_prompt, build_prompt
from .results import DEFAULT_DIFFICULTY_CSV, summarize_models, write_difficulty_results, write_main_results
from .schema import summarize_schema
from .structure_ablation import write_structure_ablation_results


def _load_existing_model_metrics(metrics_dir: Path, active_models: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metrics_dir.exists():
        return rows
    for path in sorted(metrics_dir.glob("*.jsonl")):
        if path.stem in active_models:
            continue
        rows.extend(read_jsonl(path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FormTSR-Bench main experiment.")
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", default="", help="Comma-separated model names to run.")
    parser.add_argument("--templates", default="", help="Comma-separated template names.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", default="", help="Use an existing dataset_index.jsonl instead of scanning.")
    parser.add_argument("--out-dir", default="", help="Override config output_dir. Useful for isolated robustness runs.")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have pred JSON or an error record for the model.")
    parser.add_argument("--rerun-invalid", action="store_true", help="With --resume, rerun samples recorded as invalid/error instead of treating them as complete.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split selected samples into this many modulo shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only samples whose selected-sample index maps to this shard.")
    parser.add_argument(
        "--skip-extra-reports",
        action="store_true",
        help="Write per-sample metrics and main results only; skip difficulty, constraint, and structure ablation reports.",
    )
    return parser.parse_args()


def _write_skip_outputs(out_dir: Path, model_name: str, sample: dict[str, Any], status: str, error: str | None) -> None:
    append_jsonl(
        out_dir / "errors" / f"{model_name}.jsonl",
        {
            "sample_id": sample["sample_id"],
            "template_name": sample["template_name"],
            "instance_id": sample["instance_id"],
            "status": status,
            "error": error,
        },
    )


def _latest_error_by_sample(error_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not error_path.exists():
        return latest
    for row in read_jsonl(error_path):
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            latest[sample_id] = row
    return latest


def _metric_from_existing_prediction(
    sample: dict[str, Any],
    *,
    model_cfg: dict[str, Any],
    out_dir: Path,
    weights: dict[str, float],
    layout_root: Path | None,
) -> dict[str, Any]:
    model_name = str(model_cfg["name"])
    group = str(model_cfg.get("group", model_cfg.get("provider", "vlm")))
    provider = str(model_cfg.get("provider", ""))
    model_id = str(model_cfg.get("model") or model_name)
    pred_path = out_dir / "pred" / model_name / f"{sample['sample_id']}.json"
    pred = read_json(pred_path)
    metric_row = evaluate_sample(sample, pred, valid_json=True, layout_root=layout_root, group=group)
    metric_row.update({"model": model_name, "model_id": model_id, "provider": provider, "status": "ok", "resumed": True})
    metric_row["CDS"] = compute_cds(metric_row, weights, traditional=(provider == "traditional_tsr"))
    return metric_row


def _metric_from_existing_error(
    sample: dict[str, Any],
    *,
    model_cfg: dict[str, Any],
    error_row: dict[str, Any],
    weights: dict[str, float],
    layout_root: Path | None,
) -> dict[str, Any]:
    model_name = str(model_cfg["name"])
    group = str(model_cfg.get("group", model_cfg.get("provider", "vlm")))
    provider = str(model_cfg.get("provider", ""))
    model_id = str(model_cfg.get("model") or model_name)
    status = str(error_row.get("status") or "error")
    metric_row = evaluate_sample(sample, None, valid_json=False, layout_root=layout_root, group=group)
    metric_row.update(
        {
            "model": model_name,
            "model_id": model_id,
            "provider": provider,
            "status": status,
            "error": error_row.get("error"),
            "resumed": True,
        }
    )
    metric_row["CDS"] = compute_cds(metric_row, weights, traditional=(provider == "traditional_tsr"))
    return metric_row


def run_model_on_samples(
    model_cfg: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    out_dir: Path,
    prompt: str,
    weights: dict[str, float],
    layout_root: Path | None,
    resume: bool = False,
    rerun_invalid: bool = False,
) -> list[dict[str, Any]]:
    model_name = str(model_cfg["name"])
    group = str(model_cfg.get("group", model_cfg.get("provider", "vlm")))
    provider = str(model_cfg.get("provider", ""))
    adapter = make_adapter(model_cfg)
    rows: list[dict[str, Any]] = []
    error_path = out_dir / "errors" / f"{model_name}.jsonl"
    batch_error_path = out_dir / "errors" / f"{model_name}.batch_errors.jsonl"
    existing_errors = _latest_error_by_sample(error_path)
    if error_path.exists() and not resume:
        error_path.unlink()
        existing_errors = {}
    if batch_error_path.exists() and not resume:
        batch_error_path.unlink()

    samples_to_run: list[dict[str, Any]] = []
    if resume:
        for sample in samples:
            sample_id = str(sample["sample_id"])
            pred_path = out_dir / "pred" / model_name / f"{sample_id}.json"
            if pred_path.exists():
                try:
                    rows.append(
                        _metric_from_existing_prediction(
                            sample,
                            model_cfg=model_cfg,
                            out_dir=out_dir,
                            weights=weights,
                            layout_root=layout_root,
                        )
                    )
                    continue
                except Exception:
                    pred_path.unlink()
            if not rerun_invalid and sample_id in existing_errors:
                rows.append(
                    _metric_from_existing_error(
                        sample,
                        model_cfg=model_cfg,
                        error_row=existing_errors[sample_id],
                        weights=weights,
                        layout_root=layout_root,
                    )
                )
                continue
            samples_to_run.append(sample)
    else:
        samples_to_run = list(samples)

    def handle_result(sample: dict[str, Any], result: AdapterResult) -> dict[str, Any]:
        sample_id = str(sample["sample_id"])
        raw_path = out_dir / "raw" / model_name / f"{sample_id}.txt"
        pred_path = out_dir / "pred" / model_name / f"{sample_id}.json"
        if pred_path.exists():
            pred_path.unlink()
        ensure_dir(raw_path.parent)
        raw_path.write_text(result.raw_response or "", encoding="utf-8")

        model_id = str(model_cfg.get("model") or model_name)
        if result.status != "ok":
            _write_skip_outputs(out_dir, model_name, sample, result.status, result.error)
            metric_row = evaluate_sample(sample, None, valid_json=False, layout_root=layout_root, group=group)
            metric_row.update({"model": model_name, "model_id": model_id, "provider": provider, "status": result.status, "error": result.error})
            metric_row["CDS"] = compute_cds(metric_row, weights, traditional=(provider == "traditional_tsr"))
            return metric_row

        parser_name = str(model_cfg.get("parser") or "").strip()
        if parser_name == "deepseek_ocr_markdown":
            try:
                parsed = parse_deepseek_ocr_markdown(result.raw_response)
                parse_info = {"valid": True, "error": None, "candidate": ""}
            except Exception as exc:
                parsed = None
                parse_info = {"valid": False, "error": str(exc), "candidate": result.raw_response[:4000]}
        elif parser_name == "paddleocr_vl_pipeline":
            try:
                parsed = parse_paddleocr_vl_pipeline(result.raw_response)
                parse_info = {"valid": True, "error": None, "candidate": ""}
            except Exception as exc:
                parsed = None
                parse_info = {"valid": False, "error": str(exc), "candidate": result.raw_response[:4000]}
        elif parser_name == "unlimited_ocr_markdown":
            try:
                parsed = parse_unlimited_ocr_markdown(result.raw_response)
                parse_info = {"valid": True, "error": None, "candidate": ""}
            except Exception as exc:
                parsed = None
                parse_info = {"valid": False, "error": str(exc), "candidate": result.raw_response[:4000]}
        else:
            parsed, parse_info = parse_json_response(result.raw_response)
        if parse_info["valid"]:
            write_json(pred_path, parsed)
        else:
            append_jsonl(
                out_dir / "errors" / f"{model_name}.jsonl",
                {
                    "sample_id": sample_id,
                    "status": "invalid_json",
                    "error": parse_info["error"],
                    "candidate": parse_info["candidate"],
                },
            )
        metric_row = evaluate_sample(sample, parsed, valid_json=bool(parse_info["valid"]), layout_root=layout_root, group=group)
        metric_row.update({"model": model_name, "model_id": model_id, "provider": provider, "status": "ok" if parse_info["valid"] else "invalid_json"})
        metric_row["CDS"] = compute_cds(metric_row, weights, traditional=(provider == "traditional_tsr"))
        return metric_row

    def run_one(sample: dict[str, Any]) -> dict[str, Any]:
        return handle_result(sample, adapter.run(sample, prompt))

    batch_size = max(1, int(model_cfg.get("batch_size", 1)))
    if not adapter.supports_batch():
        for sample in samples_to_run:
            rows.append(run_one(sample))
        return rows

    seen_rows: dict[str, dict[str, Any]] = {}

    def on_batch_result(sample: dict[str, Any], result: AdapterResult) -> None:
        row = handle_result(sample, result)
        seen_rows[str(sample["sample_id"])] = row
        rows.append(row)

    if not samples_to_run:
        return rows

    try:
        results = adapter.run_batch(samples_to_run, prompt, on_result=on_batch_result)
    except Exception as exc:
        append_jsonl(
            batch_error_path,
            {
                "model": model_name,
                "status": "batch_error",
                "error": str(exc),
                "n_completed_before_error": len(seen_rows),
                "n_requested": len(samples_to_run),
            },
        )
        if bool(model_cfg.get("batch_error_completes_samples", False)):
            error = f"batch_error: {exc}"
            for sample in samples_to_run:
                if str(sample["sample_id"]) in seen_rows:
                    continue
                rows.append(handle_result(sample, AdapterResult("error", error=error)))
        return rows
    if len(results) != len(samples_to_run):
        missing_count = max(0, len(samples_to_run) - len(results))
        results = [
            *results[: len(samples_to_run)],
            *[AdapterResult("error", error="batch adapter returned wrong result count") for _ in range(missing_count)],
        ]
    for sample, result in zip(samples_to_run, results):
        if str(sample["sample_id"]) in seen_rows:
            continue
        rows.append(handle_result(sample, result))
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    out_dir = Path(args.out_dir or cfg.get("output_dir", "outputs/main_exp"))
    data_root = Path(cfg.get("data_root", "./FormTSR/datasets"))
    index_path = Path(args.index or cfg.get("index_path", out_dir / "dataset_index.jsonl"))
    templates = {item.strip() for item in args.templates.split(",") if item.strip()} or None
    requested_models = {item.strip() for item in args.models.split(",") if item.strip()} or None
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

    if index_path.exists():
        all_samples = read_jsonl(index_path)
    else:
        all_samples = scan_dataset(data_root)
        write_jsonl(index_path, all_samples)
    samples = select_samples(all_samples, limit=args.limit, templates=templates, shuffle=args.shuffle, seed=args.seed)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        samples = [sample for idx, sample in enumerate(samples) if idx % args.num_shards == args.shard_index]
    summarize_schema(samples, out_dir / "schema_summary.json", max_files=int(cfg.get("schema_max_files", 25)))
    prompt = build_prompt(bool(cfg.get("compact_structure", False)))
    ensure_dir(out_dir).joinpath("prompt.txt").write_text(prompt, encoding="utf-8")

    models = enabled_models(cfg, requested_models)
    all_rows: list[dict[str, Any]] = []
    metrics_dir = out_dir / "per_model_metrics"
    active_model_names = {str(model["name"]) for model in models if model.get("name")}
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for model_cfg in models:
        model_prompt = build_model_prompt(model_cfg, prompt)
        model_rows = run_model_on_samples(
            model_cfg,
            samples,
            out_dir=out_dir,
            prompt=model_prompt,
            weights=weights,
            layout_root=layout_root,
            resume=args.resume,
            rerun_invalid=args.rerun_invalid,
        )
        write_jsonl(metrics_dir / f"{model_cfg['name']}.jsonl", model_rows)
        all_rows.extend(model_rows)

    all_rows = _load_existing_model_metrics(metrics_dir, active_model_names) + all_rows
    write_jsonl(out_dir / "per_sample_metrics.jsonl", all_rows)
    summaries = summarize_models(all_rows, model_configs=models)
    metadata: dict[str, Any] = {
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(data_root),
        "index_path": str(index_path),
        "n_indexed": len(all_samples),
        "n_selected": len(samples),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "models": models,
        "resume": args.resume,
        "rerun_invalid": args.rerun_invalid,
        "metric_weights": weights,
        "cds_rule": "Weighted average over numeric TSR-path, VAcc, R-F1, LIG-F1, and WAcc using fixed configured weights. NA metrics are excluded from that sample's weighted denominator. Traditional TSR CDS is NA.",
        "layout_root": str(layout_root) if layout_root else None,
        "difficulty_csv": str(difficulty_csv) if difficulty_csv else None,
    }
    write_main_results(out_dir, summaries, metadata)
    skip_extra_reports = args.skip_extra_reports or str(os.environ.get("FORMTSR_SKIP_EXTRA_REPORTS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if skip_extra_reports:
        print(f"wrote per-sample metrics -> {out_dir / 'per_sample_metrics.jsonl'}")
        print(f"wrote summary -> {out_dir / 'main_results.csv'}")
        print("skipped extra reports")
        return
    write_difficulty_results(
        out_dir,
        all_rows,
        index_rows=all_samples,
        difficulty_csv=difficulty_csv,
        layout_root=layout_root,
        model_configs=models,
    )
    write_constraint_slice_results(
        out_dir,
        all_rows,
        index_rows=all_samples,
        layout_root=layout_root,
        model_configs=models,
    )
    write_structure_ablation_results(
        out_dir,
        all_rows,
        index_rows=all_samples,
        pred_root=out_dir / "pred",
        layout_root=layout_root,
        model_configs=models,
    )
    print(f"wrote per-sample metrics -> {out_dir / 'per_sample_metrics.jsonl'}")
    print(f"wrote summary -> {out_dir / 'main_results.csv'}")
    print(f"wrote difficulty summary -> {out_dir / 'difficulty_results.csv'}")
    print(f"wrote constraint slices -> {out_dir / 'constraint_slice_results.csv'}")
    print(f"wrote structure ablation -> {out_dir / 'structure_ablation_deltas.csv'}")


if __name__ == "__main__":
    main()
