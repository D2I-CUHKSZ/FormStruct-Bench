from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from formtsr_exp.adapters import AdapterResult, make_adapter
from formtsr_exp.config import load_config
from formtsr_exp.io_utils import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from formtsr_exp.json_parser import parse_json_response

from .core import (
    LANGUAGES,
    METRIC_FIELDS,
    SRFUND_PROMPT,
    SRFUND_RESPONSE_SCHEMA,
    Sample,
    aggregate_rows,
    bootstrap_language_ci,
    dataset_statistics,
    evaluate_prediction,
    load_samples,
    validate_prediction,
    write_csv,
    write_index,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SRFUND tuned/base VLM benchmark.")
    parser.add_argument("--config", default="configs/srfund_qwen_transfer_benchmark.yaml")
    parser.add_argument("--models", default="", help="Comma-separated model names.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _selected_models(config: Mapping[str, Any], requested: set[str] | None) -> list[dict[str, Any]]:
    models = [dict(item) for item in config.get("models", []) if isinstance(item, dict)]
    selected = [
        item
        for item in models
        if item.get("enabled", True) and (not requested or str(item.get("name")) in requested)
    ]
    if requested:
        missing = requested - {str(item.get("name")) for item in selected}
        if missing:
            raise ValueError(f"unknown or disabled models: {sorted(missing)}")
    if not selected:
        raise ValueError("no models selected")
    return selected


def _adapter_config(spec: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    name = str(spec["name"])
    served_model = str(spec.get("served_model") or name)
    rpc_port = int(spec.get("data_parallel_rpc_port", 13400))
    data_parallel_size = int(spec.get("data_parallel_size", 2))
    data_parallel_size_local = int(
        spec.get("data_parallel_size_local", data_parallel_size)
    )
    cuda_visible_devices = str(spec.get("cuda_visible_devices", "0,1"))
    return {
        "name": name,
        "provider": "local_vllm_server_vlm",
        "model": str(spec.get("model_id") or name),
        "batch_size": int(spec.get("concurrency", 8)),
        "concurrency": int(spec.get("concurrency", 8)),
        "server_start_timeout_seconds": int(spec.get("server_start_timeout_seconds", 1800)),
        "request_timeout_seconds": int(spec.get("request_timeout_seconds", 1800)),
        "retries": int(spec.get("request_retries", 1)),
        "temperature": 0,
        "max_tokens": int(spec.get("max_tokens", 16384)),
        "response_format": "json_schema",
        "response_format_style": "structured_outputs",
        "response_json_schema_name": "srfund_semantic_v1",
        "response_json_schema": SRFUND_RESPONSE_SCHEMA,
        "include_chat_template_kwargs": True,
        "base_url": "http://127.0.0.1:8000",
        "max_model_len": int(spec.get("max_model_len", 24576)),
        "max_num_seqs": int(spec.get("max_num_seqs", 4)),
        "tensor_parallel_size": 1,
        "data_parallel_size": data_parallel_size,
        "data_parallel_size_local": data_parallel_size_local,
        "data_parallel_backend": "mp",
        "data_parallel_address": "127.0.0.1",
        "data_parallel_rpc_port": rpc_port,
        "gpu_memory_utilization": float(spec.get("gpu_memory_utilization", 0.9)),
        "mm_processor_cache_gb": 0,
        "fail_fast_server_errors": True,
        "log_dir": str(out_dir / "logs"),
        "env": {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "VLLM_MODEL_PATH": str(spec["model_path"]),
            "VLLM_BIN": "./.venv-vllm/bin/vllm",
            "VLLM_SITE_PACKAGES": "./.venv-vllm/lib/python3.12/site-packages",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
            "FORMTSR_SERVED_MODEL": served_model,
            "FORMTSR_RESPONSE_FORMAT_STYLE": "structured_outputs",
            "FORMTSR_INCLUDE_CHAT_TEMPLATE_KWARGS": "true",
            "FORMTSR_DEBUG_PAYLOAD_PATH": str(out_dir / "payloads" / f"{name}.json"),
            "SGLANG_ENABLE_THINKING": "false",
            "VLLM_EXTRA_ARGS": "--disable-log-stats --generation-config vllm",
        },
    }


def _error(
    path: Path,
    sample: Mapping[str, Any],
    *,
    status: str,
    message: str | None,
) -> None:
    append_jsonl(
        path,
        {
            "sample_id": sample["sample_id"],
            "status": status,
            "error": message,
        },
    )


def _write_status(path: Path, **values: Any) -> None:
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = read_json(path)
            if isinstance(previous, dict):
                payload.update(previous)
        except Exception:
            pass
    payload.update(values)
    write_json_atomic(path, payload)


def run_model(
    spec: Mapping[str, Any],
    samples: Sequence[Sample],
    *,
    out_dir: Path,
    resume: bool,
) -> None:
    name = str(spec["name"])
    pred_dir = out_dir / "pred" / name
    raw_dir = out_dir / "raw" / name
    error_path = out_dir / "errors" / f"{name}.jsonl"
    status_path = out_dir / "status" / f"{name}.json"
    pred_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and error_path.exists():
        error_path.unlink()

    samples_to_run: list[Sample] = []
    for sample in samples:
        pred_path = pred_dir / f"{sample.sample_id}.json"
        if resume and pred_path.is_file():
            try:
                prediction = read_json(pred_path)
                valid, _ = validate_prediction(prediction)
                if valid:
                    continue
            except Exception:
                pass
            pred_path.unlink(missing_ok=True)
        samples_to_run.append(sample)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_status(
        status_path,
        model=name,
        state="running",
        started_utc=started,
        n_total=len(samples),
        n_resumed=len(samples) - len(samples_to_run),
        n_pending=len(samples_to_run),
        pid=os.getpid(),
    )
    print(
        f"[{name}] total={len(samples)} resumed={len(samples)-len(samples_to_run)} "
        f"pending={len(samples_to_run)}",
        flush=True,
    )
    if not samples_to_run:
        _write_status(status_path, state="inference_complete", finished_utc=started)
        return

    adapter = make_adapter(_adapter_config(spec, out_dir))
    sample_lookup = {sample.sample_id: sample for sample in samples_to_run}
    completed = 0

    def on_result(adapter_sample: dict[str, Any], result: AdapterResult) -> None:
        nonlocal completed
        sample_id = str(adapter_sample["sample_id"])
        sample = sample_lookup[sample_id]
        raw_path = raw_dir / f"{sample_id}.txt"
        pred_path = pred_dir / f"{sample_id}.json"
        raw_path.write_text(result.raw_response or "", encoding="utf-8")
        pred_path.unlink(missing_ok=True)
        if result.status != "ok":
            _error(error_path, adapter_sample, status=result.status, message=result.error)
        else:
            prediction, parse_info = parse_json_response(result.raw_response)
            valid, schema_error = validate_prediction(prediction)
            if parse_info["valid"] and valid:
                write_json(pred_path, prediction)
            else:
                message = schema_error or str(parse_info.get("error") or "invalid JSON")
                _error(error_path, adapter_sample, status="invalid_json", message=message)
        completed += 1
        if completed == 1 or completed % 10 == 0 or completed == len(samples_to_run):
            n_valid = sum(1 for path in pred_dir.glob("*.json") if path.is_file())
            _write_status(
                status_path,
                n_completed_this_run=completed,
                n_valid_json=n_valid,
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            print(
                f"[{name}] completed={completed}/{len(samples_to_run)} valid_total={n_valid}",
                flush=True,
            )

    try:
        adapter.run_batch(
            [sample.adapter_sample() for sample in samples_to_run],
            SRFUND_PROMPT,
            on_result=on_result,
        )
    except Exception as exc:
        _write_status(
            status_path,
            state="failed",
            error=str(exc),
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise
    _write_status(
        status_path,
        state="inference_complete",
        n_completed_this_run=completed,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def evaluate_model(
    spec: Mapping[str, Any],
    samples: Sequence[Sample],
    *,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = str(spec["name"])
    pred_dir = out_dir / "pred" / name
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prediction: Mapping[str, Any] | None = None
        path = pred_dir / f"{sample.sample_id}.json"
        if path.is_file():
            try:
                candidate = read_json(path)
                valid, _ = validate_prediction(candidate)
                if valid:
                    prediction = candidate
            except Exception:
                pass
        rows.append(evaluate_prediction(sample, prediction, model=name))

    model_id = str(spec.get("model_id") or name)
    family = str(spec["family"])
    tuned = bool(spec["tuned"])
    summary = aggregate_rows(
        rows,
        model=name,
        model_id=model_id,
        family=family,
        tuned=tuned,
    )
    language_rows = [
        aggregate_rows(
            rows,
            model=name,
            model_id=model_id,
            family=family,
            tuned=tuned,
            language=language,
        )
        for language in LANGUAGES
    ]
    write_jsonl(out_dir / "per_sample" / f"{name}.jsonl", rows)
    write_json(out_dir / "summaries" / f"{name}.json", summary)
    write_csv(out_dir / "per_language" / f"{name}.csv", language_rows)
    _write_status(
        out_dir / "status" / f"{name}.json",
        state="complete",
        n_valid_json=summary["n_valid_json"],
        coverage=summary["coverage"],
        evaluated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    print(
        f"[{name}] evaluation complete: valid={summary['n_valid_json']}/{summary['n_total']} "
        f"entity_f1={summary['entity_f1']:.6f}",
        flush=True,
    )
    return rows, summary


def _load_model_rows(out_dir: Path, name: str) -> list[dict[str, Any]]:
    path = out_dir / "per_sample" / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def write_combined_reports(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    iterations: int,
    seed: int,
) -> None:
    summaries = [read_json(out_dir / "summaries" / f"{spec['name']}.json") for spec in model_specs]
    write_csv(out_dir / "summary.csv", summaries)

    comparison_rows: list[dict[str, Any]] = []
    report_lines = [
        "# SRFUND validation transfer benchmark",
        "",
        "All deltas are tuned minus base. The 95% interval resamples the eight language clusters.",
        "",
        "| Family | Metric | Tuned | Base | Delta | 95% CI | W/T/L |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    families = sorted({str(spec["family"]) for spec in model_specs})
    for family in families:
        family_specs = [spec for spec in model_specs if str(spec["family"]) == family]
        tuned_specs = [spec for spec in family_specs if bool(spec["tuned"])]
        base_specs = [spec for spec in family_specs if not bool(spec["tuned"])]
        if len(tuned_specs) != 1 or len(base_specs) != 1:
            continue
        tuned_spec = tuned_specs[0]
        base_spec = base_specs[0]
        tuned_rows = {row["sample_id"]: row for row in _load_model_rows(out_dir, str(tuned_spec["name"]))}
        base_rows = {row["sample_id"]: row for row in _load_model_rows(out_dir, str(base_spec["name"]))}
        if set(tuned_rows) != set(base_rows):
            raise ValueError(f"sample domain mismatch for family {family}")
        for metric in METRIC_FIELDS:
            deltas = [
                {
                    "sample_id": sample_id,
                    "language": tuned_rows[sample_id]["language"],
                    "delta": float(tuned_rows[sample_id][metric]) - float(base_rows[sample_id][metric]),
                }
                for sample_id in sorted(tuned_rows)
            ]
            tuned_mean = math.fsum(float(row[metric]) for row in tuned_rows.values()) / len(tuned_rows)
            base_mean = math.fsum(float(row[metric]) for row in base_rows.values()) / len(base_rows)
            delta = tuned_mean - base_mean
            low, high = bootstrap_language_ci(
                deltas,
                field="delta",
                iterations=iterations,
                seed=seed,
            )
            wins = sum(row["delta"] > 1e-12 for row in deltas)
            losses = sum(row["delta"] < -1e-12 for row in deltas)
            ties = len(deltas) - wins - losses
            comparison_rows.append(
                {
                    "family": family,
                    "tuned_model": tuned_spec["name"],
                    "base_model": base_spec["name"],
                    "metric": metric,
                    "n_paired": len(deltas),
                    "tuned_mean": tuned_mean,
                    "base_mean": base_mean,
                    "delta": delta,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
            report_lines.append(
                f"| {family} | {metric} | {tuned_mean:.6f} | {base_mean:.6f} | "
                f"{delta:+.6f} | [{low:+.6f}, {high:+.6f}] | {wins}/{ties}/{losses} |"
            )
    write_csv(out_dir / "comparison.csv", comparison_rows)
    (out_dir / "comparison.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    dataset_root = Path(str(config.get("dataset_root", "raw/srfund/extracted/dataset")))
    split = str(config.get("split", "validation"))
    out_dir = Path(str(config.get("output_dir", "outputs/srfund_qwen_transfer_benchmark")))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    unavailable_limit = int(config.get("unavailable_per_language", 50))
    samples = load_samples(
        dataset_root,
        split,
        unavailable_limit=unavailable_limit,
        seed=seed,
    )
    stats = dataset_statistics(samples)
    write_index(out_dir / "index.jsonl", samples)
    write_json(out_dir / "dataset_statistics.json", stats)
    write_json(
        out_dir / "protocol.json",
        {
            "dataset_root": str(dataset_root),
            "split": split,
            "seed": seed,
            "unavailable_per_language": unavailable_limit,
            "english_split_note": (
                "English files have no train/validation marker. For validation_balanced, "
                "50 English pages are selected by seeded SHA-256 rank and retain split=unavailable."
            ),
            "bbox_space": "normalized_0_1000",
            "entity_iou_threshold": 0.5,
            "e2e_text_similarity_threshold": 0.8,
        },
    )
    (out_dir / "prompt.txt").write_text(SRFUND_PROMPT + "\n", encoding="utf-8")
    write_json(out_dir / "response_schema.json", SRFUND_RESPONSE_SCHEMA)
    print(f"loaded SRFUND {split}: {len(samples)} pages; stats={stats}", flush=True)
    if args.validate_only:
        return

    requested = {item.strip() for item in args.models.split(",") if item.strip()} or None
    models = _selected_models(config, requested)
    resume = not args.no_resume
    for spec in models:
        model_path = Path(str(spec["model_path"]))
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        run_model(spec, samples, out_dir=out_dir, resume=resume)
        evaluate_model(spec, samples, out_dir=out_dir)

    all_specs = [dict(item) for item in config.get("models", []) if isinstance(item, dict) and item.get("enabled", True)]
    if all((out_dir / "summaries" / f"{spec['name']}.json").is_file() for spec in all_specs):
        write_combined_reports(
            all_specs,
            out_dir=out_dir,
            iterations=int(config.get("bootstrap_iterations", 10000)),
            seed=seed,
        )
        print(f"wrote combined comparison -> {out_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
