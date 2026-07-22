from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from formtsr_exp.config import load_config
from formtsr_exp.io_utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and verify all schema-aligned SRFUND checkpoint conditions."
    )
    parser.add_argument(
        "--config", default="configs/srfund_formstruct_aligned_benchmark.yaml"
    )
    parser.add_argument("--models", default="", help="Optional comma-separated model names.")
    parser.add_argument(
        "--report-config", default="configs/sft_transfer_figure.yaml"
    )
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def _latest_error_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row["sample_id"])
        for row in read_jsonl(path)
        if isinstance(row.get("sample_id"), str)
    }


def completion_counts(
    output_dir: Path,
    model_name: str,
    sample_ids: set[str],
) -> dict[str, int | float]:
    pred_dir = output_dir / "pred" / model_name
    prediction_ids = {
        path.stem for path in pred_dir.glob("*.json") if path.stem in sample_ids
    }
    error_ids = _latest_error_ids(output_dir / "errors" / f"{model_name}.jsonl") & sample_ids
    attempted_ids = prediction_ids | error_ids
    total = len(sample_ids)
    return {
        "n_total": total,
        "n_valid_json": len(prediction_ids),
        "n_explicit_error": len(error_ids - prediction_ids),
        "n_attempted": len(attempted_ids),
        "n_pending": total - len(attempted_ids),
        "coverage": len(prediction_ids) / total if total else 0.0,
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_status(path: Path, **values: Any) -> None:
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            rows = load_config(path)
            if isinstance(rows, dict):
                payload.update(rows)
        except Exception:
            pass
    payload.update(values)
    write_json(path, payload)


def _selected_model_names(config: dict[str, Any], requested: set[str]) -> list[str]:
    names = [
        str(model["name"])
        for model in config.get("models", [])
        if isinstance(model, dict) and model.get("enabled", True)
    ]
    if requested:
        missing = requested - set(names)
        if missing:
            raise ValueError(f"unknown models: {sorted(missing)}")
        names = [name for name in names if name in requested]
    if not names:
        raise ValueError("no aligned benchmark models selected")
    return names


def run_queue(
    config_path: Path,
    model_names: Sequence[str],
    *,
    report_config: Path,
    skip_report: bool,
) -> None:
    config = load_config(config_path)
    output_dir = Path(str(config["output_dir"]))
    index_path = Path(str(config["index_path"]))
    sample_ids = {str(row["sample_id"]) for row in read_jsonl(index_path)}
    status_path = output_dir / "queue_status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        status_path,
        state="running",
        started_utc=_utc_now(),
        config=str(config_path),
        index=str(index_path),
        models=list(model_names),
    )
    for model_name in model_names:
        before = completion_counts(output_dir, model_name, sample_ids)
        print(f"[{_utc_now()}] {model_name}: before={before}", flush=True)
        _write_status(
            status_path,
            state="running",
            current_model=model_name,
            current_model_before=before,
            updated_utc=_utc_now(),
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "formtsr_exp.run_main",
            "--config",
            str(config_path),
            "--models",
            model_name,
            "--resume",
            "--skip-extra-reports",
        ]
        result = subprocess.run(command, check=False)
        after = completion_counts(output_dir, model_name, sample_ids)
        print(f"[{_utc_now()}] {model_name}: after={after}", flush=True)
        if result.returncode != 0 or int(after["n_attempted"]) != len(sample_ids):
            _write_status(
                status_path,
                state="failed",
                current_model=model_name,
                current_model_after=after,
                returncode=result.returncode,
                finished_utc=_utc_now(),
            )
            raise RuntimeError(
                f"{model_name} incomplete: returncode={result.returncode}, counts={after}"
            )
        _write_status(
            output_dir / "status" / f"{model_name}.json",
            state="complete",
            model=model_name,
            **after,
            finished_utc=_utc_now(),
        )
    if not skip_report:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "srfund_exp.transfer_figure",
                "--config",
                str(report_config),
            ],
            check=False,
        )
        if result.returncode != 0:
            _write_status(
                status_path,
                state="report_failed",
                returncode=result.returncode,
                finished_utc=_utc_now(),
            )
            raise RuntimeError(f"transfer report failed with code {result.returncode}")
    _write_status(
        status_path,
        state="complete",
        current_model=None,
        finished_utc=_utc_now(),
    )
    print(f"[{_utc_now()}] aligned transfer queue complete", flush=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    model_names = _selected_model_names(config, requested)
    run_queue(
        config_path,
        model_names,
        report_config=Path(args.report_config),
        skip_report=args.skip_report,
    )


if __name__ == "__main__":
    main()
