from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .metrics import NA, normalize_json, unwrap_answer


RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "run_type",
    "comparison_status",
    "full_scope",
    "n_total",
    "n_prediction_files",
    "n_extra_prediction_files",
    "n_valid_json",
    "coverage",
    "n_missing_prediction",
    "n_invalid_json",
    "n_exact_match",
    "Page-EM",
    "Page-EM-valid",
]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    label_path: str
    normalized_gt: Any


@dataclass(frozen=True)
class RunSpec:
    model: str
    model_id: str
    group: str
    n_total: int


def classify_run_type(model: str) -> str:
    lowered = model.lower()
    if "smoke" in lowered:
        return "smoke"
    if "aligned_metadata" in lowered:
        return "aligned"
    return "raw"


def classify_comparison_status(
    *,
    run_type: str,
    has_prediction_dir: bool,
    full_scope: bool,
    n_valid_json: int,
    n_indexed: int,
) -> str:
    if run_type == "smoke":
        return "smoke"
    if run_type == "aligned":
        return "aligned_diagnostic"
    if not has_prediction_dir or n_valid_json == 0:
        return "failed"
    min_comparable_valid = min(5000, n_indexed)
    if full_scope and n_valid_json >= min_comparable_valid:
        return "comparable_raw"
    return "partial_raw"


def page_exact_match(pred: Any, normalized_gt: Any) -> float:
    return 1.0 if normalize_json(unwrap_answer(pred)) == normalized_gt else 0.0


def _as_int(value: Any, *, field: str, model: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} for {model}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative for {model}: {parsed}")
    return parsed


def load_samples(index_path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for row in read_jsonl(index_path):
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in index: {sample_id}")
        seen.add(sample_id)
        label_path = str(row["label_path"])
        gt = read_json(Path(label_path))
        samples.append(Sample(sample_id, label_path, normalize_json(gt)))
    return samples


def load_run_specs(main_results_path: Path, pred_root: Path, n_indexed: int) -> list[RunSpec]:
    if not main_results_path.exists():
        return [
            RunSpec(path.name, path.name, "", n_indexed)
            for path in sorted(pred_root.iterdir())
            if path.is_dir()
        ]

    specs: list[RunSpec] = []
    with main_results_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            model = str(row.get("model") or "").strip()
            if not model:
                continue
            specs.append(
                RunSpec(
                    model=model,
                    model_id=str(row.get("model_id") or model),
                    group=str(row.get("group") or ""),
                    n_total=_as_int(row.get("n_total"), field="n_total", model=model),
                )
            )
    return specs


def evaluate_run(spec: RunSpec, samples: list[Sample], pred_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_dir = pred_root / spec.model
    run_type = classify_run_type(spec.model)
    full_scope = spec.n_total == len(samples)
    if not model_dir.is_dir():
        row = {
            "model": spec.model,
            "model_id": spec.model_id,
            "group": spec.group,
            "run_type": run_type,
            "comparison_status": classify_comparison_status(
                run_type=run_type,
                has_prediction_dir=False,
                full_scope=full_scope,
                n_valid_json=0,
                n_indexed=len(samples),
            ),
            "full_scope": full_scope,
            "n_total": spec.n_total,
            "n_prediction_files": 0,
            "n_extra_prediction_files": 0,
            "n_valid_json": 0,
            "coverage": 0.0 if spec.n_total else NA,
            "n_missing_prediction": spec.n_total,
            "n_invalid_json": 0,
            "n_exact_match": 0,
            "Page-EM": 0.0 if spec.n_total else NA,
            "Page-EM-valid": NA,
        }
        return row, []

    if spec.n_total != len(samples):
        raise ValueError(
            f"run {spec.model!r} has n_total={spec.n_total}, but its prediction directory "
            f"requires the full {len(samples)}-sample index; use a matching main-results scope"
        )

    n_prediction_files = 0
    n_valid_json = 0
    n_invalid_json = 0
    n_exact_match = 0
    exact_rows: list[dict[str, Any]] = []
    sample_ids = {sample.sample_id for sample in samples}

    for sample in samples:
        pred_path = model_dir / f"{sample.sample_id}.json"
        if not pred_path.exists():
            continue
        n_prediction_files += 1
        try:
            pred = read_json(pred_path)
        except Exception:
            n_invalid_json += 1
            continue
        n_valid_json += 1
        if page_exact_match(pred, sample.normalized_gt):
            n_exact_match += 1
            exact_rows.append(
                {
                    "model": spec.model,
                    "model_id": spec.model_id,
                    "sample_id": sample.sample_id,
                    "prediction_path": str(pred_path),
                    "label_path": sample.label_path,
                }
            )

    n_extra_prediction_files = sum(
        1 for path in model_dir.glob("*.json") if path.stem not in sample_ids
    )
    row = {
        "model": spec.model,
        "model_id": spec.model_id,
        "group": spec.group,
        "run_type": run_type,
        "comparison_status": classify_comparison_status(
            run_type=run_type,
            has_prediction_dir=True,
            full_scope=full_scope,
            n_valid_json=n_valid_json,
            n_indexed=len(samples),
        ),
        "full_scope": full_scope,
        "n_total": spec.n_total,
        "n_prediction_files": n_prediction_files,
        "n_extra_prediction_files": n_extra_prediction_files,
        "n_valid_json": n_valid_json,
        "coverage": n_valid_json / spec.n_total if spec.n_total else NA,
        "n_missing_prediction": spec.n_total - n_prediction_files,
        "n_invalid_json": n_invalid_json,
        "n_exact_match": n_exact_match,
        "Page-EM": n_exact_match / spec.n_total if spec.n_total else NA,
        "Page-EM-valid": n_exact_match / n_valid_json if n_valid_json else NA,
    }
    return row, exact_rows


def _format_value(value: Any) -> str:
    if value == NA or value is None:
        return NA
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_results(out_dir: Path, rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    csv_path = out_dir / "page_em_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in RESULT_COLUMNS})

    write_jsonl(out_dir / "page_em_exact_matches.jsonl", exact_rows)
    write_json(out_dir / "page_em_results_metadata.json", metadata)

    lines = [
        "# Page-EM Results",
        "",
        "`Page-EM` is the fraction of all evaluated pages whose normalized prediction `answer` exactly equals the full GT `answer.json`. Missing and invalid predictions score zero.",
        "",
        "| Run | Type | Status | Valid/Total | Coverage | Exact pages | Page-EM | Page-EM (valid only) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["run_type"]),
                    str(row["comparison_status"]),
                    f"{row['n_valid_json']}/{row['n_total']}",
                    _format_value(row["coverage"]),
                    str(row["n_exact_match"]),
                    _format_value(row["Page-EM"]),
                    _format_value(row["Page-EM-valid"]),
                ]
            )
            + " |"
        )
    (out_dir / "page_em_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute whole-page exact match for existing FormTSR predictions.")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--main-results", default="outputs/main_exp/main_results.csv")
    parser.add_argument("--out", default="outputs/main_exp")
    parser.add_argument("--models", default="", help="Optional comma-separated run ids.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    pred_root = Path(args.pred_root)
    main_results_path = Path(args.main_results)
    samples = load_samples(index_path)
    specs = load_run_specs(main_results_path, pred_root, len(samples))

    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    if requested:
        available = {spec.model for spec in specs}
        missing = requested - available
        if missing:
            raise ValueError(f"requested run ids not found: {', '.join(sorted(missing))}")
        specs = [spec for spec in specs if spec.model in requested]

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    by_model: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_run, spec, samples, pred_root): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            by_model[spec.model] = result
            row = result[0]
            print(
                f"[Page-EM] {spec.model}: exact={row['n_exact_match']}/{row['n_total']} "
                f"valid={row['n_valid_json']}"
            )

    rows = [by_model[spec.model][0] for spec in specs]
    exact_rows = [item for spec in specs for item in by_model[spec.model][1]]
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_path": str(index_path),
        "pred_root": str(pred_root),
        "main_results_path": str(main_results_path),
        "n_indexed": len(samples),
        "n_runs": len(rows),
        "n_exact_matches_across_runs": sum(int(row["n_exact_match"]) for row in rows),
        "comparison_status": {
            "comparable_raw": "raw run over the full index with at least min(5000, n_indexed) valid predictions",
            "partial_raw": "raw run that does not meet the comparable coverage rule",
            "aligned_diagnostic": "metadata-aligned post-processing diagnostic; not a raw result",
            "smoke": "smoke-test run; not comparable",
            "failed": "no usable prediction directory or zero valid predictions",
        },
        "metric": {
            "name": "Page-EM",
            "definition": "1 when normalized prediction answer exactly equals the full normalized GT answer.json; otherwise 0",
            "aggregation": "n_exact_match / n_total; missing and invalid predictions are zero",
            "valid_only_diagnostic": "n_exact_match / n_valid_json",
            "prediction_unwrap": "If the prediction has a top-level answer key, compare only that value to GT.",
            "dict_order": "ignored by recursive key sorting",
            "list_order": "preserved",
            "key_normalization": "strip leading and trailing whitespace",
            "string_normalization": "strip and collapse whitespace only; case and Unicode form remain significant",
            "structure_note": "The instance GT is answer.json, so prediction regions/widgets/grids/relations are outside Page-EM.",
        },
    }
    write_results(Path(args.out), rows, exact_rows, metadata)
    print(f"wrote Page-EM results -> {Path(args.out) / 'page_em_results.csv'}")


if __name__ == "__main__":
    main()
