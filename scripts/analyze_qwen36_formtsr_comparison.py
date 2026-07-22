#!/usr/bin/env python3
"""Compare tuned and base FormTSR runs with template-cluster uncertainty.

The input reports are treated as immutable.  The dataset index is the left-hand
table and fixes both sample order and template membership.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formtsr_exp.document_similarity_report import (
    build_schema_tree,
    extract_normalized_values,
    load_document_samples,
    schema_nted,
    value_ned,
)
from formtsr_exp.io_utils import read_json
from formtsr_exp.metrics import unwrap_answer
from formtsr_exp.page_em_report import load_samples as load_page_em_samples
from formtsr_exp.page_em_report import page_exact_match


NA_STRINGS = {"", "na", "n/a", "nan", "none", "null", "tbd"}


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    label: str
    source: str
    field: str
    required: bool = True


METRICS = (
    MetricSpec("valid_json", "Valid JSON", "legacy", "valid_json"),
    MetricSpec("Page-EM", "Page-EM", "page_em", "Page-EM"),
    MetricSpec(
        "Schema-nTED",
        "Schema-nTED",
        "document_similarity",
        "Schema-nTED",
    ),
    MetricSpec(
        "Value-nED",
        "Value-nED",
        "document_similarity",
        "Value-nED",
    ),
    MetricSpec("TSR-path", "TSR-path", "legacy", "TSR-path"),
    MetricSpec("VAcc", "VAcc", "legacy", "VAcc"),
    MetricSpec("legacy_R-F1", "Legacy R-F1", "legacy", "R-F1"),
    MetricSpec(
        "legacy_R-F1@0.75",
        "Legacy R-F1@0.75",
        "legacy",
        "R-F1@0.75",
        required=False,
    ),
    MetricSpec("legacy_LIG-F1", "Legacy LIG-F1", "legacy", "LIG-F1"),
    MetricSpec("WAcc", "WAcc", "legacy", "WAcc"),
    MetricSpec("CDS", "Legacy CDS", "legacy", "CDS"),
    MetricSpec("corrected_R-F1", "Corrected R-F1", "corrected", "R-F1"),
    MetricSpec(
        "corrected_R-F1@0.75",
        "Corrected R-F1@0.75",
        "corrected",
        "R-F1@0.75",
    ),
    MetricSpec(
        "corrected_LIG-F1",
        "Corrected LIG-F1",
        "corrected",
        "LIG-F1",
        required=False,
    ),
    MetricSpec(
        "corrected_LIG-F1@0.75",
        "Corrected LIG-F1@0.75",
        "corrected",
        "LIG-F1@0.75",
        required=False,
    ),
    MetricSpec(
        "LG-GriTS-Top",
        "LG-GriTS-Top",
        "hierarchical",
        "LG-GriTS-Top",
    ),
    MetricSpec("WG-F1", "WG-F1", "hierarchical", "WG-F1"),
    MetricSpec("Rel-F1", "Rel-F1", "hierarchical", "Rel-F1"),
    MetricSpec(
        "Rel-F1-matched-endpoints",
        "Rel-F1 matched endpoints",
        "hierarchical",
        "Rel-F1-matched-endpoints",
        required=False,
    ),
)

SOURCE_PATHS = {
    "legacy": Path("per_model_metrics"),
    "corrected": Path("corrected_structure_per_sample"),
    "hierarchical": Path("hierarchical_structure_per_sample"),
}

AGGREGATE_REPORTS = (
    ("corrected_structure_metrics", Path("corrected_structure_metrics.csv")),
    ("hierarchical_structure_metrics", Path("hierarchical_structure_metrics.csv")),
)
AGGREGATE_REPORT_SKIP_FIELDS = {
    "model",
    "model_id",
    "group",
    "run_type",
    "comparison_status",
    "bbox_source_space",
    "sample_scope",
}
AGGREGATE_COMPARISON_COLUMNS = (
    "report",
    "field",
    "tuned",
    "base",
    "delta",
    "value_kind",
)
RELATION_TYPE_FIELDS = ("TP", "pred", "GT", "precision", "recall", "F1")
RELATION_TYPE_COMPARISON_COLUMNS = (
    "relation_type",
    "field",
    "tuned",
    "base",
    "delta",
    "value_kind",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tuned and base FormTSR reports page by page, with a "
            "template-cluster bootstrap for overall metric differences."
        )
    )
    parser.add_argument("--index", required=True, help="Frozen test index JSONL.")
    parser.add_argument("--tuned-dir", required=True, help="Tuned run output directory.")
    parser.add_argument("--tuned-model", required=True, help="Tuned run/model id.")
    parser.add_argument("--base-dir", required=True, help="Base run output directory.")
    parser.add_argument("--base-model", required=True, help="Base run/model id.")
    parser.add_argument("--out-dir", required=True, help="Directory for comparison artifacts.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10_000,
        help="Number of template-cluster bootstrap replicates (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-12,
        help="Absolute page-level delta treated as a tie.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _rows_by_sample(
    rows: Iterable[dict[str, Any]],
    *,
    path: Path,
    expected_model: str | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"missing sample_id in {path}, row {row_number}")
        if sample_id in output:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
        if expected_model is not None and str(row.get("model") or "") != expected_model:
            raise ValueError(
                f"model mismatch in {path}, sample {sample_id!r}: "
                f"expected {expected_model!r}, got {row.get('model')!r}"
            )
        output[sample_id] = row
    return output


def _load_index(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    by_id = _rows_by_sample(rows, path=path)
    for sample_id, row in by_id.items():
        template_name = row.get("template_name")
        if not isinstance(template_name, str) or not template_name:
            raise ValueError(f"missing template_name for {sample_id!r} in {path}")
        label_path = row.get("label_path")
        if not isinstance(label_path, str) or not label_path:
            raise ValueError(f"missing label_path for {sample_id!r} in {path}")
    return rows, by_id


def _source_path(run_dir: Path, model: str, source: str) -> Path:
    return run_dir / SOURCE_PATHS[source] / f"{model}.jsonl"


def _load_run_sources(
    run_dir: Path,
    model: str,
    index_path: Path,
    index_rows: Sequence[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    sources: dict[str, dict[str, dict[str, Any]]] = {}
    expected_ids = set(index_by_id)
    for source in SOURCE_PATHS:
        path = _source_path(run_dir, model, source)
        rows = _rows_by_sample(_read_jsonl(path), path=path, expected_model=model)
        actual_ids = set(rows)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"sample domain mismatch in {path}: expected={len(expected_ids)}, "
                f"actual={len(actual_ids)}, missing={missing[:10]}, extra={extra[:10]}"
            )
        for sample_id, row in rows.items():
            reported_template = row.get("template_name")
            expected_template = index_by_id[sample_id]["template_name"]
            if reported_template is not None and str(reported_template) != expected_template:
                raise ValueError(
                    f"template mismatch for {sample_id!r} in {path}: "
                    f"expected {expected_template!r}, got {reported_template!r}"
                )
        sources[source] = rows
    sources.update(
        _load_prediction_metric_sources(
            run_dir=run_dir,
            model=model,
            index_path=index_path,
            index_rows=index_rows,
            index_by_id=index_by_id,
        )
    )
    return sources


def _model_csv_row(path: Path, model: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"required aggregate CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get("model") or "") == model]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one row for model {model!r} in {path}, found {len(rows)}"
        )
    return rows[0]


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.casefold() in NA_STRINGS:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_integer_literal(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped[:1] in {"+", "-"}:
        stripped = stripped[1:]
    return bool(stripped) and stripped.isdigit()


def _paired_numeric_values(
    tuned_raw: Any,
    base_raw: Any,
) -> tuple[int | float, int | float, int | float, str] | None:
    tuned_value = _finite_float(tuned_raw)
    base_value = _finite_float(base_raw)
    if tuned_value is None or base_value is None:
        return None
    if _is_integer_literal(tuned_raw) and _is_integer_literal(base_raw):
        tuned_count = int(str(tuned_raw).strip())
        base_count = int(str(base_raw).strip())
        return tuned_count, base_count, tuned_count - base_count, "count"
    return tuned_value, base_value, tuned_value - base_value, "metric"


def _load_aggregate_report_comparison(
    *,
    tuned_dir: Path,
    tuned_model: str,
    base_dir: Path,
    base_model: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for report, relative_path in AGGREGATE_REPORTS:
        tuned_row = _model_csv_row(tuned_dir / relative_path, tuned_model)
        base_row = _model_csv_row(base_dir / relative_path, base_model)
        for field in tuned_row:
            if field in AGGREGATE_REPORT_SKIP_FIELDS or field not in base_row:
                continue
            values = _paired_numeric_values(tuned_row.get(field), base_row.get(field))
            if values is None:
                continue
            tuned_value, base_value, delta, value_kind = values
            output.append(
                {
                    "report": report,
                    "field": field,
                    "tuned": tuned_value,
                    "base": base_value,
                    "delta": delta,
                    "value_kind": value_kind,
                }
            )
    return output


def _relation_rows_by_type(path: Path, model: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required relation-type CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("model") or "") == model
        ]
    if not rows:
        raise ValueError(f"no relation-type rows for model {model!r} in {path}")

    output: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        relation_type = str(row.get("relation_type") or "")
        if not relation_type:
            raise ValueError(
                f"missing relation_type for model {model!r} in {path}, row {row_number}"
            )
        if relation_type in output:
            raise ValueError(
                f"duplicate relation_type {relation_type!r} for model {model!r} in {path}"
            )
        output[relation_type] = row
    return output


def _required_relation_value(
    value: Any,
    *,
    context: str,
    count: bool,
) -> int | float:
    parsed = _finite_float(value)
    if parsed is None:
        raise ValueError(f"missing or non-finite relation metric at {context}: {value!r}")
    if count:
        if parsed < 0 or not parsed.is_integer():
            raise ValueError(f"invalid relation count at {context}: {value!r}")
        return int(parsed)
    return parsed


def _load_relation_type_comparison(
    *,
    tuned_dir: Path,
    tuned_model: str,
    base_dir: Path,
    base_model: str,
) -> list[dict[str, Any]]:
    relative_path = Path("hierarchical_relation_type_metrics.csv")
    tuned_rows = _relation_rows_by_type(tuned_dir / relative_path, tuned_model)
    base_rows = _relation_rows_by_type(base_dir / relative_path, base_model)
    tuned_types = set(tuned_rows)
    base_types = set(base_rows)
    if tuned_types != base_types:
        raise ValueError(
            "relation_type domain mismatch: "
            f"missing_from_tuned={sorted(base_types - tuned_types)}, "
            f"missing_from_base={sorted(tuned_types - base_types)}"
        )

    output: list[dict[str, Any]] = []
    for relation_type in sorted(tuned_types):
        tuned_row = tuned_rows[relation_type]
        base_row = base_rows[relation_type]
        tuned_gt = _required_relation_value(
            tuned_row.get("GT"),
            context=f"tuned/{relation_type}/GT",
            count=True,
        )
        base_gt = _required_relation_value(
            base_row.get("GT"),
            context=f"base/{relation_type}/GT",
            count=True,
        )
        if tuned_gt != base_gt:
            raise ValueError(
                f"relation GT mismatch for {relation_type!r}: "
                f"tuned={tuned_gt}, base={base_gt}"
            )

        for field in RELATION_TYPE_FIELDS:
            is_count = field in {"TP", "pred", "GT"}
            tuned_value = _required_relation_value(
                tuned_row.get(field),
                context=f"tuned/{relation_type}/{field}",
                count=is_count,
            )
            base_value = _required_relation_value(
                base_row.get(field),
                context=f"base/{relation_type}/{field}",
                count=is_count,
            )
            output.append(
                {
                    "relation_type": relation_type,
                    "field": field,
                    "tuned": tuned_value,
                    "base": base_value,
                    "delta": tuned_value - base_value,
                    "value_kind": "count" if is_count else "metric",
                }
            )
    return output


def _validate_aggregate_count(
    row: dict[str, str],
    *,
    field: str,
    expected: int,
    context: str,
) -> None:
    raw = row.get(field)
    try:
        actual = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid aggregate count at {context}/{field}: {raw!r}") from exc
    if actual != expected:
        raise ValueError(
            f"aggregate count mismatch at {context}/{field}: expected {expected}, got {actual}"
        )


def _validate_aggregate_metric(
    row: dict[str, str],
    *,
    field: str,
    expected: float,
    context: str,
) -> None:
    actual = _as_optional_float(row.get(field), context=f"{context}/{field}")
    if actual is None:
        raise ValueError(f"missing aggregate metric at {context}/{field}")
    # Aggregate report CSVs are formatted to six decimal places.
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=5.1e-7):
        raise ValueError(
            f"aggregate metric mismatch at {context}/{field}: "
            f"recomputed {expected:.12g}, reported {actual:.12g}"
        )


def _load_prediction_metric_sources(
    *,
    run_dir: Path,
    model: str,
    index_path: Path,
    index_rows: Sequence[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    pred_dir = run_dir / "pred" / model
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"required prediction directory does not exist: {pred_dir}")

    expected_ids = [str(row["sample_id"]) for row in index_rows]
    page_samples = load_page_em_samples(index_path)
    document_samples = load_document_samples(index_path)
    page_ids = [sample.sample_id for sample in page_samples]
    document_ids = [sample.sample_id for sample in document_samples]
    if page_ids != expected_ids:
        raise ValueError("Page-EM sample order/domain does not match the frozen index")
    if document_ids != expected_ids:
        raise ValueError("document-similarity sample order/domain does not match the frozen index")

    page_rows: dict[str, dict[str, Any]] = {}
    document_rows: dict[str, dict[str, Any]] = {}
    n_prediction_files = 0
    n_valid_json = 0
    n_invalid_json = 0
    for page_sample, document_sample in zip(page_samples, document_samples, strict=True):
        sample_id = page_sample.sample_id
        if document_sample.sample_id != sample_id:
            raise ValueError(
                f"metric sample mismatch: Page-EM={sample_id!r}, "
                f"document similarity={document_sample.sample_id!r}"
            )
        common = {
            "model": model,
            "sample_id": sample_id,
            "template_name": str(index_by_id[sample_id]["template_name"]),
            "valid_json": False,
        }
        page_row = {**common, "Page-EM": 0.0}
        document_row = {**common, "Schema-nTED": 0.0, "Value-nED": 0.0}
        pred_path = pred_dir / f"{sample_id}.json"
        if pred_path.exists():
            n_prediction_files += 1
            try:
                pred = read_json(pred_path)
            except Exception:
                n_invalid_json += 1
            else:
                n_valid_json += 1
                page_row["valid_json"] = True
                document_row["valid_json"] = True
                page_row["Page-EM"] = page_exact_match(pred, page_sample.normalized_gt)
                answer = unwrap_answer(pred)
                document_row["Schema-nTED"] = schema_nted(
                    build_schema_tree(answer), document_sample.schema
                )
                document_row["Value-nED"] = value_ned(
                    extract_normalized_values(answer), document_sample.values
                )
        page_rows[sample_id] = page_row
        document_rows[sample_id] = document_row

    indexed_ids = set(expected_ids)
    n_extra_prediction_files = sum(
        path.stem not in indexed_ids for path in pred_dir.glob("*.json")
    )
    counts = {
        "n_total": len(expected_ids),
        "n_prediction_files": n_prediction_files,
        "n_extra_prediction_files": n_extra_prediction_files,
        "n_valid_json": n_valid_json,
        "n_missing_prediction": len(expected_ids) - n_prediction_files,
        "n_invalid_json": n_invalid_json,
    }
    page_aggregate = _model_csv_row(run_dir / "page_em_results.csv", model)
    document_aggregate = _model_csv_row(
        run_dir / "document_similarity_results.csv", model
    )
    for report_name, aggregate in (
        ("page_em_results", page_aggregate),
        ("document_similarity_results", document_aggregate),
    ):
        context = f"{run_dir}/{report_name}/{model}"
        for field, expected in counts.items():
            _validate_aggregate_count(
                aggregate, field=field, expected=expected, context=context
            )

    page_mean = _mean([float(row["Page-EM"]) for row in page_rows.values()])
    schema_mean = _mean(
        [float(row["Schema-nTED"]) for row in document_rows.values()]
    )
    value_mean = _mean([float(row["Value-nED"]) for row in document_rows.values()])
    if page_mean is None or schema_mean is None or value_mean is None:
        raise ValueError("cannot compute document metrics for an empty frozen index")
    _validate_aggregate_metric(
        page_aggregate,
        field="Page-EM",
        expected=page_mean,
        context=f"{run_dir}/page_em_results/{model}",
    )
    _validate_aggregate_metric(
        document_aggregate,
        field="Schema-nTED",
        expected=schema_mean,
        context=f"{run_dir}/document_similarity_results/{model}",
    )
    _validate_aggregate_metric(
        document_aggregate,
        field="Value-nED",
        expected=value_mean,
        context=f"{run_dir}/document_similarity_results/{model}",
    )
    return {"page_em": page_rows, "document_similarity": document_rows}


def _as_validity(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int) and value in {0, 1}:
        return float(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return float(value.strip().casefold() == "true")
    raise ValueError(f"invalid valid_json value at {context}: {value!r}")


def _as_optional_float(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in NA_STRINGS:
            return None
        value = stripped
    if isinstance(value, bool):
        raise ValueError(f"boolean is not a numeric metric at {context}: {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric metric at {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric metric at {context}: {value!r}")
    return parsed


def _validate_validity_consistency(
    sources: dict[str, dict[str, dict[str, Any]]],
    *,
    run_label: str,
) -> None:
    for sample_id in sources["legacy"]:
        values: dict[str, float] = {}
        for source, rows in sources.items():
            if "valid_json" not in rows[sample_id]:
                raise ValueError(
                    f"missing valid_json in {run_label}/{source}, sample {sample_id!r}"
                )
            values[source] = _as_validity(
                rows[sample_id]["valid_json"],
                context=f"{run_label}/{source}/{sample_id}/valid_json",
            )
        if len(set(values.values())) != 1:
            raise ValueError(
                f"valid_json disagreement for {run_label}, sample {sample_id!r}: {values}"
            )


def _select_metrics(
    tuned: dict[str, dict[str, dict[str, Any]]],
    base: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[MetricSpec], list[str]]:
    selected: list[MetricSpec] = []
    notes: list[str] = []
    for spec in METRICS:
        missing: list[str] = []
        for side, sources in (("tuned", tuned), ("base", base)):
            missing_ids = [
                sample_id
                for sample_id, row in sources[spec.source].items()
                if spec.field not in row
            ]
            if missing_ids:
                missing.append(f"{side}: {len(missing_ids)} rows, e.g. {missing_ids[:3]}")
        if missing:
            message = (
                f"metric {spec.name!r} is missing source field {spec.field!r} "
                f"from {spec.source}: {'; '.join(missing)}"
            )
            if spec.required:
                raise ValueError(message)
            notes.append(f"Skipped optional {message}.")
            continue
        selected.append(spec)
    return selected, notes


def _metric_value(
    spec: MetricSpec,
    sources: dict[str, dict[str, dict[str, Any]]],
    sample_id: str,
    *,
    run_label: str,
) -> float | None:
    raw = sources[spec.source][sample_id][spec.field]
    context = f"{run_label}/{spec.source}/{sample_id}/{spec.field}"
    if spec.name == "valid_json":
        return _as_validity(raw, context=context)
    return _as_optional_float(raw, context=context)


def _outcome(delta: float | None, tolerance: float) -> str:
    if delta is None:
        return ""
    if delta > tolerance:
        return "win"
    if delta < -tolerance:
        return "loss"
    return "tie"


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _cluster_bootstrap_ci(
    deltas: Sequence[tuple[str, float]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None, int]:
    by_template: dict[str, list[float]] = {}
    for template_name, delta in deltas:
        by_template.setdefault(template_name, []).append(delta)
    if not by_template:
        return None, None, 0

    templates = sorted(by_template)
    cluster_sums = np.asarray(
        [math.fsum(by_template[name]) for name in templates], dtype=np.float64
    )
    cluster_counts = np.asarray(
        [len(by_template[name]) for name in templates], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(templates), size=(iterations, len(templates)))
    estimates = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper), len(templates)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.{digits}f}"


def _format_ci(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "NA"
    return f"[{lower:.6f}, {upper:.6f}]"


def _format_comparison_value(row: Mapping[str, Any], field: str) -> str:
    value = row[field]
    if row.get("value_kind") == "count":
        return str(int(value))
    return _fmt(value)


def _append_hierarchical_aggregate_table(
    lines: list[str],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        row
        for row in aggregate_rows
        if row.get("report") == "hierarchical_structure_metrics"
        and str(row.get("field") or "").endswith(("-micro", "-corpus"))
    ]
    if not rows:
        return
    lines.extend(
        [
            "",
            "## Hierarchical aggregate diagnostics",
            "",
            "| Field | Tuned | Base | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['field']} | {_format_comparison_value(row, 'tuned')} | "
            f"{_format_comparison_value(row, 'base')} | "
            f"{_format_comparison_value(row, 'delta')} |"
        )


def _append_relation_type_table(
    lines: list[str],
    relation_type_rows: Sequence[Mapping[str, Any]],
) -> None:
    by_type: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in relation_type_rows:
        relation_type = str(row["relation_type"])
        field = str(row["field"])
        by_type.setdefault(relation_type, {})[field] = row
    if not by_type:
        return

    lines.extend(
        [
            "",
            "## Relation-type diagnostics",
            "",
            "| Relation type | GT | Tuned TP/pred | Base TP/pred | Tuned P/R/F1 | Base P/R/F1 | Delta F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for relation_type in sorted(by_type):
        fields = by_type[relation_type]
        missing = set(RELATION_TYPE_FIELDS) - set(fields)
        if missing:
            raise ValueError(
                f"relation comparison rows for {relation_type!r} are missing {sorted(missing)}"
            )
        tp = fields["TP"]
        pred = fields["pred"]
        gt = fields["GT"]
        precision = fields["precision"]
        recall = fields["recall"]
        f1 = fields["F1"]
        escaped_type = relation_type.replace("|", "\\|")
        lines.append(
            f"| {escaped_type} | {_format_comparison_value(gt, 'tuned')} | "
            f"{_format_comparison_value(tp, 'tuned')}/{_format_comparison_value(pred, 'tuned')} | "
            f"{_format_comparison_value(tp, 'base')}/{_format_comparison_value(pred, 'base')} | "
            f"{_format_comparison_value(precision, 'tuned')}/"
            f"{_format_comparison_value(recall, 'tuned')}/"
            f"{_format_comparison_value(f1, 'tuned')} | "
            f"{_format_comparison_value(precision, 'base')}/"
            f"{_format_comparison_value(recall, 'base')}/"
            f"{_format_comparison_value(f1, 'base')} | "
            f"{_format_comparison_value(f1, 'delta')} |"
        )


def _build_analysis(
    *,
    index_path: Path,
    tuned_dir: Path,
    tuned_model: str,
    base_dir: Path,
    base_model: str,
    n_samples: int,
    n_templates: int,
    iterations: int,
    seed: int,
    overall_rows: list[dict[str, Any]],
    template_rows: list[dict[str, Any]],
    notes: list[str],
    aggregate_rows: Sequence[Mapping[str, Any]] = (),
    relation_type_rows: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [
        "# Qwen3.6 FormTSR tuned/base comparison",
        "",
        "## Scope",
        "",
        f"- Index: `{index_path}` ({n_samples} pages, {n_templates} templates)",
        f"- Tuned: `{tuned_model}` from `{tuned_dir}`",
        f"- Base: `{base_model}` from `{base_dir}`",
        "- Delta and page outcomes are always tuned minus base; higher is better for every listed metric.",
        "",
        "## Overall",
        "",
        "| Metric | Paired n | Tuned | Base | Delta | Template-cluster bootstrap 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['label']} | {row['n_paired']} | {_fmt(row['tuned_mean'])} | "
            f"{_fmt(row['base_mean'])} | {_fmt(row['delta'])} | "
            f"{_format_ci(row['ci95_low'], row['ci95_high'])} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )

    _append_hierarchical_aggregate_table(lines, aggregate_rows)
    _append_relation_type_table(lines, relation_type_rows)

    directional_gains = [
        row
        for row in overall_rows
        if row["ci95_low"] is not None and row["ci95_low"] > 0.0
    ]
    directional_losses = [
        row
        for row in overall_rows
        if row["ci95_high"] is not None and row["ci95_high"] < 0.0
    ]
    lines.extend(["", "## Bootstrap reading", ""])
    if directional_gains:
        lines.append(
            "- Positive intervals excluding zero: "
            + ", ".join(f"{row['label']} ({_fmt(row['delta'])})" for row in directional_gains)
            + "."
        )
    else:
        lines.append("- No positive interval excludes zero.")
    if directional_losses:
        lines.append(
            "- Negative intervals excluding zero: "
            + ", ".join(f"{row['label']} ({_fmt(row['delta'])})" for row in directional_losses)
            + "."
        )
    else:
        lines.append("- No negative interval excludes zero.")
    lines.append(
        "- These are descriptive cluster-bootstrap intervals, not a multiple-comparison-adjusted hypothesis test."
    )

    template_metrics = {
        "Page-EM",
        "Schema-nTED",
        "Value-nED",
        "TSR-path",
        "VAcc",
        "corrected_R-F1",
        "LG-GriTS-Top",
        "WG-F1",
        "Rel-F1",
    }
    labels = {row["metric"]: row["label"] for row in overall_rows}
    lines.extend(["", "## Template extremes", ""])
    for metric in [spec.name for spec in METRICS if spec.name in template_metrics]:
        rows = [
            row
            for row in template_rows
            if row["metric"] == metric and row["delta"] is not None
        ]
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: (row["delta"], row["template_name"]))
        low = ", ".join(
            f"{row['template_name']} ({row['delta']:+.6f})" for row in ordered[:3]
        )
        high = ", ".join(
            f"{row['template_name']} ({row['delta']:+.6f})" for row in reversed(ordered[-3:])
        )
        lines.append(f"- {labels[metric]}: strongest gains {high}; strongest declines {low}.")

    availability_notes = []
    for row in overall_rows:
        if row["n_tuned_numeric"] != row["n_base_numeric"] or row["n_unpaired_numeric"]:
            availability_notes.append(
                f"{row['label']}: tuned numeric={row['n_tuned_numeric']}, "
                f"base numeric={row['n_base_numeric']}, paired={row['n_paired']}"
            )
    lines.extend(["", "## Data and method notes", ""])
    lines.extend(
        [
            "- All means and win/tie/loss counts use paired numeric pages. `NA` means the metric is not applicable and is never converted to zero.",
            "- Page-EM, Schema-nTED, and Value-nED are recomputed with the aggregate-report functions from frozen labels and prediction JSON. Missing or invalid predictions score zero, matching those reports.",
            f"- The 95% interval uses {iterations} fixed-seed ({seed}) bootstrap replicates. Each replicate samples whole templates with replacement and retains all pages in every sampled template.",
            "- Legacy R-F1/LIG-F1 and CDS are retained for continuity; corrected coordinate-normalized structure metrics should drive structure conclusions.",
            "- A cluster interval crossing zero means the observed average direction is not stable under template resampling; it does not prove equality.",
        ]
    )
    for note in availability_notes:
        lines.append(f"- Availability mismatch: {note}.")
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be at least 1")
    if args.tie_tolerance < 0:
        raise ValueError("--tie-tolerance must be non-negative")

    index_path = Path(args.index)
    tuned_dir = Path(args.tuned_dir)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    index_rows, index_by_id = _load_index(index_path)
    tuned = _load_run_sources(
        tuned_dir,
        args.tuned_model,
        index_path,
        index_rows,
        index_by_id,
    )
    base = _load_run_sources(
        base_dir,
        args.base_model,
        index_path,
        index_rows,
        index_by_id,
    )
    _validate_validity_consistency(tuned, run_label="tuned")
    _validate_validity_consistency(base, run_label="base")
    metrics, notes = _select_metrics(tuned, base)
    aggregate_rows = _load_aggregate_report_comparison(
        tuned_dir=tuned_dir,
        tuned_model=args.tuned_model,
        base_dir=base_dir,
        base_model=args.base_model,
    )
    relation_type_rows = _load_relation_type_comparison(
        tuned_dir=tuned_dir,
        tuned_model=args.tuned_model,
        base_dir=base_dir,
        base_model=args.base_model,
    )

    metric_values: dict[str, dict[str, dict[str, float | None]]] = {}
    for spec in metrics:
        metric_values[spec.name] = {"tuned": {}, "base": {}}
        for sample_id in index_by_id:
            metric_values[spec.name]["tuned"][sample_id] = _metric_value(
                spec, tuned, sample_id, run_label="tuned"
            )
            metric_values[spec.name]["base"][sample_id] = _metric_value(
                spec, base, sample_id, run_label="base"
            )

    index_columns = [
        column
        for column in ("sample_id", "template_name", "instance_id", "image_path", "label_path")
        if any(column in row for row in index_rows)
    ]
    per_sample_columns = list(index_columns)
    per_sample_rows: list[dict[str, Any]] = []
    for index_row in index_rows:
        sample_id = str(index_row["sample_id"])
        output = {column: index_row.get(column, "") for column in index_columns}
        for spec in metrics:
            tuned_value = metric_values[spec.name]["tuned"][sample_id]
            base_value = metric_values[spec.name]["base"][sample_id]
            delta = (
                tuned_value - base_value
                if tuned_value is not None and base_value is not None
                else None
            )
            output[f"tuned_{spec.name}"] = tuned_value
            output[f"base_{spec.name}"] = base_value
            output[f"delta_{spec.name}"] = delta
            output[f"outcome_{spec.name}"] = _outcome(delta, args.tie_tolerance)
        per_sample_rows.append(output)
    for spec in metrics:
        per_sample_columns.extend(
            [
                f"tuned_{spec.name}",
                f"base_{spec.name}",
                f"delta_{spec.name}",
                f"outcome_{spec.name}",
            ]
        )

    overall_rows: list[dict[str, Any]] = []
    for spec in metrics:
        tuned_values = metric_values[spec.name]["tuned"]
        base_values = metric_values[spec.name]["base"]
        paired = [
            sample_id
            for sample_id in index_by_id
            if tuned_values[sample_id] is not None and base_values[sample_id] is not None
        ]
        paired_tuned = [float(tuned_values[sample_id]) for sample_id in paired]
        paired_base = [float(base_values[sample_id]) for sample_id in paired]
        deltas = [left - right for left, right in zip(paired_tuned, paired_base)]
        outcomes = [_outcome(delta, args.tie_tolerance) for delta in deltas]
        clustered_deltas = [
            (str(index_by_id[sample_id]["template_name"]), delta)
            for sample_id, delta in zip(paired, deltas)
        ]
        ci_low, ci_high, n_clusters = _cluster_bootstrap_ci(
            clustered_deltas,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        n_tuned = sum(value is not None for value in tuned_values.values())
        n_base = sum(value is not None for value in base_values.values())
        n_union = sum(
            tuned_values[sample_id] is not None or base_values[sample_id] is not None
            for sample_id in index_by_id
        )
        overall_rows.append(
            {
                "metric": spec.name,
                "label": spec.label,
                "source": spec.source,
                "source_field": spec.field,
                "n_total": len(index_rows),
                "n_tuned_numeric": n_tuned,
                "n_base_numeric": n_base,
                "n_paired": len(paired),
                "n_unpaired_numeric": n_union - len(paired),
                "n_templates_paired": n_clusters,
                "tuned_mean": _mean(paired_tuned),
                "base_mean": _mean(paired_base),
                "delta": _mean(deltas),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wins": outcomes.count("win"),
                "ties": outcomes.count("tie"),
                "losses": outcomes.count("loss"),
            }
        )

    template_rows: list[dict[str, Any]] = []
    templates = sorted({str(row["template_name"]) for row in index_rows})
    for template_name in templates:
        sample_ids = [
            str(row["sample_id"])
            for row in index_rows
            if str(row["template_name"]) == template_name
        ]
        for spec in metrics:
            tuned_values = metric_values[spec.name]["tuned"]
            base_values = metric_values[spec.name]["base"]
            paired = [
                sample_id
                for sample_id in sample_ids
                if tuned_values[sample_id] is not None
                and base_values[sample_id] is not None
            ]
            paired_tuned = [float(tuned_values[sample_id]) for sample_id in paired]
            paired_base = [float(base_values[sample_id]) for sample_id in paired]
            deltas = [left - right for left, right in zip(paired_tuned, paired_base)]
            outcomes = [_outcome(delta, args.tie_tolerance) for delta in deltas]
            template_rows.append(
                {
                    "template_name": template_name,
                    "metric": spec.name,
                    "label": spec.label,
                    "source": spec.source,
                    "n_total": len(sample_ids),
                    "n_tuned_numeric": sum(
                        tuned_values[sample_id] is not None for sample_id in sample_ids
                    ),
                    "n_base_numeric": sum(
                        base_values[sample_id] is not None for sample_id in sample_ids
                    ),
                    "n_paired": len(paired),
                    "tuned_mean": _mean(paired_tuned),
                    "base_mean": _mean(paired_base),
                    "delta": _mean(deltas),
                    "wins": outcomes.count("win"),
                    "ties": outcomes.count("tie"),
                    "losses": outcomes.count("loss"),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_sample_comparison.csv", per_sample_columns, per_sample_rows)
    overall_columns = [
        "metric",
        "label",
        "source",
        "source_field",
        "n_total",
        "n_tuned_numeric",
        "n_base_numeric",
        "n_paired",
        "n_unpaired_numeric",
        "n_templates_paired",
        "tuned_mean",
        "base_mean",
        "delta",
        "ci95_low",
        "ci95_high",
        "wins",
        "ties",
        "losses",
    ]
    _write_csv(out_dir / "overall_comparison.csv", overall_columns, overall_rows)
    template_columns = [
        "template_name",
        "metric",
        "label",
        "source",
        "n_total",
        "n_tuned_numeric",
        "n_base_numeric",
        "n_paired",
        "tuned_mean",
        "base_mean",
        "delta",
        "wins",
        "ties",
        "losses",
    ]
    _write_csv(out_dir / "template_comparison.csv", template_columns, template_rows)
    _write_csv(
        out_dir / "aggregate_report_comparison.csv",
        AGGREGATE_COMPARISON_COLUMNS,
        aggregate_rows,
    )
    _write_csv(
        out_dir / "relation_type_comparison.csv",
        RELATION_TYPE_COMPARISON_COLUMNS,
        relation_type_rows,
    )
    analysis = _build_analysis(
        index_path=index_path,
        tuned_dir=tuned_dir,
        tuned_model=args.tuned_model,
        base_dir=base_dir,
        base_model=args.base_model,
        n_samples=len(index_rows),
        n_templates=len(templates),
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        overall_rows=overall_rows,
        template_rows=template_rows,
        notes=notes,
        aggregate_rows=aggregate_rows,
        relation_type_rows=relation_type_rows,
    )
    (out_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    print(
        f"Compared {len(index_rows)} pages across {len(templates)} templates; "
        f"wrote {len(metrics)} paired metrics, {len(aggregate_rows)} aggregate fields, "
        f"and {len(relation_type_rows)} relation-type fields to {out_dir}"
    )


if __name__ == "__main__":
    main()
