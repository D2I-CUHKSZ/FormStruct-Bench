from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from .metrics import NA
from .results import (
    DEFAULT_DIFFICULTY_CSV,
    DIFFICULTY_LEVEL_LABELS,
    DIFFICULTY_LEVEL_ORDER,
    METRIC_COLUMNS,
    format_value,
    load_difficulty_lookup,
)


LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}
LEVEL_SEQUENCE = ["low", "medium", "high"]

PER_SAMPLE_COLUMNS = [
    "model",
    "model_id",
    "group",
    "clean_model",
    "clean_model_id",
    "clean_sample_id",
    "degraded_sample_id",
    "template_name",
    "instance_id",
    "difficulty_level",
    "difficulty_name",
    "difficulty_D_main",
    "difficulty_S_form",
    "difficulty_C_context",
    "degradation_variant",
    "degradation_level",
    "clean_valid_json",
    "degraded_valid_json",
    "clean_CDS",
    "degraded_CDS",
    "CDS_drop",
    "relative_drop_pct",
    "status",
]

SUMMARY_COLUMNS = [
    "model",
    "model_id",
    "group",
    "degradation_variant",
    "degradation_level",
    "n_total",
    "n_paired_clean",
    "n_degraded_valid_json",
    "invalid_rate",
    "clean_CDS",
    "degraded_CDS",
    "CDS_drop",
    "relative_drop_pct",
]

BOUNDARY_COLUMNS = [
    "model",
    "model_id",
    "group",
    "degradation_variant",
    "clean_CDS",
    "failure_boundary",
    "failure_reason",
    "boundary_drop",
    "boundary_degraded_CDS",
    "low_drop",
    "medium_drop",
    "high_drop",
]

CLEAN_COLUMNS = [
    "model",
    "model_id",
    "group",
    "n_total",
    "n_valid_json",
    "invalid_rate",
    *METRIC_COLUMNS,
]

CLEAN_DIFFICULTY_COLUMNS = [
    "model",
    "model_id",
    "group",
    "difficulty_level",
    "difficulty_name",
    "n_clean_samples",
    "n_clean_valid_json",
    "invalid_rate",
    *[f"clean_{metric}" for metric in METRIC_COLUMNS],
]

DETAILED_GROUP_COLUMNS = [
    "model",
    "model_id",
    "group",
    "degradation_variant",
    "degradation_level",
    "difficulty_level",
    "difficulty_name",
    "n_total",
    "n_templates",
    "n_paired_clean",
    "n_degraded_valid_json",
    "invalid_rate",
]

DETAILED_METRIC_COLUMNS = [
    column
    for metric in METRIC_COLUMNS
    for column in (
        f"clean_{metric}",
        f"degraded_{metric}",
        f"{metric}_drop",
        f"{metric}_relative_drop_pct",
    )
]

DETAILED_SUMMARY_COLUMNS = [*DETAILED_GROUP_COLUMNS, *DETAILED_METRIC_COLUMNS]

BOUNDARY_DIFFICULTY_COLUMNS = [
    "model",
    "model_id",
    "group",
    "degradation_variant",
    "difficulty_level",
    "difficulty_name",
    "clean_CDS",
    "failure_boundary",
    "failure_reason",
    "boundary_drop",
    "boundary_degraded_CDS",
    "low_drop",
    "medium_drop",
    "high_drop",
]


def _split_csv(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() == NA:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _mean(values: list[Any]) -> float | str:
    nums = [num for value in values for num in [_num(value)] if num is not None]
    return (sum(nums) / len(nums)) if nums else NA


def _relative_drop(drop: Any, clean_value: Any) -> float | str:
    drop_num = _num(drop)
    clean_num = _num(clean_value)
    if drop_num is None or clean_num is None or clean_num == 0:
        return NA
    return 100.0 * drop_num / clean_num


def _drop(clean_value: Any, degraded_value: Any) -> float | str:
    clean_num = _num(clean_value)
    degraded_num = _num(degraded_value)
    if clean_num is None or degraded_num is None:
        return NA
    return clean_num - degraded_num


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in columns})


def _latex_table(path: Path, columns: list[str], rows: list[dict[str, Any]], *, max_rows: int = 80) -> None:
    ensure_dir(path.parent)
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row in rows[:max_rows]:
        lines.append(" & ".join(format_value(row.get(key)) for key in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _model_key(row: dict[str, Any]) -> str:
    return str(row.get("model", ""))


def _model_id_key(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or row.get("model") or "")


def _difficulty_sort_value(level: Any) -> int:
    value = str(level or "")
    try:
        return DIFFICULTY_LEVEL_ORDER.index(value)
    except ValueError:
        return 99


def _difficulty_fields(info: dict[str, Any] | None) -> dict[str, Any]:
    if not info:
        return {
            "difficulty_level": NA,
            "difficulty_name": NA,
            "difficulty_D_main": NA,
            "difficulty_S_form": NA,
            "difficulty_C_context": NA,
        }
    fields = {
        "difficulty_level": info.get("difficulty_level", NA),
        "difficulty_name": info.get("difficulty_name", NA),
        "difficulty_D_main": info.get("D_main", NA),
        "difficulty_S_form": info.get("S_form", NA),
        "difficulty_C_context": info.get("C_context", NA),
    }
    if "difficulty_score" in info and fields["difficulty_D_main"] == NA:
        fields["difficulty_D_main"] = info["difficulty_score"]
    return fields


def _clean_metric_lookup(clean_metrics: list[dict[str, Any]], clean_ids: set[str]) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    by_model_sample: dict[tuple[str, str], dict[str, Any]] = {}
    by_model_id_sample: dict[tuple[str, str], dict[str, Any]] = {}
    for row in clean_metrics:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in clean_ids:
            continue
        by_model_sample[(str(row.get("model", "")), sample_id)] = row
        model_id = _model_id_key(row)
        if model_id:
            key = (model_id, sample_id)
            current = by_model_id_sample.get(key)
            if current is None or (not current.get("valid_json") and row.get("valid_json")):
                by_model_id_sample[key] = row
    return by_model_sample, by_model_id_sample


def _paired_clean_row(
    degraded_row: dict[str, Any],
    clean_sample_id: str,
    clean_by_model_sample: dict[tuple[str, str], dict[str, Any]],
    clean_by_model_id_sample: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    model = str(degraded_row.get("model", ""))
    clean_row = clean_by_model_sample.get((model, clean_sample_id))
    if clean_row is not None:
        return clean_row
    model_id = _model_id_key(degraded_row)
    if model_id:
        return clean_by_model_id_sample.get((model_id, clean_sample_id))
    return None


def _summarize_clean(clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_model[_model_key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for model, rows in sorted(by_model.items()):
        n_total = len(rows)
        n_valid = sum(1 for row in rows if row.get("valid_json"))
        summary: dict[str, Any] = {
            "model": model,
            "model_id": rows[0].get("model_id", model) if rows else model,
            "group": rows[0].get("group", "") if rows else "",
            "n_total": n_total,
            "n_valid_json": n_valid,
            "invalid_rate": (1 - n_valid / n_total) if n_total else NA,
        }
        for metric in METRIC_COLUMNS:
            summary[metric] = _mean([row.get(metric) for row in rows])
        summaries.append(summary)
    return summaries


def _summarize_clean_by_difficulty(
    clean_rows: list[dict[str, Any]],
    difficulty_by_template: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        info = difficulty_by_template.get(str(row.get("template_name") or ""))
        if not info:
            continue
        grouped[(str(row.get("model", "")), str(info.get("difficulty_level", "")))].append(row)

    summaries: list[dict[str, Any]] = []
    for (model, level), rows in sorted(grouped.items(), key=lambda item: (item[0][0], _difficulty_sort_value(item[0][1]), item[0][1])):
        info = difficulty_by_template.get(str(rows[0].get("template_name") or ""))
        n_total = len(rows)
        n_valid = sum(1 for row in rows if row.get("valid_json"))
        summary: dict[str, Any] = {
            "model": model,
            "model_id": rows[0].get("model_id", model),
            "group": rows[0].get("group", ""),
            "difficulty_level": level,
            "difficulty_name": DIFFICULTY_LEVEL_LABELS.get(level, info.get("difficulty_name", "")) if info else "",
            "n_clean_samples": n_total,
            "n_clean_valid_json": n_valid,
            "invalid_rate": (1 - n_valid / n_total) if n_total else NA,
        }
        for metric in METRIC_COLUMNS:
            summary[f"clean_{metric}"] = _mean([row.get(metric) for row in rows])
        summaries.append(summary)
    return summaries


def _metric_drop_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRIC_COLUMNS:
        clean_value = _mean([row.get(f"clean_{metric}") for row in rows])
        degraded_value = _mean([row.get(f"degraded_{metric}") for row in rows])
        drop = _drop(clean_value, degraded_value)
        summary[f"clean_{metric}"] = clean_value
        summary[f"degraded_{metric}"] = degraded_value
        summary[f"{metric}_drop"] = drop
        summary[f"{metric}_relative_drop_pct"] = _relative_drop(drop, clean_value)
    return summary


def _level_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    level = str(row.get("degradation_level", ""))
    return (str(row.get("degradation_variant", "")), LEVEL_ORDER.get(level, 99), level)


def build_robustness_report(
    *,
    clean_metrics_path: Path | None,
    clean_main_metrics_path: Path | None,
    degraded_metrics_path: Path,
    clean_index_path: Path,
    degraded_index_path: Path,
    out_dir: Path,
    difficulty_csv: Path | None = None,
    layout_root: Path | None = None,
    models: set[str] | None = None,
    drop_threshold: float = 0.10,
    relative_boundary_ratio: float = 0.50,
) -> dict[str, Any]:
    clean_index = read_jsonl(clean_index_path)
    degraded_index = read_jsonl(degraded_index_path)
    clean_metrics_source = clean_main_metrics_path or clean_metrics_path
    if clean_metrics_source is None:
        raise ValueError("either clean_metrics_path or clean_main_metrics_path is required")
    clean_metrics = read_jsonl(clean_metrics_source)
    degraded_metrics = read_jsonl(degraded_metrics_path)
    if models:
        degraded_metrics = [row for row in degraded_metrics if str(row.get("model")) in models]
        selected_model_ids = {_model_id_key(row) for row in degraded_metrics if _model_id_key(row)}
        clean_metrics = [
            row
            for row in clean_metrics
            if str(row.get("model")) in models or _model_id_key(row) in selected_model_ids
        ]

    clean_ids = {str(row["sample_id"]) for row in clean_index}
    degraded_by_id = {str(row["sample_id"]): row for row in degraded_index}
    clean_metric_by_model_sample, clean_metric_by_model_id_sample = _clean_metric_lookup(clean_metrics, clean_ids)
    difficulty_lookup = load_difficulty_lookup(
        [*clean_index, *degraded_index],
        difficulty_csv=difficulty_csv,
        layout_root=layout_root,
    )

    per_sample_rows: list[dict[str, Any]] = []
    paired_clean_metrics_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for degraded_row in degraded_metrics:
        degraded_sample_id = str(degraded_row.get("sample_id", ""))
        index_row = degraded_by_id.get(degraded_sample_id)
        if index_row is None:
            continue
        model = str(degraded_row.get("model", ""))
        clean_sample_id = str(index_row.get("clean_sample_id") or "")
        clean_row = _paired_clean_row(
            degraded_row,
            clean_sample_id,
            clean_metric_by_model_sample,
            clean_metric_by_model_id_sample,
        )
        if clean_row is not None:
            paired_clean_metrics_by_key[
                (str(clean_row.get("model", "")), str(clean_row.get("sample_id", "")))
            ] = clean_row
        difficulty = _difficulty_fields(difficulty_lookup.get(str(index_row.get("template_name") or "")))
        clean_cds = clean_row.get("CDS") if clean_row else NA
        degraded_cds = degraded_row.get("CDS")
        drop = _drop(clean_cds, degraded_cds)
        status = "ok" if clean_row is not None and _num(clean_cds) is not None and _num(degraded_cds) is not None else "unpaired_or_non_numeric"
        row = {
            "model": model,
            "model_id": degraded_row.get("model_id", model),
            "group": degraded_row.get("group", ""),
            "clean_model": clean_row.get("model") if clean_row else NA,
            "clean_model_id": clean_row.get("model_id") if clean_row else NA,
            "clean_sample_id": clean_sample_id,
            "degraded_sample_id": degraded_sample_id,
            "template_name": index_row.get("template_name"),
            "instance_id": index_row.get("instance_id"),
            **difficulty,
            "degradation_variant": index_row.get("degradation_variant"),
            "degradation_level": index_row.get("degradation_level"),
            "clean_valid_json": clean_row.get("valid_json") if clean_row else False,
            "degraded_valid_json": degraded_row.get("valid_json"),
            "clean_CDS": clean_cds,
            "degraded_CDS": degraded_cds,
            "CDS_drop": drop,
            "relative_drop_pct": _relative_drop(drop, clean_cds),
            "status": status,
        }
        for metric in METRIC_COLUMNS:
            row[f"clean_{metric}"] = clean_row.get(metric) if clean_row else NA
            row[f"degraded_{metric}"] = degraded_row.get(metric)
            metric_drop = _drop(row[f"clean_{metric}"], row[f"degraded_{metric}"])
            row[f"{metric}_drop"] = metric_drop
            row[f"{metric}_relative_drop_pct"] = _relative_drop(metric_drop, row[f"clean_{metric}"])
        per_sample_rows.append(row)
    paired_clean_metrics = list(paired_clean_metrics_by_key.values())

    summary_rows: list[dict[str, Any]] = []
    summary_by_model_variant_level: dict[tuple[str, str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        grouped[(str(row["model"]), str(row["degradation_variant"]), str(row["degradation_level"]))].append(row)

    for (model, variant, level), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], LEVEL_ORDER.get(item[0][2], 99), item[0][2])):
        n_total = len(rows)
        n_valid = sum(1 for row in rows if row.get("degraded_valid_json"))
        clean_cds = _mean([row.get("clean_CDS") for row in rows])
        degraded_cds = _mean([row.get("degraded_CDS") for row in rows])
        drop = _drop(clean_cds, degraded_cds)
        summary = {
            "model": model,
            "model_id": rows[0].get("model_id", model),
            "group": rows[0].get("group", ""),
            "degradation_variant": variant,
            "degradation_level": level,
            "n_total": n_total,
            "n_paired_clean": sum(1 for row in rows if _num(row.get("clean_CDS")) is not None),
            "n_degraded_valid_json": n_valid,
            "invalid_rate": (1 - n_valid / n_total) if n_total else NA,
            "clean_CDS": clean_cds,
            "degraded_CDS": degraded_cds,
            "CDS_drop": drop,
            "relative_drop_pct": _relative_drop(drop, clean_cds),
        }
        summary_rows.append(summary)
        summary_by_model_variant_level[(model, variant, level)] = summary

    boundary_rows: list[dict[str, Any]] = []
    model_variant_pairs = sorted({(str(row["model"]), str(row["degradation_variant"])) for row in summary_rows})
    for model, variant in model_variant_pairs:
        summaries = [summary_by_model_variant_level.get((model, variant, level)) for level in LEVEL_SEQUENCE]
        summaries = [summary for summary in summaries if summary is not None]
        if not summaries:
            continue
        clean_cds = _mean([summary.get("clean_CDS") for summary in summaries])
        boundary = "not_reached"
        reason = "drop thresholds not reached"
        boundary_drop: Any = NA
        boundary_degraded: Any = NA
        clean_num = _num(clean_cds)
        for summary in summaries:
            drop_num = _num(summary.get("CDS_drop"))
            degraded_num = _num(summary.get("degraded_CDS"))
            level = str(summary.get("degradation_level"))
            if drop_num is not None and drop_num >= drop_threshold:
                boundary = level
                reason = f"mean_drop >= {drop_threshold:.4f}"
                boundary_drop = summary.get("CDS_drop")
                boundary_degraded = summary.get("degraded_CDS")
                break
            if clean_num is not None and degraded_num is not None and degraded_num < relative_boundary_ratio * clean_num:
                boundary = level
                reason = f"degraded_CDS < {relative_boundary_ratio:.2f} * clean_CDS"
                boundary_drop = summary.get("CDS_drop")
                boundary_degraded = summary.get("degraded_CDS")
                break
        first = summaries[0]
        row = {
            "model": model,
            "model_id": first.get("model_id", model),
            "group": first.get("group", ""),
            "degradation_variant": variant,
            "clean_CDS": clean_cds,
            "failure_boundary": boundary,
            "failure_reason": reason,
            "boundary_drop": boundary_drop,
            "boundary_degraded_CDS": boundary_degraded,
        }
        for level in LEVEL_SEQUENCE:
            summary = summary_by_model_variant_level.get((model, variant, level))
            row[f"{level}_drop"] = summary.get("CDS_drop") if summary else NA
        boundary_rows.append(row)

    clean_summaries = _summarize_clean(paired_clean_metrics)
    clean_difficulty_summaries = _summarize_clean_by_difficulty(paired_clean_metrics, difficulty_lookup)

    detailed_grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        detailed_grouped[
            (
                str(row["model"]),
                str(row["degradation_variant"]),
                str(row["degradation_level"]),
                str(row.get("difficulty_level", "")),
            )
        ].append(row)

    detailed_rows: list[dict[str, Any]] = []
    detailed_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for (model, variant, degradation_level, difficulty_level), rows in sorted(
        detailed_grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            LEVEL_ORDER.get(item[0][2], 99),
            item[0][2],
            _difficulty_sort_value(item[0][3]),
            item[0][3],
        ),
    ):
        n_total = len(rows)
        n_valid = sum(1 for row in rows if row.get("degraded_valid_json"))
        summary = {
            "model": model,
            "model_id": rows[0].get("model_id", model),
            "group": rows[0].get("group", ""),
            "degradation_variant": variant,
            "degradation_level": degradation_level,
            "difficulty_level": difficulty_level,
            "difficulty_name": rows[0].get("difficulty_name", ""),
            "n_total": n_total,
            "n_templates": len({str(row.get("template_name")) for row in rows if row.get("template_name")}),
            "n_paired_clean": sum(1 for row in rows if _num(row.get("clean_CDS")) is not None),
            "n_degraded_valid_json": n_valid,
            "invalid_rate": (1 - n_valid / n_total) if n_total else NA,
        }
        summary.update(_metric_drop_summary(rows))
        detailed_rows.append(summary)
        detailed_by_key[(model, variant, degradation_level, difficulty_level)] = summary

    boundary_difficulty_rows: list[dict[str, Any]] = []
    model_variant_difficulty = sorted(
        {(str(row["model"]), str(row["degradation_variant"]), str(row["difficulty_level"])) for row in detailed_rows},
        key=lambda item: (item[0], item[1], _difficulty_sort_value(item[2]), item[2]),
    )
    for model, variant, difficulty_level in model_variant_difficulty:
        summaries = [detailed_by_key.get((model, variant, level, difficulty_level)) for level in LEVEL_SEQUENCE]
        summaries = [summary for summary in summaries if summary is not None]
        if not summaries:
            continue
        clean_cds = _mean([summary.get("clean_CDS") for summary in summaries])
        boundary = "not_reached"
        reason = "drop thresholds not reached"
        boundary_drop: Any = NA
        boundary_degraded: Any = NA
        clean_num = _num(clean_cds)
        for summary in summaries:
            drop_num = _num(summary.get("CDS_drop"))
            degraded_num = _num(summary.get("degraded_CDS"))
            level = str(summary.get("degradation_level"))
            if drop_num is not None and drop_num >= drop_threshold:
                boundary = level
                reason = f"mean_drop >= {drop_threshold:.4f}"
                boundary_drop = summary.get("CDS_drop")
                boundary_degraded = summary.get("degraded_CDS")
                break
            if clean_num is not None and degraded_num is not None and degraded_num < relative_boundary_ratio * clean_num:
                boundary = level
                reason = f"degraded_CDS < {relative_boundary_ratio:.2f} * clean_CDS"
                boundary_drop = summary.get("CDS_drop")
                boundary_degraded = summary.get("degraded_CDS")
                break
        first = summaries[0]
        row = {
            "model": model,
            "model_id": first.get("model_id", model),
            "group": first.get("group", ""),
            "degradation_variant": variant,
            "difficulty_level": difficulty_level,
            "difficulty_name": first.get("difficulty_name", ""),
            "clean_CDS": clean_cds,
            "failure_boundary": boundary,
            "failure_reason": reason,
            "boundary_drop": boundary_drop,
            "boundary_degraded_CDS": boundary_degraded,
        }
        for level in LEVEL_SEQUENCE:
            summary = detailed_by_key.get((model, variant, level, difficulty_level))
            row[f"{level}_drop"] = summary.get("CDS_drop") if summary else NA
        boundary_difficulty_rows.append(row)

    write_jsonl(out_dir / "robustness_per_sample.jsonl", per_sample_rows)
    _write_csv(out_dir / "robustness_clean_baseline.csv", CLEAN_COLUMNS, clean_summaries)
    _write_csv(out_dir / "robustness_clean_by_difficulty.csv", CLEAN_DIFFICULTY_COLUMNS, clean_difficulty_summaries)
    _write_csv(out_dir / "robustness_by_degradation.csv", SUMMARY_COLUMNS, sorted(summary_rows, key=lambda row: (str(row["model"]), *_level_sort_key(row))))
    _write_csv(out_dir / "robustness_by_degradation_difficulty.csv", DETAILED_SUMMARY_COLUMNS, detailed_rows)
    _write_csv(out_dir / "robustness_failure_boundary.csv", BOUNDARY_COLUMNS, boundary_rows)
    _write_csv(out_dir / "robustness_failure_boundary_by_difficulty.csv", BOUNDARY_DIFFICULTY_COLUMNS, boundary_difficulty_rows)
    _latex_table(out_dir / "robustness_by_degradation_table.tex", SUMMARY_COLUMNS, sorted(summary_rows, key=lambda row: (str(row["model"]), *_level_sort_key(row))))
    _latex_table(out_dir / "robustness_by_degradation_difficulty_table.tex", DETAILED_SUMMARY_COLUMNS, detailed_rows)
    _latex_table(out_dir / "robustness_failure_boundary_table.tex", BOUNDARY_COLUMNS, boundary_rows)
    _latex_table(out_dir / "robustness_failure_boundary_by_difficulty_table.tex", BOUNDARY_DIFFICULTY_COLUMNS, boundary_difficulty_rows)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clean_metrics": str(clean_metrics_source),
        "clean_metrics_source": "main_experiment" if clean_main_metrics_path else "robustness_clean",
        "degraded_metrics": str(degraded_metrics_path),
        "clean_index": str(clean_index_path),
        "degraded_index": str(degraded_index_path),
        "difficulty_csv": str(difficulty_csv or DEFAULT_DIFFICULTY_CSV),
        "difficulty_csv_exists": bool((difficulty_csv or DEFAULT_DIFFICULTY_CSV).exists()),
        "layout_root": str(layout_root) if layout_root else None,
        "models": sorted(models) if models else sorted({str(row.get("model")) for row in clean_metrics + degraded_metrics}),
        "n_clean_index": len(clean_index),
        "n_degraded_index": len(degraded_index),
        "n_clean_metrics": len(clean_metrics),
        "n_paired_clean_metrics": len(paired_clean_metrics),
        "n_degraded_metrics": len(degraded_metrics),
        "n_per_sample_pairs": len(per_sample_rows),
        "n_templates_with_difficulty": len(difficulty_lookup),
        "difficulty_level_order": DIFFICULTY_LEVEL_ORDER,
        "difficulty_level_labels": DIFFICULTY_LEVEL_LABELS,
        "n_degradation_difficulty_rows": len(detailed_rows),
        "n_failure_boundary_difficulty_rows": len(boundary_difficulty_rows),
        "clean_pairing_rule": "Pair degraded rows to clean rows by (model, clean_sample_id); if not found, fall back to (model_id, clean_sample_id). This supports robustness runs whose adapter/run id differs from the clean baseline run id for the same model.",
        "drop_rule": "CDS_drop = clean_CDS - degraded_CDS on the paired clean sample. Positive values indicate robustness loss.",
        "failure_boundary_rule": "For each model and degradation variant, scan levels low -> medium -> high and select the first level where mean_drop >= drop_threshold or degraded_CDS < relative_boundary_ratio * clean_CDS. If neither condition is met, boundary is not_reached.",
        "difficulty_stratified_rule": "robustness_by_degradation_difficulty.csv groups by model, degradation_variant, degradation_level, and calibrated difficulty_level (L1-L4). It reports clean/degraded means and drops for every metric in METRIC_COLUMNS.",
        "drop_threshold": drop_threshold,
        "relative_boundary_ratio": relative_boundary_ratio,
        "isolation_note": "Degraded predictions and all report files are isolated under outputs/robustness_exp. Clean baselines may be read from outputs/main_exp/per_sample_metrics.jsonl by matching clean_sample_id, but main experiment result files are not modified.",
    }
    write_json(out_dir / "robustness_report_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FormTSR visual degradation robustness.")
    parser.add_argument("--clean-metrics", default="", help="Optional robustness-clean metric file.")
    parser.add_argument(
        "--clean-main-metrics",
        default="outputs/main_exp/per_sample_metrics.jsonl",
        help="Use existing clean main-experiment metrics as baseline. This is the default.",
    )
    parser.add_argument("--degraded-metrics", default="outputs/robustness_exp/degraded/per_sample_metrics.jsonl")
    parser.add_argument("--clean-index", default="outputs/robustness_exp/robustness_clean_index.jsonl")
    parser.add_argument("--degraded-index", default="outputs/robustness_exp/robustness_degraded_index.jsonl")
    parser.add_argument("--out", default="outputs/robustness_exp/report")
    parser.add_argument("--difficulty-csv", default=str(DEFAULT_DIFFICULTY_CSV))
    parser.add_argument("--layout-root", default="newdataset-layout")
    parser.add_argument("--models", default="", help="Optional comma-separated model names.")
    parser.add_argument("--drop-threshold", type=float, default=0.10)
    parser.add_argument("--relative-boundary-ratio", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_robustness_report(
        clean_metrics_path=Path(args.clean_metrics) if args.clean_metrics else None,
        clean_main_metrics_path=Path(args.clean_main_metrics) if args.clean_main_metrics else None,
        degraded_metrics_path=Path(args.degraded_metrics),
        clean_index_path=Path(args.clean_index),
        degraded_index_path=Path(args.degraded_index),
        out_dir=Path(args.out),
        difficulty_csv=Path(args.difficulty_csv) if args.difficulty_csv else None,
        layout_root=Path(args.layout_root) if args.layout_root else None,
        models=_split_csv(args.models),
        drop_threshold=args.drop_threshold,
        relative_boundary_ratio=args.relative_boundary_ratio,
    )
    print(f"wrote robustness report -> {args.out}")
    print(f"paired rows: {metadata['n_per_sample_pairs']}")


if __name__ == "__main__":
    main()
