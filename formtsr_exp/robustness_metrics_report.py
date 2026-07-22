from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .document_similarity_report import (
    SchemaNode,
    build_schema_tree,
    extract_normalized_values,
    schema_nted,
    value_ned,
)
from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .metrics import NA, field_accuracy, normalize_json, unwrap_answer
from .page_em_report import page_exact_match
from .results import DEFAULT_DIFFICULTY_CSV, DIFFICULTY_LEVEL_LABELS, load_difficulty_lookup


METRICS = [
    "Page-EM",
    "Schema-nTED",
    "Value-nED",
    "TSR-path",
    "R-F1@0.5",
    "R-F1@0.75",
    "LIG-F1",
]
SPATIAL_METRICS = {"R-F1@0.5", "R-F1@0.75", "LIG-F1"}
GEOMETRY_PRESERVING_VARIANTS = {"blur_noise", "erode", "occlusion_stain"}
VARIANTS = ["blur_noise", "dilate", "erode", "occlusion_stain", "perspective_skew"]
LEVELS = ["low", "medium", "high"]
LEVEL_ORDER = {level: index for index, level in enumerate(LEVELS)}
DIFFICULTY_ORDER = {level: index for index, level in enumerate(("L1", "L2", "L3", "L4"))}

EXPECTED_CLEAN_INDEX_SHA256 = "714321cd74f25e203ef8ab20bcd7b174ee4c71395bd90865d3e00ef983ec5314"
EXPECTED_DEGRADED_INDEX_SHA256 = "31d34edd790e065d800128a3d59ca50edb29b442a03391e84b3b16a78588ad59"
EXPECTED_DIFFICULTY_SHA256 = "4c446da7c43909d088cc9d5de71c4621dd388ff3b47739cb4bc5ea7318921a62"
EXPECTED_DIFFICULTY_COUNTS = {"L1": 11, "L2": 22, "L3": 24, "L4": 11}
EXPECTED_LIG_COUNTS = {"L1": 2, "L2": 6, "L3": 8, "L4": 8}

IDENTITY_COLUMNS = [
    "model",
    "model_id",
    "clean_model",
    "pairing_status",
]

COUNT_COLUMNS = [
    "n_total",
    "n_clean_valid_json",
    "n_degraded_valid_json",
    "clean_coverage",
    "degraded_coverage",
    "n_clean_exact_match",
    "n_degraded_exact_match",
    "n_structure_applicable",
    "n_lig_applicable",
]

METRIC_RESULT_COLUMNS = [
    column
    for metric in METRICS
    for column in (
        f"clean_{metric}",
        f"degraded_{metric}",
        f"{metric}_drop",
        f"{metric}_relative_drop_pct",
    )
]

RESULT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "degradation_variant",
    "degradation_level",
    "structure_metric_status",
    *COUNT_COLUMNS,
    *METRIC_RESULT_COLUMNS,
]

DIFFICULTY_RESULT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "degradation_variant",
    "degradation_level",
    "difficulty_level",
    "difficulty_name",
    "n_templates",
    "structure_metric_status",
    *COUNT_COLUMNS,
    *METRIC_RESULT_COLUMNS,
]

CLEAN_COLUMNS = [
    *IDENTITY_COLUMNS,
    "n_total",
    "n_valid_json",
    "coverage",
    "n_exact_match",
    "n_lig_applicable",
    *METRICS,
]

MODEL_SEVERITY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "degradation_level",
    "structure_metric_status",
    *COUNT_COLUMNS,
    *METRIC_RESULT_COLUMNS,
]

VARIANT_SEVERITY_COLUMNS = [
    "degradation_variant",
    "degradation_level",
    "n_models",
    "mean_clean_coverage",
    "mean_degraded_coverage",
    "coverage_drop",
    *[
        column
        for metric in METRICS
        for column in (
            f"mean_clean_{metric}",
            f"mean_degraded_{metric}",
            f"mean_{metric}_drop",
            f"{metric}_relative_drop_pct",
        )
    ],
]

OBSOLETE_REPORT_FILES = (
    "visual_degradation_results_same_backend.csv",
    "visual_degradation_cross_backend_diagnostic.csv",
    "visual_degradation_model_severity_same_backend.csv",
    "visual_degradation_variant_severity_all_models_diagnostic.csv",
)


@dataclass(frozen=True, slots=True)
class RobustRun:
    model: str
    model_id: str
    clean_model: str
    pairing_status: str
    n_valid_json: int


@dataclass(frozen=True, slots=True)
class BaseSample:
    sample_id: str
    template_name: str
    difficulty_level: str
    gt: Any
    normalized_gt: Any
    schema: SchemaNode
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DegradedSample:
    sample_id: str
    clean_sample_id: str
    template_name: str
    difficulty_level: str
    degradation_variant: str
    degradation_level: str


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip() not in {"", NA, "TBD"}:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format(value: Any, places: int = 6) -> str:
    if value is None or value == NA:
        return NA
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(
    clean_index_path: Path,
    degraded_index_path: Path,
    difficulty_csv: Path,
) -> None:
    expected = {
        clean_index_path: EXPECTED_CLEAN_INDEX_SHA256,
        degraded_index_path: EXPECTED_DEGRADED_INDEX_SHA256,
        difficulty_csv: EXPECTED_DIFFICULTY_SHA256,
    }
    for path, expected_hash in expected.items():
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"frozen robustness source changed: {path}; "
                f"expected SHA-256 {expected_hash}, got {actual_hash}"
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _raw_attempt_ids(model: str, raw_root: Path) -> set[str]:
    model_dir = raw_root / model
    if not model_dir.is_dir():
        return set()
    return {path.stem for path in model_dir.iterdir() if path.is_file()}


def select_complete_runs(
    clean_results_path: Path,
    degraded_results_path: Path,
    degraded_index_path: Path,
    degraded_raw_root: Path,
) -> tuple[list[RobustRun], list[dict[str, str]]]:
    clean_rows = _read_csv(clean_results_path)
    clean_by_model_id: dict[str, dict[str, str]] = {}
    for row in clean_rows:
        model_id = str(row.get("model_id") or "").strip()
        if not model_id:
            raise ValueError(f"missing model_id in {clean_results_path}")
        key = model_id.casefold()
        if key in clean_by_model_id:
            raise ValueError(f"duplicate clean model_id in {clean_results_path}: {model_id}")
        clean_by_model_id[key] = row

    degraded_ids = {str(row["sample_id"]) for row in read_jsonl(degraded_index_path)}
    if len(degraded_ids) != 1020:
        raise ValueError(f"expected 1020 degraded samples, got {len(degraded_ids)}")

    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    excluded: list[dict[str, str]] = []
    degraded_rows = _read_csv(degraded_results_path)
    degraded_model_ids = {
        str(row.get("model_id") or row.get("model") or "").strip().casefold()
        for row in degraded_rows
    }
    for row in degraded_rows:
        model = str(row.get("model") or "").strip()
        model_id = str(row.get("model_id") or model).strip()
        reason = ""
        if model_id.casefold() not in clean_by_model_id:
            reason = "no official clean model"
        elif int(str(row.get("n_total") or 0)) != len(degraded_ids):
            reason = "incomplete degraded metric scope"
        elif _raw_attempt_ids(model, degraded_raw_root) != degraded_ids:
            reason = "degraded raw attempts do not exactly cover the index"
        elif int(str(row.get("n_valid_json") or 0)) <= 0:
            reason = "no valid degraded predictions"
        if reason:
            excluded.append({"model": model, "model_id": model_id, "reason": reason})
            continue
        candidates[model_id.casefold()].append(row)

    for key, clean in clean_by_model_id.items():
        if key not in degraded_model_ids:
            excluded.append(
                {
                    "model": str(clean["model"]),
                    "model_id": str(clean["model_id"]),
                    "reason": "no degraded run",
                }
            )

    selected: list[RobustRun] = []
    for key, rows in candidates.items():
        degraded = max(rows, key=lambda row: int(str(row.get("n_valid_json") or 0)))
        clean = clean_by_model_id[key]
        model = str(degraded["model"])
        clean_model = str(clean["model"])
        selected.append(
            RobustRun(
                model=model,
                model_id=str(degraded["model_id"]),
                clean_model=clean_model,
                pairing_status="same_run_id"
                if model == clean_model
                else "cross_backend_model_id_pair",
                n_valid_json=int(degraded["n_valid_json"]),
            )
        )
        for row in rows:
            if row is not degraded:
                excluded.append(
                    {
                        "model": str(row["model"]),
                        "model_id": str(row["model_id"]),
                        "reason": "lower-ranked duplicate complete run",
                    }
                )
    selected.sort(key=lambda run: run.model_id.casefold())
    if len(selected) != 7:
        raise ValueError(f"expected 7 complete robustness models, got {len(selected)}")
    return selected, sorted(excluded, key=lambda row: row["model"])


def load_samples(
    clean_index_path: Path,
    degraded_index_path: Path,
    difficulty_csv: Path,
) -> tuple[
    dict[str, BaseSample],
    list[DegradedSample],
    dict[str, int],
    list[dict[str, str]],
]:
    clean_index = read_jsonl(clean_index_path)
    degraded_index = read_jsonl(degraded_index_path)
    clean_ids = [str(row["sample_id"]) for row in clean_index]
    degraded_ids = [str(row["sample_id"]) for row in degraded_index]
    if len(clean_ids) != 68 or len(set(clean_ids)) != 68:
        raise ValueError("clean robustness index must contain 68 unique samples")
    if len(degraded_ids) != 1020 or len(set(degraded_ids)) != 1020:
        raise ValueError("degraded robustness index must contain 1020 unique samples")

    difficulty_lookup = load_difficulty_lookup(
        [*clean_index, *degraded_index],
        difficulty_csv=difficulty_csv,
        layout_root=None,
    )
    clean_gt_raw = {
        str(row["sample_id"]): read_json(Path(str(row["label_path"])))
        for row in clean_index
    }
    degraded_gt_by_clean: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for row in degraded_index:
        degraded_gt_by_clean[str(row["clean_sample_id"])].append(
            (str(row["sample_id"]), read_json(Path(str(row["label_path"]))))
        )

    canonical_gt: dict[str, Any] = {}
    gt_mismatches: list[dict[str, str]] = []
    for clean_sample_id, clean_gt in clean_gt_raw.items():
        degraded_labels = degraded_gt_by_clean.get(clean_sample_id, [])
        if len(degraded_labels) != 15:
            raise ValueError(f"expected 15 degraded labels for {clean_sample_id}")
        degraded_normalized = [normalize_json(gt) for _, gt in degraded_labels]
        if any(gt != degraded_normalized[0] for gt in degraded_normalized[1:]):
            raise ValueError(f"degraded labels disagree for clean sample: {clean_sample_id}")
        if degraded_normalized[0] != normalize_json(clean_gt):
            canonical_gt[clean_sample_id] = degraded_labels[0][1]
            for degraded_sample_id, _ in degraded_labels:
                gt_mismatches.append(
                    {
                        "degraded_sample_id": degraded_sample_id,
                        "clean_sample_id": clean_sample_id,
                        "template_name": clean_sample_id.split("__", 1)[0],
                        "policy": "use the unanimous augmented GT as shared clean/degraded robustness GT",
                    }
                )
        else:
            canonical_gt[clean_sample_id] = clean_gt

    mismatch_clean_ids = {row["clean_sample_id"] for row in gt_mismatches}
    if len(gt_mismatches) != 15 or mismatch_clean_ids != {"en_13__01"}:
        raise ValueError(
            "unexpected clean/augmented GT mismatch set: "
            f"n={len(gt_mismatches)}, clean_ids={sorted(mismatch_clean_ids)}"
        )

    base_samples: dict[str, BaseSample] = {}
    difficulty_counts: Counter[str] = Counter()
    for row in clean_index:
        sample_id = str(row["sample_id"])
        template_name = str(row["template_name"])
        difficulty = difficulty_lookup.get(template_name)
        if not difficulty:
            raise ValueError(f"missing frozen difficulty mapping for {template_name}")
        level = str(difficulty["difficulty_level"])
        gt = canonical_gt[sample_id]
        base_samples[sample_id] = BaseSample(
            sample_id=sample_id,
            template_name=template_name,
            difficulty_level=level,
            gt=gt,
            normalized_gt=normalize_json(gt),
            schema=build_schema_tree(gt),
            values=extract_normalized_values(gt),
        )
        difficulty_counts[level] += 1
    if dict(difficulty_counts) != EXPECTED_DIFFICULTY_COUNTS:
        raise ValueError(
            f"robustness difficulty partition changed: "
            f"expected={EXPECTED_DIFFICULTY_COUNTS}, actual={dict(difficulty_counts)}"
        )

    degraded_samples: list[DegradedSample] = []
    condition_counts: Counter[tuple[str, str]] = Counter()
    pair_counts: Counter[str] = Counter()
    for row in degraded_index:
        sample_id = str(row["sample_id"])
        clean_sample_id = str(row.get("clean_sample_id") or "")
        base = base_samples.get(clean_sample_id)
        if base is None:
            raise ValueError(f"degraded sample has no clean pair: {sample_id}")
        variant = str(row.get("degradation_variant") or "")
        level = str(row.get("degradation_level") or "")
        if variant not in VARIANTS or level not in LEVELS:
            raise ValueError(f"unexpected degradation condition: {variant}/{level}")
        degraded_samples.append(
            DegradedSample(
                sample_id=sample_id,
                clean_sample_id=clean_sample_id,
                template_name=base.template_name,
                difficulty_level=base.difficulty_level,
                degradation_variant=variant,
                degradation_level=level,
            )
        )
        condition_counts[(variant, level)] += 1
        pair_counts[clean_sample_id] += 1
    if any(condition_counts[(variant, level)] != 68 for variant in VARIANTS for level in LEVELS):
        raise ValueError(f"degradation condition counts changed: {condition_counts}")
    if set(pair_counts.values()) != {15}:
        raise ValueError("every clean sample must have exactly 15 degraded counterparts")
    return base_samples, degraded_samples, dict(difficulty_counts), gt_mismatches


def _load_metric_map(
    path: Path,
    expected_ids: set[str],
    *,
    allow_extra: bool,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"missing sample_id in {path}")
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        rows[sample_id] = row
    actual = set(rows)
    if not expected_ids <= actual or (not allow_extra and actual != expected_ids):
        raise ValueError(
            f"sample coverage mismatch for {path}: "
            f"missing={len(expected_ids - actual)}, extra={len(actual - expected_ids)}"
        )
    return {sample_id: rows[sample_id] for sample_id in expected_ids}


def _evaluate_prediction(
    pred_path: Path,
    sample: BaseSample,
    semantic: dict[str, Any],
    structure: dict[str, Any],
) -> dict[str, Any]:
    pred: Any = None
    valid_json = False
    if pred_path.exists():
        try:
            pred = read_json(pred_path)
            valid_json = True
        except Exception:
            pred = None
    if bool(semantic.get("valid_json")) != valid_json:
        raise ValueError(f"semantic valid_json mismatch for {pred_path}")
    if bool(structure.get("valid_json")) != valid_json:
        raise ValueError(f"structure valid_json mismatch for {pred_path}")

    result: dict[str, Any] = {
        "valid_json": valid_json,
        "n_exact_match": 0,
        "Page-EM": 0.0,
        "Schema-nTED": 0.0,
        "Value-nED": 0.0,
        "TSR-path": 0.0,
        "R-F1@0.5": _numeric(structure.get("R-F1")) or 0.0,
        "R-F1@0.75": _numeric(structure.get("R-F1@0.75")) or 0.0,
        "LIG-F1": structure.get("LIG-F1", NA),
    }
    if valid_json:
        answer = unwrap_answer(pred)
        result["Page-EM"] = page_exact_match(pred, sample.normalized_gt)
        result["n_exact_match"] = int(result["Page-EM"])
        result["Schema-nTED"] = schema_nted(build_schema_tree(answer), sample.schema)
        result["Value-nED"] = value_ned(extract_normalized_values(answer), sample.values)
        tsr_path, _ = field_accuracy(pred, sample.gt)
        result["TSR-path"] = _numeric(tsr_path) or 0.0
    return result


def evaluate_run(
    run: RobustRun,
    base_samples: dict[str, BaseSample],
    degraded_samples: list[DegradedSample],
    clean_pred_root: Path,
    degraded_pred_root: Path,
    clean_semantic_dir: Path,
    degraded_semantic_dir: Path,
    clean_structure_dir: Path,
    degraded_structure_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    clean_ids = set(base_samples)
    degraded_ids = {sample.sample_id for sample in degraded_samples}
    clean_semantic = _load_metric_map(
        clean_semantic_dir / f"{run.clean_model}.jsonl", clean_ids, allow_extra=True
    )
    clean_structure = _load_metric_map(
        clean_structure_dir / f"{run.clean_model}.jsonl", clean_ids, allow_extra=True
    )
    degraded_semantic = _load_metric_map(
        degraded_semantic_dir / f"{run.model}.jsonl", degraded_ids, allow_extra=False
    )
    degraded_structure = _load_metric_map(
        degraded_structure_dir / f"{run.model}.jsonl", degraded_ids, allow_extra=False
    )

    clean_evaluations: dict[str, dict[str, Any]] = {}
    for sample_id, sample in base_samples.items():
        clean_evaluations[sample_id] = _evaluate_prediction(
            clean_pred_root / run.clean_model / f"{sample_id}.json",
            sample,
            clean_semantic[sample_id],
            clean_structure[sample_id],
        )

    pair_rows: list[dict[str, Any]] = []
    for sample in degraded_samples:
        base = base_samples[sample.clean_sample_id]
        degraded = _evaluate_prediction(
            degraded_pred_root / run.model / f"{sample.sample_id}.json",
            base,
            degraded_semantic[sample.sample_id],
            degraded_structure[sample.sample_id],
        )
        clean = clean_evaluations[sample.clean_sample_id]
        structure_valid = sample.degradation_variant in GEOMETRY_PRESERVING_VARIANTS
        row: dict[str, Any] = {
            "model": run.model,
            "model_id": run.model_id,
            "clean_model": run.clean_model,
            "pairing_status": run.pairing_status,
            "degraded_sample_id": sample.sample_id,
            "clean_sample_id": sample.clean_sample_id,
            "template_name": sample.template_name,
            "difficulty_level": sample.difficulty_level,
            "difficulty_name": DIFFICULTY_LEVEL_LABELS[sample.difficulty_level],
            "degradation_variant": sample.degradation_variant,
            "degradation_level": sample.degradation_level,
            "structure_metrics_valid": structure_valid,
            "clean_valid_json": clean["valid_json"],
            "degraded_valid_json": degraded["valid_json"],
            "clean_n_exact_match": clean["n_exact_match"],
            "degraded_n_exact_match": degraded["n_exact_match"],
        }
        for metric in METRICS:
            clean_value = clean[metric]
            degraded_value = degraded[metric]
            if metric in SPATIAL_METRICS and not structure_valid:
                clean_value = NA
                degraded_value = NA
            row[f"clean_{metric}"] = clean_value
            row[f"degraded_{metric}"] = degraded_value
            clean_number = _numeric(clean_value)
            degraded_number = _numeric(degraded_value)
            if clean_number is None or degraded_number is None:
                row[f"{metric}_drop"] = NA
                row[f"{metric}_relative_drop_pct"] = NA
            else:
                drop = clean_number - degraded_number
                row[f"{metric}_drop"] = drop
                row[f"{metric}_relative_drop_pct"] = (
                    100.0 * drop / clean_number if clean_number else NA
                )
        pair_rows.append(row)
    return pair_rows, clean_evaluations


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [_numeric(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else NA


def aggregate_pairs(
    pair_rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)

    summaries: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        first = rows[0]
        n_total = len(rows)
        summary: dict[str, Any] = {field: first[field] for field in group_fields}
        for field in IDENTITY_COLUMNS:
            summary[field] = first[field]
        if "difficulty_level" in group_fields:
            summary["difficulty_name"] = first["difficulty_name"]
            summary["n_templates"] = len({str(row["template_name"]) for row in rows})
        summary.update(
            {
                "n_total": n_total,
                "n_clean_valid_json": sum(bool(row["clean_valid_json"]) for row in rows),
                "n_degraded_valid_json": sum(bool(row["degraded_valid_json"]) for row in rows),
                "clean_coverage": sum(bool(row["clean_valid_json"]) for row in rows) / n_total,
                "degraded_coverage": sum(bool(row["degraded_valid_json"]) for row in rows) / n_total,
                "n_clean_exact_match": sum(int(row["clean_n_exact_match"]) for row in rows),
                "n_degraded_exact_match": sum(int(row["degraded_n_exact_match"]) for row in rows),
            }
        )
        n_structure_applicable = sum(bool(row["structure_metrics_valid"]) for row in rows)
        summary["n_structure_applicable"] = n_structure_applicable
        if n_structure_applicable == n_total:
            summary["structure_metric_status"] = "valid_geometry_preserving"
        elif n_structure_applicable == 0:
            summary["structure_metric_status"] = "NA_geometry_changed"
        else:
            summary["structure_metric_status"] = "partial_geometry_preserving_only"
        clean_lig_count = sum(_numeric(row.get("clean_LIG-F1")) is not None for row in rows)
        degraded_lig_count = sum(_numeric(row.get("degraded_LIG-F1")) is not None for row in rows)
        if clean_lig_count != degraded_lig_count:
            raise ValueError("clean/degraded LIG applicability mismatch")
        summary["n_lig_applicable"] = clean_lig_count

        for metric in METRICS:
            clean_value = _mean_numeric(rows, f"clean_{metric}")
            degraded_value = _mean_numeric(rows, f"degraded_{metric}")
            if metric != "LIG-F1":
                expected_numeric = n_structure_applicable if metric in SPATIAL_METRICS else n_total
                if sum(_numeric(row.get(f"clean_{metric}")) is not None for row in rows) != expected_numeric:
                    raise ValueError(f"non-numeric clean full-scope metric: {metric}")
                if sum(_numeric(row.get(f"degraded_{metric}")) is not None for row in rows) != expected_numeric:
                    raise ValueError(f"non-numeric degraded full-scope metric: {metric}")
            summary[f"clean_{metric}"] = clean_value
            summary[f"degraded_{metric}"] = degraded_value
            clean_number = _numeric(clean_value)
            degraded_number = _numeric(degraded_value)
            if clean_number is None or degraded_number is None:
                summary[f"{metric}_drop"] = NA
                summary[f"{metric}_relative_drop_pct"] = NA
            else:
                drop = clean_number - degraded_number
                summary[f"{metric}_drop"] = drop
                summary[f"{metric}_relative_drop_pct"] = (
                    100.0 * drop / clean_number if clean_number else NA
                )
        summaries.append(summary)
    return summaries


def summarize_clean(
    runs: list[RobustRun],
    clean_by_model: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in runs:
        evaluations = clean_by_model[run.model]
        rows = list(evaluations.values())
        n_total = len(rows)
        lig_count = sum(_numeric(row.get("LIG-F1")) is not None for row in rows)
        summary: dict[str, Any] = {
            "model": run.model,
            "model_id": run.model_id,
            "clean_model": run.clean_model,
            "pairing_status": run.pairing_status,
            "n_total": n_total,
            "n_valid_json": sum(bool(row["valid_json"]) for row in rows),
            "coverage": sum(bool(row["valid_json"]) for row in rows) / n_total,
            "n_exact_match": sum(int(row["n_exact_match"]) for row in rows),
            "n_lig_applicable": lig_count,
        }
        for metric in METRICS:
            summary[metric] = _mean_numeric(rows, metric)
        summaries.append(summary)
    return summaries


def summarize_variant_macro(
    condition_rows: list[dict[str, Any]],
    *,
    expected_models: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        grouped[(str(row["degradation_variant"]), str(row["degradation_level"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, level), rows in grouped.items():
        if len(rows) != expected_models:
            raise ValueError(
                f"variant macro requires {expected_models} models: {variant}/{level}"
            )
        clean_coverage = sum(float(row["clean_coverage"]) for row in rows) / len(rows)
        degraded_coverage = sum(float(row["degraded_coverage"]) for row in rows) / len(rows)
        summary: dict[str, Any] = {
            "degradation_variant": variant,
            "degradation_level": level,
            "n_models": len(rows),
            "mean_clean_coverage": clean_coverage,
            "mean_degraded_coverage": degraded_coverage,
            "coverage_drop": clean_coverage - degraded_coverage,
        }
        for metric in METRICS:
            clean_values = [_numeric(row.get(f"clean_{metric}")) for row in rows]
            degraded_values = [_numeric(row.get(f"degraded_{metric}")) for row in rows]
            clean_numeric = [value for value in clean_values if value is not None]
            degraded_numeric = [value for value in degraded_values if value is not None]
            clean_value: float | str = (
                sum(clean_numeric) / len(clean_numeric) if clean_numeric else NA
            )
            degraded_value: float | str = (
                sum(degraded_numeric) / len(degraded_numeric) if degraded_numeric else NA
            )
            clean_number = _numeric(clean_value)
            degraded_number = _numeric(degraded_value)
            drop: float | str = (
                clean_number - degraded_number
                if clean_number is not None and degraded_number is not None
                else NA
            )
            summary[f"mean_clean_{metric}"] = clean_value
            summary[f"mean_degraded_{metric}"] = degraded_value
            summary[f"mean_{metric}_drop"] = drop
            summary[f"{metric}_relative_drop_pct"] = (
                100.0 * float(drop) / clean_number
                if isinstance(drop, (int, float)) and clean_number
                else NA
            )
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (VARIANTS.index(str(row["degradation_variant"])), LEVEL_ORDER[str(row["degradation_level"])]),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in columns})


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_results(
    out_dir: Path,
    runs: list[RobustRun],
    pair_rows: list[dict[str, Any]],
    gt_mismatches: list[dict[str, str]],
    clean_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
    model_severity_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    for filename in OBSOLETE_REPORT_FILES:
        (out_dir / filename).unlink(missing_ok=True)
    _write_csv(out_dir / "visual_degradation_clean_baseline.csv", clean_rows, CLEAN_COLUMNS)
    _write_csv(out_dir / "visual_degradation_results.csv", condition_rows, RESULT_COLUMNS)
    _write_csv(
        out_dir / "visual_degradation_by_difficulty.csv",
        difficulty_rows,
        DIFFICULTY_RESULT_COLUMNS,
    )
    _write_csv(
        out_dir / "visual_degradation_model_severity.csv",
        model_severity_rows,
        MODEL_SEVERITY_COLUMNS,
    )
    _write_csv(
        out_dir / "visual_degradation_variant_severity.csv",
        variant_rows,
        VARIANT_SEVERITY_COLUMNS,
    )
    write_jsonl(out_dir / "visual_degradation_per_sample.jsonl", pair_rows)
    _write_csv(
        out_dir / "visual_degradation_gt_mismatches.csv",
        gt_mismatches,
        ["degraded_sample_id", "clean_sample_id", "template_name", "policy"],
    )
    write_json(out_dir / "visual_degradation_results_metadata.json", metadata)

    markdown = [
        "# Visual degradation results",
        "",
        "The report includes all seven fully attempted degraded runs in one unified comparison. Backend differences are treated as negligible for aggregation. Positive drops mean lower performance after degradation. Spatial metrics are NA for dilate and perspective_skew because those transforms change geometry without transformed bbox GT.",
        "",
        "## Selected runs",
        "",
        "| Model | Degraded run | Clean run | Pairing | Degraded valid/1020 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for run in runs:
        markdown.append(
            f"| {run.model_id} | {run.model} | {run.clean_model} | "
            f"{run.pairing_status} | {run.n_valid_json}/1020 |"
        )
    markdown.extend(
        [
            "",
            "## Five-variant mean by severity",
            "",
            "All seven completed model pairs are included. Semantic drops average all five variants; spatial drops average only the three geometry-preserving variants.",
            "",
            "| Model | Level | Clean valid | Degraded valid | Schema drop | Value drop | TSR drop | R-F1@0.5 drop | R-F1@0.75 drop | LIG-F1 drop |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in model_severity_rows:
        markdown.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["degradation_level"]),
                    f"{row['n_clean_valid_json']}/{row['n_total']}",
                    f"{row['n_degraded_valid_json']}/{row['n_total']}",
                    _format(row["Schema-nTED_drop"]),
                    _format(row["Value-nED_drop"]),
                    _format(row["TSR-path_drop"]),
                    _format(row["R-F1@0.5_drop"]),
                    _format(row["R-F1@0.75_drop"]),
                    _format(row["LIG-F1_drop"]),
                ]
            )
            + " |"
        )
    (out_dir / "visual_degradation_results.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Level & Schema drop & Value drop & TSR drop & R-F1@0.5 drop & R-F1@0.75 drop & LIG-F1 drop \\",
        r"\midrule",
    ]
    for row in model_severity_rows:
        latex.append(
            " & ".join(
                [
                    _latex_escape(str(row["model_id"])),
                    str(row["degradation_level"]),
                    _format(row["Schema-nTED_drop"], 4),
                    _format(row["Value-nED_drop"], 4),
                    _format(row["TSR-path_drop"], 4),
                    _format(row["R-F1@0.5_drop"], 4),
                    _format(row["R-F1@0.75_drop"], 4),
                    _format(row["LIG-F1_drop"], 4),
                ]
            )
            + r" \\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "visual_degradation_results_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean visual-degradation slices with the latest formal metrics."
    )
    parser.add_argument("--clean-results", default="outputs/main_exp/main_experiment_results.csv")
    parser.add_argument(
        "--degraded-results", default="outputs/robustness_exp/degraded/main_results.csv"
    )
    parser.add_argument(
        "--clean-index", default="outputs/robustness_exp/robustness_clean_index.jsonl"
    )
    parser.add_argument(
        "--degraded-index", default="outputs/robustness_exp/robustness_degraded_index.jsonl"
    )
    parser.add_argument("--clean-pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--degraded-pred-root", default="outputs/robustness_exp/degraded/pred")
    parser.add_argument("--degraded-raw-root", default="outputs/robustness_exp/degraded/raw")
    parser.add_argument("--clean-semantic-dir", default="outputs/main_exp/per_model_metrics")
    parser.add_argument(
        "--degraded-semantic-dir",
        default="outputs/robustness_exp/degraded/per_model_metrics",
    )
    parser.add_argument(
        "--clean-structure-dir", default="outputs/main_exp/corrected_structure_per_sample"
    )
    parser.add_argument(
        "--degraded-structure-dir",
        default="outputs/robustness_exp/latest_metrics/degraded/corrected_structure_per_sample",
    )
    parser.add_argument("--difficulty-csv", default=str(DEFAULT_DIFFICULTY_CSV))
    parser.add_argument("--out", default="outputs/robustness_exp/report_latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_results_path = Path(args.clean_results)
    degraded_results_path = Path(args.degraded_results)
    clean_index_path = Path(args.clean_index)
    degraded_index_path = Path(args.degraded_index)
    clean_pred_root = Path(args.clean_pred_root)
    degraded_pred_root = Path(args.degraded_pred_root)
    degraded_raw_root = Path(args.degraded_raw_root)
    clean_semantic_dir = Path(args.clean_semantic_dir)
    degraded_semantic_dir = Path(args.degraded_semantic_dir)
    clean_structure_dir = Path(args.clean_structure_dir)
    degraded_structure_dir = Path(args.degraded_structure_dir)
    difficulty_csv = Path(args.difficulty_csv)
    out_dir = Path(args.out)

    validate_frozen_sources(clean_index_path, degraded_index_path, difficulty_csv)
    runs, excluded = select_complete_runs(
        clean_results_path,
        degraded_results_path,
        degraded_index_path,
        degraded_raw_root,
    )
    base_samples, degraded_samples, difficulty_counts, gt_mismatches = load_samples(
        clean_index_path,
        degraded_index_path,
        difficulty_csv,
    )

    all_pairs: list[dict[str, Any]] = []
    clean_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        pairs, clean_evaluations = evaluate_run(
            run,
            base_samples,
            degraded_samples,
            clean_pred_root,
            degraded_pred_root,
            clean_semantic_dir,
            degraded_semantic_dir,
            clean_structure_dir,
            degraded_structure_dir,
        )
        all_pairs.extend(pairs)
        clean_by_model[run.model] = clean_evaluations
        n_degraded_valid = sum(bool(row["degraded_valid_json"]) for row in pairs)
        if n_degraded_valid != run.n_valid_json:
            raise ValueError(
                f"degraded valid count mismatch for {run.model}: "
                f"report={n_degraded_valid}, main_results={run.n_valid_json}"
            )
        print(f"[Visual degradation] {run.model}: {len(pairs)} pairs", flush=True)

    if len(all_pairs) != 7140:
        raise ValueError(f"expected 7140 clean/degraded pairs, got {len(all_pairs)}")
    clean_rows = summarize_clean(runs, clean_by_model)
    condition_rows = aggregate_pairs(
        all_pairs,
        ("model", "degradation_variant", "degradation_level"),
    )
    difficulty_rows = aggregate_pairs(
        all_pairs,
        ("model", "degradation_variant", "degradation_level", "difficulty_level"),
    )
    model_severity_rows = aggregate_pairs(all_pairs, ("model", "degradation_level"))
    condition_rows.sort(
        key=lambda row: (
            str(row["model_id"]).casefold(),
            VARIANTS.index(str(row["degradation_variant"])),
            LEVEL_ORDER[str(row["degradation_level"])],
        )
    )
    difficulty_rows.sort(
        key=lambda row: (
            str(row["model_id"]).casefold(),
            VARIANTS.index(str(row["degradation_variant"])),
            LEVEL_ORDER[str(row["degradation_level"])],
            DIFFICULTY_ORDER[str(row["difficulty_level"])],
        )
    )
    model_severity_rows.sort(
        key=lambda row: (
            str(row["model_id"]).casefold(),
            LEVEL_ORDER[str(row["degradation_level"])],
        )
    )
    variant_rows = summarize_variant_macro(
        condition_rows,
        expected_models=7,
    )

    for row in difficulty_rows:
        geometry_preserving = str(row["degradation_variant"]) in GEOMETRY_PRESERVING_VARIANTS
        expected_structure = int(row["n_total"]) if geometry_preserving else 0
        if int(row["n_structure_applicable"]) != expected_structure:
            raise ValueError(f"difficulty-specific structure denominator changed: {row}")
        expected_lig = (
            EXPECTED_LIG_COUNTS[str(row["difficulty_level"])] if geometry_preserving else 0
        )
        if int(row["n_lig_applicable"]) != expected_lig:
            raise ValueError(f"difficulty-specific LIG denominator changed: {row}")
    for row in condition_rows:
        geometry_preserving = str(row["degradation_variant"]) in GEOMETRY_PRESERVING_VARIANTS
        expected_structure = 68 if geometry_preserving else 0
        expected_lig = 24 if geometry_preserving else 0
        if int(row["n_structure_applicable"]) != expected_structure:
            raise ValueError(f"condition structure denominator changed: {row}")
        if int(row["n_lig_applicable"]) != expected_lig:
            raise ValueError(f"condition LIG denominator changed: {row}")

    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "clean_results": str(clean_results_path),
            "degraded_results": str(degraded_results_path),
            "clean_index": str(clean_index_path),
            "clean_index_sha256": _sha256(clean_index_path),
            "degraded_index": str(degraded_index_path),
            "degraded_index_sha256": _sha256(degraded_index_path),
            "difficulty_csv": str(difficulty_csv),
            "difficulty_csv_sha256": _sha256(difficulty_csv),
            "clean_pred_root": str(clean_pred_root),
            "degraded_pred_root": str(degraded_pred_root),
            "clean_structure_dir": str(clean_structure_dir),
            "degraded_structure_dir": str(degraded_structure_dir),
        },
        "selection": "one fully attempted 1020-sample degraded raw run per official main-table model_id; incomplete and absent runs are excluded",
        "selected_runs": [
            {
                "model": run.model,
                "model_id": run.model_id,
                "clean_model": run.clean_model,
                "pairing_status": run.pairing_status,
                "n_degraded_valid_json": run.n_valid_json,
            }
            for run in runs
        ],
        "excluded_runs": excluded,
        "n_models": len(runs),
        "n_clean_samples": len(base_samples),
        "n_degraded_samples_per_model": len(degraded_samples),
        "n_pairs": len(all_pairs),
        "n_augmented_gt_mismatches": len(gt_mismatches),
        "n_condition_rows": len(condition_rows),
        "n_difficulty_rows": len(difficulty_rows),
        "n_model_severity_rows": len(model_severity_rows),
        "n_variant_severity_rows": len(variant_rows),
        "variants": VARIANTS,
        "levels": LEVELS,
        "difficulty_counts": difficulty_counts,
        "metric_policy": {
            "full_scope": "Page-EM, Schema-nTED, Value-nED, and TSR-path include all 68 pages in each condition; missing/invalid predictions are zero. Spatial metrics use all 68 pages only when the degradation preserves geometry.",
            "LIG-F1": "page-macro over the 24 GT-applicable pages per geometry-preserving condition; missing/invalid applicable pages are zero",
            "spatial_scope": "R-F1@0.5, R-F1@0.75, and LIG-F1 are reported only for blur_noise, erode, and occlusion_stain. Dilate contains a local warp and perspective_skew contains global affine/perspective transforms, but transformed bbox GT is unavailable, so their spatial metrics are NA.",
            "drop": "clean score minus degraded score; positive means degradation loss",
            "variant_macro": "visual_degradation_variant_severity.csv is the unweighted mean of all seven completed model pairs; backend differences are treated as negligible for aggregation",
            "Page-EM": "report exact counts because the score is sparse",
        },
        "aggregation_policy": "all seven completed clean/degraded model-id pairs are included in formal tables and macros; backend differences are assumed negligible",
        "shared_gt_policy": "Both clean and degraded predictions are scored against one shared GT per clean sample. For en_13__01, all 15 augmented labels unanimously contain two image-visible fields missing from the clean label, so that unanimous augmented label overrides the clean label for both sides. The 15 affected pairs are recorded in visual_degradation_gt_mismatches.csv. TSR-path is recomputed inside this reporter against the shared GT.",
        "legacy_note": "Legacy CDS/VAcc/WAcc and failure-boundary tables are not part of this latest report. A CDS threshold cannot be carried forward after the composite was retired.",
    }
    write_results(
        out_dir,
        runs,
        all_pairs,
        gt_mismatches,
        clean_rows,
        condition_rows,
        difficulty_rows,
        model_severity_rows,
        variant_rows,
        metadata,
    )
    print(f"wrote latest visual degradation report -> {out_dir}")


if __name__ == "__main__":
    main()
