from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .metrics import NA, extract_line_item_groups, normalize_metadata_layout
from .page_em_report import (
    RunSpec,
    classify_comparison_status,
    classify_run_type,
    load_run_specs,
)


SOURCE_SPACES = {"pixel", "normalized_1000", "normalized_1", "none"}

RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "run_type",
    "comparison_status",
    "bbox_source_space",
    "n_total",
    "n_valid_json",
    "coverage",
    "n_missing_prediction",
    "n_invalid_json",
    "n_pages_with_regions",
    "R-Precision@0.5",
    "R-Recall@0.5",
    "R-micro-F1@0.5",
    "R-F1",
    "R-F1@0.75",
    "R-F1-valid",
    "n_region_pred",
    "n_region_gt",
    "n_lig_applicable",
    "LIG-F1",
    "LIG-F1@0.75",
    "LIG-F1-valid",
    "region_bbox_declared",
    "region_bbox_valid",
    "region_bbox_clipped",
    "region_bbox_dropped",
    "region_unknown_type",
    "lig_bbox_declared",
    "lig_bbox_valid",
    "lig_bbox_clipped",
    "lig_bbox_dropped",
]

PER_SAMPLE_COLUMNS = [
    "model",
    "sample_id",
    "valid_json",
    "R-F1",
    "R-F1@0.75",
    "R-TP@0.5",
    "R-TP@0.75",
    "R-pred",
    "R-gt",
    "LIG-F1",
    "LIG-F1@0.75",
    "LIG-TP@0.5",
    "LIG-TP@0.75",
    "LIG-pred",
    "LIG-gt",
]

# GT metadata distinguishes data types such as date and number, while the model
# schema asks for semantic region roles. Both sides are mapped to one shared
# ontology before matching.
GT_TYPE_MAP = {
    "field_group": "group",
    "section": "group",
    "region": "group",
    "field": "label",
    "text": "value",
    "number": "value",
    "date": "value",
    "value": "value",
    "checkbox": "widget",
    "checkbox_multi": "widget",
    "check_box": "widget",
    "radio": "widget",
    "radio_button": "widget",
    "widget": "widget",
    "table": "table",
}

PRED_TYPE_MAP = {
    "title": "group",
    "section": "group",
    "section_header": "group",
    "field_group": "group",
    "region": "group",
    "field": "label",
    "label": "label",
    "field_label": "label",
    "value": "value",
    "field_value": "value",
    "number": "value",
    "date": "value",
    "text": "text",
    "checkbox": "widget",
    "checkbox_multi": "widget",
    "check_box": "widget",
    "radio": "widget",
    "radio_button": "widget",
    "widget": "widget",
    "table": "table",
    "other": "other",
}

PRED_COMPATIBILITY = {
    "group": {"group"},
    "label": {"label"},
    "value": {"value"},
    # The prompt's generic text type can describe either a visible label or a
    # filled text value. Geometry still has to meet the configured IoU.
    "text": {"label", "value"},
    "widget": {"widget"},
    "table": {"table", "group"},
    "other": set(),
    "unknown": set(),
}

LIG_REGION_TYPES = {
    "line_item_group",
    "line-item-group",
    "lineitemgroup",
    "line_item",
}


@dataclass(frozen=True, slots=True)
class CanonicalRegion:
    category: str
    bbox: tuple[float, float, float, float]
    raw_type: str = ""


@dataclass(frozen=True, slots=True)
class StructureSample:
    sample_id: str
    template_name: str
    image_width: float
    image_height: float
    regions: tuple[CanonicalRegion, ...]
    line_item_groups: tuple[tuple[float, float, float, float], ...]


@dataclass(slots=True)
class BBoxAudit:
    declared: int = 0
    valid: int = 0
    clipped: int = 0
    dropped: int = 0
    unknown_type: int = 0

    def update(self, other: "BBoxAudit") -> None:
        self.declared += other.declared
        self.valid += other.valid
        self.clipped += other.clipped
        self.dropped += other.dropped
        self.unknown_type += other.unknown_type


def _raw_region_type(item: dict[str, Any]) -> str:
    raw = item.get("type") or item.get("region_type") or item.get("data_type") or "unknown"
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def _strict_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        for key in ("bbox", "box", "region_box", "bounds"):
            if key in value:
                return _strict_bbox(value[key])
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in coords):
        return None
    return coords  # type: ignore[return-value]


def normalize_bbox(
    value: Any,
    source_space: str,
    image_width: float,
    image_height: float,
) -> tuple[tuple[float, float, float, float] | None, str]:
    """Convert one box to [0, 1] without repairing reversed endpoints."""
    raw = _strict_bbox(value)
    if raw is None:
        return None, "malformed"
    if source_space == "pixel":
        if image_width <= 0 or image_height <= 0:
            return None, "invalid_image_size"
        transformed = (
            raw[0] / image_width,
            raw[1] / image_height,
            raw[2] / image_width,
            raw[3] / image_height,
        )
    elif source_space == "normalized_1000":
        transformed = tuple(item / 1000.0 for item in raw)
    elif source_space == "normalized_1":
        transformed = raw
    elif source_space == "none":
        return None, "unsupported_space"
    else:
        raise ValueError(f"unsupported bbox source space: {source_space!r}")

    if transformed[2] <= transformed[0] or transformed[3] <= transformed[1]:
        return None, "reversed_or_degenerate"
    clipped = tuple(min(1.0, max(0.0, item)) for item in transformed)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, "outside_page"
    return clipped, "clipped" if clipped != transformed else "ok"  # type: ignore[return-value]


def load_bbox_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), dict):
        raise ValueError(f"invalid bbox manifest: {path}")
    if manifest.get("canonical_space") != "normalized_0_1":
        raise ValueError("bbox manifest canonical_space must be normalized_0_1")
    return manifest


def resolve_bbox_space(manifest: dict[str, Any], model: str) -> tuple[str, str]:
    runs = manifest["runs"]
    current = model
    seen: set[str] = set()
    while True:
        if current in seen:
            raise ValueError(f"bbox manifest inheritance cycle at {current!r}")
        seen.add(current)
        entry = runs.get(current)
        if not isinstance(entry, dict):
            raise ValueError(f"bbox source space missing for run {model!r}")
        parent = entry.get("inherits")
        if parent:
            current = str(parent)
            continue
        source_space = str(entry.get("source_space") or "")
        if source_space not in SOURCE_SPACES:
            raise ValueError(f"invalid bbox source space for {current!r}: {source_space!r}")
        return source_space, current


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _deduplicate_gt_regions(regions: list[CanonicalRegion]) -> tuple[CanonicalRegion, ...]:
    # layout_structure commonly emits a section and a near-identical generic
    # region for the same area. Keep the typed section and drop that duplicate,
    # while retaining genuine subdivisions.
    sections = [item for item in regions if item.raw_type == "section"]
    output: list[CanonicalRegion] = []
    for item in regions:
        if item.raw_type == "region" and any(_iou(item.bbox, section.bbox) >= 0.9 for section in sections):
            continue
        output.append(item)
    return tuple(output)


def _image_size(layout: dict[str, Any], template_name: str) -> tuple[float, float]:
    width = layout.get("original_width")
    height = layout.get("original_height")
    try:
        parsed = float(width), float(height)
    except (TypeError, ValueError):
        page_bbox = (
            ((layout.get("metadata") or {}).get("layout_structure") or {}).get("page_bbox")
            if isinstance(layout.get("metadata"), dict)
            else None
        )
        box = _strict_bbox(page_bbox)
        if box is None:
            raise ValueError(f"missing image dimensions for template {template_name!r}")
        parsed = box[2] - box[0], box[3] - box[1]
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError(f"invalid image dimensions for template {template_name!r}: {parsed}")
    return parsed


def _load_template_structure(layout_path: Path, template_name: str) -> tuple[float, float, tuple[CanonicalRegion, ...], tuple[tuple[float, float, float, float], ...]]:
    raw_layout = read_json(layout_path)
    if not isinstance(raw_layout, dict):
        raise ValueError(f"layout must be an object: {layout_path}")
    width, height = _image_size(raw_layout, template_name)
    layout = normalize_metadata_layout(raw_layout)

    regions: list[CanonicalRegion] = []
    for item in layout.get("regions", []) if isinstance(layout, dict) else []:
        if not isinstance(item, dict):
            continue
        raw_type = _raw_region_type(item)
        category = GT_TYPE_MAP.get(raw_type)
        if category is None:
            raise ValueError(f"unmapped GT region type {raw_type!r} in {layout_path}")
        box, status = normalize_bbox(item, "pixel", width, height)
        if box is None or status != "ok":
            raise ValueError(f"invalid GT bbox in {layout_path}: {item!r} ({status})")
        regions.append(CanonicalRegion(category, box, raw_type))

    line_item_groups: list[tuple[float, float, float, float]] = []
    seen_groups: set[tuple[float, float, float, float]] = set()
    for item in extract_line_item_groups(layout):
        box, status = normalize_bbox(item, "pixel", width, height)
        if box is None or status != "ok":
            raise ValueError(f"invalid GT line-item-group bbox in {layout_path}: {item!r} ({status})")
        key = tuple(round(value, 8) for value in box)
        if key not in seen_groups:
            seen_groups.add(key)
            line_item_groups.append(box)
    return width, height, _deduplicate_gt_regions(regions), tuple(line_item_groups)


def load_structure_samples(index_path: Path, layout_root: Path) -> list[StructureSample]:
    samples: list[StructureSample] = []
    templates: dict[str, tuple[float, float, tuple[CanonicalRegion, ...], tuple[tuple[float, float, float, float], ...]]] = {}
    seen: set[str] = set()
    for row in read_jsonl(index_path):
        sample_id = str(row["sample_id"])
        template_name = str(row["template_name"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in index: {sample_id}")
        seen.add(sample_id)
        if template_name not in templates:
            path = layout_root / f"{template_name}.json"
            if not path.exists():
                raise ValueError(f"missing layout metadata: {path}")
            templates[template_name] = _load_template_structure(path, template_name)
        width, height, regions, groups = templates[template_name]
        samples.append(StructureSample(sample_id, template_name, width, height, regions, groups))
    return samples


def _prediction_region_items(prediction: Any) -> list[Any]:
    if not isinstance(prediction, dict):
        return []
    regions = prediction.get("regions")
    if isinstance(regions, list) and regions:
        return regions
    fallback = prediction.get("region_boxes")
    return fallback if isinstance(fallback, list) else regions if isinstance(regions, list) else []


def normalize_prediction_regions(
    prediction: Any,
    source_space: str,
    image_width: float,
    image_height: float,
) -> tuple[list[CanonicalRegion], int, BBoxAudit]:
    items = _prediction_region_items(prediction)
    audit = BBoxAudit(declared=len(items))
    regions: list[CanonicalRegion] = []
    for item in items:
        if not isinstance(item, dict):
            audit.dropped += 1
            continue
        raw_type = _raw_region_type(item)
        category = PRED_TYPE_MAP.get(raw_type, "unknown")
        if category == "unknown":
            audit.unknown_type += 1
        box, status = normalize_bbox(item, source_space, image_width, image_height)
        if box is None:
            audit.dropped += 1
            continue
        audit.valid += 1
        audit.clipped += status == "clipped"
        regions.append(CanonicalRegion(category, box, raw_type))
    return regions, len(items), audit


def _normalized_declared_box(
    item: Any,
    source_space: str,
    width: float,
    height: float,
    audit: BBoxAudit,
) -> tuple[float, float, float, float] | None:
    audit.declared += 1
    box, status = normalize_bbox(item, source_space, width, height)
    if box is None:
        audit.dropped += 1
        return None
    audit.valid += 1
    audit.clipped += status == "clipped"
    return box


def normalize_prediction_line_item_groups(
    prediction: Any,
    source_space: str,
    image_width: float,
    image_height: float,
) -> tuple[list[tuple[float, float, float, float]], BBoxAudit]:
    if not isinstance(prediction, dict):
        return [], BBoxAudit()
    audit = BBoxAudit()
    groups: list[tuple[float, float, float, float]] = []

    for item in prediction.get("line_item_groups", []) if isinstance(prediction.get("line_item_groups"), list) else []:
        box = _normalized_declared_box(item, source_space, image_width, image_height, audit)
        if box is not None:
            groups.append(box)

    for item in _prediction_region_items(prediction):
        if not isinstance(item, dict) or _raw_region_type(item) not in LIG_REGION_TYPES:
            continue
        box = _normalized_declared_box(item, source_space, image_width, image_height, audit)
        if box is not None:
            groups.append(box)

    grids: list[Any] = []
    for key in ("local_grids", "grids"):
        value = prediction.get(key)
        if isinstance(value, list):
            grids.extend(value)
    for grid in grids:
        if not isinstance(grid, dict):
            audit.declared += 1
            audit.dropped += 1
            continue
        if _strict_bbox(grid) is not None:
            box = _normalized_declared_box(grid, source_space, image_width, image_height, audit)
            if box is not None:
                groups.append(box)
            continue
        cell_boxes: list[tuple[float, float, float, float]] = []
        for cell in grid.get("cells", []) if isinstance(grid.get("cells"), list) else []:
            box = _normalized_declared_box(cell, source_space, image_width, image_height, audit)
            if box is not None:
                cell_boxes.append(box)
        if cell_boxes:
            groups.append(
                (
                    min(box[0] for box in cell_boxes),
                    min(box[1] for box in cell_boxes),
                    max(box[2] for box in cell_boxes),
                    max(box[3] for box in cell_boxes),
                )
            )

    unique: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for box in groups:
        key = tuple(round(value, 8) for value in box)
        if key not in seen:
            seen.add(key)
            unique.append(box)
    return unique, audit


def _iou_matrix(
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    pred = np.asarray(pred_boxes, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    ix1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    iy1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    ix2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    iy2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    intersection = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _maximum_matches(eligible: np.ndarray) -> int:
    if eligible.size == 0 or not eligible.any():
        return 0
    pred_indices, gt_indices = linear_sum_assignment(eligible, maximize=True)
    return int(eligible[pred_indices, gt_indices].sum())


def region_match_counts(
    pred_regions: list[CanonicalRegion],
    n_pred_declared: int,
    gt_regions: tuple[CanonicalRegion, ...],
    threshold: float,
) -> tuple[int, int, int]:
    if not pred_regions or not gt_regions:
        return 0, n_pred_declared, len(gt_regions)
    ious = _iou_matrix([item.bbox for item in pred_regions], [item.bbox for item in gt_regions])
    compatible = np.asarray(
        [
            [gt.category in PRED_COMPATIBILITY.get(pred.category, set()) for gt in gt_regions]
            for pred in pred_regions
        ],
        dtype=bool,
    )
    return _maximum_matches((ious >= threshold) & compatible), n_pred_declared, len(gt_regions)


def box_match_counts(
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: tuple[tuple[float, float, float, float], ...],
    threshold: float,
) -> tuple[int, int, int]:
    if not pred_boxes or not gt_boxes:
        return 0, len(pred_boxes), len(gt_boxes)
    return _maximum_matches(_iou_matrix(pred_boxes, list(gt_boxes)) >= threshold), len(pred_boxes), len(gt_boxes)


def _f1(tp: int, n_pred: int, n_gt: int) -> float:
    return 2.0 * tp / (n_pred + n_gt) if n_pred + n_gt else 1.0


def _failed_row(spec: RunSpec, source_space: str, n_indexed: int) -> dict[str, Any]:
    run_type = classify_run_type(spec.model)
    full_scope = spec.n_total == n_indexed
    return {
        "model": spec.model,
        "model_id": spec.model_id,
        "group": spec.group,
        "run_type": run_type,
        "comparison_status": classify_comparison_status(
            run_type=run_type,
            has_prediction_dir=False,
            full_scope=full_scope,
            n_valid_json=0,
            n_indexed=n_indexed,
        ),
        "bbox_source_space": source_space,
        "n_total": spec.n_total,
        "n_valid_json": 0,
        "coverage": 0.0 if spec.n_total else NA,
        "n_missing_prediction": spec.n_total,
        "n_invalid_json": 0,
        "n_pages_with_regions": 0,
        "R-Precision@0.5": 0.0 if spec.n_total else NA,
        "R-Recall@0.5": 0.0 if spec.n_total else NA,
        "R-micro-F1@0.5": 0.0 if spec.n_total else NA,
        "R-F1": 0.0 if spec.n_total else NA,
        "R-F1@0.75": 0.0 if spec.n_total else NA,
        "R-F1-valid": NA,
        "n_region_pred": 0,
        "n_region_gt": NA,
        "n_lig_applicable": NA,
        "LIG-F1": NA,
        "LIG-F1@0.75": NA,
        "LIG-F1-valid": NA,
        **{column: 0 for column in RESULT_COLUMNS if column.startswith("region_bbox_") or column.startswith("lig_bbox_")},
        "region_unknown_type": 0,
    }


def evaluate_run(
    spec: RunSpec,
    samples: list[StructureSample],
    pred_root: Path,
    manifest: dict[str, Any],
    per_sample_dir: Path | None = None,
) -> dict[str, Any]:
    source_space, inherited_from = resolve_bbox_space(manifest, spec.model)
    model_dir = pred_root / spec.model
    if not model_dir.is_dir():
        return _failed_row(spec, source_space, len(samples))
    if spec.n_total != len(samples):
        raise ValueError(
            f"run {spec.model!r} has n_total={spec.n_total}, but structure evaluation "
            f"requires the full {len(samples)}-sample index"
        )

    n_valid_json = 0
    n_invalid_json = 0
    n_missing_prediction = 0
    n_pages_with_regions = 0
    region_macro_05 = 0.0
    region_macro_075 = 0.0
    region_valid_macro_05 = 0.0
    region_tp_05 = 0
    region_pred_total = 0
    region_gt_total = 0
    lig_macro_05 = 0.0
    lig_macro_075 = 0.0
    lig_valid_macro_05 = 0.0
    n_lig_applicable = 0
    n_lig_valid = 0
    region_audit = BBoxAudit()
    lig_audit = BBoxAudit()
    per_sample_rows: list[dict[str, Any]] = []

    for sample in samples:
        pred_path = model_dir / f"{sample.sample_id}.json"
        prediction: Any = None
        valid_json = False
        if not pred_path.exists():
            n_missing_prediction += 1
        else:
            try:
                prediction = read_json(pred_path)
                valid_json = True
                n_valid_json += 1
            except Exception:
                n_invalid_json += 1

        pred_regions: list[CanonicalRegion] = []
        n_pred_declared = 0
        pred_groups: list[tuple[float, float, float, float]] = []
        if valid_json:
            pred_regions, n_pred_declared, page_region_audit = normalize_prediction_regions(
                prediction,
                source_space,
                sample.image_width,
                sample.image_height,
            )
            pred_groups, page_lig_audit = normalize_prediction_line_item_groups(
                prediction,
                source_space,
                sample.image_width,
                sample.image_height,
            )
            region_audit.update(page_region_audit)
            lig_audit.update(page_lig_audit)
            n_pages_with_regions += n_pred_declared > 0

        region_05 = region_match_counts(pred_regions, n_pred_declared, sample.regions, 0.5)
        region_075 = region_match_counts(pred_regions, n_pred_declared, sample.regions, 0.75)
        page_region_f1_05 = _f1(*region_05)
        page_region_f1_075 = _f1(*region_075)
        region_macro_05 += page_region_f1_05
        region_macro_075 += page_region_f1_075
        if valid_json:
            region_valid_macro_05 += page_region_f1_05
        region_tp_05 += region_05[0]
        region_pred_total += region_05[1]
        region_gt_total += region_05[2]

        lig_05 = box_match_counts(pred_groups, sample.line_item_groups, 0.5)
        lig_075 = box_match_counts(pred_groups, sample.line_item_groups, 0.75)
        if sample.line_item_groups:
            n_lig_applicable += 1
            page_lig_f1_05 = _f1(*lig_05)
            page_lig_f1_075 = _f1(*lig_075)
            lig_macro_05 += page_lig_f1_05
            lig_macro_075 += page_lig_f1_075
            if valid_json:
                n_lig_valid += 1
                lig_valid_macro_05 += page_lig_f1_05
            lig_value: float | str = page_lig_f1_05
            lig_075_value: float | str = page_lig_f1_075
        else:
            lig_value = NA
            lig_075_value = NA

        if per_sample_dir is not None:
            per_sample_rows.append(
                {
                    "model": spec.model,
                    "sample_id": sample.sample_id,
                    "valid_json": valid_json,
                    "R-F1": page_region_f1_05,
                    "R-F1@0.75": page_region_f1_075,
                    "R-TP@0.5": region_05[0],
                    "R-TP@0.75": region_075[0],
                    "R-pred": region_05[1],
                    "R-gt": region_05[2],
                    "LIG-F1": lig_value,
                    "LIG-F1@0.75": lig_075_value,
                    "LIG-TP@0.5": lig_05[0] if sample.line_item_groups else NA,
                    "LIG-TP@0.75": lig_075[0] if sample.line_item_groups else NA,
                    "LIG-pred": lig_05[1] if sample.line_item_groups else NA,
                    "LIG-gt": lig_05[2] if sample.line_item_groups else NA,
                }
            )

    if per_sample_dir is not None:
        write_jsonl(per_sample_dir / f"{spec.model}.jsonl", per_sample_rows)

    run_type = classify_run_type(spec.model)
    precision = region_tp_05 / region_pred_total if region_pred_total else 0.0
    recall = region_tp_05 / region_gt_total if region_gt_total else 0.0
    row = {
        "model": spec.model,
        "model_id": spec.model_id,
        "group": spec.group,
        "run_type": run_type,
        "comparison_status": classify_comparison_status(
            run_type=run_type,
            has_prediction_dir=True,
            full_scope=True,
            n_valid_json=n_valid_json,
            n_indexed=len(samples),
        ),
        "bbox_source_space": source_space,
        "bbox_space_inherited_from": inherited_from,
        "n_total": spec.n_total,
        "n_valid_json": n_valid_json,
        "coverage": n_valid_json / spec.n_total if spec.n_total else NA,
        "n_missing_prediction": n_missing_prediction,
        "n_invalid_json": n_invalid_json,
        "n_pages_with_regions": n_pages_with_regions,
        "R-Precision@0.5": precision,
        "R-Recall@0.5": recall,
        "R-micro-F1@0.5": 2.0 * region_tp_05 / (region_pred_total + region_gt_total)
        if region_pred_total + region_gt_total
        else 1.0,
        "R-F1": region_macro_05 / spec.n_total if spec.n_total else NA,
        "R-F1@0.75": region_macro_075 / spec.n_total if spec.n_total else NA,
        "R-F1-valid": region_valid_macro_05 / n_valid_json if n_valid_json else NA,
        "n_region_pred": region_pred_total,
        "n_region_gt": region_gt_total,
        "n_lig_applicable": n_lig_applicable,
        "LIG-F1": lig_macro_05 / n_lig_applicable if n_lig_applicable else NA,
        "LIG-F1@0.75": lig_macro_075 / n_lig_applicable if n_lig_applicable else NA,
        "LIG-F1-valid": lig_valid_macro_05 / n_lig_valid if n_lig_valid else NA,
        "region_bbox_declared": region_audit.declared,
        "region_bbox_valid": region_audit.valid,
        "region_bbox_clipped": region_audit.clipped,
        "region_bbox_dropped": region_audit.dropped,
        "region_unknown_type": region_audit.unknown_type,
        "lig_bbox_declared": lig_audit.declared,
        "lig_bbox_valid": lig_audit.valid,
        "lig_bbox_clipped": lig_audit.clipped,
        "lig_bbox_dropped": lig_audit.dropped,
    }
    return row


def _format_value(value: Any) -> str:
    if value == NA or value is None:
        return NA
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_results(out_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    with (out_dir / "corrected_structure_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in RESULT_COLUMNS})
    write_json(out_dir / "corrected_structure_metrics_metadata.json", metadata)

    lines = [
        "# Corrected Structure Metrics",
        "",
        "Prediction boxes are converted from an explicit run-level coordinate space to normalized [0, 1] coordinates. GT and prediction region types are mapped to a shared ontology before one-to-one IoU matching.",
        "",
        "| Run | Space | Status | Valid/Total | R-P@.5 | R-R@.5 | R-F1@.5 | R-F1@.75 | LIG-F1@.5 | Clipped | Dropped |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["bbox_source_space"]),
                    str(row["comparison_status"]),
                    f"{row['n_valid_json']}/{row['n_total']}",
                    _format_value(row["R-Precision@0.5"]),
                    _format_value(row["R-Recall@0.5"]),
                    _format_value(row["R-F1"]),
                    _format_value(row["R-F1@0.75"]),
                    _format_value(row["LIG-F1"]),
                    str(row["region_bbox_clipped"]),
                    str(row["region_bbox_dropped"]),
                ]
            )
            + " |"
        )
    (out_dir / "corrected_structure_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute canonical structure metrics for existing predictions.")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--layout-root", default="newdataset-layout")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--main-results", default="outputs/main_exp/main_results.csv")
    parser.add_argument("--bbox-manifest", default="configs/bbox_coordinate_spaces.json")
    parser.add_argument("--out", default="outputs/main_exp")
    parser.add_argument("--models", default="", help="Optional comma-separated run ids.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-per-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    index_path = Path(args.index)
    layout_root = Path(args.layout_root)
    pred_root = Path(args.pred_root)
    main_results = Path(args.main_results)
    manifest_path = Path(args.bbox_manifest)
    samples = load_structure_samples(index_path, layout_root)
    manifest = load_bbox_manifest(manifest_path)
    specs = load_run_specs(main_results, pred_root, len(samples))

    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    if requested:
        available = {spec.model for spec in specs}
        missing = requested - available
        if missing:
            raise ValueError(f"requested run ids not found: {', '.join(sorted(missing))}")
        specs = [spec for spec in specs if spec.model in requested]
    for spec in specs:
        resolve_bbox_space(manifest, spec.model)

    per_sample_dir = None if args.skip_per_sample else Path(args.out) / "corrected_structure_per_sample"
    by_model: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_run, spec, samples, pred_root, manifest, per_sample_dir): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            row = future.result()
            by_model[spec.model] = row
            print(
                f"[Structure] {spec.model}: R-F1={_format_value(row['R-F1'])} "
                f"R-F1@0.75={_format_value(row['R-F1@0.75'])} "
                f"valid={row['n_valid_json']}/{row['n_total']}",
                flush=True,
            )

    rows = [by_model[spec.model] for spec in specs]
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_path": str(index_path),
        "layout_root": str(layout_root),
        "pred_root": str(pred_root),
        "main_results_path": str(main_results),
        "bbox_manifest": str(manifest_path),
        "n_indexed": len(samples),
        "n_runs": len(rows),
        "coordinate_policy": manifest.get("policy"),
        "region_metric": {
            "primary": "page-macro R-F1@0.5 over the complete run scope; missing and invalid pages are zero",
            "strict": "page-macro R-F1@0.75 over the complete run scope",
            "diagnostics": "micro precision, recall, and F1 at IoU 0.5 plus valid-only page macro F1",
            "matching": "maximum one-to-one matching over IoU-qualified and canonical-type-compatible pairs",
            "gt_duplicate_policy": "drop a generic metadata region only when it overlaps a typed section by IoU >= 0.9",
            "gt_type_map": GT_TYPE_MAP,
            "prediction_type_map": PRED_TYPE_MAP,
            "prediction_compatibility": {key: sorted(value) for key, value in PRED_COMPATIBILITY.items()},
            "invalid_prediction_box": "counts in the prediction denominator but cannot match",
            "prompt_scope_note": "Adapters have different region output caps; this metric evaluates emitted system output, not prompt-independent detector capacity.",
        },
        "lig_metric": {
            "primary": "page-macro LIG-F1@0.5 over GT-applicable pages; missing and invalid applicable pages are zero",
            "strict": "page-macro LIG-F1@0.75 over GT-applicable pages",
        },
        "legacy_note": "This report supersedes the unnormalized R-F1/LIG-F1 values in legacy main_results.csv. Raw prediction JSON files are not modified.",
    }
    write_results(Path(args.out), rows, metadata)
    print(f"wrote corrected structure metrics -> {Path(args.out) / 'corrected_structure_metrics.csv'}")


if __name__ == "__main__":
    main()
