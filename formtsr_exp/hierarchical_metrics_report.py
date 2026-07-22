from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .hierarchical_metrics import (
    BBox,
    GridCell,
    LocalGrid,
    Relation,
    RelationCounts,
    Widget,
    WidgetGroup,
    as_serializable_counts,
    bbox_iou,
    flatten_widgets,
    lg_grits_top,
    match_widgets_for_relations,
    maximum_weight_matching,
    numeric_mean,
    relation_f1,
    widget_group_f1,
)
from .hierarchical_prediction_adapter import adapt_legacy_prediction
from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .metrics import (
    NA,
    SELECTION_TYPES,
    extract_line_item_groups,
    flatten_leaf_fields,
    node_id,
    node_label,
    normalize_metadata_layout,
)
from .page_em_report import (
    RunSpec,
    classify_comparison_status,
    classify_run_type,
    load_run_specs,
)
from .structure_metrics_report import (
    GT_TYPE_MAP,
    LIG_REGION_TYPES,
    PRED_COMPATIBILITY,
    PRED_TYPE_MAP,
    load_bbox_manifest,
    normalize_bbox,
    resolve_bbox_space,
)


RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "run_type",
    "comparison_status",
    "bbox_source_space",
    "sample_scope",
    "n_total",
    "n_valid_json",
    "coverage",
    "n_missing_prediction",
    "n_invalid_json",
    "n_lg_gt_applicable",
    "n_lg_scored",
    "LG-GriTS-Top",
    "LG-GriTS-Top-corpus",
    "n_grid_pred",
    "n_grid_gt",
    "n_grid_matches",
    "n_wg_gt_applicable",
    "n_wg_scored",
    "WG-F1",
    "WG-F1-corpus",
    "n_widget_group_pred",
    "n_widget_group_gt",
    "n_widget_gt_unknown_state",
    "n_rel_gt_applicable",
    "n_rel_scored",
    "Rel-F1",
    "Rel-Precision-micro",
    "Rel-Recall-micro",
    "Rel-F1-micro",
    "Rel-F1-matched-endpoints",
    "Rel-F1-matched-endpoints-micro",
    "n_relation_tp",
    "n_relation_pred",
    "n_relation_gt",
    "adapter_relation_declared_items",
    "adapter_relation_accepted_items",
    "adapter_relation_rejected_items",
    "adapter_relation_endpoint_aliases",
    "adapter_relation_ambiguous_endpoints",
    "adapter_relation_type_aliases",
    "adapter_relation_types_inferred",
    "adapter_grid_cells_enriched",
    "adapter_grid_fragments_merged",
    "adapter_grid_index_offsets_normalized",
    "adapter_grid_rejected_items",
    "adapter_grid_parent_inferred",
    "adapter_widget_groups_recovered",
]

PER_SAMPLE_COLUMNS = [
    "model",
    "sample_id",
    "template_name",
    "valid_json",
    "LG-GriTS-Top",
    "LG-similarity-sum",
    "LG-pred",
    "LG-gt",
    "LG-matches",
    "WG-F1",
    "WG-similarity-sum",
    "WG-pred",
    "WG-gt",
    "WG-unknown-gt-state",
    "Rel-F1",
    "Rel-F1-matched-endpoints",
    "Rel-TP",
    "Rel-pred",
    "Rel-gt",
    "Rel-matched-endpoint-pred",
    "Rel-matched-endpoint-gt",
    "adapter-relation-accepted",
    "adapter-relation-rejected",
    "adapter-grid-parent-inferred",
]


@dataclass(frozen=True, slots=True)
class RegionNode:
    id: str
    category: str
    bbox: BBox
    raw_type: str


@dataclass(frozen=True, slots=True)
class BoxNode:
    id: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class CellCandidate:
    id: str
    field_id: str
    bbox: BBox
    raw_row: Any
    raw_col: Any
    rowspan: int
    colspan: int


@dataclass(frozen=True, slots=True)
class RawAnnotationItem:
    id: str
    role: str
    bbox: BBox | None
    semantic_key: str
    original_label: str
    data_type: str
    mark_type: str
    row_id: str
    column_id: str
    row_start: int | None
    row_end: int | None
    col_start: int | None
    col_end: int | None


@dataclass(frozen=True, slots=True)
class WidgetSpec:
    id: str
    bbox: BBox
    widget_type: str
    group_id: str
    group_type: str
    field_path: tuple[str, ...]
    owner_path: tuple[str, ...]
    option_labels: tuple[str, ...]
    state_rule: str


@dataclass(frozen=True, slots=True)
class TemplateStructure:
    template_name: str
    width: float
    height: float
    regions: tuple[RegionNode, ...]
    grids: tuple[LocalGrid, ...]
    widget_specs: tuple[WidgetSpec, ...]
    relations: tuple[Relation, ...]
    line_item_groups: tuple[BoxNode, ...]
    field_paths: Mapping[tuple[str, ...], str]
    region_relation_ids: Mapping[str, str]
    line_item_relation_ids: Mapping[str, str]
    audit: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    sample_id: str
    template_name: str
    gt_answer: Any
    template: TemplateStructure
    widget_groups: tuple[WidgetGroup, ...]
    widget_unknown_states: int
    widget_state_sources: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PredictionStructure:
    regions: tuple[RegionNode, ...]
    grids: tuple[LocalGrid, ...]
    widget_groups: tuple[WidgetGroup, ...]
    relations: tuple[Relation, ...]
    line_item_groups: tuple[BoxNode, ...]
    answer_paths: tuple[tuple[str, ...], ...]
    audit: Mapping[str, int]


def _canonical_token(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _canonical_path(path: Iterable[Any]) -> tuple[str, ...]:
    return tuple(token for part in path for token in [_canonical_token(part)] if token)


def _raw_type(item: Mapping[str, Any]) -> str:
    raw = item.get("type") or item.get("region_type") or item.get("data_type") or "unknown"
    return _canonical_token(raw).replace("-", "_").replace(" ", "_")


def _widget_type(item: Mapping[str, Any]) -> str:
    raw = _raw_type(item)
    aliases = {
        "check_box": "checkbox",
        "checkbox_multi": "checkbox",
        "radio_button": "radio",
        "characterbox": "character_box",
        "blankline": "blank_line",
    }
    return aliases.get(raw, raw)


def _raw_widget_type(item: RawAnnotationItem) -> str | None:
    aliases = {
        "check_box": "checkbox",
        "checkbox_multi": "checkbox",
        "radio_button": "radio",
        "characterbox": "character_box",
        "blankline": "blank_line",
    }
    widget_type = aliases.get(item.data_type, item.data_type)
    return widget_type if widget_type in {
        "checkbox",
        "radio",
        "character_box",
        "blank_line",
        "signature",
    } else None


def _is_selection(item: Mapping[str, Any]) -> bool:
    return _raw_type(item) in SELECTION_TYPES | {"checkbox_multi"} or item.get("mark") not in (
        None,
        "",
        False,
    )


def _strict_positive_int(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return -1
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return -1
    return parsed if parsed >= 1 else -1


def _parse_index(value: Any) -> int:
    if isinstance(value, bool):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _normalized_box(
    value: Any,
    source_space: str,
    width: float,
    height: float,
) -> BBox | None:
    box, _status = normalize_bbox(value, source_space, width, height)
    return box


def _required_gt_box(value: Any, width: float, height: float, context: str) -> BBox:
    box, status = normalize_bbox(value, "pixel", width, height)
    if box is None or status != "ok":
        raise ValueError(f"invalid GT bbox for {context}: {value!r} ({status})")
    return box


def _image_size(layout: Mapping[str, Any], template_name: str) -> tuple[float, float]:
    try:
        width = float(layout["original_width"])
        height = float(layout["original_height"])
    except (KeyError, TypeError, ValueError):
        structure = (layout.get("metadata") or {}).get("layout_structure", {})
        raw_page = structure.get("page_bbox") if isinstance(structure, dict) else None
        if not isinstance(raw_page, list) or len(raw_page) != 4:
            raise ValueError(f"missing image dimensions for template {template_name!r}")
        width = float(raw_page[2]) - float(raw_page[0])
        height = float(raw_page[3]) - float(raw_page[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions for template {template_name!r}")
    return width, height


def _unique_rows(items: Iterable[Relation]) -> tuple[Relation, ...]:
    return tuple(sorted(set(items)))


def _relation_from_item(item: Mapping[str, Any], index: int, *, gt: bool = False) -> Relation:
    source = item.get("source") or item.get("from") or item.get("u") or item.get("parent")
    target = item.get("target") or item.get("to") or item.get("v") or item.get("child")
    relation_type = item.get("type") or item.get("relation_type") or item.get("r")
    if source is None and "key" in item:
        source = item.get("key")
    if target is None and "value" in item:
        target = item.get("value")
    if gt and (source in (None, "") or target in (None, "") or relation_type in (None, "")):
        raise ValueError(f"malformed GT relation: {item!r}")
    return Relation(
        str(source) if source not in (None, "") else f"__missing_source_{index}",
        str(relation_type) if relation_type not in (None, "") else "__missing_type__",
        str(target) if target not in (None, "") else f"__missing_target_{index}",
    )


def _inside(inner: BBox, outer: BBox) -> bool:
    center_x = (inner[0] + inner[2]) / 2.0
    center_y = (inner[1] + inner[3]) / 2.0
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def _contains_all(outer: BBox, inner_boxes: Sequence[BBox]) -> bool:
    return all(_inside(inner, outer) for inner in inner_boxes)


def _bbox_union(boxes: Sequence[BBox]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, _canonical_token(value))


def _build_grid(
    table_id: str,
    table_bbox: BBox,
    candidates: Sequence[CellCandidate],
) -> LocalGrid | None:
    selected = [candidate for candidate in candidates if _inside(candidate.bbox, table_bbox)]
    if not selected:
        return None
    raw_rows = sorted({str(candidate.raw_row) for candidate in selected}, key=_sort_key)
    row_lookup = {value: index for index, value in enumerate(raw_rows)}

    column_keys: dict[str, list[CellCandidate]] = {}
    for candidate in selected:
        key = f"explicit:{candidate.raw_col}" if candidate.raw_col not in (None, "") else f"field:{candidate.field_id}"
        column_keys.setdefault(key, []).append(candidate)

    def column_order(item: tuple[str, list[CellCandidate]]) -> tuple[int, float]:
        key, cells = item
        if key.startswith("explicit:"):
            raw = key.split(":", 1)[1]
            try:
                return (0, float(raw))
            except ValueError:
                pass
        centers = sorted((cell.bbox[0] + cell.bbox[2]) / 2.0 for cell in cells)
        return (1, centers[len(centers) // 2])

    ordered_columns = sorted(column_keys.items(), key=column_order)
    col_lookup = {key: index for index, (key, _cells) in enumerate(ordered_columns)}
    cells: list[GridCell] = []
    for candidate in selected:
        col_key = (
            f"explicit:{candidate.raw_col}"
            if candidate.raw_col not in (None, "")
            else f"field:{candidate.field_id}"
        )
        cells.append(
            GridCell(
                id=candidate.id,
                row=row_lookup[str(candidate.raw_row)],
                col=col_lookup[col_key],
                rowspan=candidate.rowspan,
                colspan=candidate.colspan,
                bbox=candidate.bbox,
            )
        )
    return LocalGrid(table_id.replace("table_", "grid_", 1), table_id, tuple(cells), table_bbox)


def _template_widget_specs(
    layout: Mapping[str, Any], width: float, height: float
) -> tuple[
    tuple[WidgetSpec, ...],
    Mapping[tuple[str, ...], str],
    tuple[CellCandidate, ...],
    int,
]:
    specs: list[WidgetSpec] = []
    field_paths: dict[tuple[str, ...], str] = {}
    ambiguous_field_paths: set[tuple[str, ...]] = set()
    cell_candidates: list[CellCandidate] = []

    def visit(
        node: Mapping[str, Any],
        path: tuple[str, ...],
        parent_path: tuple[str, ...],
        index_path: tuple[int, ...],
    ) -> None:
        label = node_label(dict(node), path[-1] if path else "field")
        current_path = path + (label or f"node{'.'.join(map(str, index_path))}",)
        current_id = node_id("field", current_path)
        canonical_path = _canonical_path(current_path)
        if canonical_path not in ambiguous_field_paths:
            previous = field_paths.get(canonical_path)
            if previous is None:
                field_paths[canonical_path] = current_id
            elif previous != current_id:
                field_paths.pop(canonical_path, None)
                ambiguous_field_paths.add(canonical_path)

        value = node.get("value")
        if isinstance(value, dict):
            raw_row = value.get("row_id")
            box = _normalized_box(value, "pixel", width, height)
            if raw_row is not None and box is not None:
                cell_candidates.append(
                    CellCandidate(
                        node_id("value", current_path),
                        current_id,
                        box,
                        raw_row,
                        value.get("column_id"),
                        _strict_positive_int(value.get("rowspan")),
                        _strict_positive_int(value.get("colspan")),
                    )
                )

        value_items = node.get("values") if isinstance(node.get("values"), list) else []
        value_widget_ids: list[str] = []
        for value_index, item in enumerate(value_items):
            if not isinstance(item, dict):
                continue
            item_id = node_id("value", current_path + (str(value_index),))
            item_box = _normalized_box(item, "pixel", width, height)
            if item.get("row_id") is not None and item_box is not None:
                cell_candidates.append(
                    CellCandidate(
                        item_id,
                        current_id,
                        item_box,
                        item.get("row_id"),
                        item.get("column_id"),
                        _strict_positive_int(item.get("rowspan")),
                        _strict_positive_int(item.get("colspan")),
                    )
                )
            if not _is_selection(item) or item_box is None:
                continue
            widget_id = node_id("widget", current_path + (str(value_index),))
            value_widget_ids.append(widget_id)
            item_type = _widget_type(item)
            if item_type in {"unknown", "text", "value"}:
                item_type = "checkbox"
            specs.append(
                WidgetSpec(
                    id=widget_id,
                    bbox=item_box,
                    widget_type=item_type,
                    group_id=f"widget-group:{current_id}",
                    group_type=item_type,
                    field_path=current_path,
                    owner_path=current_path,
                    option_labels=(),
                    state_rule="answer_presence",
                )
            )

        if _is_selection(node):
            box = _normalized_box(node, "pixel", width, height)
            if box is not None:
                widget_type = _widget_type(node)
                if widget_type in {"unknown", "text", "value"}:
                    widget_type = "checkbox"
                specs.append(
                    WidgetSpec(
                        id=node_id("widget", current_path),
                        bbox=box,
                        widget_type=widget_type,
                        group_id=f"widget-group:{current_id}",
                        group_type=widget_type,
                        field_path=current_path,
                        owner_path=parent_path or current_path,
                        option_labels=tuple(
                            value
                            for value in (
                                str(node.get("original_label") or "").strip(),
                                str(node.get("semantic_key") or "").strip(),
                            )
                            if value
                        ),
                        state_rule="option_membership" if parent_path else "answer_presence",
                    )
                )

        children = node.get("keys") if isinstance(node.get("keys"), list) else []
        for child_index, child in enumerate(children):
            if isinstance(child, dict):
                visit(child, current_path, current_path, index_path + (child_index,))

    fields = layout.get("fields") if isinstance(layout.get("fields"), list) else []
    for index, root in enumerate(fields):
        if isinstance(root, dict):
            visit(root, (), (), (index,))
    return (
        tuple(specs),
        field_paths,
        tuple(cell_candidates),
        len(ambiguous_field_paths),
    )


def _annotation_scalar(result: Mapping[str, Any]) -> Any:
    value = result.get("value")
    if not isinstance(value, dict):
        return None
    for key in ("text", "choices"):
        candidate = value.get(key)
        if isinstance(candidate, list) and candidate:
            return candidate[0]
    return None


def _annotation_bbox(result: Mapping[str, Any]) -> BBox | None:
    value = result.get("value")
    if not isinstance(value, dict):
        return None
    try:
        x = float(value["x"]) / 100.0
        y = float(value["y"]) / 100.0
        width = float(value["width"]) / 100.0
        height = float(value["height"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    box = (x, y, x + width, y + height)
    if not all(math.isfinite(coord) for coord in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return tuple(min(1.0, max(0.0, coord)) for coord in box)  # type: ignore[return-value]


def _optional_annotation_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_raw_annotation(
    raw: Any,
) -> tuple[dict[str, RawAnnotationItem], list[tuple[str, str, str]], float, float]:
    tasks = raw if isinstance(raw, list) else [raw]
    if len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ValueError("template metadata must contain exactly one Label Studio task")
    task = tasks[0]
    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations or not isinstance(annotations[0], dict):
        raise ValueError("template metadata has no annotation result")
    results = annotations[0].get("result")
    if not isinstance(results, list):
        raise ValueError("template metadata annotation result must be a list")

    partial: dict[str, dict[str, Any]] = {}
    relations: list[tuple[str, str, str]] = []
    width: float | None = None
    height: float | None = None
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("type") == "relation":
            source = result.get("from_id")
            target = result.get("to_id")
            if source not in (None, "") and target not in (None, ""):
                direction = _canonical_token(result.get("direction"))
                source_id = str(source)
                target_id = str(target)
                if direction == "left":
                    source_id, target_id = target_id, source_id
                relations.append((source_id, target_id, direction))
            continue
        item_id = result.get("id")
        if item_id in (None, ""):
            continue
        if width is None and result.get("original_width") is not None:
            width = float(result["original_width"])
        if height is None and result.get("original_height") is not None:
            height = float(result["original_height"])
        item = partial.setdefault(str(item_id), {"id": str(item_id)})
        from_name = str(result.get("from_name") or "")
        if from_name == "bbox" and result.get("type") == "rectanglelabels":
            labels = (result.get("value") or {}).get("rectanglelabels", [])
            item["role"] = str(labels[0]) if isinstance(labels, list) and labels else ""
            item["bbox"] = _annotation_bbox(result)
        elif from_name in {
            "semantic_key",
            "original_label",
            "data_type",
            "mark_type",
            "mark_type_multi",
            "row_id",
            "column_id",
            "row_start",
            "row_end",
            "col_start",
            "col_end",
        }:
            key = "mark_type" if from_name == "mark_type_multi" else from_name
            item[key] = _annotation_scalar(result)
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("template metadata is missing original image dimensions")

    items = {
        item_id: RawAnnotationItem(
            id=item_id,
            role=_canonical_token(item.get("role")).replace("-", "_").replace(" ", "_"),
            bbox=item.get("bbox"),
            semantic_key=str(item.get("semantic_key") or "").strip(),
            original_label=str(item.get("original_label") or "").strip(),
            data_type=_canonical_token(item.get("data_type")).replace("-", "_").replace(" ", "_"),
            mark_type=_canonical_token(item.get("mark_type")),
            row_id=str(item.get("row_id") or "").strip(),
            column_id=str(item.get("column_id") or "").strip(),
            row_start=_optional_annotation_int(item.get("row_start")),
            row_end=_optional_annotation_int(item.get("row_end")),
            col_start=_optional_annotation_int(item.get("col_start")),
            col_end=_optional_annotation_int(item.get("col_end")),
        )
        for item_id, item in partial.items()
    }
    return items, relations, width, height


def _raw_item_label(item: RawAnnotationItem) -> str:
    return item.original_label or item.semantic_key or item.id


def _raw_field_paths(
    items: Mapping[str, RawAnnotationItem],
    raw_relations: Sequence[tuple[str, str, str]],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[tuple[str, ...], str],
    int,
]:
    key_ids = {item.id for item in items.values() if item.role == "key"}
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    for source, target, _direction in raw_relations:
        if source in key_ids and target in key_ids:
            parents.setdefault(target, []).append(source)
            children.setdefault(source, []).append(target)
    roots = sorted(key_ids - set(parents))
    paths_by_id: dict[str, tuple[str, ...]] = {}

    def visit(item_id: str, path: tuple[str, ...], visiting: set[str]) -> None:
        if item_id in visiting:
            return
        current = path + (_raw_item_label(items[item_id]),)
        previous = paths_by_id.get(item_id)
        if previous is None or (len(current), _canonical_path(current)) < (
            len(previous),
            _canonical_path(previous),
        ):
            paths_by_id[item_id] = current
        for child in sorted(children.get(item_id, [])):
            visit(child, current, visiting | {item_id})

    for root in roots:
        visit(root, (), set())
    for item_id in sorted(key_ids - set(paths_by_id)):
        visit(item_id, (), set())
    field_paths: dict[tuple[str, ...], str] = {}
    ambiguous_paths: set[tuple[str, ...]] = set()
    for item_id, path in paths_by_id.items():
        canonical_path = _canonical_path(path)
        if canonical_path in ambiguous_paths:
            continue
        previous = field_paths.get(canonical_path)
        if previous is None:
            field_paths[canonical_path] = item_id
        elif previous != item_id:
            field_paths.pop(canonical_path, None)
            ambiguous_paths.add(canonical_path)
    return paths_by_id, field_paths, len(ambiguous_paths)


def _raw_region_category(
    item: RawAnnotationItem,
    key_with_children: set[str],
) -> str | None:
    if item.role in {"section", "region", "header"}:
        return "group"
    if item.role == "table_region":
        return "table"
    if item.role == "key":
        if item.data_type in {"checkbox", "checkbox_multi"}:
            return "widget"
        return "group" if item.id in key_with_children else "label"
    if item.role == "value":
        return "widget" if item.data_type in {"checkbox", "checkbox_multi"} else "value"
    return None


def _cell_components(cells: Sequence[RawAnnotationItem]) -> list[list[RawAnnotationItem]]:
    if not cells:
        return []
    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def connected(left: RawAnnotationItem, right: RawAnnotationItem) -> bool:
        assert left.bbox is not None and right.bbox is not None
        horizontal_gap = max(0.0, max(left.bbox[0], right.bbox[0]) - min(left.bbox[2], right.bbox[2]))
        vertical_gap = max(0.0, max(left.bbox[1], right.bbox[1]) - min(left.bbox[3], right.bbox[3]))
        x_overlap = min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0])
        y_overlap = min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1])
        spatially_adjacent = (horizontal_gap <= 0.015 and y_overlap >= -0.002) or (
            vertical_gap <= 0.015 and x_overlap >= -0.002
        )
        if not spatially_adjacent:
            return False
        assert left.row_start is not None and left.row_end is not None
        assert left.col_start is not None and left.col_end is not None
        assert right.row_start is not None and right.row_end is not None
        assert right.col_start is not None and right.col_end is not None
        topology_adjacent = (
            left.row_start <= right.row_end + 1
            and right.row_start <= left.row_end + 1
            and left.col_start <= right.col_end + 1
            and right.col_start <= left.col_end + 1
        )
        return topology_adjacent

    for left in range(len(cells)):
        for right in range(left + 1, len(cells)):
            if connected(cells[left], cells[right]):
                union(left, right)
    groups: dict[int, list[RawAnnotationItem]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(find(index), []).append(cell)
    return sorted(groups.values(), key=lambda group: min(cell.id for cell in group))


def _raw_grid_from_cells(
    grid_id: str,
    parent_id: str,
    cells: Sequence[RawAnnotationItem],
) -> tuple[LocalGrid | None, int]:
    min_row = min(cell.row_start for cell in cells if cell.row_start is not None)
    min_col = min(cell.col_start for cell in cells if cell.col_start is not None)
    boxes = [cell.bbox for cell in cells if cell.bbox is not None]
    grid_cells = [
        GridCell(
            id=cell.id,
            row=int(cell.row_start) - int(min_row),
            col=int(cell.col_start) - int(min_col),
            rowspan=int(cell.row_end) - int(cell.row_start) + 1,
            colspan=int(cell.col_end) - int(cell.col_start) + 1,
            bbox=cell.bbox,
        )
        for cell in cells
        if cell.bbox is not None
        and cell.row_start is not None
        and cell.row_end is not None
        and cell.col_start is not None
        and cell.col_end is not None
    ]
    occupied: set[tuple[int, int]] = set()
    for cell in grid_cells:
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                if (row, col) in occupied:
                    return None, 0
                occupied.add((row, col))
    n_rows = max(cell.row + cell.rowspan for cell in grid_cells)
    n_cols = max(cell.col + cell.colspan for cell in grid_cells)
    implicit_count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) in occupied:
                continue
            grid_cells.append(
                GridCell(f"{grid_id}:implicit:{row}:{col}", row, col, 1, 1, None)
            )
            implicit_count += 1
    return LocalGrid(grid_id, parent_id, tuple(grid_cells), _bbox_union(boxes)), implicit_count


def load_raw_template_structure(path: Path, template_name: str) -> TemplateStructure:
    items, raw_relations, width, height = parse_raw_annotation(read_json(path))
    paths_by_id, field_paths, ambiguous_field_paths = _raw_field_paths(
        items, raw_relations
    )
    key_with_children = {
        source
        for source, target, _direction in raw_relations
        if items.get(source) is not None
        and items.get(target) is not None
        and items[source].role == "key"
        and items[target].role == "key"
    }
    regions = [
        RegionNode(item.id, category, item.bbox, item.role)
        for item in items.values()
        for category in [_raw_region_category(item, key_with_children)]
        if category is not None and item.bbox is not None
    ]

    localized_cells = [
        item for item in items.values() if item.role == "cell" and item.bbox is not None
    ]
    complete_cells = [
        item
        for item in localized_cells
        if item.bbox is not None
        and item.row_start is not None
        and item.row_end is not None
        and item.col_start is not None
        and item.col_end is not None
        and item.row_end >= item.row_start
        and item.col_end >= item.col_start
    ]
    all_cell_count = sum(item.role == "cell" for item in items.values())
    incomplete_cell_count = all_cell_count - len(complete_cells)
    complete_cell_ids = {item.id for item in complete_cells}
    table_regions = [
        item for item in items.values() if item.role == "table_region" and item.bbox is not None
    ]
    parent_regions = [
        item
        for item in items.values()
        if item.role in {"table_region", "region", "section"} and item.bbox is not None
    ]
    cells_by_parent: dict[str, list[RawAnnotationItem]] = {}
    unassigned_cells: list[RawAnnotationItem] = []
    for cell in localized_cells:
        assert cell.bbox is not None
        containing = [
            item
            for item in parent_regions
            if item.bbox is not None and _contains_all(item.bbox, [cell.bbox])
        ]
        parent_item = min(containing, key=lambda item: _bbox_area(item.bbox), default=None)
        if parent_item is None:
            unassigned_cells.append(cell)
        else:
            cells_by_parent.setdefault(parent_item.id, []).append(cell)
    grids: list[LocalGrid] = []
    synthetic_regions: list[RegionNode] = []
    implicit_cells = 0
    invalid_overlapping_grids = 0
    invalid_incomplete_grids = 0
    grid_index = 0
    for parent_id, parent_cells in sorted(cells_by_parent.items()):
        if any(cell.id not in complete_cell_ids for cell in parent_cells):
            invalid_incomplete_grids += 1
            grid_index += 1
            continue
        grid, added_implicit = _raw_grid_from_cells(
            f"grid:{parent_id}:{grid_index}", parent_id, parent_cells
        )
        if grid is None:
            invalid_overlapping_grids += 1
        else:
            grids.append(grid)
            implicit_cells += added_implicit
        grid_index += 1

    unassigned_complete = [
        cell for cell in unassigned_cells if cell.id in complete_cell_ids
    ]
    unassigned_incomplete = [
        cell for cell in unassigned_cells if cell.id not in complete_cell_ids
    ]
    unassigned_components = _cell_components(unassigned_complete)
    for cell in unassigned_incomplete:
        assert cell.bbox is not None
        cell_center = (
            (cell.bbox[0] + cell.bbox[2]) / 2.0,
            (cell.bbox[1] + cell.bbox[3]) / 2.0,
        )
        if not unassigned_components:
            unassigned_components.append([cell])
            continue
        nearest = min(
            unassigned_components,
            key=lambda component: min(
                (
                    cell_center[0] - (member.bbox[0] + member.bbox[2]) / 2.0
                )
                ** 2
                + (
                    cell_center[1] - (member.bbox[1] + member.bbox[3]) / 2.0
                )
                ** 2
                for member in component
                if member.bbox is not None
            ),
        )
        nearest.append(cell)

    for component in unassigned_components:
        boxes = [cell.bbox for cell in component if cell.bbox is not None]
        parent_id = f"synthetic-table-region:{grid_index}"
        parent_bbox = _bbox_union(boxes)
        synthetic_regions.append(RegionNode(parent_id, "table", parent_bbox, "table_region"))
        if any(cell.id not in complete_cell_ids for cell in component):
            invalid_incomplete_grids += 1
            grid_index += 1
            continue
        grid, added_implicit = _raw_grid_from_cells(
            f"grid:{parent_id}:{grid_index}", parent_id, component
        )
        if grid is None:
            invalid_overlapping_grids += 1
        else:
            grids.append(grid)
            implicit_cells += added_implicit
        grid_index += 1
    regions.extend(synthetic_regions)

    widget_items = {
        item.id: item
        for item in items.values()
        if _raw_widget_type(item) is not None and item.bbox is not None
    }
    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for source, target, _direction in raw_relations:
        incoming.setdefault(target, []).append(source)
        outgoing.setdefault(source, []).append(target)
    widget_owner: dict[str, str] = {}
    self_widget_relations_ignored = 0
    widgets_with_multiple_key_parents = 0
    for widget_id, item in widget_items.items():
        self_widget_relations_ignored += sum(
            source == widget_id for source in incoming.get(widget_id, [])
        )
        key_parents = sorted(
            {
            source
            for source in incoming.get(widget_id, [])
            if source != widget_id
            and source in items
            and items[source].role == "key"
            }
        )
        widgets_with_multiple_key_parents += len(key_parents) > 1
        if item.role == "key" and key_parents:
            widget_owner[widget_id] = key_parents[0]
        elif key_parents:
            widget_owner[widget_id] = key_parents[0]
        else:
            widget_owner[widget_id] = widget_id
    owner_members: dict[str, list[RawAnnotationItem]] = {}
    for widget_id, owner_id in widget_owner.items():
        owner_members.setdefault(owner_id, []).append(widget_items[widget_id])
    widget_specs: list[WidgetSpec] = []
    for owner_id, members in sorted(owner_members.items()):
        member_types = {_raw_widget_type(member) for member in members}
        group_type = (
            "checkbox_multi"
            if any(member.data_type == "checkbox_multi" for member in members)
            else next(iter(member_types))
            if len(member_types) == 1
            else "mixed"
        )
        owner_path = paths_by_id.get(owner_id, paths_by_id.get(members[0].id, ()))
        for member in sorted(members, key=lambda value: value.id):
            member_path = paths_by_id.get(member.id, owner_path + (_raw_item_label(member),))
            widget_specs.append(
                WidgetSpec(
                    id=member.id,
                    bbox=member.bbox,  # type: ignore[arg-type]
                    widget_type=_raw_widget_type(member) or "unknown",
                    group_id=f"widget-group:{owner_id}",
                    group_type=group_type,
                    field_path=member_path,
                    owner_path=owner_path,
                    option_labels=tuple(
                        value
                        for value in (member.original_label, member.semantic_key)
                        if value
                    ),
                    state_rule="option_membership" if owner_id != member.id else "answer_presence",
                )
            )

    line_groups = tuple(
        BoxNode(item.id, item.bbox)
        for item in items.values()
        if item.role == "line_item_group" and item.bbox is not None
    )

    typed_relations: list[Relation] = []
    widget_ids = set(widget_items)
    for source, target, _direction in raw_relations:
        source_item = items.get(source)
        target_item = items.get(target)
        if source_item is None or target_item is None:
            continue
        if target in widget_ids:
            relation_type = "field-widget"
        elif source_item.role == "key" and target_item.role == "key":
            relation_type = "parent-child"
        elif source_item.role == "key" and target_item.role == "value":
            relation_type = "key-value"
        elif source_item.role == "section":
            relation_type = "section-membership"
        elif source_item.role == "line_item_group" or target_item.role == "line_item_group":
            relation_type = "line-item-membership"
        else:
            relation_type = f"{source_item.role}-to-{target_item.role}"
        typed_relations.append(Relation(source, relation_type, target))

    return TemplateStructure(
        template_name=template_name,
        width=width,
        height=height,
        regions=tuple(regions),
        grids=tuple(grids),
        widget_specs=tuple(widget_specs),
        relations=_unique_rows(typed_relations),
        line_item_groups=line_groups,
        field_paths=field_paths,
        region_relation_ids={region.id: region.id for region in regions},
        line_item_relation_ids={group.id: group.id for group in line_groups},
        audit={
            "gt_source": str(path),
            "raw_objects": len(items),
            "raw_relations": len(raw_relations),
            "table_regions": len(table_regions),
            "reconstructable_local_grids": len(grids),
            "table_regions_without_row_col_topology": 0,
            "complete_cells": len(complete_cells),
            "incomplete_cells": incomplete_cell_count,
            "implicit_empty_cells": implicit_cells,
            "invalid_overlapping_grids_excluded": invalid_overlapping_grids,
            "invalid_incomplete_grids_excluded": invalid_incomplete_grids,
            "unlocalized_incomplete_cells": all_cell_count - len(localized_cells),
            "metadata_observable_widgets": len(widget_specs),
            "explicit_widget_groups": len(owner_members),
            "self_widget_relations_ignored": self_widget_relations_ignored,
            "widgets_with_multiple_key_parents": widgets_with_multiple_key_parents,
            "widget_mark_styles": {
                style: sum(item.mark_type == style for item in widget_items.values())
                for style in sorted({item.mark_type for item in widget_items.values() if item.mark_type})
            },
            "derived_typed_relations": len(set(typed_relations)),
            "synthetic_grid_parent_regions": len(synthetic_regions),
            "ambiguous_field_schema_paths": ambiguous_field_paths,
        },
    )


def load_template_structure(path: Path, template_name: str) -> TemplateStructure:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"layout must be an object: {path}")
    width, height = _image_size(raw, template_name)
    normalized = normalize_metadata_layout(raw)
    if not isinstance(normalized, dict):
        raise ValueError(f"normalized layout must be an object: {path}")

    regions: list[RegionNode] = []
    region_id_counts: dict[str, int] = {}
    for index, item in enumerate(normalized.get("regions", [])):
        if not isinstance(item, dict):
            continue
        raw_type = _raw_type(item)
        category = GT_TYPE_MAP.get(raw_type)
        if category is None:
            raise ValueError(f"unmapped GT region type {raw_type!r} in {path}")
        base_id = str(item.get("id") or f"gt-region-{index}")
        occurrence = region_id_counts.get(base_id, 0)
        region_id_counts[base_id] = occurrence + 1
        region_id = base_id if occurrence == 0 else f"{base_id}#{occurrence}"
        regions.append(
            RegionNode(
                region_id,
                category,
                _required_gt_box(item, width, height, f"{path}:region:{index}"),
                raw_type,
            )
        )

    (
        widget_specs,
        field_paths,
        cell_candidates,
        ambiguous_field_paths,
    ) = _template_widget_specs(raw, width, height)
    tables: list[tuple[str, BBox]] = []
    structure = (raw.get("metadata") or {}).get("layout_structure", {})
    sections = structure.get("sections", []) if isinstance(structure, dict) else []
    for section_index, section in enumerate(sections if isinstance(sections, list) else []):
        if not isinstance(section, dict):
            continue
        for table_index, table in enumerate(
            section.get("table_regions", []) if isinstance(section.get("table_regions"), list) else []
        ):
            if not isinstance(table, dict):
                continue
            table_id = str(table.get("table_id") or table.get("source_region_id") or f"table-{section_index}-{table_index}")
            table_bbox = _required_gt_box(table, width, height, f"{path}:table:{table_id}")
            tables.append((table_id, table_bbox))

    # Match the corrected R-F1 policy: drop near-identical generic regions in
    # favor of the typed section, while retaining genuine subdivisions.
    section_regions = [region for region in regions if region.raw_type == "section"]
    regions = [
        region
        for region in regions
        if not (
            region.raw_type == "region"
            and any(bbox_iou(region.bbox, section.bbox) >= 0.9 for section in section_regions)
        )
    ]

    line_groups: list[BoxNode] = []
    line_group_id_counts: dict[str, int] = {}
    for index, group in enumerate(extract_line_item_groups(normalized)):
        base_id = str(
            group.get("id")
            or group.get("line_item_group_id")
            or group.get("source_region_id")
            or f"line-item-{index}"
        )
        occurrence = line_group_id_counts.get(base_id, 0)
        line_group_id_counts[base_id] = occurrence + 1
        group_id = base_id if occurrence == 0 else f"{base_id}#{occurrence}"
        line_groups.append(
            BoxNode(
                group_id,
                _required_gt_box(group, width, height, f"{path}:line-item:{group_id}"),
            )
        )

    grids: list[LocalGrid] = []
    tables_without_topology = 0
    for table_id, table_bbox in tables:
        grid = _build_grid(table_id, table_bbox, cell_candidates)
        if grid is None:
            tables_without_topology += 1
        else:
            parent_candidates = [
                region
                for region in regions
                if region.category in PRED_COMPATIBILITY.get("table", set())
            ]
            parent = max(
                parent_candidates,
                key=lambda region: bbox_iou(table_bbox, region.bbox),
                default=None,
            )
            parent_id = (
                parent.id
                if parent is not None and bbox_iou(table_bbox, parent.bbox) >= 0.5
                else f"unmatched-r-f1-parent:{table_id}"
            )
            grids.append(replace(grid, parent_region_id=parent_id))

    relations = tuple(
        _relation_from_item(item, index, gt=True)
        for index, item in enumerate(normalized.get("relations", []))
        if isinstance(item, dict)
    )
    return TemplateStructure(
        template_name=template_name,
        width=width,
        height=height,
        regions=tuple(regions),
        grids=tuple(grids),
        widget_specs=widget_specs,
        relations=_unique_rows(relations),
        line_item_groups=tuple(line_groups),
        field_paths=field_paths,
        region_relation_ids={region.id: region.id for region in regions},
        line_item_relation_ids={group.id: group.id for group in line_groups},
        audit={
            "table_regions": len(tables),
            "r_f1_regions": len(regions),
            "reconstructable_local_grids": len(grids),
            "table_regions_without_row_col_topology": tables_without_topology,
            "row_id_cell_candidates": len(cell_candidates),
            "metadata_observable_widgets": len(widget_specs),
            "derived_typed_relations": len(set(relations)),
            "ambiguous_field_schema_paths": ambiguous_field_paths,
        },
    )


def reconcile_with_r_f1_structure(
    raw_template: TemplateStructure,
    r_f1_template: TemplateStructure,
) -> TemplateStructure:
    """Use the exact corrected R-F1/LIG GT universes with raw topology and edges."""
    raw_regions = {region.id: region for region in raw_template.regions}
    remapped_grids: list[LocalGrid] = []
    unmatched_grid_parents = 0
    for grid in raw_template.grids:
        raw_parent = raw_regions.get(grid.parent_region_id)
        candidates = (
            [
                region
                for region in r_f1_template.regions
                if raw_parent is not None
                and region.category
                in PRED_COMPATIBILITY.get(raw_parent.category, set())
            ]
            if raw_parent is not None
            else []
        )
        target = max(
            candidates,
            key=lambda region: bbox_iou(raw_parent.bbox, region.bbox),  # type: ignore[union-attr]
            default=None,
        )
        if (
            raw_parent is None
            or target is None
            or bbox_iou(raw_parent.bbox, target.bbox) < 0.5
        ):
            parent_id = f"unmatched-r-f1-parent:{grid.parent_region_id}"
            unmatched_grid_parents += 1
        else:
            parent_id = target.id
        remapped_grids.append(replace(grid, parent_region_id=parent_id))

    region_relation_ids = match_regions(
        r_f1_template.regions, raw_template.regions
    )
    line_item_relation_ids = match_box_nodes(
        r_f1_template.line_item_groups, raw_template.line_item_groups
    )
    recoverable_relation_endpoints = (
        set(region_relation_ids.values())
        | set(line_item_relation_ids.values())
        | {spec.id for spec in raw_template.widget_specs}
        | set(raw_template.field_paths.values())
        | {grid.id for grid in raw_template.grids}
        | {
            cell.id
            for grid in raw_template.grids
            for cell in grid.cells
        }
    )
    relations = tuple(
        relation
        for relation in raw_template.relations
        if relation.source in recoverable_relation_endpoints
        and relation.target in recoverable_relation_endpoints
    )
    excluded_relation_endpoints = {
        endpoint
        for relation in raw_template.relations
        for endpoint in (relation.source, relation.target)
        if endpoint not in recoverable_relation_endpoints
    }
    audit = dict(raw_template.audit)
    audit.update(
        {
            "r_f1_regions": len(r_f1_template.regions),
            "raw_semantic_regions": len(raw_template.regions),
            "grid_parents_without_r_f1_correspondence": unmatched_grid_parents,
            "r_f1_region_relation_aliases": len(region_relation_ids),
            "r_f1_line_item_groups": len(r_f1_template.line_item_groups),
            "raw_line_item_groups": len(raw_template.line_item_groups),
            "lig_relation_aliases": len(line_item_relation_ids),
            "recoverable_typed_relations": len(relations),
            "unrecoverable_typed_relations_excluded": (
                len(raw_template.relations) - len(relations)
            ),
            "unrecoverable_relation_endpoints": len(excluded_relation_endpoints),
        }
    )
    return replace(
        raw_template,
        regions=r_f1_template.regions,
        grids=tuple(remapped_grids),
        relations=relations,
        line_item_groups=r_f1_template.line_item_groups,
        region_relation_ids=region_relation_ids,
        line_item_relation_ids=line_item_relation_ids,
        audit=audit,
    )


def load_samples(
    index_path: Path,
    metadata_root: Path,
    layout_root: Path,
) -> list[EvaluationSample]:
    templates: dict[str, TemplateStructure] = {}
    samples: list[EvaluationSample] = []
    seen: set[str] = set()
    for row in read_jsonl(index_path):
        sample_id = str(row["sample_id"])
        template_name = str(row["template_name"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in index: {sample_id}")
        seen.add(sample_id)
        if template_name not in templates:
            metadata_path = metadata_root / f"{template_name}.json"
            layout_path = layout_root / f"{template_name}.json"
            if metadata_path.exists():
                raw_template = load_raw_template_structure(
                    metadata_path, template_name
                )
                if not layout_path.exists():
                    raise ValueError(
                        f"raw metadata requires corrected R-F1 layout metadata: {layout_path}"
                    )
                r_f1_template = load_template_structure(layout_path, template_name)
                templates[template_name] = reconcile_with_r_f1_structure(
                    raw_template, r_f1_template
                )
            elif layout_path.exists():
                templates[template_name] = load_template_structure(
                    layout_path, template_name
                )
            else:
                raise ValueError(
                    f"missing raw and converted template metadata for {template_name!r}"
                )
        gt_answer = read_json(Path(str(row["label_path"])))
        widget_groups, unknown_states, state_sources = resolve_widget_groups(
            templates[template_name], gt_answer
        )
        samples.append(
            EvaluationSample(
                sample_id,
                template_name,
                gt_answer,
                templates[template_name],
                widget_groups,
                unknown_states,
                state_sources,
            )
        )
    return samples


def _truthy_answer(value: Any) -> bool:
    if value in (None, "", False):
        return False
    if isinstance(value, str):
        return _canonical_token(value) not in {"", "false", "no", "none", "0", "unchecked"}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def resolve_widget_groups(
    template: TemplateStructure, gt_answer: Any
) -> tuple[tuple[WidgetGroup, ...], int, Mapping[str, int]]:
    fields = flatten_leaf_fields(gt_answer)
    canonical_fields = {_canonical_path(path): value for path, value in fields.items()}

    def answer_owner_prefix(
        answer_path: tuple[str, ...], owner_path: tuple[str, ...]
    ) -> tuple[str, ...] | None:
        owner_index = 0
        for answer_index, part in enumerate(answer_path):
            if owner_index < len(owner_path) and part == owner_path[owner_index]:
                owner_index += 1
            elif part.isdecimal():
                continue
            else:
                return None
            if owner_index == len(owner_path):
                return answer_path[: answer_index + 1]
        return None

    specs_by_signature: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[WidgetSpec]
    ] = {}
    for spec in template.widget_specs:
        signature = (
            _canonical_path(spec.owner_path),
            _canonical_path(spec.field_path),
        )
        specs_by_signature.setdefault(signature, []).append(spec)

    candidate_owners: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for owner_path, _field_path in specs_by_signature:
        if owner_path in candidate_owners:
            continue
        candidates = sorted(
            {
                candidate
                for answer_path in canonical_fields
                for candidate in [answer_owner_prefix(answer_path, owner_path)]
                if candidate is not None
            },
            key=lambda path: (
                tuple(int(part) for part in path if part.isdecimal()),
                path,
            ),
        )
        candidate_owners[owner_path] = candidates

    actual_owner_by_widget: dict[str, tuple[str, ...]] = {}
    for (owner_path, _field_path), specs in specs_by_signature.items():
        candidates = candidate_owners[owner_path]
        ordered_specs = sorted(
            specs, key=lambda spec: (spec.bbox[1], spec.bbox[0], spec.id)
        )
        if len(candidates) == len(ordered_specs):
            actual_owner_by_widget.update(
                (spec.id, candidate)
                for spec, candidate in zip(ordered_specs, candidates)
            )
        elif len(candidates) == 1:
            actual_owner_by_widget.update(
                (spec.id, candidates[0]) for spec in ordered_specs
            )
        elif len(candidates) > 1:
            actual_owner_by_widget.update(
                (spec.id, candidate)
                for spec, candidate in zip(ordered_specs, candidates)
            )

    actual_field_by_widget: dict[str, tuple[str, ...]] = {}
    member_paths_by_owner: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for spec in template.widget_specs:
        raw_owner = _canonical_path(spec.owner_path)
        actual_owner = actual_owner_by_widget.get(spec.id, raw_owner)
        raw_field = _canonical_path(spec.field_path)
        actual_field = (
            actual_owner + raw_field[len(raw_owner) :]
            if raw_field[: len(raw_owner)] == raw_owner
            else raw_field
        )
        actual_field_by_widget[spec.id] = actual_field
        member_paths_by_owner.setdefault(actual_owner, []).append(actual_field)

    def values_under(path: tuple[str, ...]) -> list[Any]:
        return [
            value
            for field_path, value in canonical_fields.items()
            if field_path[: len(path)] == path
        ]

    owner_member_evidence = {
        owner_path: any(
            field_path[: len(member_path)] == member_path
            for member_path in member_paths
            for field_path in canonical_fields
        )
        for owner_path, member_paths in member_paths_by_owner.items()
    }

    widgets_by_group: dict[str, list[Widget]] = {}
    group_types: dict[str, str] = {}
    unknown_count = 0
    state_sources = {"answer_presence": 0, "option_membership": 0, "unknown": 0}
    for spec in template.widget_specs:
        actual_owner = actual_owner_by_widget.get(
            spec.id, _canonical_path(spec.owner_path)
        )
        actual_field = actual_field_by_widget[spec.id]
        field_values = values_under(actual_field)
        owner_value = canonical_fields.get(actual_owner)
        state = "unknown"
        if spec.state_rule == "answer_presence" and field_values:
            state = "selected" if any(_truthy_answer(value) for value in field_values) else "unselected"
            state_sources["answer_presence"] += 1
        elif spec.state_rule == "option_membership" and field_values:
            state = "selected" if any(_truthy_answer(value) for value in field_values) else "unselected"
            state_sources["answer_presence"] += 1
        elif (
            spec.state_rule == "option_membership"
            and actual_owner in canonical_fields
            and spec.option_labels
        ):
            answer_tokens = {
                _canonical_token(part)
                for value in [owner_value]
                for part in (value if isinstance(value, list) else [value])
            }
            label_tokens = {_canonical_token(value) for value in spec.option_labels}
            state = "selected" if answer_tokens & label_tokens else "unselected"
            state_sources["option_membership"] += 1
        elif spec.state_rule == "option_membership" and owner_member_evidence.get(
            actual_owner, False
        ):
            state = "unselected"
            state_sources["option_membership"] += 1
        else:
            unknown_count += 1
            state_sources["unknown"] += 1
        widgets_by_group.setdefault(spec.group_id, []).append(
            Widget(spec.id, spec.bbox, spec.widget_type, state)
        )
        group_types.setdefault(spec.group_id, spec.group_type)
    groups = tuple(
        WidgetGroup(group_id, group_types[group_id], tuple(members))
        for group_id, members in sorted(widgets_by_group.items())
    )
    return groups, unknown_count, state_sources


def _prediction_widget_state(item: Mapping[str, Any], widget_type: str) -> str:
    explicit = _canonical_token(item.get("state"))
    if explicit:
        return explicit
    if widget_type in {"checkbox", "radio"}:
        selected = item.get("selected", item.get("checked"))
        if isinstance(selected, bool):
            return "selected" if selected else "unselected"
        mark = _canonical_token(item.get("mark"))
        if mark:
            return "unselected" if mark in {"blank", "unchecked", "unselected"} else "selected"
        return "unknown"
    if isinstance(item.get("filled"), bool):
        return "filled" if item["filled"] else "blank"
    if isinstance(item.get("blank"), bool):
        return "blank" if item["blank"] else "filled"
    selected = item.get("selected")
    if isinstance(selected, bool):
        return "filled" if selected else "blank"
    return "filled" if _truthy_answer(item.get("value") or item.get("text")) else "unknown"


def _prediction_widget_groups(
    prediction: Mapping[str, Any], source_space: str, width: float, height: float
) -> tuple[tuple[WidgetGroup, ...], int]:
    raw_widgets = prediction.get("widgets") if isinstance(prediction.get("widgets"), list) else []
    widgets: list[Widget] = []
    widget_by_id: dict[str, Widget] = {}
    for index, item in enumerate(raw_widgets):
        if not isinstance(item, dict):
            widget = Widget(f"pred-widget-{index}", None, "unknown", "unknown")
        else:
            widget_id = str(item.get("id") or f"pred-widget-{index}")
            if widget_id in widget_by_id:
                widget_id = f"{widget_id}#{index}"
            widget_type = _widget_type(item)
            widget = Widget(
                widget_id,
                _normalized_box(item, source_space, width, height),
                widget_type,
                _prediction_widget_state(item, widget_type),
            )
        widgets.append(widget)
        widget_by_id[widget.id] = widget

    raw_groups = prediction.get("widget_groups") if isinstance(prediction.get("widget_groups"), list) else []
    groups: list[WidgetGroup] = []
    consumed: set[str] = set()
    for index, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            groups.append(WidgetGroup(f"pred-widget-group-{index}", "unknown", ()))
            continue
        raw_members = item.get("members") or item.get("widgets") or item.get("items") or []
        members: list[Widget] = []
        for member_index, raw_member in enumerate(raw_members if isinstance(raw_members, list) else []):
            if isinstance(raw_member, str):
                member = widget_by_id.get(raw_member)
            elif isinstance(raw_member, dict):
                member_id = str(raw_member.get("id") or f"pred-group-widget-{index}-{member_index}")
                member_type = _widget_type(raw_member)
                member = Widget(
                    member_id,
                    _normalized_box(raw_member, source_space, width, height),
                    member_type,
                    _prediction_widget_state(raw_member, member_type),
                )
            else:
                member = None
            if member is not None and member.id not in {existing.id for existing in members}:
                members.append(member)
                consumed.add(member.id)
        group_type = _canonical_token(item.get("group_type") or item.get("type") or item.get("tau"))
        if not group_type and members:
            member_types = {member.widget_type for member in members}
            group_type = next(iter(member_types)) if len(member_types) == 1 else "mixed"
        groups.append(
            WidgetGroup(
                str(item.get("id") or f"pred-widget-group-{index}"),
                group_type or "unknown",
                tuple(members),
            )
        )
    for widget in widgets:
        if widget.id not in consumed:
            groups.append(
                WidgetGroup(f"singleton:{widget.id}", widget.widget_type, (widget,))
            )
    return tuple(groups), len(raw_widgets)


def _prediction_grids(
    prediction: Mapping[str, Any], source_space: str, width: float, height: float
) -> tuple[tuple[LocalGrid, ...], Mapping[str, int]]:
    raw_grids = (
        prediction.get("local_grids")
        if isinstance(prediction.get("local_grids"), list)
        else []
    )
    grids: list[LocalGrid] = []
    audit: Counter[str] = Counter()
    used_grid_ids: set[str] = set()
    used_cell_ids: set[str] = set()
    for grid_index, raw_grid in enumerate(raw_grids):
        if not isinstance(raw_grid, dict):
            audit["adapter_grid_rejected_items"] += 1
            continue
        raw_cells = raw_grid.get("cells")
        if not isinstance(raw_cells, list) or not raw_cells:
            # A flattened cell or arbitrary object placed in local_grids is not
            # a two-dimensional grid prediction.
            audit["adapter_grid_rejected_items"] += 1
            continue
        grid_id = str(raw_grid.get("id") or f"pred-grid-{grid_index}")
        if grid_id in used_grid_ids:
            grid_id = f"{grid_id}#{grid_index}"
        used_grid_ids.add(grid_id)
        parent_id = str(
            raw_grid.get("region_id")
            or raw_grid.get("parent_region_id")
            or raw_grid.get("parent_id")
            or raw_grid.get("parent")
            or ""
        )
        cells: list[GridCell] = []
        for cell_index, raw_cell in enumerate(raw_cells):
            if not isinstance(raw_cell, dict):
                cells.append(GridCell(f"{grid_id}:cell:{cell_index}", -1, -1, -1, -1, None))
                continue
            cell_id = str(raw_cell.get("id") or f"{grid_id}:cell:{cell_index}")
            if cell_id in used_cell_ids:
                cell_id = f"{grid_id}/{cell_id}"
            used_cell_ids.add(cell_id)
            row = _parse_index(
                raw_cell.get(
                    "row",
                    raw_cell.get("row_index", raw_cell.get("row_start")),
                )
            )
            col = _parse_index(
                raw_cell.get(
                    "col",
                    raw_cell.get(
                        "column",
                        raw_cell.get("col_index", raw_cell.get("col_start")),
                    ),
                )
            )
            rowspan = _strict_positive_int(raw_cell.get("rowspan"))
            colspan = _strict_positive_int(raw_cell.get("colspan"))
            if raw_cell.get("rowspan") is None and raw_cell.get("row_end") is not None:
                row_end = _parse_index(raw_cell.get("row_end"))
                rowspan = row_end - row + 1 if row >= 0 and row_end >= row else -1
            if raw_cell.get("colspan") is None and raw_cell.get("col_end") is not None:
                col_end = _parse_index(raw_cell.get("col_end"))
                colspan = col_end - col + 1 if col >= 0 and col_end >= col else -1
            cells.append(
                GridCell(
                    cell_id,
                    row,
                    col,
                    rowspan,
                    colspan,
                    _normalized_box(raw_cell, source_space, width, height),
                )
            )
        if cells and all(cell.row >= 0 and cell.col >= 0 for cell in cells):
            min_row = min(cell.row for cell in cells)
            min_col = min(cell.col for cell in cells)
            if min_row or min_col:
                cells = [
                    replace(cell, row=cell.row - min_row, col=cell.col - min_col)
                    for cell in cells
                ]
                audit["adapter_grid_index_offsets_normalized"] += 1
        grid_bbox = _normalized_box(raw_grid, source_space, width, height)
        if grid_bbox is None:
            boxes = [cell.bbox for cell in cells if cell.bbox is not None]
            if boxes and len(boxes) == len(cells):
                grid_bbox = (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                )
            elif boxes:
                audit["adapter_grid_partial_bbox_union_blocked"] += 1
        grids.append(LocalGrid(grid_id, parent_id, tuple(cells), grid_bbox))
    return tuple(grids), dict(audit)


def _prediction_line_item_groups(
    prediction: Mapping[str, Any],
    raw_regions: Sequence[Any],
    grids: Sequence[LocalGrid],
    source_space: str,
    width: float,
    height: float,
) -> tuple[BoxNode, ...]:
    candidates: list[BoxNode] = []
    raw_groups = (
        prediction.get("line_item_groups")
        if isinstance(prediction.get("line_item_groups"), list)
        else []
    )
    for index, item in enumerate(raw_groups):
        box = _normalized_box(item, source_space, width, height)
        if box is None:
            continue
        item_id = (
            str(item.get("id") or item.get("line_item_group_id") or f"pred-line-item-{index}")
            if isinstance(item, dict)
            else f"pred-line-item-{index}"
        )
        candidates.append(BoxNode(item_id, box))

    for index, item in enumerate(raw_regions):
        if not isinstance(item, dict) or _raw_type(item) not in LIG_REGION_TYPES:
            continue
        box = _normalized_box(item, source_space, width, height)
        if box is not None:
            candidates.append(
                BoxNode(str(item.get("id") or f"pred-lig-region-{index}"), box)
            )

    candidates.extend(
        BoxNode(grid.id, grid.bbox) for grid in grids if grid.bbox is not None
    )
    unique: list[BoxNode] = []
    seen: set[tuple[float, float, float, float]] = set()
    for candidate in candidates:
        key = tuple(round(value, 8) for value in candidate.bbox)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def parse_prediction(
    prediction: Any,
    source_space: str,
    template: TemplateStructure,
) -> PredictionStructure:
    prediction, adapter_audit = adapt_legacy_prediction(prediction)
    regions: list[RegionNode] = []
    raw_regions = prediction.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raw_regions = prediction.get("region_boxes") if isinstance(prediction.get("region_boxes"), list) else []
    region_dropped = 0
    for index, item in enumerate(raw_regions):
        if not isinstance(item, dict):
            region_dropped += 1
            continue
        box = _normalized_box(item, source_space, template.width, template.height)
        if box is None:
            region_dropped += 1
            continue
        raw_type = _raw_type(item)
        regions.append(
            RegionNode(
                str(item.get("id") or f"pred-region-{index}"),
                PRED_TYPE_MAP.get(raw_type, "unknown"),
                box,
                raw_type,
            )
        )

    widget_groups, n_widget_declared = _prediction_widget_groups(
        prediction, source_space, template.width, template.height
    )
    grids, grid_audit = _prediction_grids(
        prediction, source_space, template.width, template.height
    )
    line_groups = _prediction_line_item_groups(
        prediction,
        raw_regions,
        grids,
        source_space,
        template.width,
        template.height,
    )

    raw_relations = prediction.get("relations")
    if not isinstance(raw_relations, list):
        raw_relations = []
    relations = tuple(
        _relation_from_item(item, index)
        for index, item in enumerate(raw_relations if isinstance(raw_relations, list) else [])
        if isinstance(item, dict)
    )
    answer = prediction.get("answer") if isinstance(prediction.get("answer"), (dict, list)) else {}
    answer_paths = tuple(flatten_leaf_fields(answer).keys())
    return PredictionStructure(
        tuple(regions),
        grids,
        widget_groups,
        _unique_rows(relations),
        line_groups,
        answer_paths,
        {
            **adapter_audit,
            **grid_audit,
            "region_declared": len(raw_regions),
            "region_dropped": region_dropped,
            "widget_declared": n_widget_declared,
            "grid_declared": len(grids),
            "relation_declared": len(raw_relations) if isinstance(raw_relations, list) else 0,
        },
    )


def _bbox_iou_matrix(left_boxes: Sequence[BBox], right_boxes: Sequence[BBox]) -> np.ndarray:
    if not left_boxes or not right_boxes:
        return np.zeros((len(left_boxes), len(right_boxes)), dtype=np.float64)
    left = np.asarray(left_boxes, dtype=np.float64)
    right = np.asarray(right_boxes, dtype=np.float64)
    ix1 = np.maximum(left[:, None, 0], right[None, :, 0])
    iy1 = np.maximum(left[:, None, 1], right[None, :, 1])
    ix2 = np.minimum(left[:, None, 2], right[None, :, 2])
    iy2 = np.minimum(left[:, None, 3], right[None, :, 3])
    intersection = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    left_area = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
    right_area = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_regions(
    pred_regions: Sequence[RegionNode], gt_regions: Sequence[RegionNode]
) -> dict[str, str]:
    weights = _bbox_iou_matrix(
        [region.bbox for region in pred_regions],
        [region.bbox for region in gt_regions],
    )
    compatible = np.asarray(
        [
            [gt.category in PRED_COMPATIBILITY.get(pred.category, set()) for gt in gt_regions]
            for pred in pred_regions
        ],
        dtype=bool,
    ).reshape(weights.shape)
    eligible = (weights >= 0.5) & compatible
    return {
        pred_regions[pred_index].id: gt_regions[gt_index].id
        for pred_index, gt_index in maximum_weight_matching(
            weights, eligible, cardinality_first=True
        )
    }


def _valid_grid_topology(grid: LocalGrid) -> bool:
    if not grid.cells or any(
        cell.row < 0
        or cell.col < 0
        or cell.rowspan < 1
        or cell.colspan < 1
        for cell in grid.cells
    ):
        return False
    n_rows = max(cell.row + cell.rowspan for cell in grid.cells)
    n_cols = max(cell.col + cell.colspan for cell in grid.cells)
    if sum(cell.rowspan * cell.colspan for cell in grid.cells) != n_rows * n_cols:
        return False
    for left_index, left in enumerate(grid.cells):
        for right in grid.cells[left_index + 1 :]:
            rows_overlap = max(left.row, right.row) < min(
                left.row + left.rowspan, right.row + right.rowspan
            )
            cols_overlap = max(left.col, right.col) < min(
                left.col + left.colspan, right.col + right.colspan
            )
            if rows_overlap and cols_overlap:
                return False
    return True


def _box_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def recover_grid_parents(
    prediction: PredictionStructure,
    region_mapping: Mapping[str, str],
) -> tuple[PredictionStructure, int]:
    """Recover only uniquely implied legacy parents after R-F1 is frozen."""
    region_ids = {region.id for region in prediction.regions}
    recovered = 0
    grids: list[LocalGrid] = []
    for grid in prediction.grids:
        if grid.parent_region_id and grid.parent_region_id in region_ids:
            grids.append(grid)
            continue
        if not _valid_grid_topology(grid) or any(
            cell.bbox is None for cell in grid.cells
        ):
            grids.append(grid)
            continue
        cell_boxes = [cell.bbox for cell in grid.cells if cell.bbox is not None]
        cell_union = _bbox_union(cell_boxes)
        candidates = [
            region
            for region in prediction.regions
            if region.id in region_mapping and _box_contains(region.bbox, cell_union)
        ]
        if len(candidates) != 1:
            grids.append(grid)
            continue
        grids.append(
            replace(
                grid,
                parent_region_id=candidates[0].id,
                bbox=grid.bbox or cell_union,
            )
        )
        recovered += 1
    if not recovered:
        return prediction, 0
    audit = dict(prediction.audit)
    audit["adapter_grid_parent_inferred"] = (
        int(audit.get("adapter_grid_parent_inferred", 0)) + recovered
    )
    return replace(prediction, grids=tuple(grids), audit=audit), recovered


def match_box_nodes(pred_nodes: Sequence[BoxNode], gt_nodes: Sequence[BoxNode]) -> dict[str, str]:
    weights = _bbox_iou_matrix(
        [node.bbox for node in pred_nodes], [node.bbox for node in gt_nodes]
    )
    eligible = weights >= 0.5
    return {
        pred_nodes[pred_index].id: gt_nodes[gt_index].id
        for pred_index, gt_index in maximum_weight_matching(
            weights, eligible, cardinality_first=True
        )
    }


def _field_endpoint_mapping(
    prediction: PredictionStructure,
    template: TemplateStructure,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in prediction.answer_paths:
        canonical = _canonical_path(path)
        gt_id = template.field_paths.get(canonical)
        if gt_id is None:
            continue
        aliases = {
            "/".join(str(part).strip() for part in path),
            ".".join(str(part).strip() for part in path),
        }
        for alias in aliases:
            mapping[alias] = gt_id

    gt_aliases: dict[str, str | None] = {}
    for path, gt_id in template.field_paths.items():
        for alias in ("/".join(path), ".".join(path)):
            gt_aliases[alias] = gt_id if alias not in gt_aliases else None
    relation_endpoints = {
        endpoint
        for relation in prediction.relations
        for endpoint in (relation.source, relation.target)
    }
    for endpoint in relation_endpoints:
        canonical = _canonical_token(endpoint.replace("__", "/"))
        if gt_aliases.get(canonical):
            mapping[endpoint] = str(gt_aliases[canonical])
    return mapping


def _empty_prediction() -> PredictionStructure:
    return PredictionStructure((), (), (), (), (), (), {})


def _format(value: Any) -> str:
    if value == NA or value is None:
        return NA
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _sum_counts(left: RelationCounts, right: RelationCounts) -> RelationCounts:
    return RelationCounts(left.tp + right.tp, left.pred + right.pred, left.gt + right.gt)


def _add_endpoint_mapping(
    destination: dict[str, str],
    namespace: str,
    mapping: Mapping[str, str],
) -> None:
    destination.update(mapping)
    destination.update(
        {f"{namespace}.{pred_id}": gt_id for pred_id, gt_id in mapping.items()}
    )


def evaluate_run(
    spec: RunSpec,
    samples: Sequence[EvaluationSample],
    pred_root: Path,
    manifest: Mapping[str, Any],
    symmetric_types: set[str],
    per_sample_dir: Path | None,
) -> tuple[dict[str, Any], Mapping[str, RelationCounts], Mapping[str, int]]:
    source_space, inherited_from = resolve_bbox_space(dict(manifest), spec.model)
    model_dir = pred_root / spec.model
    has_prediction_dir = model_dir.is_dir()
    n_valid = 0
    n_missing = 0
    n_invalid = 0
    lg_values: list[float | str] = []
    wg_values: list[float | str] = []
    rel_values: list[float | str] = []
    rel_conditional_values: list[float | str] = []
    lg_similarity_sum = 0.0
    lg_pred_total = 0
    lg_gt_total = 0
    lg_match_total = 0
    wg_similarity_sum = 0.0
    wg_pred_total = 0
    wg_gt_total = 0
    widget_unknown_total = 0
    rel_total = RelationCounts(0, 0, 0)
    rel_conditional_total = RelationCounts(0, 0, 0)
    relation_type_totals: dict[str, RelationCounts] = {}
    n_lg_gt_applicable = 0
    n_wg_gt_applicable = 0
    n_rel_gt_applicable = 0
    state_source_totals = {"answer_presence": 0, "option_membership": 0, "unknown": 0}
    adapter_totals: Counter[str] = Counter()
    per_sample_rows: list[dict[str, Any]] = []

    if spec.n_total == len(samples):
        run_samples = list(samples)
        sample_scope = "full_index"
    elif spec.n_total < len(samples):
        sample_ids = {sample.sample_id for sample in samples}
        prediction_ids = (
            {path.stem for path in model_dir.glob("*.json")} & sample_ids
            if has_prediction_dir
            else set()
        )
        if len(prediction_ids) == spec.n_total:
            run_samples = [
                sample for sample in samples if sample.sample_id in prediction_ids
            ]
            sample_scope = "prediction_files"
        else:
            run_samples = []
            sample_scope = "unrecoverable_partial"
            n_missing = spec.n_total
    else:
        run_samples = []
        sample_scope = "invalid_oversized_scope"
        n_missing = spec.n_total
    for sample in run_samples:
        prediction = _empty_prediction()
        valid_json = False
        pred_path = model_dir / f"{sample.sample_id}.json"
        if not has_prediction_dir or not pred_path.exists():
            n_missing += 1
        else:
            try:
                raw_prediction = read_json(pred_path)
                prediction = parse_prediction(raw_prediction, source_space, sample.template)
                valid_json = True
                n_valid += 1
            except Exception:
                n_invalid += 1

        gt_widget_groups = sample.widget_groups
        unknown_states = sample.widget_unknown_states
        for key, value in sample.widget_state_sources.items():
            state_source_totals[key] += value

        region_mapping = match_regions(prediction.regions, sample.template.regions)
        prediction, _n_parents_recovered = recover_grid_parents(
            prediction, region_mapping
        )
        adapter_totals.update(
            {
                key: int(value)
                for key, value in prediction.audit.items()
                if key.startswith("adapter_")
            }
        )
        lg_score = lg_grits_top(prediction.grids, sample.template.grids, region_mapping)
        wg_score = widget_group_f1(prediction.widget_groups, gt_widget_groups)

        pred_widgets = flatten_widgets(prediction.widget_groups)
        gt_widgets = flatten_widgets(gt_widget_groups)
        endpoint_mapping: dict[str, str] = {}
        _add_endpoint_mapping(
            endpoint_mapping,
            "regions",
            {
                pred_id: sample.template.region_relation_ids.get(gt_id, gt_id)
                for pred_id, gt_id in region_mapping.items()
            },
        )
        _add_endpoint_mapping(
            endpoint_mapping,
            "widgets",
            match_widgets_for_relations(pred_widgets, gt_widgets),
        )
        line_item_mapping = match_box_nodes(
            prediction.line_item_groups, sample.template.line_item_groups
        )
        _add_endpoint_mapping(
            endpoint_mapping,
            "line_item_groups",
            {
                pred_id: sample.template.line_item_relation_ids.get(gt_id, gt_id)
                for pred_id, gt_id in line_item_mapping.items()
            },
        )
        for pred_id, gt_id in _field_endpoint_mapping(
            prediction, sample.template
        ).items():
            endpoint_mapping.setdefault(pred_id, gt_id)
        grid_endpoint_mapping: dict[str, str] = {}
        for pred_index, gt_index, _similarity in lg_score.matches:
            grid_endpoint_mapping[prediction.grids[pred_index].id] = (
                sample.template.grids[gt_index].id
            )
        _add_endpoint_mapping(endpoint_mapping, "local_grids", grid_endpoint_mapping)
        _add_endpoint_mapping(
            endpoint_mapping, "cells", lg_score.cell_mapping
        )

        rel_score = relation_f1(
            prediction.relations,
            sample.template.relations,
            endpoint_mapping,
            symmetric_types=symmetric_types,
        )
        lg_values.append(lg_score.score)
        wg_values.append(wg_score.score)
        rel_values.append(rel_score.counts.f1)
        rel_conditional_values.append(rel_score.matched_endpoint_counts.f1)
        n_lg_gt_applicable += bool(sample.template.grids)
        n_wg_gt_applicable += bool(gt_widget_groups)
        n_rel_gt_applicable += bool(sample.template.relations)
        lg_similarity_sum += lg_score.similarity_sum
        lg_pred_total += lg_score.n_pred
        lg_gt_total += lg_score.n_gt
        lg_match_total += len(lg_score.matches)
        wg_similarity_sum += wg_score.similarity_sum
        wg_pred_total += wg_score.n_pred
        wg_gt_total += wg_score.n_gt
        widget_unknown_total += unknown_states
        rel_total = _sum_counts(rel_total, rel_score.counts)
        rel_conditional_total = _sum_counts(
            rel_conditional_total, rel_score.matched_endpoint_counts
        )
        for relation_type, counts in rel_score.by_type.items():
            relation_type_totals[relation_type] = _sum_counts(
                relation_type_totals.get(relation_type, RelationCounts(0, 0, 0)), counts
            )

        if per_sample_dir is not None:
            per_sample_rows.append(
                {
                    "model": spec.model,
                    "sample_id": sample.sample_id,
                    "template_name": sample.template_name,
                    "valid_json": valid_json,
                    "LG-GriTS-Top": lg_score.score,
                    "LG-similarity-sum": lg_score.similarity_sum,
                    "LG-pred": lg_score.n_pred,
                    "LG-gt": lg_score.n_gt,
                    "LG-matches": len(lg_score.matches),
                    "WG-F1": wg_score.score,
                    "WG-similarity-sum": wg_score.similarity_sum,
                    "WG-pred": wg_score.n_pred,
                    "WG-gt": wg_score.n_gt,
                    "WG-unknown-gt-state": unknown_states,
                    "Rel-F1": rel_score.counts.f1,
                    "Rel-F1-matched-endpoints": rel_score.matched_endpoint_counts.f1,
                    "Rel-TP": rel_score.counts.tp,
                    "Rel-pred": rel_score.counts.pred,
                    "Rel-gt": rel_score.counts.gt,
                    "Rel-matched-endpoint-pred": rel_score.matched_endpoint_counts.pred,
                    "Rel-matched-endpoint-gt": rel_score.matched_endpoint_counts.gt,
                    "adapter-relation-accepted": prediction.audit.get(
                        "adapter_relation_accepted_items", 0
                    ),
                    "adapter-relation-rejected": prediction.audit.get(
                        "adapter_relation_rejected_items", 0
                    ),
                    "adapter-grid-parent-inferred": prediction.audit.get(
                        "adapter_grid_parent_inferred", 0
                    ),
                }
            )

    if per_sample_dir is not None:
        write_jsonl(per_sample_dir / f"{spec.model}.jsonl", per_sample_rows)

    full_scope = sample_scope == "full_index"
    run_type = classify_run_type(spec.model)
    row = {
        "model": spec.model,
        "model_id": spec.model_id,
        "group": spec.group,
        "run_type": run_type,
        "comparison_status": classify_comparison_status(
            run_type=run_type,
            has_prediction_dir=has_prediction_dir,
            full_scope=full_scope,
            n_valid_json=n_valid,
            n_indexed=len(samples),
        ),
        "bbox_source_space": source_space,
        "bbox_space_inherited_from": inherited_from,
        "sample_scope": sample_scope,
        "n_total": spec.n_total,
        "n_valid_json": n_valid,
        "coverage": n_valid / spec.n_total if spec.n_total else NA,
        "n_missing_prediction": n_missing,
        "n_invalid_json": n_invalid,
        "n_lg_gt_applicable": n_lg_gt_applicable,
        "n_lg_scored": sum(isinstance(value, (int, float)) for value in lg_values),
        "LG-GriTS-Top": numeric_mean(lg_values),
        "LG-GriTS-Top-corpus": (
            2.0 * lg_similarity_sum / (lg_pred_total + lg_gt_total)
            if lg_pred_total + lg_gt_total
            else NA
        ),
        "n_grid_pred": lg_pred_total,
        "n_grid_gt": lg_gt_total,
        "n_grid_matches": lg_match_total,
        "n_wg_gt_applicable": n_wg_gt_applicable,
        "n_wg_scored": sum(isinstance(value, (int, float)) for value in wg_values),
        "WG-F1": numeric_mean(wg_values),
        "WG-F1-corpus": (
            2.0 * wg_similarity_sum / (wg_pred_total + wg_gt_total)
            if wg_pred_total + wg_gt_total
            else NA
        ),
        "n_widget_group_pred": wg_pred_total,
        "n_widget_group_gt": wg_gt_total,
        "n_widget_gt_unknown_state": widget_unknown_total,
        "n_rel_gt_applicable": n_rel_gt_applicable,
        "n_rel_scored": sum(isinstance(value, (int, float)) for value in rel_values),
        "Rel-F1": numeric_mean(rel_values),
        "Rel-Precision-micro": rel_total.precision,
        "Rel-Recall-micro": rel_total.recall,
        "Rel-F1-micro": rel_total.f1,
        "Rel-F1-matched-endpoints": numeric_mean(rel_conditional_values),
        "Rel-F1-matched-endpoints-micro": rel_conditional_total.f1,
        "n_relation_tp": rel_total.tp,
        "n_relation_pred": rel_total.pred,
        "n_relation_gt": rel_total.gt,
        "adapter_relation_declared_items": adapter_totals[
            "adapter_relation_declared_items"
        ],
        "adapter_relation_accepted_items": adapter_totals[
            "adapter_relation_accepted_items"
        ],
        "adapter_relation_rejected_items": adapter_totals[
            "adapter_relation_rejected_items"
        ],
        "adapter_relation_endpoint_aliases": adapter_totals[
            "adapter_relation_endpoint_aliases"
        ],
        "adapter_relation_ambiguous_endpoints": adapter_totals[
            "adapter_relation_ambiguous_endpoints"
        ],
        "adapter_relation_type_aliases": adapter_totals[
            "adapter_relation_type_aliases"
        ],
        "adapter_relation_types_inferred": adapter_totals[
            "adapter_relation_types_inferred"
        ],
        "adapter_grid_cells_enriched": adapter_totals[
            "adapter_grid_cells_enriched"
        ],
        "adapter_grid_fragments_merged": adapter_totals[
            "adapter_grid_fragments_merged"
        ],
        "adapter_grid_index_offsets_normalized": adapter_totals[
            "adapter_grid_index_offsets_normalized"
        ],
        "adapter_grid_rejected_items": adapter_totals[
            "adapter_grid_rejected_items"
        ],
        "adapter_grid_parent_inferred": adapter_totals[
            "adapter_grid_parent_inferred"
        ],
        "adapter_widget_groups_recovered": (
            adapter_totals["adapter_widget_groups_from_fields"]
            + adapter_totals["adapter_widget_groups_from_member_ids"]
        ),
    }
    return row, relation_type_totals, state_source_totals


def write_results(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    relation_types: Mapping[str, Mapping[str, RelationCounts]],
    metadata: Mapping[str, Any],
) -> None:
    ensure_dir(out_dir)
    with (out_dir / "hierarchical_structure_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in RESULT_COLUMNS})

    relation_type_columns = [
        "model",
        "relation_type",
        "TP",
        "pred",
        "GT",
        "precision",
        "recall",
        "F1",
    ]
    with (out_dir / "hierarchical_relation_type_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=relation_type_columns)
        writer.writeheader()
        for row in rows:
            model = str(row["model"])
            type_counts = relation_types.get(model, {})
            for relation_type, counts in sorted(type_counts.items()):
                writer.writerow(
                    {
                        "model": model,
                        "relation_type": relation_type,
                        "TP": counts.tp,
                        "pred": counts.pred,
                        "GT": counts.gt,
                        "precision": _format(counts.precision),
                        "recall": _format(counts.recall),
                        "F1": _format(counts.f1),
                    }
                )
    write_json(out_dir / "hierarchical_structure_metrics_metadata.json", dict(metadata))

    lines = [
        "# Hierarchical Structure Metrics",
        "",
        "LG-GriTS-Top uses parent R-F1 matches, grid IoU/singleton eligibility, reference factored 2D-MSS GriTS_Top, and page-level Hungarian matching. WG-F1 uses strict type/state/IoU member matching followed by group-level Hungarian matching. Rel-F1 freezes all endpoint matches before scoring typed directed edges.",
        "",
        "| Run | Status | Valid/Total | LG-GriTS-Top | WG-F1 | Rel-F1 | Rel micro | Rel matched endpoints |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["comparison_status"]),
                    f"{row['n_valid_json']}/{row['n_total']}",
                    _format(row["LG-GriTS-Top"]),
                    _format(row["WG-F1"]),
                    _format(row["Rel-F1"]),
                    _format(row["Rel-F1-micro"]),
                    _format(row["Rel-F1-matched-endpoints"]),
                ]
            )
            + " |"
        )
    (out_dir / "hierarchical_structure_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LG-GriTS-Top, WG-F1, and fixed-endpoint Rel-F1."
    )
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--metadata-root", default="new-dataset-json")
    parser.add_argument("--layout-root", default="newdataset-layout")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--main-results", default="outputs/main_exp/main_results.csv")
    parser.add_argument("--bbox-manifest", default="configs/bbox_coordinate_spaces.json")
    parser.add_argument("--out", default="outputs/main_exp")
    parser.add_argument("--models", default="", help="Optional comma-separated run ids.")
    parser.add_argument(
        "--symmetric-relations",
        default="",
        help="Optional comma-separated relation types canonicalized as symmetric.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-per-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    index_path = Path(args.index)
    metadata_root = Path(args.metadata_root)
    layout_root = Path(args.layout_root)
    pred_root = Path(args.pred_root)
    main_results_path = Path(args.main_results)
    manifest_path = Path(args.bbox_manifest)
    samples = load_samples(index_path, metadata_root, layout_root)
    manifest = load_bbox_manifest(manifest_path)
    specs = load_run_specs(main_results_path, pred_root, len(samples))
    requested = {value.strip() for value in args.models.split(",") if value.strip()}
    if requested:
        available = {spec.model for spec in specs}
        missing = requested - available
        if missing:
            raise ValueError(f"requested run ids not found: {', '.join(sorted(missing))}")
        specs = [spec for spec in specs if spec.model in requested]
    for spec in specs:
        resolve_bbox_space(manifest, spec.model)
    symmetric_types = {
        value.strip() for value in args.symmetric_relations.split(",") if value.strip()
    }
    per_sample_dir = (
        None if args.skip_per_sample else Path(args.out) / "hierarchical_structure_per_sample"
    )

    by_model: dict[str, dict[str, Any]] = {}
    relation_types: dict[str, Mapping[str, RelationCounts]] = {}
    state_sources: dict[str, Mapping[str, int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_run,
                spec,
                samples,
                pred_root,
                manifest,
                symmetric_types,
                per_sample_dir,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            row, type_counts, run_state_sources = future.result()
            by_model[spec.model] = row
            relation_types[spec.model] = type_counts
            state_sources[spec.model] = run_state_sources
            print(
                f"[Hierarchical] {spec.model}: LG={_format(row['LG-GriTS-Top'])} "
                f"WG={_format(row['WG-F1'])} Rel={_format(row['Rel-F1'])} "
                f"valid={row['n_valid_json']}/{row['n_total']}",
                flush=True,
            )

    rows = [by_model[spec.model] for spec in specs]
    templates = {sample.template_name: sample.template for sample in samples}
    template_audit = {
        "n_templates": len(templates),
        "n_table_regions": sum(
            int(template.audit["table_regions"]) for template in templates.values()
        ),
        "n_reconstructable_local_grids": sum(
            int(template.audit["reconstructable_local_grids"])
            for template in templates.values()
        ),
        "n_table_regions_without_row_col_topology": sum(
            int(template.audit["table_regions_without_row_col_topology"])
            for template in templates.values()
        ),
        "n_metadata_observable_widgets": sum(
            int(template.audit["metadata_observable_widgets"])
            for template in templates.values()
        ),
        "n_derived_typed_relations": sum(
            int(template.audit["derived_typed_relations"])
            for template in templates.values()
        ),
        "n_r_f1_regions": sum(
            int(template.audit.get("r_f1_regions", len(template.regions)))
            for template in templates.values()
        ),
        "n_r_f1_line_item_groups": sum(
            int(template.audit.get("r_f1_line_item_groups", len(template.line_item_groups)))
            for template in templates.values()
        ),
        "n_implicit_empty_cells": sum(
            int(template.audit.get("implicit_empty_cells", 0))
            for template in templates.values()
        ),
        "n_invalid_overlapping_grids_excluded": sum(
            int(template.audit.get("invalid_overlapping_grids_excluded", 0))
            for template in templates.values()
        ),
        "n_invalid_incomplete_grids_excluded": sum(
            int(template.audit.get("invalid_incomplete_grids_excluded", 0))
            for template in templates.values()
        ),
        "n_grid_parents_without_r_f1_correspondence": sum(
            int(template.audit.get("grid_parents_without_r_f1_correspondence", 0))
            for template in templates.values()
        ),
        "n_ambiguous_field_schema_paths": sum(
            int(template.audit.get("ambiguous_field_schema_paths", 0))
            for template in templates.values()
        ),
        "n_self_widget_relations_ignored": sum(
            int(template.audit.get("self_widget_relations_ignored", 0))
            for template in templates.values()
        ),
        "by_template": {name: dict(template.audit) for name, template in sorted(templates.items())},
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_path": str(index_path),
        "metadata_root": str(metadata_root),
        "layout_root": str(layout_root),
        "pred_root": str(pred_root),
        "main_results_path": str(main_results_path),
        "bbox_manifest": str(manifest_path),
        "n_indexed": len(samples),
        "n_runs": len(rows),
        "symmetric_relation_types": sorted(symmetric_types),
        "empty_page_policy": "A page is NA only when both prediction and GT sets are empty. Prediction-only pages score zero and remain in page-macro aggregation.",
        "partial_scope_policy": "A run shorter than the evaluation index is scored only when its exact sample IDs can be recovered from prediction filenames; otherwise metrics are NA rather than assuming an index prefix.",
        "lg_metric": {
            "grid_gt": "Raw Label Studio cells provide topology via inclusive row_start/row_end/col_start/col_end spans; converted layout provides the corrected R-F1 parent universe and is the topology fallback only when raw metadata is absent.",
            "parent_mapping": "Frozen mapping over the exact corrected R-F1 GT region universe and deduplication policy: type/IoU>=0.5 maximum-cardinality, with IoU tie-break.",
            "pair_eligibility": "Matched parent and either grid bbox IoU>=0.5 or one grid on each matched parent.",
            "similarity": "Reference-compatible factored 2D-MSS GriTS_Top over relative span matrices.",
            "invalid_topology": "A localized incomplete-span cell excludes its entire local grid; overlapping grids are also excluded and audited.",
        },
        "wg_metric": {
            "scope": "All checkbox controls preserved in raw Label Studio metadata; converted layout is fallback only.",
            "groups": "Raw parent-to-widget relations define groups; controls without an explicit parent are singleton groups.",
            "state": "Resolved per instance from exact member paths or direct owner option values in answer.json; unresolved state is literal unknown and only matches predicted unknown.",
            "type": "data_type is authoritative; mark_type check/circle is an annotation style and does not rewrite widget type or state.",
            "strict_match": "IoU>=0.5, exact canonical widget type, exact state; no center-distance fallback.",
            "state_resolution_by_run": state_sources,
        },
        "relation_metric": {
            "gt": "Raw directed Label Studio edges with relation type derived deterministically from endpoint roles.",
            "prediction": "Only explicit predicted relation triples are used; malformed container items are rejected and there is no answer-tree relation fallback.",
            "endpoint_mapping": "Frozen corrected R-F1 region, state-agnostic widget, schema-path field, corrected LIG-F1, and LG grid/cell mappings; relations never optimize endpoints.",
            "exactness": "Relation type, direction, and endpoint IDs are exact; only explicitly configured symmetric relation types reorder endpoints.",
            "ambiguous_fields": "A duplicated normalized schema path is left unmapped unless another fixed endpoint matcher resolves it; first-occurrence guessing is forbidden.",
            "primary": "Page-macro Rel-F1",
            "appendix": "Corpus micro, relation-type micro, and Rel-F1 conditioned on matched endpoints.",
        },
        "prediction_adapter": {
            "scope": "In-memory normalization of legacy prediction syntax; source prediction JSON files are not changed.",
            "relation_fields": "Accept source/from/u/parent, target/to/v/child, and type/relation_type/r only when aliases are non-conflicting scalar values.",
            "relation_types": "Canonicalize documented legacy spellings (for example label-value and label_to_widget); infer a missing type only when the two declared prediction node roles uniquely determine the ontology type.",
            "endpoint_ids": "Accept exact IDs only when their prediction namespace is unique, plus unambiguous explicit namespaces such as cells.c1 and regions.r1; text/label equality and relation IDs are never endpoint fallbacks.",
            "grid_cells": "An exact nested/top-level cell ID may fill missing topology or bbox fields; standalone top-level cells never create a grid.",
            "split_grids": "Fragments whose region_id explicitly references another predicted grid are merged only when the combined cell spans form one complete non-overlapping matrix; fragment IDs remain relation aliases of the root grid.",
            "grid_indices": "Row and column indices are translated to a zero-based local origin because GriTS topology is invariant to a constant per-grid offset.",
            "grid_parent": "Recover a missing or dangling parent only for valid topology with complete cell bboxes when exactly one already R-F1-matched predicted region fully contains the cell union.",
            "widget_groups": "Use explicit widget_groups, legacy fields with explicit members, or explicit widget group_id; otherwise retain the required singleton normalization.",
            "gt_relation_leakage": False,
        },
        "template_audit": template_audit,
        "raw_predictions_modified": False,
    }
    write_results(Path(args.out), rows, relation_types, metadata)
    print(
        f"wrote hierarchical metrics -> {Path(args.out) / 'hierarchical_structure_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
