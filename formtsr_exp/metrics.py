from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json


NA = "NA"


@dataclass
class MatchStats:
    tp: int
    pred: int
    gt: int

    @property
    def precision(self) -> float | str:
        if self.pred == 0:
            return NA
        return self.tp / self.pred

    @property
    def recall(self) -> float | str:
        if self.gt == 0:
            return NA
        return self.tp / self.gt

    @property
    def f1(self) -> float | str:
        if self.pred == 0 and self.gt == 0:
            return NA
        if self.pred == 0 or self.gt == 0:
            return 0.0
        denom = self.pred + self.gt
        return (2 * self.tp / denom) if denom else NA


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k).strip(): normalize_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def unwrap_answer(pred: Any) -> Any:
    if isinstance(pred, dict) and "answer" in pred:
        return pred["answer"]
    return pred


def flatten_leaf_fields(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        fields: dict[tuple[str, ...], Any] = {}
        for key, child in value.items():
            fields.update(flatten_leaf_fields(child, prefix + (str(key).strip(),)))
        return fields
    if isinstance(value, list):
        fields: dict[tuple[str, ...], Any] = {}
        for index, child in enumerate(value):
            fields.update(flatten_leaf_fields(child, prefix + (str(index),)))
        return fields
    return {prefix: normalize_json(value)}


def field_accuracy(pred: Any, gt: Any) -> tuple[float | str, dict[str, Any]]:
    pred_fields = flatten_leaf_fields(unwrap_answer(pred))
    gt_fields = flatten_leaf_fields(gt)
    total = len(gt_fields)
    if total == 0:
        return NA, {"reason": "GT has no leaf fields"}
    correct = 0
    mismatches: list[dict[str, Any]] = []
    for path, gt_value in gt_fields.items():
        pred_value = pred_fields.get(path)
        if pred_value == gt_value:
            correct += 1
        elif len(mismatches) < 20:
            mismatches.append({"path": list(path), "gt": gt_value, "pred": pred_value})
    return correct / total, {"correct": correct, "total": total, "mismatches_sample": mismatches}


def _normalize_value_for_vacc(value: Any) -> str:
    normalized = normalize_json(value)
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
        return str(normalized)
    if isinstance(normalized, (dict, list)):
        text = json_dumps_compact(normalized)
    else:
        text = str(normalized)
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=\d)[\s,]+(?=\d)", "", text)
    return text.strip()


def value_accuracy(pred: Any, gt: Any) -> tuple[float | str, dict[str, Any]]:
    pred_fields = flatten_leaf_fields(unwrap_answer(pred))
    gt_fields = flatten_leaf_fields(gt)
    pred_values = Counter(
        token
        for token in (_normalize_value_for_vacc(value) for value in pred_fields.values())
        if token
    )
    total = 0
    correct = 0
    ignored_empty_gt = 0
    mismatches: list[dict[str, Any]] = []
    for path, gt_value in gt_fields.items():
        token = _normalize_value_for_vacc(gt_value)
        if not token:
            ignored_empty_gt += 1
            continue
        total += 1
        if pred_values[token] > 0:
            pred_values[token] -= 1
            correct += 1
        elif len(mismatches) < 20:
            mismatches.append({"path": list(path), "gt": gt_value, "normalized_gt": token})
    if total == 0:
        return NA, {"reason": "GT has no non-empty leaf values", "ignored_empty_gt": ignored_empty_gt}
    return correct / total, {
        "correct": correct,
        "total": total,
        "ignored_empty_gt": ignored_empty_gt,
        "pred_non_empty_values": sum(
            1 for value in pred_fields.values() if _normalize_value_for_vacc(value)
        ),
        "normalization": "path-independent multiset match over non-empty normalized leaf values",
        "mismatches_sample": mismatches,
    }


def exact_json_match(pred: Any, gt: Any) -> float:
    return 1.0 if normalize_json(unwrap_answer(pred)) == normalize_json(gt) else 0.0


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def bbox_from(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        for key in ("bbox", "box", "region_box", "bounds"):
            if key in value:
                return bbox_from(value[key])
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _region_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("region_type") or item.get("data_type") or "unknown").strip().lower()


def extract_regions(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    candidates = data.get("regions") or data.get("region_boxes")
    regions: list[dict[str, Any]] = []
    for item in as_list(candidates):
        if isinstance(item, dict) and bbox_from(item):
            regions.append(item)
    return regions


SELECTION_TYPES = {"checkbox", "check_box", "radio", "radio_button"}


def node_label(node: dict[str, Any], fallback: str = "") -> str:
    return str(node.get("original_label") or node.get("semantic_key") or node.get("title") or fallback).strip()


def node_id(prefix: str, path: tuple[str, ...]) -> str:
    cleaned = [part.replace(" ", "_").replace("/", "_") for part in path if part]
    return prefix + "_" + "__".join(cleaned) if cleaned else prefix


def normalize_metadata_layout(layout: Any) -> Any:
    if not isinstance(layout, dict):
        return layout
    if any(key in layout for key in ("regions", "widgets", "relations", "cells", "local_grids")) and not layout.get("fields"):
        return layout

    regions: list[dict[str, Any]] = []
    widgets: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    line_item_groups: list[dict[str, Any]] = []
    widget_answer_paths: set[tuple[str, ...]] = set()
    column_by_parent: dict[tuple[str, ...], dict[str, int]] = {}

    def add_region(item_id: str, item_type: str, bbox: Any, text: str = "") -> None:
        if bbox_from(bbox):
            regions.append({"id": item_id, "type": item_type, "bbox": bbox, "text": text})

    def add_relation(source: str, target: str, relation_type: str, source_label: str = "", target_label: str = "") -> None:
        if source and target:
            row = {"source": source, "target": target, "type": relation_type}
            if source_label:
                row["source_label"] = source_label
            if target_label:
                row["target_label"] = target_label
            relations.append(row)

    def add_widget_answer_path(path: tuple[str, ...]) -> None:
        if path:
            widget_answer_paths.add(path)

    def visit(node: dict[str, Any], path: tuple[str, ...], parent_id: str | None, parent_path: tuple[str, ...]) -> None:
        label = node_label(node, path[-1] if path else "field")
        current_path = path + (label or f"node{len(regions)}",)
        current_id = node_id("field", current_path)
        data_type = str(node.get("data_type") or "field").lower()
        region_type = "widget" if data_type in SELECTION_TYPES else ("field_group" if node.get("keys") else "field")
        add_region(current_id, region_type, node.get("bbox"), label)
        if parent_id:
            add_relation(parent_id, current_id, "parent-child", target_label=label)

        value = node.get("value")
        if isinstance(value, dict):
            value_id = node_id("value", current_path)
            add_region(value_id, str(value.get("data_type") or "value").lower(), value.get("bbox"), "")
            add_relation(current_id, value_id, "key-value", source_label=label, target_label="")
            row_id = value.get("row_id")
            if row_id is not None and bbox_from(value):
                parent_cols = column_by_parent.setdefault(parent_path, {})
                col = parent_cols.setdefault(current_id, len(parent_cols))
                cells.append({"id": value_id, "row": int(str(row_id)) - 1 if str(row_id).isdigit() else str(row_id), "col": col, "rowspan": 1, "colspan": 1, "bbox": value.get("bbox"), "text": label})

        for index, item in enumerate(as_list(node.get("values"))):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("data_type") or node.get("data_type") or "value").lower()
            item_id = node_id("value", current_path + (str(index),))
            add_region(item_id, item_type, item.get("bbox"), label)
            add_relation(current_id, item_id, "key-value", source_label=label, target_label="")
            if item_type in SELECTION_TYPES or item.get("mark") == "check":
                add_widget_answer_path(current_path)
                widget_id = node_id("widget", current_path + (str(index),))
                widgets.append({"id": widget_id, "type": item_type if item_type in SELECTION_TYPES else "checkbox", "bbox": item.get("bbox"), "label": label, "selected": item.get("mark") == "check"})
                add_relation(current_id, widget_id, "field-widget", source_label=label, target_label=label)
                add_relation(current_id, widget_id, "label-option", source_label=label, target_label=label)
            row_id = item.get("row_id")
            if row_id is not None and bbox_from(item):
                parent_cols = column_by_parent.setdefault(parent_path, {})
                col = parent_cols.setdefault(current_id, len(parent_cols))
                cells.append({"id": item_id, "row": int(str(row_id)) - 1 if str(row_id).isdigit() else str(row_id), "col": col, "rowspan": 1, "colspan": 1, "bbox": item.get("bbox"), "text": label})
                add_relation(item_id, f"row:{row_id}", "row-membership")
                add_relation(item_id, f"column:{col}", "column-membership")

        if data_type in SELECTION_TYPES:
            widgets.append({"id": node_id("widget", current_path), "type": data_type, "bbox": node.get("bbox"), "label": label, "selected": node.get("mark") == "check"})

        for child in as_list(node.get("keys")):
            if isinstance(child, dict):
                child_type = str(child.get("data_type") or "").lower()
                if child_type in SELECTION_TYPES or child.get("mark") == "check":
                    add_widget_answer_path(current_path)
                visit(child, current_path, current_id, current_path)

    for root in as_list(layout.get("fields")):
        if isinstance(root, dict):
            visit(root, (), None, ())

    metadata = layout.get("metadata")
    if isinstance(metadata, dict):
        for section in as_list(metadata.get("layout_structure", {}).get("sections")):
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or section.get("source_region_id") or f"section_{len(regions)}")
            add_region(section_id, str(section.get("region_type") or "section"), section.get("bbox"), str(section.get("title") or ""))
            for region in as_list(section.get("regions")):
                if isinstance(region, dict):
                    region_id = str(region.get("region_id") or region.get("source_region_id") or f"region_{len(regions)}")
                    add_region(region_id, str(region.get("region_type") or "region"), region.get("bbox"), str(region.get("title") or ""))
                    add_relation(section_id, region_id, "parent-child")
            for group in as_list(section.get("line_item_groups")):
                if isinstance(group, dict) and bbox_from(group):
                    group_id = str(group.get("line_item_group_id") or group.get("source_region_id") or f"line_item_group_{len(line_item_groups)}")
                    row = {
                        "id": group_id,
                        "type": str(group.get("region_type") or "line_item_group"),
                        "bbox": group.get("bbox"),
                        "text": str(group.get("title") or ""),
                        "member_counts": group.get("member_counts", {}),
                    }
                    line_item_groups.append(row)
                    add_relation(section_id, group_id, "section-line-item-group")

    normalized = dict(layout)
    normalized["regions"] = regions
    normalized["widgets"] = [w for w in widgets if bbox_from(w)]
    normalized["line_item_groups"] = line_item_groups
    normalized["relations"] = relations
    normalized["_widget_answer_paths"] = [list(path) for path in sorted(widget_answer_paths)]
    if cells:
        normalized["cells"] = cells
        normalized["local_grids"] = [{"id": "metadata_row_id_grid", "cells": cells}]
    normalized["_metadata_normalizer"] = {
        "source": "newdataset-layout fields/keys/value/values",
        "regions": len(regions),
        "widgets": len(normalized["widgets"]),
        "relations": len(relations),
        "cells": len(cells),
        "line_item_groups": len(line_item_groups),
        "widget_answer_paths": len(widget_answer_paths),
        "lig_note": "line_item_groups derived from metadata layout_structure sections",
    }
    return normalized


def answer_tree_relations(data: Any) -> list[dict[str, Any]]:
    answer = unwrap_answer(data)
    relations: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            parent = "/".join(path)
            for key, child in value.items():
                child_path = path + (str(key).strip(),)
                child_id = "/".join(child_path)
                if parent:
                    relations.append({"source": parent, "target": child_id, "type": "parent-child"})
                if not isinstance(child, (dict, list)):
                    relations.append({"source": child_id, "target": str(normalize_json(child)), "type": "key-value"})
                visit(child, child_path)
        elif isinstance(value, list):
            parent = "/".join(path)
            for index, child in enumerate(value):
                child_path = path + (str(index),)
                child_id = "/".join(child_path)
                if parent:
                    relations.append({"source": parent, "target": child_id, "type": "parent-child"})
                visit(child, child_path)

    visit(answer, ())
    return relations


def max_bipartite_matches(edges: list[tuple[int, int, float]], n_left: int, n_right: int) -> int:
    matched_right: dict[int, int] = {}
    graph: dict[int, list[int]] = {}
    for left, right, _score in sorted(edges, key=lambda item: item[2], reverse=True):
        graph.setdefault(left, []).append(right)

    def dfs(left: int, seen: set[int]) -> bool:
        for right in graph.get(left, []):
            if right in seen:
                continue
            seen.add(right)
            if right not in matched_right or dfs(matched_right[right], seen):
                matched_right[right] = left
                return True
        return False

    total = 0
    for left in range(n_left):
        if dfs(left, set()):
            total += 1
    return total


def region_f1(pred: Any, gt: Any, *, threshold: float = 0.5) -> tuple[float | str, dict[str, Any]]:
    pred_regions = extract_regions(pred)
    gt_regions = extract_regions(gt)
    if not gt_regions:
        return NA, {"reason": "GT has no explicit regions/region boxes"}
    edges: list[tuple[int, int, float]] = []
    for i, p in enumerate(pred_regions):
        pb = bbox_from(p)
        if not pb:
            continue
        for j, g in enumerate(gt_regions):
            gb = bbox_from(g)
            if not gb:
                continue
            score = iou(pb, gb)
            if score >= threshold and _region_type(p) == _region_type(g):
                edges.append((i, j, score))
    stats = MatchStats(max_bipartite_matches(edges, len(pred_regions), len(gt_regions)), len(pred_regions), len(gt_regions))
    return stats.f1, {"tp": stats.tp, "pred": stats.pred, "gt": stats.gt, "iou_threshold": threshold}


def _cell_signature(cell: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        cell.get("row", cell.get("row_index")),
        cell.get("col", cell.get("column", cell.get("col_index"))),
        cell.get("rowspan", 1),
        cell.get("colspan", 1),
    )


def extract_cells(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    cells = [item for item in as_list(data.get("cells")) if isinstance(item, dict)]
    for grid in as_list(data.get("local_grids")) + as_list(data.get("grids")):
        if isinstance(grid, dict):
            cells.extend(item for item in as_list(grid.get("cells")) if isinstance(item, dict))
    return cells


def topology_score(pred: Any, gt: Any) -> tuple[float | str, dict[str, Any]]:
    gt_cells = extract_cells(gt)
    pred_cells = extract_cells(pred)
    if not gt_cells:
        return NA, {"reason": "GT has no explicit local grid/cell topology"}
    gt_sigs = {_cell_signature(cell) for cell in gt_cells}
    pred_sigs = {_cell_signature(cell) for cell in pred_cells}
    if not gt_sigs:
        return NA, {"reason": "GT grid cells lack row/column topology fields"}
    tp = len(gt_sigs & pred_sigs)
    stats = MatchStats(tp, len(pred_sigs), len(gt_sigs))
    return stats.f1, {"tp": stats.tp, "pred": stats.pred, "gt": stats.gt, "method": "row/col/rowspan/colspan signature F1"}


def _bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def extract_line_item_groups(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    groups: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()

    def add_group(item: dict[str, Any], source: str) -> None:
        box = bbox_from(item)
        if not box:
            return
        key = tuple(round(coord, 3) for coord in box)
        if key in seen:
            return
        seen.add(key)
        row = dict(item)
        row["bbox"] = [box[0], box[1], box[2], box[3]]
        row["_source"] = source
        groups.append(row)

    for item in as_list(data.get("line_item_groups")):
        if isinstance(item, dict):
            add_group(item, "line_item_groups")

    for region in extract_regions(data):
        region_type = _region_type(region)
        if region_type in {"line_item_group", "line-item-group", "lineitemgroup", "line_item"}:
            add_group(region, "regions")

    for grid in as_list(data.get("local_grids")) + as_list(data.get("grids")):
        if not isinstance(grid, dict):
            continue
        if bbox_from(grid):
            add_group(grid, "local_grids")
            continue
        cell_boxes = [box for cell in as_list(grid.get("cells")) if isinstance(cell, dict) for box in [bbox_from(cell)] if box]
        union_box = _bbox_union(cell_boxes)
        if union_box:
            row = dict(grid)
            row["bbox"] = [union_box[0], union_box[1], union_box[2], union_box[3]]
            add_group(row, "local_grids.cells_union")
    return groups


def line_item_group_f1(pred: Any, gt: Any, *, threshold: float = 0.5) -> tuple[float | str, dict[str, Any]]:
    gt_groups = extract_line_item_groups(gt)
    pred_groups = extract_line_item_groups(pred)
    if not gt_groups:
        return NA, {"reason": "GT has no line_item_groups"}
    edges: list[tuple[int, int, float]] = []
    for i, pred_group in enumerate(pred_groups):
        pb = bbox_from(pred_group)
        if not pb:
            continue
        for j, gt_group in enumerate(gt_groups):
            gb = bbox_from(gt_group)
            if not gb:
                continue
            score = iou(pb, gb)
            if score >= threshold:
                edges.append((i, j, score))
    stats = MatchStats(max_bipartite_matches(edges, len(pred_groups), len(gt_groups)), len(pred_groups), len(gt_groups))
    return stats.f1, {
        "tp": stats.tp,
        "pred": stats.pred,
        "gt": stats.gt,
        "iou_threshold": threshold,
        "method": "line_item_group bbox IoU bipartite matching",
    }


def _widget_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("data_type") or "unknown").lower()


def _widget_selected(item: dict[str, Any]) -> bool:
    return bool(item.get("selected", item.get("checked", item.get("mark") == "check")))


def _bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_diag(box: tuple[float, float, float, float]) -> float:
    return max(1.0, ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5)


def _widget_match_score(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    if _widget_type(pred) != _widget_type(gt):
        return 0.0
    if _widget_selected(pred) != _widget_selected(gt):
        return 0.0
    pb = bbox_from(pred)
    gb = bbox_from(gt)
    if not pb or not gb:
        return 0.0
    overlap = iou(pb, gb)
    pcx, pcy = _bbox_center(pb)
    gcx, gcy = _bbox_center(gb)
    center_dist = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5
    # Checkbox/radio boxes are small; VLM boxes are often shifted. Accept either IoU or nearby centers.
    center_ok = center_dist <= max(35.0, 1.5 * _bbox_diag(gb))
    if overlap >= 0.1 or center_ok:
        return max(overlap, 1.0 - min(center_dist / max(35.0, 1.5 * _bbox_diag(gb)), 1.0))
    return 0.0


def extract_widgets(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    widgets = [item for item in as_list(data.get("widgets")) if isinstance(item, dict)]
    groups = data.get("widget_groups") or data.get("fields")
    for item in as_list(groups):
        if isinstance(item, dict) and (item.get("type") in {"checkbox", "radio"} or item.get("data_type") in {"checkbox", "radio"}):
            widgets.append(item)
    return widgets


def set_f1(pred_items: set[Any], gt_items: set[Any], no_gt_reason: str) -> tuple[float | str, dict[str, Any]]:
    if not gt_items:
        return NA, {"reason": no_gt_reason}
    tp = len(pred_items & gt_items)
    stats = MatchStats(tp, len(pred_items), len(gt_items))
    return stats.f1, {"tp": stats.tp, "pred": stats.pred, "gt": stats.gt}


def widget_group_f1(pred: Any, gt: Any) -> tuple[float | str, dict[str, Any]]:
    gt_widgets = extract_widgets(gt)
    pred_widgets = extract_widgets(pred)
    if not gt_widgets:
        return NA, {"reason": "GT has no explicit widgets/widget groups"}
    edges: list[tuple[int, int, float]] = []
    for i, pred_widget in enumerate(pred_widgets):
        for j, gt_widget in enumerate(gt_widgets):
            score = _widget_match_score(pred_widget, gt_widget)
            if score > 0:
                edges.append((i, j, score))
    stats = MatchStats(max_bipartite_matches(edges, len(pred_widgets), len(gt_widgets)), len(pred_widgets), len(gt_widgets))
    return stats.f1, {"tp": stats.tp, "pred": stats.pred, "gt": stats.gt, "method": "type/selected plus IoU-or-center-distance bipartite matching"}


def widget_answer_accuracy(pred: Any, gt_answer: Any, structural_gt: Any) -> tuple[float | str, dict[str, Any]]:
    if not isinstance(structural_gt, dict):
        return NA, {"reason": "no structural GT for widget answer paths"}
    raw_paths = structural_gt.get("_widget_answer_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        return NA, {"reason": "metadata has no selectable widget answer paths"}
    pred_fields = flatten_leaf_fields(unwrap_answer(pred))
    gt_fields = flatten_leaf_fields(gt_answer)
    paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_path in raw_paths:
        if isinstance(raw_path, list):
            path = tuple(str(part).strip() for part in raw_path if str(part).strip())
            if path and path in gt_fields and path not in seen:
                paths.append(path)
                seen.add(path)
    if not paths:
        return NA, {"reason": "metadata widget paths do not map to GT answer leaves", "raw_path_count": len(raw_paths)}
    correct = 0
    mismatches: list[dict[str, Any]] = []
    for path in paths:
        gt_value = gt_fields.get(path)
        pred_value = pred_fields.get(path)
        if pred_value == gt_value:
            correct += 1
        elif len(mismatches) < 20:
            mismatches.append({"path": list(path), "gt": gt_value, "pred": pred_value})
    return correct / len(paths), {"correct": correct, "total": len(paths), "paths": [list(path) for path in paths], "mismatches_sample": mismatches}


def _rel_endpoint(value: Any) -> str:
    text = str(value or "").strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "__" in text:
        text = text.rsplit("__", 1)[-1]
    return " ".join(text.replace("field_", "").replace("value_", "").replace("widget_", "").split())


def _relation_signature(item: dict[str, Any]) -> tuple[str, str, str]:
    relation_type = str(item.get("type") or item.get("relation_type") or "relation").lower()
    source = _rel_endpoint(item.get("source_label") or item.get("source") or item.get("from") or item.get("parent"))
    target = _rel_endpoint(item.get("target_label") or item.get("target") or item.get("to") or item.get("child"))
    if relation_type == "key-value":
        # Template metadata has value slots, not per-instance value text.
        return (source, "__value__", relation_type)
    if relation_type == "parent-child":
        # Some metadata parent ids are generated; compare child membership by label.
        return ("__parent__", target, relation_type)
    return (source, target, relation_type)


def extract_relations(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    relations = data.get("relations") or data.get("edges") or data.get("relation_edges")
    return [item for item in as_list(relations) if isinstance(item, dict)]


def relation_f1(pred: Any, gt: Any) -> tuple[float | str, dict[str, Any]]:
    gt_rel_items = extract_relations(gt)
    pred_rel_items = extract_relations(pred)
    if not pred_rel_items:
        pred_rel_items = answer_tree_relations(pred)
    gt_rels = {_relation_signature(item) for item in gt_rel_items}
    pred_rels = {_relation_signature(item) for item in pred_rel_items}
    return set_f1(pred_rels, gt_rels, "GT has no explicit relation edge set")


@lru_cache(maxsize=None)
def _load_optional_layout_gt_cached(layout_root: str, template_name: str) -> Any | None:
    path = Path(layout_root) / f"{template_name}.json"
    if not path.exists():
        return None
    return normalize_metadata_layout(read_json(path))


def load_optional_layout_gt(sample: dict[str, Any], layout_root: Path | None) -> Any | None:
    if not layout_root:
        return None
    return _load_optional_layout_gt_cached(str(layout_root), str(sample["template_name"]))


def evaluate_sample(
    sample: dict[str, Any],
    pred: Any | None,
    *,
    valid_json: bool,
    layout_root: Path | None = None,
    group: str = "vlm",
) -> dict[str, Any]:
    gt_answer = read_json(Path(sample["label_path"]))
    layout_gt = load_optional_layout_gt(sample, layout_root)
    structural_gt = gt_answer if any(k in gt_answer for k in ("regions", "local_grids", "cells", "widgets", "relations")) else layout_gt

    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "template_name": sample["template_name"],
        "instance_id": sample["instance_id"],
        "valid_json": valid_json,
        "group": group,
    }
    details: dict[str, Any] = {}
    if not valid_json or pred is None:
        row.update({"TSR-path": 0.0, "VAcc": 0.0, "R-F1": NA, "R-F1@0.75": NA, "LIG-F1": NA, "WAcc": NA, "CDS": NA})
        row["details"] = {"invalid_json": True}
        return row

    row["TSR-path"], details["TSR-path"] = field_accuracy(pred, gt_answer)
    row["VAcc"], details["VAcc"] = value_accuracy(pred, gt_answer)
    if structural_gt is None:
        row.update({"R-F1": NA, "R-F1@0.75": NA, "LIG-F1": NA, "WAcc": NA})
        details["structural_gt"] = "NA: no instance structural fields and no layout annotation found"
    else:
        row["R-F1"], details["R-F1"] = region_f1(pred, structural_gt, threshold=0.5)
        row["R-F1@0.75"], details["R-F1@0.75"] = region_f1(pred, structural_gt, threshold=0.75)
        row["LIG-F1"], details["LIG-F1"] = line_item_group_f1(pred, structural_gt)
        row["WAcc"], details["WAcc"] = widget_answer_accuracy(pred, gt_answer, structural_gt)
        details["structural_gt"] = "layout_root" if structural_gt is layout_gt else "answer.json"
    row["details"] = details
    return row


def compute_cds(row: dict[str, Any], weights: dict[str, float], *, traditional: bool = False) -> float | str:
    if traditional:
        return NA
    values: list[float] = []
    used_weights: list[float] = []
    for metric, weight in weights.items():
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value) * float(weight))
            used_weights.append(float(weight))
    if not values or sum(used_weights) == 0:
        return NA
    return sum(values) / sum(used_weights)


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return (sum(values) / len(values)) if values else NA


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
