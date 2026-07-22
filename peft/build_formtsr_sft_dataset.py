#!/usr/bin/env python3
"""Build an index-preserving multimodal SFT jsonl for FormTSR.

The frozen FormTSR split files are path indexes, while most multimodal SFT
tools expect conversational records.  This converter deliberately does not
re-split or shuffle the index: it only resolves paths and wraps each semantic
answer in the seven-key output contract used by ``formtsr_exp``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA_KEYS = (
    "regions",
    "widgets",
    "local_grids",
    "cells",
    "line_item_groups",
    "relations",
    "answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        default="outputs/dataset_splits/template_stratified_seed42/train_index.jsonl",
    )
    parser.add_argument("--output", default="outputs/qwen35_formtsr_sft/train.jsonl")
    parser.add_argument("--prompt-file", default="", help="Optional text file overriding the formal prompt.")
    parser.add_argument("--structure-root", default="newdataset-layout")
    parser.add_argument("--metadata-root", default="new-dataset-json")
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--max-widgets", type=int, default=80)
    parser.add_argument("--max-local-grids", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=160)
    parser.add_argument("--max-line-item-groups", type=int, default=None)
    parser.add_argument("--max-relations", type=int, default=220)
    parser.add_argument("--max-structure-text-length", type=int, default=80)
    parser.add_argument(
        "--hierarchical-structure",
        action="store_true",
        help=(
            "Build widget groups, grids/cells, and typed relations through the "
            "formal hierarchical GT loader."
        ),
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Use empty structural arrays instead of template layout supervision.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"index row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def formal_prompt(prompt_file: str, *, hierarchical_structure: bool = False) -> str:
    if prompt_file:
        return resolve_path(prompt_file).read_text(encoding="utf-8")
    from formtsr_exp.prompt import HIERARCHICAL_PROMPT, PROMPT

    return HIERARCHICAL_PROMPT if hierarchical_structure else PROMPT


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_bbox(box: Any, width: float, height: float) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4 or width <= 0 or height <= 0:
        return None
    values = [_number(item) for item in box]
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values  # type: ignore[misc]
    left, right = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    top, bottom = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if right <= left or bottom <= top:
        return None
    return [
        round(left / width * 1000, 2),
        round(top / height * 1000, 2),
        round(right / width * 1000, 2),
        round(bottom / height * 1000, 2),
    ]


def compact_normalized_bbox(box: Any) -> list[int] | None:
    """Convert an evaluator-native 0..1 box to compact 0..1000 integers."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [_number(item) for item in box]
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values  # type: ignore[misc]
    left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if right <= left or bottom <= top:
        return None
    result = [
        round(left * 1000),
        round(top * 1000),
        round(right * 1000),
        round(bottom * 1000),
    ]
    # Rounding can collapse a sub-pixel box. Preserve a positive extent.
    if result[2] <= result[0]:
        result[2] = min(1000, result[0] + 1)
    if result[3] <= result[1]:
        result[3] = min(1000, result[1] + 1)
    return result if result[2] > result[0] and result[3] > result[1] else None


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def raw_region_type(item: dict[str, Any]) -> str:
    value = item.get("type") or item.get("region_type") or item.get("data_type") or "other"
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def region_type(raw_type: Any) -> str:
    value = str(raw_type or "other").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"title", "header"}:
        return "title"
    if value in {"section", "section_header", "field_group", "group", "region"}:
        return "section"
    if value in {
        "checkbox",
        "checkbox_multi",
        "check_box",
        "radio",
        "radio_button",
        "input",
        "signature",
        "widget",
    }:
        return "widget"
    if value in {"value", "date", "number"}:
        return "value"
    if value in {"table", "row", "column", "cell"}:
        return "table"
    if value in {"text", "paragraph"}:
        return "text"
    if value == "field":
        return "field"
    return "other"


@lru_cache(maxsize=None)
def layout_target(template_name: str, structure_root: str, max_regions: int, max_ligs: int) -> dict[str, Any]:
    """Convert static template metadata to the benchmark's compact schema."""
    layout_path = Path(structure_root) / f"{template_name}.json"
    if not layout_path.exists():
        return {"regions": [], "line_item_groups": []}
    raw = json.loads(layout_path.read_text(encoding="utf-8"))
    from formtsr_exp.metrics import normalize_metadata_layout

    normalized = normalize_metadata_layout(raw)
    width = _number(raw.get("original_width")) or 1.0
    height = _number(raw.get("original_height")) or 1.0
    raw_regions = [item for item in normalized.get("regions", []) if isinstance(item, dict)]
    section_boxes = [
        bbox
        for item in raw_regions
        if raw_region_type(item) == "section"
        and (bbox := normalize_bbox(item.get("bbox"), width, height)) is not None
    ]
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    priority = {
        "title": 0,
        "section": 0,
        "table": 1,
        "field": 2,
        "widget": 2,
        "text": 3,
        "value": 4,
        "other": 5,
    }
    for index, item in enumerate(raw_regions):
        bbox = normalize_bbox(item.get("bbox"), width, height)
        if bbox is None:
            continue
        # The corrected evaluator drops generic regions that duplicate a typed
        # section, while keeping genuine subdivisions. Match that GT contract.
        if raw_region_type(item) == "region" and any(
            bbox_iou(bbox, section_bbox) >= 0.9 for section_bbox in section_boxes
        ):
            continue
        kind = region_type(item.get("type"))
        text = str(item.get("text") or "").strip()
        candidates.append(
            (
                priority.get(kind, 5),
                index,
                {
                    "id": str(item.get("id") or f"r{index + 1}"),
                    "type": kind,
                    "bbox": bbox,
                    "text": text[:160],
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    regions: list[dict[str, Any]] = []
    for selected_index, (_, _, item) in enumerate(
        candidates[: max(0, max_regions)], start=1
    ):
        region = dict(item)
        region["id"] = f"r{selected_index}"
        regions.append(region)

    line_item_groups: list[dict[str, Any]] = []
    seen_lig_boxes: set[tuple[float, float, float, float]] = set()

    def add_line_item_group(bbox: list[float], text: str = "") -> None:
        key = tuple(round(value, 3) for value in bbox)
        if key in seen_lig_boxes:
            return
        seen_lig_boxes.add(key)
        line_item_groups.append(
            {
                "id": f"lig{len(line_item_groups) + 1}",
                "bbox": bbox,
                "text": text[:160],
            }
        )

    for item in normalized.get("line_item_groups", []):
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"), width, height)
        if bbox is None:
            continue
        add_line_item_group(bbox, str(item.get("text") or "").strip())

    # The evaluator treats a grid without its own bbox as one additional
    # line-item group spanning the union of its cells. Preserve that derived
    # evidence so training and corrected LIG-F1 use the same contract.
    grids: list[Any] = []
    for key in ("local_grids", "grids"):
        value = normalized.get(key)
        if isinstance(value, list):
            grids.extend(value)
    for grid in grids:
        if not isinstance(grid, dict):
            continue
        grid_bbox = normalize_bbox(grid.get("bbox"), width, height)
        if grid_bbox is not None:
            add_line_item_group(grid_bbox, str(grid.get("text") or "").strip())
            continue
        cell_boxes: list[list[float]] = []
        for cell in grid.get("cells", []) if isinstance(grid.get("cells"), list) else []:
            if not isinstance(cell, dict):
                continue
            cell_bbox = normalize_bbox(cell.get("bbox"), width, height)
            if cell_bbox is not None:
                cell_boxes.append(cell_bbox)
        if cell_boxes:
            add_line_item_group(
                [
                    min(box[0] for box in cell_boxes),
                    min(box[1] for box in cell_boxes),
                    max(box[2] for box in cell_boxes),
                    max(box[3] for box in cell_boxes),
                ]
            )
    return {"regions": regions, "line_item_groups": line_item_groups[: max(0, max_ligs)]}


@lru_cache(maxsize=None)
def metadata_item_labels(template_name: str, metadata_root: str) -> dict[str, str]:
    """Read visible labels keyed by the raw relation endpoint IDs."""
    metadata_path = Path(metadata_root) / f"{template_name}.json"
    if not metadata_path.exists():
        return {}
    from formtsr_exp.hierarchical_metrics_report import parse_raw_annotation

    items, _relations, _width, _height = parse_raw_annotation(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    return {
        item_id: str(item.original_label or item.semantic_key or "").strip()
        for item_id, item in items.items()
    }


def _compact_text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[: max(0, limit)]


def _prediction_region_type(category: Any) -> str:
    aliases = {"group": "section", "label": "field"}
    value = aliases.get(str(category), str(category))
    return value if value in {"title", "section", "field", "value", "text", "widget", "table"} else "other"


def _hierarchical_target(
    sample: Any,
    *,
    metadata_root: str,
    max_regions: int,
    max_widgets: int,
    max_grids: int,
    max_cells: int,
    max_ligs: int,
    max_relations: int,
    max_text_length: int,
) -> dict[str, Any]:
    """Serialize the formal hierarchical GT into a compact prediction target."""
    template = sample.template
    labels = metadata_item_labels(sample.template_name, metadata_root)
    region_by_id = {region.id: region for region in template.regions}
    raw_to_region = {
        raw_id: output_id
        for output_id, raw_id in template.region_relation_ids.items()
    }
    relation_degree: Counter[str] = Counter()
    for relation in template.relations:
        for endpoint in (relation.source, relation.target):
            region_id = raw_to_region.get(endpoint, endpoint)
            if region_id in region_by_id:
                relation_degree[region_id] += 1

    required_region_ids = {
        grid.parent_region_id
        for grid in template.grids
        if grid.parent_region_id in region_by_id
    }
    region_priority = {
        "title": 0,
        "section": 0,
        "table": 1,
        "field": 2,
        "widget": 2,
        "text": 3,
        "value": 4,
        "other": 5,
    }
    region_order = {region.id: index for index, region in enumerate(template.regions)}
    selected_region_ids = sorted(
        region_by_id,
        key=lambda region_id: (
            region_id not in required_region_ids,
            -relation_degree[region_id],
            region_priority.get(_prediction_region_type(region_by_id[region_id].category), 5),
            region_order[region_id],
        ),
    )[: max(0, max_regions)]

    endpoint_maps: dict[str, dict[str, str]] = {
        "regions": {},
        "widgets": {},
        "local_grids": {},
        "cells": {},
        "line_item_groups": {},
    }
    regions: list[dict[str, Any]] = []
    for region_id in selected_region_ids:
        region = region_by_id[region_id]
        bbox = compact_normalized_bbox(region.bbox)
        if bbox is None:
            continue
        compact_id = f"r{len(regions) + 1}"
        raw_id = template.region_relation_ids.get(region_id, region_id)
        endpoint_maps["regions"][region_id] = compact_id
        endpoint_maps["regions"][raw_id] = compact_id
        regions.append(
            {
                "id": compact_id,
                "type": _prediction_region_type(region.category),
                "bbox": bbox,
                "text": _compact_text(labels.get(raw_id, ""), max_text_length),
            }
        )

    widget_specs = {spec.id: spec for spec in template.widget_specs}
    widgets: list[dict[str, Any]] = []
    emitted_group_index = 0
    for group in sample.widget_groups:
        valid_members = [
            (member, compact_normalized_bbox(member.bbox))
            for member in group.members
        ]
        if any(bbox is None for _member, bbox in valid_members):
            continue
        if len(widgets) + len(valid_members) > max(0, max_widgets):
            continue
        emitted_group_index += 1
        group_id = f"wg{emitted_group_index}"
        for member, bbox in valid_members:
            assert bbox is not None
            compact_id = f"w{len(widgets) + 1}"
            endpoint_maps["widgets"][member.id] = compact_id
            spec = widget_specs.get(member.id)
            label_candidates = (
                tuple(spec.option_labels) + tuple(reversed(spec.field_path))
                if spec is not None
                else ()
            )
            label = next((str(value).strip() for value in label_candidates if str(value).strip()), "")
            widgets.append(
                {
                    "id": compact_id,
                    "type": member.widget_type,
                    "bbox": bbox,
                    "label": _compact_text(label, max_text_length),
                    "state": member.state,
                    "group_id": group_id,
                    "group_type": group.group_type,
                }
            )

    grids: list[dict[str, Any]] = []
    emitted_cells = 0
    for grid in template.grids:
        parent_id = endpoint_maps["regions"].get(grid.parent_region_id)
        if parent_id is None:
            continue
        if len(grids) >= max(0, max_grids):
            break
        if emitted_cells + len(grid.cells) > max(0, max_cells):
            continue
        compact_grid_id = f"g{len(grids) + 1}"
        endpoint_maps["local_grids"][grid.id] = compact_grid_id
        cells: list[dict[str, Any]] = []
        for cell in grid.cells:
            compact_cell_id = f"c{emitted_cells + 1}"
            emitted_cells += 1
            endpoint_maps["cells"][cell.id] = compact_cell_id
            cell_item: dict[str, Any] = {
                "id": compact_cell_id,
                "row": cell.row,
                "col": cell.col,
            }
            if cell.rowspan != 1:
                cell_item["rowspan"] = cell.rowspan
            if cell.colspan != 1:
                cell_item["colspan"] = cell.colspan
            cell_bbox = compact_normalized_bbox(cell.bbox)
            if cell_bbox is not None:
                cell_item["bbox"] = cell_bbox
            cells.append(cell_item)
        grids.append(
            {
                "id": compact_grid_id,
                "region_id": parent_id,
                "cells": cells,
            }
        )

    line_item_groups: list[dict[str, Any]] = []
    for group in template.line_item_groups[: max(0, max_ligs)]:
        bbox = compact_normalized_bbox(group.bbox)
        if bbox is None:
            continue
        compact_id = f"l{len(line_item_groups) + 1}"
        raw_id = template.line_item_relation_ids.get(group.id, group.id)
        endpoint_maps["line_item_groups"][group.id] = compact_id
        endpoint_maps["line_item_groups"][raw_id] = compact_id
        line_item_groups.append({"id": compact_id, "bbox": bbox})

    def resolve_endpoint(raw_id: str, relation_type: str, *, target: bool) -> str | None:
        if target and relation_type == "field-widget":
            return endpoint_maps["widgets"].get(raw_id)
        if target and relation_type == "key-to-cell":
            return endpoint_maps["cells"].get(raw_id)
        for namespace in (
            "regions",
            "widgets",
            "cells",
            "line_item_groups",
            "local_grids",
        ):
            if raw_id in endpoint_maps[namespace]:
                return endpoint_maps[namespace][raw_id]
        return None

    relations: list[dict[str, str]] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in template.relations:
        if len(relations) >= max(0, max_relations):
            break
        source = resolve_endpoint(relation.source, relation.relation_type, target=False)
        target = resolve_endpoint(relation.target, relation.relation_type, target=True)
        if source is None or target is None or source == target:
            continue
        key = (source, relation.relation_type, target)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append({"u": source, "r": relation.relation_type, "v": target})
        if len(relations) >= max(0, max_relations):
            break

    target = {
        "regions": regions,
        "widgets": widgets,
        "local_grids": grids,
        "cells": [],
        "line_item_groups": line_item_groups,
        "relations": relations,
        "answer": sample.gt_answer,
    }
    _validate_hierarchical_target(
        target,
        max_regions=max_regions,
        max_widgets=max_widgets,
        max_grids=max_grids,
        max_cells=max_cells,
        max_ligs=max_ligs,
        max_relations=max_relations,
    )
    return target


def _validate_hierarchical_target(
    target: dict[str, Any],
    *,
    max_regions: int,
    max_widgets: int,
    max_grids: int,
    max_cells: int,
    max_ligs: int,
    max_relations: int,
) -> None:
    if tuple(target) != SCHEMA_KEYS:
        raise ValueError(f"hierarchical target keys differ from schema: {tuple(target)}")
    limits = {
        "regions": max_regions,
        "widgets": max_widgets,
        "local_grids": max_grids,
        "cells": max_cells,
        "line_item_groups": max_ligs,
        "relations": max_relations,
    }
    for key, limit in limits.items():
        value = target.get(key)
        if not isinstance(value, list) or len(value) > max(0, limit):
            raise ValueError(f"invalid {key} count: {len(value) if isinstance(value, list) else value!r}")

    registry: set[str] = set()
    region_ids = {str(item["id"]) for item in target["regions"]}
    widget_group_types: dict[str, str] = {}
    for namespace in ("regions", "widgets", "line_item_groups"):
        for item in target[namespace]:
            item_id = str(item["id"])
            if item_id in registry:
                raise ValueError(f"duplicate hierarchical target id: {item_id}")
            registry.add(item_id)
            if namespace == "widgets":
                state = item.get("state")
                if state not in {"selected", "unselected", "unknown"}:
                    raise ValueError(f"invalid widget state for {item_id}: {state!r}")
                group_id = str(item.get("group_id") or "")
                group_type = str(item.get("group_type") or "")
                if not group_id or not group_type:
                    raise ValueError(f"missing widget group metadata for {item_id}")
                previous = widget_group_types.setdefault(group_id, group_type)
                if previous != group_type:
                    raise ValueError(f"inconsistent group type for {group_id}")

    nested_cells = 0
    for grid in target["local_grids"]:
        grid_id = str(grid["id"])
        if grid_id in registry:
            raise ValueError(f"duplicate hierarchical target id: {grid_id}")
        registry.add(grid_id)
        if str(grid.get("region_id") or "") not in region_ids:
            raise ValueError(f"grid {grid_id} references a missing region")
        cells = grid.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"grid {grid_id} has no cells")
        nested_cells += len(cells)
        for cell in cells:
            cell_id = str(cell["id"])
            if cell_id in registry:
                raise ValueError(f"duplicate hierarchical target id: {cell_id}")
            registry.add(cell_id)
    if nested_cells > max(0, max_cells):
        raise ValueError(f"nested cell count exceeds cap: {nested_cells} > {max_cells}")

    for relation in target["relations"]:
        source = str(relation.get("source") or relation.get("u") or "")
        destination = str(relation.get("target") or relation.get("v") or "")
        if source not in registry or destination not in registry:
            raise ValueError(f"relation has an unresolved endpoint: {relation}")
        if not (relation.get("type") or relation.get("r")):
            raise ValueError(f"relation has no type: {relation}")


def target_for(
    answer: Any,
    *,
    template_name: str,
    structure_root: str,
    max_regions: int,
    max_ligs: int,
    include_structure: bool,
) -> dict[str, Any]:
    if not isinstance(answer, dict):
        raise ValueError("FormTSR answer.json must contain a top-level object")
    structure = (
        layout_target(template_name, structure_root, max_regions, max_ligs)
        if include_structure
        else {"regions": [], "line_item_groups": []}
    )
    # FormTSR answer.json contains instance-specific semantic labels.  The
    # optional regions/line-item groups come from template metadata and use
    # normalized 0..1000 coordinates, matching the benchmark prompt.
    return {
        "regions": structure["regions"],
        "widgets": [],
        "local_grids": [],
        "cells": [],
        "line_item_groups": structure["line_item_groups"],
        "relations": [],
        "answer": answer,
    }


def make_record(
    row: dict[str, Any],
    prompt: str,
    *,
    structure_root: str,
    metadata_root: str,
    max_regions: int,
    max_widgets: int,
    max_grids: int,
    max_cells: int,
    max_ligs: int,
    max_relations: int,
    max_text_length: int,
    include_structure: bool,
    hierarchical_sample: Any | None = None,
) -> dict[str, Any]:
    image_path = resolve_path(str(row["image_path"]))
    label_path = resolve_path(str(row["label_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"label not found: {label_path}")
    answer = json.loads(label_path.read_text(encoding="utf-8"))
    target_object = (
        _hierarchical_target(
            hierarchical_sample,
            metadata_root=metadata_root,
            max_regions=max_regions,
            max_widgets=max_widgets,
            max_grids=max_grids,
            max_cells=max_cells,
            max_ligs=max_ligs,
            max_relations=max_relations,
            max_text_length=max_text_length,
        )
        if hierarchical_sample is not None
        else target_for(
            answer,
            template_name=str(row.get("template_name", "")),
            structure_root=structure_root,
            max_regions=max_regions,
            max_ligs=max_ligs,
            include_structure=include_structure,
        )
    )
    target = json.dumps(
        target_object,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    sample_id = str(row["sample_id"])
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "template_name": str(row.get("template_name", "")),
        "instance_id": str(row.get("instance_id", "")),
        "image": str(image_path),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": str(image_path)},
                ],
            },
            {"role": "assistant", "content": target},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize_targets(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    maxima: Counter[str] = Counter()
    nonempty_pages: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    widget_states: Counter[str] = Counter()
    character_lengths: list[int] = []
    structural_keys = SCHEMA_KEYS[:-1]
    for record in records:
        content = str(record["messages"][-1]["content"])
        character_lengths.append(len(content))
        target = json.loads(content)
        for key in structural_keys:
            items = target[key]
            totals[key] += len(items)
            maxima[key] = max(maxima[key], len(items))
            nonempty_pages[key] += bool(items)
        nested_cells = sum(len(grid.get("cells", [])) for grid in target["local_grids"])
        totals["nested_cells"] += nested_cells
        maxima["nested_cells"] = max(maxima["nested_cells"], nested_cells)
        nonempty_pages["nested_cells"] += nested_cells > 0
        relation_types.update(
            str(item.get("type") or item.get("r") or "")
            for item in target["relations"]
        )
        widget_states.update(str(item.get("state") or "") for item in target["widgets"])
    ordered_lengths = sorted(character_lengths)

    def percentile(fraction: float) -> int:
        index = min(len(ordered_lengths) - 1, int(len(ordered_lengths) * fraction))
        return ordered_lengths[index]

    return {
        "total_items": dict(totals),
        "max_items_per_page": dict(maxima),
        "nonempty_pages": dict(nonempty_pages),
        "relation_types": dict(relation_types),
        "widget_states": dict(widget_states),
        "target_characters": {
            "min": min(character_lengths),
            "mean": sum(character_lengths) / len(character_lengths),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(character_lengths),
        },
    }


def main() -> None:
    args = parse_args()
    if args.hierarchical_structure and args.no_structure:
        raise ValueError("--hierarchical-structure and --no-structure are mutually exclusive")
    if args.max_regions is None:
        args.max_regions = 80 if args.hierarchical_structure else 60
    if args.max_line_item_groups is None:
        args.max_line_item_groups = 20 if args.hierarchical_structure else 8
    for name in (
        "max_regions",
        "max_widgets",
        "max_local_grids",
        "max_cells",
        "max_line_item_groups",
        "max_relations",
        "max_structure_text_length",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    index_path = resolve_path(args.index)
    output_path = resolve_path(args.output)
    prompt = formal_prompt(
        args.prompt_file,
        hierarchical_structure=args.hierarchical_structure,
    )
    rows = read_jsonl(index_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    structure_root_path = resolve_path(args.structure_root)
    metadata_root_path = resolve_path(args.metadata_root)
    structure_root = str(structure_root_path)
    metadata_root = str(metadata_root_path)
    hierarchical_samples: dict[str, Any] = {}
    if args.hierarchical_structure:
        from formtsr_exp.hierarchical_metrics_report import load_samples

        samples = load_samples(index_path, metadata_root_path, structure_root_path)
        all_hierarchical_samples = {sample.sample_id: sample for sample in samples}
        index_ids = [str(row["sample_id"]) for row in rows]
        missing_ids = [sample_id for sample_id in index_ids if sample_id not in all_hierarchical_samples]
        if missing_ids:
            raise ValueError(f"hierarchical GT is missing frozen index samples: {missing_ids[:10]}")
        hierarchical_samples = {
            sample_id: all_hierarchical_samples[sample_id] for sample_id in index_ids
        }
    records = [
        make_record(
            row,
            prompt,
            structure_root=structure_root,
            metadata_root=metadata_root,
            max_regions=args.max_regions,
            max_widgets=args.max_widgets,
            max_grids=args.max_local_grids,
            max_cells=args.max_cells,
            max_ligs=args.max_line_item_groups,
            max_relations=args.max_relations,
            max_text_length=args.max_structure_text_length,
            include_structure=not args.no_structure,
            hierarchical_sample=hierarchical_samples.get(str(row["sample_id"])),
        )
        for row in rows
    ]
    write_jsonl(output_path, records)

    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    metadata = {
        "index": str(index_path),
        "index_sha256": index_sha256,
        "output": str(output_path),
        "count": len(records),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "target_keys": list(SCHEMA_KEYS),
        "target_mode": (
            "hierarchical_template_structure_plus_instance_widget_state_plus_answer_tree"
            if args.hierarchical_structure
            else "template_structure_plus_answer_tree"
            if not args.no_structure
            else "empty_structure_plus_answer_tree"
        ),
        "structure_root": structure_root,
        "metadata_root": metadata_root,
        "max_regions": args.max_regions,
        "max_widgets": args.max_widgets,
        "max_local_grids": args.max_local_grids,
        "max_cells": args.max_cells,
        "max_line_item_groups": args.max_line_item_groups,
        "max_relations": args.max_relations,
        "max_structure_text_length": args.max_structure_text_length,
        "target_statistics": summarize_targets(records),
        "path_policy": "absolute paths resolved from repository root",
    }
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
