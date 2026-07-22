from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .document_similarity_report import (
    build_schema_tree,
    extract_normalized_values,
    schema_nted,
    value_ned,
)
from .hierarchical_metrics import (
    BBox,
    GridCell,
    LocalGrid,
    Relation,
    Widget,
    WidgetGroup,
    bbox_iou,
    flatten_widgets,
    lg_grits_top,
    match_widgets_for_relations,
    relation_f1,
    widget_group_f1,
)
from .hierarchical_metrics_report import (
    BoxNode,
    EvaluationSample,
    PredictionStructure,
    RegionNode,
    TemplateStructure,
    _add_endpoint_mapping,
    _field_endpoint_mapping,
    load_samples,
    match_box_nodes,
    match_regions,
)
from .io_utils import ensure_dir, write_json
from .metrics import NA, field_accuracy, flatten_leaf_fields
from .structure_metrics_report import PRED_COMPATIBILITY


METRICS = (
    "Schema-nTED",
    "Value-nED",
    "TSR-path",
    "R-F1@0.5",
    "LIG-F1",
    "LG-GriTS-Top",
    "WG-F1",
    "Rel-F1",
)

ERRORS = (
    "value",
    "hierarchy",
    "region",
    "line_item",
    "local_grid",
    "widget",
    "relation",
)

ERROR_LABELS = {
    "value": "Value corruption",
    "hierarchy": "Hierarchy corruption",
    "region": "Region corruption",
    "line_item": "Line-item corruption",
    "local_grid": "Local-grid corruption",
    "widget": "Widget corruption",
    "relation": "Relation corruption",
}

PRIMARY_TARGETS = {
    "value": ("Value-nED",),
    "hierarchy": ("Schema-nTED", "TSR-path"),
    "region": ("R-F1@0.5",),
    "line_item": ("LIG-F1",),
    "local_grid": ("LG-GriTS-Top",),
    "widget": ("WG-F1",),
    "relation": ("Rel-F1",),
}

# These are downstream consequences of frozen endpoint/parent matching, not
# leakage into an unrelated diagnostic dimension.
EXPECTED_DEPENDENCIES = {
    "value": ("TSR-path",),
    "hierarchy": ("Rel-F1",),
    "region": ("LG-GriTS-Top", "Rel-F1"),
    "line_item": ("Rel-F1",),
    "local_grid": ("Rel-F1",),
    "widget": (),
    "relation": (),
}

DISPLAY_METRICS = {
    "Schema-nTED": "Schema-nTED",
    "Value-nED": "Value-nED",
    "TSR-path": "TSR-path",
    "R-F1@0.5": "R-F1@0.5",
    "LIG-F1": "LIG-F1",
    "LG-GriTS-Top": "LG-GriTS",
    "WG-F1": "WG-F1",
    "Rel-F1": "Rel-F1",
}


@dataclass(frozen=True, slots=True)
class DiagnosticPrediction:
    answer: Any
    regions: tuple[RegionNode, ...]
    line_item_groups: tuple[BoxNode, ...]
    grids: tuple[LocalGrid, ...]
    widget_groups: tuple[WidgetGroup, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True, slots=True)
class ItemBox:
    id: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class LineGroupPlan:
    group_index: int
    members: tuple[ItemBox, ...]
    outsiders: tuple[ItemBox, ...]


@dataclass(frozen=True, slots=True)
class TemplatePerturbationPlan:
    shifted_region_boxes: Mapping[int, BBox]
    line_groups: tuple[LineGroupPlan, ...]
    alternative_grids: Mapping[int, LocalGrid]


@dataclass(frozen=True, slots=True)
class PairedPageResult:
    error: str
    severity: float
    seed: int
    sample_id: str
    template_name: str
    eligible_units: int
    injected_units: int
    clean_scores: Mapping[str, float | None]
    corrupted_scores: Mapping[str, float | None]

    def drop(self, metric: str) -> float | None:
        clean = self.clean_scores[metric]
        corrupted = self.corrupted_scores[metric]
        if clean is None or corrupted is None:
            return None
        return 100.0 * (clean - corrupted)


def _numeric(value: Any) -> float | None:
    if value == NA or value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"expected numeric score or NA, got {value!r}")


def _f1_from_mapping(n_matches: int, n_pred: int, n_gt: int) -> float | None:
    denominator = n_pred + n_gt
    return 2.0 * n_matches / denominator if denominator else None


def clean_prediction(sample: EvaluationSample) -> DiagnosticPrediction:
    endpoint_aliases: dict[str, str] = {}

    # Prefer the most specific structural namespace when a raw annotation ID
    # is observable through more than one node family.
    for widget in flatten_widgets(sample.widget_groups):
        endpoint_aliases.setdefault(widget.id, widget.id)
    for pred_id, gt_id in sample.template.line_item_relation_ids.items():
        endpoint_aliases.setdefault(gt_id, pred_id)
    for grid in sample.template.grids:
        endpoint_aliases.setdefault(grid.id, grid.id)
        for cell in grid.cells:
            endpoint_aliases.setdefault(cell.id, cell.id)
    for pred_id, gt_id in sample.template.region_relation_ids.items():
        endpoint_aliases.setdefault(gt_id, pred_id)
    for path, gt_id in sample.template.field_paths.items():
        endpoint_aliases.setdefault(gt_id, "/".join(path))

    missing_endpoints = {
        endpoint
        for relation in sample.template.relations
        for endpoint in (relation.source, relation.target)
        if endpoint not in endpoint_aliases
    }
    if missing_endpoints:
        raise AssertionError(
            "gold relation endpoints are not recoverable: "
            + ", ".join(sorted(missing_endpoints))
        )
    predicted_relations = tuple(
        Relation(
            endpoint_aliases[relation.source],
            relation.relation_type,
            endpoint_aliases[relation.target],
        )
        for relation in sample.template.relations
    )
    return DiagnosticPrediction(
        answer=copy.deepcopy(sample.gt_answer),
        regions=sample.template.regions,
        line_item_groups=sample.template.line_item_groups,
        grids=sample.template.grids,
        widget_groups=sample.widget_groups,
        relations=predicted_relations,
    )


def score_prediction(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
) -> dict[str, float | None]:
    if prediction.answer == sample.gt_answer:
        schema_score = 1.0
        value_score = 1.0
        tsr_score: float | str = 1.0
    else:
        pred_schema = build_schema_tree(prediction.answer)
        gt_schema = build_schema_tree(sample.gt_answer)
        schema_score = schema_nted(pred_schema, gt_schema)
        pred_values = extract_normalized_values(prediction.answer)
        gt_values = extract_normalized_values(sample.gt_answer)
        value_score = (
            1.0
            if Counter(pred_values) == Counter(gt_values)
            else value_ned(pred_values, gt_values)
        )
        tsr_score, _ = field_accuracy(prediction.answer, sample.gt_answer)

    region_mapping = match_regions(prediction.regions, sample.template.regions)
    region_score = _f1_from_mapping(
        len(region_mapping), len(prediction.regions), len(sample.template.regions)
    )

    line_item_mapping = match_box_nodes(
        prediction.line_item_groups, sample.template.line_item_groups
    )
    line_item_score = (
        _f1_from_mapping(
            len(line_item_mapping),
            len(prediction.line_item_groups),
            len(sample.template.line_item_groups),
        )
        if sample.template.line_item_groups
        else None
    )

    grid_score = lg_grits_top(
        prediction.grids, sample.template.grids, region_mapping
    )
    widget_score = widget_group_f1(
        prediction.widget_groups, sample.widget_groups
    )

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
        match_widgets_for_relations(
            flatten_widgets(prediction.widget_groups),
            flatten_widgets(sample.widget_groups),
        ),
    )
    _add_endpoint_mapping(
        endpoint_mapping,
        "line_item_groups",
        {
            pred_id: sample.template.line_item_relation_ids.get(gt_id, gt_id)
            for pred_id, gt_id in line_item_mapping.items()
        },
    )
    structural_prediction = PredictionStructure(
        regions=prediction.regions,
        grids=prediction.grids,
        widget_groups=prediction.widget_groups,
        relations=prediction.relations,
        line_item_groups=prediction.line_item_groups,
        answer_paths=tuple(flatten_leaf_fields(prediction.answer).keys()),
        audit={},
    )
    for pred_id, gt_id in _field_endpoint_mapping(
        structural_prediction, sample.template
    ).items():
        endpoint_mapping.setdefault(pred_id, gt_id)
    grid_endpoint_mapping: dict[str, str] = {}
    for pred_index, gt_index, _similarity in grid_score.matches:
        grid_endpoint_mapping[prediction.grids[pred_index].id] = (
            sample.template.grids[gt_index].id
        )
    _add_endpoint_mapping(endpoint_mapping, "local_grids", grid_endpoint_mapping)
    _add_endpoint_mapping(endpoint_mapping, "cells", grid_score.cell_mapping)
    relation_score = relation_f1(
        prediction.relations,
        sample.template.relations,
        endpoint_mapping,
    )

    return {
        "Schema-nTED": float(schema_score),
        "Value-nED": float(value_score),
        "TSR-path": _numeric(tsr_score),
        "R-F1@0.5": region_score,
        "LIG-F1": line_item_score,
        "LG-GriTS-Top": _numeric(grid_score.score),
        "WG-F1": _numeric(widget_score.score),
        "Rel-F1": _numeric(relation_score.counts.f1),
    }


def _unit_uniform(seed: int, error: str, sample_id: str, unit_id: str) -> float:
    payload = f"{seed}\0{error}\0{sample_id}\0{unit_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _selected(
    seed: int,
    error: str,
    sample_id: str,
    unit_id: str,
    severity: float,
) -> bool:
    return _unit_uniform(seed, error, sample_id, unit_id) < severity


def _set_leaf(root: Any, path: Sequence[str], value: Any) -> None:
    current = root
    for part in path[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    last = path[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def _corrupt_equal_length_text(value: str) -> str:
    output: list[str] = []
    for character in value:
        if character.isspace():
            output.append(character)
        elif "0" <= character <= "9":
            output.append(str((int(character) + 5) % 10))
        elif "a" <= character <= "z":
            output.append(chr((ord(character) - ord("a") + 13) % 26 + ord("a")))
        elif "A" <= character <= "Z":
            output.append(chr((ord(character) - ord("A") + 13) % 26 + ord("A")))
        else:
            output.append("y" if character == "x" else "x")
    corrupted = "".join(output)
    if corrupted == value:
        raise AssertionError("equal-length value corruption did not change the value")
    if len(corrupted) != len(value):
        raise AssertionError("value corruption changed character length")
    return corrupted


def corrupt_values(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    fields = flatten_leaf_fields(prediction.answer)
    eligible = [
        (path, value)
        for path, value in fields.items()
        if isinstance(value, str) and bool(value.strip())
    ]
    answer = copy.deepcopy(prediction.answer)
    selected_count = 0
    for path, value in eligible:
        unit_id = "/".join(path)
        if _selected(seed, "value", sample.sample_id, unit_id, severity):
            _set_leaf(answer, path, _corrupt_equal_length_text(value))
            selected_count += 1

    before = flatten_leaf_fields(prediction.answer)
    after = flatten_leaf_fields(answer)
    if before.keys() != after.keys():
        raise AssertionError("value corruption changed field paths")
    for path in before:
        if len(str(before[path])) != len(str(after[path])):
            raise AssertionError(f"value corruption changed length at {path!r}")
    return replace(prediction, answer=answer), len(eligible), selected_count


PathToken = str | int


@dataclass(frozen=True, slots=True)
class DictLeaf:
    parent_path: tuple[PathToken, ...]
    key: Any
    value: Any


def _dict_leaves(value: Any, path: tuple[PathToken, ...] = ()) -> list[DictLeaf]:
    leaves: list[DictLeaf] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                leaves.extend(_dict_leaves(child, path + (key,)))
            else:
                leaves.append(DictLeaf(path, key, child))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaves.extend(_dict_leaves(child, path + (index,)))
    return leaves


def _container(root: Any, path: Sequence[PathToken]) -> Any:
    current = root
    for part in path:
        current = current[part]
    return current


def _leaf_id(leaf: DictLeaf) -> str:
    return json.dumps(
        [*leaf.parent_path, leaf.key], ensure_ascii=False, separators=(",", ":")
    )


def _leaf_pair_multiset(value: Any) -> Counter[tuple[str, str]]:
    return Counter(
        (
            str(leaf.key),
            json.dumps(leaf.value, ensure_ascii=False, sort_keys=True),
        )
        for leaf in _dict_leaves(value)
    )


def _hierarchy_pairs(
    answer: Any,
    sample_id: str,
    seed: int,
) -> list[tuple[DictLeaf, DictLeaf]]:
    leaves = _dict_leaves(answer)
    ordered = sorted(
        leaves,
        key=lambda leaf: _unit_uniform(
            seed, "hierarchy-order", sample_id, _leaf_id(leaf)
        ),
    )
    used: set[str] = set()
    planned_incoming: dict[tuple[PathToken, ...], set[Any]] = defaultdict(set)
    pairs: list[tuple[DictLeaf, DictLeaf]] = []
    for left in ordered:
        left_id = _leaf_id(left)
        if left_id in used:
            continue
        candidates = sorted(
            ordered,
            key=lambda right: _unit_uniform(
                seed,
                "hierarchy-pair",
                sample_id,
                f"{left_id}->{_leaf_id(right)}",
            ),
        )
        left_parent = _container(answer, left.parent_path)
        for right in candidates:
            right_id = _leaf_id(right)
            if right_id in used or right_id == left_id:
                continue
            if left.parent_path == right.parent_path or left.key == right.key:
                continue
            right_parent = _container(answer, right.parent_path)
            if left.key in right_parent or right.key in left_parent:
                continue
            if left.key in planned_incoming[right.parent_path]:
                continue
            if right.key in planned_incoming[left.parent_path]:
                continue
            pairs.append((left, right))
            used.update((left_id, right_id))
            planned_incoming[right.parent_path].add(left.key)
            planned_incoming[left.parent_path].add(right.key)
            break
    return pairs


def corrupt_hierarchy(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    pairs = _hierarchy_pairs(prediction.answer, sample.sample_id, seed)
    chosen = [
        pair
        for pair_index, pair in enumerate(pairs)
        if _selected(
            seed,
            "hierarchy",
            sample.sample_id,
            f"pair:{pair_index}:{_leaf_id(pair[0])}:{_leaf_id(pair[1])}",
            severity,
        )
    ]
    answer = copy.deepcopy(prediction.answer)
    for left, right in chosen:
        del _container(answer, left.parent_path)[left.key]
        del _container(answer, right.parent_path)[right.key]
    for left, right in chosen:
        _container(answer, right.parent_path)[left.key] = copy.deepcopy(left.value)
        _container(answer, left.parent_path)[right.key] = copy.deepcopy(right.value)

    if _leaf_pair_multiset(answer) != _leaf_pair_multiset(prediction.answer):
        raise AssertionError("hierarchy corruption changed a key/value pair")
    return replace(prediction, answer=answer), 2 * len(pairs), 2 * len(chosen)


def _bbox_union(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise ValueError("cannot union an empty box sequence")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _translated_box(
    region: RegionNode,
    gt_regions: Sequence[RegionNode],
) -> BBox | None:
    x1, y1, x2, y2 = region.bbox
    width = x2 - x1
    height = y2 - y1
    max_x = 1.0 - width
    max_y = 1.0 - height
    if max_x < 0 or max_y < 0:
        return None
    x_positions = {max_x * index / 8.0 for index in range(9)}
    y_positions = {max_y * index / 8.0 for index in range(9)}
    compatible = [
        gt
        for gt in gt_regions
        if gt.category in PRED_COMPATIBILITY.get(region.category, set())
    ]
    candidates: list[tuple[float, float, BBox]] = []
    for candidate_x in x_positions:
        for candidate_y in y_positions:
            box = (
                candidate_x,
                candidate_y,
                candidate_x + width,
                candidate_y + height,
            )
            own_overlap = bbox_iou(box, region.bbox)
            maximum_overlap = max(
                (bbox_iou(box, gt.bbox) for gt in compatible), default=0.0
            )
            if own_overlap < 0.5 and maximum_overlap < 0.5:
                displacement = abs(candidate_x - x1) + abs(candidate_y - y1)
                candidates.append((maximum_overlap, -displacement, box))
    return min(candidates, default=(0.0, 0.0, None))[2]


def _alternative_grid(grid: LocalGrid) -> LocalGrid | None:
    if len(grid.cells) < 2:
        return None
    n_rows = max(cell.row + cell.rowspan for cell in grid.cells)
    n_cols = max(cell.col + cell.colspan for cell in grid.cells)
    make_column = n_cols >= n_rows

    def rearrange(column: bool) -> tuple[GridCell, ...]:
        return tuple(
            replace(
                cell,
                row=index if column else 0,
                col=0 if column else index,
                rowspan=1,
                colspan=1,
            )
            for index, cell in enumerate(grid.cells)
        )

    original = tuple(
        (cell.row, cell.col, cell.rowspan, cell.colspan) for cell in grid.cells
    )
    cells = rearrange(make_column)
    changed = tuple(
        (cell.row, cell.col, cell.rowspan, cell.colspan) for cell in cells
    )
    if changed == original:
        cells = rearrange(not make_column)
        changed = tuple(
            (cell.row, cell.col, cell.rowspan, cell.colspan) for cell in cells
        )
    return replace(grid, cells=cells) if changed != original else None


def build_template_plan(template: TemplateStructure) -> TemplatePerturbationPlan:
    shifted = {
        index: box
        for index, region in enumerate(template.regions)
        for box in [_translated_box(region, template.regions)]
        if box is not None
    }

    items: list[ItemBox] = []
    for region in template.regions:
        if region.category not in {"group", "table"}:
            items.append(ItemBox(f"region:{region.id}", region.bbox))
    for grid in template.grids:
        for cell in grid.cells:
            if cell.bbox is not None:
                items.append(ItemBox(f"cell:{cell.id}", cell.bbox))
    deduplicated: dict[tuple[float, float, float, float], ItemBox] = {}
    for item in items:
        key = tuple(round(value, 8) for value in item.bbox)
        deduplicated.setdefault(key, item)
    unique_items = tuple(deduplicated.values())

    line_plans: list[LineGroupPlan] = []
    for group_index, group in enumerate(template.line_item_groups):
        members: list[ItemBox] = []
        outsiders: list[ItemBox] = []
        for item in unique_items:
            center_x = (item.bbox[0] + item.bbox[2]) / 2.0
            center_y = (item.bbox[1] + item.bbox[3]) / 2.0
            destination = (
                members
                if group.bbox[0] <= center_x <= group.bbox[2]
                and group.bbox[1] <= center_y <= group.bbox[3]
                else outsiders
            )
            destination.append(item)
        if members and outsiders:
            line_plans.append(
                LineGroupPlan(group_index, tuple(members), tuple(outsiders))
            )

    alternatives = {
        index: alternative
        for index, grid in enumerate(template.grids)
        for alternative in [_alternative_grid(grid)]
        if alternative is not None
    }
    return TemplatePerturbationPlan(shifted, tuple(line_plans), alternatives)


def corrupt_regions(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    plan: TemplatePerturbationPlan,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    regions = list(prediction.regions)
    selected_count = 0
    for index, box in plan.shifted_region_boxes.items():
        if _selected(
            seed, "region", sample.sample_id, f"region:{regions[index].id}", severity
        ):
            regions[index] = replace(regions[index], bbox=box)
            selected_count += 1
    if len(regions) != len(prediction.regions):
        raise AssertionError("region corruption changed region count")
    if [item.category for item in regions] != [
        item.category for item in prediction.regions
    ]:
        raise AssertionError("region corruption changed region types")
    return (
        replace(prediction, regions=tuple(regions)),
        len(plan.shifted_region_boxes),
        selected_count,
    )


def _seeded_item_order(
    items: Sequence[ItemBox], seed: int, sample_id: str, prefix: str
) -> list[ItemBox]:
    return sorted(
        items,
        key=lambda item: _unit_uniform(seed, prefix, sample_id, item.id),
    )


def corrupt_line_items(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    plan: TemplatePerturbationPlan,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    groups = list(prediction.line_item_groups)
    eligible_count = 0
    selected_count = 0
    for group_plan in plan.line_groups:
        members = _seeded_item_order(
            group_plan.members, seed, sample.sample_id, "line-member-order"
        )
        outsiders = _seeded_item_order(
            group_plan.outsiders, seed, sample.sample_id, "line-outsider-order"
        )
        pair_count = min(len(members), len(outsiders))
        eligible_count += pair_count
        membership = {item.id: item for item in group_plan.members}
        touched = False
        for pair_index, (member, outsider) in enumerate(
            zip(members[:pair_count], outsiders[:pair_count])
        ):
            if not _selected(
                seed,
                "line_item",
                sample.sample_id,
                f"group:{group_plan.group_index}:pair:{pair_index}",
                severity,
            ):
                continue
            membership.pop(member.id, None)
            membership[outsider.id] = outsider
            selected_count += 1
            touched = True
        if touched:
            groups[group_plan.group_index] = replace(
                groups[group_plan.group_index],
                bbox=_bbox_union([item.bbox for item in membership.values()]),
            )
    if len(groups) != len(prediction.line_item_groups):
        raise AssertionError("line-item corruption changed group count")
    return (
        replace(prediction, line_item_groups=tuple(groups)),
        eligible_count,
        selected_count,
    )


def corrupt_local_grids(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    plan: TemplatePerturbationPlan,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    grids = list(prediction.grids)
    selected_count = 0
    for index, alternative in plan.alternative_grids.items():
        if _selected(
            seed, "local_grid", sample.sample_id, f"grid:{grids[index].id}", severity
        ):
            grids[index] = alternative
            selected_count += 1
    before_ids = [(grid.id, tuple(cell.id for cell in grid.cells)) for grid in prediction.grids]
    after_ids = [(grid.id, tuple(cell.id for cell in grid.cells)) for grid in grids]
    if before_ids != after_ids:
        raise AssertionError("local-grid corruption changed grid or cell identities")
    return (
        replace(prediction, grids=tuple(grids)),
        len(plan.alternative_grids),
        selected_count,
    )


def _flipped_widget_state(state: str) -> str:
    opposites = {
        "selected": "unselected",
        "unselected": "selected",
        "filled": "blank",
        "blank": "filled",
        "checked": "unchecked",
        "unchecked": "checked",
    }
    return opposites.get(state, "diagnostic-corrupted-state")


def corrupt_widgets(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    groups: list[WidgetGroup] = []
    selected_count = 0
    eligible_count = 0
    for group in prediction.widget_groups:
        members: list[Widget] = []
        for widget in group.members:
            eligible_count += 1
            if _selected(
                seed, "widget", sample.sample_id, f"widget:{widget.id}", severity
            ):
                members.append(replace(widget, state=_flipped_widget_state(widget.state)))
                selected_count += 1
            else:
                members.append(widget)
        groups.append(replace(group, members=tuple(members)))
    before_ids = [widget.id for widget in flatten_widgets(prediction.widget_groups)]
    after_ids = [widget.id for widget in flatten_widgets(groups)]
    if before_ids != after_ids:
        raise AssertionError("widget corruption changed widget count or identity")
    return replace(prediction, widget_groups=tuple(groups)), eligible_count, selected_count


def _wrong_relation(
    relation: Relation,
    gt_relations: set[Relation],
    occupied: set[Relation],
    prefer_reverse: bool,
) -> Relation:
    canonical_types = (
        "key-value",
        "parent-child",
        "field-widget",
        "key-to-cell",
        "key-to-field",
        "section-membership",
        "line-item-membership",
        "reading-order",
    )
    reversed_relation = Relation(
        relation.target, relation.relation_type, relation.source
    )
    typed = [
        Relation(relation.source, relation_type, relation.target)
        for relation_type in canonical_types
        if relation_type != relation.relation_type
    ]
    candidates = ([reversed_relation] + typed) if prefer_reverse else (typed + [reversed_relation])
    for candidate in candidates:
        if candidate not in gt_relations and candidate not in occupied:
            return candidate
    fallback_index = 0
    while True:
        candidate = Relation(
            relation.source,
            f"diagnostic-wrong-type-{fallback_index}",
            relation.target,
        )
        if candidate not in gt_relations and candidate not in occupied:
            return candidate
        fallback_index += 1


def corrupt_relations(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    relations = list(prediction.relations)
    gt_set = set(sample.template.relations)
    selected_indices = [
        index
        for index, relation in enumerate(relations)
        if _selected(
            seed,
            "relation",
            sample.sample_id,
            f"relation:{index}:{relation.source}:{relation.relation_type}:{relation.target}",
            severity,
        )
    ]
    occupied = set(relations)
    for index in selected_indices:
        original = relations[index]
        occupied.discard(original)
        prefer_reverse = _unit_uniform(
            seed, "relation-mode", sample.sample_id, str(index)
        ) < 0.5
        replacement = _wrong_relation(original, gt_set, occupied, prefer_reverse)
        relations[index] = replacement
        occupied.add(replacement)
    before_endpoints = Counter(
        tuple(sorted((relation.source, relation.target)))
        for relation in prediction.relations
    )
    after_endpoints = Counter(
        tuple(sorted((relation.source, relation.target))) for relation in relations
    )
    if len(relations) != len(prediction.relations) or before_endpoints != after_endpoints:
        raise AssertionError("relation corruption changed endpoint pairs or edge count")
    return (
        replace(prediction, relations=tuple(relations)),
        len(prediction.relations),
        len(selected_indices),
    )


def corrupt_prediction(
    prediction: DiagnosticPrediction,
    sample: EvaluationSample,
    plan: TemplatePerturbationPlan,
    error: str,
    severity: float,
    seed: int,
) -> tuple[DiagnosticPrediction, int, int]:
    if not 0.0 < severity < 1.0:
        raise ValueError("severity must lie strictly between zero and one")
    if error == "value":
        return corrupt_values(prediction, sample, severity, seed)
    if error == "hierarchy":
        return corrupt_hierarchy(prediction, sample, severity, seed)
    if error == "region":
        return corrupt_regions(prediction, sample, plan, severity, seed)
    if error == "line_item":
        return corrupt_line_items(prediction, sample, plan, severity, seed)
    if error == "local_grid":
        return corrupt_local_grids(prediction, sample, plan, severity, seed)
    if error == "widget":
        return corrupt_widgets(prediction, sample, severity, seed)
    if error == "relation":
        return corrupt_relations(prediction, sample, severity, seed)
    raise ValueError(f"unknown corruption: {error!r}")


def verify_gold_identity(
    samples: Sequence[EvaluationSample],
) -> tuple[dict[str, dict[str, float | None]], list[dict[str, Any]]]:
    clean_scores: dict[str, dict[str, float | None]] = {}
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    page_counts: dict[str, set[str]] = {metric: set() for metric in METRICS}
    template_counts: dict[str, set[str]] = {metric: set() for metric in METRICS}
    failures: list[tuple[str, str, float]] = []
    for sample in samples:
        scores = score_prediction(clean_prediction(sample), sample)
        clean_scores[sample.sample_id] = scores
        for metric, score in scores.items():
            if score is None:
                continue
            values[metric].append(score)
            page_counts[metric].add(sample.sample_id)
            template_counts[metric].add(sample.template_name)
            if abs(score - 1.0) > 1e-12:
                failures.append((sample.sample_id, metric, score))
    if failures:
        preview = "; ".join(
            f"{sample_id}/{metric}={score:.9f}"
            for sample_id, metric, score in failures[:20]
        )
        raise AssertionError(
            f"gold-as-prediction identity check failed on {len(failures)} scores: {preview}"
        )
    rows = [
        {
            "metric": metric,
            "applicable_pages": len(page_counts[metric]),
            "applicable_templates": len(template_counts[metric]),
            "minimum_score": min(values[metric]) if values[metric] else None,
            "mean_score": float(np.mean(values[metric])) if values[metric] else None,
            "maximum_score": max(values[metric]) if values[metric] else None,
            "all_scores_equal_100": bool(values[metric])
            and all(abs(value - 1.0) <= 1e-12 for value in values[metric]),
        }
        for metric in METRICS
    ]
    return clean_scores, rows


def run_condition(
    samples: Sequence[EvaluationSample],
    plans: Mapping[str, TemplatePerturbationPlan],
    clean_scores: Mapping[str, Mapping[str, float | None]],
    error: str,
    severity: float,
    seed: int,
) -> list[PairedPageResult]:
    rows: list[PairedPageResult] = []
    for sample in samples:
        clean = clean_prediction(sample)
        corrupted, eligible_units, injected_units = corrupt_prediction(
            clean,
            sample,
            plans[sample.template_name],
            error,
            severity,
            seed,
        )
        if eligible_units == 0:
            continue
        corrupted_scores = score_prediction(corrupted, sample)
        rows.append(
            PairedPageResult(
                error=error,
                severity=severity,
                seed=seed,
                sample_id=sample.sample_id,
                template_name=sample.template_name,
                eligible_units=eligible_units,
                injected_units=injected_units,
                clean_scores=clean_scores[sample.sample_id],
                corrupted_scores=corrupted_scores,
            )
        )
    return rows


_WORKER_SAMPLES: list[EvaluationSample] = []
_WORKER_PLANS: dict[str, TemplatePerturbationPlan] = {}
_WORKER_CLEAN_SCORES: dict[str, dict[str, float | None]] = {}


def _init_condition_worker(
    index_path: str,
    metadata_root: str,
    layout_root: str,
) -> None:
    global _WORKER_SAMPLES, _WORKER_PLANS, _WORKER_CLEAN_SCORES
    _WORKER_SAMPLES = load_samples(
        Path(index_path), Path(metadata_root), Path(layout_root)
    )
    _WORKER_PLANS = {}
    for sample in _WORKER_SAMPLES:
        if sample.template_name not in _WORKER_PLANS:
            _WORKER_PLANS[sample.template_name] = build_template_plan(
                sample.template
            )
    _WORKER_CLEAN_SCORES, _identity_rows = verify_gold_identity(_WORKER_SAMPLES)


def _run_condition_worker(
    condition: tuple[str, float, int],
) -> list[PairedPageResult]:
    error, severity, seed = condition
    return run_condition(
        _WORKER_SAMPLES,
        _WORKER_PLANS,
        _WORKER_CLEAN_SCORES,
        error,
        severity,
        seed,
    )


def run_all_conditions(
    samples: Sequence[EvaluationSample],
    plans: Mapping[str, TemplatePerturbationPlan],
    clean_scores: Mapping[str, Mapping[str, float | None]],
    errors: Sequence[str],
    severities: Sequence[float],
    seeds: Sequence[int],
    *,
    workers: int,
    index_path: Path,
    metadata_root: Path,
    layout_root: Path,
) -> list[PairedPageResult]:
    conditions = [
        (error, severity, seed)
        for error in errors
        for severity in severities
        for seed in seeds
    ]
    results: list[PairedPageResult] = []
    if workers == 1:
        for condition_index, (error, severity, seed) in enumerate(conditions, start=1):
            results.extend(
                run_condition(
                    samples, plans, clean_scores, error, severity, seed
                )
            )
            print(
                f"condition {condition_index}/{len(conditions)}: "
                f"{error} severity={severity:.2f} seed={seed}",
                flush=True,
            )
        return results

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_condition_worker,
        initargs=(str(index_path), str(metadata_root), str(layout_root)),
    ) as executor:
        futures = {
            executor.submit(_run_condition_worker, condition): condition
            for condition in conditions
        }
        completed = 0
        for future in as_completed(futures):
            condition = futures[future]
            results.extend(future.result())
            completed += 1
            print(
                f"condition {completed}/{len(conditions)}: "
                f"{condition[0]} severity={condition[1]:.2f} seed={condition[2]}",
                flush=True,
            )
    return sorted(
        results,
        key=lambda row: (
            ERRORS.index(row.error),
            row.severity,
            row.seed,
            row.sample_id,
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_identity_check(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        (
            "metric",
            "applicable_pages",
            "applicable_templates",
            "minimum_score",
            "mean_score",
            "maximum_score",
            "all_scores_equal_100",
        ),
    )


def write_paired_page_drops(path: Path, rows: Sequence[PairedPageResult]) -> None:
    fields = [
        "error",
        "severity",
        "seed",
        "sample_id",
        "template_name",
        "eligible_units",
        "injected_units",
        "actual_injection_pct",
    ]
    for metric in METRICS:
        fields.extend(
            (
                f"clean_{metric}",
                f"corrupted_{metric}",
                f"drop_pp_{metric}",
            )
        )
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in rows:
            row: dict[str, Any] = {
                "error": result.error,
                "severity": result.severity,
                "seed": result.seed,
                "sample_id": result.sample_id,
                "template_name": result.template_name,
                "eligible_units": result.eligible_units,
                "injected_units": result.injected_units,
                "actual_injection_pct": (
                    100.0 * result.injected_units / result.eligible_units
                ),
            }
            for metric in METRICS:
                clean = result.clean_scores[metric]
                corrupted = result.corrupted_scores[metric]
                row[f"clean_{metric}"] = "" if clean is None else f"{clean:.9f}"
                row[f"corrupted_{metric}"] = (
                    "" if corrupted is None else f"{corrupted:.9f}"
                )
                drop = result.drop(metric)
                row[f"drop_pp_{metric}"] = "" if drop is None else f"{drop:.9f}"
            writer.writerow(row)


def read_paired_page_drops(path: Path) -> list[PairedPageResult]:
    rows: list[PairedPageResult] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            clean_scores: dict[str, float | None] = {}
            corrupted_scores: dict[str, float | None] = {}
            for metric in METRICS:
                clean_value = raw[f"clean_{metric}"]
                corrupted_value = raw[f"corrupted_{metric}"]
                clean_scores[metric] = float(clean_value) if clean_value else None
                corrupted_scores[metric] = (
                    float(corrupted_value) if corrupted_value else None
                )
            rows.append(
                PairedPageResult(
                    error=raw["error"],
                    severity=float(raw["severity"]),
                    seed=int(raw["seed"]),
                    sample_id=raw["sample_id"],
                    template_name=raw["template_name"],
                    eligible_units=int(raw["eligible_units"]),
                    injected_units=int(raw["injected_units"]),
                    clean_scores=clean_scores,
                    corrupted_scores=corrupted_scores,
                )
            )
    return rows


def _derived_seed(base_seed: int, *parts: Any) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)]).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap_interval(
    template_values: Sequence[float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(template_values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    statistics = values[indices].mean(axis=1)
    lower, upper = np.quantile(statistics, [0.025, 0.975])
    return float(lower), float(upper)


def _template_drop_map(
    rows: Iterable[PairedPageResult], metric: str
) -> dict[str, float]:
    by_template: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        drop = row.drop(metric)
        if drop is not None:
            by_template[row.template_name].append(drop)
    return {
        template: float(np.mean(values))
        for template, values in by_template.items()
        if values
    }


def aggregate_severity_curves(
    rows: Sequence[PairedPageResult],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[PairedPageResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.error, row.severity)].append(row)
    output: list[dict[str, Any]] = []
    for error in ERRORS:
        for severity in sorted({row.severity for row in rows}):
            condition_rows = grouped.get((error, severity), [])
            if not condition_rows:
                continue
            for metric in METRICS:
                template_values = _template_drop_map(condition_rows, metric)
                if not template_values:
                    continue
                values = list(template_values.values())
                lower, upper = _bootstrap_interval(
                    values,
                    bootstrap_replicates,
                    _derived_seed(bootstrap_seed, error, severity, metric),
                )
                applicable_pages = {
                    row.sample_id
                    for row in condition_rows
                    if row.drop(metric) is not None
                }
                output.append(
                    {
                        "error": error,
                        "error_label": ERROR_LABELS[error],
                        "severity": severity,
                        "metric": metric,
                        "applicable_pages": len(applicable_pages),
                        "applicable_templates": len(template_values),
                        "mean_drop_pp": float(np.mean(values)),
                        "ci95_low_pp": lower,
                        "ci95_high_pp": upper,
                    }
                )
    return output


def aggregate_injection_rates(
    rows: Sequence[PairedPageResult],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int], list[PairedPageResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.error, row.severity, row.seed)].append(row)
    output: list[dict[str, Any]] = []
    for (error, severity, seed), condition_rows in sorted(grouped.items()):
        eligible = sum(row.eligible_units for row in condition_rows)
        injected = sum(row.injected_units for row in condition_rows)
        output.append(
            {
                "error": error,
                "severity": severity,
                "seed": seed,
                "applicable_pages": len({row.sample_id for row in condition_rows}),
                "applicable_templates": len(
                    {row.template_name for row in condition_rows}
                ),
                "eligible_units": eligible,
                "injected_units": injected,
                "actual_injection_pct": 100.0 * injected / eligible,
            }
        )
    return output


def _curve_lookup(
    curve_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, float, str], Mapping[str, Any]]:
    return {
        (str(row["error"]), float(row["severity"]), str(row["metric"])): row
        for row in curve_rows
    }


def build_response_matrix(
    curve_rows: Sequence[Mapping[str, Any]], severity: float = 0.25
) -> list[dict[str, Any]]:
    lookup = _curve_lookup(curve_rows)
    output: list[dict[str, Any]] = []
    for error in ERRORS:
        row: dict[str, Any] = {
            "error": error,
            "error_label": ERROR_LABELS[error],
            "severity": severity,
        }
        for metric in METRICS:
            result = lookup.get((error, severity, metric))
            row[metric] = "" if result is None else result["mean_drop_pp"]
        output.append(row)
    return output


def _seed_level_macro(
    rows: Sequence[PairedPageResult],
    error: str,
    severity: float,
    seed: int,
    metric: str,
) -> float | None:
    selected = [
        row
        for row in rows
        if row.error == error and row.severity == severity and row.seed == seed
    ]
    template_values = _template_drop_map(selected, metric)
    return float(np.mean(list(template_values.values()))) if template_values else None


def _target_monotonicity(
    rows: Sequence[PairedPageResult],
    error: str,
    metric: str,
) -> float | None:
    severities = sorted({row.severity for row in rows if row.error == error})
    seeds = sorted({row.seed for row in rows if row.error == error})
    severity_means: list[float] = []
    for severity in severities:
        seed_values: list[float] = []
        for seed in seeds:
            value = _seed_level_macro(rows, error, severity, seed, metric)
            if value is not None:
                seed_values.append(value)
        if seed_values:
            severity_means.append(float(np.mean(seed_values)))
    if len(severity_means) != len(severities) or len(set(severity_means)) < 2:
        return None
    statistic = spearmanr(severities, severity_means).statistic
    return None if np.isnan(statistic) else float(statistic)


def _selectivity_for_error(
    rows: Sequence[PairedPageResult],
    error: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    condition_rows = [
        row for row in rows if row.error == error and row.severity == 0.25
    ]
    templates = sorted({row.template_name for row in condition_rows})
    metric_maps = {
        metric: _template_drop_map(condition_rows, metric) for metric in METRICS
    }
    targets = PRIMARY_TARGETS[error]
    excluded = set(targets) | set(EXPECTED_DEPENDENCIES[error])
    off_targets = [metric for metric in METRICS if metric not in excluded]

    target_statistics = {
        metric: float(np.mean(list(metric_maps[metric].values())))
        for metric in targets
    }
    target_response = float(np.mean(list(target_statistics.values())))
    off_target_statistics = {
        metric: float(np.mean(list(metric_maps[metric].values())))
        for metric in off_targets
        if metric_maps[metric]
    }
    max_off_metric, max_off_response = max(
        off_target_statistics.items(), key=lambda item: (item[1], item[0])
    )

    rng = np.random.default_rng(_derived_seed(bootstrap_seed, error, "selectivity"))
    target_bootstrap: list[float] = []
    off_bootstrap: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled = rng.choice(templates, size=len(templates), replace=True)
        target_values: list[float] = []
        for metric in targets:
            values = [metric_maps[metric][template] for template in sampled]
            target_values.append(float(np.mean(values)))
        target_bootstrap.append(float(np.mean(target_values)))

        metric_values: list[float] = []
        for metric in off_targets:
            values = [
                metric_maps[metric][template]
                for template in sampled
                if template in metric_maps[metric]
            ]
            if values:
                metric_values.append(float(np.mean(values)))
        off_bootstrap.append(max(metric_values, default=0.0))
    target_low, target_high = np.quantile(target_bootstrap, [0.025, 0.975])
    off_low, off_high = np.quantile(off_bootstrap, [0.025, 0.975])
    return {
        "error": error,
        "error_label": ERROR_LABELS[error],
        "applicable_pages": len({row.sample_id for row in condition_rows}),
        "applicable_templates": len(templates),
        "target_metrics": " / ".join(targets),
        "target_response_pp": target_response,
        "target_ci95_low_pp": float(target_low),
        "target_ci95_high_pp": float(target_high),
        "max_off_target_metric": max_off_metric,
        "max_off_target_drop_pp": max_off_response,
        "off_target_ci95_low_pp": float(off_low),
        "off_target_ci95_high_pp": float(off_high),
        "excluded_expected_dependencies": " / ".join(EXPECTED_DEPENDENCIES[error]),
        "selectivity_margin_pp": target_response - max_off_response,
    }


def build_selectivity_rows(
    rows: Sequence[PairedPageResult],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    return [
        _selectivity_for_error(rows, error, bootstrap_replicates, bootstrap_seed)
        for error in ERRORS
    ]


def build_target_rows(
    rows: Sequence[PairedPageResult],
    curve_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _curve_lookup(curve_rows)
    output: list[dict[str, Any]] = []
    for error in ERRORS:
        applicable_pages = len(
            {row.sample_id for row in rows if row.error == error}
        )
        applicable_templates = len(
            {row.template_name for row in rows if row.error == error}
        )
        for metric in PRIMARY_TARGETS[error]:
            row: dict[str, Any] = {
                "error": error,
                "error_label": ERROR_LABELS[error],
                "applicable_pages": applicable_pages,
                "applicable_templates": applicable_templates,
                "target_metric": metric,
                "monotonicity_rho": _target_monotonicity(rows, error, metric),
            }
            for severity in (0.10, 0.25, 0.50):
                curve = lookup[(error, severity, metric)]
                suffix = str(int(severity * 100))
                row[f"drop_pp_{suffix}"] = curve["mean_drop_pp"]
                row[f"ci95_low_pp_{suffix}"] = curve["ci95_low_pp"]
                row[f"ci95_high_pp_{suffix}"] = curve["ci95_high_pp"]
            output.append(row)
    return output


def write_main_table(
    path: Path,
    target_rows: Sequence[Mapping[str, Any]],
    selectivity_rows: Sequence[Mapping[str, Any]],
) -> None:
    targets_by_error: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        targets_by_error[str(row["error"])].append(row)
    selectivity = {str(row["error"]): row for row in selectivity_rows}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled diagnostic validation on the 1,100-page test split. Drops are absolute percentage points from each page's clean gold baseline, followed by page averaging within template and macro-averaging across applicable test templates. Hierarchy cells report Schema-nTED/TSR-path. Max off-target is measured at 25\% after excluding the declared downstream dependencies shown in Fig.~\ref{fig:controlled-diagnostic}. Monotonicity is Spearman's $\rho$ over the three severity-level macro means after averaging five seeds.}",
        r"\label{tab:controlled-diagnostic}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l r l r r r c r@{}}",
        r"\toprule",
        r"Injected error & Pages & Target metric & $\Delta$@10\% & $\Delta$@25\% & $\Delta$@50\% & $\rho$ & Max off-target $\Delta$@25\% \\",
        r"\midrule",
    ]
    for error in ERRORS:
        target_group = targets_by_error[error]
        metric_cell = " / ".join(
            DISPLAY_METRICS[str(row["target_metric"])] for row in target_group
        )

        def joined(field: str, digits: int) -> str:
            return "/".join(
                (
                    "--"
                    if row.get(field) is None
                    else f"{float(row[field]):.{digits}f}"
                )
                for row in target_group
            )

        lines.append(
            " & ".join(
                (
                    ERROR_LABELS[error],
                    str(target_group[0]["applicable_pages"]),
                    metric_cell,
                    joined("drop_pp_10", 1),
                    joined("drop_pp_25", 1),
                    joined("drop_pp_50", 1),
                    joined("monotonicity_rho", 2),
                    f"{float(selectivity[error]['max_off_target_drop_pp']):.1f}",
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _curve_index(
    curve_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    output: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        output[(str(row["error"]), str(row["metric"]))].append(row)
    for values in output.values():
        values.sort(key=lambda row: float(row["severity"]))
    return output


def plot_main_figure(
    output_base: Path,
    response_rows: Sequence[Mapping[str, Any]],
    selectivity_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(
        [
            [float(row[metric]) if row[metric] != "" else np.nan for metric in METRICS]
            for row in response_rows
        ],
        dtype=np.float64,
    )
    maximum = max(1.0, float(np.nanmax(matrix)))
    figure, (matrix_axis, selectivity_axis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.2),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
        constrained_layout=True,
    )
    image = matrix_axis.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=maximum)
    matrix_axis.set_xticks(range(len(METRICS)))
    matrix_axis.set_xticklabels(
        [DISPLAY_METRICS[metric] for metric in METRICS], rotation=42, ha="right"
    )
    matrix_axis.set_yticks(range(len(ERRORS)))
    matrix_axis.set_yticklabels([ERROR_LABELS[error] for error in ERRORS])
    matrix_axis.set_title("(a) Metric response matrix at 25% injection", loc="left")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                label = "NA"
                color = "#4b5563"
            else:
                label = f"{value:.1f}"
                color = "white" if value > maximum * 0.55 else "#111827"
            matrix_axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Absolute drop (percentage points)")

    y_positions = np.arange(len(ERRORS), dtype=np.float64)
    width = 0.36
    target = np.asarray(
        [float(row["target_response_pp"]) for row in selectivity_rows]
    )
    off_target = np.asarray(
        [float(row["max_off_target_drop_pp"]) for row in selectivity_rows]
    )
    target_errors = np.asarray(
        [
            [
                float(row["target_response_pp"]) - float(row["target_ci95_low_pp"])
                for row in selectivity_rows
            ],
            [
                float(row["target_ci95_high_pp"]) - float(row["target_response_pp"])
                for row in selectivity_rows
            ],
        ]
    )
    off_errors = np.asarray(
        [
            [
                float(row["max_off_target_drop_pp"])
                - float(row["off_target_ci95_low_pp"])
                for row in selectivity_rows
            ],
            [
                float(row["off_target_ci95_high_pp"])
                - float(row["max_off_target_drop_pp"])
                for row in selectivity_rows
            ],
        ]
    )
    selectivity_axis.barh(
        y_positions - width / 2,
        target,
        height=width,
        xerr=target_errors,
        color="#c43c39",
        label="Target response",
        capsize=2,
    )
    selectivity_axis.barh(
        y_positions + width / 2,
        off_target,
        height=width,
        xerr=off_errors,
        color="#3f7f93",
        label="Max unrelated response",
        capsize=2,
    )
    selectivity_axis.set_yticks(y_positions)
    selectivity_axis.set_yticklabels([ERROR_LABELS[error] for error in ERRORS])
    selectivity_axis.invert_yaxis()
    selectivity_axis.set_xlabel("Absolute drop at 25% (percentage points)")
    selectivity_axis.set_title("(b) Diagnostic selectivity", loc="left")
    selectivity_axis.grid(axis="x", color="#d1d5db", linewidth=0.7)
    selectivity_axis.set_axisbelow(True)
    selectivity_axis.legend(frameon=False, loc="lower right")
    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=300)
    plt.close(figure)


def plot_severity_curves(
    output_base: Path,
    curve_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = _curve_index(curve_rows)
    figure, axes = plt.subplots(
        4, 2, figsize=(11.5, 13.0), sharex=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P", "X", "h")
    for error_index, error in enumerate(ERRORS):
        axis = axes.flat[error_index]
        for metric_index, metric in enumerate(METRICS):
            values = index.get((error, metric), [])
            if not values:
                continue
            x = np.asarray([100.0 * float(row["severity"]) for row in values])
            y = np.asarray([float(row["mean_drop_pp"]) for row in values])
            low = np.asarray([float(row["ci95_low_pp"]) for row in values])
            high = np.asarray([float(row["ci95_high_pp"]) for row in values])
            axis.plot(
                x,
                y,
                marker=markers[metric_index],
                color=colors[metric_index],
                linewidth=1.5,
                markersize=4,
                label=DISPLAY_METRICS[metric],
            )
            axis.fill_between(x, low, high, color=colors[metric_index], alpha=0.10)
        axis.set_title(f"({chr(ord('a') + error_index)}) {ERROR_LABELS[error]}", loc="left")
        axis.set_ylabel("Drop (pp)")
        axis.grid(color="#e5e7eb", linewidth=0.7)
        axis.set_axisbelow(True)
    legend_axis = axes.flat[7]
    legend_axis.axis("off")
    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("Injected units (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(
    path: Path,
    target_rows: Sequence[Mapping[str, Any]],
    selectivity_rows: Sequence[Mapping[str, Any]],
) -> None:
    target_by_error: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        target_by_error[str(row["error"])].append(row)
    selectivity = {str(row["error"]): row for row in selectivity_rows}
    lines = [
        "# Controlled Diagnostic Validation",
        "",
        "Gold annotations from the 1,100-page, 11-template test split are treated as perfect predictions. Each run changes one structural factor only. Drops are absolute percentage points.",
        "",
        "| Error | Pages | Target metric | Drop @10% | Drop @25% | Drop @50% | rho | Max unrelated @25% |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for error in ERRORS:
        group = target_by_error[error]

        def join(field: str, digits: int = 2) -> str:
            return "/".join(
                "NA" if row.get(field) is None else f"{float(row[field]):.{digits}f}"
                for row in group
            )

        lines.append(
            "| "
            + " | ".join(
                (
                    ERROR_LABELS[error],
                    str(group[0]["applicable_pages"]),
                    "/".join(DISPLAY_METRICS[str(row["target_metric"])] for row in group),
                    join("drop_pp_10"),
                    join("drop_pp_25"),
                    join("drop_pp_50"),
                    join("monotonicity_rho"),
                    f"{float(selectivity[error]['max_off_target_drop_pp']):.2f}",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Confidence intervals in `severity_curves.csv` use a template-clustered bootstrap. Expected downstream responses are shown in the response matrix but excluded from the unrelated-response statistic.",
            "",
            "`paired_page_drops.csv` is the page-paired source of every aggregate. `injection_rates.csv` records realized rates for all five fixed seeds.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate metric selectivity with isolated gold perturbations."
    )
    parser.add_argument(
        "--index",
        default="outputs/dataset_splits/template_stratified_seed42/test_index.jsonl",
    )
    parser.add_argument("--metadata-root", default="new-dataset-json")
    parser.add_argument("--layout-root", default="newdataset-layout")
    parser.add_argument(
        "--out", default="outputs/aux_exp/controlled_diagnostic"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--severities", default="0.10,0.25,0.50")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild aggregate tables and figures from an existing paired_page_drops.csv.",
    )
    parser.add_argument(
        "--allow-nonstandard-scope",
        action="store_true",
        help="Allow a scope other than the formal 1,100 pages and 11 templates.",
    )
    return parser.parse_args()


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")
    seeds = _parse_int_list(args.seeds)
    severities = _parse_float_list(args.severities)
    if len(set(seeds)) != len(seeds):
        raise ValueError("fixed random seeds must be unique")
    if sorted(severities) != list(severities) or any(
        not 0.0 < severity < 1.0 for severity in severities
    ):
        raise ValueError("severities must be unique, increasing, and in (0, 1)")
    if not args.allow_nonstandard_scope:
        if seeds != (0, 1, 2, 3, 4):
            raise ValueError("formal protocol requires fixed seeds 0,1,2,3,4")
        if severities != (0.10, 0.25, 0.50):
            raise ValueError("formal protocol requires severities 0.10,0.25,0.50")

    index_path = Path(args.index)
    metadata_root = Path(args.metadata_root)
    layout_root = Path(args.layout_root)
    out_dir = ensure_dir(Path(args.out))
    samples = load_samples(index_path, metadata_root, layout_root)
    templates = sorted({sample.template_name for sample in samples})
    if not args.allow_nonstandard_scope and (len(samples), len(templates)) != (1100, 11):
        raise ValueError(
            "formal controlled diagnostic scope must contain exactly "
            f"1,100 pages and 11 templates, got {len(samples)} and {len(templates)}"
        )

    print("running gold-as-prediction identity check", flush=True)
    clean_scores, identity_rows = verify_gold_identity(samples)
    write_identity_check(out_dir / "gold_identity_check.csv", identity_rows)
    print("gold identity check passed for all applicable metrics", flush=True)

    if args.report_only:
        paired_path = out_dir / "paired_page_drops.csv"
        if not paired_path.exists():
            raise FileNotFoundError(
                f"--report-only requires an existing {paired_path}"
            )
        paired_rows = read_paired_page_drops(paired_path)
    else:
        plans: dict[str, TemplatePerturbationPlan] = {}
        for sample in samples:
            if sample.template_name not in plans:
                plans[sample.template_name] = build_template_plan(sample.template)
        paired_rows = run_all_conditions(
            samples,
            plans,
            clean_scores,
            ERRORS,
            severities,
            seeds,
            workers=args.workers,
            index_path=index_path,
            metadata_root=metadata_root,
            layout_root=layout_root,
        )
        write_paired_page_drops(out_dir / "paired_page_drops.csv", paired_rows)

    curve_rows = aggregate_severity_curves(
        paired_rows, args.bootstrap_replicates, args.bootstrap_seed
    )
    injection_rows = aggregate_injection_rates(paired_rows)
    response_rows = build_response_matrix(curve_rows)
    selectivity_rows = build_selectivity_rows(
        paired_rows, args.bootstrap_replicates, args.bootstrap_seed
    )
    target_rows = build_target_rows(paired_rows, curve_rows)

    _write_csv(
        out_dir / "severity_curves.csv",
        curve_rows,
        (
            "error",
            "error_label",
            "severity",
            "metric",
            "applicable_pages",
            "applicable_templates",
            "mean_drop_pp",
            "ci95_low_pp",
            "ci95_high_pp",
        ),
    )
    _write_csv(
        out_dir / "injection_rates.csv",
        injection_rows,
        (
            "error",
            "severity",
            "seed",
            "applicable_pages",
            "applicable_templates",
            "eligible_units",
            "injected_units",
            "actual_injection_pct",
        ),
    )
    _write_csv(
        out_dir / "metric_response_matrix_25pct.csv",
        response_rows,
        ("error", "error_label", "severity", *METRICS),
    )
    _write_csv(
        out_dir / "diagnostic_selectivity_25pct.csv",
        selectivity_rows,
        (
            "error",
            "error_label",
            "applicable_pages",
            "applicable_templates",
            "target_metrics",
            "target_response_pp",
            "target_ci95_low_pp",
            "target_ci95_high_pp",
            "max_off_target_metric",
            "max_off_target_drop_pp",
            "off_target_ci95_low_pp",
            "off_target_ci95_high_pp",
            "excluded_expected_dependencies",
            "selectivity_margin_pp",
        ),
    )
    _write_csv(
        out_dir / "target_metric_summary.csv",
        target_rows,
        (
            "error",
            "error_label",
            "applicable_pages",
            "applicable_templates",
            "target_metric",
            "drop_pp_10",
            "ci95_low_pp_10",
            "ci95_high_pp_10",
            "drop_pp_25",
            "ci95_low_pp_25",
            "ci95_high_pp_25",
            "drop_pp_50",
            "ci95_low_pp_50",
            "ci95_high_pp_50",
            "monotonicity_rho",
        ),
    )
    write_main_table(
        out_dir / "controlled_diagnostic_table.tex", target_rows, selectivity_rows
    )
    plot_main_figure(
        out_dir / "controlled_diagnostic_figure", response_rows, selectivity_rows
    )
    plot_severity_curves(
        out_dir / "controlled_diagnostic_severity_curves", curve_rows
    )
    write_readme(out_dir / "README.md", target_rows, selectivity_rows)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "controlled diagnostic validation with gold annotations as predictions",
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "n_pages": len(samples),
        "n_templates": len(templates),
        "templates": templates,
        "severities": list(severities),
        "fixed_seeds": list(seeds),
        "bootstrap": {
            "method": "template-clustered percentile bootstrap over template means",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "aggregation": (
            "paired page drop in percentage points; mean within template over pages "
            "and five seeds; macro mean over applicable templates"
        ),
        "applicability": (
            "pages without a corruptible unit are excluded from that corruption; "
            "metric-level NA follows the current evaluator"
        ),
        "primary_targets": PRIMARY_TARGETS,
        "expected_dependencies_excluded_from_off_target": EXPECTED_DEPENDENCIES,
        "line_item_protocol": (
            "primitive item boxes remain fixed; spatially inferred member/nonmember "
            "assignments are swapped and the canonical group envelope is rebuilt"
        ),
        "identity_check": identity_rows,
        "relation_reachability_audit": {
            sample.template_name: {
                "recoverable_typed_relations": sample.template.audit.get(
                    "recoverable_typed_relations",
                    len(sample.template.relations),
                ),
                "unrecoverable_typed_relations_excluded": sample.template.audit.get(
                    "unrecoverable_typed_relations_excluded", 0
                ),
                "unrecoverable_relation_endpoints": sample.template.audit.get(
                    "unrecoverable_relation_endpoints", 0
                ),
            }
            for sample in {
                item.template_name: item for item in samples
            }.values()
        },
        "drop_unit": "absolute percentage points",
        "output_rows": {
            "paired_page_drops": len(paired_rows),
            "severity_curves": len(curve_rows),
            "target_metric_summary": len(target_rows),
        },
    }
    write_json(out_dir / "controlled_diagnostic_metadata.json", metadata)
    print(f"wrote controlled diagnostic report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
