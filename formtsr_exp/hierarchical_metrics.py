from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import NA


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GridCell:
    id: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    bbox: BBox | None = None


@dataclass(frozen=True, slots=True)
class LocalGrid:
    id: str
    parent_region_id: str
    cells: tuple[GridCell, ...]
    bbox: BBox | None = None


@dataclass(frozen=True, slots=True)
class Widget:
    id: str
    bbox: BBox | None
    widget_type: str
    state: str


@dataclass(frozen=True, slots=True)
class WidgetGroup:
    id: str
    group_type: str
    members: tuple[Widget, ...]


@dataclass(frozen=True, slots=True, order=True)
class Relation:
    source: str
    relation_type: str
    target: str


@dataclass(frozen=True, slots=True)
class GritsTopResult:
    f1: float
    precision: float
    recall: float
    upper_bound: float
    aligned_gt_rows: tuple[int, ...]
    aligned_pred_rows: tuple[int, ...]
    aligned_gt_cols: tuple[int, ...]
    aligned_pred_cols: tuple[int, ...]
    cell_mapping: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LocalGridScore:
    score: float | str
    similarity_sum: float
    n_pred: int
    n_gt: int
    matches: tuple[tuple[int, int, float], ...]
    cell_mapping: Mapping[str, str]
    eligible_pairs: int


@dataclass(frozen=True, slots=True)
class WidgetGroupScore:
    score: float | str
    similarity_sum: float
    n_pred: int
    n_gt: int
    matches: tuple[tuple[int, int, float], ...]
    member_matches: int


@dataclass(frozen=True, slots=True)
class RelationCounts:
    tp: int
    pred: int
    gt: int

    @property
    def precision(self) -> float | str:
        if self.pred == 0:
            return NA if self.gt == 0 else 0.0
        return self.tp / self.pred

    @property
    def recall(self) -> float | str:
        if self.gt == 0:
            return NA if self.pred == 0 else 0.0
        return self.tp / self.gt

    @property
    def f1(self) -> float | str:
        denominator = self.pred + self.gt
        return 2.0 * self.tp / denominator if denominator else NA


@dataclass(frozen=True, slots=True)
class RelationScore:
    counts: RelationCounts
    matched_endpoint_counts: RelationCounts
    by_type: Mapping[str, RelationCounts]
    mapped_pred_relations: tuple[Relation, ...]


def bbox_iou(left: BBox | None, right: BBox | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def maximum_weight_matching(
    weights: np.ndarray,
    eligible: np.ndarray | None = None,
    *,
    cardinality_first: bool = False,
) -> list[tuple[int, int]]:
    """Return an optional one-to-one matching, leaving either side unmatched."""
    if weights.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    n_left, n_right = weights.shape
    if n_left == 0 or n_right == 0:
        return []
    if eligible is None:
        eligible = np.ones_like(weights, dtype=bool)
    if eligible.shape != weights.shape:
        raise ValueError("eligible and weights must have the same shape")
    if not cardinality_first:
        # Zero-weight edges cannot improve the maximum total similarity and do
        # not establish a useful downstream endpoint correspondence.
        eligible = eligible & (weights > 0)
    if not eligible.any():
        return []

    # Dummy vertices make the assignment optional.
    size = n_left + n_right
    augmented = np.zeros((size, size), dtype=np.float64)
    augmented[:n_left, :n_right] = -1e12
    if cardinality_first:
        finite = np.where(eligible, np.maximum(weights, 0.0), 0.0)
        scale = float(finite.max()) or 1.0
        real_weights = 1.0 + finite / scale * 1e-6
    else:
        real_weights = weights.astype(np.float64, copy=False)
    augmented[:n_left, :n_right][eligible] = real_weights[eligible]

    left_indices, right_indices = linear_sum_assignment(augmented, maximize=True)
    pairs = [
        (int(left), int(right))
        for left, right in zip(left_indices, right_indices)
        if left < n_left and right < n_right and eligible[left, right]
    ]
    return sorted(pairs)


def _traceback(pointers: np.ndarray) -> tuple[list[int], list[int]]:
    left_index = pointers.shape[0] - 1
    right_index = pointers.shape[1] - 1
    aligned_left: list[int] = []
    aligned_right: list[int] = []
    while left_index or right_index:
        pointer = pointers[left_index, right_index]
        if pointer == -1:
            left_index -= 1
        elif pointer == 1:
            right_index -= 1
        else:
            left_index -= 1
            right_index -= 1
            aligned_left.append(left_index)
            aligned_right.append(right_index)
    return aligned_left[::-1], aligned_right[::-1]


def _align_1d(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
    rewards: Mapping[tuple[int, int, int, int], float],
) -> float:
    scores = np.zeros((len(left) + 1, len(right) + 1), dtype=np.float64)
    for left_index in range(1, len(left) + 1):
        for right_index in range(1, len(right) + 1):
            reward = rewards[left[left_index - 1] + right[right_index - 1]]
            scores[left_index, right_index] = max(
                scores[left_index - 1, right_index - 1] + reward,
                scores[left_index - 1, right_index],
                scores[left_index, right_index - 1],
            )
    return float(scores[-1, -1])


def _align_2d_outer(
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
    rewards: Mapping[tuple[int, int, int, int], float],
) -> tuple[list[int], list[int], float]:
    scores = np.zeros((left_shape[0] + 1, right_shape[0] + 1), dtype=np.float64)
    pointers = np.zeros_like(scores, dtype=np.int8)
    pointers[1:, 0] = -1
    pointers[0, 1:] = 1
    for left_index in range(1, left_shape[0] + 1):
        for right_index in range(1, right_shape[0] + 1):
            reward = _align_1d(
                [(left_index - 1, col) for col in range(left_shape[1])],
                [(right_index - 1, col) for col in range(right_shape[1])],
                rewards,
            )
            diagonal = scores[left_index - 1, right_index - 1] + reward
            skip_left = scores[left_index - 1, right_index]
            skip_right = scores[left_index, right_index - 1]
            best = max(diagonal, skip_left, skip_right)
            scores[left_index, right_index] = best
            if diagonal == best:
                pointers[left_index, right_index] = 0
            elif skip_left == best:
                pointers[left_index, right_index] = -1
            else:
                pointers[left_index, right_index] = 1
    aligned_left, aligned_right = _traceback(pointers)
    return aligned_left, aligned_right, float(scores[-1, -1])


def _span_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    """Match the reference GriTS relative-span rectangle overlap."""
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ux1 = min(left[0], right[0])
    uy1 = min(left[1], right[1])
    ux2 = max(left[2], right[2])
    uy2 = max(left[3], right[3])
    enclosing_area = max(0, ux2 - ux1) * max(0, uy2 - uy1)
    return intersection / enclosing_area if enclosing_area else 0.0


def _relative_span_grid(
    grid: LocalGrid,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[str]]] | None:
    if not grid.cells:
        return None
    if any(
        cell.row < 0
        or cell.col < 0
        or cell.rowspan < 1
        or cell.colspan < 1
        for cell in grid.cells
    ):
        return None
    n_rows = max(cell.row + cell.rowspan for cell in grid.cells)
    n_cols = max(cell.col + cell.colspan for cell in grid.cells)
    if n_rows <= 0 or n_cols <= 0:
        return None
    spans: list[list[tuple[int, int, int, int] | None]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]
    cell_ids: list[list[str | None]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    for cell in grid.cells:
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                if spans[row][col] is not None:
                    return None
                spans[row][col] = (
                    cell.col - col,
                    cell.row - row,
                    cell.col + cell.colspan - col,
                    cell.row + cell.rowspan - row,
                )
                cell_ids[row][col] = cell.id
    if any(value is None for row in spans for value in row):
        return None
    return (
        [[value for value in row if value is not None] for row in spans],
        [[value for value in row if value is not None] for row in cell_ids],
    )


def grits_top(pred_grid: LocalGrid, gt_grid: LocalGrid) -> GritsTopResult | None:
    """Compute reference-compatible factored 2D-MSS GriTS_Top."""
    pred_data = _relative_span_grid(pred_grid)
    gt_data = _relative_span_grid(gt_grid)
    if pred_data is None or gt_data is None:
        return None
    pred_spans, pred_ids = pred_data
    gt_spans, gt_ids = gt_data
    gt_shape = (len(gt_spans), len(gt_spans[0]))
    pred_shape = (len(pred_spans), len(pred_spans[0]))

    rewards: dict[tuple[int, int, int, int], float] = {}
    transposed_rewards: dict[tuple[int, int, int, int], float] = {}
    for gt_row, gt_col, pred_row, pred_col in itertools.product(
        range(gt_shape[0]),
        range(gt_shape[1]),
        range(pred_shape[0]),
        range(pred_shape[1]),
    ):
        reward = _span_iou(gt_spans[gt_row][gt_col], pred_spans[pred_row][pred_col])
        rewards[(gt_row, gt_col, pred_row, pred_col)] = reward
        transposed_rewards[(gt_col, gt_row, pred_col, pred_row)] = reward

    gt_rows, pred_rows, row_score = _align_2d_outer(gt_shape, pred_shape, rewards)
    gt_cols, pred_cols, col_score = _align_2d_outer(
        gt_shape[::-1], pred_shape[::-1], transposed_rewards
    )
    n_gt = gt_shape[0] * gt_shape[1]
    n_pred = pred_shape[0] * pred_shape[1]
    upper_tp = min(row_score, col_score)
    upper_bound = 2.0 * upper_tp / (n_gt + n_pred)

    positive_score = 0.0
    cell_pair_weights: dict[tuple[str, str], float] = {}
    for gt_row, pred_row in zip(gt_rows, pred_rows):
        for gt_col, pred_col in zip(gt_cols, pred_cols):
            reward = rewards[(gt_row, gt_col, pred_row, pred_col)]
            positive_score += reward
            if reward > 0:
                key = (pred_ids[pred_row][pred_col], gt_ids[gt_row][gt_col])
                cell_pair_weights[key] = cell_pair_weights.get(key, 0.0) + reward

    pred_cell_ids = sorted({cell.id for cell in pred_grid.cells})
    gt_cell_ids = sorted({cell.id for cell in gt_grid.cells})
    cell_mapping: dict[str, str] = {}
    if pred_cell_ids and gt_cell_ids and cell_pair_weights:
        pred_lookup = {cell_id: index for index, cell_id in enumerate(pred_cell_ids)}
        gt_lookup = {cell_id: index for index, cell_id in enumerate(gt_cell_ids)}
        cell_weights = np.zeros((len(pred_cell_ids), len(gt_cell_ids)), dtype=np.float64)
        cell_eligible = np.zeros_like(cell_weights, dtype=bool)
        for (pred_id, gt_id), reward in cell_pair_weights.items():
            cell_weights[pred_lookup[pred_id], gt_lookup[gt_id]] = reward
            cell_eligible[pred_lookup[pred_id], gt_lookup[gt_id]] = True
        for pred_index, gt_index in maximum_weight_matching(cell_weights, cell_eligible):
            cell_mapping[pred_cell_ids[pred_index]] = gt_cell_ids[gt_index]

    return GritsTopResult(
        f1=2.0 * positive_score / (n_gt + n_pred),
        precision=positive_score / n_pred,
        recall=positive_score / n_gt,
        upper_bound=upper_bound,
        aligned_gt_rows=tuple(gt_rows),
        aligned_pred_rows=tuple(pred_rows),
        aligned_gt_cols=tuple(gt_cols),
        aligned_pred_cols=tuple(pred_cols),
        cell_mapping=cell_mapping,
    )


def lg_grits_top(
    pred_grids: Sequence[LocalGrid],
    gt_grids: Sequence[LocalGrid],
    region_mapping: Mapping[str, str],
    *,
    bbox_iou_threshold: float = 0.5,
) -> LocalGridScore:
    denominator = len(pred_grids) + len(gt_grids)
    if denominator == 0:
        return LocalGridScore(NA, 0.0, 0, 0, (), {}, 0)

    pred_counts: dict[str, int] = {}
    gt_counts: dict[str, int] = {}
    for grid in pred_grids:
        pred_counts[grid.parent_region_id] = pred_counts.get(grid.parent_region_id, 0) + 1
    for grid in gt_grids:
        gt_counts[grid.parent_region_id] = gt_counts.get(grid.parent_region_id, 0) + 1

    weights = np.zeros((len(pred_grids), len(gt_grids)), dtype=np.float64)
    eligible = np.zeros_like(weights, dtype=bool)
    alignments: dict[tuple[int, int], GritsTopResult | None] = {}
    for pred_index, pred_grid in enumerate(pred_grids):
        matched_parent = region_mapping.get(pred_grid.parent_region_id)
        for gt_index, gt_grid in enumerate(gt_grids):
            if matched_parent != gt_grid.parent_region_id:
                continue
            singleton_fallback = (
                pred_counts.get(pred_grid.parent_region_id) == 1
                and gt_counts.get(gt_grid.parent_region_id) == 1
            )
            if bbox_iou(pred_grid.bbox, gt_grid.bbox) < bbox_iou_threshold and not singleton_fallback:
                continue
            eligible[pred_index, gt_index] = True
            alignment = grits_top(pred_grid, gt_grid)
            alignments[(pred_index, gt_index)] = alignment
            weights[pred_index, gt_index] = alignment.f1 if alignment is not None else 0.0

    pairs = maximum_weight_matching(weights, eligible)
    matches: list[tuple[int, int, float]] = []
    cell_mapping: dict[str, str] = {}
    for pred_index, gt_index in pairs:
        similarity = float(weights[pred_index, gt_index])
        matches.append((pred_index, gt_index, similarity))
        alignment = alignments[(pred_index, gt_index)]
        if alignment is not None:
            cell_mapping.update(alignment.cell_mapping)
    similarity_sum = sum(item[2] for item in matches)
    return LocalGridScore(
        score=2.0 * similarity_sum / denominator,
        similarity_sum=similarity_sum,
        n_pred=len(pred_grids),
        n_gt=len(gt_grids),
        matches=tuple(matches),
        cell_mapping=cell_mapping,
        eligible_pairs=int(eligible.sum()),
    )


def _widget_member_mapping(
    pred_members: Sequence[Widget], gt_members: Sequence[Widget]
) -> list[tuple[int, int]]:
    if len(pred_members) == 1 and len(gt_members) == 1:
        pred_widget = pred_members[0]
        gt_widget = gt_members[0]
        return (
            [(0, 0)]
            if bbox_iou(pred_widget.bbox, gt_widget.bbox) >= 0.5
            and pred_widget.widget_type == gt_widget.widget_type
            and pred_widget.state == gt_widget.state
            else []
        )
    if len(pred_members) == 1:
        pred_widget = pred_members[0]
        candidates = [
            (bbox_iou(pred_widget.bbox, gt_widget.bbox), gt_index)
            for gt_index, gt_widget in enumerate(gt_members)
            if pred_widget.widget_type == gt_widget.widget_type
            and pred_widget.state == gt_widget.state
            and bbox_iou(pred_widget.bbox, gt_widget.bbox) >= 0.5
        ]
        return [(0, max(candidates)[1])] if candidates else []
    if len(gt_members) == 1:
        gt_widget = gt_members[0]
        candidates = [
            (bbox_iou(pred_widget.bbox, gt_widget.bbox), pred_index)
            for pred_index, pred_widget in enumerate(pred_members)
            if pred_widget.widget_type == gt_widget.widget_type
            and pred_widget.state == gt_widget.state
            and bbox_iou(pred_widget.bbox, gt_widget.bbox) >= 0.5
        ]
        return [(max(candidates)[1], 0)] if candidates else []
    weights = np.zeros((len(pred_members), len(gt_members)), dtype=np.float64)
    eligible = np.zeros_like(weights, dtype=bool)
    for pred_index, pred_widget in enumerate(pred_members):
        for gt_index, gt_widget in enumerate(gt_members):
            overlap = bbox_iou(pred_widget.bbox, gt_widget.bbox)
            is_match = (
                overlap >= 0.5
                and pred_widget.widget_type == gt_widget.widget_type
                and pred_widget.state == gt_widget.state
            )
            eligible[pred_index, gt_index] = is_match
            weights[pred_index, gt_index] = overlap
    return maximum_weight_matching(weights, eligible, cardinality_first=True)


def match_widgets_for_relations(
    pred_widgets: Sequence[Widget], gt_widgets: Sequence[Widget]
) -> dict[str, str]:
    """Freeze relation endpoint matches using type and IoU, ignoring state."""
    weights = np.zeros((len(pred_widgets), len(gt_widgets)), dtype=np.float64)
    eligible = np.zeros_like(weights, dtype=bool)
    for pred_index, pred_widget in enumerate(pred_widgets):
        for gt_index, gt_widget in enumerate(gt_widgets):
            overlap = bbox_iou(pred_widget.bbox, gt_widget.bbox)
            eligible[pred_index, gt_index] = (
                pred_widget.widget_type == gt_widget.widget_type and overlap >= 0.5
            )
            weights[pred_index, gt_index] = overlap
    return {
        pred_widgets[pred_index].id: gt_widgets[gt_index].id
        for pred_index, gt_index in maximum_weight_matching(
            weights, eligible, cardinality_first=True
        )
    }


def widget_group_f1(
    pred_groups: Sequence[WidgetGroup], gt_groups: Sequence[WidgetGroup]
) -> WidgetGroupScore:
    denominator = len(pred_groups) + len(gt_groups)
    if denominator == 0:
        return WidgetGroupScore(NA, 0.0, 0, 0, (), 0)
    weights = np.zeros((len(pred_groups), len(gt_groups)), dtype=np.float64)
    eligible = np.zeros_like(weights, dtype=bool)
    member_counts: dict[tuple[int, int], int] = {}
    for pred_index, pred_group in enumerate(pred_groups):
        for gt_index, gt_group in enumerate(gt_groups):
            if pred_group.group_type != gt_group.group_type:
                continue
            eligible[pred_index, gt_index] = True
            member_mapping = _widget_member_mapping(pred_group.members, gt_group.members)
            count = len(member_mapping)
            member_counts[(pred_index, gt_index)] = count
            member_denominator = len(pred_group.members) + len(gt_group.members)
            weights[pred_index, gt_index] = (
                2.0 * count / member_denominator if member_denominator else 0.0
            )
    pairs = maximum_weight_matching(weights, eligible)
    matches = tuple(
        (pred_index, gt_index, float(weights[pred_index, gt_index]))
        for pred_index, gt_index in pairs
    )
    similarity_sum = sum(item[2] for item in matches)
    return WidgetGroupScore(
        score=2.0 * similarity_sum / denominator,
        similarity_sum=similarity_sum,
        n_pred=len(pred_groups),
        n_gt=len(gt_groups),
        matches=matches,
        member_matches=sum(member_counts.get((left, right), 0) for left, right in pairs),
    )


def _canonical_relation(relation: Relation, symmetric_types: set[str]) -> Relation:
    source = relation.source
    target = relation.target
    if relation.relation_type in symmetric_types and target < source:
        source, target = target, source
    return Relation(source, relation.relation_type, target)


def relation_f1(
    pred_relations: Iterable[Relation],
    gt_relations: Iterable[Relation],
    endpoint_mapping: Mapping[str, str],
    *,
    symmetric_types: Iterable[str] = (),
) -> RelationScore:
    symmetric = {str(value) for value in symmetric_types}
    pred_set = {_canonical_relation(relation, symmetric) for relation in pred_relations}
    gt_set = {_canonical_relation(relation, symmetric) for relation in gt_relations}
    mapped_gt_nodes = set(endpoint_mapping.values())

    mapped_pred: set[Relation] = set()
    pred_with_matched_endpoints = 0
    for relation in pred_set:
        source = endpoint_mapping.get(relation.source)
        target = endpoint_mapping.get(relation.target)
        if source is None or target is None:
            continue
        pred_with_matched_endpoints += 1
        mapped_pred.add(
            _canonical_relation(Relation(source, relation.relation_type, target), symmetric)
        )
    true_positives = mapped_pred & gt_set
    gt_with_matched_endpoints = {
        relation
        for relation in gt_set
        if relation.source in mapped_gt_nodes and relation.target in mapped_gt_nodes
    }

    relation_types = sorted(
        {relation.relation_type for relation in pred_set}
        | {relation.relation_type for relation in gt_set}
    )
    by_type: dict[str, RelationCounts] = {}
    for relation_type in relation_types:
        by_type[relation_type] = RelationCounts(
            tp=sum(item.relation_type == relation_type for item in true_positives),
            pred=sum(item.relation_type == relation_type for item in pred_set),
            gt=sum(item.relation_type == relation_type for item in gt_set),
        )

    return RelationScore(
        counts=RelationCounts(len(true_positives), len(pred_set), len(gt_set)),
        matched_endpoint_counts=RelationCounts(
            len(true_positives), pred_with_matched_endpoints, len(gt_with_matched_endpoints)
        ),
        by_type=by_type,
        mapped_pred_relations=tuple(sorted(mapped_pred)),
    )


def flatten_widgets(groups: Sequence[WidgetGroup]) -> list[Widget]:
    output: list[Widget] = []
    seen: set[str] = set()
    for group in groups:
        for widget in group.members:
            if widget.id not in seen:
                seen.add(widget.id)
                output.append(widget)
    return output


def f1_from_counts(tp: int, pred: int, gt: int) -> float | str:
    denominator = pred + gt
    return 2.0 * tp / denominator if denominator else NA


def numeric_mean(values: Iterable[float | str]) -> float | str:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else NA


def as_serializable_counts(counts: RelationCounts) -> dict[str, Any]:
    return {
        "tp": counts.tp,
        "pred": counts.pred,
        "gt": counts.gt,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
    }
