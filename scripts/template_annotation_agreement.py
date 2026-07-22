#!/usr/bin/env python3
"""Compute template-only agreement between the raw and reviewed annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from formtsr_exp.hierarchical_metrics import (
    LocalGrid,
    RelationCounts,
    Widget,
    bbox_iou,
    grits_top,
    maximum_weight_matching,
    relation_f1,
)
from formtsr_exp.hierarchical_metrics_report import (
    RegionNode,
    TemplateStructure,
    load_raw_template_structure,
    load_template_structure,
    reconcile_with_r_f1_structure,
)


TEST_TEMPLATES = (
    "Arabic-2",
    "Arabic-7",
    "de_2",
    "en_15",
    "en_5",
    "en_7",
    "en_9",
    "ja_20",
    "ja_6",
    "pt_2",
    "zn_4",
)


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right):
        raise ValueError("rating vectors must have equal length")
    if not left:
        return None
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] / total * right_counts[label] / total
        for label in set(left_counts) | set(right_counts)
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def match_regions_by_geometry(
    reviewed: Sequence[RegionNode],
    existing: Sequence[RegionNode],
    *,
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    weights = np.asarray(
        [[bbox_iou(left.bbox, right.bbox) for right in existing] for left in reviewed],
        dtype=np.float64,
    ).reshape((len(reviewed), len(existing)))
    pairs = maximum_weight_matching(
        weights,
        weights >= threshold,
        cardinality_first=True,
    )
    return pairs, weights


def topology_match(
    reviewed: Sequence[LocalGrid], existing: Sequence[LocalGrid]
) -> tuple[float | None, int, int, int]:
    """Match grids spatially, then score only their cell/span topology."""
    if not reviewed and not existing:
        return None, 0, 0, 0
    weights = np.zeros((len(reviewed), len(existing)), dtype=np.float64)
    eligible = np.zeros_like(weights, dtype=bool)
    for reviewed_index, reviewed_grid in enumerate(reviewed):
        for existing_index, existing_grid in enumerate(existing):
            overlap = bbox_iou(reviewed_grid.bbox, existing_grid.bbox)
            if overlap < 0.5:
                continue
            alignment = grits_top(reviewed_grid, existing_grid)
            if alignment is None:
                continue
            eligible[reviewed_index, existing_index] = True
            weights[reviewed_index, existing_index] = alignment.f1
    pairs = maximum_weight_matching(weights, eligible)
    similarity_sum = sum(float(weights[left, right]) for left, right in pairs)
    denominator = len(reviewed) + len(existing)
    score = 2.0 * similarity_sum / denominator if denominator else 1.0
    return score, len(pairs), len(reviewed), len(existing)


def blank_widgets(template: TemplateStructure) -> list[Widget]:
    return [
        Widget(spec.id, spec.bbox, spec.widget_type, "unselected")
        for spec in template.widget_specs
    ]


def widget_object_match(
    reviewed: Sequence[Widget], existing: Sequence[Widget]
) -> tuple[float | None, int]:
    if not reviewed and not existing:
        return None, 0
    weights = np.asarray(
        [[bbox_iou(left.bbox, right.bbox) for right in existing] for left in reviewed],
        dtype=np.float64,
    ).reshape((len(reviewed), len(existing)))
    eligible = np.asarray(
        [
            [
                weights[left_index, right_index] >= 0.5
                and left.widget_type == right.widget_type
                and left.state == right.state
                for right_index, right in enumerate(existing)
            ]
            for left_index, left in enumerate(reviewed)
        ],
        dtype=bool,
    ).reshape(weights.shape)
    pairs = maximum_weight_matching(weights, eligible, cardinality_first=True)
    denominator = len(reviewed) + len(existing)
    score = 2.0 * len(pairs) / denominator if denominator else 1.0
    return score, len(pairs)


def add_counts(left: RelationCounts, right: RelationCounts) -> RelationCounts:
    return RelationCounts(left.tp + right.tp, left.pred + right.pred, left.gt + right.gt)


def load_pair(
    template_name: str, raw_root: Path, layout_root: Path
) -> tuple[TemplateStructure, TemplateStructure]:
    existing = load_raw_template_structure(
        raw_root / f"{template_name}.json", template_name
    )
    corrected_regions = load_template_structure(
        layout_root / f"{template_name}.json", template_name
    )
    reviewed = reconcile_with_r_f1_structure(existing, corrected_regions)
    return existing, reviewed


def compute(raw_root: Path, layout_root: Path) -> dict[str, Any]:
    per_template: list[dict[str, Any]] = []
    region_ious: list[float] = []
    existing_region_types: list[str] = []
    reviewed_region_types: list[str] = []
    existing_regions = reviewed_regions = region_matches = 0
    topology_similarity = 0.0
    existing_grids = reviewed_grids = grid_matches = 0
    existing_cells = reviewed_cells = 0
    existing_widgets = reviewed_widgets = widget_matches = 0
    relation_counts = RelationCounts(0, 0, 0)

    for template_name in TEST_TEMPLATES:
        existing, reviewed = load_pair(template_name, raw_root, layout_root)
        pairs, weights = match_regions_by_geometry(reviewed.regions, existing.regions)
        template_ious = [float(weights[left, right]) for left, right in pairs]
        existing_types = [existing.regions[right].category for left, right in pairs]
        reviewed_types = [reviewed.regions[left].category for left, right in pairs]

        topology_f1, n_grid_matches, n_reviewed_grids, n_existing_grids = topology_match(
            reviewed.grids, existing.grids
        )
        existing_widget_items = blank_widgets(existing)
        reviewed_widget_items = blank_widgets(reviewed)
        widget_f1, n_widget_matches = widget_object_match(
            reviewed_widget_items, existing_widget_items
        )

        endpoints = {
            endpoint
            for relation in (*existing.relations, *reviewed.relations)
            for endpoint in (relation.source, relation.target)
        }
        relation_score = relation_f1(
            reviewed.relations,
            existing.relations,
            {endpoint: endpoint for endpoint in endpoints},
        )

        existing_count = len(existing.regions)
        reviewed_count = len(reviewed.regions)
        boundary_f1 = (
            2.0 * len(pairs) / (existing_count + reviewed_count)
            if existing_count + reviewed_count
            else 1.0
        )
        per_template.append(
            {
                "template": template_name,
                "existing_regions": existing_count,
                "reviewed_regions": reviewed_count,
                "matched_regions": len(pairs),
                "mean_iou": sum(template_ious) / len(template_ious),
                "boundary_f1_at_0_5": boundary_f1,
                "region_type_kappa": cohen_kappa(existing_types, reviewed_types),
                "existing_grids": n_existing_grids,
                "reviewed_grids": n_reviewed_grids,
                "matched_grids": n_grid_matches,
                "topology_f1": topology_f1,
                "existing_widgets": len(existing_widget_items),
                "reviewed_widgets": len(reviewed_widget_items),
                "matched_widgets": n_widget_matches,
                "widget_object_f1": widget_f1,
                "relation_tp": relation_score.counts.tp,
                "existing_relations": relation_score.counts.gt,
                "reviewed_relations": relation_score.counts.pred,
                "typed_relation_f1": relation_score.counts.f1,
            }
        )

        region_ious.extend(template_ious)
        existing_region_types.extend(existing_types)
        reviewed_region_types.extend(reviewed_types)
        existing_regions += existing_count
        reviewed_regions += reviewed_count
        region_matches += len(pairs)
        if topology_f1 is not None:
            topology_similarity += (
                topology_f1 * (n_existing_grids + n_reviewed_grids) / 2.0
            )
        existing_grids += n_existing_grids
        reviewed_grids += n_reviewed_grids
        grid_matches += n_grid_matches
        existing_cells += sum(len(grid.cells) for grid in existing.grids)
        reviewed_cells += sum(len(grid.cells) for grid in reviewed.grids)
        existing_widgets += len(existing_widget_items)
        reviewed_widgets += len(reviewed_widget_items)
        widget_matches += n_widget_matches
        relation_counts = add_counts(relation_counts, relation_score.counts)

    region_denominator = existing_regions + reviewed_regions
    grid_denominator = existing_grids + reviewed_grids
    widget_denominator = existing_widgets + reviewed_widgets
    totals = {
        "templates": len(TEST_TEMPLATES),
        "existing_regions": existing_regions,
        "reviewed_regions": reviewed_regions,
        "matched_regions": region_matches,
        "region_mean_iou": sum(region_ious) / len(region_ious),
        "region_boundary_f1_at_0_5": 2.0 * region_matches / region_denominator,
        "region_type_kappa": cohen_kappa(existing_region_types, reviewed_region_types),
        "existing_grids": existing_grids,
        "reviewed_grids": reviewed_grids,
        "matched_grids": grid_matches,
        "existing_cells": existing_cells,
        "reviewed_cells": reviewed_cells,
        "topology_f1": 2.0 * topology_similarity / grid_denominator,
        "existing_widgets": existing_widgets,
        "reviewed_widgets": reviewed_widgets,
        "matched_widgets": widget_matches,
        "widget_object_f1": 2.0 * widget_matches / widget_denominator,
        "relation_tp": relation_counts.tp,
        "existing_relations": relation_counts.gt,
        "reviewed_relations": relation_counts.pred,
        "typed_relation_f1": relation_counts.f1,
        "three_way_review_kappa": None,
    }
    return {
        "scope": {
            "unit": "template",
            "templates": list(TEST_TEMPLATES),
            "instances_reviewed": 0,
        },
        "protocol": {
            "existing_pass": "raw Label Studio template annotation in new-dataset-json",
            "review_pass": (
                "template-only visual re-review using the corrected region universe; "
                "incomplete grids and relations with unreachable endpoints are excluded"
            ),
            "region_matching": "maximum-cardinality one-to-one geometry matching at IoU >= 0.5",
            "type_statistic": "Cohen's kappa on geometry-matched regions",
            "topology_statistic": "spatially matched grid GriTS topology F1, without a second parent-region penalty",
            "widget_statistic": "matched-object F1 with exact type and blank-template state",
            "relation_statistic": "micro typed directed-relation F1",
            "three_way_review": "not applicable because no instances or independent decision vectors were reviewed",
            "independence_note": (
                "This is agreement between an existing annotation and an assisted review pass, "
                "not a blinded two-human inter-annotator study."
            ),
        },
        "totals": totals,
        "per_template": per_template,
    }


def format_metric(value: Any) -> str:
    return "N/A" if value is None or value == "NA" else f"{float(value):.3f}"


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "template_annotation_agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = result["per_template"]
    with (out_dir / "template_annotation_agreement.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    totals = result["totals"]
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Agreement between the existing template annotations and one template-only review pass on the 11 test templates.}}
\label{{tab:annotation_agreement}}
\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\columnwidth}}{{@{{}}>{{\raggedright\arraybackslash}}X>{{\raggedright\arraybackslash}}X>{{\raggedright\arraybackslash}}X@{{}}}}
\toprule
Target & Agreement statistic & Result \\
\midrule
Region boundaries & Mean IoU; object $F_1$ at IoU $\geq 0.5$ & {format_metric(totals['region_mean_iou'])}; {format_metric(totals['region_boundary_f1_at_0_5'])} \\
Region types & Cohen's $\kappa$ on matched regions & {format_metric(totals['region_type_kappa'])} \\
Local-grid topology & Cell/span topology $F_1$ & {format_metric(totals['topology_f1'])} \\
Widget types and states & Matched-object $F_1$ & {format_metric(totals['widget_object_f1'])} \\
Structural relations & Typed-relation $F_1$ & {format_metric(totals['typed_relation_f1'])} \\
Three-way review decision & Cohen's $\kappa$ & N/A (template-only scope) \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    (out_dir / "annotation_agreement_table.tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("new-dataset-json"))
    parser.add_argument("--layout-root", type=Path, default=Path("newdataset-layout"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/annotation_agreement")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute(args.raw_root, args.layout_root)
    write_outputs(result, args.out_dir)
    print(json.dumps(result["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
