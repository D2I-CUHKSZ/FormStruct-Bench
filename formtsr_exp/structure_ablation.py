from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json
from .metrics import (
    NA,
    compute_cds,
    MatchStats,
    _relation_signature,
    extract_relations,
    line_item_group_f1,
    load_optional_layout_gt,
    mean_numeric,
    region_f1,
    topology_score,
    widget_answer_accuracy,
)
from .results import format_value


ABLATION_COMPONENT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "component",
    "mean",
    "n_numeric",
    "n_total",
    "coverage",
]

ABLATION_RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "variant",
    "variant_name",
    "n_total",
    "score",
    "components",
]

ABLATION_DELTA_COLUMNS = [
    "model",
    "model_id",
    "group",
    "comparison",
    "with_variant",
    "without_variant",
    "score_with",
    "score_without",
    "delta",
    "relative_delta_pct",
    "interpretation",
]

TARGETED_ABLATION_COLUMNS = [
    "model",
    "model_id",
    "group",
    "comparison",
    "scope",
    "n_scope",
    "with_variant",
    "without_variant",
    "score_with",
    "score_without",
    "delta",
    "relative_delta_pct",
    "interpretation",
]

VARIANTS: dict[str, dict[str, Any]] = {
    "answer_only": {
        "name": "Answer only",
        "components": ["TSR-path", "VAcc"],
    },
    "global_grid_struct": {
        "name": "Global-grid structural",
        "components": ["TSR-path", "VAcc", "R-F1", "GlobalGrid-F1"],
    },
    "region_local_grid_struct": {
        "name": "Region-local-grid structural",
        "components": ["TSR-path", "VAcc", "R-F1", "RegionLocalGrid-F1"],
    },
    "region_local_grid_widget": {
        "name": "Region-local-grid + widget",
        "components": ["TSR-path", "VAcc", "R-F1", "RegionLocalGrid-F1", "WAcc"],
    },
    "region_local_grid_widget_box": {
        "name": "Region-local-grid + widget boxes",
        "components": ["TSR-path", "VAcc", "R-F1", "RegionLocalGrid-F1", "WidgetBox-F1"],
    },
    "region_local_grid_widget_full": {
        "name": "Region-local-grid + widget answers + boxes",
        "components": ["TSR-path", "VAcc", "R-F1", "RegionLocalGrid-F1", "WAcc", "WidgetBox-F1"],
    },
    "region_local_grid_relation": {
        "name": "Region-local-grid + relation",
        "components": ["TSR-path", "VAcc", "R-F1", "RegionLocalGrid-F1", "Rel-F1"],
    },
    "full_region_local_struct": {
        "name": "Region-local-grid + widget + relation",
        "components": ["TSR-path", "VAcc", "R-F1", "RegionLocalGrid-F1", "WAcc", "WidgetBox-F1", "Rel-F1"],
    },
}

COMPARISONS = [
    {
        "comparison": "region_local_grid_vs_global_grid",
        "with_variant": "region_local_grid_struct",
        "without_variant": "global_grid_struct",
    },
    {
        "comparison": "widget_level_effect",
        "with_variant": "region_local_grid_widget",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "relation_level_effect_without_widget",
        "with_variant": "region_local_grid_relation",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "relation_level_effect_with_widget",
        "with_variant": "full_region_local_struct",
        "without_variant": "region_local_grid_widget_full",
    },
    {
        "comparison": "full_structural_vs_answer_only",
        "with_variant": "full_region_local_struct",
        "without_variant": "answer_only",
    },
]

TARGETED_COMPARISONS = [
    {
        "comparison": "region_local_grid_vs_global_grid_on_grid_samples",
        "scope": "grid_applicable",
        "with_variant": "region_local_grid_struct",
        "without_variant": "global_grid_struct",
    },
    {
        "comparison": "widget_answer_effect_on_widget_answer_samples",
        "scope": "widget_answer_applicable",
        "with_variant": "region_local_grid_widget",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "widget_box_effect_on_widget_box_samples",
        "scope": "widget_box_applicable",
        "with_variant": "region_local_grid_widget_box",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "widget_full_effect_on_widget_samples",
        "scope": "widget_any_applicable",
        "with_variant": "region_local_grid_widget_full",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "relation_effect_on_relation_samples",
        "scope": "relation_applicable",
        "with_variant": "region_local_grid_relation",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "full_structural_vs_answer_only_on_structural_samples",
        "scope": "any_structural_applicable",
        "with_variant": "full_region_local_struct",
        "without_variant": "answer_only",
    },
]


def _model_config_lookup(model_configs: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {
        str(cfg.get("name")): cfg
        for cfg in (model_configs or [])
        if isinstance(cfg, dict) and cfg.get("name")
    }


def _model_names(rows: list[dict[str, Any]], model_configs: list[dict[str, Any]] | None) -> list[str]:
    present = {str(row.get("model")) for row in rows if row.get("model")}
    configured = set(_model_config_lookup(model_configs))
    return sorted(present | configured)


def _weighted_average(row: dict[str, Any], components: list[str]) -> float | str:
    weights = {component: 1.0 for component in components}
    return compute_cds(row, weights)


def _load_prediction(pred_root: Path, model: str, sample_id: str) -> Any | None:
    path = pred_root / model / f"{sample_id}.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _index_by_sample(index_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in index_rows if row.get("sample_id")}


def _extra_structure_metrics(
    sample: dict[str, Any],
    pred: Any | None,
    *,
    valid_json: bool,
    layout_root: Path | None,
) -> dict[str, Any]:
    if not valid_json or pred is None:
        return {
            "GlobalGrid-F1": NA,
            "RegionLocalGrid-F1": NA,
            "Rel-F1": NA,
            "WidgetBox-F1": NA,
        }

    gt_answer = read_json(Path(sample["label_path"]))
    layout_gt = load_optional_layout_gt(sample, layout_root)
    structural_gt = gt_answer if any(k in gt_answer for k in ("regions", "local_grids", "cells", "widgets", "relations")) else layout_gt
    if structural_gt is None:
        return {
            "GlobalGrid-F1": NA,
            "RegionLocalGrid-F1": NA,
            "Rel-F1": NA,
            "WidgetBox-F1": NA,
        }

    global_grid, _ = topology_score(pred, structural_gt)
    region_local_grid, _ = line_item_group_f1(pred, structural_gt)
    rel_f1 = _explicit_relation_f1(pred, structural_gt)
    widget_box_f1, _ = region_f1({"regions": pred.get("widgets", [])} if isinstance(pred, dict) else pred, {"regions": structural_gt.get("widgets", [])} if isinstance(structural_gt, dict) else structural_gt)
    wacc, _ = widget_answer_accuracy(pred, gt_answer, structural_gt)
    return {
        "GlobalGrid-F1": global_grid,
        "RegionLocalGrid-F1": region_local_grid,
        "Rel-F1": rel_f1,
        "WidgetBox-F1": widget_box_f1,
        "WAcc": wacc,
    }


def _explicit_relation_f1(pred: Any, gt: Any) -> float | str:
    gt_rel_items = extract_relations(gt)
    if not gt_rel_items:
        return NA
    pred_rel_items = extract_relations(pred)
    gt_rels = {_relation_signature(item) for item in gt_rel_items}
    pred_rels = {_relation_signature(item) for item in pred_rel_items}
    stats = MatchStats(len(pred_rels & gt_rels), len(pred_rels), len(gt_rels))
    return stats.f1


def build_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    index_rows: list[dict[str, Any]],
    pred_root: Path,
    layout_root: Path | None = None,
) -> list[dict[str, Any]]:
    sample_by_id = _index_by_sample(index_rows)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        model = str(row.get("model") or "")
        sample = sample_by_id.get(sample_id)
        if not sample or not model:
            continue
        pred = _load_prediction(pred_root, model, sample_id)
        valid_json = bool(row.get("valid_json")) and pred is not None
        ablation_row = dict(row)
        ablation_row.update(
            _extra_structure_metrics(
                sample,
                pred,
                valid_json=valid_json,
                layout_root=layout_root,
            )
        )
        for variant, spec in VARIANTS.items():
            ablation_row[f"score:{variant}"] = _weighted_average(ablation_row, list(spec["components"]))
        output_rows.append(ablation_row)
    return output_rows


def summarize_components(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    components = sorted(
        {
            component
            for spec in VARIANTS.values()
            for component in spec["components"]
        }
        | {"WidgetBox-F1"}
    )
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    config_by_model = _model_config_lookup(model_configs)

    summaries: list[dict[str, Any]] = []
    for model in _model_names(rows, model_configs):
        model_rows = by_model.get(model, [])
        cfg = config_by_model.get(model, {})
        if model_rows:
            model_id = str(model_rows[0].get("model_id") or cfg.get("model") or model)
            group = str(model_rows[0].get("group") or cfg.get("group") or "")
        else:
            model_id = str(cfg.get("model", model))
            group = str(cfg.get("group", cfg.get("provider", "")))
        for component in components:
            n_numeric = sum(1 for row in model_rows if isinstance(row.get(component), (int, float)))
            n_total = len(model_rows)
            summaries.append(
                {
                    "model": model,
                    "model_id": model_id,
                    "group": group,
                    "component": component,
                    "mean": mean_numeric(model_rows, component),
                    "n_numeric": n_numeric,
                    "n_total": n_total,
                    "coverage": (n_numeric / n_total) if n_total else NA,
                }
            )
    return summaries


def summarize_variants(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    config_by_model = _model_config_lookup(model_configs)

    summaries: list[dict[str, Any]] = []
    for model in _model_names(rows, model_configs):
        model_rows = by_model.get(model, [])
        cfg = config_by_model.get(model, {})
        if model_rows:
            model_id = str(model_rows[0].get("model_id") or cfg.get("model") or model)
            group = str(model_rows[0].get("group") or cfg.get("group") or "")
        else:
            model_id = str(cfg.get("model", model))
            group = str(cfg.get("group", cfg.get("provider", "")))
        for variant, spec in VARIANTS.items():
            summaries.append(
                {
                    "model": model,
                    "model_id": model_id,
                    "group": group,
                    "variant": variant,
                    "variant_name": spec["name"],
                    "n_total": len(model_rows),
                    "score": mean_numeric(model_rows, f"score:{variant}"),
                    "components": ",".join(spec["components"]),
                }
            )
    return summaries


def summarize_deltas(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in variant_rows:
        by_model.setdefault(str(row["model"]), {})[str(row["variant"])] = row

    output_rows: list[dict[str, Any]] = []
    for model, variants in sorted(by_model.items()):
        any_row = next(iter(variants.values()))
        for comparison in COMPARISONS:
            with_variant = comparison["with_variant"]
            without_variant = comparison["without_variant"]
            with_row = variants.get(with_variant, {})
            without_row = variants.get(without_variant, {})
            score_with = with_row.get("score", NA)
            score_without = without_row.get("score", NA)
            delta: float | str = NA
            relative_delta: float | str = NA
            interpretation = "insufficient_data"
            if isinstance(score_with, (int, float)) and isinstance(score_without, (int, float)):
                delta = float(score_with) - float(score_without)
                relative_delta = (delta / float(score_without) * 100.0) if float(score_without) else NA
                if delta < 0:
                    interpretation = "stricter_lower_score"
                elif delta > 0:
                    interpretation = "higher_score_under_added_dimension"
                else:
                    interpretation = "no_change"
            output_rows.append(
                {
                    "model": model,
                    "model_id": any_row.get("model_id", model),
                    "group": any_row.get("group", ""),
                    "comparison": comparison["comparison"],
                    "with_variant": with_variant,
                    "without_variant": without_variant,
                    "score_with": score_with,
                    "score_without": score_without,
                    "delta": delta,
                    "relative_delta_pct": relative_delta,
                    "interpretation": interpretation,
                }
            )
    return output_rows


def _row_in_scope(row: dict[str, Any], scope: str) -> bool:
    if scope == "grid_applicable":
        return isinstance(row.get("GlobalGrid-F1"), (int, float)) or isinstance(row.get("RegionLocalGrid-F1"), (int, float))
    if scope == "widget_answer_applicable":
        return isinstance(row.get("WAcc"), (int, float))
    if scope == "widget_box_applicable":
        return isinstance(row.get("WidgetBox-F1"), (int, float))
    if scope == "widget_any_applicable":
        return isinstance(row.get("WAcc"), (int, float)) or isinstance(row.get("WidgetBox-F1"), (int, float))
    if scope == "relation_applicable":
        return isinstance(row.get("Rel-F1"), (int, float))
    if scope == "any_structural_applicable":
        return any(
            isinstance(row.get(component), (int, float))
            for component in ("R-F1", "GlobalGrid-F1", "RegionLocalGrid-F1", "WAcc", "WidgetBox-F1", "Rel-F1")
        )
    return True


def summarize_targeted_deltas(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    config_by_model = _model_config_lookup(model_configs)
    output_rows: list[dict[str, Any]] = []
    for model in _model_names(rows, model_configs):
        model_rows = by_model.get(model, [])
        cfg = config_by_model.get(model, {})
        if model_rows:
            model_id = str(model_rows[0].get("model_id") or cfg.get("model") or model)
            group = str(model_rows[0].get("group") or cfg.get("group") or "")
        else:
            model_id = str(cfg.get("model", model))
            group = str(cfg.get("group", cfg.get("provider", "")))

        for comparison in TARGETED_COMPARISONS:
            scope = str(comparison["scope"])
            scope_rows = [row for row in model_rows if _row_in_scope(row, scope)]
            with_variant = str(comparison["with_variant"])
            without_variant = str(comparison["without_variant"])
            score_with = mean_numeric(scope_rows, f"score:{with_variant}")
            score_without = mean_numeric(scope_rows, f"score:{without_variant}")
            delta: float | str = NA
            relative_delta: float | str = NA
            interpretation = "insufficient_data"
            if isinstance(score_with, (int, float)) and isinstance(score_without, (int, float)):
                delta = float(score_with) - float(score_without)
                relative_delta = (delta / float(score_without) * 100.0) if float(score_without) else NA
                if delta < 0:
                    interpretation = "stricter_lower_score"
                elif delta > 0:
                    interpretation = "higher_score_under_added_dimension"
                else:
                    interpretation = "no_change"
            output_rows.append(
                {
                    "model": model,
                    "model_id": model_id,
                    "group": group,
                    "comparison": comparison["comparison"],
                    "scope": scope,
                    "n_scope": len(scope_rows),
                    "with_variant": with_variant,
                    "without_variant": without_variant,
                    "score_with": score_with,
                    "score_without": score_without,
                    "delta": delta,
                    "relative_delta_pct": relative_delta,
                    "interpretation": interpretation,
                }
            )
    return output_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in columns})


def _write_delta_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\hline",
        "Run & Comparison & With & Without & Delta & Interpretation \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    format_value(row.get("model")),
                    format_value(row.get("comparison")),
                    format_value(row.get("score_with")),
                    format_value(row.get("score_without")),
                    format_value(row.get("delta")),
                    format_value(row.get("interpretation")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_targeted_delta_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\hline",
        "Run & Comparison & Scope N & With & Without & Delta & Interpretation \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    format_value(row.get("model")),
                    format_value(row.get("comparison")),
                    format_value(row.get("n_scope")),
                    format_value(row.get("score_with")),
                    format_value(row.get("score_without")),
                    format_value(row.get("delta")),
                    format_value(row.get("interpretation")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_structure_ablation_results(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    index_rows: list[dict[str, Any]],
    pred_root: Path,
    layout_root: Path | None = None,
    model_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    ablation_rows = build_ablation_rows(
        rows,
        index_rows=index_rows,
        pred_root=pred_root,
        layout_root=layout_root,
    )
    component_rows = summarize_components(ablation_rows, model_configs=model_configs)
    variant_rows = summarize_variants(ablation_rows, model_configs=model_configs)
    delta_rows = summarize_deltas(variant_rows)
    targeted_delta_rows = summarize_targeted_deltas(ablation_rows, model_configs=model_configs)

    _write_csv(out_dir / "structure_ablation_components.csv", component_rows, ABLATION_COMPONENT_COLUMNS)
    _write_csv(out_dir / "structure_ablation_variants.csv", variant_rows, ABLATION_RESULT_COLUMNS)
    _write_csv(out_dir / "structure_ablation_deltas.csv", delta_rows, ABLATION_DELTA_COLUMNS)
    _write_csv(out_dir / "structure_ablation_targeted_deltas.csv", targeted_delta_rows, TARGETED_ABLATION_COLUMNS)
    _write_delta_table_tex(out_dir / "structure_ablation_deltas_table.tex", delta_rows)
    _write_targeted_delta_table_tex(out_dir / "structure_ablation_targeted_deltas_table.tex", targeted_delta_rows)

    metadata = {
        "pred_root": str(pred_root),
        "layout_root": str(layout_root) if layout_root else None,
        "n_rows_input": len(rows),
        "n_rows_with_ablation_metrics": len(ablation_rows),
        "variants": VARIANTS,
        "comparisons": COMPARISONS,
        "targeted_comparisons": TARGETED_COMPARISONS,
        "delta_rule": "Delta = score(with added structural dimension) - score(without that dimension). Negative values mean the added dimension lowers the score and exposes errors hidden by the simpler metric.",
        "targeted_delta_rule": "Targeted deltas use only samples where the added structure dimension is applicable, reducing dilution from samples without that annotation type.",
        "global_grid_metric": "GlobalGrid-F1 uses row/col/rowspan/colspan signatures over all cells, ignoring local region membership.",
        "region_local_grid_metric": "RegionLocalGrid-F1 uses local grid or line-item-group region localization via bbox matching.",
        "widget_metric": "WAcc compares widget-selected answer fields; WidgetBox-F1 is reported as a component diagnostic but is not part of default ablation variants.",
        "relation_metric": "Rel-F1 compares explicit normalized structural relation signatures only; predictions without explicit relations receive zero when GT relation edges exist.",
    }
    write_json(out_dir / "structure_ablation_metadata.json", metadata)
    return {
        "components": component_rows,
        "variants": variant_rows,
        "deltas": delta_rows,
        "targeted_deltas": targeted_delta_rows,
        "metadata": metadata,
    }
