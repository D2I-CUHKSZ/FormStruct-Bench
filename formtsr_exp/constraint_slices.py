from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json
from .metrics import NA, mean_numeric
from .results import METRIC_COLUMNS, format_value


CONSTRAINT_OUTPUT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "constraint",
    "constraint_name",
    "metric",
    "n_with",
    "n_without",
    "templates_with",
    "templates_without",
    "mean_with",
    "mean_without",
    "delta",
    "relative_delta_pct",
    "status",
]

CONSTRAINT_TEMPLATE_COLUMNS = [
    "template_name",
    "constraint",
    "constraint_name",
    "present",
    "signal",
    "rule",
]

DEFAULT_CONSTRAINT_ORDER = [
    "region_local_grids",
    "widget_grouping",
    "key_field_relations",
    "line_item_groups",
    "mixed_layout",
    "visual_degradation",
]

CONSTRAINT_NAMES = {
    "region_local_grids": "Region-local grids",
    "widget_grouping": "Widget grouping",
    "key_field_relations": "Dense key-field relations",
    "line_item_groups": "Line-item groups",
    "mixed_layout": "Mixed layout",
    "visual_degradation": "Visual degradation",
}

IGNORED_CONSTRAINTS = {
    "weak_borderless_grids": "Ignored by default: current weak/borderless grid annotations are known to be wrong.",
}

DEGRADATION_TERMS = {
    "blur",
    "blurred",
    "degraded",
    "degradation",
    "distorted",
    "distortion",
    "low_contrast",
    "noise",
    "noisy",
    "occlusion",
    "shadow",
    "watermark",
}


@dataclass(frozen=True)
class ConstraintFlag:
    present: bool
    signal: float | str
    rule: str


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _num(value: Any, default: float = 0.0) -> float:
    number = _as_float(value)
    return default if number is None else number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "degraded", "blurred", "noisy"}
    return bool(value)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _template_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("template_name")) for row in rows if row.get("template_name")}


def _load_layout_metadata(layout_root: Path | None, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata_by_template: dict[str, dict[str, Any]] = {}
    if not layout_root:
        return metadata_by_template
    for template_name in sorted(_template_names(rows)):
        path = layout_root / f"{template_name}.json"
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if isinstance(metadata, dict):
            metadata_by_template[template_name] = metadata
    return metadata_by_template


def _contains_degradation_tag(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_degradation_tag(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_degradation_tag(item) for item in value)
    if not isinstance(value, str):
        return False
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    return any(term in text for term in DEGRADATION_TERMS)


def _explicit_visual_degradation(metadata: dict[str, Any]) -> tuple[bool, str]:
    visual = metadata.get("V") if isinstance(metadata.get("V"), dict) else {}
    assert isinstance(visual, dict)
    candidates = [
        "degraded",
        "visual_degradation",
        "degradation",
        "is_degraded",
        "blurred",
        "noisy",
        "low_contrast",
        "occluded",
        "shadowed",
    ]
    for key in candidates:
        if key in visual:
            return _truthy(visual[key]), f"metadata.V.{key}"

    for key, value in visual.items():
        key_text = str(key).lower()
        if "degrad" in key_text:
            return _truthy(value), f"metadata.V.{key}"
        if key_text in {"readability_tag", "visual_tag", "quality_tag"} and _contains_degradation_tag(value):
            return True, f"metadata.V.{key}"

    difficulty = metadata.get("difficulty") if isinstance(metadata.get("difficulty"), dict) else {}
    assert isinstance(difficulty, dict)
    visual_score = _as_float(difficulty.get("visual_score"))
    if visual_score is not None and visual_score > 1:
        return True, "metadata.difficulty.visual_score>1"
    return False, "no explicit degradation tag"


def build_constraint_lookup(
    rows: list[dict[str, Any]],
    *,
    layout_root: Path | None = None,
) -> tuple[dict[str, dict[str, ConstraintFlag]], dict[str, Any]]:
    metadata_by_template = _load_layout_metadata(layout_root, rows)

    relation_counts: list[float] = []
    option_group_counts: list[float] = []
    for metadata in metadata_by_template.values():
        structural = metadata.get("S") if isinstance(metadata.get("S"), dict) else {}
        assert isinstance(structural, dict)
        relation_counts.append(_num(structural.get("relation_edge_count")))
        option_group_counts.append(max(_num(structural.get("option_group_count")), _num(structural.get("multi_value_key_count"))))

    relation_q75 = _percentile(relation_counts, 0.75)
    option_group_q75 = _percentile(option_group_counts, 0.75)

    lookup: dict[str, dict[str, ConstraintFlag]] = {}
    for template_name, metadata in metadata_by_template.items():
        structural = metadata.get("S") if isinstance(metadata.get("S"), dict) else {}
        context = metadata.get("C") if isinstance(metadata.get("C"), dict) else {}
        layout = metadata.get("layout_structure") if isinstance(metadata.get("layout_structure"), dict) else {}
        assert isinstance(structural, dict)
        assert isinstance(context, dict)
        assert isinstance(layout, dict)

        cell_signal = max(
            _num(structural.get("cell_count")),
            _num(structural.get("row_count")),
            _num(structural.get("col_count")),
            _num(layout.get("table_region_count")),
        )
        line_item_signal = max(_num(structural.get("line_item_group_count")), _num(layout.get("line_item_group_count")))
        widget_signal = max(
            _num(structural.get("selection_control_count")),
            _num(structural.get("option_group_count")),
            _num(structural.get("multi_value_key_count")),
            max(_num(structural.get("max_values_per_key")) - 1.0, 0.0),
        )
        relation_signal = _num(structural.get("relation_edge_count"))
        table_count = _num(context.get("table_count_on_page"))
        table_regions = _num(layout.get("table_region_count"))
        region_count = _num(layout.get("region_count"))
        section_count = _num(layout.get("section_count"))
        mixed_signal = 1.0 if (
            _truthy(context.get("multi_table_page"))
            or (table_count > 0 and (region_count > table_regions or section_count > 1))
            or (table_regions > 0 and region_count > table_regions)
        ) else 0.0
        visual_present, visual_rule = _explicit_visual_degradation(metadata)

        lookup[template_name] = {
            "region_local_grids": ConstraintFlag(
                cell_signal > 0,
                cell_signal,
                "metadata.S cell/row/col count > 0 or metadata.layout_structure.table_region_count > 0",
            ),
            "widget_grouping": ConstraintFlag(
                widget_signal >= option_group_q75 and widget_signal > 0,
                widget_signal,
                f"max(selection_control_count, option_group_count, multi_value_key_count, max_values_per_key-1) >= q75 ({option_group_q75:.4f})",
            ),
            "key_field_relations": ConstraintFlag(
                relation_signal >= relation_q75 and relation_signal > 0,
                relation_signal,
                f"metadata.S.relation_edge_count >= q75 ({relation_q75:.4f})",
            ),
            "line_item_groups": ConstraintFlag(
                line_item_signal > 0,
                line_item_signal,
                "metadata.S.line_item_group_count > 0 or metadata.layout_structure.line_item_group_count > 0",
            ),
            "mixed_layout": ConstraintFlag(
                mixed_signal > 0,
                mixed_signal,
                "multi_table_page or table regions mixed with non-table/section regions",
            ),
            "visual_degradation": ConstraintFlag(
                visual_present,
                1.0 if visual_present else 0.0,
                visual_rule,
            ),
        }

    counts_by_constraint = {
        constraint: Counter(flags[constraint].present for flags in lookup.values())
        for constraint in DEFAULT_CONSTRAINT_ORDER
    }
    metadata = {
        "layout_root": str(layout_root) if layout_root else None,
        "n_templates_with_layout_metadata": len(metadata_by_template),
        "constraints": DEFAULT_CONSTRAINT_ORDER,
        "constraint_names": CONSTRAINT_NAMES,
        "ignored_constraints": IGNORED_CONSTRAINTS,
        "thresholds": {
            "key_field_relations_relation_edge_count_q75": relation_q75,
            "widget_grouping_signal_q75": option_group_q75,
        },
        "template_counts_by_constraint": {
            constraint: {
                "with": counts.get(True, 0),
                "without": counts.get(False, 0),
            }
            for constraint, counts in counts_by_constraint.items()
        },
        "delta_rule": "Delta(c) = mean(metric | constraint absent) - mean(metric | constraint present). Positive values indicate performance degradation when the constraint is present.",
    }
    return lookup, metadata


def _annotate_rows_with_constraints(
    rows: list[dict[str, Any]],
    constraint_lookup: dict[str, dict[str, ConstraintFlag]],
) -> tuple[list[dict[str, Any]], set[str]]:
    annotated: list[dict[str, Any]] = []
    missing_templates: set[str] = set()
    for row in rows:
        template_name = str(row.get("template_name") or "")
        flags = constraint_lookup.get(template_name)
        if not flags:
            if template_name:
                missing_templates.add(template_name)
            continue
        copied = dict(row)
        copied["_constraints"] = flags
        annotated.append(copied)
    return annotated, missing_templates


def _model_names(rows: list[dict[str, Any]], model_configs: list[dict[str, Any]] | None) -> list[str]:
    configured = {
        str(cfg.get("name"))
        for cfg in (model_configs or [])
        if isinstance(cfg, dict) and cfg.get("name")
    }
    present = {str(row.get("model")) for row in rows if row.get("model")}
    return sorted(present | configured)


def _model_config_lookup(model_configs: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {
        str(cfg.get("name")): cfg
        for cfg in (model_configs or [])
        if isinstance(cfg, dict) and cfg.get("name")
    }


def summarize_constraint_slices(
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

        for constraint in DEFAULT_CONSTRAINT_ORDER:
            rows_with = [
                row
                for row in model_rows
                if row.get("_constraints", {}).get(constraint) and row["_constraints"][constraint].present
            ]
            rows_without = [
                row
                for row in model_rows
                if row.get("_constraints", {}).get(constraint) and not row["_constraints"][constraint].present
            ]
            templates_with = len({str(row.get("template_name")) for row in rows_with if row.get("template_name")})
            templates_without = len({str(row.get("template_name")) for row in rows_without if row.get("template_name")})
            status = "ok" if rows_with and rows_without else "insufficient_contrast"

            for metric in METRIC_COLUMNS:
                mean_with = mean_numeric(rows_with, metric)
                mean_without = mean_numeric(rows_without, metric)
                delta: float | str = NA
                relative_delta: float | str = NA
                if isinstance(mean_with, (int, float)) and isinstance(mean_without, (int, float)):
                    delta = float(mean_without) - float(mean_with)
                    relative_delta = (delta / float(mean_without) * 100.0) if float(mean_without) else NA
                output_rows.append(
                    {
                        "model": model,
                        "model_id": model_id,
                        "group": group,
                        "constraint": constraint,
                        "constraint_name": CONSTRAINT_NAMES[constraint],
                        "metric": metric,
                        "n_with": len(rows_with),
                        "n_without": len(rows_without),
                        "templates_with": templates_with,
                        "templates_without": templates_without,
                        "mean_with": mean_with,
                        "mean_without": mean_without,
                        "delta": delta,
                        "relative_delta_pct": relative_delta,
                        "status": status,
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


def _write_constraint_table_tex(path: Path, rows: list[dict[str, Any]], *, metric: str = "CDS") -> None:
    metric_rows = [row for row in rows if row.get("metric") == metric]
    lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\hline",
        "Run & Constraint & Metric & N+ & N- & With & Without & Delta \\\\",
        "\\hline",
    ]
    for row in metric_rows:
        lines.append(
            " & ".join(
                [
                    format_value(row.get("model")),
                    format_value(row.get("constraint")),
                    format_value(row.get("metric")),
                    format_value(row.get("n_with")),
                    format_value(row.get("n_without")),
                    format_value(row.get("mean_with")),
                    format_value(row.get("mean_without")),
                    format_value(row.get("delta")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _template_membership_rows(
    constraint_lookup: dict[str, dict[str, ConstraintFlag]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template_name, flags in sorted(constraint_lookup.items()):
        for constraint in DEFAULT_CONSTRAINT_ORDER:
            flag = flags[constraint]
            rows.append(
                {
                    "template_name": template_name,
                    "constraint": constraint,
                    "constraint_name": CONSTRAINT_NAMES[constraint],
                    "present": flag.present,
                    "signal": flag.signal,
                    "rule": flag.rule,
                }
            )
    return rows


def write_constraint_slice_results(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    index_rows: list[dict[str, Any]] | None = None,
    layout_root: Path | None = None,
    model_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    lookup_rows = [*(index_rows or []), *rows]
    constraint_lookup, metadata = build_constraint_lookup(lookup_rows, layout_root=layout_root)
    annotated_rows, missing_templates = _annotate_rows_with_constraints(rows, constraint_lookup)
    summaries = summarize_constraint_slices(annotated_rows, model_configs=model_configs)
    membership_rows = _template_membership_rows(constraint_lookup)

    _write_csv(out_dir / "constraint_slice_results.csv", summaries, CONSTRAINT_OUTPUT_COLUMNS)
    _write_csv(out_dir / "constraint_slice_template_membership.csv", membership_rows, CONSTRAINT_TEMPLATE_COLUMNS)
    _write_constraint_table_tex(out_dir / "constraint_slice_results_table.tex", summaries, metric="CDS")

    row_counts = {
        constraint: Counter(row["_constraints"][constraint].present for row in annotated_rows)
        for constraint in DEFAULT_CONSTRAINT_ORDER
    }
    metadata.update(
        {
            "n_rows_input": len(rows),
            "n_rows_with_constraint_metadata": len(annotated_rows),
            "missing_templates": sorted(missing_templates),
            "row_counts_by_constraint": {
                constraint: {
                    "with": counts.get(True, 0),
                    "without": counts.get(False, 0),
                }
                for constraint, counts in row_counts.items()
            },
        }
    )
    write_json(out_dir / "constraint_slice_metadata.json", metadata)
    return {
        "summaries": summaries,
        "template_membership": membership_rows,
        "metadata": metadata,
    }
