from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constraint_slices import (
    CONSTRAINT_NAMES,
    ConstraintFlag,
    build_constraint_lookup,
)
from .io_utils import ensure_dir, read_jsonl, write_json
from .metrics import NA
from .robustness_metrics_report import (
    COUNT_COLUMNS,
    GEOMETRY_PRESERVING_VARIANTS,
    IDENTITY_COLUMNS,
    LEVEL_ORDER,
    METRIC_RESULT_COLUMNS,
    METRICS,
    VARIANTS,
    _format,
    _numeric,
    aggregate_pairs,
)


COMPONENTS = (
    "region_local_grids",
    "widget_grouping",
    "key_field_relations",
    "line_item_groups",
    "mixed_layout",
)

MEMBERSHIP_COLUMNS = [
    "clean_sample_id",
    "template_name",
    "component",
    "component_name",
    "present",
    "signal",
    "rule",
]

SLICE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "degradation_variant",
    "degradation_level",
    "component",
    "component_name",
    "component_present",
    "n_templates",
    "structure_metric_status",
    *COUNT_COLUMNS,
    *METRIC_RESULT_COLUMNS,
]

CONTRAST_IDENTITY_COLUMNS = [
    *IDENTITY_COLUMNS,
    "degradation_variant",
    "degradation_level",
    "component",
    "component_name",
    "n_with",
    "n_without",
    "templates_with",
    "templates_without",
    "status",
]

CONTRAST_METRIC_COLUMNS = [
    column
    for metric in METRICS
    for column in (
        f"clean_{metric}_with",
        f"degraded_{metric}_with",
        f"{metric}_drop_with",
        f"clean_{metric}_without",
        f"degraded_{metric}_without",
        f"{metric}_drop_without",
        f"{metric}_excess_drop",
    )
]

CONTRAST_COLUMNS = [*CONTRAST_IDENTITY_COLUMNS, *CONTRAST_METRIC_COLUMNS]

MACRO_COLUMNS = [
    "component",
    "component_name",
    "degradation_variant",
    "degradation_level",
    "n_models",
    "n_templates_per_condition",
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

SEVERITY_COLUMNS = [
    "component",
    "component_name",
    "degradation_level",
    "n_models",
    "n_templates_per_condition",
    "n_variants",
    "spatial_variants",
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

EXCESS_SEVERITY_COLUMNS = [
    "component",
    "component_name",
    "degradation_level",
    "n_models",
    "n_with_templates",
    "n_without_templates",
    *[
        column
        for metric in METRICS
        for column in (
            f"mean_{metric}_drop_with",
            f"mean_{metric}_drop_without",
            f"mean_{metric}_excess_drop",
        )
    ],
]

OBSOLETE_REPORT_FILES = (
    "visual_degradation_by_component_all_models.csv",
    "visual_degradation_by_component_same_backend.csv",
    "visual_degradation_component_contrast_all_models.csv",
    "visual_degradation_component_contrast_same_backend.csv",
    "visual_degradation_component_condition_all_models_diagnostic.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in columns})


def load_component_lookup(
    layout_root: Path,
    clean_index_rows: list[dict[str, Any]],
    reference_index_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, ConstraintFlag]], dict[str, Any], list[dict[str, Any]]]:
    reference_templates = sorted(
        {str(row["template_name"]) for row in reference_index_rows}
    )
    all_template_rows = [{"template_name": template} for template in reference_templates]
    lookup, metadata = build_constraint_lookup(all_template_rows, layout_root=layout_root)
    if set(lookup) != set(reference_templates):
        raise ValueError(
            "formal reference templates and available component metadata differ: "
            f"missing={sorted(set(reference_templates) - set(lookup))}, "
            f"extra={sorted(set(lookup) - set(reference_templates))}"
        )
    clean_templates = {str(row["template_name"]) for row in clean_index_rows}
    missing = clean_templates - set(lookup)
    if missing:
        raise ValueError(f"robustness templates missing component metadata: {sorted(missing)}")

    membership: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for row in clean_index_rows:
        sample_id = str(row["sample_id"])
        template_name = str(row["template_name"])
        if sample_id in seen_samples:
            raise ValueError(f"duplicate clean robustness sample: {sample_id}")
        seen_samples.add(sample_id)
        for component in COMPONENTS:
            flag = lookup[template_name][component]
            membership.append(
                {
                    "clean_sample_id": sample_id,
                    "template_name": template_name,
                    "component": component,
                    "component_name": CONSTRAINT_NAMES[component],
                    "present": flag.present,
                    "signal": flag.signal,
                    "rule": flag.rule,
                }
            )
    return lookup, metadata, membership


def annotate_pairs(
    pair_rows: list[dict[str, Any]],
    component_lookup: dict[str, dict[str, ConstraintFlag]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in pair_rows:
        template_name = str(row.get("template_name") or "")
        flags = component_lookup.get(template_name)
        if flags is None:
            raise ValueError(f"pair row has no template component label: {template_name}")
        for component in COMPONENTS:
            copied = dict(row)
            copied["component"] = component
            copied["component_name"] = CONSTRAINT_NAMES[component]
            copied["component_present"] = flags[component].present
            annotated.append(copied)
    return annotated


def aggregate_component_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_fields = (
        "model",
        "degradation_variant",
        "degradation_level",
        "component",
        "component_present",
    )
    summaries = aggregate_pairs(rows, group_fields)
    template_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        template_sets[key].add(str(row["template_name"]))
    for summary in summaries:
        key = tuple(str(summary[field]) for field in group_fields)
        summary["component_name"] = CONSTRAINT_NAMES[str(summary["component"])]
        summary["n_templates"] = len(template_sets[key])
    return summaries


def build_contrasts(slice_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in slice_rows:
        key = (
            str(row["model"]),
            str(row["degradation_variant"]),
            str(row["degradation_level"]),
            str(row["component"]),
        )
        grouped[key][bool(row["component_present"])] = row

    output: list[dict[str, Any]] = []
    for key, by_presence in grouped.items():
        present = by_presence.get(True)
        absent = by_presence.get(False)
        first = present or absent
        assert first is not None
        status = "ok" if present is not None and absent is not None else "insufficient_contrast"
        result: dict[str, Any] = {
            **{field: first[field] for field in IDENTITY_COLUMNS},
            "degradation_variant": key[1],
            "degradation_level": key[2],
            "component": key[3],
            "component_name": CONSTRAINT_NAMES[key[3]],
            "n_with": present["n_total"] if present else 0,
            "n_without": absent["n_total"] if absent else 0,
            "templates_with": present["n_templates"] if present else 0,
            "templates_without": absent["n_templates"] if absent else 0,
            "status": status,
        }
        for metric in METRICS:
            for side, row in (("with", present), ("without", absent)):
                result[f"clean_{metric}_{side}"] = row.get(f"clean_{metric}", NA) if row else NA
                result[f"degraded_{metric}_{side}"] = row.get(f"degraded_{metric}", NA) if row else NA
                result[f"{metric}_drop_{side}"] = row.get(f"{metric}_drop", NA) if row else NA
            drop_with = _numeric(result[f"{metric}_drop_with"])
            drop_without = _numeric(result[f"{metric}_drop_without"])
            result[f"{metric}_excess_drop"] = (
                drop_with - drop_without
                if drop_with is not None and drop_without is not None
                else NA
            )
        output.append(result)
    return output


def _macro_rows(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    *,
    expected_models: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for _, group in grouped.items():
        if len(group) != expected_models:
            raise ValueError(
                f"component macro expected {expected_models} model rows for "
                f"{[group[0][field] for field in group_fields]}, got {len(group)}"
            )
        first = group[0]
        template_counts = {int(row["n_templates"]) for row in group}
        if len(template_counts) != 1:
            raise ValueError("component template denominator differs across models")
        clean_coverage = sum(float(row["clean_coverage"]) for row in group) / len(group)
        degraded_coverage = sum(float(row["degraded_coverage"]) for row in group) / len(group)
        summary: dict[str, Any] = {
            field: first[field] for field in group_fields
        }
        summary.update(
            {
                "component_name": CONSTRAINT_NAMES[str(first["component"])],
                "n_models": len(group),
                "n_templates_per_condition": next(iter(template_counts)),
                "mean_clean_coverage": clean_coverage,
                "mean_degraded_coverage": degraded_coverage,
                "coverage_drop": clean_coverage - degraded_coverage,
            }
        )
        for metric in METRICS:
            clean_values = [_numeric(row.get(f"clean_{metric}")) for row in group]
            degraded_values = [_numeric(row.get(f"degraded_{metric}")) for row in group]
            clean_numeric = [value for value in clean_values if value is not None]
            degraded_numeric = [value for value in degraded_values if value is not None]
            clean_score: float | str = sum(clean_numeric) / len(clean_numeric) if clean_numeric else NA
            degraded_score: float | str = (
                sum(degraded_numeric) / len(degraded_numeric) if degraded_numeric else NA
            )
            clean_number = _numeric(clean_score)
            degraded_number = _numeric(degraded_score)
            drop: float | str = (
                clean_number - degraded_number
                if clean_number is not None and degraded_number is not None
                else NA
            )
            summary[f"mean_clean_{metric}"] = clean_score
            summary[f"mean_degraded_{metric}"] = degraded_score
            summary[f"mean_{metric}_drop"] = drop
            summary[f"{metric}_relative_drop_pct"] = (
                100.0 * float(drop) / clean_number
                if isinstance(drop, (int, float)) and clean_number
                else NA
            )
        output.append(summary)
    return output


def build_condition_macro(
    present_slices: list[dict[str, Any]],
    *,
    expected_models: int,
) -> list[dict[str, Any]]:
    return _macro_rows(
        present_slices,
        ("component", "degradation_variant", "degradation_level"),
        expected_models=expected_models,
    )


def build_component_severity(
    present_pair_rows: list[dict[str, Any]],
    *,
    expected_models: int,
) -> list[dict[str, Any]]:
    model_rows = aggregate_pairs(
        present_pair_rows,
        ("model", "component", "degradation_level"),
    )
    templates_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in present_pair_rows:
        key = (str(row["model"]), str(row["component"]), str(row["degradation_level"]))
        templates_by_key[key].add(str(row["template_name"]))
    for row in model_rows:
        key = (str(row["model"]), str(row["component"]), str(row["degradation_level"]))
        row["n_templates"] = len(templates_by_key[key])
    macro = _macro_rows(
        model_rows,
        ("component", "degradation_level"),
        expected_models=expected_models,
    )
    for row in macro:
        row["n_variants"] = len(VARIANTS)
        row["spatial_variants"] = len(GEOMETRY_PRESERVING_VARIANTS)
    return macro


def build_excess_drop_severity(
    annotated_pairs: list[dict[str, Any]],
    *,
    expected_models: int,
) -> list[dict[str, Any]]:
    model_slices = aggregate_pairs(
        annotated_pairs,
        ("model", "component", "component_present", "degradation_level"),
    )
    template_sets: dict[tuple[str, str, bool, str], set[str]] = defaultdict(set)
    for row in annotated_pairs:
        key = (
            str(row["model"]),
            str(row["component"]),
            bool(row["component_present"]),
            str(row["degradation_level"]),
        )
        template_sets[key].add(str(row["template_name"]))
    for row in model_slices:
        key = (
            str(row["model"]),
            str(row["component"]),
            bool(row["component_present"]),
            str(row["degradation_level"]),
        )
        row["n_templates"] = len(template_sets[key])

    by_model: dict[tuple[str, str, str], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in model_slices:
        key = (
            str(row["model"]),
            str(row["component"]),
            str(row["degradation_level"]),
        )
        by_model[key][bool(row["component_present"])] = row

    model_contrasts: list[dict[str, Any]] = []
    for (model, component, level), by_presence in by_model.items():
        present = by_presence.get(True)
        absent = by_presence.get(False)
        if present is None or absent is None:
            continue
        result: dict[str, Any] = {
            "model": model,
            "component": component,
            "degradation_level": level,
            "n_with_templates": present["n_templates"],
            "n_without_templates": absent["n_templates"],
        }
        for metric in METRICS:
            drop_with = _numeric(present.get(f"{metric}_drop"))
            drop_without = _numeric(absent.get(f"{metric}_drop"))
            result[f"{metric}_drop_with"] = drop_with if drop_with is not None else NA
            result[f"{metric}_drop_without"] = drop_without if drop_without is not None else NA
            result[f"{metric}_excess_drop"] = (
                drop_with - drop_without
                if drop_with is not None and drop_without is not None
                else NA
            )
        model_contrasts.append(result)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in model_contrasts:
        grouped[(str(row["component"]), str(row["degradation_level"]))].append(row)
    output: list[dict[str, Any]] = []
    for (component, level), rows in grouped.items():
        if len(rows) != expected_models:
            raise ValueError(
                f"severity component contrast expected {expected_models} models for "
                f"{component}/{level}, got {len(rows)}"
            )
        with_counts = {int(row["n_with_templates"]) for row in rows}
        without_counts = {int(row["n_without_templates"]) for row in rows}
        if len(with_counts) != 1 or len(without_counts) != 1:
            raise ValueError("component contrast template counts differ across models")
        summary: dict[str, Any] = {
            "component": component,
            "component_name": CONSTRAINT_NAMES[component],
            "degradation_level": level,
            "n_models": len(rows),
            "n_with_templates": next(iter(with_counts)),
            "n_without_templates": next(iter(without_counts)),
        }
        for metric in METRICS:
            for suffix in ("drop_with", "drop_without", "excess_drop"):
                values = [
                    _numeric(row.get(f"{metric}_{suffix}"))
                    for row in rows
                ]
                numeric = [value for value in values if value is not None]
                summary[f"mean_{metric}_{suffix}"] = (
                    sum(numeric) / len(numeric) if numeric else NA
                )
        output.append(summary)
    return output


def _sort_slice(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["model_id"]).casefold(),
        COMPONENTS.index(str(row["component"])),
        VARIANTS.index(str(row["degradation_variant"])),
        LEVEL_ORDER[str(row["degradation_level"])],
        0 if bool(row.get("component_present")) else 1,
    )


def _write_markdown(
    path: Path,
    membership: list[dict[str, Any]],
    severity: list[dict[str, Any]],
    excess_severity: list[dict[str, Any]],
    n_reference_templates: int,
) -> None:
    counts = Counter(
        row["component"]
        for row in membership
        if bool(row["present"])
    )
    rule_by_component = {
        str(row["component"]): str(row["rule"])
        for row in membership
    }
    lines = [
        "# Visual degradation by template component",
        "",
        "Component labels are read from the metadata of each clean instance's template. They are never inferred from predictions, degraded images, or sample filenames. The five component slices overlap and are not a partition.",
        "",
        f"Widget grouping and dense key-field relation labels use the same corpus-level q75 rules as the clean constraint report. The thresholds are frozen from the {n_reference_templates} templates in the formal main index before selecting the 68 robustness templates.",
        "",
        "All seven completed model pairs are used in the macro below; backend differences are treated as negligible. Semantic drops average five degradation variants; spatial drops average only blur_noise, erode, and occlusion_stain.",
        "",
        "## Component labels",
        "",
        "| Component | Present | Absent | Metadata rule |",
        "| --- | ---: | ---: | --- |",
    ]
    for component in COMPONENTS:
        present = counts[component]
        lines.append(
            f"| {CONSTRAINT_NAMES[component]} | {present} | {68 - present} | {rule_by_component[component]} |"
        )
    lines.extend(
        [
            "",
            "## Seven-model macro by severity",
            "",
            "Positive drops mean the degraded prediction is worse than its paired clean prediction.",
            "",
            "| Component | Level | Templates | Schema drop | Value drop | TSR drop | R-F1@0.5 drop | R-F1@0.75 drop | LIG-F1 drop |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        severity,
        key=lambda item: (
            COMPONENTS.index(str(item["component"])),
            LEVEL_ORDER[str(item["degradation_level"])],
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["component_name"]),
                    str(row["degradation_level"]),
                    str(row["n_templates_per_condition"]),
                    _format(row["mean_Schema-nTED_drop"]),
                    _format(row["mean_Value-nED_drop"]),
                    _format(row["mean_TSR-path_drop"]),
                    _format(row["mean_R-F1@0.5_drop"]),
                    _format(row["mean_R-F1@0.75_drop"]),
                    _format(row["mean_LIG-F1_drop"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## High-severity component contrast",
            "",
            "Excess drop is the component-present drop minus the component-absent drop on the same seven models and five degradation variants. Positive values indicate greater sensitivity on component-present pages.",
            "",
            "| Component | With/without templates | Schema drop with | Schema drop without | Schema excess | Value drop with | Value drop without | Value excess |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in excess_severity:
        if row["degradation_level"] != "high":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["component_name"]),
                    f"{row['n_with_templates']}/{row['n_without_templates']}",
                    _format(row["mean_Schema-nTED_drop_with"]),
                    _format(row["mean_Schema-nTED_drop_without"]),
                    _format(row["mean_Schema-nTED_excess_drop"]),
                    _format(row["mean_Value-nED_drop_with"]),
                    _format(row["mean_Value-nED_drop_without"]),
                    _format(row["mean_Value-nED_excess_drop"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build visual-degradation component slices from template metadata labels."
    )
    parser.add_argument(
        "--pairs",
        default="outputs/robustness_exp/report_latest/visual_degradation_per_sample.jsonl",
    )
    parser.add_argument(
        "--clean-index",
        default="outputs/robustness_exp/robustness_clean_index.jsonl",
    )
    parser.add_argument("--main-index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--layout-root", default="newdataset-layout")
    parser.add_argument("--out", default="outputs/robustness_exp/report_latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_path = Path(args.pairs)
    clean_index_path = Path(args.clean_index)
    main_index_path = Path(args.main_index)
    layout_root = Path(args.layout_root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    pair_rows = read_jsonl(pair_path)
    clean_index_rows = read_jsonl(clean_index_path)
    reference_index_rows = read_jsonl(main_index_path)
    if len(pair_rows) != 7140:
        raise ValueError(f"expected 7140 visual-degradation pairs, got {len(pair_rows)}")
    if len(clean_index_rows) != 68:
        raise ValueError(f"expected 68 clean robustness samples, got {len(clean_index_rows)}")
    models = {str(row["model"]) for row in pair_rows}
    if len(models) != 7:
        raise ValueError(f"unexpected robustness model count: {len(models)}")

    lookup, label_metadata, membership = load_component_lookup(
        layout_root,
        clean_index_rows,
        reference_index_rows,
    )
    annotated = annotate_pairs(pair_rows, lookup)
    slices = aggregate_component_slices(annotated)
    present_slices = [row for row in slices if bool(row["component_present"])]
    contrasts = build_contrasts(slices)
    condition_macro = build_condition_macro(present_slices, expected_models=7)
    present_pairs = [
        row
        for row in annotated
        if bool(row["component_present"])
    ]
    severity = build_component_severity(present_pairs, expected_models=7)
    excess_severity = build_excess_drop_severity(
        annotated,
        expected_models=7,
    )

    slices.sort(key=_sort_slice)
    contrasts.sort(
        key=lambda row: (
            str(row["model_id"]).casefold(),
            COMPONENTS.index(str(row["component"])),
            VARIANTS.index(str(row["degradation_variant"])),
            LEVEL_ORDER[str(row["degradation_level"])],
        )
    )
    condition_macro.sort(
        key=lambda row: (
            COMPONENTS.index(str(row["component"])),
            VARIANTS.index(str(row["degradation_variant"])),
            LEVEL_ORDER[str(row["degradation_level"])],
        )
    )
    severity.sort(
        key=lambda row: (
            COMPONENTS.index(str(row["component"])),
            LEVEL_ORDER[str(row["degradation_level"])],
        )
    )
    excess_severity.sort(
        key=lambda row: (
            COMPONENTS.index(str(row["component"])),
            LEVEL_ORDER[str(row["degradation_level"])],
        )
    )

    for filename in OBSOLETE_REPORT_FILES:
        (out_dir / filename).unlink(missing_ok=True)
    _write_csv(out_dir / "visual_degradation_component_membership.csv", membership, MEMBERSHIP_COLUMNS)
    _write_csv(out_dir / "visual_degradation_by_component.csv", slices, SLICE_COLUMNS)
    _write_csv(out_dir / "visual_degradation_component_contrast.csv", contrasts, CONTRAST_COLUMNS)
    _write_csv(
        out_dir / "visual_degradation_component_condition_macro.csv",
        condition_macro,
        MACRO_COLUMNS,
    )
    _write_csv(
        out_dir / "visual_degradation_component_severity.csv",
        severity,
        SEVERITY_COLUMNS,
    )
    _write_csv(
        out_dir / "visual_degradation_component_excess_drop_severity.csv",
        excess_severity,
        EXCESS_SEVERITY_COLUMNS,
    )
    n_reference_templates = len(
        {str(row["template_name"]) for row in reference_index_rows}
    )
    _write_markdown(
        out_dir / "visual_degradation_component_results.md",
        membership,
        severity,
        excess_severity,
        n_reference_templates,
    )

    present_counts = Counter(
        str(row["component"])
        for row in membership
        if bool(row["present"])
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "pairs": str(pair_path),
            "pairs_sha256": _sha256(pair_path),
            "clean_index": str(clean_index_path),
            "clean_index_sha256": _sha256(clean_index_path),
            "main_index": str(main_index_path),
            "main_index_sha256": _sha256(main_index_path),
            "layout_root": str(layout_root),
        },
        "label_source": "each robustness instance is joined by template_name to its template metadata; no component label is inferred from predictions, degraded images, or filenames",
        "threshold_scope": "q75 thresholds are computed from templates in the formal main index before selecting the robustness subset; unrelated layout files are excluded",
        "n_reference_templates": n_reference_templates,
        "components": list(COMPONENTS),
        "component_names": {component: CONSTRAINT_NAMES[component] for component in COMPONENTS},
        "label_metadata": label_metadata,
        "n_clean_samples": len(clean_index_rows),
        "n_models": len(models),
        "n_pair_rows": len(pair_rows),
        "n_annotated_pair_component_rows": len(annotated),
        "n_membership_rows": len(membership),
        "component_present_counts": dict(present_counts),
        "component_absent_counts": {
            component: len(clean_index_rows) - present_counts[component]
            for component in COMPONENTS
        },
        "slice_overlap": "component slices overlap and must not be summed as a partition",
        "formal_macro": "unweighted mean of all seven completed model pairs on component-present pages; backend differences are treated as negligible",
        "spatial_policy": (
            "R-F1@0.5, R-F1@0.75, and LIG-F1 are numeric only for "
            f"{sorted(GEOMETRY_PRESERVING_VARIANTS)}; dilate and perspective_skew remain NA"
        ),
        "drop": "clean score minus degraded score; positive means degradation loss",
        "contrast": "excess_drop = component-present robustness drop minus component-absent robustness drop; component contrast is insufficient when either side has no template",
        "outputs": {
            "membership_rows": len(membership),
            "slice_rows": len(slices),
            "contrast_rows": len(contrasts),
            "condition_macro_rows": len(condition_macro),
            "severity_rows": len(severity),
            "excess_drop_severity_rows": len(excess_severity),
        },
    }
    write_json(out_dir / "visual_degradation_component_metadata.json", metadata)
    print(f"wrote visual-degradation component report -> {out_dir}")


if __name__ == "__main__":
    main()
