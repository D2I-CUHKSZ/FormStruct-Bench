#!/usr/bin/env python3
"""Compute paper-ready hierarchical form-table structure statistics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STRUCTURAL_UNITS = [
    ("semantic_sections", "Semantic sections"),
    ("table_regions", "Table regions"),
    ("field_groups", "Field groups"),
    ("label_field_units", "Label-field units"),
    ("multi_field_units", "Multi-field units"),
    ("selection_fields", "Selection fields"),
    ("structural_relation_edges", "Structural relation edges"),
    ("maximum_hierarchy_depth", "Maximum hierarchy depth"),
]

SELECTION_DATA_TYPES = {"checkbox", "check_box", "radio", "radio_button"}
LARGE_DIFF_ABSOLUTE = 3
LARGE_DIFF_RELATIVE = 0.25


@dataclass
class WarningRecord:
    sample_id: Any
    image: str
    file: str
    field: str
    message: str
    primary: Any = None
    fallback: Any = None


@dataclass
class TreeStats:
    field_groups: int = 0
    label_field_units: int = 0
    multi_field_units: int = 0
    selection_fields: int = 0
    structural_relation_edges: int = 0
    maximum_hierarchy_depth: int = 0


@dataclass
class SampleContext:
    path: Path | None
    index: int
    sample_id: Any
    image: str
    warnings: list[WarningRecord] = field(default_factory=list)

    @property
    def file(self) -> str:
        if self.path is None:
            return f"<sample:{self.index}>"
        return str(self.path)

    def warn(self, field_name: str, message: str, primary: Any = None, fallback: Any = None) -> None:
        self.warnings.append(
            WarningRecord(
                sample_id=self.sample_id,
                image=self.image,
                file=self.file,
                field=field_name,
                message=message,
                primary=primary,
                fallback=fallback,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hierarchical form-table structure statistics from annotation JSON."
    )
    parser.add_argument("--input", required=True, help="Annotation JSON file or directory of JSON files.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--template-root",
        default="",
        help="Optional dataset root whose immediate directory names define the templates to include.",
    )
    parser.add_argument(
        "--selection-counts",
        default="",
        help="Optional CSV with template_name,selection_fields overrides for corrected control counts.",
    )
    return parser.parse_args()


def load_selection_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"template_name", "selection_fields"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"selection-count CSV must contain {sorted(required)}")
        return {
            str(row["template_name"]): int(row["selection_fields"])
            for row in reader
        }


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    return None


def nested_get(data: dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_json_payload(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def iter_input_samples(input_path: Path) -> tuple[list[tuple[Path | None, int, dict[str, Any]]], list[dict[str, Any]]]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.json"))
    samples: list[tuple[Path | None, int, dict[str, Any]]] = []
    load_warnings: list[dict[str, Any]] = []

    for path in files:
        payload = load_json_payload(path)
        if payload is None:
            load_warnings.append(
                {
                    "sample_id": None,
                    "image": "",
                    "file": str(path),
                    "field": "input",
                    "message": "failed to parse JSON file",
                    "primary": None,
                    "fallback": None,
                }
            )
            continue
        if isinstance(payload, dict):
            samples.append((path, 0, payload))
        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    samples.append((path, index, item))
                else:
                    load_warnings.append(
                        {
                            "sample_id": None,
                            "image": "",
                            "file": f"{path}#{index}",
                            "field": "input",
                            "message": "list item is not a JSON object",
                            "primary": type(item).__name__,
                            "fallback": None,
                        }
                    )
        else:
            load_warnings.append(
                {
                    "sample_id": None,
                    "image": "",
                    "file": str(path),
                    "field": "input",
                    "message": "top-level JSON is neither object nor list",
                    "primary": type(payload).__name__,
                    "fallback": None,
                }
            )

    return samples, load_warnings


def child_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [child for child in as_list(node.get("keys")) if isinstance(child, dict)]


def value_items(node: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    value = node.get("value")
    if isinstance(value, dict):
        items.append(value)
    for item in as_list(node.get("values")):
        if isinstance(item, dict):
            items.append(item)
    return items


def is_selection_type(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in SELECTION_DATA_TYPES


def traverse_annotation_tree(fields: list[Any]) -> TreeStats:
    stats = TreeStats()
    root_nodes = [node for node in fields if isinstance(node, dict)]

    def visit(node: dict[str, Any], depth: int, is_outermost_root: bool) -> None:
        children = child_nodes(node)
        values = value_items(node)
        stats.maximum_hierarchy_depth = max(stats.maximum_hierarchy_depth, depth)

        if children and not is_outermost_root:
            stats.field_groups += 1

        if not children and isinstance(node.get("value"), dict):
            stats.label_field_units += 1

        node_values = as_list(node.get("values"))
        if len(node_values) > 1:
            stats.multi_field_units += 1

        if is_selection_type(node.get("data_type")):
            stats.selection_fields += 1
        for item in values:
            if is_selection_type(item.get("data_type")):
                stats.selection_fields += 1

        stats.structural_relation_edges += len(children)
        if isinstance(node.get("value"), dict):
            stats.structural_relation_edges += 1
        stats.structural_relation_edges += len(node_values)

        for child in children:
            visit(child, depth + 1, False)

    for node in root_nodes:
        visit(node, 1, True)

    return stats


def count_sections(layout: dict[str, Any]) -> int:
    sections = as_list(layout.get("sections"))
    return len(sections)


def count_table_regions(layout: dict[str, Any]) -> int:
    total = 0
    for section in as_list(layout.get("sections")):
        if isinstance(section, dict):
            total += len(as_list(section.get("regions")))
    return total


def choose_count(
    ctx: SampleContext,
    field_name: str,
    primary: Any,
    fallback: int,
    *,
    prefer_primary: bool = True,
) -> int:
    primary_int = as_int(primary)
    if primary_int is None:
        ctx.warn(field_name, "primary source missing or non-numeric; using fallback", primary, fallback)
        return fallback

    if primary_int != fallback:
        message = "primary source differs from recursive fallback"
        if is_large_difference(primary_int, fallback):
            message = "primary source differs substantially from recursive fallback"
        ctx.warn(field_name, message, primary_int, fallback)

    return primary_int if prefer_primary else fallback


def is_large_difference(primary: int, fallback: int) -> bool:
    diff = abs(primary - fallback)
    scale = max(abs(primary), abs(fallback), 1)
    return diff >= LARGE_DIFF_ABSOLUTE and diff / scale >= LARGE_DIFF_RELATIVE


def summarize(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    array = np.array(values, dtype=float)
    median = float(np.median(array))
    return {
        "min": int(np.min(array)),
        "median": int(median) if median.is_integer() else round(median, 1),
        "mean": round(float(np.mean(array)), 1),
        "max": int(np.max(array)),
    }


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}" if "." in str(value) else str(int(value))
        return f"{value:.1f}"
    return str(value)


def process_sample(path: Path | None, index: int, sample: dict[str, Any]) -> tuple[dict[str, Any], list[WarningRecord]]:
    sample_id = sample.get("id", "")
    image = str(sample.get("img", ""))
    ctx = SampleContext(path=path, index=index, sample_id=sample_id, image=image)

    metadata = as_dict(sample.get("metadata"))
    source_has_metadata = bool(metadata)
    if not source_has_metadata:
        ctx.warn("metadata", "sample has no metadata object")

    s_metadata = as_dict(metadata.get("S"))
    layout = as_dict(metadata.get("layout_structure"))
    fields = as_list(sample.get("fields"))
    tree_stats = traverse_annotation_tree(fields)

    semantic_sections = choose_count(
        ctx,
        "semantic_sections",
        layout.get("section_count"),
        count_sections(layout),
    )
    table_regions = choose_count(
        ctx,
        "table_regions",
        layout.get("region_count"),
        count_table_regions(layout),
    )
    multi_field_units = choose_count(
        ctx,
        "multi_field_units",
        s_metadata.get("multi_value_key_count"),
        tree_stats.multi_field_units,
    )
    selection_fields = choose_count(
        ctx,
        "selection_fields",
        s_metadata.get("selection_control_count"),
        tree_stats.selection_fields,
    )
    structural_relation_edges = choose_count(
        ctx,
        "structural_relation_edges",
        s_metadata.get("relation_edge_count"),
        tree_stats.structural_relation_edges,
    )
    maximum_hierarchy_depth = choose_count(
        ctx,
        "maximum_hierarchy_depth",
        s_metadata.get("max_hierarchy_depth"),
        tree_stats.maximum_hierarchy_depth,
    )

    row = {
        "sample_id": sample_id,
        "image": image,
        "language": metadata.get("language", ""),
        "semantic_sections": semantic_sections,
        "table_regions": table_regions,
        "field_groups": tree_stats.field_groups,
        "label_field_units": tree_stats.label_field_units,
        "multi_field_units": multi_field_units,
        "selection_fields": selection_fields,
        "structural_relation_edges": structural_relation_edges,
        "maximum_hierarchy_depth": maximum_hierarchy_depth,
        "key_region_count_raw": as_int(s_metadata.get("key_region_count")) or 0,
        "value_region_count_raw": as_int(s_metadata.get("value_region_count")) or 0,
        "option_group_count_raw": as_int(s_metadata.get("option_group_count")) or 0,
        "source_has_metadata": source_has_metadata,
        "warning_count": len(ctx.warnings),
    }
    return row, ctx.warnings


def build_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    return {
        key: summarize([as_int(row.get(key)) or 0 for row in rows])
        for key, _ in STRUCTURAL_UNITS
    }


def write_sample_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "image",
        "language",
        "semantic_sections",
        "table_regions",
        "field_groups",
        "label_field_units",
        "multi_field_units",
        "selection_fields",
        "structural_relation_edges",
        "maximum_hierarchy_depth",
        "key_region_count_raw",
        "value_region_count_raw",
        "option_group_count_raw",
        "source_has_metadata",
        "warning_count",
    ]
    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_summary_json(summary: dict[str, dict[str, int | float]], output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown_table(summary: dict[str, dict[str, int | float]], output_path: Path) -> None:
    lines = [
        "| Structural unit | Min | Median | Mean | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in STRUCTURAL_UNITS:
        stats = summary[key]
        lines.append(
            "| {label} | {min} | {median} | {mean} | {max} |".format(
                label=label,
                min=format_number(stats["min"]),
                median=format_number(stats["median"]),
                mean=f"{float(stats['mean']):.1f}",
                max=format_number(stats["max"]),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(
    summary: dict[str, dict[str, int | float]], output_path: Path, template_count: int
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Template-level structural statistics.}",
        r"\label{tab:hierarchical_stats}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Structural unit & Min & Median & Mean & Max\\",
        r"\midrule",
    ]
    for key, label in STRUCTURAL_UNITS:
        stats = summary[key]
        lines.append(
            "{label}\n  & {min} & {median} & {mean:.1f} & {max} \\\\".format(
                label=label,
                min=format_number(stats["min"]),
                median=format_number(stats["median"]),
                mean=float(stats["mean"]),
                max=format_number(stats["max"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tablenotes}",
            r"\small",
            (
                f"Statistics are computed over {template_count} templates from object-level "
                "form-table annotations. "
                "We use label-field units to denote the smallest structural units "
                "consisting of a localized label region, its associated field region, "
                "and their structural link. Current annotations do not include explicit "
                "cell-level grid coordinates; therefore, cell counts, merged-cell ratios, "
                "and local-grid counts are not reported."
            ),
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_warnings(
    load_warnings: list[dict[str, Any]],
    warnings: list[WarningRecord],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in load_warnings:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for warning in warnings:
            handle.write(json.dumps(warning.__dict__, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_counts: dict[str, int] | None = None
    if args.selection_counts:
        selection_path = Path(args.selection_counts)
        if not selection_path.is_file():
            raise SystemExit(f"selection-count CSV does not exist: {selection_path}")
        selection_counts = load_selection_counts(selection_path)

    samples, load_warnings = iter_input_samples(input_path)
    template_filter: set[str] | None = None
    if args.template_root:
        template_root = Path(args.template_root)
        if not template_root.is_dir():
            raise SystemExit(f"template root does not exist or is not a directory: {template_root}")
        template_filter = {path.name for path in template_root.iterdir() if path.is_dir()}
        samples = [
            item
            for item in samples
            if item[0] is not None and item[0].stem in template_filter
        ]
        loaded_templates = {path.stem for path, _, _ in samples if path is not None}
        missing_templates = sorted(template_filter - loaded_templates)
        if missing_templates:
            raise SystemExit(
                "templates missing from structural annotations: " + ", ".join(missing_templates)
            )
    rows: list[dict[str, Any]] = []
    all_warnings: list[WarningRecord] = []

    for path, index, sample in samples:
        row, warnings = process_sample(path, index, sample)
        if selection_counts is not None:
            template_name = path.stem if path is not None else ""
            if template_name not in selection_counts:
                raise SystemExit(
                    f"selection-count override missing template_name: {template_name}"
                )
            row["selection_fields"] = selection_counts[template_name]
            row["template_name"] = template_name
        rows.append(row)
        all_warnings.extend(warnings)
        row["warning_count"] = len(warnings)

    summary = build_summary(rows)
    if selection_counts is not None:
        row_templates = {str(row["template_name"]) for row in rows}
        extra_selection_templates = sorted(set(selection_counts) - row_templates)
        if extra_selection_templates:
            raise SystemExit(
                "selection-count overrides not present in filtered annotations: "
                + ", ".join(extra_selection_templates)
            )

    sample_csv = output_dir / "hierarchical_structure_sample_stats.csv"
    summary_json = output_dir / "hierarchical_structure_summary.json"
    table_tex = output_dir / "hierarchical_structure_table.tex"
    table_md = output_dir / "hierarchical_structure_table.md"
    warnings_jsonl = output_dir / "warnings.jsonl"
    metadata_json = output_dir / "hierarchical_structure_metadata.json"

    write_sample_csv(rows, sample_csv)
    write_summary_json(summary, summary_json)
    write_latex_table(summary, table_tex, len(rows))
    write_markdown_table(summary, table_md)
    write_warnings(load_warnings, all_warnings, warnings_jsonl)
    metadata_json.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "template_root": args.template_root or None,
                "template_count": len(rows),
                "template_filter_count": len(template_filter) if template_filter is not None else None,
                "selection_counts": args.selection_counts or None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"loaded samples: {len(samples) + len(load_warnings)}")
    print(f"valid samples: {len(rows)}")
    print(f"warnings: {len(load_warnings) + len(all_warnings)}")
    print("output file paths:")
    for path in [sample_csv, summary_json, table_tex, table_md, warnings_jsonl, metadata_json]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
