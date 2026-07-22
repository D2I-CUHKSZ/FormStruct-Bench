#!/usr/bin/env python3
"""Summarize structural complexity from difficulty_main_v0_4.S_form."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


GROUP_ORDER = [
    "region_section_complexity",
    "hierarchy_complexity",
    "field_relation_density",
    "widget_grouping_complexity",
    "local_grid_irregularity",
]

RAW_FEATURE_ORDER = [
    "section_count",
    "region_count",
    "max_hierarchy_depth",
    "key_value_region_count",
    "relation_edge_count",
    "multi_value_key_count",
    "selection_control_count",
    "option_group_count",
    "max_values_per_key",
    "local_grid_count",
    "line_item_group_count",
    "irregular_grid",
]

GROUP_LABELS = {
    "region_section_complexity": "Region/Section",
    "hierarchy_complexity": "Hierarchy",
    "field_relation_density": "Field Relations",
    "widget_grouping_complexity": "Widgets",
    "local_grid_irregularity": "Grid/Line Items",
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rounded(value: float) -> float:
    return round(float(value), 4)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "min": rounded(min(values)),
        "p25": rounded(percentile(values, 25)),
        "median": rounded(statistics.median(values)),
        "mean": rounded(statistics.mean(values)),
        "p75": rounded(percentile(values, 75)),
        "p90": rounded(percentile(values, 90)),
        "p95": rounded(percentile(values, 95)),
        "max": rounded(max(values)),
    }


def read_records(layout_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(layout_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        difficulty = data.get("metadata", {}).get("difficulty_main_v0_4", {})
        s_form = difficulty.get("S_form", {})
        groups = s_form.get("groups", {})
        if not isinstance(groups, dict):
            continue

        group_scores = {
            group_name: float(groups.get(group_name, {}).get("score", 0.0))
            for group_name in GROUP_ORDER
        }
        raw_features: dict[str, float] = {}
        normalized_features: dict[str, float] = {}
        for group in groups.values():
            for feature_name, feature in group.get("features", {}).items():
                raw = feature.get("raw", 0)
                if isinstance(raw, bool):
                    raw_features[feature_name] = float(int(raw))
                elif isinstance(raw, (int, float)):
                    raw_features[feature_name] = float(raw)
                normalized_features[feature_name] = float(feature.get("normalized", 0.0))

        records.append(
            {
                "file": path.name,
                "id": data.get("id"),
                "difficulty_level": difficulty.get("difficulty_level"),
                "D_main": float(difficulty.get("D_main", 0.0)),
                "S_form": float(s_form.get("score", 0.0)),
                "C_context": float(difficulty.get("C_context", {}).get("score", 0.0)),
                "group_scores": group_scores,
                "raw_features": raw_features,
                "normalized_features": normalized_features,
            }
        )
    return records


def s_form_bucket(score: float) -> str:
    if score < 0.2:
        return "[0.0,0.2)"
    if score < 0.4:
        return "[0.2,0.4)"
    if score < 0.6:
        return "[0.4,0.6)"
    if score < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "id",
        "difficulty_level",
        "D_main",
        "S_form",
        "C_context",
        *GROUP_ORDER,
        *RAW_FEATURE_ORDER,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["S_form"], reverse=True):
            row: dict[str, Any] = {
                "file": record["file"],
                "id": record["id"],
                "difficulty_level": record["difficulty_level"],
                "D_main": rounded(record["D_main"]),
                "S_form": rounded(record["S_form"]),
                "C_context": rounded(record["C_context"]),
            }
            row.update({name: rounded(record["group_scores"][name]) for name in GROUP_ORDER})
            row.update(
                {
                    name: rounded(record["raw_features"].get(name, 0.0))
                    for name in RAW_FEATURE_ORDER
                }
            )
            writer.writerow(row)


def make_markdown(records: list[dict[str, Any]]) -> str:
    s_scores = [record["S_form"] for record in records]
    bucket_counts = Counter(s_form_bucket(score) for score in s_scores)
    group_summaries = {
        group: summarize([record["group_scores"][group] for record in records])
        for group in GROUP_ORDER
    }
    feature_summaries = {
        feature: summarize([record["raw_features"].get(feature, 0.0) for record in records])
        for feature in RAW_FEATURE_ORDER
    }
    top_records = sorted(records, key=lambda record: record["S_form"], reverse=True)[:10]

    lines = [
        "# Structural Complexity Report",
        "",
        f"Total samples: {len(records)}",
        "",
        "## S_form Distribution",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summarize(s_scores).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## S_form Buckets", "", "| bucket | count |", "| --- | ---: |"])
    for bucket in ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]:
        lines.append(f"| {bucket} | {bucket_counts.get(bucket, 0)} |")

    lines.extend(["", "## Group Score Summary", ""])
    lines.append("| group | min | p25 | median | mean | p75 | p90 | p95 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in GROUP_ORDER:
        summary = group_summaries[group]
        lines.append(
            "| {group} | {min} | {p25} | {median} | {mean} | {p75} | {p90} | {p95} | {max} |".format(
                group=GROUP_LABELS[group], **summary
            )
        )

    lines.extend(["", "## Raw Feature Summary", ""])
    lines.append("| feature | min | p25 | median | mean | p75 | p90 | p95 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for feature in RAW_FEATURE_ORDER:
        summary = feature_summaries[feature]
        lines.append(
            "| {feature} | {min} | {p25} | {median} | {mean} | {p75} | {p90} | {p95} | {max} |".format(
                feature=feature, **summary
            )
        )

    lines.extend(["", "## Top Structural Samples", ""])
    lines.append("| rank | file | S_form | level | dominant groups |")
    lines.append("| ---: | --- | ---: | --- | --- |")
    for rank, record in enumerate(top_records, start=1):
        dominant = sorted(
            record["group_scores"].items(), key=lambda item: item[1], reverse=True
        )[:2]
        dominant_text = ", ".join(
            f"{GROUP_LABELS[name]}={rounded(score)}" for name, score in dominant
        )
        lines.append(
            f"| {rank} | {record['file']} | {rounded(record['S_form'])} | "
            f"{record['difficulty_level']} | {dominant_text} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_svg(records: list[dict[str, Any]], output_path: Path) -> None:
    bucket_order = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    counts = Counter(s_form_bucket(record["S_form"]) for record in records)
    max_count = max(counts.values(), default=1)
    width, height = 860, 520
    chart_left, chart_bottom = 90, 430
    bar_width, gap = 105, 34
    max_bar_height = 290
    colors = ["#8fcfda", "#74b9a0", "#f0c45c", "#ee8b5a", "#d95f5f"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="520" viewBox="0 0 860 520">',
        '<rect width="860" height="520" fill="#ffffff"/>',
        '<text x="44" y="54" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#263238">S_form Structural Complexity Distribution</text>',
        f'<text x="44" y="84" font-family="Arial, sans-serif" font-size="15" fill="#263238">Total samples: {len(records)}</text>',
        f'<line x1="{chart_left}" y1="{chart_bottom}" x2="800" y2="{chart_bottom}" stroke="#546e7a" stroke-width="1"/>',
    ]
    for index, bucket in enumerate(bucket_order):
        count = counts.get(bucket, 0)
        bar_height = 0 if max_count == 0 else (count / max_count) * max_bar_height
        x = chart_left + 35 + index * (bar_width + gap)
        y = chart_bottom - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[index]}"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#263238">{count}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{chart_bottom + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#263238">{bucket}</text>'
        )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structural complexity statistics.")
    parser.add_argument("--layout-dir", default="newdataset-layout")
    parser.add_argument("--report", default="reports/structural_complexity_report.md")
    parser.add_argument("--csv", default="reports/structural_complexity_samples.csv")
    parser.add_argument("--svg", default="outputs/structural_complexity_distribution.svg")
    args = parser.parse_args()

    records = read_records(Path(args.layout_dir))
    if not records:
        raise SystemExit("No difficulty_main_v0_4.S_form records found.")

    write_csv(records, Path(args.csv))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(make_markdown(records), encoding="utf-8")
    write_svg(records, Path(args.svg))

    summary = summarize([record["S_form"] for record in records])
    print(f"Samples: {len(records)}")
    print(f"S_form summary: {summary}")
    print(f"Wrote: {args.report}")
    print(f"Wrote: {args.csv}")
    print(f"Wrote: {args.svg}")


if __name__ == "__main__":
    main()
