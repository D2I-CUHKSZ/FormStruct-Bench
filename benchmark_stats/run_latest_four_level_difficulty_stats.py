#!/usr/bin/env python3
"""Compute the latest four-level template difficulty distribution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.add_form_difficulty import (  # noqa: E402
    DIFFICULTY_SOURCE,
    FORMULA,
    build_context_score,
    build_normalization_config,
    clamp,
    load_records,
    score_structural,
)


LEVEL_ORDER = ["L1", "L2", "L3", "L4"]
LEVEL_NAMES = {"L1": "easy", "L2": "medium", "L3": "hard", "L4": "expert"}
LEVEL_COLORS = {"L1": "#3D8F70", "L2": "#4C89C6", "L3": "#D49A36", "L4": "#C65353"}
CALIBRATION_QUANTILES = {
    "L1_L2": 15.8655,
    "L2_L3": 50.0,
    "L3_L4": 84.1345,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute v0.4.1 D_main and normal-shaped calibrated L1-L4 statistics."
    )
    parser.add_argument("--layout-dir", default="newdataset-layout")
    parser.add_argument(
        "--domain-csv", default="outputs/domain_stats/title_domain_sample_stats.csv"
    )
    parser.add_argument("--output", default="outputs/difficulty_stats/latest_four_level")
    parser.add_argument("--instances-per-template", type=int, default=100)
    return parser.parse_args()


def read_domain_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            Path(row.get("file", "")).stem: row
            for row in csv.DictReader(handle)
            if row.get("file")
        }


def normalized_language(template_name: str, metadata: dict[str, Any]) -> str:
    raw = str(metadata.get("language") or "").strip()
    if raw == "Japan":
        return "Japanese"
    if raw:
        return raw
    if template_name.startswith("Arabic-"):
        return "Arabic"
    if template_name.startswith("de_"):
        return "German"
    if template_name.startswith("en_"):
        return "English"
    if template_name.startswith("es_"):
        return "Spanish"
    if template_name.startswith("ja_"):
        return "Japanese"
    if template_name.startswith("pt_"):
        return "Portuguese"
    if template_name.startswith("zn_en_"):
        return "Chinese+English"
    if template_name.startswith("zn_"):
        return "Chinese"
    return "Unknown"


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def assign_level(score: float, thresholds: dict[str, float]) -> str:
    if score < thresholds["L1_L2"]:
        return "L1"
    if score < thresholds["L2_L3"]:
        return "L2"
    if score < thresholds["L3_L4"]:
        return "L3"
    return "L4"


def score_templates(
    layout_dir: Path,
    domain_lookup: dict[str, dict[str, str]],
    instances_per_template: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    paths = sorted(layout_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No JSON files found under {layout_dir}")
    records = load_records(paths)
    config = build_normalization_config(records, "dataset_percentile_80_templates")

    scored: list[dict[str, Any]] = []
    d_main_values: list[float] = []
    for record in records:
        structural = score_structural(record["structural"], config)
        context = build_context_score(
            record["data"], record["structural"]["features"], config
        )
        d_main = clamp(float(structural["score"]) + float(context["score"]), 0.0, 2.0)
        d_main_values.append(d_main)
        data = record["data"]
        metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        template_name = record["path"].stem
        domain = domain_lookup.get(template_name, {})
        scored.append(
            {
                "template_name": template_name,
                "file": record["path"].name,
                "sample_id": data.get("id", ""),
                "language": normalized_language(template_name, metadata),
                "coarse_domain": domain.get("coarse_domain", ""),
                "fine_domain": domain.get("fine_domain", ""),
                "S_form": round(float(structural["score"]), 4),
                "C_context": round(float(context["score"]), 4),
                "D_main": round(d_main, 4),
                "benchmark_instances": instances_per_template,
            }
        )

    thresholds = {
        name: percentile(d_main_values, quantile)
        for name, quantile in CALIBRATION_QUANTILES.items()
    }
    for row in scored:
        level = assign_level(float(row["D_main"]), thresholds)
        row["difficulty_level"] = level
        row["difficulty_name"] = LEVEL_NAMES[level]
    return scored, config, thresholds


def summarize_levels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_templates = len(rows)
    total_instances = sum(int(row["benchmark_instances"]) for row in rows)
    output = []
    for level in LEVEL_ORDER:
        level_rows = [row for row in rows if row["difficulty_level"] == level]
        scores = [float(row["D_main"]) for row in level_rows]
        instances = sum(int(row["benchmark_instances"]) for row in level_rows)
        output.append(
            {
                "difficulty_level": level,
                "difficulty_name": LEVEL_NAMES[level],
                "templates": len(level_rows),
                "template_pct": len(level_rows) / total_templates * 100,
                "benchmark_instances": instances,
                "instance_pct": instances / total_instances * 100 if total_instances else 0.0,
                "D_main_min": min(scores) if scores else None,
                "D_main_median": float(np.median(scores)) if scores else None,
                "D_main_mean": float(np.mean(scores)) if scores else None,
                "D_main_max": max(scores) if scores else None,
            }
        )
    return output


def cross_tab(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        name = str(row.get(key) or "Unknown")
        grouped[name][str(row["difficulty_level"])] += 1
    output = []
    for name, counts in sorted(grouped.items()):
        output.append(
            {
                key: name,
                **{level: counts[level] for level in LEVEL_ORDER},
                "Total": sum(counts.values()),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in value)


def threshold_ranges(thresholds: dict[str, float]) -> dict[str, str]:
    return {
        "L1": rf"$D < {thresholds['L1_L2']:.4f}$",
        "L2": rf"${thresholds['L1_L2']:.4f} \leq D < {thresholds['L2_L3']:.4f}$",
        "L3": rf"${thresholds['L2_L3']:.4f} \leq D < {thresholds['L3_L4']:.4f}$",
        "L4": rf"$D \geq {thresholds['L3_L4']:.4f}$",
    }


def write_latex(
    path: Path, summary: list[dict[str, Any]], thresholds: dict[str, float]
) -> None:
    ranges = threshold_ranges(thresholds)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Latest four-level template difficulty distribution.}",
        r"\label{tab:latest_four_level_difficulty}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Level & Category & $D_{\mathrm{main}}$ range & Templates & (\%) & Instances \\",
        r"\midrule",
    ]
    for row in summary:
        level = str(row["difficulty_level"])
        lines.append(
            f"{level} & {latex_escape(str(row['difficulty_name']).title())} & "
            f"{ranges[level]} & {row['templates']} & {float(row['template_pct']):.2f} & "
            f"{int(row['benchmark_instances']):,} " + r"\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"\\textbf{{Total}} & & & \\textbf{{{sum(int(row['templates']) for row in summary)}}} & "
            f"\\textbf{{100.00}} & \\textbf{{{sum(int(row['benchmark_instances']) for row in summary):,}}} "
            + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_distribution(path: Path, summary: list[dict[str, Any]]) -> None:
    levels = [str(row["difficulty_level"]) for row in summary]
    counts = [int(row["templates"]) for row in summary]
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    bars = axis.bar(levels, counts, color=[LEVEL_COLORS[level] for level in levels], width=0.62)
    axis.set_xlabel("Difficulty level")
    axis.set_ylabel("Templates")
    axis.set_title("Latest four-level difficulty distribution")
    axis.set_ylim(0, max(counts) * 1.22)
    axis.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, count + 0.35, str(count), ha="center")
    figure.tight_layout()
    figure.savefig(path / "difficulty_distribution.png", dpi=200)
    figure.savefig(path / "difficulty_distribution.pdf")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    layout_dir = Path(args.layout_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    domain_lookup = read_domain_lookup(Path(args.domain_csv))
    rows, config, thresholds = score_templates(
        layout_dir, domain_lookup, args.instances_per_template
    )
    summary = summarize_levels(rows)
    language_counts = cross_tab(rows, "language")
    domain_counts = cross_tab(rows, "coarse_domain")

    metadata = {
        "difficulty_source": DIFFICULTY_SOURCE,
        "formula": FORMULA,
        "scoring_version": "v0.4.1",
        "calibration_method": "normal-shaped empirical calibration using D_main percentiles at normal CDF z=-1,0,+1",
        "calibration_quantiles": CALIBRATION_QUANTILES,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "templates": len(rows),
        "instances_per_template": args.instances_per_template,
        "benchmark_instances": sum(int(row["benchmark_instances"]) for row in rows),
        "level_counts": {row["difficulty_level"]: row["templates"] for row in summary},
        "does_not_modify_layout_metadata": True,
    }

    write_csv(output / "difficulty_template_assignments.csv", rows)
    write_csv(output / "difficulty_level_distribution.csv", summary)
    write_csv(output / "difficulty_by_language.csv", language_counts)
    write_csv(output / "difficulty_by_domain.csv", domain_counts)
    write_json(output / "difficulty_metadata.json", metadata)
    write_json(output / "difficulty_normalization_config.json", config)
    write_latex(output / "difficulty_level_distribution_table.tex", summary, thresholds)
    plot_distribution(output, summary)

    print(f"templates: {len(rows)}")
    print(f"thresholds: {metadata['thresholds']}")
    print(f"level counts: {metadata['level_counts']}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
