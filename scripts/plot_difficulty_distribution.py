#!/usr/bin/env python3
"""Plot difficulty_main_v0_4 level distribution as a pie chart SVG."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5"]
LEVEL_COLORS = {
    "L1": "#6fbf73",
    "L2": "#8fcfda",
    "L3": "#f0c45c",
    "L4": "#ee8b5a",
    "L5": "#d95f5f",
}


def collect_distribution(layout_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted(layout_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        difficulty = metadata.get("difficulty_main_v0_4", {})
        level = difficulty.get("difficulty_level")
        if isinstance(level, str):
            counts[level] += 1
    return counts


def polar_to_cartesian(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def pie_slice_path(
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
) -> str:
    start_x, start_y = polar_to_cartesian(cx, cy, radius, start_angle)
    end_x, end_y = polar_to_cartesian(cx, cy, radius, end_angle)
    large_arc = 1 if end_angle - start_angle > math.pi else 0
    return (
        f"M {cx:.3f} {cy:.3f} "
        f"L {start_x:.3f} {start_y:.3f} "
        f"A {radius:.3f} {radius:.3f} 0 {large_arc} 1 {end_x:.3f} {end_y:.3f} "
        "Z"
    )


def svg_text(x: float, y: float, text: str, size: int = 16, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#263238">{text}</text>'
    )


def render_svg(counts: Counter[str], output_path: Path) -> None:
    total = sum(counts.values())
    if total == 0:
        raise SystemExit("No difficulty_main_v0_4.difficulty_level values found.")

    width, height = 860, 560
    cx, cy, radius = 285, 285, 185
    angle = -math.pi / 2
    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="560" viewBox="0 0 860 560">',
        '<rect width="860" height="560" fill="#ffffff"/>',
        svg_text(44, 54, "difficulty_main_v0_4 Level Distribution", 24, "700"),
        svg_text(44, 84, f"Total samples: {total}", 15),
    ]

    legend_y = 165
    for index, level in enumerate(LEVEL_ORDER):
        count = counts.get(level, 0)
        if count <= 0:
            continue
        fraction = count / total
        next_angle = angle + (2 * math.pi * fraction)
        color = LEVEL_COLORS[level]
        parts.append(f'<path d="{pie_slice_path(cx, cy, radius, angle, next_angle)}" fill="{color}"/>')

        mid_angle = (angle + next_angle) / 2
        label_x, label_y = polar_to_cartesian(cx, cy, radius * 0.62, mid_angle)
        percent = fraction * 100
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="Arial, sans-serif" '
            f'font-size="16" font-weight="700" fill="#263238">{level}</text>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y + 22:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#263238">{percent:.1f}%</text>'
        )

        y = legend_y + index * 54
        parts.extend(
            [
                f'<rect x="565" y="{y - 17}" width="22" height="22" rx="3" fill="{color}"/>',
                svg_text(600, y, f"{level}: {count} samples ({percent:.1f}%)", 17, "700"),
            ]
        )
        angle = next_angle

    parts.append(
        '<circle cx="285" cy="285" r="185" fill="none" stroke="#ffffff" stroke-width="2"/>'
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw difficulty level distribution as an SVG pie chart.")
    parser.add_argument("--layout-dir", default="newdataset-layout")
    parser.add_argument(
        "--output",
        default="outputs/difficulty_main_v0_4_distribution.svg",
        help="SVG output path.",
    )
    args = parser.parse_args()

    counts = collect_distribution(Path(args.layout_dir))
    render_svg(counts, Path(args.output))
    print(f"Distribution: {dict((level, counts[level]) for level in LEVEL_ORDER if counts[level])}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
