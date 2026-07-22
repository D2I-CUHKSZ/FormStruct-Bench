#!/usr/bin/env python3
"""Summarize template annotation metadata from a directory of JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def count_value(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def bool_value(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}

    array = np.array(values, dtype=float)
    return {
        "min": int(np.min(array)),
        "median": round(float(np.median(array)), 2),
        "mean": round(float(np.mean(array)), 2),
        "max": int(np.max(array)),
    }


def read_annotation(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <annotation_json_dir>", file=sys.stderr)
        return 2

    annotation_dir = Path(sys.argv[1])

    regions: list[float] = []
    fields: list[float] = []
    local_grids: list[float] = []
    widgets: list[float] = []
    irregular: list[bool] = []

    for path in sorted(annotation_dir.glob("*.json")):
        data = read_annotation(path)
        if data is None:
            continue

        metadata = nested_dict(data.get("metadata", {}))
        structure = nested_dict(metadata.get("layout_structure", {}))
        s_metadata = nested_dict(metadata.get("S", {}))

        regions.append(count_value(structure.get("region_count", 0)))
        fields.append(count_value(s_metadata.get("key_region_count", 0)))
        local_grids.append(count_value(structure.get("table_region_count", 0)))
        widgets.append(count_value(s_metadata.get("selection_control_count", 0)))
        irregular.append(bool_value(s_metadata.get("irregular_grid", False)))

    n = len(irregular)
    irregular_ratio = round(float(np.sum(irregular)) / n * 100, 1) if n else 0.0

    output = {
        "regions_per_template": summarize(regions),
        "fields_per_template": summarize(fields),
        "local_grids_per_template": summarize(local_grids),
        "widgets_per_template": summarize(widgets),
        "irregular_region_ratio_pct": irregular_ratio,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
