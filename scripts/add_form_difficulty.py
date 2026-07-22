#!/usr/bin/env python3
"""Compute form-like TSR difficulty metadata.

This script performs the two required passes:

1. Dataset pass: extract observable form/layout features and build a frozen
   percentile normalization config.
2. Per-sample pass: write metadata.difficulty_main_v0_4 without replacing any
   legacy difficulty fields.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DIFFICULTY_SOURCE = "auto_rule_v0.4.1_s_plus_c_no_vision_calibrated_level"
FORMULA = "D_main = S_form + C_context; visual readability is tag only"
DEFAULT_NORMALIZATION_SOURCE = "pilot_dataset_percentile"
NORMALIZATION_METHOD = "percentile_rank_midrank_zero_floor_v0_4"
GROUP_AGGREGATION = "max"
LEVELING_METHOD = "pilot_d_main_quantile_calibrated_v0_4_1"
LEVEL_CALIBRATION_QUANTILES = {
    "L1_L2": 15,
    "L2_L3": 35,
    "L3_L4": 65,
    "L4_L5": 85,
}

STRUCTURAL_GROUP_WEIGHTS = {
    "region_section_complexity": 0.20,
    "hierarchy_complexity": 0.20,
    "field_relation_density": 0.20,
    "widget_grouping_complexity": 0.20,
    "local_grid_irregularity": 0.20,
}

STRUCTURAL_FEATURE_GROUPS = {
    "region_section_complexity": ["section_count", "region_count"],
    "hierarchy_complexity": ["max_hierarchy_depth"],
    "field_relation_density": [
        "key_value_region_count",
        "relation_edge_count",
        "multi_value_key_count",
    ],
    "widget_grouping_complexity": [
        "selection_control_count",
        "option_group_count",
        "max_values_per_key",
    ],
    "local_grid_irregularity": [
        "local_grid_count",
        "line_item_group_count",
        "irregular_grid",
    ],
}

NORMALIZED_STRUCTURAL_FEATURES = [
    "section_count",
    "region_count",
    "key_region_count",
    "value_region_count",
    "key_value_region_count",
    "selection_control_count",
    "option_group_count",
    "multi_value_key_count",
    "max_values_per_key",
    "relation_edge_count",
    "max_hierarchy_depth",
    "local_grid_count",
    "line_item_group_count",
    "irregular_grid",
]

CONTEXT_WEIGHTS = {
    "page_scope": 0.20,
    "multi_section_layout": 0.15,
    "multi_table_context": 0.15,
    "mixed_layout": 0.15,
    "reading_order_dependency": 0.15,
    "text_direction": 0.15,
    "cross_region_dependency": 0.05,
}

ABSOLUTE_DIFFICULTY_LEVELS = [
    ("L1", 0.0, 0.4),
    ("L2", 0.4, 0.8),
    ("L3", 0.8, 1.2),
    ("L4", 1.2, 1.6),
    ("L5", 1.6, 2.000000001),
]

WIDGET_DATA_TYPES = {
    "checkbox",
    "check_box",
    "radio",
    "radio_button",
    "char_box",
    "character_box",
    "signature_line",
    "signature",
    "select",
    "dropdown",
    "option",
    "toggle",
    "switch",
}

APPROVAL_DEPENDENCY_TERMS = {
    "approval",
    "approver",
    "comment",
    "date",
    "name",
    "signature",
    "review",
    "verification",
}


def clean_number(value: float) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if float(value).is_integer():
        return int(value)
    return round(float(value), 6)


def round_score(value: float) -> float:
    return round(clamp(value, 0.0, 1.0), 4)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no"}:
            return 0.0
        try:
            number = float(stripped)
        except ValueError:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return None


def normalize_count(value: Any) -> int | float | bool:
    if isinstance(value, bool):
        return value
    number = as_number(value)
    if number is None:
        return 0
    return clean_number(number)


def is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def is_list(value: Any) -> bool:
    return isinstance(value, list)


def bbox_is_valid(bbox: Any) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(item, (int, float)) for item in bbox)
    )


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def data_type_of(node: dict[str, Any]) -> str:
    return str(node.get("data_type", "")).strip().lower().replace("-", "_")


def is_selection_control(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if data_type_of(node) in WIDGET_DATA_TYPES:
        return True
    mark = node.get("mark")
    if mark not in (None, "", False):
        return True
    if isinstance(node.get("checked"), bool):
        return True
    return False


def iter_field_nodes(fields: Any) -> Any:
    if isinstance(fields, dict):
        yield fields
        for key in ("keys",):
            for child in fields.get(key, []) if isinstance(fields.get(key), list) else []:
                yield from iter_field_nodes(child)
    elif isinstance(fields, list):
        for item in fields:
            yield from iter_field_nodes(item)


def extract_tree_features(data: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "section_count": 0,
        "region_count": 0,
        "key_region_count": 0,
        "value_region_count": 0,
        "selection_control_count": 0,
        "option_group_count": 0,
        "multi_value_key_count": 0,
        "max_values_per_key": 0,
        "relation_edge_count": 0,
        "max_hierarchy_depth": 0,
        "local_grid_count": 0,
        "line_item_group_count": 0,
        "irregular_grid": False,
    }
    semantic_keys: list[str] = []
    key_bboxes: list[list[float]] = []
    container_node_count = 0

    def count_value_candidate(value: Any, depth: int) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                count_value_candidate(item, depth)
            return
        stats["value_region_count"] += 1
        if isinstance(value, dict):
            if bbox_is_valid(value.get("bbox")):
                key_bboxes.append(value["bbox"])
            if is_selection_control(value):
                stats["selection_control_count"] += 1
            if value.get("value") is not None:
                stats["relation_edge_count"] += 1
                count_value_candidate(value.get("value"), depth + 1)
            nested_values = value.get("values")
            if isinstance(nested_values, list):
                stats["relation_edge_count"] += len(nested_values)
                for nested_item in nested_values:
                    count_value_candidate(nested_item, depth + 1)
            nested_keys = value.get("keys")
            if isinstance(nested_keys, list):
                child_count = sum(1 for child in nested_keys if isinstance(child, dict))
                stats["relation_edge_count"] += child_count
                for child in nested_keys:
                    walk_node(child, depth + 1)

    def walk_node(node: Any, depth: int) -> None:
        nonlocal container_node_count
        if not isinstance(node, dict):
            return
        stats["max_hierarchy_depth"] = max(stats["max_hierarchy_depth"], depth)

        semantic_key = node.get("semantic_key")
        if semantic_key not in (None, ""):
            stats["key_region_count"] += 1
            semantic_keys.append(str(semantic_key))
            if bbox_is_valid(node.get("bbox")):
                key_bboxes.append(node["bbox"])

        if is_selection_control(node):
            stats["selection_control_count"] += 1

        label = " ".join(
            str(node.get(name, "")) for name in ("semantic_key", "original_label", "data_type")
        ).lower()
        if "line item" in label or "line_item" in label:
            stats["line_item_group_count"] += 1
        if "grid" in label or "table" in label:
            stats["local_grid_count"] += 1
        if "irregular" in label:
            stats["irregular_grid"] = True

        child_keys = node.get("keys")
        if isinstance(child_keys, list):
            container_node_count += 1
            child_count = sum(1 for child in child_keys if isinstance(child, dict))
            stats["relation_edge_count"] += child_count
            for child in child_keys:
                walk_node(child, depth + 1)

        if node.get("value") is not None:
            stats["relation_edge_count"] += 1
            count_value_candidate(node.get("value"), depth + 1)

        values = node.get("values")
        if isinstance(values, list):
            stats["relation_edge_count"] += len(values)
            stats["max_values_per_key"] = max(stats["max_values_per_key"], len(values))
            if len(values) > 1:
                stats["multi_value_key_count"] += 1
            if any(is_selection_control(item) for item in values):
                stats["option_group_count"] += 1
            for item in values:
                count_value_candidate(item, depth + 1)

    fields = data.get("fields", [])
    if isinstance(fields, list):
        stats["section_count"] = max(1, len(fields))
        for item in fields:
            walk_node(item, 2)
    elif isinstance(fields, dict):
        stats["section_count"] = 1
        walk_node(fields, 2)

    stats["region_count"] = max(stats["section_count"], container_node_count)
    return {
        "features": stats,
        "semantic_keys": semantic_keys,
        "bboxes": key_bboxes,
        "evidence": {
            "source": "fields_recursive_extraction",
            "relation_edge_count_estimated": True,
            "section_region_count_estimated": True,
        },
    }


def choose_feature(
    candidates: list[tuple[Any, str, bool]],
    default: Any = 0,
) -> tuple[Any, dict[str, Any]]:
    for value, source, estimated in candidates:
        if value is None:
            continue
        if isinstance(value, bool):
            return value, {"source": source, "estimated": estimated}
        number = as_number(value)
        if number is not None:
            return clean_number(number), {"source": source, "estimated": estimated}
    return default, {"source": "default", "estimated": True}


def build_structural_features(
    data: dict[str, Any], extracted: dict[str, Any]
) -> dict[str, Any]:
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    structural = metadata.get("S", {}) if isinstance(metadata.get("S"), dict) else {}
    layout = (
        metadata.get("layout_structure", {})
        if isinstance(metadata.get("layout_structure"), dict)
        else {}
    )
    extracted_features = extracted["features"]

    feature_specs: dict[str, list[tuple[Any, str, bool]]] = {
        "section_count": [
            (layout.get("section_count"), "metadata.layout_structure.section_count", False),
            (structural.get("section_count"), "metadata.S.section_count", False),
            (extracted_features.get("section_count"), "fields_recursive_extraction", True),
        ],
        "region_count": [
            (layout.get("region_count"), "metadata.layout_structure.region_count", False),
            (structural.get("region_count"), "metadata.S.region_count", False),
            (extracted_features.get("region_count"), "fields_recursive_extraction", True),
        ],
        "key_region_count": [
            (structural.get("key_region_count"), "metadata.S.key_region_count", False),
            (extracted_features.get("key_region_count"), "fields_recursive_extraction", True),
        ],
        "value_region_count": [
            (structural.get("value_region_count"), "metadata.S.value_region_count", False),
            (extracted_features.get("value_region_count"), "fields_recursive_extraction", True),
        ],
        "selection_control_count": [
            (
                structural.get("selection_control_count"),
                "metadata.S.selection_control_count",
                False,
            ),
            (
                extracted_features.get("selection_control_count"),
                "fields_recursive_extraction",
                True,
            ),
        ],
        "option_group_count": [
            (structural.get("option_group_count"), "metadata.S.option_group_count", False),
            (extracted_features.get("option_group_count"), "fields_recursive_extraction", True),
        ],
        "multi_value_key_count": [
            (
                structural.get("multi_value_key_count"),
                "metadata.S.multi_value_key_count",
                False,
            ),
            (
                extracted_features.get("multi_value_key_count"),
                "fields_recursive_extraction",
                True,
            ),
        ],
        "max_values_per_key": [
            (structural.get("max_values_per_key"), "metadata.S.max_values_per_key", False),
            (extracted_features.get("max_values_per_key"), "fields_recursive_extraction", True),
        ],
        "relation_edge_count": [
            (
                structural.get("relation_edge_count"),
                "metadata.S.relation_edge_count",
                False,
            ),
            (
                extracted_features.get("relation_edge_count"),
                "fields_recursive_extraction_estimate",
                True,
            ),
        ],
        "max_hierarchy_depth": [
            (
                structural.get("max_hierarchy_depth"),
                "metadata.S.max_hierarchy_depth",
                False,
            ),
            (
                extracted_features.get("max_hierarchy_depth"),
                "fields_recursive_extraction",
                True,
            ),
        ],
        "local_grid_count": [
            (structural.get("local_grid_count"), "metadata.S.local_grid_count", False),
            (
                layout.get("table_region_count"),
                "metadata.layout_structure.table_region_count_as_local_grid_count",
                True,
            ),
            (extracted_features.get("local_grid_count"), "fields_recursive_extraction", True),
        ],
        "line_item_group_count": [
            (
                structural.get("line_item_group_count"),
                "metadata.S.line_item_group_count",
                False,
            ),
            (
                layout.get("line_item_group_count"),
                "metadata.layout_structure.line_item_group_count",
                False,
            ),
            (
                extracted_features.get("line_item_group_count"),
                "fields_recursive_extraction",
                True,
            ),
        ],
        "irregular_grid": [
            (structural.get("irregular_grid"), "metadata.S.irregular_grid", False),
            (extracted_features.get("irregular_grid"), "fields_recursive_extraction", True),
        ],
    }

    features: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for name, candidates in feature_specs.items():
        value, value_evidence = choose_feature(candidates)
        features[name] = normalize_count(value)
        evidence[name] = value_evidence

    key_value_region_count = as_number(features["key_region_count"]) or 0.0
    key_value_region_count += as_number(features["value_region_count"]) or 0.0
    features["key_value_region_count"] = clean_number(key_value_region_count)
    evidence["key_value_region_count"] = {
        "source": "derived:key_region_count+value_region_count",
        "estimated": evidence["key_region_count"]["estimated"]
        or evidence["value_region_count"]["estimated"],
    }

    return {"features": features, "evidence": evidence}


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
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def numeric_feature_value(value: Any) -> float:
    number = as_number(value)
    return 0.0 if number is None else number


def build_normalization_config(
    records: list[dict[str, Any]], normalization_source: str
) -> dict[str, Any]:
    config_features: dict[str, Any] = {}
    for feature_name in NORMALIZED_STRUCTURAL_FEATURES:
        values = [
            numeric_feature_value(record["structural"]["features"].get(feature_name, 0))
            for record in records
        ]
        distribution = sorted(clean_number(value) for value in values)
        config_features[feature_name] = {
            "count": len(distribution),
            "min": clean_number(min(values) if values else 0.0),
            "max": clean_number(max(values) if values else 0.0),
            "quantiles": {
                "P25": clean_number(percentile(values, 25)),
                "P50": clean_number(percentile(values, 50)),
                "P75": clean_number(percentile(values, 75)),
                "P90": clean_number(percentile(values, 90)),
                "P95": clean_number(percentile(values, 95)),
            },
            "distribution": distribution,
        }

    return {
        "version": "v0.4.1",
        "difficulty_source": DIFFICULTY_SOURCE,
        "normalization_source": normalization_source,
        "normalization_method": NORMALIZATION_METHOD,
        "dataset_file_count": len(records),
        "dataset_files": [record["path"].name for record in records],
        "single_sample_policy": "error_without_existing_config",
        "feature_group_aggregation": GROUP_AGGREGATION,
        "structural_feature_groups": STRUCTURAL_FEATURE_GROUPS,
        "structural_group_weights": STRUCTURAL_GROUP_WEIGHTS,
        "context_weights": CONTEXT_WEIGHTS,
        "context_rules": {
            "page_scope": "1.0 if table_area_ratio >= 0.75; 0.5 if >= 0.40; else 0.0",
            "multi_table_context": "min(max(table_count_on_page - 1, 0) / 2, 1.0)",
            "text_direction": "text_direction_complexity / 3",
        },
        "leveling_method": LEVELING_METHOD,
        "level_calibration_quantiles": LEVEL_CALIBRATION_QUANTILES,
        "absolute_difficulty_levels": [
            {"level": level, "min_inclusive": low, "max_exclusive": high}
            for level, low, high in ABSOLUTE_DIFFICULTY_LEVELS
        ],
        "difficulty_levels": [],
        "features": config_features,
    }


def percentile_norm(value: Any, distribution: list[Any]) -> float:
    numeric_value = numeric_feature_value(value)
    ordered = sorted(numeric_feature_value(item) for item in distribution)
    if not ordered:
        return 0.0
    if numeric_value <= 0.0 and min(ordered) >= 0.0:
        return 0.0
    if ordered[0] == ordered[-1]:
        return 0.5 if ordered[0] > 0.0 else 0.0
    left = bisect.bisect_left(ordered, numeric_value)
    right = bisect.bisect_right(ordered, numeric_value)
    if left == right:
        return clamp(left / len(ordered))
    midrank = left + ((right - left) / 2.0)
    return clamp(midrank / len(ordered))


def aggregate_scores(values: list[float]) -> float:
    if not values:
        return 0.0
    if GROUP_AGGREGATION == "mean":
        return sum(values) / len(values)
    return max(values)


def score_structural(
    structural: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    features = structural["features"]
    feature_evidence = structural["evidence"]
    config_features = config["features"]
    groups: dict[str, Any] = {}
    weighted_total = 0.0

    for group_name, feature_names in STRUCTURAL_FEATURE_GROUPS.items():
        normalized_values: list[float] = []
        feature_records: dict[str, Any] = {}
        for feature_name in feature_names:
            raw_value = features.get(feature_name, 0)
            distribution = config_features.get(feature_name, {}).get("distribution", [])
            normalized = percentile_norm(raw_value, distribution)
            normalized_values.append(normalized)
            evidence = feature_evidence.get(feature_name, {})
            feature_records[feature_name] = {
                "raw": raw_value,
                "normalized": round_score(normalized),
                "source": evidence.get("source", "unknown"),
                "estimated": bool(evidence.get("estimated", True)),
            }
        group_score = aggregate_scores(normalized_values)
        groups[group_name] = {
            "score": round_score(group_score),
            "features": feature_records,
            "evidence": {
                "aggregation": f"{GROUP_AGGREGATION}_percentile_normalized_feature",
                "weight": STRUCTURAL_GROUP_WEIGHTS[group_name],
            },
        }
        weighted_total += STRUCTURAL_GROUP_WEIGHTS[group_name] * group_score

    return {
        "score": round_score(weighted_total),
        "normalization_source": config["normalization_source"],
        "groups": groups,
    }


def get_context_value(
    metadata: dict[str, Any], key: str
) -> tuple[float | None, dict[str, Any] | None]:
    context = metadata.get("C", {}) if isinstance(metadata.get("C"), dict) else {}
    if key not in context:
        return None, None
    number = as_number(context[key])
    if number is None:
        return None, None
    return number, {"source": f"metadata.C.{key}", "estimated": False}


def collect_all_bboxes(value: Any) -> list[list[float]]:
    bboxes: list[list[float]] = []
    if isinstance(value, dict):
        bbox = value.get("bbox")
        if bbox_is_valid(bbox):
            bboxes.append(bbox)
        for child in value.values():
            bboxes.extend(collect_all_bboxes(child))
    elif isinstance(value, list):
        for child in value:
            bboxes.extend(collect_all_bboxes(child))
    return bboxes


def estimate_area_ratio(data: dict[str, Any], metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    width = as_number(data.get("original_width"))
    height = as_number(data.get("original_height"))
    page_area = (width or 0.0) * (height or 0.0)
    layout = (
        metadata.get("layout_structure", {})
        if isinstance(metadata.get("layout_structure"), dict)
        else {}
    )
    page_bbox = layout.get("page_bbox")
    if bbox_is_valid(page_bbox):
        page_area = bbox_area(page_bbox)

    if page_area <= 0.0:
        return 0.0, {
            "source": "default:no_page_size",
            "estimated": True,
            "method": "no valid page area",
        }

    section_bboxes = [
        section["bbox"]
        for section in layout.get("sections", [])
        if isinstance(section, dict) and bbox_is_valid(section.get("bbox"))
    ]
    if section_bboxes:
        area = min(sum(bbox_area(bbox) for bbox in section_bboxes) / page_area, 1.0)
        return clamp(area), {
            "source": "metadata.layout_structure.sections.bbox",
            "estimated": True,
            "method": "sum_section_area_capped_at_page",
        }

    bboxes = collect_all_bboxes(data.get("fields", []))
    if not bboxes:
        return 0.0, {"source": "default:no_bboxes", "estimated": True}

    left = min(float(bbox[0]) for bbox in bboxes)
    top = min(float(bbox[1]) for bbox in bboxes)
    right = max(float(bbox[2]) for bbox in bboxes)
    bottom = max(float(bbox[3]) for bbox in bboxes)
    area = bbox_area([left, top, right, bottom]) / page_area
    return clamp(area), {
        "source": "fields_bbox_extent",
        "estimated": True,
        "method": "bounding_extent_area_ratio",
    }


def page_scope_from_area_ratio(table_area_ratio: float) -> float:
    if table_area_ratio >= 0.75:
        return 1.0
    if table_area_ratio >= 0.40:
        return 0.5
    return 0.0


def text_direction_complexity(metadata: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    direction = (
        metadata.get("text_direction", {})
        if isinstance(metadata.get("text_direction"), dict)
        else {}
    )
    direct = as_number(direction.get("text_direction_complexity"))
    if direct is not None:
        return int(clamp(direct, 0.0, 3.0)), {
            "source": "metadata.text_direction.text_direction_complexity",
            "estimated": False,
            "text_direction": direction,
        }

    values = " ".join(
        str(direction.get(key, ""))
        for key in (
            "reading_direction",
            "line_orientation",
            "writing_mode",
            "column_direction",
            "orientation",
        )
    ).lower()
    if any(
        bool(direction.get(key))
        for key in (
            "direction_impacts_grouping",
            "affects_field_grouping",
            "impacts_reading_order",
            "cross_region_direction_dependency",
        )
    ):
        complexity = 3
    elif "mixed" in values or "bidi" in values or ("ltr" in values and "rtl" in values):
        complexity = 2
    elif (
        direction.get("reading_direction") == "rtl"
        or str(direction.get("line_orientation", "")).lower() == "vertical"
        or str(direction.get("writing_mode", "")).lower().startswith("vertical")
        or "rotated" in values
    ):
        complexity = 1
    else:
        complexity = 0
    return complexity, {
        "source": "metadata.text_direction",
        "estimated": "text_direction_complexity" not in direction,
        "text_direction": direction,
    }


def tree_order_bboxes(data: dict[str, Any]) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for node in iter_field_nodes(data.get("fields", [])):
        if not isinstance(node, dict):
            continue
        if node.get("semantic_key") in (None, ""):
            continue
        bbox = node.get("bbox")
        if bbox_is_valid(bbox):
            bboxes.append(bbox)
    return bboxes


def has_non_monotonic_reading_order(
    data: dict[str, Any], metadata: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    bboxes = tree_order_bboxes(data)
    if len(bboxes) < 4:
        return False, {"checked_pairs": max(0, len(bboxes) - 1), "violation_ratio": 0.0}

    direction = (
        metadata.get("text_direction", {})
        if isinstance(metadata.get("text_direction"), dict)
        else {}
    )
    reading_direction = str(direction.get("reading_direction", "ltr")).lower()
    violations = 0
    checked = 0
    heights = [abs(float(bbox[3] - bbox[1])) for bbox in bboxes]
    row_tolerance = max(8.0, percentile(heights, 50) * 0.75)
    for previous, current in zip(bboxes, bboxes[1:]):
        checked += 1
        previous_x = (float(previous[0]) + float(previous[2])) / 2.0
        current_x = (float(current[0]) + float(current[2])) / 2.0
        previous_y = (float(previous[1]) + float(previous[3])) / 2.0
        current_y = (float(current[1]) + float(current[3])) / 2.0
        if current_y + row_tolerance < previous_y:
            violations += 1
        elif abs(current_y - previous_y) <= row_tolerance:
            if reading_direction == "rtl" and current_x > previous_x + row_tolerance:
                violations += 1
            elif reading_direction != "rtl" and current_x + row_tolerance < previous_x:
                violations += 1
    ratio = violations / checked if checked else 0.0
    return ratio >= 0.15, {"checked_pairs": checked, "violation_ratio": round(ratio, 4)}


def estimate_reading_order_dependency(
    data: dict[str, Any],
    metadata: dict[str, Any],
    structural_features: dict[str, Any],
    text_complexity: int,
) -> tuple[float, dict[str, Any]]:
    direct, evidence = get_context_value(metadata, "reading_order_dependency")
    if direct is not None:
        return clamp(direct), evidence or {}

    section_count = numeric_feature_value(structural_features.get("section_count", 0))
    region_count = numeric_feature_value(structural_features.get("region_count", 0))
    non_monotonic, order_evidence = has_non_monotonic_reading_order(data, metadata)

    score = 0.0
    if text_complexity >= 2:
        score += 0.35
    elif text_complexity == 1:
        score += 0.25
    if section_count >= 4:
        score += 0.12
    elif section_count >= 3:
        score += 0.08
    elif section_count > 1:
        score += 0.04
    if region_count >= 3:
        score += 0.08
    elif region_count > 1:
        score += 0.04
    if non_monotonic:
        score += 0.25

    return clamp(score), {
        "source": "estimated:text_direction+section_count+region_count+bbox_order",
        "estimated": True,
        "text_direction_complexity": text_complexity,
        "section_count": clean_number(section_count),
        "region_count": clean_number(region_count),
        "bbox_order": order_evidence,
        "rule_version": "conservative_v0_4_1",
    }


def normalized_key(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def estimate_cross_region_dependency(
    data: dict[str, Any], structural_features: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    direct, evidence = get_context_value(metadata, "cross_region_dependency")
    if direct is not None:
        return clamp(direct), evidence or {}

    semantic_keys = [
        normalized_key(str(node.get("semantic_key")))
        for node in iter_field_nodes(data.get("fields", []))
        if isinstance(node, dict) and node.get("semantic_key") not in (None, "")
    ]
    counts = Counter(semantic_keys)
    repeated_approval_keys = {
        key: count
        for key, count in counts.items()
        if count > 1 and any(term in key for term in APPROVAL_DEPENDENCY_TERMS)
    }
    section_count = numeric_feature_value(structural_features.get("section_count", 0))
    multi_value_count = numeric_feature_value(
        structural_features.get("multi_value_key_count", 0)
    )
    relation_edges = numeric_feature_value(structural_features.get("relation_edge_count", 0))
    key_value_count = numeric_feature_value(
        structural_features.get("key_value_region_count", 0)
    )

    score = 0.0
    if section_count >= 2 and repeated_approval_keys:
        score += min(0.35, 0.10 + 0.05 * len(repeated_approval_keys))
    if section_count > 2 and multi_value_count > 0:
        score += 0.10
    if section_count > 2 and relation_edges > key_value_count:
        score += 0.05

    return clamp(score), {
        "source": "estimated:repeated_approval_fields+multi_section_relations",
        "estimated": True,
        "repeated_approval_keys": repeated_approval_keys,
        "section_count": clean_number(section_count),
        "rule_version": "conservative_v0_4_1",
    }


def build_context_score(
    data: dict[str, Any],
    structural_features: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    layout = (
        metadata.get("layout_structure", {})
        if isinstance(metadata.get("layout_structure"), dict)
        else {}
    )

    table_area_ratio, table_area_evidence = get_context_value(metadata, "table_area_ratio")
    if table_area_ratio is None:
        table_area_ratio, table_area_evidence = estimate_area_ratio(data, metadata)
    table_area_ratio = clamp(table_area_ratio)
    page_scope = page_scope_from_area_ratio(table_area_ratio)

    table_count, table_count_evidence = get_context_value(metadata, "table_count_on_page")
    if table_count is None:
        layout_table_count = as_number(layout.get("table_region_count"))
        if layout_table_count is not None and layout_table_count > 0:
            table_count = layout_table_count
            table_count_evidence = {
                "source": "metadata.layout_structure.table_region_count",
                "estimated": True,
            }
        else:
            table_count = 1.0
            table_count_evidence = {
                "source": "default:one_form_like_table_on_page",
                "estimated": True,
            }

    section_count = structural_features.get("section_count", 0)
    section_distribution = config["features"].get("section_count", {}).get("distribution", [])
    multi_section_layout = percentile_norm(section_count, section_distribution)
    multi_table_context = clamp(max(float(table_count) - 1.0, 0.0) / 2.0)

    mixed_layout, mixed_layout_evidence = get_context_value(metadata, "mixed_layout")
    if mixed_layout is None:
        mixed_layout = 0.0
        mixed_layout_evidence = {
            "source": "default_missing_mixed_layout",
            "estimated": True,
        }
    mixed_layout = clamp(mixed_layout)

    text_complexity, text_evidence = text_direction_complexity(metadata)
    text_direction_score = clamp(text_complexity / 3.0)

    reading_order, reading_order_evidence = estimate_reading_order_dependency(
        data, metadata, structural_features, text_complexity
    )
    cross_region, cross_region_evidence = estimate_cross_region_dependency(
        data, structural_features
    )

    components = {
        "page_scope": page_scope,
        "multi_section_layout": multi_section_layout,
        "multi_table_context": multi_table_context,
        "mixed_layout": mixed_layout,
        "reading_order_dependency": reading_order,
        "text_direction": text_direction_score,
        "cross_region_dependency": cross_region,
    }
    score = sum(CONTEXT_WEIGHTS[name] * value for name, value in components.items())

    return {
        "score": round_score(score),
        "normalization_source": config["normalization_source"],
        "features": {
            "full_page_context": round_score(page_scope),
            "table_count_on_page": clean_number(float(table_count)),
            "table_area_ratio": round_score(table_area_ratio),
            "multi_section_layout": round_score(multi_section_layout),
            "mixed_layout": round_score(mixed_layout),
            "reading_order_dependency": round_score(reading_order),
            "text_direction_complexity": int(text_complexity),
            "text_direction": round_score(text_direction_score),
            "cross_region_dependency": round_score(cross_region),
        },
        "evidence": {
            "weights": CONTEXT_WEIGHTS,
            "page_scope": {
                "table_area_ratio": round_score(table_area_ratio),
                "rule": "1.0 if ratio >= 0.75; 0.5 if ratio >= 0.40; else 0.0",
                **(table_area_evidence or {}),
            },
            "table_count_on_page": table_count_evidence or {},
            "multi_section_layout": {
                "source": "section_count percentile normalization",
                "section_count": structural_features.get("section_count", 0),
                "estimated": False,
            },
            "mixed_layout": mixed_layout_evidence or {},
            "reading_order_dependency": reading_order_evidence,
            "text_direction_complexity": text_evidence,
            "cross_region_dependency": cross_region_evidence,
        },
    }


def visual_readability_tags(metadata: dict[str, Any]) -> dict[str, Any]:
    visual = metadata.get("V", {}) if isinstance(metadata.get("V"), dict) else {}
    tags: list[str] = []

    def add_tag(tag: Any) -> None:
        if tag in (None, "", False):
            return
        tag_text = str(tag)
        if tag_text.lower() in {"false", "none", "0"}:
            return
        if tag_text not in tags:
            tags.append(tag_text)

    for key, value in visual.items():
        if isinstance(value, bool) and value:
            add_tag(key)
        elif key in {"borderless_or_weak_grid", "readability_tag", "visual_tag"}:
            add_tag(value)

    result: dict[str, Any] = {"used_in_main_score": False, "tags": tags}
    for key, value in visual.items():
        result[key] = value
    return result


def legacy_fields_ignored(metadata: dict[str, Any]) -> list[str]:
    ignored: list[str] = []
    legacy = metadata.get("difficulty", {}) if isinstance(metadata.get("difficulty"), dict) else {}
    if "visual_score" in legacy:
        ignored.append("metadata.difficulty.visual_score")
    if "difficulty_score" in legacy:
        ignored.append("metadata.difficulty.difficulty_score")
    return ignored


def difficulty_level(score: float, config: dict[str, Any]) -> str:
    configured_levels = config.get("difficulty_levels")
    if isinstance(configured_levels, list) and configured_levels:
        for item in configured_levels:
            if not isinstance(item, dict):
                continue
            low = as_number(item.get("min_inclusive"))
            high = as_number(item.get("max_exclusive"))
            level = item.get("level")
            if low is None or high is None or not isinstance(level, str):
                continue
            if low <= score < high:
                return level
    for level, low, high in ABSOLUTE_DIFFICULTY_LEVELS:
        if low <= score < high:
            return level
    return "L5" if score >= 2.0 else "L1"


def absolute_difficulty_level(score: float) -> str:
    for level, low, high in ABSOLUTE_DIFFICULTY_LEVELS:
        if low <= score < high:
            return level
    return "L5" if score >= 2.0 else "L1"


def build_difficulty_metadata(
    data: dict[str, Any],
    structural: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    s_form = score_structural(structural, config)
    c_context = build_context_score(data, structural["features"], config)
    d_main = clamp(s_form["score"] + c_context["score"], 0.0, 2.0)
    return {
        "difficulty_source": DIFFICULTY_SOURCE,
        "formula": FORMULA,
        "D_main": round(d_main, 4),
        "difficulty_level": difficulty_level(d_main, config),
        "difficulty_leveling": {
            "method": config.get("leveling_method", "absolute_threshold_v0_4"),
            "normalization_source": config.get("normalization_source"),
            "thresholds": config.get("difficulty_levels", []),
            "absolute_threshold_level": absolute_difficulty_level(d_main),
        },
        "S_form": s_form,
        "C_context": c_context,
        "visual_readability_tags": visual_readability_tags(metadata),
        "legacy_fields_ignored": legacy_fields_ignored(metadata),
    }


def calibrate_difficulty_levels(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    d_main_values: list[float] = []
    for record in records:
        s_form = score_structural(record["structural"], config)
        c_context = build_context_score(record["data"], record["structural"]["features"], config)
        d_main_values.append(clamp(s_form["score"] + c_context["score"], 0.0, 2.0))

    thresholds = {
        name: percentile(d_main_values, quantile)
        for name, quantile in LEVEL_CALIBRATION_QUANTILES.items()
    }
    levels = [
        ("L1", 0.0, thresholds["L1_L2"]),
        ("L2", thresholds["L1_L2"], thresholds["L2_L3"]),
        ("L3", thresholds["L2_L3"], thresholds["L3_L4"]),
        ("L4", thresholds["L3_L4"], thresholds["L4_L5"]),
        ("L5", thresholds["L4_L5"], 2.000000001),
    ]
    config["difficulty_levels"] = [
        {
            "level": level,
            "min_inclusive": round(float(low), 4),
            "max_exclusive": round(float(high), 4),
        }
        for level, low, high in levels
    ]
    config["d_main_distribution"] = {
        "count": len(d_main_values),
        "min": round(min(d_main_values), 4),
        "max": round(max(d_main_values), 4),
        "quantiles": {
            "P15": round(thresholds["L1_L2"], 4),
            "P35": round(thresholds["L2_L3"], 4),
            "P65": round(thresholds["L3_L4"], 4),
            "P85": round(thresholds["L4_L5"], 4),
        },
        "distribution": [round(value, 4) for value in sorted(d_main_values)],
    }
    return config


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        extracted = extract_tree_features(data)
        structural = build_structural_features(data, extracted)
        records.append(
            {
                "path": path,
                "data": data,
                "extracted": extracted,
                "structural": structural,
            }
        )
    return records


def load_or_build_config(
    records: list[dict[str, Any]],
    config_path: Path,
    normalization_source: str,
    rebuild: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    if config_path.exists() and not rebuild:
        return json.loads(config_path.read_text(encoding="utf-8")), "loaded"
    if len(records) < 2:
        raise SystemExit(
            "Cannot build a formal difficulty normalization config from one sample. "
            "Provide --config pointing to an existing config."
        )
    config = calibrate_difficulty_levels(
        records, build_normalization_config(records, normalization_source)
    )
    if not dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return config, "built"


def write_records(records: list[dict[str, Any]], config: dict[str, Any], dry_run: bool) -> Counter:
    level_counts: Counter = Counter()
    for record in records:
        data = record["data"]
        metadata = data.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{record['path']}: metadata must be an object when present")
        difficulty = build_difficulty_metadata(data, record["structural"], config)
        metadata["difficulty_main_v0_4"] = difficulty
        level_counts[difficulty["difficulty_level"]] += 1
        if not dry_run:
            record["path"].write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return level_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add metadata.difficulty_main_v0_4 to form-like TSR JSON files."
    )
    parser.add_argument("--layout-dir", default="newdataset-layout")
    parser.add_argument(
        "--config",
        default="config/difficulty_main_v0_4_normalization.json",
        help="Frozen percentile normalization config path.",
    )
    parser.add_argument(
        "--normalization-source",
        default=DEFAULT_NORMALIZATION_SOURCE,
        help="Value stored in S_form.normalization_source.",
    )
    parser.add_argument("--rebuild-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir)
    config_path = Path(args.config)
    paths = sorted(layout_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No JSON files found in {layout_dir}")

    records = load_records(paths)
    config, config_action = load_or_build_config(
        records,
        config_path,
        args.normalization_source,
        args.rebuild_config,
        args.dry_run,
    )
    level_counts = write_records(records, config, args.dry_run)

    print(f"Processed {len(records)} files from {layout_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Normalization config: {config_action} {config_path}")
    print(f"Normalization source: {config.get('normalization_source')}")
    print(f"Difficulty levels: {dict(sorted(level_counts.items()))}")

    arabic_one = next((record for record in records if record["path"].name == "Arabic-1.json"), None)
    if arabic_one is not None:
        difficulty = arabic_one["data"]["metadata"]["difficulty_main_v0_4"]
        print(
            "Arabic-1.json: D_main={D_main}, level={difficulty_level}, "
            "S_form={S}, C_context={C}, text_direction_complexity={T}, tags={tags}".format(
                D_main=difficulty["D_main"],
                difficulty_level=difficulty["difficulty_level"],
                S=difficulty["S_form"]["score"],
                C=difficulty["C_context"]["score"],
                T=difficulty["C_context"]["features"]["text_direction_complexity"],
                tags=difficulty["visual_readability_tags"]["tags"],
            )
        )


if __name__ == "__main__":
    main()
