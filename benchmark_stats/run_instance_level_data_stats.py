#!/usr/bin/env python3
"""Build paper-ready instance-level data statistics for FormTSR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NUMERIC_VALUE_RE = re.compile(r"[\d\s.,:+\-/%()]+")
DATE_VALUE_RE = re.compile(
    r"(?:\d{4}[-/.年]\d{1,2}|\d{1,2}[-/.月]\d{1,2}|\d{1,2}月\d{1,2}日)"
)
WHITESPACE_RE = re.compile(r"\s+")

STRUCTURE_METRICS = [
    ("leaf_count", "Leaf fields"),
    ("internal_semantic_group_count", "Internal semantic groups"),
    ("semantic_tree_edge_count", "Semantic tree edges"),
    ("mean_leaf_depth", "Mean leaf depth"),
    ("tree_depth", "Maximum hierarchy depth"),
    ("active_answer_path_count", "Active answer paths"),
]

CONTENT_METRICS = [
    ("total_characters", "Total answer characters"),
    ("mean_value_length", "Mean value length"),
    ("max_value_length", "Maximum value length"),
    ("long_value_count_40", "Values with at least 40 characters"),
    ("numeric_value_ratio", "Numeric-value ratio"),
    ("date_value_ratio", "Date-value ratio"),
    ("duplicate_value_ratio", "Duplicate-value ratio"),
]

INSTANCE_METRICS = [*STRUCTURE_METRICS, *CONTENT_METRICS]

VISUAL_METRICS = [
    ("blur_score_cv", "Laplacian sharpness"),
    ("blur_score_cv_denoised", "Smoothed Laplacian sharpness"),
    ("foreground_ratio", "Non-white-pixel ratio"),
    ("mean_gray", "Mean grayscale intensity"),
    ("std_gray", "Grayscale contrast"),
    ("edge_density", "Edge density"),
    ("visual_entropy_bits", "Grayscale entropy"),
    ("colorfulness", "Colorfulness"),
]

TAG_DEFINITIONS = {
    "conditional_schema_variant": "answer-path set differs from the modal path set of its template",
    "field_count_variant": "leaf-field count differs from the template mode",
    "hierarchy_depth_variant": "answer-tree depth differs from the template mode",
    "repeated_group_variant": "list-shape signature differs from the template mode",
    "long_context": "total answer characters >= 1.25 times the template median",
    "short_context": "total answer characters <= 0.75 times the template median",
    "long_single_value": "maximum value length >= 1.5 times the template median",
}

TAG_LABELS = {
    "conditional_schema_variant": "Conditional schema variant",
    "field_count_variant": "Field-count variant",
    "hierarchy_depth_variant": "Hierarchy-depth variant",
    "repeated_group_variant": "Repeated-group variant",
    "long_context": "Long context",
    "short_context": "Short context",
    "long_single_value": "Long single value",
}

TEMPLATE_DIVERSITY_METRICS = [
    ("unique_answer_ratio", "Unique-answer ratio"),
    ("unique_schema_count", "Unique schemas"),
    ("schema_entropy_bits", "Schema entropy"),
    ("modal_schema_share", "Modal-schema share"),
    ("variable_path_count", "Variable answer paths"),
    ("mean_variable_path_entropy_bits", "Mean variable-path entropy"),
    ("mean_field_unique_value_ratio", "Mean field unique-value ratio"),
    ("total_characters_cv", "Total-character CV"),
    ("max_value_length_cv", "Maximum-value-length CV"),
]

TEMPLATE_VISUAL_DIVERSITY_METRICS = [
    ("blur_score_cv_denoised_cv", "Smoothed-sharpness CV"),
    ("foreground_ratio_cv", "Non-white-pixel-ratio CV"),
    ("mean_gray_cv", "Mean-grayscale CV"),
    ("std_gray_cv", "Grayscale-contrast CV"),
    ("edge_density_cv", "Edge-density CV"),
]

FEATURE_LABELS = dict([*INSTANCE_METRICS, *VISUAL_METRICS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute instance-level structure, content, diversity, duplicate, and visual statistics."
    )
    parser.add_argument("--data-root", default="FormTSR/datasets")
    parser.add_argument("--output", default="outputs/instance_stats")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional global instance limit for smoke tests.")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--value-jaccard-threshold", type=float, default=0.90)
    parser.add_argument("--phash-distance-threshold", type=int, default=5)
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    text = unicodedata.normalize("NFKC", text)
    return WHITESPACE_RE.sub(" ", text).strip().casefold()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def signature_hash(items: Iterable[str]) -> str:
    encoded = "\n".join(items).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def analyze_answer(payload: Any) -> dict[str, Any]:
    scalar_values: list[str] = []
    scalar_paths: list[str] = []
    path_values: dict[str, list[str]] = defaultdict(list)
    list_lengths: list[int] = []
    branching_factors: list[int] = []
    leaf_depths: list[int] = []
    dict_node_count = 0
    maximum_depth = 0

    def walk(value: Any, path: tuple[str, ...] = (), depth: int = 0) -> None:
        nonlocal dict_node_count, maximum_depth
        maximum_depth = max(maximum_depth, depth)
        if isinstance(value, dict):
            dict_node_count += 1
            branching_factors.append(len(value))
            for key, child in value.items():
                walk(child, (*path, str(key)), depth + 1)
            return
        if isinstance(value, list):
            list_lengths.append(len(value))
            branching_factors.append(len(value))
            for child in value:
                walk(child, (*path, "[]"), depth + 1)
            return

        normalized = normalize_value(value)
        path_text = "/".join(path)
        scalar_values.append(normalized)
        scalar_paths.append(path_text)
        leaf_depths.append(depth)
        if normalized:
            path_values[path_text].append(normalized)

    walk(payload)
    nonempty_values = [value for value in scalar_values if value]
    lengths = [len(value) for value in nonempty_values]
    numeric_count = sum(bool(NUMERIC_VALUE_RE.fullmatch(value)) for value in nonempty_values)
    date_count = sum(bool(DATE_VALUE_RE.search(value)) for value in nonempty_values)
    duplicate_count = len(nonempty_values) - len(set(nonempty_values))
    schema_paths = tuple(sorted(set(scalar_paths)))
    list_shape = tuple(list_lengths)
    container_count = dict_node_count + len(list_lengths)
    tree_node_count = container_count + len(scalar_values)

    return {
        "leaf_count": len(scalar_values),
        "internal_semantic_group_count": max(dict_node_count - 1, 0),
        "semantic_tree_edge_count": max(tree_node_count - 1, 0),
        "mean_branching_factor": (
            statistics.mean(branching_factors) if branching_factors else 0.0
        ),
        "mean_leaf_depth": statistics.mean(leaf_depths) if leaf_depths else 0.0,
        "active_answer_path_count": len(schema_paths),
        "nonempty_leaf_count": len(nonempty_values),
        "empty_leaf_count": len(scalar_values) - len(nonempty_values),
        "total_characters": sum(lengths),
        "mean_value_length": statistics.mean(lengths) if lengths else 0.0,
        "max_value_length": max(lengths, default=0),
        "long_value_count_40": sum(length >= 40 for length in lengths),
        "long_value_count_80": sum(length >= 80 for length in lengths),
        "numeric_value_count": numeric_count,
        "numeric_value_ratio": numeric_count / len(nonempty_values) if nonempty_values else 0.0,
        "date_value_count": date_count,
        "date_value_ratio": date_count / len(nonempty_values) if nonempty_values else 0.0,
        "duplicate_value_count": duplicate_count,
        "duplicate_value_ratio": duplicate_count / len(nonempty_values) if nonempty_values else 0.0,
        "list_node_count": len(list_lengths),
        "list_item_count": sum(list_lengths),
        "max_list_length": max(list_lengths, default=0),
        "tree_depth": maximum_depth,
        "schema_signature": signature_hash(schema_paths),
        "list_shape_signature": signature_hash(str(length) for length in list_shape),
        "canonical_answer_hash": canonical_json_hash(payload),
        "_schema_paths": schema_paths,
        "_list_shape": list_shape,
        "_path_values": dict(path_values),
        "_value_set": frozenset(nonempty_values),
    }


def discover_instances(data_root: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_path in sorted(data_root.glob("*/*/answer.json")):
        template_name = label_path.parents[1].name
        instance_id = label_path.parent.name
        expected_image = label_path.parent / f"{template_name}-{instance_id}.png"
        image_candidates = sorted(label_path.parent.glob("*.png"))
        image_path = expected_image if expected_image.exists() else (image_candidates[0] if image_candidates else None)
        rows.append(
            {
                "sample_id": f"{template_name}__{instance_id}",
                "template_name": template_name,
                "instance_id": instance_id,
                "label_path": str(label_path),
                "image_path": str(image_path) if image_path else "",
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def load_answer_rows(instances: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for instance in instances:
        label_path = Path(instance["label_path"])
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "sample_id": instance["sample_id"],
                    "path": str(label_path),
                    "error": str(exc),
                }
            )
            continue
        rows.append({**instance, **analyze_answer(payload)})
    return rows, errors


def perceptual_hash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8]
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = (low_frequency > median).flatten()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def grayscale_entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def image_colorfulness(image: np.ndarray) -> float:
    blue, green, red = cv2.split(image.astype(np.float32))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    chroma_std = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    chroma_mean = math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    return chroma_std + 0.3 * chroma_mean


def analyze_image(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    image_path = Path(str(row.get("image_path") or ""))
    if not image_path.exists():
        return row["sample_id"], None, f"missing image: {image_path}"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return row["sample_id"], None, f"failed to decode image: {image_path}"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 100, 200)
    height, width = gray.shape
    image_phash = perceptual_hash(gray)
    metrics = {
        "image_width": width,
        "image_height": height,
        "image_orientation": "landscape" if width > height else "portrait",
        "image_file_size_bytes": image_path.stat().st_size,
        "blur_score_cv": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "blur_score_cv_denoised": float(cv2.Laplacian(denoised, cv2.CV_64F).var()),
        "foreground_ratio": float(np.mean(gray < 245)),
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "edge_density": float(np.mean(edges > 0)),
        "visual_entropy_bits": grayscale_entropy(gray),
        "colorfulness": image_colorfulness(image),
        "perceptual_hash": f"{image_phash:016x}",
        "image_pixel_hash": hashlib.sha256(image.tobytes()).hexdigest(),
        "_perceptual_hash_int": image_phash,
    }
    return row["sample_id"], metrics, None


def add_visual_metrics(
    rows: list[dict[str, Any]], workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_sample = {row["sample_id"]: row for row in rows}
    errors: list[dict[str, str]] = []
    worker_count = max(1, workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, (sample_id, metrics, error) in enumerate(executor.map(analyze_image, rows), start=1):
            if metrics is not None:
                by_sample[sample_id].update(metrics)
            if error:
                errors.append({"sample_id": sample_id, "path": by_sample[sample_id]["image_path"], "error": error})
            if index % 250 == 0 or index == len(rows):
                print(f"visual metrics: {index}/{len(rows)}", flush=True)
    return rows, errors


def mode(values: Iterable[Any]) -> Any:
    return Counter(values).most_common(1)[0][0]


def coefficient_of_variation(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if not len(array) or float(array.mean()) == 0.0:
        return 0.0
    return float(array.std(ddof=0) / abs(array.mean()))


def shannon_entropy(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=float)
    if not len(values) or float(values.sum()) == 0.0:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))


def annotate_instance_tags(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["template_name"]].append(row)

    for template_rows in grouped.values():
        modal_schema = mode(row["schema_signature"] for row in template_rows)
        modal_leaf_count = mode(row["leaf_count"] for row in template_rows)
        modal_depth = mode(row["tree_depth"] for row in template_rows)
        modal_list_shape = mode(row["list_shape_signature"] for row in template_rows)
        median_characters = statistics.median(row["total_characters"] for row in template_rows)
        median_max_value = statistics.median(row["max_value_length"] for row in template_rows)

        for row in template_rows:
            row["conditional_schema_variant"] = row["schema_signature"] != modal_schema
            row["field_count_variant"] = row["leaf_count"] != modal_leaf_count
            row["hierarchy_depth_variant"] = row["tree_depth"] != modal_depth
            row["repeated_group_variant"] = row["list_shape_signature"] != modal_list_shape
            row["long_context"] = row["total_characters"] >= 1.25 * median_characters
            row["short_context"] = row["total_characters"] <= 0.75 * median_characters
            row["long_single_value"] = row["max_value_length"] >= 1.5 * max(median_max_value, 1)


def build_template_diversity(rows: list[dict[str, Any]], include_visual: bool) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["template_name"]].append(row)

    output: list[dict[str, Any]] = []
    for template_name, template_rows in sorted(grouped.items()):
        n_instances = len(template_rows)
        schema_counts = Counter(row["schema_signature"] for row in template_rows)
        all_paths = set().union(*(set(row["_schema_paths"]) for row in template_rows))
        path_counts = Counter(
            path for row in template_rows for path in set(row["_schema_paths"])
        )
        variable_paths = [path for path in all_paths if 0 < path_counts[path] < n_instances]
        variable_path_entropies = [
            binary_entropy(path_counts[path] / n_instances) for path in variable_paths
        ]

        values_by_path: dict[str, list[str]] = defaultdict(list)
        for row in template_rows:
            for path, values in row["_path_values"].items():
                values_by_path[path].extend(values)
        field_unique_ratios = [
            len(set(values)) / len(values)
            for values in values_by_path.values()
            if values
        ]

        unique_answers = len({row["canonical_answer_hash"] for row in template_rows})
        result = {
            "template_name": template_name,
            "instances": n_instances,
            "unique_answers": unique_answers,
            "unique_answer_ratio": unique_answers / n_instances,
            "duplicate_answer_instances": n_instances - unique_answers,
            "unique_schema_count": len(schema_counts),
            "schema_entropy_bits": shannon_entropy(schema_counts.values()),
            "schema_entropy_normalized": (
                shannon_entropy(schema_counts.values()) / math.log2(len(schema_counts))
                if len(schema_counts) > 1
                else 0.0
            ),
            "modal_schema_share": max(schema_counts.values()) / n_instances,
            "rare_schema_count_lt5pct": sum(
                count / n_instances < 0.05 for count in schema_counts.values()
            ),
            "variable_path_count": len(variable_paths),
            "mean_variable_path_entropy_bits": (
                statistics.mean(variable_path_entropies) if variable_path_entropies else 0.0
            ),
            "mean_field_unique_value_ratio": (
                statistics.mean(field_unique_ratios) if field_unique_ratios else 0.0
            ),
            "median_field_unique_value_ratio": (
                statistics.median(field_unique_ratios) if field_unique_ratios else 0.0
            ),
            "leaf_count_min": min(row["leaf_count"] for row in template_rows),
            "leaf_count_max": max(row["leaf_count"] for row in template_rows),
            "total_characters_cv": coefficient_of_variation(
                row["total_characters"] for row in template_rows
            ),
            "max_value_length_cv": coefficient_of_variation(
                row["max_value_length"] for row in template_rows
            ),
        }
        if include_visual:
            for metric, _ in VISUAL_METRICS:
                result[f"{metric}_cv"] = coefficient_of_variation(
                    row[metric] for row in template_rows if metric in row
                )
        output.append(result)
    return output


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {key: 0.0 for key in ["count", "min", "p25", "median", "mean", "p75", "p95", "max"]}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def build_metric_summary(
    rows: list[dict[str, Any]], metrics: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for key, label in metrics:
        values = [float(row[key]) for row in rows if key in row and pd.notna(row[key])]
        output.append({"metric": key, "label": label, **numeric_summary(values)})
    return output


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root

    def component_count(self) -> int:
        return len({self.find(index) for index in range(len(self.parent))})


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


def build_duplicate_summary(
    rows: list[dict[str, Any]],
    value_threshold: float,
    phash_threshold: int,
    include_visual: bool,
) -> dict[str, Any]:
    canonical_counts = Counter(row["canonical_answer_hash"] for row in rows)
    pixel_hash_counts = (
        Counter(row["image_pixel_hash"] for row in rows) if include_visual else Counter()
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["template_name"]].append(row)

    value_pairs = 0
    value_affected: set[str] = set()
    value_components = 0
    nearest_value_similarities: list[float] = []
    phash_pairs = 0
    phash_affected: set[str] = set()
    nearest_phash_distances: list[int] = []

    for template_rows in grouped.values():
        size = len(template_rows)
        value_union = UnionFind(size)
        nearest_values = [0.0] * size
        nearest_phash = [65] * size
        for first in range(size):
            for second in range(first + 1, size):
                similarity = jaccard(
                    template_rows[first]["_value_set"], template_rows[second]["_value_set"]
                )
                nearest_values[first] = max(nearest_values[first], similarity)
                nearest_values[second] = max(nearest_values[second], similarity)
                if similarity >= value_threshold:
                    value_pairs += 1
                    value_affected.update(
                        [template_rows[first]["sample_id"], template_rows[second]["sample_id"]]
                    )
                    value_union.union(first, second)

                if include_visual:
                    distance = (
                        template_rows[first]["_perceptual_hash_int"]
                        ^ template_rows[second]["_perceptual_hash_int"]
                    ).bit_count()
                    nearest_phash[first] = min(nearest_phash[first], distance)
                    nearest_phash[second] = min(nearest_phash[second], distance)
                    if distance <= phash_threshold:
                        phash_pairs += 1
                        phash_affected.update(
                            [template_rows[first]["sample_id"], template_rows[second]["sample_id"]]
                        )
        value_components += value_union.component_count()
        nearest_value_similarities.extend(nearest_values)
        if include_visual:
            nearest_phash_distances.extend(distance for distance in nearest_phash if distance <= 64)

    duplicated_groups = [count for count in canonical_counts.values() if count > 1]
    duplicated_pixel_groups = [count for count in pixel_hash_counts.values() if count > 1]
    result: dict[str, Any] = {
        "instances": len(rows),
        "unique_canonical_answers": len(canonical_counts),
        "unique_canonical_answer_ratio": len(canonical_counts) / len(rows) if rows else 0.0,
        "exact_duplicate_groups": len(duplicated_groups),
        "exact_duplicate_instances_beyond_first": sum(count - 1 for count in duplicated_groups),
        "largest_exact_duplicate_group": max(duplicated_groups, default=1),
        "value_jaccard_threshold": value_threshold,
        "value_near_duplicate_pairs": value_pairs,
        "value_near_duplicate_instances": len(value_affected),
        "value_similarity_components": value_components,
        "value_similarity_effective_ratio": value_components / len(rows) if rows else 0.0,
        "median_nearest_value_jaccard": (
            statistics.median(nearest_value_similarities) if nearest_value_similarities else 0.0
        ),
    }
    if include_visual:
        result.update(
            {
                "unique_image_pixel_hashes": len(pixel_hash_counts),
                "unique_image_pixel_hash_ratio": len(pixel_hash_counts) / len(rows) if rows else 0.0,
                "exact_duplicate_image_groups": len(duplicated_pixel_groups),
                "exact_duplicate_images_beyond_first": sum(
                    count - 1 for count in duplicated_pixel_groups
                ),
                "phash_distance_threshold": phash_threshold,
                "phash_near_duplicate_pairs": phash_pairs,
                "phash_near_duplicate_instances": len(phash_affected),
                "median_nearest_phash_distance": (
                    statistics.median(nearest_phash_distances) if nearest_phash_distances else 0.0
                ),
            }
        )
    return result


def build_exact_duplicate_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_answer_hash"]].append(row)
    output = []
    for answer_hash, duplicate_rows in grouped.items():
        if len(duplicate_rows) < 2:
            continue
        output.append(
            {
                "canonical_answer_hash": answer_hash,
                "count": len(duplicate_rows),
                "templates": ";".join(sorted({row["template_name"] for row in duplicate_rows})),
                "sample_ids": ";".join(row["sample_id"] for row in duplicate_rows),
                "unique_image_pixel_hashes": len(
                    {row.get("image_pixel_hash") for row in duplicate_rows if row.get("image_pixel_hash")}
                ),
            }
        )
    return sorted(output, key=lambda row: (-int(row["count"]), str(row["canonical_answer_hash"])))


def build_exact_image_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_hash = str(row.get("image_pixel_hash") or "")
        if image_hash:
            grouped[image_hash].append(row)
    output = []
    for image_hash, duplicate_rows in grouped.items():
        if len(duplicate_rows) < 2:
            continue
        output.append(
            {
                "image_pixel_hash": image_hash,
                "count": len(duplicate_rows),
                "templates": ";".join(sorted({row["template_name"] for row in duplicate_rows})),
                "sample_ids": ";".join(row["sample_id"] for row in duplicate_rows),
                "unique_canonical_answers": len(
                    {row["canonical_answer_hash"] for row in duplicate_rows}
                ),
            }
        )
    return sorted(output, key=lambda row: (-int(row["count"]), str(row["image_pixel_hash"])))


def safe_correlation(first: pd.Series, second: pd.Series, method: str) -> float:
    mask = first.notna() & second.notna()
    x = first[mask].astype(float)
    y = second[mask].astype(float)
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def build_cross_modal_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows
        ]
    )
    content_keys = [
        "leaf_count",
        "total_characters",
        "mean_value_length",
        "max_value_length",
        "list_item_count",
        "tree_depth",
    ]
    visual_keys = [
        "blur_score_cv_denoised",
        "foreground_ratio",
        "mean_gray",
        "std_gray",
        "edge_density",
        "visual_entropy_bits",
    ]
    centered = frame[["template_name", *content_keys, *visual_keys]].copy()
    centered[[*content_keys, *visual_keys]] = centered.groupby("template_name")[
        [*content_keys, *visual_keys]
    ].transform(lambda column: column - column.mean())

    output = []
    for content_key in content_keys:
        for visual_key in visual_keys:
            output.append(
                {
                    "content_feature": content_key,
                    "visual_feature": visual_key,
                    "n": int((frame[content_key].notna() & frame[visual_key].notna()).sum()),
                    "pearson_global": safe_correlation(
                        frame[content_key], frame[visual_key], "pearson"
                    ),
                    "spearman_global": safe_correlation(
                        frame[content_key], frame[visual_key], "spearman"
                    ),
                    "pearson_within_template": safe_correlation(
                        centered[content_key], centered[visual_key], "pearson"
                    ),
                }
            )
    return output


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


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


def format_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    if abs(number) < 0.01:
        return f"{number:.4f}"
    return f"{number:.2f}"


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Metric | Min | P25 | Median | Mean | P75 | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {min} | {p25} | {median} | {mean} | {p75} | {p95} | {max} |".format(
                label=row["label"],
                **{key: format_number(row[key]) for key in ["min", "p25", "median", "mean", "p75", "p95", "max"]},
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_latex(
    path: Path, rows: list[dict[str, Any]], caption: str, label: str
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Metric & Min & P25 & Median & Mean & P75 & P95 & Max \\",
        r"\midrule",
    ]
    for row in rows:
        values = [format_number(row[key]) for key in ["min", "p25", "median", "mean", "p75", "p95", "max"]]
        lines.append(f"{latex_escape(row['label'])} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_tag_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    templates = sorted({row["template_name"] for row in rows})
    for key, label in TAG_LABELS.items():
        count = sum(bool(row[key]) for row in rows)
        varying_templates = sum(
            len({bool(row[key]) for row in rows if row["template_name"] == template}) > 1
            for template in templates
        )
        output.append(
            {
                "tag": key,
                "label": label,
                "instances": count,
                "pct": count / len(rows) * 100 if rows else 0.0,
                "templates": varying_templates,
            }
        )
    return output


def build_structure_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    list_rows = [row for row in rows if int(row.get("list_node_count", 0)) > 0]
    return [
        {
            "feature": "instances_with_repeated_lists",
            "label": "Instances with repeated lists",
            "instances": len(list_rows),
            "pct": len(list_rows) / len(rows) * 100 if rows else 0.0,
            "templates": len({row["template_name"] for row in list_rows}),
        }
    ]


def write_structure_coverage_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Coverage of instance-level repeated structures.}",
        r"\label{tab:instance_repeated_structure_coverage}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Feature & Instances & (\%) & Templates \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['label'])} & {row['instances']:,} & {row['pct']:.2f} & {row['templates']} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tag_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Distribution of instance-level semantic-structure and content-constraint tags.}",
        r"\label{tab:instance_constraint_distribution}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Tag & Instances & (\%) & Templates \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['label'])} & {row['instances']:,} & {row['pct']:.2f} & {row['templates']} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_duplicate_latex(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        ("Unique canonical answers", summary["unique_canonical_answers"]),
        ("Unique canonical answers (\\%)", summary["unique_canonical_answer_ratio"] * 100),
        ("Exact duplicate groups", summary["exact_duplicate_groups"]),
        ("Duplicate instances beyond first", summary["exact_duplicate_instances_beyond_first"]),
        ("Value-near-duplicate pairs", summary["value_near_duplicate_pairs"]),
        ("Value-similarity effective ratio (\\%)", summary["value_similarity_effective_ratio"] * 100),
    ]
    if "phash_near_duplicate_pairs" in summary:
        rows.extend(
            [
                ("Unique image pixel hashes", summary["unique_image_pixel_hashes"]),
                ("Exact duplicate images", summary["exact_duplicate_images_beyond_first"]),
                ("Median nearest full-page pHash distance", summary["median_nearest_phash_distance"]),
            ]
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Instance-level uniqueness and near-duplicate audit.}",
        r"\label{tab:instance_duplicate_audit}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
    ]
    for name, value in rows:
        display_name = name.replace(r"\%", "%")
        lines.append(f"{latex_escape(display_name)} & {format_number(value)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_latex(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        ("Instances", summary["valid_answer_instances"]),
        ("Templates", summary["templates"]),
        ("Answer parse errors", summary["answer_parse_errors"]),
        ("Missing or invalid images", summary["missing_or_invalid_images"]),
        ("Total leaf fields", summary["total_leaf_fields"]),
        ("Empty leaf fields", summary["empty_leaf_fields"]),
        ("Complete instances (\\%)", summary["complete_instance_ratio"] * 100),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Instance-level data quality statistics.}",
        r"\label{tab:instance_data_quality}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
    ]
    for name, value in rows:
        display_name = name.replace(r"\%", "%")
        lines.append(f"{latex_escape(display_name)} & {format_number(value)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_correlation_latex(path: Path, rows: list[dict[str, Any]], limit: int = 10) -> None:
    selected = sorted(
        (row for row in rows if not math.isnan(float(row["pearson_within_template"]))),
        key=lambda row: abs(float(row["pearson_within_template"])),
        reverse=True,
    )[:limit]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Strongest instance-level content--visual correlations.}",
        r"\label{tab:cross_modal_correlations}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Content feature & Visual feature & Pearson & Spearman & Within-template Pearson \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{latex_escape(FEATURE_LABELS.get(str(row['content_feature']), str(row['content_feature'])))} & "
            f"{latex_escape(FEATURE_LABELS.get(str(row['visual_feature']), str(row['visual_feature'])))} & "
            f"{float(row['pearson_global']):.3f} & {float(row['spearman_global']):.3f} & "
            f"{float(row['pearson_within_template']):.3f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_feature_distributions(output: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("leaf_count", "Leaf fields"),
        ("total_characters", "Total answer characters"),
        ("max_value_length", "Maximum value length"),
        ("tree_depth", "Answer-tree depth"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (key, label) in zip(axes.flat, metrics):
        axis.hist([row[key] for row in rows], bins=30, color="#276FBF", edgecolor="white")
        axis.set_title(label)
        axis.set_ylabel("Instances")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "instance_feature_distributions.png", dpi=200)
    figure.savefig(output / "instance_feature_distributions.pdf")
    plt.close(figure)


def plot_template_diversity(output: Path, rows: list[dict[str, Any]]) -> None:
    unique_ratios = [row["unique_answer_ratio"] for row in rows]
    value_diversity = [row["mean_field_unique_value_ratio"] for row in rows]
    schema_entropy = [row["schema_entropy_bits"] for row in rows]
    sizes = [25 + 4 * row["unique_schema_count"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(unique_ratios, bins=np.linspace(0.4, 1.0, 25), color="#C94C4C", edgecolor="white")
    axes[0].set_xlabel("Unique-answer ratio")
    axes[0].set_ylabel("Templates")
    axes[0].set_title("Full-answer uniqueness")
    axes[0].grid(axis="y", alpha=0.25)
    scatter = axes[1].scatter(
        value_diversity,
        schema_entropy,
        c=unique_ratios,
        s=sizes,
        alpha=0.78,
        cmap="viridis",
        vmin=0.4,
        vmax=1.0,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Mean field unique-value ratio")
    axes[1].set_ylabel("Schema entropy (bits)")
    axes[1].set_title("Value and schema diversity")
    axes[1].grid(alpha=0.25)
    colorbar = figure.colorbar(scatter, ax=axes[1], shrink=0.85)
    colorbar.set_label("Unique-answer ratio")
    figure.tight_layout()
    figure.savefig(output / "template_diversity.png", dpi=200)
    figure.savefig(output / "template_diversity.pdf")
    plt.close(figure)


def plot_visual_distributions(output: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("blur_score_cv_denoised", "Smoothed sharpness"),
        ("foreground_ratio", "Non-white-pixel ratio"),
        ("mean_gray", "Mean grayscale"),
        ("std_gray", "Grayscale contrast"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (key, label) in zip(axes.flat, metrics):
        axis.hist([row[key] for row in rows], bins=35, color="#278A72", edgecolor="white")
        axis.set_title(label)
        axis.set_ylabel("Instances")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "clean_visual_feature_distributions.png", dpi=200)
    figure.savefig(output / "clean_visual_feature_distributions.pdf")
    plt.close(figure)


def plot_correlation_heatmap(output: Path, rows: list[dict[str, Any]]) -> None:
    content_keys = list(dict.fromkeys(row["content_feature"] for row in rows))
    visual_keys = list(dict.fromkeys(row["visual_feature"] for row in rows))
    lookup = {
        (row["content_feature"], row["visual_feature"]): row["pearson_within_template"]
        for row in rows
    }
    matrix = np.asarray(
        [[lookup[(content, visual)] for visual in visual_keys] for content in content_keys],
        dtype=float,
    )
    color_limit = max(0.2, math.ceil(float(np.nanmax(np.abs(matrix))) * 10) / 10)
    figure, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-color_limit, vmax=color_limit, aspect="auto")
    axis.set_xticks(
        range(len(visual_keys)),
        [FEATURE_LABELS.get(key, key) for key in visual_keys],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(
        range(len(content_keys)),
        [FEATURE_LABELS.get(key, key) for key in content_keys],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Within-template content--visual correlations")
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(output / "cross_modal_correlation_heatmap.png", dpi=200)
    figure.savefig(output / "cross_modal_correlation_heatmap.pdf")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    started = time.time()
    data_root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    instances = discover_instances(data_root, args.limit)
    rows, answer_errors = load_answer_rows(instances)
    if not rows:
        raise SystemExit(f"no valid instances found under {data_root}")
    print(f"answer statistics: {len(rows)}/{len(instances)}", flush=True)

    visual_errors: list[dict[str, str]] = []
    include_visual = not args.skip_visual
    if include_visual:
        rows, visual_errors = add_visual_metrics(rows, args.workers)
        include_visual = all("blur_score_cv" in row for row in rows)
        if not include_visual:
            print("warning: incomplete visual metrics; visual aggregate outputs are skipped", flush=True)

    annotate_instance_tags(rows)
    template_rows = build_template_diversity(rows, include_visual)
    summary_metrics = [*INSTANCE_METRICS, *(VISUAL_METRICS if include_visual else [])]
    instance_summary = build_metric_summary(rows, summary_metrics)
    content_structure_summary = build_metric_summary(rows, INSTANCE_METRICS)
    template_summary_metrics = [
        *TEMPLATE_DIVERSITY_METRICS,
        *(TEMPLATE_VISUAL_DIVERSITY_METRICS if include_visual else []),
    ]
    template_summary = build_metric_summary(template_rows, template_summary_metrics)
    tag_rows = build_tag_distribution(rows)
    structure_coverage_rows = build_structure_coverage(rows)
    duplicate_summary = build_duplicate_summary(
        rows,
        value_threshold=args.value_jaccard_threshold,
        phash_threshold=args.phash_distance_threshold,
        include_visual=include_visual,
    )
    duplicate_clusters = build_exact_duplicate_clusters(rows)
    image_duplicate_clusters = build_exact_image_clusters(rows) if include_visual else []
    correlations = build_cross_modal_correlations(rows) if include_visual else []

    quality_summary = {
        "discovered_instances": len(instances),
        "valid_answer_instances": len(rows),
        "answer_parse_errors": len(answer_errors),
        "missing_or_invalid_images": len(visual_errors),
        "templates": len({row["template_name"] for row in rows}),
        "total_leaf_fields": sum(row["leaf_count"] for row in rows),
        "empty_leaf_fields": sum(row["empty_leaf_count"] for row in rows),
        "empty_leaf_field_ratio": (
            sum(row["empty_leaf_count"] for row in rows)
            / max(sum(row["leaf_count"] for row in rows), 1)
        ),
        "complete_instances": sum(row["empty_leaf_count"] == 0 for row in rows),
        "complete_instance_ratio": sum(row["empty_leaf_count"] == 0 for row in rows) / len(rows),
    }
    metadata = {
        "data_root": str(data_root),
        "instances": len(rows),
        "templates": quality_summary["templates"],
        "visual_metrics_included": include_visual,
        "tag_definitions": TAG_DEFINITIONS,
        "structural_metric_definitions": {
            "leaf_count": "number of scalar leaves in the instance answer tree",
            "internal_semantic_group_count": "number of non-root JSON object nodes",
            "semantic_tree_edge_count": "parent-child edges among JSON containers and scalar leaves",
            "mean_leaf_depth": "mean root-to-leaf depth over scalar answer fields",
            "tree_depth": "maximum root-to-node depth",
            "active_answer_path_count": "number of unique scalar answer paths activated by the instance",
        },
        "visual_metric_definitions": {
            "blur_score_cv": "variance of the grayscale Laplacian",
            "blur_score_cv_denoised": "variance of the Laplacian after 3x3 Gaussian smoothing",
            "foreground_ratio": "fraction of grayscale pixels below 245",
            "mean_gray": "mean grayscale intensity",
            "std_gray": "standard deviation of grayscale intensity",
            "edge_density": "fraction of Canny edge pixels at thresholds 100 and 200",
            "visual_entropy_bits": "Shannon entropy of the 256-bin grayscale histogram",
            "colorfulness": "Hasler--Suesstrunk colorfulness estimate",
        },
        "duplicate_definitions": {
            "canonical_answer": "SHA-256 of normalized-key-order compact JSON",
            "exact_image": "SHA-256 of decoded BGR pixel bytes",
            "value_near_duplicate": f"within-template scalar-value-set Jaccard >= {args.value_jaccard_threshold}",
            "image_near_duplicate": (
                f"within-template 64-bit perceptual-hash Hamming distance <= {args.phash_distance_threshold}"
                if include_visual
                else None
            ),
            "image_near_duplicate_caveat": (
                "full-page perceptual hashes are dominated by shared template layout and are reported as a visual-similarity diagnostic, not as proof of duplicate content"
                if include_visual
                else None
            ),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }

    write_csv(output / "instance_level_sample_stats.csv", public_rows(rows))
    write_csv(output / "instance_level_summary.csv", instance_summary)
    write_json(output / "instance_level_summary.json", instance_summary)
    write_summary_markdown(output / "instance_level_summary.md", instance_summary)
    instance_caption = "Instance-level semantic-structure and content statistics."
    if include_visual:
        instance_caption = "Instance-level semantic-structure, content, and visual statistics."
    write_summary_latex(
        output / "instance_level_summary_table.tex",
        instance_summary,
        instance_caption,
        "tab:instance_level_statistics",
    )
    write_summary_latex(
        output / "instance_content_structure_summary_table.tex",
        content_structure_summary,
        "Instance-level semantic-structure and content statistics.",
        "tab:instance_content_structure_statistics",
    )
    write_csv(
        output / "instance_content_structure_summary.csv", content_structure_summary
    )
    if include_visual:
        write_summary_latex(
            output / "clean_visual_summary_table.tex",
            build_metric_summary(rows, VISUAL_METRICS),
            "Instance-level clean-image visual statistics.",
            "tab:clean_visual_statistics",
        )

    write_csv(output / "template_diversity_stats.csv", template_rows)
    write_csv(output / "template_diversity_summary.csv", template_summary)
    write_json(output / "template_diversity_summary.json", template_summary)
    write_summary_latex(
        output / "template_diversity_summary_table.tex",
        template_summary,
        "Within-template instance diversity statistics.",
        "tab:template_diversity_statistics",
    )

    write_csv(output / "instance_constraint_tag_distribution.csv", tag_rows)
    write_tag_latex(output / "instance_constraint_tag_distribution_table.tex", tag_rows)
    write_csv(output / "instance_structure_coverage.csv", structure_coverage_rows)
    write_structure_coverage_latex(
        output / "instance_structure_coverage_table.tex", structure_coverage_rows
    )
    write_json(output / "duplicate_summary.json", duplicate_summary)
    write_duplicate_latex(output / "duplicate_summary_table.tex", duplicate_summary)
    write_csv(output / "exact_duplicate_answer_clusters.csv", duplicate_clusters)
    if include_visual:
        write_csv(output / "exact_duplicate_image_clusters.csv", image_duplicate_clusters)
    write_json(output / "data_quality_summary.json", quality_summary)
    write_quality_latex(output / "data_quality_summary_table.tex", quality_summary)
    write_json(output / "instance_stats_metadata.json", metadata)
    write_json(output / "errors.json", {"answer_errors": answer_errors, "visual_errors": visual_errors})

    if correlations:
        write_csv(output / "cross_modal_correlations.csv", correlations)
        write_correlation_latex(output / "cross_modal_correlations_table.tex", correlations)

    plot_feature_distributions(output, rows)
    plot_template_diversity(output, template_rows)
    if include_visual:
        plot_visual_distributions(output, rows)
        plot_correlation_heatmap(output, correlations)

    print("output files:")
    for path in sorted(output.iterdir()):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
