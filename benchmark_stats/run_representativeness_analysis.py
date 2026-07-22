#!/usr/bin/env python3
"""Analyze FormStruct-Bench template representativeness against SRFUND."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SEED = 42
BOOTSTRAP_ROUNDS = 1000
EPSILON = 1e-12
SRFUND_LANGUAGES = ("de", "en", "es", "fr", "it", "ja", "pt", "zh")
ENTITY_LABELS = ("header", "question", "answer", "other")

LANGUAGE_INFO: dict[str, tuple[str, str, str]] = {
    "ar": ("Arabic", "Arabic", "RTL"),
    "de": ("German", "Latin", "LTR"),
    "en": ("English", "Latin", "LTR"),
    "es": ("Spanish", "Latin", "LTR"),
    "fr": ("French", "Latin", "LTR"),
    "it": ("Italian", "Latin", "LTR"),
    "ja": ("Japanese", "Han + Hiragana + Katakana", "LTR"),
    "pt": ("Portuguese", "Latin", "LTR"),
    "zh": ("Chinese", "Han", "LTR"),
    "zh-en": ("Chinese--English", "Han + Latin", "LTR"),
}

LANGUAGE_ALIASES = {
    "arabic": "ar",
    "german": "de",
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "japanese": "ja",
    "japan": "ja",
    "portuguese": "pt",
    "chinese": "zh",
    "chinese+english": "zh-en",
    "chinese--english": "zh-en",
    "zh": "zh",
    "zn": "zh",
}

JS_FEATURES = {
    "language": "direct",
    "script": "derived_conditional",
    "writing_direction": "conditional",
    "table_presence": "conditional",
    "hierarchy_depth_bin": "direct",
    "entity_type_distribution": "conditional",
}

WASSERSTEIN_FEATURES = {
    "num_entities": "conditional",
    "num_item_tables": "conditional",
    "num_relation_links": "direct",
    "max_hierarchy_depth": "direct",
    "mean_branching_factor": "direct",
    "num_root_children": "direct",
    "relation_density": "direct",
    "spatial_layout_density": "conditional",
    "bbox_area_mean": "direct",
}

GOWER_GROUPS: dict[str, dict[str, list[str]]] = {
    "hierarchy": {
        "continuous": ["max_hierarchy_depth", "mean_branching_factor", "num_root_children"],
        "categorical": [],
    },
    "entity_composition": {
        "continuous": [
            "num_entities",
            "header_entity_ratio",
            "question_entity_ratio",
            "answer_entity_ratio",
            "other_entity_ratio",
        ],
        "categorical": [],
    },
    "table_structure": {
        "continuous": ["num_item_tables"],
        "categorical": ["table_presence"],
    },
    "relation_structure": {
        "continuous": ["num_relation_links", "relation_density"],
        "categorical": [],
    },
    "spatial_layout": {
        "continuous": [
            "bbox_area_mean",
            "bbox_width_mean",
            "bbox_height_mean",
            "bbox_center_x_mean",
            "bbox_center_y_mean",
            "bbox_center_x_std",
            "bbox_center_y_std",
            "spatial_layout_density",
        ],
        "categorical": [],
    },
    "language_script": {
        "continuous": [],
        "categorical": ["language_code", "script", "writing_direction"],
    },
}

# Sensitivity configuration using only fields whose units/semantics are
# shared without an ontology conversion.  The mapped configuration above is
# retained as the primary coverage view because table and entity composition
# are useful structural signals, but this view exposes how much the result
# depends on those conditional mappings.
DIRECT_GOWER_GROUPS: dict[str, dict[str, list[str]]] = {
    "hierarchy": {
        "continuous": ["max_hierarchy_depth", "mean_branching_factor", "num_root_children"],
        "categorical": [],
    },
    "relation_structure": {
        "continuous": ["num_relation_links", "relation_density"],
        "categorical": [],
    },
    "spatial_layout": {
        "continuous": [
            "bbox_area_mean",
            "bbox_width_mean",
            "bbox_height_mean",
            "bbox_center_x_mean",
            "bbox_center_y_mean",
            "bbox_center_x_std",
            "bbox_center_y_std",
        ],
        "categorical": [],
    },
    "language": {
        "continuous": [],
        "categorical": ["language_code"],
    },
}

REQUIRED_OUTPUTS = (
    "dedup_candidates.csv",
    "dedup_excluded.csv",
    "srfund_layout_clusters.csv",
    "dedup_report.md",
    "feature_mapping.md",
    "formstruct_template_features.csv",
    "srfund_reference_features.csv",
    "js_results.csv",
    "wasserstein_results.csv",
    "coverage_results.csv",
    "nearest_template_matches.csv",
    "calibration_bootstrap.csv",
    "calibration_summary.csv",
    "representativeness_distance_heatmap.pdf",
    "continuous_feature_ecdf.pdf",
    "coverage_by_slice.pdf",
    "nearest_neighbor_gap.pdf",
    "representativeness_table.tex",
    "representativeness_summary.md",
)


@dataclass
class Unit:
    unit_id: str
    dataset: str
    source_id: str
    language_code: str
    split: str
    image_path: Path
    annotation_path: Path
    features: dict[str, Any]
    boxes: list[tuple[float, float, float, float]]
    text: str
    source_document_id: str = ""
    template_id: str = ""
    file_hash: str = ""
    normalized_image_hash: str = ""
    phash_variants: tuple[int, ...] = ()
    occupancy_variants: tuple[np.ndarray, ...] = ()
    text_shingles: frozenset[str] = frozenset()
    cluster_id: str = ""
    cluster_size: int = 1
    cluster_members: tuple[str, ...] = ()
    dedup_status: str = "included"
    dedup_reason: str = ""


@dataclass(frozen=True)
class PairMetrics:
    phash_distance: int
    layout_cosine: float
    ocr_jaccard: float
    entity_count_ratio: float
    aspect_ratio_difference: float
    rotation_quadrants: int


@dataclass
class GowerConfig:
    groups: dict[str, dict[str, list[str]]]
    ranges: dict[str, float]
    active_groups: list[str]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FormStruct-Bench templates with deduplicated SRFUND layouts."
    )
    parser.add_argument("--srfund-root", default="raw/srfund/extracted/dataset")
    parser.add_argument("--srfund-archive", default="raw/srfund/srfund_download.bin")
    parser.add_argument("--layout-dir", default="newdataset-layout")
    parser.add_argument("--formstruct-image-dir", default="new-dataset")
    parser.add_argument("--template-root", default="FormTSR/datasets")
    parser.add_argument("--source-metadata-dir", default="metadata-test")
    parser.add_argument(
        "--split-assignments",
        default="outputs/dataset_splits/template_stratified_seed42/template_assignments.csv",
    )
    parser.add_argument("--output", default="outputs/representativeness")
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-render-validation", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zip_archive(path: Path) -> None:
    """Read every member CRC so a rerun cannot silently use a damaged archive."""
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"SRFUND archive is not a readable ZIP: {path}") from exc
    if bad_member is not None:
        raise ValueError(f"SRFUND archive CRC check failed for member: {bad_member}")


def normalize_language(raw: Any, template_name: str = "") -> str:
    value = str(raw or "").strip().lower()
    if value in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[value]
    if template_name.startswith("Arabic-"):
        return "ar"
    prefix = template_name.split("_", 1)[0].lower()
    if prefix == "zn" and template_name.startswith("zn_en_"):
        return "zh-en"
    return {"de": "de", "en": "en", "es": "es", "ja": "ja", "pt": "pt", "zn": "zh"}.get(
        prefix, value
    )


def language_metadata(code: str) -> tuple[str, str, str]:
    if code not in LANGUAGE_INFO:
        raise ValueError(f"unsupported language code: {code!r}")
    return LANGUAGE_INFO[code]


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_bbox(
    value: Any, width: float, height: float
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or width <= 0 or height <= 0:
        return None
    if len(value) > 4 and len(value) % 2 == 0:
        raw = [finite_number(item) for item in value]
        if any(item is None for item in raw):
            return None
        xs = [float(item) for item in raw[0::2]]
        ys = [float(item) for item in raw[1::2]]
        value = [min(xs), min(ys), max(xs), max(ys)]
    if len(value) != 4:
        return None
    coords = [finite_number(item) for item in value]
    if any(item is None for item in coords):
        return None
    x1, y1, x2, y2 = (float(item) for item in coords)
    if x2 <= x1 or y2 <= y1:
        return None
    clipped = (
        min(1.0, max(0.0, x1 / width)),
        min(1.0, max(0.0, y1 / height)),
        min(1.0, max(0.0, x2 / width)),
        min(1.0, max(0.0, y2 / height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def rasterize_boxes(
    boxes: Sequence[tuple[float, float, float, float]], size: int = 24
) -> np.ndarray:
    raster = np.zeros((size, size), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        left = max(0, min(size - 1, int(math.floor(x1 * size))))
        top = max(0, min(size - 1, int(math.floor(y1 * size))))
        right = max(left + 1, min(size, int(math.ceil(x2 * size))))
        bottom = max(top + 1, min(size, int(math.ceil(y2 * size))))
        raster[top:bottom, left:right] = 1.0
    return raster


def geometry_features(
    boxes: Sequence[tuple[float, float, float, float]]
) -> dict[str, float]:
    if not boxes:
        return {
            key: 0.0
            for key in (
                "bbox_area_mean",
                "bbox_area_median",
                "bbox_area_std",
                "bbox_width_mean",
                "bbox_height_mean",
                "bbox_center_x_mean",
                "bbox_center_y_mean",
                "bbox_center_x_std",
                "bbox_center_y_std",
                "spatial_layout_density",
            )
        }
    array = np.asarray(boxes, dtype=float)
    widths = array[:, 2] - array[:, 0]
    heights = array[:, 3] - array[:, 1]
    areas = widths * heights
    center_x = (array[:, 0] + array[:, 2]) / 2.0
    center_y = (array[:, 1] + array[:, 3]) / 2.0
    return {
        "bbox_area_mean": float(np.mean(areas)),
        "bbox_area_median": float(np.median(areas)),
        "bbox_area_std": float(np.std(areas)),
        "bbox_width_mean": float(np.mean(widths)),
        "bbox_height_mean": float(np.mean(heights)),
        "bbox_center_x_mean": float(np.mean(center_x)),
        "bbox_center_y_mean": float(np.mean(center_y)),
        "bbox_center_x_std": float(np.std(center_x)),
        "bbox_center_y_std": float(np.std(center_y)),
        "spatial_layout_density": float(np.mean(rasterize_boxes(boxes, 64))),
    }


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return "".join(char for char in normalized if char.isalnum() or char == "#")


def character_shingles(text: str, size: int = 3) -> frozenset[str]:
    normalized = normalize_text(text)
    if not normalized:
        return frozenset()
    if len(normalized) <= size:
        return frozenset({normalized})
    return frozenset(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def perceptual_hash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8]
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = (low_frequency > median).flatten()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def analyze_image(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    variants = tuple(perceptual_hash(np.rot90(gray, turns)) for turns in range(4))
    normalized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    return {
        "image_width": width,
        "image_height": height,
        "image_aspect_ratio": width / height,
        "image_file_size_bytes": path.stat().st_size,
        "file_hash": sha256_file(path),
        "normalized_image_hash": hashlib.sha256(normalized.tobytes()).hexdigest(),
        "phash_variants": variants,
    }


def attach_image_metrics(units: list[Unit], workers: int) -> None:
    worker_count = max(1, workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(analyze_image, [unit.image_path for unit in units]))
    for unit, metrics in zip(units, results, strict=True):
        unit.file_hash = str(metrics.pop("file_hash"))
        unit.normalized_image_hash = str(metrics.pop("normalized_image_hash"))
        unit.phash_variants = tuple(metrics.pop("phash_variants"))
        unit.features.update(metrics)
        base = rasterize_boxes(unit.boxes)
        variants: list[np.ndarray] = []
        for turns in range(4):
            raster = np.rot90(base, turns).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(raster))
            variants.append(raster / norm if norm else raster)
        unit.occupancy_variants = tuple(variants)
        unit.text_shingles = character_shingles(unit.text)


def graph_statistics(
    nodes: set[int], edges: set[tuple[int, int]], roots: set[int], depths: dict[int, int]
) -> dict[str, float | int]:
    outdegree = Counter(parent for parent, _ in edges)
    branch_values = list(outdegree.values())
    return {
        "num_relation_links": len(edges),
        "max_hierarchy_depth": max(depths.values(), default=0),
        "mean_branching_factor": float(np.mean(branch_values)) if branch_values else 0.0,
        "num_root_children": len(roots),
        "relation_density": len(edges) / max(len(nodes), 1),
    }


def parse_srfund_relation_tree(tree: Any) -> dict[str, float | int]:
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    roots: set[int] = set()
    depths: dict[int, int] = {}

    def add_node(node: int, parent: int | None, depth: int) -> None:
        nodes.add(node)
        depths[node] = max(depths.get(node, 0), depth)
        if parent is None:
            roots.add(node)
        elif parent != node:
            edges.add((parent, node))

    def visit(value: Any, parent: int | None, parent_depth: int) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            add_node(value, parent, parent_depth + 1)
            return
        if isinstance(value, list):
            for item in value:
                visit(item, parent, parent_depth)
            return
        if not isinstance(value, dict):
            return
        for raw_key, child_value in value.items():
            try:
                node = int(raw_key)
            except (TypeError, ValueError):
                visit(child_value, parent, parent_depth)
                continue
            depth = parent_depth + 1
            add_node(node, parent, depth)
            visit(child_value, node, depth)

    visit(tree, None, 0)
    return {"relation_graph_nodes": len(nodes), **graph_statistics(nodes, edges, roots, depths)}


def add_entity_ratios(features: dict[str, Any]) -> None:
    total = max(int(features.get("num_entities") or 0), 1)
    for label in ENTITY_LABELS:
        count = int(features.get(f"num_{label}_entities") or 0)
        features[f"{label}_entity_ratio"] = count / total


def hierarchy_bin(value: Any) -> str:
    depth = int(finite_number(value) or 0)
    if depth <= 2:
        return "shallow_0_2"
    if depth <= 4:
        return "medium_3_4"
    return "deep_5_plus"


def infer_srfund_split(image_name: str) -> str:
    if "_train_" in image_name:
        return "train"
    if "_val_" in image_name:
        return "validation"
    return "unavailable"


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def find_srfund_image(root: Path, language: str, image_name: str) -> Path:
    path = root / "images" / language / image_name
    if not path.is_file():
        raise FileNotFoundError(f"SRFUND image missing for annotation key: {path}")
    return path


def load_srfund_units(root: Path) -> tuple[list[Unit], dict[str, Any]]:
    item_tables = read_json(root / "item_table_info.json")
    table_lookup: dict[str, list[dict[str, Any]]] = {}
    for raw_name, records in item_tables.items():
        table_lookup[str(raw_name)] = records
        table_lookup[Path(str(raw_name)).stem] = records

    units: list[Unit] = []
    inventory: dict[str, Any] = {"languages": {}, "annotation_schema": {}}
    for language in SRFUND_LANGUAGES:
        instance_path = root / "instance_annotation" / f"{language}.json"
        relation_path = root / "relation_annotation" / f"{language}.json"
        instances = read_json(instance_path)
        relations = read_json(relation_path)
        if set(instances) != set(relations):
            raise ValueError(f"instance/relation keys differ for SRFUND language {language}")
        image_keys = {path.name for path in (root / "images" / language).iterdir() if path.is_file()}
        if set(instances) != image_keys:
            raise ValueError(f"image/annotation keys differ for SRFUND language {language}")

        split_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for image_name, entities in instances.items():
            image_path = find_srfund_image(root, language, image_name)
            width, height = image_dimensions(image_path)
            boxes: list[tuple[float, float, float, float]] = []
            word_boxes: list[tuple[float, float, float, float]] = []
            entity_counts = Counter()
            num_lines = 0
            num_words = 0
            linking_edges: set[tuple[int, int]] = set()
            texts: list[str] = []
            for entity in entities:
                label = str(entity.get("label") or "other").lower()
                if label not in ENTITY_LABELS:
                    label = "other"
                entity_counts[label] += 1
                label_counts[label] += 1
                box = normalize_bbox(entity.get("box"), width, height)
                if box is not None:
                    boxes.append(box)
                text = str(entity.get("text") or "")
                if text:
                    texts.append(text)
                lines = entity.get("lines") if isinstance(entity.get("lines"), list) else []
                num_lines += len(lines)
                for line in lines:
                    words = line.get("words") if isinstance(line, dict) and isinstance(line.get("words"), list) else []
                    num_words += len(words)
                    for word in words:
                        if isinstance(word, dict):
                            word_box = normalize_bbox(word.get("box"), width, height)
                            if word_box is not None:
                                word_boxes.append(word_box)
                linking = entity.get("linking") if isinstance(entity.get("linking"), list) else []
                for pair in linking:
                    if isinstance(pair, list) and len(pair) == 2:
                        try:
                            linking_edges.add((int(pair[0]), int(pair[1])))
                        except (TypeError, ValueError):
                            pass

            source_id = Path(image_name).stem
            table_records = table_lookup.get(image_name, table_lookup.get(source_id, []))
            num_table_items = sum(
                len(record.get("item_value_ids", {}))
                for record in table_records
                if isinstance(record, dict) and isinstance(record.get("item_value_ids"), dict)
            )
            relation_stats = parse_srfund_relation_tree(relations[image_name])
            language_name, script, direction = language_metadata(language)
            split = infer_srfund_split(image_name)
            split_counts[split] += 1
            features: dict[str, Any] = {
                "language": language_name,
                "language_code": language,
                "script": script,
                "writing_direction": direction,
                "num_words": num_words,
                "num_text_lines": num_lines,
                "num_entities": len(entities),
                "num_header_entities": entity_counts["header"],
                "num_question_entities": entity_counts["question"],
                "num_answer_entities": entity_counts["answer"],
                "num_other_entities": entity_counts["other"],
                "num_item_tables": len(table_records),
                "num_table_items": num_table_items,
                "raw_linking_edge_count": len(linking_edges),
                "table_presence": bool(table_records),
                "page_text_density": float(np.mean(rasterize_boxes(word_boxes, 64))),
                "source_width": width,
                "source_height": height,
                **relation_stats,
                **geometry_features(boxes),
            }
            features["hierarchy_depth_bin"] = hierarchy_bin(features["max_hierarchy_depth"])
            add_entity_ratios(features)
            units.append(
                Unit(
                    unit_id=f"srfund::{language}::{image_name}",
                    dataset="srfund",
                    source_id=source_id,
                    source_document_id="",
                    language_code=language,
                    split=split,
                    image_path=image_path,
                    annotation_path=instance_path,
                    features=features,
                    boxes=boxes,
                    text="\n".join(texts),
                )
            )
        inventory["languages"][language] = {
            "images": len(image_keys),
            "instance_annotations": len(instances),
            "relation_annotations": len(relations),
            "splits": dict(sorted(split_counts.items())),
            "entity_labels": dict(sorted(label_counts.items())),
        }

    inventory["annotation_schema"] = {
        "instance_annotation": "filename -> list of entities with id, box, text, lines/words, label, linking",
        "relation_annotation": "filename -> heterogeneous nested dict/list hierarchy of entity IDs",
        "item_table_info": "filename -> list of tables with table box, entity IDs, item IDs, item boxes",
        "document_metadata": "not present",
        "domain_labels": "not present",
    }
    inventory["total_images"] = len(units)
    inventory["item_table_pages"] = len(item_tables)
    inventory["item_tables"] = sum(len(records) for records in item_tables.values())
    return units, inventory


def formstruct_value_objects(node: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(node.get("value"), dict):
        output.append(node["value"])
    if isinstance(node.get("values"), list):
        output.extend(item for item in node["values"] if isinstance(item, dict))
    return output


def extract_formstruct_graph(
    fields: Any, width: float, height: float
) -> tuple[dict[str, Any], list[tuple[float, float, float, float]], str]:
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    roots: set[int] = set()
    depths: dict[int, int] = {}
    boxes: list[tuple[float, float, float, float]] = []
    labels = Counter()
    texts: list[str] = []
    next_id = 0

    def allocate() -> int:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        nodes.add(node_id)
        return node_id

    def visit(node: dict[str, Any], parent: int | None, depth: int) -> None:
        node_id = allocate()
        depths[node_id] = depth
        if parent is None:
            roots.add(node_id)
        else:
            edges.add((parent, node_id))
        children = [item for item in node.get("keys", []) if isinstance(item, dict)] if isinstance(node.get("keys"), list) else []
        values = formstruct_value_objects(node)
        labels["header" if children and not values else "question"] += 1
        box = normalize_bbox(node.get("bbox"), width, height)
        if box is not None:
            boxes.append(box)
        label_text = str(node.get("original_label") or node.get("semantic_key") or "")
        if label_text:
            texts.append(label_text)
        for value in values:
            value_id = allocate()
            depths[value_id] = depth + 1
            edges.add((node_id, value_id))
            labels["answer"] += 1
            value_box = normalize_bbox(value.get("bbox"), width, height)
            if value_box is not None:
                boxes.append(value_box)
        for child in children:
            visit(child, node_id, depth + 1)

    if isinstance(fields, list):
        for item in fields:
            if isinstance(item, dict):
                visit(item, None, 1)
    features: dict[str, Any] = {
        "num_entities": len(nodes),
        "num_header_entities": labels["header"],
        "num_question_entities": labels["question"],
        "num_answer_entities": labels["answer"],
        "num_other_entities": 0,
        **graph_statistics(nodes, edges, roots, depths),
        **geometry_features(boxes),
    }
    add_entity_ratios(features)
    return features, boxes, "\n".join(texts)


def count_formstruct_data_types(fields: Any) -> Counter[str]:
    counts: Counter[str] = Counter()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            data_type = value.get("data_type")
            if data_type:
                counts[str(data_type).strip().lower()] += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(fields)
    return counts


def load_source_metadata(source_dir: Path) -> tuple[dict[int, str], dict[int, Path]]:
    names: dict[int, str] = {}
    paths: dict[int, Path] = {}
    for path in sorted(source_dir.glob("*.json")):
        payload = read_json(path)
        source_id = payload.get("id") if isinstance(payload, dict) else None
        if isinstance(source_id, int):
            names[source_id] = path.stem
            paths[source_id] = path
    return names, paths


def load_split_assignments(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["template_name"]: row["split"] for row in csv.DictReader(handle)}


def load_formstruct_units(
    layout_dir: Path,
    image_dir: Path,
    template_root: Path,
    source_metadata_dir: Path,
    split_path: Path,
) -> tuple[list[Unit], dict[str, Any]]:
    selected = sorted(path.name for path in template_root.iterdir() if path.is_dir())
    if len(selected) != 70:
        raise ValueError(f"expected 70 selected FormStruct templates, found {len(selected)}")
    splits = load_split_assignments(split_path)
    source_names, source_paths = load_source_metadata(source_metadata_dir)
    units: list[Unit] = []
    missing_source_metadata: list[str] = []
    for template_id in selected:
        layout_path = layout_dir / f"{template_id}.json"
        image_path = image_dir / f"{template_id}.jpg"
        if not layout_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"missing FormStruct source layout/image for {template_id}")
        payload = read_json(layout_path)
        width = float(payload.get("original_width") or 0)
        height = float(payload.get("original_height") or 0)
        if width <= 0 or height <= 0:
            width, height = image_dimensions(image_path)
        graph_features, boxes, text = extract_formstruct_graph(payload.get("fields"), width, height)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        structural = metadata.get("S") if isinstance(metadata.get("S"), dict) else {}
        visual = metadata.get("V") if isinstance(metadata.get("V"), dict) else {}
        layout = metadata.get("layout_structure") if isinstance(metadata.get("layout_structure"), dict) else {}
        language_code = normalize_language(metadata.get("language"), template_id)
        language_name, script, direction = language_metadata(language_code)
        table_regions = int(layout.get("table_region_count") or 0)
        data_types = count_formstruct_data_types(payload.get("fields"))
        record_id = payload.get("id")
        source_id = source_names.get(record_id, "") if isinstance(record_id, int) else ""
        source_annotation = source_paths.get(record_id, layout_path) if isinstance(record_id, int) else layout_path
        if not source_id:
            missing_source_metadata.append(template_id)
            source_id = Path(str(payload.get("img") or template_id)).stem
        features: dict[str, Any] = {
            "language": language_name,
            "language_code": language_code,
            "script": script,
            "writing_direction": direction,
            "num_words": np.nan,
            "num_text_lines": np.nan,
            "page_text_density": np.nan,
            "num_item_tables": table_regions,
            "num_table_items": np.nan,
            "table_presence": bool(table_regions),
            "raw_linking_edge_count": np.nan,
            "source_width": width,
            "source_height": height,
            "semantic_section_count": int(layout.get("section_count") or 0),
            "semantic_region_count": int(layout.get("region_count") or 0),
            "table_region_count": table_regions,
            "line_item_group_count": int(layout.get("line_item_group_count") or 0),
            "cell_count": int(structural.get("cell_count") or 0),
            "merged_cell_count": int(structural.get("merged_cell_count") or 0),
            "selection_control_count": int(structural.get("selection_control_count") or 0),
            "option_group_count": int(structural.get("option_group_count") or 0),
            "signature_field_count": sum(count for name, count in data_types.items() if "signature" in name),
            "checkbox_field_count": sum(count for name, count in data_types.items() if "checkbox" in name or "check_box" in name),
            "radio_field_count": sum(count for name, count in data_types.items() if "radio" in name),
            "character_box_count": sum(count for name, count in data_types.items() if "character" in name),
            "weak_or_borderless_grid": bool(visual.get("borderless") or visual.get("weak_grid")),
            "annotated_label_token_count": len(re.findall(r"\w+", text, flags=re.UNICODE)),
            **graph_features,
        }
        features["hierarchy_depth_bin"] = hierarchy_bin(features["max_hierarchy_depth"])
        units.append(
            Unit(
                unit_id=f"formstruct::{template_id}",
                dataset="formstruct",
                source_id=source_id,
                source_document_id=source_id,
                template_id=template_id,
                language_code=language_code,
                split=splits.get(template_id, "unassigned"),
                image_path=image_path,
                annotation_path=source_annotation,
                features=features,
                boxes=boxes,
                text=text,
                dedup_status="benchmark_template",
            )
        )
    split_counts = Counter(unit.split for unit in units)
    inventory = {
        "selected_templates": len(units),
        "instances": sum(
            1
            for template in selected
            for path in (template_root / template).iterdir()
            if path.is_dir()
        ),
        "templates_by_split": dict(sorted(split_counts.items())),
        "missing_source_metadata": missing_source_metadata,
        "source_metadata_records": len(source_names),
    }
    return units, inventory


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if not np.any(first) and not np.any(second):
        return 1.0
    if not np.any(first) or not np.any(second):
        return 0.0
    return float(np.clip(np.dot(first, second), 0.0, 1.0))


def unit_pair_metrics(first: Unit, second: Unit) -> PairMetrics:
    best_phash = 65
    best_rotation = 0
    for first_turns, first_hash in enumerate(first.phash_variants):
        for second_turns, second_hash in enumerate(second.phash_variants):
            distance = (first_hash ^ second_hash).bit_count()
            if distance < best_phash:
                best_phash = distance
                best_rotation = (second_turns - first_turns) % 4
    layout_scores = [
        cosine_similarity(first.occupancy_variants[0], second.occupancy_variants[turns])
        for turns in range(4)
    ]
    layout_cosine = max(layout_scores, default=0.0)
    layout_rotation = int(np.argmax(layout_scores)) if layout_scores else 0
    union = first.text_shingles | second.text_shingles
    text_similarity = len(first.text_shingles & second.text_shingles) / len(union) if union else 1.0
    first_count = int(first.features.get("num_entities") or 0)
    second_count = int(second.features.get("num_entities") or 0)
    count_ratio = min(first_count, second_count) / max(first_count, second_count, 1)
    first_aspect = float(first.features.get("image_aspect_ratio") or 0.0)
    second_aspect = float(second.features.get("image_aspect_ratio") or 0.0)
    direct_aspect = abs(first_aspect - second_aspect) / max(first_aspect, second_aspect, EPSILON)
    rotated_second = 1.0 / second_aspect if second_aspect else second_aspect
    rotated_aspect = abs(first_aspect - rotated_second) / max(first_aspect, rotated_second, EPSILON)
    return PairMetrics(
        phash_distance=best_phash,
        layout_cosine=layout_cosine,
        ocr_jaccard=text_similarity,
        entity_count_ratio=count_ratio,
        aspect_ratio_difference=min(direct_aspect, rotated_aspect),
        rotation_quadrants=layout_rotation if layout_cosine >= 0.5 else best_rotation,
    )


def internal_duplicate_reason(first: Unit, second: Unit, metrics: PairMetrics) -> str:
    if first.file_hash == second.file_hash:
        return "exact_file_sha256"
    if first.normalized_image_hash == second.normalized_image_hash:
        return "exact_normalized_image_sha256"
    if metrics.phash_distance <= 4 and metrics.layout_cosine >= 0.90:
        return "phash_le_4_and_layout_ge_0.90"
    if (
        metrics.phash_distance <= 8
        and metrics.layout_cosine >= 0.95
        and metrics.ocr_jaccard >= 0.45
    ):
        return "phash_le_8_layout_ge_0.95_text_ge_0.45"
    if (
        metrics.layout_cosine >= 0.985
        and metrics.ocr_jaccard >= 0.82
        and metrics.entity_count_ratio >= 0.75
    ):
        return "layout_ge_0.985_text_ge_0.82_count_ratio_ge_0.75"
    return ""


def relaxed_duplicate_edge(metrics: PairMetrics) -> bool:
    return (
        (metrics.phash_distance <= 10 and metrics.layout_cosine >= 0.88)
        or (
            metrics.phash_distance <= 14
            and metrics.layout_cosine >= 0.93
            and metrics.ocr_jaccard >= 0.35
        )
        or (metrics.layout_cosine >= 0.97 and metrics.ocr_jaccard >= 0.65)
    )


def strict_duplicate_edge(metrics: PairMetrics) -> bool:
    return (
        (metrics.phash_distance <= 3 and metrics.layout_cosine >= 0.93)
        or (
            metrics.phash_distance <= 6
            and metrics.layout_cosine >= 0.97
            and metrics.ocr_jaccard >= 0.55
        )
        or (metrics.layout_cosine >= 0.992 and metrics.ocr_jaccard >= 0.88)
    )


def cluster_cohesion(metrics: PairMetrics) -> bool:
    return (
        (metrics.phash_distance <= 16 and metrics.layout_cosine >= 0.78)
        or (metrics.layout_cosine >= 0.92 and metrics.ocr_jaccard >= 0.30)
        or (metrics.layout_cosine >= 0.84 and metrics.ocr_jaccard >= 0.72)
    ) and metrics.aspect_ratio_difference <= 0.20


def candidate_pair_indices(units: list[Unit]) -> set[tuple[int, int]]:
    size = len(units)
    candidates: set[tuple[int, int]] = set()
    by_file_hash: dict[str, list[int]] = defaultdict(list)
    by_normalized_hash: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        by_file_hash[unit.file_hash].append(index)
        by_normalized_hash[unit.normalized_image_hash].append(index)
    for groups in (by_file_hash, by_normalized_hash):
        for members in groups.values():
            for offset, first in enumerate(members):
                for second in members[offset + 1 :]:
                    candidates.add((min(first, second), max(first, second)))

    for first in range(size):
        first_hashes = units[first].phash_variants
        for second in range(first + 1, size):
            distance = min(
                (left ^ right).bit_count()
                for left in first_hashes
                for right in units[second].phash_variants
            )
            if distance <= 20:
                candidates.add((first, second))

    similarities = np.full((size, size), -1.0, dtype=np.float32)
    for rotation in range(4):
        base = np.vstack([unit.occupancy_variants[rotation] for unit in units])
        similarities = np.maximum(similarities, base @ base.T)
    np.fill_diagonal(similarities, -1.0)
    neighbor_count = min(25, max(size - 1, 1))
    for first in range(size):
        neighbors = np.argpartition(similarities[first], -neighbor_count)[-neighbor_count:]
        for second in neighbors:
            if similarities[first, second] >= 0.88:
                candidates.add((min(first, int(second)), max(first, int(second))))
    return candidates


def simple_component_count(
    size: int,
    candidate_metrics: dict[tuple[int, int], PairMetrics],
    predicate: Callable[[PairMetrics], bool],
) -> int:
    union = UnionFind(size)
    for (first, second), metrics in candidate_metrics.items():
        if predicate(metrics):
            union.union(first, second)
    return len({union.find(index) for index in range(size)})


def cluster_srfund_pages(
    units: list[Unit],
) -> tuple[dict[int, str], list[dict[str, Any]], dict[str, Any], dict[tuple[int, int], PairMetrics]]:
    candidates = candidate_pair_indices(units)
    metrics_by_pair = {
        pair: unit_pair_metrics(units[pair[0]], units[pair[1]]) for pair in sorted(candidates)
    }
    accepted: list[tuple[int, int, str, PairMetrics]] = []
    candidate_rows: list[dict[str, Any]] = []
    for (first, second), metrics in metrics_by_pair.items():
        reason = internal_duplicate_reason(units[first], units[second], metrics)
        candidate_rows.append(
            {
                "source_id_a": units[first].source_id,
                "source_id_b": units[second].source_id,
                "language_a": units[first].language_code,
                "language_b": units[second].language_code,
                "phash_distance": metrics.phash_distance,
                "layout_cosine": metrics.layout_cosine,
                "ocr_text_jaccard": metrics.ocr_jaccard,
                "entity_count_ratio": metrics.entity_count_ratio,
                "aspect_ratio_difference": metrics.aspect_ratio_difference,
                "rotation_quadrants": metrics.rotation_quadrants,
                "accepted_duplicate_edge": bool(reason),
                "edge_reason": reason,
            }
        )
        if reason:
            accepted.append((first, second, reason, metrics))

    clusters: dict[int, set[int]] = {index: {index} for index in range(len(units))}
    owner = list(range(len(units)))

    def root(item: int) -> int:
        while owner[item] != item:
            owner[item] = owner[owner[item]]
            item = owner[item]
        return item

    metric_cache = dict(metrics_by_pair)

    def metrics_for(first: int, second: int) -> PairMetrics:
        pair = (min(first, second), max(first, second))
        if pair not in metric_cache:
            metric_cache[pair] = unit_pair_metrics(units[pair[0]], units[pair[1]])
        return metric_cache[pair]

    accepted.sort(
        key=lambda item: (
            0 if item[2].startswith("exact") else 1,
            item[3].phash_distance,
            -item[3].layout_cosine,
            -item[3].ocr_jaccard,
        )
    )
    rejected_by_cohesion = 0
    for first, second, _, _ in accepted:
        first_root = root(first)
        second_root = root(second)
        if first_root == second_root:
            continue
        cross_pairs = [
            metrics_for(left, right)
            for left in clusters[first_root]
            for right in clusters[second_root]
        ]
        if not all(cluster_cohesion(metrics) for metrics in cross_pairs):
            rejected_by_cohesion += 1
            continue
        if len(clusters[first_root]) < len(clusters[second_root]):
            first_root, second_root = second_root, first_root
        owner[second_root] = first_root
        clusters[first_root].update(clusters.pop(second_root))

    ordered_components = sorted(
        (sorted(members) for members in clusters.values()),
        key=lambda members: min(units[index].unit_id for index in members),
    )
    membership: dict[int, str] = {}
    for number, members in enumerate(ordered_components, start=1):
        cluster_id = f"srfund_layout_{number:04d}"
        for index in members:
            membership[index] = cluster_id

    base_count = len(ordered_components)
    sensitivity = {
        "page_count": len(units),
        "candidate_pair_count": len(candidates),
        "accepted_edge_count": len(accepted),
        "cohesion_rejected_merges": rejected_by_cohesion,
        "strict_component_count": simple_component_count(len(units), metrics_by_pair, strict_duplicate_edge),
        "primary_cluster_count": base_count,
        "relaxed_component_count": simple_component_count(len(units), metrics_by_pair, relaxed_duplicate_edge),
        "multi_page_clusters": sum(len(members) > 1 for members in ordered_components),
        "largest_cluster_size": max(map(len, ordered_components), default=0),
    }
    return membership, candidate_rows, sensitivity, metric_cache


def canonical_source_alias(source_id: str) -> str:
    normalized = unicodedata.normalize("NFKC", source_id).casefold().strip()
    normalized = Path(normalized).stem
    normalized = re.sub(r"[^\w-]+", "_", normalized, flags=re.UNICODE).strip("_")
    match = re.fullmatch(r"(de|es|ja|pt|zh)_(?:train|val)_(\d+)", normalized)
    if match:
        return f"{match.group(1)}_{int(match.group(2))}"
    match = re.fullmatch(r"(de|es|ja|pt|zh)_(\d+)", normalized)
    if match:
        return f"{match.group(1)}_{int(match.group(2))}"
    return normalized


def cross_dataset_dedup_candidates(
    form_units: list[Unit], srfund_units: list[Unit]
) -> tuple[list[dict[str, Any]], dict[int, tuple[str, str]]]:
    form_base = np.vstack([unit.occupancy_variants[0] for unit in form_units])
    layout_similarities = np.zeros((len(form_units), len(srfund_units)), dtype=np.float32)
    for turns in range(4):
        srfund_variant = np.vstack([unit.occupancy_variants[turns] for unit in srfund_units])
        layout_similarities = np.maximum(layout_similarities, form_base @ srfund_variant.T)

    all_units = [*form_units, *srfund_units]
    normalized_texts = [normalize_text(unit.text) or "empty" for unit in all_units]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=1, max_features=50000, lowercase=False, dtype=np.float32
    )
    matrix = normalize(vectorizer.fit_transform(normalized_texts), norm="l2")
    form_text = matrix[: len(form_units)]
    srfund_text = matrix[len(form_units) :]
    text_similarities = (form_text @ srfund_text.T).toarray()

    candidate_pairs: set[tuple[int, int]] = set()
    aliases = [canonical_source_alias(unit.source_id) for unit in srfund_units]
    form_aliases = [canonical_source_alias(unit.source_id) for unit in form_units]
    for form_index, form_unit in enumerate(form_units):
        layout_neighbors = np.argpartition(layout_similarities[form_index], -20)[-20:]
        text_neighbors = np.argpartition(text_similarities[form_index], -20)[-20:]
        candidate_pairs.update((form_index, int(index)) for index in layout_neighbors)
        candidate_pairs.update((form_index, int(index)) for index in text_neighbors)
        for srfund_index, srfund_unit in enumerate(srfund_units):
            source_alias = bool(form_aliases[form_index]) and form_aliases[form_index] == aliases[srfund_index]
            phash_distance = min(
                (left ^ right).bit_count()
                for left in form_unit.phash_variants
                for right in srfund_unit.phash_variants
            )
            if source_alias or phash_distance <= 20:
                candidate_pairs.add((form_index, srfund_index))

    rows: list[dict[str, Any]] = []
    excluded: dict[int, tuple[str, str]] = {}
    for form_index, srfund_index in sorted(candidate_pairs):
        form_unit = form_units[form_index]
        srfund_unit = srfund_units[srfund_index]
        metrics = unit_pair_metrics(form_unit, srfund_unit)
        exact_source = form_unit.source_id.casefold() == srfund_unit.source_id.casefold() and bool(form_unit.source_id)
        source_alias = bool(form_aliases[form_index]) and form_aliases[form_index] == aliases[srfund_index]
        text_cosine = float(text_similarities[form_index, srfund_index])
        exact_hash = form_unit.file_hash == srfund_unit.file_hash
        normalized_hash = form_unit.normalized_image_hash == srfund_unit.normalized_image_hash
        status = "candidate_only"
        reasons: list[str] = []
        if exact_source:
            status = "confirmed_overlap"
            reasons.append("exact_source_id")
        if exact_hash:
            status = "confirmed_overlap"
            reasons.append("exact_file_sha256")
        if normalized_hash:
            status = "confirmed_overlap"
            reasons.append("exact_normalized_image_sha256")
        # Language/index aliases are useful provenance only when their split or
        # independent visual/text evidence supports the match.  Validation
        # aliases are not unique document IDs: the same numeric index can name
        # a different form there, as verified by the image audit.
        alias_visual_confirmation = source_alias and (
            (metrics.phash_distance <= 8 and metrics.layout_cosine >= 0.65)
            or (
                metrics.phash_distance <= 14
                and metrics.layout_cosine >= 0.65
                and (metrics.entity_count_ratio >= 0.75 or text_cosine >= 0.35)
            )
            or (metrics.phash_distance <= 14 and text_cosine >= 0.70)
            or (metrics.layout_cosine >= 0.90 and metrics.ocr_jaccard >= 0.35)
        )
        alias_train_provenance = source_alias and srfund_unit.split == "train"
        if status == "candidate_only" and alias_train_provenance:
            status = "confirmed_overlap"
            reasons.append("source_alias_train_split_provenance")
        elif status == "candidate_only" and alias_visual_confirmation:
            status = "suspected_overlap"
            reasons.append("source_alias_with_visual_or_text_confirmation")
        strong_signals = sum(
            (
                metrics.phash_distance <= 14,
                metrics.layout_cosine >= 0.88,
                text_cosine >= 0.55,
                metrics.ocr_jaccard >= 0.45,
            )
        )
        if status == "candidate_only" and (
            (metrics.phash_distance <= 8 and metrics.layout_cosine >= 0.82)
            or strong_signals >= 3
            or (metrics.layout_cosine >= 0.96 and text_cosine >= 0.70)
        ):
            status = "suspected_overlap"
            reasons.append("multiple_strong_visual_text_signals")
        if source_alias and status == "candidate_only":
            reasons.append("ambiguous_source_alias_without_signal_confirmation")
        if not reasons:
            reasons.append("nearest_signal_candidate_for_audit")
        row = {
            "formstruct_template_id": form_unit.template_id,
            "formstruct_source_id": form_unit.source_id,
            "srfund_source_id": srfund_unit.source_id,
            "srfund_language": srfund_unit.language_code,
            "srfund_split": srfund_unit.split,
            "exact_source_id": exact_source,
            "source_alias_match": source_alias,
            "exact_file_hash": exact_hash,
            "exact_normalized_image_hash": normalized_hash,
            "phash_distance": metrics.phash_distance,
            "layout_cosine": metrics.layout_cosine,
            "ocr_text_jaccard": metrics.ocr_jaccard,
            "ocr_tfidf_cosine": text_cosine,
            "entity_count_ratio": metrics.entity_count_ratio,
            "aspect_ratio_difference": metrics.aspect_ratio_difference,
            "rotation_quadrants": metrics.rotation_quadrants,
            "dedup_status": status,
            "reason": ";".join(reasons),
            "formstruct_image": str(form_unit.image_path),
            "srfund_image": str(srfund_unit.image_path),
        }
        rows.append(row)
        if status in {"confirmed_overlap", "suspected_overlap"}:
            previous = excluded.get(srfund_index)
            if previous is None or status == "confirmed_overlap":
                excluded[srfund_index] = (status, row["reason"])
    rows.sort(
        key=lambda row: (
            {"confirmed_overlap": 0, "suspected_overlap": 1, "candidate_only": 2}[row["dedup_status"]],
            int(row["phash_distance"]),
            -float(row["layout_cosine"]),
            -float(row["ocr_tfidf_cosine"]),
        )
    )
    return rows, excluded


def choose_cluster_representative(
    members: list[int], units: list[Unit], metric_cache: dict[tuple[int, int], PairMetrics]
) -> int:
    if len(members) == 1:
        return members[0]

    def metrics(first: int, second: int) -> PairMetrics:
        pair = (min(first, second), max(first, second))
        if pair not in metric_cache:
            metric_cache[pair] = unit_pair_metrics(units[pair[0]], units[pair[1]])
        return metric_cache[pair]

    scores: dict[int, float] = {}
    for candidate in members:
        distances = []
        for other in members:
            if candidate == other:
                continue
            pair_metrics = metrics(candidate, other)
            distances.append(
                0.45 * pair_metrics.phash_distance / 64.0
                + 0.40 * (1.0 - pair_metrics.layout_cosine)
                + 0.15 * (1.0 - pair_metrics.ocr_jaccard)
            )
        scores[candidate] = float(np.mean(distances))
    return min(members, key=lambda index: (scores[index], units[index].unit_id))


def build_reference_clusters(
    units: list[Unit],
    membership: dict[int, str],
    page_exclusions: dict[int, tuple[str, str]],
    metric_cache: dict[tuple[int, int], PairMetrics],
) -> tuple[list[Unit], list[dict[str, Any]], list[dict[str, Any]]]:
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in membership.items():
        by_cluster[cluster_id].append(index)
    reference_units: list[Unit] = []
    membership_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for cluster_id, members in sorted(by_cluster.items()):
        representative_index = choose_cluster_representative(members, units, metric_cache)
        representative = units[representative_index]
        excluded_members = [index for index in members if index in page_exclusions]
        status = "included"
        reason = ""
        if excluded_members:
            statuses = [page_exclusions[index][0] for index in excluded_members]
            status = "confirmed_overlap" if "confirmed_overlap" in statuses else "suspected_overlap"
            reason = ";".join(sorted({page_exclusions[index][1] for index in excluded_members}))
        member_ids = tuple(sorted(units[index].source_id for index in members))
        cluster_unit = replace(
            representative,
            unit_id=cluster_id,
            cluster_id=cluster_id,
            cluster_size=len(members),
            cluster_members=member_ids,
            dedup_status=status,
            dedup_reason=reason,
        )
        reference_units.append(cluster_unit)
        for index in members:
            page = units[index]
            pair_metrics = (
                PairMetrics(0, 1.0, 1.0, 1.0, 0.0, 0)
                if index == representative_index
                else unit_pair_metrics(page, representative)
            )
            membership_rows.append(
                {
                    "srfund_cluster_id": cluster_id,
                    "source_id": page.source_id,
                    "language": page.features["language"],
                    "language_code": page.language_code,
                    "split": page.split,
                    "image_path": str(page.image_path),
                    "is_representative": index == representative_index,
                    "cluster_size": len(members),
                    "phash_distance_to_representative": pair_metrics.phash_distance,
                    "layout_cosine_to_representative": pair_metrics.layout_cosine,
                    "ocr_jaccard_to_representative": pair_metrics.ocr_jaccard,
                    "cluster_dedup_status": status,
                    "cluster_dedup_reason": reason,
                }
            )
        if status != "included":
            excluded_rows.append(
                {
                    "srfund_cluster_id": cluster_id,
                    "representative_source_id": representative.source_id,
                    "cluster_size": len(members),
                    "member_source_ids": json.dumps(member_ids, ensure_ascii=False),
                    "dedup_status": status,
                    "reason": reason,
                }
            )
    return reference_units, membership_rows, excluded_rows


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def public_feature_row(unit: Unit) -> dict[str, Any]:
    row: dict[str, Any] = {
        "unit_id": unit.unit_id,
        "template_id": unit.template_id,
        "srfund_cluster_id": unit.cluster_id,
        "source_id": unit.source_id,
        "source_document_id": unit.source_document_id,
        "dataset": unit.dataset,
        "language": unit.features.get("language"),
        "language_code": unit.language_code,
        "script": unit.features.get("script"),
        "writing_direction": unit.features.get("writing_direction"),
        "split": unit.split,
        "dedup_status": unit.dedup_status,
        "dedup_reason": unit.dedup_reason,
        "cluster_size": unit.cluster_size,
        "cluster_member_source_ids": json.dumps(unit.cluster_members, ensure_ascii=False),
        "source_image": str(unit.image_path),
        "source_annotation": str(unit.annotation_path),
        "image_file_sha256": unit.file_hash,
        "normalized_image_sha256": unit.normalized_image_hash,
        "perceptual_hash": f"{unit.phash_variants[0]:016x}" if unit.phash_variants else "",
    }
    row.update(unit.features)
    return {key: json_ready(value) for key, value in row.items()}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def build_analysis_subsets(
    form_units: list[Unit], reference_units: list[Unit]
) -> dict[str, tuple[list[Unit], list[Unit]]]:
    included_reference = [unit for unit in reference_units if unit.dedup_status == "included"]
    subsets: dict[str, tuple[list[Unit], list[Unit]]] = {
        "all_comparable": (form_units, included_reference)
    }
    form_languages = {unit.language_code for unit in form_units}
    reference_languages = {unit.language_code for unit in included_reference}
    shared = sorted(form_languages & reference_languages)
    subsets["shared_languages"] = (
        [unit for unit in form_units if unit.language_code in shared],
        [unit for unit in included_reference if unit.language_code in shared],
    )
    for language in shared:
        form_slice = [unit for unit in form_units if unit.language_code == language]
        reference_slice = [unit for unit in included_reference if unit.language_code == language]
        if len(form_slice) >= 2 and len(reference_slice) >= 4:
            subsets[f"language_{language}"] = (form_slice, reference_slice)
    for presence, label in ((True, "table_present"), (False, "table_absent")):
        form_slice = [unit for unit in form_units if bool(unit.features.get("table_presence")) is presence]
        reference_slice = [
            unit for unit in included_reference if bool(unit.features.get("table_presence")) is presence
        ]
        if len(form_slice) >= 2 and len(reference_slice) >= 4:
            subsets[label] = (form_slice, reference_slice)
    for depth_label in ("shallow_0_2", "medium_3_4", "deep_5_plus"):
        form_slice = [unit for unit in form_units if unit.features.get("hierarchy_depth_bin") == depth_label]
        reference_slice = [
            unit
            for unit in included_reference
            if unit.features.get("hierarchy_depth_bin") == depth_label
        ]
        if len(form_slice) >= 2 and len(reference_slice) >= 4:
            subsets[f"hierarchy_{depth_label}"] = (form_slice, reference_slice)
    return subsets


def categorical_distribution(units: Sequence[Unit], feature: str) -> dict[str, float]:
    if not units:
        return {}
    if feature == "entity_type_distribution":
        values = {
            label: float(np.mean([float(unit.features.get(f"{label}_entity_ratio") or 0.0) for unit in units]))
            for label in ENTITY_LABELS
        }
        total = sum(values.values())
        return {label: value / total for label, value in values.items()} if total else values
    source_key = {
        "language": "language",
        "script": "script",
        "writing_direction": "writing_direction",
        "table_presence": "table_presence",
        "hierarchy_depth_bin": "hierarchy_depth_bin",
    }[feature]
    counts = Counter(str(unit.features.get(source_key)) for unit in units)
    total = sum(counts.values())
    return {key: count / total for key, count in sorted(counts.items())}


def jensen_shannon_divergence(
    first: dict[str, float], second: dict[str, float], smoothing: float = EPSILON
) -> float:
    categories = sorted(set(first) | set(second))
    if not categories:
        return 0.0
    first_array = np.asarray([first.get(category, 0.0) + smoothing for category in categories], dtype=float)
    second_array = np.asarray([second.get(category, 0.0) + smoothing for category in categories], dtype=float)
    first_array /= first_array.sum()
    second_array /= second_array.sum()
    midpoint = (first_array + second_array) / 2.0
    first_kl = np.sum(first_array * np.log2(first_array / midpoint))
    second_kl = np.sum(second_array * np.log2(second_array / midpoint))
    return float(0.5 * (first_kl + second_kl))


def numeric_values(units: Sequence[Unit], feature: str) -> np.ndarray:
    values = []
    for unit in units:
        value = finite_number(unit.features.get(feature))
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=float)


def reference_quantile_span(units: Sequence[Unit], feature: str) -> float:
    values = numeric_values(units, feature)
    if not len(values):
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def normalized_wasserstein(
    first: Sequence[Unit], second: Sequence[Unit], feature: str, span: float
) -> tuple[float, float]:
    first_values = numeric_values(first, feature)
    second_values = numeric_values(second, feature)
    if not len(first_values) or not len(second_values):
        return float("nan"), float("nan")
    raw = float(wasserstein_distance(first_values, second_values))
    return raw, raw / (span + EPSILON)


def fit_gower_config(
    reference_units: Sequence[Unit],
    group_specification: dict[str, dict[str, list[str]]] | None = None,
) -> GowerConfig:
    ranges: dict[str, float] = {}
    active_groups: list[str] = []
    groups: dict[str, dict[str, list[str]]] = {}
    source_groups = group_specification or GOWER_GROUPS
    for group_name, specification in source_groups.items():
        continuous = []
        categorical = []
        for feature in specification["continuous"]:
            values = numeric_values(reference_units, feature)
            if len(values):
                ranges[feature] = max(float(np.max(values) - np.min(values)), EPSILON)
                continuous.append(feature)
        for feature in specification["categorical"]:
            if any(unit.features.get(feature) is not None for unit in reference_units):
                categorical.append(feature)
        if continuous or categorical:
            active_groups.append(group_name)
            groups[group_name] = {"continuous": continuous, "categorical": categorical}
    return GowerConfig(groups=groups, ranges=ranges, active_groups=active_groups)


def unit_feature_value(unit: Unit, feature: str) -> Any:
    if feature == "language_code":
        return unit.language_code
    return unit.features.get(feature)


def gower_distance_matrix(
    first: Sequence[Unit], second: Sequence[Unit], config: GowerConfig
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    shape = (len(first), len(second))
    group_matrices: dict[str, np.ndarray] = {}
    for group_name in config.active_groups:
        feature_matrices: list[np.ndarray] = []
        specification = config.groups[group_name]
        for feature in specification["continuous"]:
            left = np.asarray([finite_number(unit_feature_value(unit, feature)) for unit in first], dtype=float)
            right = np.asarray([finite_number(unit_feature_value(unit, feature)) for unit in second], dtype=float)
            feature_matrices.append(np.minimum(np.abs(left[:, None] - right[None, :]) / config.ranges[feature], 1.0))
        for feature in specification["categorical"]:
            left = np.asarray([str(unit_feature_value(unit, feature)) for unit in first], dtype=object)
            right = np.asarray([str(unit_feature_value(unit, feature)) for unit in second], dtype=object)
            feature_matrices.append((left[:, None] != right[None, :]).astype(float))
        group_matrices[group_name] = (
            np.mean(np.stack(feature_matrices, axis=0), axis=0) if feature_matrices else np.zeros(shape)
        )
    if not group_matrices:
        raise ValueError("no active Gower feature groups")
    total = np.mean(np.stack(list(group_matrices.values()), axis=0), axis=0)
    return total, group_matrices


def gower_pair_contributions(first: Unit, second: Unit, config: GowerConfig) -> list[tuple[str, float]]:
    contributions: list[tuple[str, float]] = []
    for group_name in config.active_groups:
        specification = config.groups[group_name]
        for feature in specification["continuous"]:
            left = finite_number(unit_feature_value(first, feature))
            right = finite_number(unit_feature_value(second, feature))
            if left is not None and right is not None:
                contributions.append((feature, min(abs(left - right) / config.ranges[feature], 1.0)))
        for feature in specification["categorical"]:
            contributions.append(
                (feature, float(str(unit_feature_value(first, feature)) != str(unit_feature_value(second, feature))))
            )
    return sorted(contributions, key=lambda item: (-item[1], item[0]))


def choose_stratum_function(
    form_units: Sequence[Unit], reference_units: Sequence[Unit]
) -> tuple[str, Callable[[Unit], str]]:
    form_languages = {unit.language_code for unit in form_units}
    reference_languages = {unit.language_code for unit in reference_units}
    if len(form_languages) > 1 and form_languages.issubset(reference_languages):
        return "language", lambda unit: unit.language_code
    return (
        "table_presence+hierarchy_depth",
        lambda unit: f"{int(bool(unit.features.get('table_presence')))}|{unit.features.get('hierarchy_depth_bin')}",
    )


def allocate_stratified_counts(
    target_labels: Sequence[str], reference_labels: Sequence[str], size: int
) -> dict[str, int]:
    target_counts = Counter(target_labels)
    reference_counts = Counter(reference_labels)
    capacities = {label: count // 2 for label, count in reference_counts.items()}
    target_total = max(sum(target_counts.values()), 1)
    desired = {label: size * target_counts.get(label, 0) / target_total for label in capacities}
    allocation = {
        label: min(int(math.floor(desired[label])), capacities[label]) for label in capacities
    }
    while sum(allocation.values()) < size:
        available = [label for label in capacities if allocation[label] < capacities[label]]
        if not available:
            raise ValueError("insufficient per-stratum capacity for two disjoint reference sets")
        label = max(
            available,
            key=lambda item: (
                desired.get(item, 0.0) - allocation[item],
                target_counts.get(item, 0),
                capacities[item] - allocation[item],
                item,
            ),
        )
        allocation[label] += 1
    return {label: count for label, count in allocation.items() if count}


def sample_reference_sets(
    form_units: Sequence[Unit],
    reference_units: Sequence[Unit],
    size: int,
    rng: np.random.Generator,
    design: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if design == "unstratified":
        permutation = rng.permutation(len(reference_units))
        return permutation[:size], permutation[size : 2 * size], "none"
    stratum_name, stratum_function = choose_stratum_function(form_units, reference_units)
    form_labels = [stratum_function(unit) for unit in form_units]
    reference_labels = [stratum_function(unit) for unit in reference_units]
    allocation = allocate_stratified_counts(form_labels, reference_labels, size)
    first: list[int] = []
    second: list[int] = []
    for label, count in sorted(allocation.items()):
        members = np.asarray([index for index, value in enumerate(reference_labels) if value == label])
        selected = rng.choice(members, size=2 * count, replace=False)
        first.extend(int(index) for index in selected[:count])
        second.extend(int(index) for index in selected[count:])
    rng.shuffle(first)
    rng.shuffle(second)
    return np.asarray(first), np.asarray(second), stratum_name


def sample_form_indices(
    form_units: Sequence[Unit], size: int, rng: np.random.Generator
) -> np.ndarray:
    if len(form_units) == size:
        return np.arange(size)
    return np.sort(rng.choice(len(form_units), size=size, replace=False))


def run_calibration(
    subsets: dict[str, tuple[list[Unit], list[Unit]]], rounds: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if rounds < 1000:
        raise ValueError("Real-Real calibration requires at least 1000 rounds")
    all_rows: list[dict[str, Any]] = []
    runtime: dict[str, dict[str, Any]] = {}
    for subset_offset, (subset_name, (form_units, reference_units)) in enumerate(subsets.items()):
        size = min(70, len(form_units), len(reference_units) // 2)
        if size < 2:
            continue
        gower_variants = {
            "mapped": fit_gower_config(reference_units),
            "direct_only": fit_gower_config(reference_units, DIRECT_GOWER_GROUPS),
        }
        gower_matrices: dict[str, np.ndarray] = {}
        gower_reference_matrices: dict[str, np.ndarray] = {}
        gower_group_matrices: dict[str, dict[str, np.ndarray]] = {}
        for variant, config in gower_variants.items():
            template_matrix, group_matrices = gower_distance_matrix(form_units, reference_units, config)
            gower_matrices[variant] = template_matrix
            gower_group_matrices[variant] = group_matrices
            gower_reference_matrices[variant], _ = gower_distance_matrix(
                reference_units, reference_units, config
            )
        spans = {
            feature: reference_quantile_span(reference_units, feature)
            for feature in WASSERSTEIN_FEATURES
        }
        runtime[subset_name] = {
            "form_units": form_units,
            "reference_units": reference_units,
            "size": size,
            "gower_configs": gower_variants,
            "template_reference": gower_matrices["mapped"],
            "reference_reference": gower_reference_matrices["mapped"],
            "gower_matrices": gower_matrices,
            "gower_reference_matrices": gower_reference_matrices,
            "group_matrices": gower_group_matrices,
            "spans": spans,
        }
        for design_offset, design in enumerate(("unstratified", "stratified")):
            rng = np.random.default_rng(seed + subset_offset * 100003 + design_offset * 10000019)
            for round_index in range(rounds):
                template_indices = sample_form_indices(form_units, size, rng)
                real_a_indices, real_b_indices, stratum_name = sample_reference_sets(
                    form_units, reference_units, size, rng, design
                )
                template_sample = [form_units[index] for index in template_indices]
                real_a_sample = [reference_units[index] for index in real_a_indices]
                real_b_sample = [reference_units[index] for index in real_b_indices]
                common = {
                    "round": round_index,
                    "seed": seed,
                    "analysis_subset": subset_name,
                    "calibration_design": design,
                    "stratification": stratum_name,
                    "n_formstruct": size,
                    "n_real_a": size,
                    "n_real_b": size,
                }
                for feature, tier in JS_FEATURES.items():
                    template_value = jensen_shannon_divergence(
                        categorical_distribution(template_sample, feature),
                        categorical_distribution(real_b_sample, feature),
                    )
                    real_value = jensen_shannon_divergence(
                        categorical_distribution(real_a_sample, feature),
                        categorical_distribution(real_b_sample, feature),
                    )
                    all_rows.append(
                        {
                            **common,
                            "metric_family": "js",
                            "feature": feature,
                            "comparison_tier": tier,
                            "template_real": template_value,
                            "real_real": real_value,
                            "calibrated_difference": template_value - real_value,
                            "calibrated_ratio": template_value / (real_value + EPSILON),
                            "tau": np.nan,
                            "template_coverage": np.nan,
                            "real_real_coverage": np.nan,
                        }
                    )
                for feature, tier in WASSERSTEIN_FEATURES.items():
                    span = spans[feature]
                    _, template_value = normalized_wasserstein(template_sample, real_b_sample, feature, span)
                    _, real_value = normalized_wasserstein(real_a_sample, real_b_sample, feature, span)
                    if not np.isfinite(template_value) or not np.isfinite(real_value):
                        continue
                    all_rows.append(
                        {
                            **common,
                            "metric_family": "wasserstein",
                            "feature": feature,
                            "comparison_tier": tier,
                            "template_real": template_value,
                            "real_real": real_value,
                            "calibrated_difference": template_value - real_value,
                            "calibrated_ratio": template_value / (real_value + EPSILON),
                            "tau": np.nan,
                            "template_coverage": np.nan,
                            "real_real_coverage": np.nan,
                        }
                    )
                for variant, template_matrix in gower_matrices.items():
                    reference_matrix = gower_reference_matrices[variant]
                    template_real_distances = template_matrix[np.ix_(template_indices, real_b_indices)]
                    real_real_distances = reference_matrix[np.ix_(real_a_indices, real_b_indices)]
                    template_nearest = np.min(template_real_distances, axis=0)
                    real_nearest = np.min(real_real_distances, axis=0)
                    tau = float(np.quantile(real_nearest, 0.95))
                    template_value = float(np.mean(template_nearest))
                    real_value = float(np.mean(real_nearest))
                    feature_name = (
                        "group_balanced_gower"
                        if variant == "mapped"
                        else "group_balanced_gower_direct_only"
                    )
                    tier = (
                        "direct+conditional_group_balanced"
                        if variant == "mapped"
                        else "direct_only_group_balanced"
                    )
                    all_rows.append(
                        {
                            **common,
                            "metric_family": "gower_nn",
                            "feature": feature_name,
                            "comparison_tier": tier,
                            "template_real": template_value,
                            "real_real": real_value,
                            "calibrated_difference": template_value - real_value,
                            "calibrated_ratio": template_value / (real_value + EPSILON),
                            "tau": tau,
                            "template_coverage": float(np.mean(template_nearest <= tau)),
                            "real_real_coverage": float(np.mean(real_nearest <= tau)),
                        }
                    )
        print(
            f"calibrated {subset_name}: form={len(form_units)}, reference={len(reference_units)}, matched={size}, rounds={rounds}"
        )
    return all_rows, runtime


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def summarize_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    group_columns = [
        "analysis_subset",
        "calibration_design",
        "stratification",
        "metric_family",
        "feature",
        "comparison_tier",
        "n_formstruct",
        "n_real_a",
        "n_real_b",
    ]
    output: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        for column in (
            "template_real",
            "real_real",
            "calibrated_difference",
            "calibrated_ratio",
            "tau",
            "template_coverage",
            "real_real_coverage",
        ):
            values = group[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                row[f"{column}_median"] = np.nan
                row[f"{column}_ci_low"] = np.nan
                row[f"{column}_ci_high"] = np.nan
                continue
            low, high = percentile_interval(values)
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        template_median = float(row["template_real_median"])
        real_values = group["real_real"].to_numpy(dtype=float)
        real_values = real_values[np.isfinite(real_values)]
        row["template_real_percentile_in_real_real"] = (
            float(np.mean(real_values <= template_median) * 100.0) if len(real_values) else np.nan
        )
        row["rounds"] = len(group)
        output.append(row)
    return output


def calibration_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row["analysis_subset"]),
            str(row["calibration_design"]),
            str(row["metric_family"]),
            str(row["feature"]),
        ): row
        for row in rows
    }


def empirical_interpretation(percentile: float) -> str:
    if not math.isfinite(percentile):
        return "not estimable"
    if percentile <= 50:
        return "Template-Real distance is at or below the Real-Real median"
    if percentile <= 95:
        return f"Template-Real distance is in the upper Real-Real range ({percentile:.1f}th percentile)"
    return f"Template-Real distance exceeds the Real-Real 95th percentile ({percentile:.1f}th percentile)"


def build_js_results(
    subsets: dict[str, tuple[list[Unit], list[Unit]]], summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = calibration_lookup(summary_rows)
    output: list[dict[str, Any]] = []
    for subset_name, (form_units, reference_units) in subsets.items():
        for feature, tier in JS_FEATURES.items():
            form_distribution = categorical_distribution(form_units, feature)
            reference_distribution = categorical_distribution(reference_units, feature)
            full_distance = jensen_shannon_divergence(form_distribution, reference_distribution)
            for design in ("unstratified", "stratified"):
                summary = lookup.get((subset_name, design, "js", feature))
                if summary is None:
                    continue
                percentile = float(summary["template_real_percentile_in_real_real"])
                output.append(
                    {
                        "feature": feature,
                        "analysis_subset": subset_name,
                        "calibration_design": design,
                        "comparison_tier": tier,
                        "formstruct_distribution": json.dumps(form_distribution, sort_keys=True),
                        "srfund_distribution": json.dumps(reference_distribution, sort_keys=True),
                        "full_sample_js": full_distance,
                        "js_template_real": summary["template_real_median"],
                        "js_template_real_ci_low": summary["template_real_ci_low"],
                        "js_template_real_ci_high": summary["template_real_ci_high"],
                        "js_real_real_median": summary["real_real_median"],
                        "js_real_real_ci_low": summary["real_real_ci_low"],
                        "js_real_real_ci_high": summary["real_real_ci_high"],
                        "calibrated_difference": summary["calibrated_difference_median"],
                        "calibrated_ratio": summary["calibrated_ratio_median"],
                        "template_real_percentile_in_real_real": percentile,
                        "interpretation": empirical_interpretation(percentile),
                        "category_support": json.dumps(sorted(set(form_distribution) | set(reference_distribution))),
                        "zero_frequency_smoothing": EPSILON,
                        "n_formstruct": len(form_units),
                        "n_srfund_clusters": len(reference_units),
                        "size_matched_n": summary["n_formstruct"],
                    }
                )
    return output


def bootstrap_wasserstein_ci(
    form_units: Sequence[Unit],
    reference_units: Sequence[Unit],
    feature: str,
    span: float,
    rounds: int,
    seed: int,
) -> tuple[float, float, float, float]:
    form_values = numeric_values(form_units, feature)
    reference_values = numeric_values(reference_units, feature)
    if not len(form_values) or not len(reference_values):
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    raw_values = []
    normalized_values = []
    for _ in range(rounds):
        form_sample = rng.choice(form_values, size=len(form_values), replace=True)
        reference_sample = rng.choice(reference_values, size=len(reference_values), replace=True)
        raw = float(wasserstein_distance(form_sample, reference_sample))
        raw_values.append(raw)
        normalized_values.append(raw / (span + EPSILON))
    raw_low, raw_high = percentile_interval(raw_values)
    normalized_low, normalized_high = percentile_interval(normalized_values)
    return raw_low, raw_high, normalized_low, normalized_high


def build_wasserstein_results(
    subsets: dict[str, tuple[list[Unit], list[Unit]]],
    summary_rows: list[dict[str, Any]],
    rounds: int,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = calibration_lookup(summary_rows)
    output: list[dict[str, Any]] = []
    for subset_offset, (subset_name, (form_units, reference_units)) in enumerate(subsets.items()):
        for feature_offset, (feature, tier) in enumerate(WASSERSTEIN_FEATURES.items()):
            span = reference_quantile_span(reference_units, feature)
            raw_w1, normalized_w1 = normalized_wasserstein(form_units, reference_units, feature, span)
            if not np.isfinite(raw_w1):
                continue
            raw_low, raw_high, normalized_low, normalized_high = bootstrap_wasserstein_ci(
                form_units,
                reference_units,
                feature,
                span,
                rounds,
                seed + subset_offset * 1009 + feature_offset * 100003,
            )
            reference_values = numeric_values(reference_units, feature)
            q05 = float(np.percentile(reference_values, 5))
            q95 = float(np.percentile(reference_values, 95))
            for design in ("unstratified", "stratified"):
                summary = lookup.get((subset_name, design, "wasserstein", feature))
                if summary is None:
                    continue
                percentile = float(summary["template_real_percentile_in_real_real"])
                output.append(
                    {
                        "feature": feature,
                        "analysis_subset": subset_name,
                        "calibration_design": design,
                        "comparison_tier": tier,
                        "raw_w1": raw_w1,
                        "normalized_w1": normalized_w1,
                        "bootstrap_raw_w1_ci_low": raw_low,
                        "bootstrap_raw_w1_ci_high": raw_high,
                        "bootstrap_normalized_w1_ci_low": normalized_low,
                        "bootstrap_normalized_w1_ci_high": normalized_high,
                        "template_real_size_matched_median": summary["template_real_median"],
                        "template_real_size_matched_ci_low": summary["template_real_ci_low"],
                        "template_real_size_matched_ci_high": summary["template_real_ci_high"],
                        "real_real_median": summary["real_real_median"],
                        "real_real_ci_low": summary["real_real_ci_low"],
                        "real_real_ci_high": summary["real_real_ci_high"],
                        "calibrated_difference": summary["calibrated_difference_median"],
                        "calibrated_ratio": summary["calibrated_ratio_median"],
                        "template_real_percentile_in_real_real": percentile,
                        "interpretation": empirical_interpretation(percentile),
                        "srfund_q05": q05,
                        "srfund_q95": q95,
                        "normalization_span": span,
                        "epsilon": EPSILON,
                        "n_formstruct": len(form_units),
                        "n_srfund_clusters": len(reference_units),
                        "size_matched_n": summary["n_formstruct"],
                    }
                )
    return output


def bootstrap_proportion_ci(
    values: Sequence[bool], rounds: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(rng.choice(array, size=len(array), replace=True))) for _ in range(rounds)]
    return percentile_interval(estimates)


def build_coverage_results(
    runtime: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    rounds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lookup = calibration_lookup(summary_rows)
    coverage_rows: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    plot_payload: dict[str, Any] = {}
    for subset_offset, (subset_name, state) in enumerate(runtime.items()):
        form_units: list[Unit] = state["form_units"]
        reference_units: list[Unit] = state["reference_units"]
        mapped_matrix: np.ndarray = state["gower_matrices"]["mapped"]
        mapped_nearest_indices = np.argmin(mapped_matrix, axis=0)
        mapped_nearest_distances = np.min(mapped_matrix, axis=0)
        for design in ("unstratified", "stratified"):
            for variant, feature_name in (
                ("mapped", "group_balanced_gower"),
                ("direct_only", "group_balanced_gower_direct_only"),
            ):
                summary = lookup.get((subset_name, design, "gower_nn", feature_name))
                if summary is None:
                    continue
                # Coverage itself is reported from the same size-matched
                # repetitions used to calibrate tau.  This avoids silently
                # replacing the 70-vs-70 design with a full 1,528-cluster
                # nearest-neighbour calculation.
                coverage = float(summary["template_coverage_median"])
                coverage_low = float(summary["template_coverage_ci_low"])
                coverage_high = float(summary["template_coverage_ci_high"])
                real_coverage = float(summary["real_real_coverage_median"])
                coverage_rows.append(
                    {
                        "analysis_subset": subset_name,
                        "slice_type": "analysis_subset",
                        "slice_value": subset_name,
                        "calibration_design": design,
                        "stratification": summary["stratification"],
                        "gower_feature_set": variant,
                        "n_formstruct": len(form_units),
                        "n_srfund_clusters": len(reference_units),
                        "size_matched_reference_set": int(summary["n_real_a"]),
                        "tau": float(summary["tau_median"]),
                        "tau_ci_low": summary["tau_ci_low"],
                        "tau_ci_high": summary["tau_ci_high"],
                        "coverage": coverage,
                        "coverage_ci_low": coverage_low,
                        "coverage_ci_high": coverage_high,
                        "real_real_coverage_median": real_coverage,
                        "real_real_coverage_ci_low": summary["real_real_coverage_ci_low"],
                        "real_real_coverage_ci_high": summary["real_real_coverage_ci_high"],
                        "calibrated_coverage_difference": coverage - real_coverage,
                        "calibrated_coverage_ratio": coverage / (real_coverage + EPSILON),
                        "mean_nearest_distance": float(summary["template_real_median"]),
                        "median_nearest_distance": np.nan,
                        "template_real_distance_calibration_median": summary["template_real_median"],
                        "real_real_distance_median": summary["real_real_median"],
                        "real_real_distance_ci_low": summary["real_real_ci_low"],
                        "real_real_distance_ci_high": summary["real_real_ci_high"],
                        "distance_calibrated_ratio": summary["calibrated_ratio_median"],
                        "template_real_percentile_in_real_real": summary[
                            "template_real_percentile_in_real_real"
                        ],
                    }
                )
        if subset_name == "all_comparable":
            summary = lookup[(subset_name, "unstratified", "gower_nn", "group_balanced_gower")]
            tau = float(summary["tau_median"])
            config: GowerConfig = state["gower_configs"]["mapped"]
            for reference_index, reference in enumerate(reference_units):
                template = form_units[int(mapped_nearest_indices[reference_index])]
                differences = gower_pair_contributions(reference, template, config)[:5]
                nearest_rows.append(
                    {
                        "srfund_cluster_id": reference.cluster_id,
                        "srfund_source_id": reference.source_id,
                        "nearest_formstruct_template_id": template.template_id,
                        "nearest_formstruct_source_id": template.source_id,
                        "gower_distance": float(mapped_nearest_distances[reference_index]),
                        "covered_at_tau": bool(mapped_nearest_distances[reference_index] <= tau),
                        "coverage_scope": "full_reference_descriptive",
                        "size_matched_n": int(summary["n_real_a"]),
                        "tau": tau,
                        "language": reference.features["language"],
                        "language_code": reference.language_code,
                        "table_presence": bool(reference.features["table_presence"]),
                        "hierarchy_depth": int(reference.features["max_hierarchy_depth"]),
                        "hierarchy_depth_bin": reference.features["hierarchy_depth_bin"],
                        "main_difference_features": ";".join(
                            f"{feature}={value:.3f}" for feature, value in differences
                        ),
                        "dedup_status": reference.dedup_status,
                    }
                )
            plot_payload = {
                "nearest_distances": mapped_nearest_distances,
                "tau": tau,
                "reference_units": reference_units,
                "covered": mapped_nearest_distances <= tau,
                "real_real_mean_ci": (
                    float(summary["real_real_ci_low"]),
                    float(summary["real_real_ci_high"]),
                ),
            }

    all_state = runtime.get("all_comparable")
    if all_state and plot_payload:
        references: list[Unit] = all_state["reference_units"]
        distances = np.asarray(plot_payload["nearest_distances"])
        tau = float(plot_payload["tau"])
        slice_specs: list[tuple[str, Callable[[Unit], str]]] = [
            ("language", lambda unit: unit.features["language"]),
            ("table_presence", lambda unit: "present" if unit.features["table_presence"] else "absent"),
            ("hierarchy_depth", lambda unit: str(unit.features["hierarchy_depth_bin"])),
        ]
        for slice_offset, (slice_type, key_function) in enumerate(slice_specs):
            groups: dict[str, list[int]] = defaultdict(list)
            for index, unit in enumerate(references):
                groups[key_function(unit)].append(index)
            for value, indices in sorted(groups.items()):
                covered = distances[indices] <= tau
                low, high = bootstrap_proportion_ci(
                    covered, rounds, seed + 900000 + slice_offset * 10007 + len(coverage_rows)
                )
                coverage_rows.append(
                    {
                        "analysis_subset": "all_comparable",
                        "slice_type": slice_type,
                        "slice_value": value,
                        "calibration_design": "unstratified_global_tau",
                        "stratification": "none",
                        "gower_feature_set": "mapped",
                        "n_formstruct": len(all_state["form_units"]),
                        "n_srfund_clusters": len(indices),
                        "size_matched_reference_set": all_state["size"],
                        "tau": tau,
                        "tau_ci_low": np.nan,
                        "tau_ci_high": np.nan,
                        "coverage": float(np.mean(covered)),
                        "coverage_ci_low": low,
                        "coverage_ci_high": high,
                        "real_real_coverage_median": np.nan,
                        "real_real_coverage_ci_low": np.nan,
                        "real_real_coverage_ci_high": np.nan,
                        "calibrated_coverage_difference": np.nan,
                        "calibrated_coverage_ratio": np.nan,
                        "mean_nearest_distance": float(np.mean(distances[indices])),
                        "median_nearest_distance": float(np.median(distances[indices])),
                        "template_real_distance_calibration_median": np.nan,
                        "real_real_distance_median": np.nan,
                        "real_real_distance_ci_low": np.nan,
                        "real_real_distance_ci_high": np.nan,
                        "distance_calibrated_ratio": np.nan,
                        "template_real_percentile_in_real_real": np.nan,
                    }
                )
    return coverage_rows, nearest_rows, plot_payload


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def plot_distance_heatmap(
    js_rows: list[dict[str, Any]], wasserstein_rows: list[dict[str, Any]], output: Path
) -> None:
    rows: list[tuple[str, float, float, float]] = []
    for row in js_rows:
        if row["analysis_subset"] == "all_comparable" and row["calibration_design"] == "unstratified":
            rows.append(
                (
                    f"JS: {str(row['feature']).replace('_', ' ')}",
                    float(row["js_template_real"]),
                    float(row["js_real_real_median"]),
                    float(row["calibrated_ratio"]),
                )
            )
    for row in wasserstein_rows:
        if row["analysis_subset"] == "all_comparable" and row["calibration_design"] == "unstratified":
            rows.append(
                (
                    f"W1: {str(row['feature']).replace('_', ' ')}",
                    float(row["template_real_size_matched_median"]),
                    float(row["real_real_median"]),
                    float(row["calibrated_ratio"]),
                )
            )
    labels = [row[0] for row in rows]
    distances = np.asarray([[row[1], row[2]] for row in rows], dtype=float)
    ratios = np.asarray([[row[3]] for row in rows], dtype=float)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, max(4.5, 0.30 * len(rows))),
        gridspec_kw={"width_ratios": [2.0, 0.8]},
    )
    distance_image = axes[0].imshow(distances, aspect="auto", cmap="YlGnBu", vmin=0)
    axes[0].set_xticks([0, 1], ["Template-Real", "Real-Real median"])
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].set_title("Size-matched distances")
    for row_index in range(len(rows)):
        for column_index in range(2):
            axes[0].text(
                column_index,
                row_index,
                f"{distances[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
            )
    log_ratios = np.log10(np.maximum(ratios, EPSILON))
    ratio_cap = max(0.5, float(np.nanpercentile(log_ratios, 95)))
    ratio_image = axes[1].imshow(
        np.minimum(log_ratios, ratio_cap), aspect="auto", cmap="YlOrRd", vmin=0, vmax=ratio_cap
    )
    axes[1].set_xticks([0], ["log10 calibrated ratio"])
    axes[1].set_yticks(np.arange(len(labels)), [""] * len(labels))
    axes[1].set_title("Relative to Real-Real")
    for row_index in range(len(rows)):
        axes[1].text(0, row_index, f"{ratios[row_index, 0]:.2g}", ha="center", va="center", fontsize=6.5)
    figure.colorbar(distance_image, ax=axes[0], fraction=0.025, pad=0.02, label="Distance")
    figure.colorbar(ratio_image, ax=axes[1], fraction=0.06, pad=0.04, label="log10 ratio")
    figure.suptitle("FormStruct-Bench vs. deduplicated SRFUND", y=1.002, fontsize=10)
    save_figure(figure, output, "representativeness_distance_heatmap")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)
    probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, probabilities


def plot_continuous_ecdf(
    form_units: list[Unit], reference_units: list[Unit], output: Path
) -> None:
    features = [
        ("num_entities", "Entities"),
        ("num_relation_links", "Relation links"),
        ("max_hierarchy_depth", "Maximum hierarchy depth"),
        ("mean_branching_factor", "Mean branching factor"),
        ("num_item_tables", "Item tables"),
        ("spatial_layout_density", "Spatial layout density"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    colors = {"FormStruct": "#2878B5", "SRFUND": "#D95F02"}
    for axis, (feature, label) in zip(axes.flat, features, strict=True):
        for name, units in (("FormStruct", form_units), ("SRFUND", reference_units)):
            values = numeric_values(units, feature)
            x, y = ecdf(values)
            axis.step(x, y, where="post", label=f"{name} (n={len(values)})", color=colors[name], linewidth=1.5)
        axis.set_xlabel(label)
        axis.set_ylabel("ECDF")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(frameon=False)
    figure.suptitle("Template/layout-cluster structural feature distributions", y=1.01, fontsize=10)
    figure.tight_layout()
    save_figure(figure, output, "continuous_feature_ecdf")


def plot_coverage_by_slice(coverage_rows: list[dict[str, Any]], output: Path) -> None:
    slice_rows = [
        row
        for row in coverage_rows
        if row["calibration_design"] == "unstratified_global_tau"
        and row["slice_type"] in {"language", "table_presence", "hierarchy_depth"}
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.6, 1.0, 1.2]})
    titles = {
        "language": "Language",
        "table_presence": "Table presence",
        "hierarchy_depth": "Hierarchy depth",
    }
    colors = {"language": "#2878B5", "table_presence": "#3D8F70", "hierarchy_depth": "#C65353"}
    for axis, slice_type in zip(axes, ("language", "table_presence", "hierarchy_depth"), strict=True):
        rows = sorted(
            [row for row in slice_rows if row["slice_type"] == slice_type],
            key=lambda row: (float(row["coverage"]), str(row["slice_value"])),
        )
        positions = np.arange(len(rows))
        estimates = np.asarray([float(row["coverage"]) for row in rows])
        lows = np.asarray([float(row["coverage_ci_low"]) for row in rows])
        highs = np.asarray([float(row["coverage_ci_high"]) for row in rows])
        errors = np.vstack([estimates - lows, highs - estimates])
        axis.barh(positions, estimates, color=colors[slice_type], alpha=0.82, height=0.65)
        axis.errorbar(estimates, positions, xerr=errors, fmt="none", ecolor="#222222", capsize=2, linewidth=0.8)
        axis.set_yticks(positions, [str(row["slice_value"]).replace("_", " ") for row in rows])
        axis.set_xlim(0, 1.03)
        axis.set_xlabel("Coverage at calibrated tau")
        axis.set_title(titles[slice_type])
        axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    figure.suptitle("Structural coverage by reference slice (cluster bootstrap 95% CI)", y=1.01, fontsize=10)
    figure.tight_layout()
    save_figure(figure, output, "coverage_by_slice")


def plot_nearest_neighbor_gap(plot_payload: dict[str, Any], output: Path) -> None:
    distances = np.asarray(plot_payload["nearest_distances"], dtype=float)
    tau = float(plot_payload["tau"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].hist(distances, bins=24, color="#2878B5", alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[0].axvline(tau, color="#C65353", linestyle="--", linewidth=1.5, label=f"Real-Real tau={tau:.3f}")
    axes[0].set_xlabel("Nearest FormStruct Gower distance")
    axes[0].set_ylabel("SRFUND layout clusters")
    axes[0].legend(frameon=False)
    x, y = ecdf(distances)
    axes[1].step(x, y, where="post", color="#2878B5", linewidth=1.6)
    axes[1].axvline(tau, color="#C65353", linestyle="--", linewidth=1.5)
    axes[1].axhline(0.95, color="#777777", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel("Nearest FormStruct Gower distance")
    axes[1].set_ylabel("ECDF")
    axes[1].grid(alpha=0.18, linewidth=0.5)
    figure.suptitle("SRFUND-to-FormStruct nearest-neighbor gap", y=1.02, fontsize=10)
    figure.tight_layout()
    save_figure(figure, output, "nearest_neighbor_gap")


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_representativeness_table(path: Path, coverage_rows: list[dict[str, Any]]) -> None:
    preferred = [
        "all_comparable",
        "shared_languages",
        "table_present",
        "hierarchy_shallow_0_2",
        "hierarchy_medium_3_4",
        "hierarchy_deep_5_plus",
    ]
    by_subset = {
        row["analysis_subset"]: row
        for row in coverage_rows
        if row["slice_type"] == "analysis_subset"
        and row["calibration_design"] == "unstratified"
        and row.get("gower_feature_set", "mapped") == "mapped"
    }
    rows = [by_subset[name] for name in preferred if name in by_subset]
    display_labels = {
        "all_comparable": "All comparable",
        "shared_languages": "Shared languages",
        "table_present": "Table present",
        "hierarchy_shallow_0_2": "Hierarchy: shallow (0--2)",
        "hierarchy_medium_3_4": "Hierarchy: medium (3--4)",
        "hierarchy_deep_5_plus": r"Hierarchy: deep ($\geq 5$)",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Size-matched structural representativeness against deduplicated, non-overlapping SRFUND layout clusters. The table uses the mapped direct+conditional Gower configuration; each slice has its own calibrated tau. Real--Real and coverage intervals are empirical 95\% intervals over layout-cluster resampling.}",
        r"\label{tab:representativeness}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Subset & matched $n_T/n_R$ & T--R $d_{\min}$ & R--R $d_{\min}$ & Ratio & Coverage \\",
        r"\midrule",
    ]
    for row in rows:
        raw_label = display_labels.get(str(row["analysis_subset"]), str(row["analysis_subset"]))
        label = raw_label if "$" in raw_label else latex_escape(raw_label)
        lines.append(
            f"{label} & {int(row['n_formstruct'])}/{int(row['size_matched_reference_set'])} & "
            f"{float(row['template_real_distance_calibration_median']):.3f} & "
            f"{float(row['real_real_distance_median']):.3f} "
            f"[{float(row['real_real_distance_ci_low']):.3f}, {float(row['real_real_distance_ci_high']):.3f}] & "
            f"{float(row['distance_calibrated_ratio']):.2f} & "
            f"{100 * float(row['coverage']):.1f} [{100 * float(row['coverage_ci_low']):.1f}, {100 * float(row['coverage_ci_high']):.1f}] "
            + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_directory_tree(path: Path, srfund_root: Path) -> None:
    lines = ["dataset/"]
    for directory in ("images", "instance_annotation", "relation_annotation"):
        lines.append(f"  {directory}/")
        current = srfund_root / directory
        if directory == "images":
            for language_dir in sorted(item for item in current.iterdir() if item.is_dir()):
                files = sorted(item.name for item in language_dir.iterdir() if item.is_file())
                extensions = dict(sorted(Counter(Path(name).suffix.lower() for name in files).items()))
                lines.append(
                    f"    {language_dir.name}/ ({len(files)} files; extensions={json.dumps(extensions, sort_keys=True)})"
                )
        else:
            for file_path in sorted(item for item in current.iterdir() if item.is_file()):
                payload = read_json(file_path)
                lines.append(f"    {file_path.name} ({len(payload)} document records)")
    item_tables = read_json(srfund_root / "item_table_info.json")
    lines.append(
        f"  item_table_info.json ({len(item_tables)} document records; "
        f"{sum(len(records) for records in item_tables.values())} tables)"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_mapping(path: Path) -> None:
    text = """# FormStruct-Bench to SRFUND feature mapping

The mapping was defined after inspecting both annotation schemas. A shared name is not treated as evidence of shared semantics. All distances use one FormStruct template or one deduplicated SRFUND layout cluster as the statistical unit.

## A. Directly comparable shared structure

| Feature | FormStruct extraction | SRFUND extraction | Use |
|---|---|---|---|
| language code | Verified template metadata/language naming | Dataset language directory | Main categorical analysis; language mismatch is also conditioned out in the shared-language subset |
| num relation links | Directed parent-child and key-value edges in the annotation tree | Directed edges parsed from the nested relation annotation | Main analysis; both count untyped structural adjacency |
| maximum hierarchy depth | Maximum entity-node depth, including answer leaves | Maximum entity-node depth, including answer leaves | Main analysis with the same root-depth convention |
| mean branching factor | Mean out-degree among nodes with children | Mean out-degree among nodes with children | Main analysis |
| num root children | Distinct top-level field entities | Distinct top-level relation entities | Main analysis |
| relation density | Unique relation edges divided by graph entities | Unique relation edges divided by graph entities | Main analysis |
| normalized entity bounding-box statistics | Key/value boxes normalized by page size | Entity boxes normalized by page size | Main analysis |

## B. Comparable only through an explicit conditional mapping

| Feature | Mapping and caveat | Use |
|---|---|---|
| script | Both datasets are assigned a deterministic script family from language; page text can contain mixed scripts, so this is derived rather than a page-level annotation. | Supplemental JS/Gower, marked `derived_conditional` |
| entity counts and entity-type composition | FormStruct group-only keys map to `header`, other keys to `question`, and value boxes to `answer`; FormStruct has no `other` counterpart. SRFUND uses native labels. | Supplemental JS/Gower/W1, marked `conditional` |
| writing direction | Neither schema stores a page-level reading-direction annotation. Direction is derived conservatively from verified language/script (`RTL` only for Arabic; otherwise horizontal `LTR`). | Supplemental JS/Gower, marked `conditional` |
| item-table count and table presence | FormStruct explicit `table_region_count` maps to SRFUND `item_table_info` records. Generic semantic regions and line-item groups are not counted as tables. | Supplemental main-distance group, marked `conditional` |
| spatial/layout density | Union occupancy of normalized entity boxes in both datasets. Different entity segmentation can still affect it. | Supplemental W1/Gower, marked `conditional` |
| semantic region vs entity group | Only entity boxes are compared; region membership is not assumed equivalent. | Region counts excluded from cross-dataset distances |
| line-item group vs item table/item | FormStruct line-item groups are column/group annotations, while SRFUND item values are row records. | Not used as an item-count mapping |

## C. Not directly comparable or FormStruct-specific

| Feature | Reason |
|---|---|
| num words, num text lines, page text density | SRFUND has full OCR words/lines, including filled values; FormStruct has semantic labels and blank field boxes rather than equivalent page OCR. Values are retained where observed but excluded from distances. |
| num table items | SRFUND has explicit item-value records; FormStruct's current metadata reports no semantically equivalent record count. |
| relation type distribution | FormStruct has typed relations; SRFUND's nested hierarchy does not expose the same relation ontology. |
| semantic sections and semantic regions | No equivalent SRFUND annotation unit. |
| region-local grid topology, row/column coordinates, row/column spans | No equivalent SRFUND annotation. Existing FormStruct metadata also lacks reliable explicit coordinates for every cell. |
| widget groups/states, checkbox, radio, character box, signature field | No equivalent SRFUND annotation. |
| typed key-to-field, field-to-widget, and section-membership relations | SRFUND relations are not typed with these semantics. |
| domain | SRFUND contains no reliable domain labels. No domain labels are inferred. |

These category-C features describe FormStruct annotation richness only. They cannot validate population fidelity relative to SRFUND.
"""
    path.write_text(text, encoding="utf-8")


def write_dedup_report(
    path: Path,
    archive_path: Path,
    archive_sha256: str,
    inventory: dict[str, Any],
    sensitivity: dict[str, Any],
    cross_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> None:
    confirmed = [row for row in cross_rows if row["dedup_status"] == "confirmed_overlap"]
    suspected = [row for row in cross_rows if row["dedup_status"] == "suspected_overlap"]
    unresolved_aliases = [
        row
        for row in cross_rows
        if row["dedup_status"] == "candidate_only" and row["source_alias_match"]
    ]
    exact_sources = sorted(
        {
            f"{row['srfund_source_id']} -> {row['formstruct_template_id']}"
            for row in confirmed
            if row["exact_source_id"]
        }
    )
    text = f"""# SRFUND deduplication and overlap report

## Source integrity

- Download URL: `https://drive.google.com/file/d/1cx1WhcFiDJz8WIzwWEs_D8Ml6KP8W9wB/view`
- Archive: `{archive_path}`
- Archive bytes: {archive_path.stat().st_size:,}
- SHA-256: `{archive_sha256}`
- ZIP integrity test: passed before extraction
- Extracted raw files were read only; derived files are written outside `raw/srfund/`.

The compact complete directory inventory is in `srfund_directory_tree.txt`. SRFUND contains {inventory['total_images']:,} images, eight languages, matching instance/relation annotation keys, {inventory['item_table_pages']:,} pages with explicit item-table metadata, and no document-level metadata or domain labels.

## Annotation format

- `instance_annotation/<language>.json`: image filename to entity list. Entities carry `id`, `box`, `text`, OCR `lines`/`words`, native `label` (`header`, `question`, `answer`, `other`), and `linking`.
- `relation_annotation/<language>.json`: image filename to a heterogeneous nested dict/list hierarchy of numeric entity references plus special keys such as `other`, `note`, and `item_table_*`.
- `item_table_info.json`: image filename to table records with table boxes, member entity IDs, item-value IDs, and item-value boxes.

## Independent layout clustering

The archive has no reliable original-document ID. Therefore no filename-only document grouping was invented. Exact file SHA-256, normalized image hash, rotation-aware 64-bit pHash, normalized entity-box occupancy, OCR character shingles, aspect ratio, and entity-count agreement were used together. Candidate retrieval includes all pHash-near pairs and the highest-layout-neighbour candidates; sensitivity counts are conditional on that retrieval graph, not an exhaustive proof over every page pair. Strong duplicate edges require either an exact hash or one of the multi-signal rules recorded in the analysis manifest. Complete-link-style cohesion checks prevent transitive chaining through visually generic forms.

Sensitivity counts:

- Raw pages: {sensitivity['page_count']:,}
- Exact/near-duplicate candidate pairs: {sensitivity['candidate_pair_count']:,}
- Accepted strong edges: {sensitivity['accepted_edge_count']:,}
- Primary independent layout clusters: {sensitivity['primary_cluster_count']:,}
- Strict-threshold components: {sensitivity['strict_component_count']:,}
- Relaxed-threshold components: {sensitivity['relaxed_component_count']:,}
- Multi-page clusters: {sensitivity['multi_page_clusters']:,}
- Largest cluster: {sensitivity['largest_cluster_size']} pages
- Proposed merges rejected by cluster cohesion: {sensitivity['cohesion_rejected_merges']:,}

Every cluster is represented by its multi-signal medoid rather than by aggregating its pages as independent samples.

## Cross-dataset overlap

Source provenance is recovered by matching FormStruct source-annotation IDs to `metadata-test/`. Candidate generation combines exact and normalized source IDs with image, OCR, and layout signals. A train-split language/index alias is treated as provenance evidence; validation aliases are non-unique and require independent visual/text confirmation. Exact byte hashes alone found no cross-dataset match because source images were resized or re-encoded.

- Confirmed page-pair matches: {len(confirmed):,}
- Suspected page-pair matches: {len(suspected):,}
- Unresolved non-unique language/index alias candidates retained for audit: {len(unresolved_aliases):,}
- Excluded SRFUND layout clusters: {len(excluded_rows):,}
- Exact source-ID matches: {', '.join(exact_sources) if exact_sources else 'none'}

All clusters containing a confirmed or independently corroborated suspected overlap are excluded from every primary representativeness result. A language/index alias by itself is not a globally unique document ID: alias-only rows with weak visual/OCR agreement remain candidate-only rather than being falsely excluded. `dedup_candidates.csv` retains all audited candidates and signal values; `dedup_excluded.csv` gives the excluded cluster set. Excluding every unresolved alias as a sensitivity analysis would remove an additional {len(unresolved_aliases):,} page-pair candidates (the primary analysis does not do so). `srfund_near_duplicate_candidates.csv` records within-SRFUND clustering evidence.

## Boundaries

Clustering is evidence-based but cannot reconstruct missing publisher document IDs. The strict/primary/relaxed sensitivity counts expose threshold dependence. Potential domain overlap cannot be tested because SRFUND has no domain labels; no domain labels were fabricated.
"""
    path.write_text(text, encoding="utf-8")


def dependency_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "opencv-python-headless",
        "opencv-python",
        "matplotlib",
        "Pillow",
        "PyMuPDF",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    versions["cv2_runtime"] = getattr(cv2, "__version__", "unknown")
    return versions


def format_ci(estimate: float, low: float, high: float, scale: float = 1.0) -> str:
    return f"{estimate * scale:.3f} [{low * scale:.3f}, {high * scale:.3f}]"


def write_representativeness_summary(
    path: Path,
    archive_sha256: str,
    srfund_inventory: dict[str, Any],
    form_inventory: dict[str, Any],
    sensitivity: dict[str, Any],
    reference_units: list[Unit],
    form_units: list[Unit],
    cross_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    js_rows: list[dict[str, Any]],
    wasserstein_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    rounds: int,
    seed: int,
) -> None:
    primary_coverage = {
        row["analysis_subset"]: row
        for row in coverage_rows
        if row["slice_type"] == "analysis_subset"
        and row["calibration_design"] == "unstratified"
        and row.get("gower_feature_set", "mapped") == "mapped"
    }
    direct_coverage = {
        row["analysis_subset"]: row
        for row in coverage_rows
        if row["slice_type"] == "analysis_subset"
        and row["calibration_design"] == "unstratified"
        and row.get("gower_feature_set") == "direct_only"
    }
    all_coverage = primary_coverage["all_comparable"]
    all_direct_coverage = direct_coverage["all_comparable"]
    shared_coverage = primary_coverage.get("shared_languages")
    primary_js = [
        row
        for row in js_rows
        if row["analysis_subset"] == "all_comparable" and row["calibration_design"] == "unstratified"
    ]
    shared_js = [
        row
        for row in js_rows
        if row["analysis_subset"] == "shared_languages" and row["calibration_design"] == "unstratified"
    ]
    primary_w1 = [
        row
        for row in wasserstein_rows
        if row["analysis_subset"] == "all_comparable" and row["calibration_design"] == "unstratified"
    ]
    shared_w1 = [
        row
        for row in wasserstein_rows
        if row["analysis_subset"] == "shared_languages" and row["calibration_design"] == "unstratified"
    ]
    js_above = sum(float(row["template_real_percentile_in_real_real"]) > 95 for row in primary_js)
    shared_js_above = sum(float(row["template_real_percentile_in_real_real"]) > 95 for row in shared_js)
    w1_above = sum(float(row["template_real_percentile_in_real_real"]) > 95 for row in primary_w1)
    shared_w1_above = sum(float(row["template_real_percentile_in_real_real"]) > 95 for row in shared_w1)
    confirmed = [row for row in cross_rows if row["dedup_status"] == "confirmed_overlap"]
    suspected = [row for row in cross_rows if row["dedup_status"] == "suspected_overlap"]
    unresolved_aliases = [
        row
        for row in cross_rows
        if row["dedup_status"] == "candidate_only" and row["source_alias_match"]
    ]
    included_reference = [unit for unit in reference_units if unit.dedup_status == "included"]
    form_languages = sorted({unit.language_code for unit in form_units})
    reference_languages = sorted({unit.language_code for unit in included_reference})
    shared_languages = sorted(set(form_languages) & set(reference_languages))
    form_only = sorted(set(form_languages) - set(reference_languages))
    reference_only = sorted(set(reference_languages) - set(form_languages))
    language_slices = sorted(
        [
            row
            for row in coverage_rows
            if row["slice_type"] == "language" and row["calibration_design"] == "unstratified_global_tau"
        ],
        key=lambda row: float(row["coverage"]),
    )
    structure_slices = [
        row
        for row in coverage_rows
        if row["slice_type"] in {"table_presence", "hierarchy_depth"}
        and row["calibration_design"] == "unstratified_global_tau"
    ]
    lowest_language_gap = (
        f"{language_slices[0]['slice_value']} at {100 * float(language_slices[0]['coverage']):.1f}%"
        if language_slices
        else "unavailable"
    )
    form_table_templates = sum(bool(unit.features.get("table_presence")) for unit in form_units)
    form_selection_templates = sum(
        int(unit.features.get("checkbox_field_count") or 0)
        + int(unit.features.get("radio_field_count") or 0)
        > 0
        for unit in form_units
    )
    form_signature_templates = sum(int(unit.features.get("signature_field_count") or 0) > 0 for unit in form_units)

    slice_lines = [
        "| Slice | Clusters | Coverage (95% CI) | Mean nearest distance |",
        "|---|---:|---:|---:|",
    ]
    for row in [*language_slices, *structure_slices]:
        slice_lines.append(
            f"| {row['slice_type']}: {row['slice_value']} | {int(row['n_srfund_clusters'])} | "
            f"{100 * float(row['coverage']):.1f}% [{100 * float(row['coverage_ci_low']):.1f}, "
            f"{100 * float(row['coverage_ci_high']):.1f}] | {float(row['mean_nearest_distance']):.3f} |"
        )

    shared_sentence = "not estimable"
    if shared_coverage:
        shared_sentence = (
            f"{100 * float(shared_coverage['coverage']):.1f}% "
            f"(95% CI {100 * float(shared_coverage['coverage_ci_low']):.1f}-"
            f"{100 * float(shared_coverage['coverage_ci_high']):.1f}%)"
        )
    text = f"""# FormStruct-Bench real-template structural representativeness

## 1. Data and statistical unit

FormStruct-Bench contributes exactly {len(form_units)} selected, real, manually structured templates and {form_inventory['instances']:,} synthetic instances. The verified split is {form_inventory['templates_by_split']}. Synthetic refers to filled content/schema variation/visual perturbation; the template skeletons are real. SRFUND contributes {srfund_inventory['total_images']:,} pages in eight languages, but pages are not treated as independent templates. Rotation-aware image/OCR/layout clustering produces {sensitivity['primary_cluster_count']:,} independent layout clusters, of which {len(included_reference):,} remain after overlap exclusion. Every analysis row and bootstrap draw is a template or layout cluster. FormStruct source metadata is unavailable for {len(form_inventory.get('missing_source_metadata', []))} templates ({', '.join(form_inventory.get('missing_source_metadata', [])) or 'none'}), so those provenance checks rely on image/OCR/layout evidence.

## 2. SRFUND download and integrity

The supplied Google Drive object was downloaded to `raw/srfund/srfund_download.bin` ({Path('raw/srfund/srfund_download.bin').stat().st_size:,} bytes), tested as a ZIP, and extracted without modifying archive contents. SHA-256 is `{archive_sha256}`. Image, instance-annotation, and relation-annotation key sets agree for all {srfund_inventory['total_images']:,} pages. There are {srfund_inventory['item_table_pages']:,} item-table pages and {srfund_inventory['item_tables']:,} table records. The exact directory inventory is in `srfund_directory_tree.txt`.

## 3. Deduplication and potential leakage

SRFUND lacks document metadata and reliable original-document IDs, so clustering uses exact file and normalized-image hashes, rotation-aware pHash, OCR shingles, normalized entity-box occupancy, entity-count agreement, and aspect ratio. Candidate retrieval examines all pHash-near pairs plus the highest-layout-neighbour candidates; the sensitivity counts are therefore evidence-based conditional counts rather than a proof that every possible pair was compared. Strict/primary/relaxed sensitivity yields {sensitivity['strict_component_count']:,}/{sensitivity['primary_cluster_count']:,}/{sensitivity['relaxed_component_count']:,} clusters. Cross-dataset provenance and similarity checks find {len(confirmed):,} confirmed and {len(suspected):,} corroborated suspected page-pair matches, affecting {len(excluded_rows):,} SRFUND clusters. {len(unresolved_aliases):,} additional non-unique language/index aliases remain candidate-only because their visual/OCR evidence is weak. Every confirmed/corroborated cluster is excluded from the main analysis. Notably, resizing/re-encoding makes exact byte hashes insufficient; source-ID evidence detects real leakage that hashes miss.

## 4. Feature mapping and non-comparable features

`feature_mapping.md` records the schema-derived mapping. Untyped relation links, aligned hierarchy statistics, language code, and normalized bbox summaries are direct. Script and writing direction are derived from language/script conventions rather than page-level annotations. Entity-type composition, table presence/count, and spatial density require explicit conditional mappings and are labeled accordingly. FormStruct semantic regions, grid topology, spans, widget state/types, signatures, and typed relations are annotation richness, not SRFUND-validated representativeness. Full OCR word/line counts, page text density, table-item records, relation-type distributions, and domain are not directly comparable and are excluded rather than imputed.

## 5. Jensen-Shannon results

JS divergence uses the union of categories, additive smoothing `{EPSILON:g}`, base-2 logarithms, and equal template/cluster contribution. The primary shared-language calibration has {shared_js_above} of {len(shared_js)} categorical features above the Real-Real 95th percentile. The all-comparable calibration, which also reflects language-composition differences, has {js_above} of {len(primary_js)}. Detailed distributions, Real-Real intervals, calibrated differences/ratios, and empirical percentiles are in `js_results.csv`. Language JS is interpreted as corpus sampling composition, not as a world-language prior.

## 6. Normalized Wasserstein results

Continuous W1 is normalized by SRFUND's Q05-Q95 span plus epsilon. The primary shared-language calibration has {shared_w1_above} of {len(shared_w1)} analyzed continuous features above the Real-Real 95th percentile; the all-comparable calibration has {w1_above} of {len(primary_w1)}. `wasserstein_results.csv` reports raw W1, normalized W1, cluster-bootstrap 95% CIs, size-matched Real-Real calibration, sample sizes, and empirical percentiles. Word/line and table-item counts are absent because their units are not equivalent.

## 7. Structural coverage

Group-balanced Gower distance gives equal weight to hierarchy, entity composition, table structure, relation structure, spatial layout, and language/script; correlated features are averaged within groups before groups are averaged. The primary shared-language mapped coverage is {shared_sentence}. The all-comparable mapped coverage is {100 * float(all_coverage['coverage']):.1f}% (95% CI {100 * float(all_coverage['coverage_ci_low']):.1f}-{100 * float(all_coverage['coverage_ci_high']):.1f}%) at tau={float(all_coverage['tau']):.3f}. A direct-only sensitivity (hierarchy, relation, bbox layout, and language code) gives {100 * float(all_direct_coverage['coverage']):.1f}% (95% CI {100 * float(all_direct_coverage['coverage_ci_low']):.1f}-{100 * float(all_direct_coverage['coverage_ci_high']):.1f}%) for all-comparable. All coverage estimates come from the same size-matched repetitions: {int(all_coverage['size_matched_reference_set'])} FormStruct/reference units for the all-comparable set; each smaller slice uses its own matched size.

## 8. Real-Real calibration

For each subset and each of unstratified/stratified designs, {rounds:,} seeded repetitions draw two disjoint SRFUND sets. The main all-comparable comparison uses 70 vs 70 when at least 140 reference clusters are available; smaller slices use `n=min(|T|, floor(|R|/2))` and sample the same number of FormStruct templates. Tau is the 95th percentile of Real-B-to-Real-A nearest distances. Calibrated difference is `D_TemplateReal - D_RealReal`; calibrated ratio is `D_TemplateReal/(D_RealReal+epsilon)`. No arbitrary ratio acceptance threshold is used. Seed: {seed}.

## 9. Language and structural slices

Shared languages are {', '.join(shared_languages)}. FormStruct-only language coverage is {', '.join(form_only) or 'none'}; SRFUND-only coverage is {', '.join(reference_only) or 'none'}. Arabic/RTL and Chinese-English templates are reported descriptively rather than forced into a mismatched reference language. Coverage slices are:

{chr(10).join(slice_lines)}

The slice table above uses one global all-comparable tau and the fixed 70-template set. The LaTeX table reports the separate within-slice calibrated tau for each named subset; these are intentionally different estimands and must not be compared as if they used one threshold. `nearest_template_matches.csv` retains every included reference cluster as a descriptive full-reference map; its `covered_at_tau` flag is not the size-matched aggregate coverage estimate reported above.

## 10. FormStruct structural coverage strengths

The benchmark includes all {len(form_units)} independent real template skeletons, {len(form_languages)} language categories, {form_table_templates} templates with explicit table regions, and additional annotation not present in SRFUND. Group-balanced nearest-neighbor results quantify the reference configurations covered without letting entity/relation counts dominate. FormStruct-specific checkbox/radio controls occur in {form_selection_templates} templates; the current extracted ontology records signature-typed fields in {form_signature_templates} templates. These are annotation-scope observations, not evidence about SRFUND prevalence.

## 11. Observed coverage gaps

The lowest global language-slice coverage is {lowest_language_gap}. `nearest_template_matches.csv` identifies each non-overlapping reference cluster's closest template and its largest feature differences. Gaps can reflect benchmark sampling, language mismatch, domain mismatch, intentional difficulty balancing, ontology mismatch, or SRFUND's own boundary. They are not evidence that the benchmark is invalid.

## 12. Recommended paper claim

The evidence does **not** justify `population-representative`. Multiple population-fidelity distances can exceed natural Real-Real fluctuation, SRFUND has a balanced and bounded language composition, domain labels are unavailable, and confirmed source overlap had to be removed. The defensible claim is: **"The benchmark is coverage-oriented rather than prevalence-matched."** Report the calibrated distances and coverage interval rather than a binary representativeness threshold.

## 13. Limitations

SRFUND lacks document IDs, domain metadata, and a relation ontology aligned to FormStruct. Near-duplicate clustering is therefore multi-signal and threshold-sensitive, with candidate-retrieval limits and sensitivity counts reported. Cluster medoids approximate layout templates but cannot recover unavailable publisher provenance; the {len(form_inventory.get('missing_source_metadata', []))} FormStruct templates without source metadata are especially dependent on visual/OCR checks. Only one external reference collection is used. These results validate template-level shared structure only; they do not validate the content, handwriting, blank-field behavior, schema-change frequency, scan/camera noise, or perturbation prevalence of the {form_inventory['instances']:,} synthetic instances.

To validate those synthetic instances, collect independent real filled forms for the same template families and compare field-population/missingness patterns, value types and lengths, handwriting/printed mixtures, checkbox/radio states, schema optionality, document damage, capture geometry, blur/noise/color, and within-template covariance. Use real-template-disjoint sampling and the same Real-Real calibration.

## 14. Reproduction

```bash
.venv/bin/python -m benchmark_stats.run_representativeness_analysis \\
  --srfund-root raw/srfund/extracted/dataset \\
  --srfund-archive raw/srfund/srfund_download.bin \\
  --layout-dir newdataset-layout \\
  --formstruct-image-dir new-dataset \\
  --template-root FormTSR/datasets \\
  --source-metadata-dir metadata-test \\
  --split-assignments outputs/dataset_splits/template_stratified_seed42/template_assignments.csv \\
  --bootstrap-rounds {rounds} --seed {seed} \\
  --output outputs/representativeness
```

Exact runtime versions, thresholds, the SRFUND archive checksum, input paths, and command arguments are in `analysis_manifest.json`; machine checks are in `validation_report.json`. The four PDFs were rasterized after generation and checked for nonblank, finite-size pages.
"""
    path.write_text(text, encoding="utf-8")


def render_and_validate_pdfs(output: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for post-generation PDF render validation") from exc
    render_dir = output / "render_checks"
    render_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for filename in (
        "representativeness_distance_heatmap.pdf",
        "continuous_feature_ecdf.pdf",
        "coverage_by_slice.pdf",
        "nearest_neighbor_gap.pdf",
    ):
        document = fitz.open(output / filename)
        if document.page_count < 1:
            raise ValueError(f"PDF has no pages: {filename}")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        render_path = render_dir / f"{Path(filename).stem}.png"
        pixmap.save(render_path)
        image = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"failed to decode rendered PDF: {filename}")
        ink_ratio = float(np.mean(image < 250))
        if image.shape[0] < 300 or image.shape[1] < 300 or ink_ratio < 0.002:
            raise ValueError(f"blank or undersized rendered PDF: {filename}")
        results.append(
            {
                "pdf": filename,
                "pages": document.page_count,
                "render_width": int(image.shape[1]),
                "render_height": int(image.shape[0]),
                "nonwhite_pixel_ratio": ink_ratio,
                "render_path": str(render_path),
            }
        )
        document.close()
    return results


def validate_outputs(output: Path, render_pdfs: bool) -> dict[str, Any]:
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    empty = [name for name in REQUIRED_OUTPUTS if (output / name).is_file() and (output / name).stat().st_size == 0]
    if missing or empty:
        raise ValueError(f"required output validation failed: missing={missing}, empty={empty}")
    form_frame = pd.read_csv(output / "formstruct_template_features.csv")
    reference_frame = pd.read_csv(output / "srfund_reference_features.csv")
    js_frame = pd.read_csv(output / "js_results.csv")
    wasserstein_frame = pd.read_csv(output / "wasserstein_results.csv")
    coverage_frame = pd.read_csv(output / "coverage_results.csv")
    calibration_frame = pd.read_csv(output / "calibration_bootstrap.csv")
    if len(form_frame) != 70 or form_frame["template_id"].nunique() != 70:
        raise ValueError("FormStruct feature table does not contain 70 unique templates")
    if not len(reference_frame) or reference_frame["srfund_cluster_id"].nunique() != len(reference_frame):
        raise ValueError("SRFUND feature table is empty or has duplicate cluster rows")
    for column in ("full_sample_js", "js_template_real", "js_real_real_median"):
        values = js_frame[column].dropna()
        if not values.between(0, 1).all():
            raise ValueError(f"JS range check failed for {column}")
    if (wasserstein_frame["raw_w1"].dropna() < 0).any() or (
        wasserstein_frame["normalized_w1"].dropna() < 0
    ).any():
        raise ValueError("Wasserstein distances must be non-negative")
    for column in ("coverage", "coverage_ci_low", "coverage_ci_high", "tau"):
        values = coverage_frame[column].dropna()
        if not values.between(0, 1).all():
            raise ValueError(f"coverage range check failed for {column}")
    if int(calibration_frame["round"].max()) + 1 < 1000:
        raise ValueError("calibration contains fewer than 1000 rounds")
    for filename in REQUIRED_OUTPUTS:
        if filename.endswith(".pdf") and not (output / filename).read_bytes().startswith(b"%PDF"):
            raise ValueError(f"invalid PDF signature: {filename}")
    render_results = render_and_validate_pdfs(output) if render_pdfs else []
    result = {
        "status": "passed",
        "required_outputs": len(REQUIRED_OUTPUTS),
        "formstruct_templates": len(form_frame),
        "srfund_clusters": len(reference_frame),
        "srfund_included_clusters": int((reference_frame["dedup_status"] == "included").sum()),
        "js_rows": len(js_frame),
        "wasserstein_rows": len(wasserstein_frame),
        "coverage_rows": len(coverage_frame),
        "calibration_rows": len(calibration_frame),
        "pdf_render_validation": render_results,
    }
    (output / "validation_report.json").write_text(
        json.dumps(json_ready(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def build_manifest(
    args: argparse.Namespace,
    archive_sha256: str,
    srfund_inventory: dict[str, Any],
    form_inventory: dict[str, Any],
    sensitivity: dict[str, Any],
    reference_units: list[Unit],
    excluded_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "analysis": "FormStruct-Bench real-template structural representativeness against SRFUND",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "bootstrap_rounds": args.bootstrap_rounds,
        "input": {
            "srfund_url": "https://drive.google.com/file/d/1cx1WhcFiDJz8WIzwWEs_D8Ml6KP8W9wB/view",
            "srfund_archive": str(args.srfund_archive),
            "srfund_archive_bytes": Path(args.srfund_archive).stat().st_size,
            "srfund_archive_sha256": archive_sha256,
            "srfund_archive_zip_integrity": "passed (ZipFile.testzip)",
            "srfund_root": str(args.srfund_root),
            "layout_dir": str(args.layout_dir),
            "formstruct_image_dir": str(args.formstruct_image_dir),
            "template_root": str(args.template_root),
            "source_metadata_dir": str(args.source_metadata_dir),
            "split_assignments": str(args.split_assignments),
        },
        "verified_inventory": {
            "srfund": srfund_inventory,
            "formstruct": form_inventory,
            "srfund_layout_clusters": len(reference_units),
            "excluded_overlap_clusters": len(excluded_rows),
            "included_reference_clusters": sum(unit.dedup_status == "included" for unit in reference_units),
        },
        "deduplication": {
            "signals": [
                "file_sha256",
                "normalized_image_sha256",
                "rotation_aware_64_bit_phash",
                "normalized_entity_bbox_occupancy",
                "ocr_character_3gram_jaccard",
                "ocr_char_3_to_5gram_tfidf_cosine_for_cross_dataset_audit",
                "entity_count_ratio",
                "aspect_ratio",
                "source_id_and_normalized_source_alias",
            ],
            "primary_edge_rules": [
                "exact file or normalized-image hash",
                "pHash <= 4 and layout cosine >= 0.90",
                "pHash <= 8 and layout cosine >= 0.95 and OCR Jaccard >= 0.45",
                "layout cosine >= 0.985 and OCR Jaccard >= 0.82 and entity-count ratio >= 0.75",
            ],
            "cluster_cohesion": "all cross-component pairs must pass a looser multi-signal cohesion rule",
            "candidate_retrieval": "all rotation-aware pHash pairs within distance 20 plus top-25 layout-neighbour candidates using all rotations; sensitivity counts are conditional on this retrieval graph",
            "cross_dataset_alias_rule": "train-split language/index aliases are provenance-confirmed; validation aliases require independent visual/text confirmation",
            "sensitivity": sensitivity,
        },
        "statistics": {
            "js": {
                "log_base": 2,
                "category_support": "union",
                "zero_frequency_additive_smoothing": EPSILON,
                "entity_distribution_weighting": "equal template/layout-cluster contribution",
            },
            "wasserstein": {
                "order": 1,
                "normalization": "SRFUND Q95-Q05 span plus epsilon",
                "epsilon": EPSILON,
            },
            "gower": {
                "continuous_normalization": "range of non-overlapping SRFUND subset",
                "categorical": "0/1 mismatch",
                "groups": GOWER_GROUPS,
                "direct_only_groups": DIRECT_GOWER_GROUPS,
                "aggregation": "mean within group, then equal mean across groups",
            },
            "coverage_tau": "per-round Q95 of Real-B nearest distance to disjoint Real-A",
            "calibration_designs": ["unstratified", "stratified"],
            "size_matching": "n=min(70, |T|, floor(|R|/2)); both source sets contain n units",
            "confidence_intervals": "2.5th and 97.5th percentiles at template/layout-cluster level",
        },
        "feature_status": {
            "js": JS_FEATURES,
            "wasserstein": WASSERSTEIN_FEATURES,
            "not_directly_comparable": [
                "num_words",
                "num_text_lines",
                "page_text_density",
                "num_table_items",
                "relation_type_distribution",
                "domain",
                "grid_topology",
                "cell_spans",
                "widgets_and_states",
                "typed_formstruct_relations",
            ],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
            "matplotlib_font_family": "DejaVu Sans (repository default)",
        },
        "reproduction_command": (
            ".venv/bin/python -m benchmark_stats.run_representativeness_analysis "
            f"--srfund-root {args.srfund_root} --srfund-archive {args.srfund_archive} "
            f"--layout-dir {args.layout_dir} --formstruct-image-dir {args.formstruct_image_dir} "
            f"--template-root {args.template_root} --source-metadata-dir {args.source_metadata_dir} "
            f"--split-assignments {args.split_assignments} --bootstrap-rounds {args.bootstrap_rounds} "
            f"--seed {args.seed} --output {args.output}"
        ),
    }


def main() -> int:
    args = parse_args()
    srfund_root = Path(args.srfund_root)
    archive_path = Path(args.srfund_archive)
    layout_dir = Path(args.layout_dir)
    formstruct_image_dir = Path(args.formstruct_image_dir)
    template_root = Path(args.template_root)
    source_metadata_dir = Path(args.source_metadata_dir)
    split_path = Path(args.split_assignments)
    output = Path(args.output)
    for path in (
        srfund_root,
        archive_path,
        layout_dir,
        formstruct_image_dir,
        template_root,
        source_metadata_dir,
        split_path,
    ):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    output.mkdir(parents=True, exist_ok=True)

    print("verifying source archive")
    verify_zip_archive(archive_path)
    archive_sha256 = sha256_file(archive_path)
    print("loading SRFUND annotations")
    srfund_pages, srfund_inventory = load_srfund_units(srfund_root)
    print("loading selected FormStruct templates")
    form_units, form_inventory = load_formstruct_units(
        layout_dir, formstruct_image_dir, template_root, source_metadata_dir, split_path
    )
    print(f"extracting image hashes for {len(srfund_pages) + len(form_units)} source images")
    attach_image_metrics(srfund_pages, args.workers)
    attach_image_metrics(form_units, args.workers)

    print("clustering SRFUND pages into independent layouts")
    membership, internal_candidate_rows, sensitivity, metric_cache = cluster_srfund_pages(srfund_pages)
    print("auditing cross-dataset overlap")
    cross_rows, page_exclusions = cross_dataset_dedup_candidates(form_units, srfund_pages)
    reference_units, membership_rows, excluded_rows = build_reference_clusters(
        srfund_pages, membership, page_exclusions, metric_cache
    )

    write_rows(output / "dedup_candidates.csv", cross_rows)
    write_rows(output / "dedup_excluded.csv", excluded_rows)
    write_rows(output / "srfund_layout_clusters.csv", membership_rows)
    write_rows(output / "srfund_near_duplicate_candidates.csv", internal_candidate_rows)
    write_rows(
        output / "formstruct_template_features.csv",
        [public_feature_row(unit) for unit in form_units],
    )
    write_rows(
        output / "srfund_reference_features.csv",
        [public_feature_row(unit) for unit in reference_units],
    )
    write_directory_tree(output / "srfund_directory_tree.txt", srfund_root)
    (output / "data_inventory.json").write_text(
        json.dumps(
            json_ready({"srfund": srfund_inventory, "formstruct": form_inventory}),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_feature_mapping(output / "feature_mapping.md")
    write_dedup_report(
        output / "dedup_report.md",
        archive_path,
        archive_sha256,
        srfund_inventory,
        sensitivity,
        cross_rows,
        excluded_rows,
    )

    subsets = build_analysis_subsets(form_units, reference_units)
    calibration_rows, runtime = run_calibration(
        subsets, args.bootstrap_rounds, args.seed
    )
    calibration_summary_rows = summarize_calibration(calibration_rows)
    write_rows(output / "calibration_bootstrap.csv", calibration_rows)
    write_rows(output / "calibration_summary.csv", calibration_summary_rows)

    js_rows = build_js_results(subsets, calibration_summary_rows)
    wasserstein_rows = build_wasserstein_results(
        subsets, calibration_summary_rows, args.bootstrap_rounds, args.seed
    )
    coverage_rows, nearest_rows, plot_payload = build_coverage_results(
        runtime, calibration_summary_rows, args.bootstrap_rounds, args.seed
    )
    write_rows(output / "js_results.csv", js_rows)
    write_rows(output / "wasserstein_results.csv", wasserstein_rows)
    write_rows(output / "coverage_results.csv", coverage_rows)
    write_rows(output / "nearest_template_matches.csv", nearest_rows)

    configure_plot_style()
    plot_distance_heatmap(js_rows, wasserstein_rows, output)
    included_reference = [unit for unit in reference_units if unit.dedup_status == "included"]
    plot_continuous_ecdf(form_units, included_reference, output)
    plot_coverage_by_slice(coverage_rows, output)
    plot_nearest_neighbor_gap(plot_payload, output)
    write_representativeness_table(output / "representativeness_table.tex", coverage_rows)

    manifest = build_manifest(
        args,
        archive_sha256,
        srfund_inventory,
        form_inventory,
        sensitivity,
        reference_units,
        excluded_rows,
    )
    (output / "analysis_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_representativeness_summary(
        output / "representativeness_summary.md",
        archive_sha256,
        srfund_inventory,
        form_inventory,
        sensitivity,
        reference_units,
        form_units,
        cross_rows,
        excluded_rows,
        js_rows,
        wasserstein_rows,
        coverage_rows,
        args.bootstrap_rounds,
        args.seed,
    )
    validation = validate_outputs(output, render_pdfs=not args.skip_render_validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
