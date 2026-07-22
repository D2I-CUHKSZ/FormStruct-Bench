from __future__ import annotations

import csv
import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from formtsr_exp.hierarchical_metrics import bbox_iou, maximum_weight_matching
from formtsr_exp.io_utils import read_json, write_json, write_jsonl


LANGUAGES = ("de", "en", "es", "fr", "it", "ja", "pt", "zh")
LABELS = ("header", "question", "answer", "other")
METRIC_FIELDS = (
    "valid_json",
    "entity_f1",
    "entity_e2e_f1",
    "entity_text_score",
    "label_accuracy",
    "link_f1",
    "link_matched_endpoint_f1",
    "hierarchy_f1",
    "hierarchy_matched_endpoint_f1",
)


SRFUND_RESPONSE_SCHEMA: dict[str, Any] = {
    "title": "SRFUNDSemanticPrediction",
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "links"],
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 320,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "bbox", "text"],
                "properties": {
                    "id": {"type": "string", "maxLength": 32},
                    "label": {"type": "string", "enum": list(LABELS)},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "text": {"type": "string", "maxLength": 512},
                },
            },
        },
        "links": {
            "type": "array",
            "maxItems": 1024,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "target"],
                "properties": {
                    "source": {"type": "string", "maxLength": 32},
                    "target": {"type": "string", "maxLength": 32},
                },
            },
        },
    },
}


SRFUND_PROMPT = """Extract the semantic form structure from this document image.

Return exactly one JSON object with two keys: "entities" and "links".

Entity rules:
- One entity is one semantic text block, not an individual OCR word.
- "label" must be exactly one of: "header", "question", "answer", "other".
- header: a title or section heading that governs other fields.
- question: a field name, prompt, key, or column label.
- answer: a filled value associated with a question.
- other: relevant document text that is none of the above.
- "bbox" is [x1,y1,x2,y2] normalized to 0..1000 relative to the full image.
- "text" must transcribe the visible entity text in reading order.
- Give every entity a unique short ID such as e1, e2, ... .

Link rules:
- Each link is directed from parent to child: header -> question, question -> answer, or a semantic parent -> nested child.
- Link endpoints must reference emitted entity IDs.
- Do not invent links when the relationship is unclear.

Do not include Markdown, commentary, OCR word lists, or keys other than "entities" and "links"."""


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    label: str
    bbox: tuple[float, float, float, float]
    text: str


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    language: str
    split: str
    image_name: str
    image_path: Path
    width: int
    height: int
    entities: tuple[Entity, ...]
    links: frozenset[tuple[str, str]]
    hierarchy_edges: frozenset[tuple[str, str]]

    def adapter_sample(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "image_path": str(self.image_path)}

    def index_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "language": self.language,
            "split": self.split,
            "image_name": self.image_name,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "n_entities": len(self.entities),
            "n_links": len(self.links),
            "n_hierarchy_edges": len(self.hierarchy_edges),
        }


def split_from_name(name: str) -> str:
    if "_val_" in name:
        return "validation"
    if "_train_" in name:
        return "train"
    return "unavailable"


def parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in coords):
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return coords


def normalize_bbox(
    value: Any, width: int, height: int
) -> tuple[float, float, float, float] | None:
    box = parse_bbox(value)
    if box is None or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    result = (
        max(0.0, min(1000.0, x1 * 1000.0 / width)),
        max(0.0, min(1000.0, y1 * 1000.0 / height)),
        max(0.0, min(1000.0, x2 * 1000.0 / width)),
        max(0.0, min(1000.0, y2 * 1000.0 / height)),
    )
    return result if result[2] > result[0] and result[3] > result[1] else None


def relation_tree_edges(tree: Any) -> frozenset[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()

    def visit(value: Any, parent: int | None) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            if parent is not None and value != parent:
                edges.add((str(parent), str(value)))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, parent)
            return
        if not isinstance(value, dict):
            return
        for raw_key, child in value.items():
            try:
                node = int(raw_key)
            except (TypeError, ValueError):
                visit(child, parent)
                continue
            if parent is not None and node != parent:
                edges.add((str(parent), str(node)))
            visit(child, node)

    visit(tree, None)
    return frozenset(edges)


def _load_language(root: Path, language: str, split: str) -> list[Sample]:
    instances = read_json(root / "instance_annotation" / f"{language}.json")
    relations = read_json(root / "relation_annotation" / f"{language}.json")
    if not isinstance(instances, dict) or not isinstance(relations, dict):
        raise ValueError(f"SRFUND annotations must be objects for {language}")
    if set(instances) != set(relations):
        raise ValueError(f"instance/relation key mismatch for {language}")

    output: list[Sample] = []
    for image_name in sorted(instances):
        image_split = split_from_name(image_name)
        if split != "all" and image_split != split:
            continue
        image_path = root / "images" / language / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        raw_entities = instances[image_name]
        if not isinstance(raw_entities, list):
            raise ValueError(f"entities must be a list: {language}/{image_name}")

        entities: list[Entity] = []
        entity_ids: set[str] = set()
        direct_links: set[tuple[str, str]] = set()
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue
            entity_id = str(raw.get("id"))
            if entity_id in entity_ids:
                raise ValueError(f"duplicate entity ID {entity_id}: {language}/{image_name}")
            label = str(raw.get("label") or "other").strip().lower()
            if label not in LABELS:
                label = "other"
            box = normalize_bbox(raw.get("box"), width, height)
            if box is None:
                continue
            entity_ids.add(entity_id)
            entities.append(Entity(entity_id, label, box, str(raw.get("text") or "")))
            linking = raw.get("linking") if isinstance(raw.get("linking"), list) else []
            for pair in linking:
                if isinstance(pair, list) and len(pair) == 2:
                    direct_links.add((str(pair[0]), str(pair[1])))

        direct_links = {
            edge for edge in direct_links if edge[0] in entity_ids and edge[1] in entity_ids
        }
        hierarchy = {
            edge
            for edge in relation_tree_edges(relations[image_name])
            if edge[0] in entity_ids and edge[1] in entity_ids
        }
        output.append(
            Sample(
                sample_id=f"{language}__{Path(image_name).stem}",
                language=language,
                split=image_split,
                image_name=image_name,
                image_path=image_path,
                width=width,
                height=height,
                entities=tuple(entities),
                links=frozenset(direct_links),
                hierarchy_edges=frozenset(hierarchy),
            )
        )
    return output


def load_samples(
    root: Path,
    split: str = "validation",
    *,
    unavailable_limit: int = 50,
    seed: int = 42,
) -> list[Sample]:
    if split not in {"train", "validation", "validation_balanced", "all"}:
        raise ValueError(f"unsupported SRFUND split: {split}")
    samples: list[Sample] = []
    for language in LANGUAGES:
        requested_split = "validation" if split == "validation_balanced" else split
        language_samples = _load_language(root, language, requested_split)
        if split == "validation_balanced" and not language_samples:
            unavailable = [
                sample
                for sample in _load_language(root, language, "all")
                if sample.split == "unavailable"
            ]
            unavailable.sort(
                key=lambda sample: hashlib.sha256(
                    f"{seed}:{sample.sample_id}".encode("utf-8")
                ).hexdigest()
            )
            language_samples = unavailable[:unavailable_limit]
        samples.extend(language_samples)
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate SRFUND sample IDs")
    if not samples:
        raise ValueError(f"no SRFUND samples for split {split}")
    return samples


def write_index(path: Path, samples: Sequence[Sample]) -> None:
    write_jsonl(path, (sample.index_row() for sample in samples))


def dataset_statistics(samples: Sequence[Sample]) -> dict[str, Any]:
    entity_counts = [len(sample.entities) for sample in samples]
    link_counts = [len(sample.links) for sample in samples]
    hierarchy_counts = [len(sample.hierarchy_edges) for sample in samples]
    return {
        "n_samples": len(samples),
        "by_language": {
            language: sum(sample.language == language for sample in samples)
            for language in LANGUAGES
        },
        "n_entities": sum(entity_counts),
        "max_entities_per_page": max(entity_counts, default=0),
        "n_links": sum(link_counts),
        "max_links_per_page": max(link_counts, default=0),
        "n_hierarchy_edges": sum(hierarchy_counts),
        "max_hierarchy_edges_per_page": max(hierarchy_counts, default=0),
    }


def validate_prediction(value: Any) -> tuple[bool, str | None]:
    if not isinstance(value, dict):
        return False, "prediction root must be an object"
    if set(value) != {"entities", "links"}:
        return False, "prediction must contain exactly entities and links"
    if not isinstance(value["entities"], list) or not isinstance(value["links"], list):
        return False, "entities and links must be arrays"
    if len(value["entities"]) > 320 or len(value["links"]) > 1024:
        return False, "prediction exceeds schema item limits"
    for index, raw in enumerate(value["entities"]):
        if not isinstance(raw, dict) or set(raw) != {"id", "label", "bbox", "text"}:
            return False, f"invalid entity object at index {index}"
        if not isinstance(raw["id"], str) or not raw["id"] or len(raw["id"]) > 32:
            return False, f"invalid entity id at index {index}"
        if raw["label"] not in LABELS:
            return False, f"invalid entity label at index {index}"
        if parse_bbox(raw["bbox"]) is None:
            return False, f"invalid entity bbox at index {index}"
        if not isinstance(raw["text"], str):
            return False, f"invalid entity text at index {index}"
    for index, raw in enumerate(value["links"]):
        if not isinstance(raw, dict) or set(raw) != {"source", "target"}:
            return False, f"invalid link object at index {index}"
        if not isinstance(raw["source"], str) or not isinstance(raw["target"], str):
            return False, f"invalid link endpoint at index {index}"
    return True, None


def normalize_prediction(
    value: Mapping[str, Any],
) -> tuple[list[Entity], set[tuple[str, str]], dict[str, int]]:
    entities: list[Entity] = []
    seen: set[str] = set()
    duplicate_ids = 0
    for raw in value.get("entities", []):
        entity_id = str(raw["id"])
        if entity_id in seen:
            duplicate_ids += 1
            continue
        seen.add(entity_id)
        box = parse_bbox(raw["bbox"])
        if box is not None:
            entities.append(Entity(entity_id, str(raw["label"]), box, str(raw["text"])))
    links = {
        (str(raw["source"]), str(raw["target"]))
        for raw in value.get("links", [])
        if isinstance(raw, dict)
    }
    return entities, links, {
        "duplicate_entity_ids": duplicate_ids,
        "invalid_link_endpoints": sum(
            source not in seen or target not in seen for source, target in links
        ),
    }


_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _SPACE_RE.sub(" ", normalized).strip()


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return float(Levenshtein.normalized_similarity(left_norm, right_norm))


def f1(tp: int, pred: int, gt: int) -> float:
    return 2.0 * tp / (pred + gt) if pred + gt else 1.0


def _entity_matching(
    pred: Sequence[Entity],
    gt: Sequence[Entity],
    *,
    require_label: bool,
    require_text: bool = False,
    iou_threshold: float = 0.5,
    text_threshold: float = 0.8,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    weights = np.zeros((len(pred), len(gt)), dtype=np.float64)
    eligible = np.zeros_like(weights, dtype=bool)
    similarities = np.zeros_like(weights)
    for pred_index, pred_entity in enumerate(pred):
        for gt_index, gt_entity in enumerate(gt):
            overlap = bbox_iou(pred_entity.bbox, gt_entity.bbox)
            similarity = text_similarity(pred_entity.text, gt_entity.text)
            weights[pred_index, gt_index] = overlap + similarity * 1e-3
            similarities[pred_index, gt_index] = similarity
            eligible[pred_index, gt_index] = (
                overlap >= iou_threshold
                and (not require_label or pred_entity.label == gt_entity.label)
                and (not require_text or similarity >= text_threshold)
            )
    return (
        maximum_weight_matching(weights, eligible, cardinality_first=True),
        similarities,
    )


def _edge_metrics(
    pred_edges: set[tuple[str, str]],
    gt_edges: frozenset[tuple[str, str]],
    entity_mapping: Mapping[str, str],
) -> dict[str, Any]:
    mapped: set[tuple[str, str]] = set()
    endpoint_pred = 0
    for source, target in pred_edges:
        mapped_source = entity_mapping.get(source)
        mapped_target = entity_mapping.get(target)
        if mapped_source is None or mapped_target is None:
            continue
        endpoint_pred += 1
        mapped.add((mapped_source, mapped_target))
    tp = len(mapped & gt_edges)
    matched_gt_ids = set(entity_mapping.values())
    endpoint_gt = {
        edge for edge in gt_edges if edge[0] in matched_gt_ids and edge[1] in matched_gt_ids
    }
    endpoint_tp = len(mapped & endpoint_gt)
    return {
        "tp": tp,
        "pred": len(pred_edges),
        "gt": len(gt_edges),
        "f1": f1(tp, len(pred_edges), len(gt_edges)),
        "matched_endpoint_tp": endpoint_tp,
        "matched_endpoint_pred": endpoint_pred,
        "matched_endpoint_gt": len(endpoint_gt),
        "matched_endpoint_f1": f1(endpoint_tp, endpoint_pred, len(endpoint_gt)),
    }


def evaluate_prediction(
    sample: Sample,
    prediction: Mapping[str, Any] | None,
    *,
    model: str,
) -> dict[str, Any]:
    pred_entities: list[Entity] = []
    pred_links: set[tuple[str, str]] = set()
    diagnostics = {"duplicate_entity_ids": 0, "invalid_link_endpoints": 0}
    if prediction is not None:
        pred_entities, pred_links, diagnostics = normalize_prediction(prediction)
    gt_entities = list(sample.entities)

    label_pairs, similarities = _entity_matching(
        pred_entities, gt_entities, require_label=True
    )
    geometry_pairs, _ = _entity_matching(
        pred_entities, gt_entities, require_label=False
    )
    e2e_pairs, _ = _entity_matching(
        pred_entities, gt_entities, require_label=True, require_text=True
    )
    entity_mapping = {
        pred_entities[pred_index].entity_id: gt_entities[gt_index].entity_id
        for pred_index, gt_index in label_pairs
    }
    text_sum = sum(similarities[pred_index, gt_index] for pred_index, gt_index in label_pairs)
    label_correct = sum(
        pred_entities[pred_index].label == gt_entities[gt_index].label
        for pred_index, gt_index in geometry_pairs
    )
    link = _edge_metrics(pred_links, sample.links, entity_mapping)
    hierarchy = _edge_metrics(pred_links, sample.hierarchy_edges, entity_mapping)
    if prediction is None:
        link["f1"] = 0.0
        link["matched_endpoint_f1"] = 0.0
        hierarchy["f1"] = 0.0
        hierarchy["matched_endpoint_f1"] = 0.0
    return {
        "model": model,
        "sample_id": sample.sample_id,
        "language": sample.language,
        "valid_json": prediction is not None,
        "n_entity_tp": len(label_pairs),
        "n_entity_pred": len(pred_entities),
        "n_entity_gt": len(gt_entities),
        "entity_f1": f1(len(label_pairs), len(pred_entities), len(gt_entities)),
        "n_entity_e2e_tp": len(e2e_pairs),
        "n_entity_e2e_pred": len(pred_entities),
        "n_entity_e2e_gt": len(gt_entities),
        "entity_e2e_f1": f1(len(e2e_pairs), len(pred_entities), len(gt_entities)),
        "entity_text_similarity_sum": text_sum,
        "entity_text_score": text_sum / max(len(pred_entities), len(gt_entities), 1),
        "n_geometry_matches": len(geometry_pairs),
        "n_label_correct": label_correct,
        "label_accuracy": label_correct / len(geometry_pairs) if geometry_pairs else 0.0,
        "n_link_tp": link["tp"],
        "n_link_pred": link["pred"],
        "n_link_gt": link["gt"],
        "link_f1": link["f1"],
        "n_link_matched_endpoint_tp": link["matched_endpoint_tp"],
        "n_link_matched_endpoint_pred": link["matched_endpoint_pred"],
        "n_link_matched_endpoint_gt": link["matched_endpoint_gt"],
        "link_matched_endpoint_f1": link["matched_endpoint_f1"],
        "n_hierarchy_tp": hierarchy["tp"],
        "n_hierarchy_pred": hierarchy["pred"],
        "n_hierarchy_gt": hierarchy["gt"],
        "hierarchy_f1": hierarchy["f1"],
        "n_hierarchy_matched_endpoint_tp": hierarchy["matched_endpoint_tp"],
        "n_hierarchy_matched_endpoint_pred": hierarchy["matched_endpoint_pred"],
        "n_hierarchy_matched_endpoint_gt": hierarchy["matched_endpoint_gt"],
        "hierarchy_matched_endpoint_f1": hierarchy["matched_endpoint_f1"],
        **diagnostics,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return math.fsum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def _micro(rows: Sequence[Mapping[str, Any]], prefix: str) -> float:
    return f1(
        sum(int(row[f"n_{prefix}_tp"]) for row in rows),
        sum(int(row[f"n_{prefix}_pred"]) for row in rows),
        sum(int(row[f"n_{prefix}_gt"]) for row in rows),
    )


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    model_id: str,
    family: str,
    tuned: bool,
    language: str = "all",
) -> dict[str, Any]:
    selected = [row for row in rows if language == "all" or row["language"] == language]
    valid = sum(bool(row["valid_json"]) for row in selected)
    return {
        "model": model,
        "model_id": model_id,
        "family": family,
        "tuned": tuned,
        "language": language,
        "n_total": len(selected),
        "n_valid_json": valid,
        "coverage": valid / len(selected) if selected else 0.0,
        "entity_f1": _mean(selected, "entity_f1"),
        "entity_f1_micro": _micro(selected, "entity"),
        "entity_e2e_f1": _mean(selected, "entity_e2e_f1"),
        "entity_e2e_f1_micro": _micro(selected, "entity_e2e"),
        "entity_text_score": _mean(selected, "entity_text_score"),
        "label_accuracy": _mean(selected, "label_accuracy"),
        "link_f1": _mean(selected, "link_f1"),
        "link_f1_micro": _micro(selected, "link"),
        "link_matched_endpoint_f1": _mean(selected, "link_matched_endpoint_f1"),
        "link_matched_endpoint_f1_micro": _micro(selected, "link_matched_endpoint"),
        "hierarchy_f1": _mean(selected, "hierarchy_f1"),
        "hierarchy_f1_micro": _micro(selected, "hierarchy"),
        "hierarchy_matched_endpoint_f1": _mean(
            selected, "hierarchy_matched_endpoint_f1"
        ),
        "hierarchy_matched_endpoint_f1_micro": _micro(
            selected, "hierarchy_matched_endpoint"
        ),
        "n_entity_tp": sum(int(row["n_entity_tp"]) for row in selected),
        "n_entity_pred": sum(int(row["n_entity_pred"]) for row in selected),
        "n_entity_gt": sum(int(row["n_entity_gt"]) for row in selected),
        "n_link_tp": sum(int(row["n_link_tp"]) for row in selected),
        "n_link_pred": sum(int(row["n_link_pred"]) for row in selected),
        "n_link_gt": sum(int(row["n_link_gt"]) for row in selected),
        "n_hierarchy_tp": sum(int(row["n_hierarchy_tp"]) for row in selected),
        "n_hierarchy_pred": sum(int(row["n_hierarchy_pred"]) for row in selected),
        "n_hierarchy_gt": sum(int(row["n_hierarchy_gt"]) for row in selected),
        "n_duplicate_entity_ids": sum(
            int(row["duplicate_entity_ids"]) for row in selected
        ),
        "n_invalid_link_endpoints": sum(
            int(row["invalid_link_endpoints"]) for row in selected
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def bootstrap_language_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    by_language: dict[str, list[float]] = {}
    for row in rows:
        by_language.setdefault(str(row["language"]), []).append(float(row[field]))
    languages = sorted(by_language)
    if not languages:
        return 0.0, 0.0
    sums = np.asarray([math.fsum(by_language[key]) for key in languages], dtype=np.float64)
    counts = np.asarray([len(by_language[key]) for key in languages], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(languages), size=(iterations, len(languages)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)
