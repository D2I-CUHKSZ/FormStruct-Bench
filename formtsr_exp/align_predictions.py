from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, read_jsonl, write_json
from .metrics import bbox_from, flatten_leaf_fields, iou, normalize_metadata_layout, unwrap_answer


GENERIC_LABELS = {
    "answer",
    "answers",
    "data",
    "field",
    "fields",
    "form",
    "forms",
    "information",
    "additional information",
    "section",
    "value",
}


@dataclass
class FieldDescriptor:
    path: tuple[str, ...]
    label_variants: list[str]
    key_bbox: tuple[float, float, float, float] | None = None
    value_bbox: tuple[float, float, float, float] | None = None


@dataclass
class CandidateValue:
    value: Any
    label_variants: list[str]
    source: str
    bbox: tuple[float, float, float, float] | None = None
    score_debug: dict[str, Any] = field(default_factory=dict)


_DESCRIPTOR_CACHE: dict[tuple[str, str, tuple[str, ...]], list[FieldDescriptor]] = {}


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.strip().lower().split())
    text = re.sub(r"[_:/\\|]+", " ", text)
    text = " ".join(text.split())
    return text


def _label_parts(path: tuple[str, ...]) -> list[str]:
    parts: list[str] = []
    for item in path:
        text = str(item).strip()
        if not text:
            continue
        if _norm_text(text) in GENERIC_LABELS:
            continue
        parts.append(text)
    return parts


def _dedupe_texts(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _norm_text(text)
        if not key or key in seen or key in GENERIC_LABELS:
            continue
        seen.add(key)
        out.append(text)
    return out


@lru_cache(maxsize=250_000)
def _label_similarity_norm_cached(an: str, bn: str) -> float:
    if not an or not bn:
        return 0.0
    if an == bn:
        return 1.0
    a_tokens = set(an.split())
    b_tokens = set(bn.split())
    if a_tokens and b_tokens:
        smaller = min(len(a_tokens), len(b_tokens))
        overlap = len(a_tokens & b_tokens)
        if overlap == smaller:
            return 0.9
        token_jaccard = overlap / len(a_tokens | b_tokens)
    else:
        token_jaccard = 0.0
    if an in bn or bn in an:
        return max(0.85, token_jaccard)
    return max(token_jaccard, SequenceMatcher(None, an, bn).ratio())


def _label_similarity(a: str, b: str) -> float:
    return _label_similarity_norm_cached(_norm_text(a), _norm_text(b))


def _best_label_similarity(candidate: CandidateValue, descriptor: FieldDescriptor) -> tuple[float, tuple[str, str]]:
    best = 0.0
    best_pair = ("", "")
    for a in candidate.label_variants:
        for b in descriptor.label_variants:
            score = _label_similarity(a, b)
            if score > best:
                best = score
                best_pair = (a, b)
    return best, best_pair


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diag(box: tuple[float, float, float, float]) -> float:
    return max(1.0, math.hypot(box[2] - box[0], box[3] - box[1]))


def _bbox_similarity(candidate_box: tuple[float, float, float, float] | None, target_box: tuple[float, float, float, float] | None) -> float:
    if not candidate_box or not target_box:
        return 0.0
    overlap = iou(candidate_box, target_box)
    if overlap > 0:
        return min(1.0, 0.5 + overlap / 2.0)
    cx, cy = _center(candidate_box)
    tx, ty = _center(target_box)
    dist = math.hypot(cx - tx, cy - ty)
    return max(0.0, 1.0 - dist / max(50.0, 3.0 * _diag(target_box)))


def _candidate_descriptor_score(candidate: CandidateValue, descriptor: FieldDescriptor) -> tuple[float, dict[str, Any]]:
    label_score, label_pair = _best_label_similarity(candidate, descriptor)
    bbox_score = _bbox_similarity(candidate.bbox, descriptor.value_bbox)
    if candidate.bbox and bbox_score > 0:
        score = 0.65 * label_score + 0.35 * bbox_score
    else:
        score = label_score
    return score, {"label_score": label_score, "bbox_score": bbox_score, "label_pair": label_pair}


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = root
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


def _debug_value(value: Any, max_chars: int = 160) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "..."
    return value


def _write_json_compact(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _metadata_descriptor_paths(layout: Any, gt_paths: set[tuple[str, ...]]) -> dict[tuple[str, ...], FieldDescriptor]:
    descriptors: dict[tuple[str, ...], FieldDescriptor] = {}
    if not isinstance(layout, dict):
        return descriptors

    def visit(node: dict[str, Any], original_path: tuple[str, ...], semantic_path: tuple[str, ...]) -> None:
        original = str(node.get("original_label") or "").strip()
        semantic = str(node.get("semantic_key") or "").strip()
        label = original or semantic
        if not label:
            return
        current_original = original_path + (label,)
        current_semantic = semantic_path + ((semantic or original),)
        value_bbox = bbox_from(node.get("value"))
        if value_bbox is None:
            value_boxes = [box for item in node.get("values", []) if isinstance(item, dict) for box in [bbox_from(item)] if box]
            if value_boxes:
                value_bbox = (
                    min(box[0] for box in value_boxes),
                    min(box[1] for box in value_boxes),
                    max(box[2] for box in value_boxes),
                    max(box[3] for box in value_boxes),
                )
        if current_original in gt_paths:
            descriptors[current_original] = FieldDescriptor(
                path=current_original,
                label_variants=_dedupe_texts(
                    [
                        original,
                        semantic,
                        " / ".join(current_original),
                        " / ".join(current_semantic),
                    ]
                ),
                key_bbox=bbox_from(node),
                value_bbox=value_bbox,
            )
        for child in node.get("keys", []) or []:
            if isinstance(child, dict):
                visit(child, current_original, current_semantic)

    for root in layout.get("fields", []) or []:
        if isinstance(root, dict):
            visit(root, (), ())
    return descriptors


@lru_cache(maxsize=None)
def _read_layout_cached(layout_root: str, template_name: str) -> Any:
    layout_path = Path(layout_root) / f"{template_name}.json"
    return read_json(layout_path) if layout_path.exists() else {}


def _field_descriptors(sample: dict[str, Any], layout_root: Path) -> list[FieldDescriptor]:
    gt = read_json(Path(sample["label_path"]))
    gt_paths = set(flatten_leaf_fields(gt))
    cache_key = (
        str(layout_root),
        str(sample["template_name"]),
        tuple(sorted("/".join(path) for path in gt_paths)),
    )
    cached = _DESCRIPTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    layout = _read_layout_cached(str(layout_root), str(sample["template_name"]))
    descriptors = _metadata_descriptor_paths(layout, gt_paths)
    out: list[FieldDescriptor] = []
    for path in sorted(gt_paths):
        descriptor = descriptors.get(path)
        if descriptor is None:
            parts = _label_parts(path)
            descriptor = FieldDescriptor(path=path, label_variants=_dedupe_texts([path[-1], " / ".join(parts)]))
        out.append(descriptor)
    _DESCRIPTOR_CACHE[cache_key] = out
    return out


def _region_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("label") or item.get("value") or "").strip()


def _region_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("region_type") or item.get("data_type") or "").strip().lower()


def _extract_answer_candidates(pred: Any) -> list[CandidateValue]:
    answer = unwrap_answer(pred)
    candidates: list[CandidateValue] = []
    for path, value in flatten_leaf_fields(answer).items():
        if value in (None, ""):
            continue
        parts = _label_parts(path)
        labels = _dedupe_texts([*(parts[-2:] if len(parts) >= 2 else parts), " / ".join(parts)])
        if not labels:
            continue
        candidates.append(CandidateValue(value=value, label_variants=labels, source="answer"))
    return candidates


def _extract_region_candidates(pred: Any) -> list[CandidateValue]:
    if not isinstance(pred, dict):
        return []
    regions = [item for item in pred.get("regions", []) or [] if isinstance(item, dict)]
    by_id = {str(item.get("id")): item for item in regions if item.get("id") is not None}
    candidates: list[CandidateValue] = []

    for rel in pred.get("relations", []) or []:
        if not isinstance(rel, dict):
            continue
        rel_type = str(rel.get("type") or rel.get("relation_type") or "").lower()
        if rel_type != "key-value":
            continue
        source = by_id.get(str(rel.get("source") or rel.get("from") or ""))
        target = by_id.get(str(rel.get("target") or rel.get("to") or ""))
        label = str(rel.get("source_label") or (source and _region_text(source)) or "").strip()
        value = str(rel.get("target_label") or (target and _region_text(target)) or "").strip()
        if label and value:
            candidates.append(
                CandidateValue(
                    value=value,
                    label_variants=_dedupe_texts([label]),
                    source="relation",
                    bbox=bbox_from(target) if target else None,
                )
            )

    label_regions = []
    value_regions = []
    for item in regions:
        text = _region_text(item)
        box = bbox_from(item)
        if not text or not box:
            continue
        kind = _region_type(item)
        if kind in {"value", "field_value", "answer"}:
            value_regions.append((item, box))
        elif kind in {"field", "key", "label", "question"}:
            label_regions.append((item, box))

    for value_region, value_box in value_regions:
        value_text = _region_text(value_region)
        best_label: str | None = None
        best_score = 0.0
        vcx, vcy = _center(value_box)
        for label_region, label_box in label_regions:
            lcx, lcy = _center(label_box)
            row_dist = abs(vcy - lcy)
            if row_dist > max(45.0, 2.5 * max(value_box[3] - value_box[1], label_box[3] - label_box[1])):
                continue
            horizontal_gap = max(0.0, max(value_box[0], label_box[0]) - min(value_box[2], label_box[2]))
            score = 1.0 / (1.0 + row_dist + 0.01 * horizontal_gap + 0.001 * abs(vcx - lcx))
            if score > best_score:
                best_score = score
                best_label = _region_text(label_region)
        if best_label:
            candidates.append(
                CandidateValue(
                    value=value_text,
                    label_variants=_dedupe_texts([best_label]),
                    source="region_pair",
                    bbox=value_box,
                )
            )
    return candidates


def extract_candidates(pred: Any) -> list[CandidateValue]:
    candidates = _extract_answer_candidates(pred) + _extract_region_candidates(pred)
    deduped: list[CandidateValue] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (
            json.dumps(candidate.value, ensure_ascii=False, sort_keys=True) if isinstance(candidate.value, (dict, list)) else str(candidate.value),
            "|".join(_norm_text(label) for label in candidate.label_variants),
            candidate.source,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def align_prediction(
    sample: dict[str, Any],
    pred: dict[str, Any],
    *,
    layout_root: Path,
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptors = _field_descriptors(sample, layout_root)
    candidates = extract_candidates(pred)
    edges: list[tuple[float, int, int, dict[str, Any]]] = []
    for d_index, descriptor in enumerate(descriptors):
        for c_index, candidate in enumerate(candidates):
            score, debug = _candidate_descriptor_score(candidate, descriptor)
            if score >= min_score:
                edges.append((score, d_index, c_index, debug))

    assigned_descriptors: set[int] = set()
    assigned_candidates: set[int] = set()
    assignments: list[dict[str, Any]] = []
    aligned_answer: dict[str, Any] = {}
    for score, d_index, c_index, debug in sorted(edges, reverse=True, key=lambda item: item[0]):
        if d_index in assigned_descriptors or c_index in assigned_candidates:
            continue
        assigned_descriptors.add(d_index)
        assigned_candidates.add(c_index)
        descriptor = descriptors[d_index]
        candidate = candidates[c_index]
        _set_path(aligned_answer, descriptor.path, candidate.value)
        assignments.append(
            {
                "path": list(descriptor.path),
                "source": candidate.source,
                "score": score,
                "value": _debug_value(candidate.value),
                "debug": debug,
            }
        )
    unaligned_values = [candidates[index].value for index in range(len(candidates)) if index not in assigned_candidates]
    if unaligned_values:
        aligned_answer["__unaligned__"] = unaligned_values

    aligned = copy.deepcopy(pred)
    aligned["answer"] = aligned_answer
    aligned["_alignment"] = {
        "method": "metadata_label_bbox_v1",
        "min_score": min_score,
        "candidate_count": len(candidates),
        "gt_field_count": len(descriptors),
        "assigned_count": len(assignments),
        "unaligned_count": len(unaligned_values),
        "note": "Uses newdataset-layout labels/bboxes and prediction text only; does not use GT answer values for matching.",
        "assignments_sample": assignments[:10],
    }
    report = {
        "sample_id": sample["sample_id"],
        "candidate_count": len(candidates),
        "gt_field_count": len(descriptors),
        "assigned_count": len(assignments),
    }
    return aligned, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align FormTSR prediction answers to metadata field paths.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--pred-in", required=True)
    parser.add_argument("--pred-out", required=True)
    parser.add_argument("--layout-root", default="./newdataset-layout")
    parser.add_argument("--report", default="")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _align_one_task(args: tuple[dict[str, Any], str, str, str, float]) -> dict[str, Any]:
    sample, pred_in_str, pred_out_str, layout_root_str, min_score = args
    pred_in = Path(pred_in_str)
    pred_out = Path(pred_out_str)
    layout_root = Path(layout_root_str)
    in_path = pred_in / f"{sample['sample_id']}.json"
    if not in_path.exists():
        return {"sample_id": sample["sample_id"], "status": "missing"}
    try:
        pred = read_json(in_path)
        if not isinstance(pred, dict):
            raise ValueError("prediction JSON is not an object")
        aligned, report = align_prediction(sample, pred, layout_root=layout_root, min_score=min_score)
        _write_json_compact(pred_out / in_path.name, aligned)
        report["status"] = "ok"
        return report
    except Exception as exc:
        return {"sample_id": sample["sample_id"], "status": "failed", "error": str(exc)}


def main() -> None:
    args = parse_args()
    index_rows = read_jsonl(Path(args.index))
    if args.limit is not None:
        index_rows = index_rows[: args.limit]
    pred_in = Path(args.pred_in)
    pred_out = Path(args.pred_out)
    layout_root = Path(args.layout_root)
    reports: list[dict[str, Any]] = []
    tasks = [(sample, str(pred_in), str(pred_out), str(layout_root), args.min_score) for sample in index_rows]
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for report in executor.map(_align_one_task, tasks, chunksize=10):
                reports.append(report)
    else:
        for task in tasks:
            reports.append(_align_one_task(task))
    written = sum(1 for row in reports if row.get("status") == "ok")
    missing = sum(1 for row in reports if row.get("status") == "missing")
    failed = sum(1 for row in reports if row.get("status") == "failed")
    summary = {
        "pred_in": str(pred_in),
        "pred_out": str(pred_out),
        "layout_root": str(layout_root),
        "min_score": args.min_score,
        "written": written,
        "missing": missing,
        "failed": failed,
        "mean_assigned": (sum(row.get("assigned_count", 0) for row in reports) / written) if written else 0,
    }
    if args.report:
        write_json(Path(args.report), {"summary": summary, "samples": reports}, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
