from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from apted import APTED, Config
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .io_utils import ensure_dir, read_json, read_jsonl, write_json
from .metrics import NA, _normalize_value_for_vacc, flatten_leaf_fields, unwrap_answer
from .page_em_report import (
    RunSpec,
    classify_comparison_status,
    classify_run_type,
    load_run_specs,
)


RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "run_type",
    "comparison_status",
    "full_scope",
    "n_total",
    "n_prediction_files",
    "n_extra_prediction_files",
    "n_valid_json",
    "coverage",
    "n_missing_prediction",
    "n_invalid_json",
    "Schema-nTED",
    "Schema-nTED-valid",
    "Value-nED",
    "Value-nED-valid",
]


@dataclass(frozen=True, slots=True)
class SchemaNode:
    name: str
    children: tuple["SchemaNode", ...] = ()
    size: int = field(init=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", 1 + sum(child.size for child in self.children))


@dataclass(frozen=True, slots=True)
class DocumentSample:
    sample_id: str
    schema: SchemaNode
    values: tuple[str, ...]


class SchemaEditConfig(Config):
    def rename(self, node1: SchemaNode, node2: SchemaNode) -> int:
        return 0 if node1.name == node2.name else 2


SCHEMA_EDIT_CONFIG = SchemaEditConfig()
_WORKER_SAMPLES: list[DocumentSample] = []


def _value_kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "scalar"


def build_schema_tree(value: Any, name: str = "root") -> SchemaNode:
    kind = _value_kind(value)
    if isinstance(value, dict):
        children = tuple(
            build_schema_tree(child, f"key:{str(key).strip()}")
            for key, child in sorted(value.items(), key=lambda item: str(item[0]).strip())
        )
    elif isinstance(value, list):
        children = tuple(build_schema_tree(child, "item") for child in value)
    else:
        children = ()
    return SchemaNode(f"{name}|{kind}", children)


def extract_normalized_values(value: Any) -> tuple[str, ...]:
    return tuple(
        token
        for token in (
            _normalize_value_for_vacc(item)
            for item in flatten_leaf_fields(unwrap_answer(value)).values()
        )
        if token
    )


@lru_cache(maxsize=4096)
def schema_nted(pred: SchemaNode, gt: SchemaNode) -> float:
    if pred == gt:
        return 1.0
    distance = float(APTED(pred, gt, SCHEMA_EDIT_CONFIG).compute_edit_distance())
    denominator = pred.size + gt.size
    return max(0.0, 1.0 - distance / denominator) if denominator else 1.0


def value_ned(pred_values: tuple[str, ...], gt_values: tuple[str, ...]) -> float:
    if not pred_values and not gt_values:
        return 1.0
    if not pred_values or not gt_values:
        return 0.0
    similarities = process.cdist(
        pred_values,
        gt_values,
        scorer=Levenshtein.normalized_similarity,
        dtype=np.float32,
    )
    pred_indices, gt_indices = linear_sum_assignment(similarities, maximize=True)
    matched_similarity = float(similarities[pred_indices, gt_indices].sum())
    return matched_similarity / max(len(pred_values), len(gt_values))


def load_document_samples(index_path: Path) -> list[DocumentSample]:
    samples: list[DocumentSample] = []
    seen: set[str] = set()
    for row in read_jsonl(index_path):
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in index: {sample_id}")
        seen.add(sample_id)
        gt = read_json(Path(str(row["label_path"])))
        samples.append(
            DocumentSample(
                sample_id=sample_id,
                schema=build_schema_tree(gt),
                values=extract_normalized_values(gt),
            )
        )
    return samples


def evaluate_run(
    spec: RunSpec,
    samples: list[DocumentSample],
    pred_root: Path,
) -> dict[str, Any]:
    model_dir = pred_root / spec.model
    run_type = classify_run_type(spec.model)
    full_scope = spec.n_total == len(samples)
    if not model_dir.is_dir():
        return {
            "model": spec.model,
            "model_id": spec.model_id,
            "group": spec.group,
            "run_type": run_type,
            "comparison_status": classify_comparison_status(
                run_type=run_type,
                has_prediction_dir=False,
                full_scope=full_scope,
                n_valid_json=0,
                n_indexed=len(samples),
            ),
            "full_scope": full_scope,
            "n_total": spec.n_total,
            "n_prediction_files": 0,
            "n_extra_prediction_files": 0,
            "n_valid_json": 0,
            "coverage": 0.0 if spec.n_total else NA,
            "n_missing_prediction": spec.n_total,
            "n_invalid_json": 0,
            "Schema-nTED": 0.0 if spec.n_total else NA,
            "Schema-nTED-valid": NA,
            "Value-nED": 0.0 if spec.n_total else NA,
            "Value-nED-valid": NA,
        }

    if not full_scope:
        raise ValueError(
            f"run {spec.model!r} has n_total={spec.n_total}, but its prediction directory "
            f"requires the full {len(samples)}-sample index"
        )

    n_prediction_files = 0
    n_valid_json = 0
    n_invalid_json = 0
    schema_sum = 0.0
    value_sum = 0.0
    sample_ids = {sample.sample_id for sample in samples}

    for sample in samples:
        pred_path = model_dir / f"{sample.sample_id}.json"
        if not pred_path.exists():
            continue
        n_prediction_files += 1
        try:
            pred = read_json(pred_path)
        except Exception:
            n_invalid_json += 1
            continue
        n_valid_json += 1
        answer = unwrap_answer(pred)
        schema_sum += schema_nted(build_schema_tree(answer), sample.schema)
        value_sum += value_ned(extract_normalized_values(answer), sample.values)

    n_extra_prediction_files = sum(
        1 for path in model_dir.glob("*.json") if path.stem not in sample_ids
    )
    return {
        "model": spec.model,
        "model_id": spec.model_id,
        "group": spec.group,
        "run_type": run_type,
        "comparison_status": classify_comparison_status(
            run_type=run_type,
            has_prediction_dir=True,
            full_scope=full_scope,
            n_valid_json=n_valid_json,
            n_indexed=len(samples),
        ),
        "full_scope": full_scope,
        "n_total": spec.n_total,
        "n_prediction_files": n_prediction_files,
        "n_extra_prediction_files": n_extra_prediction_files,
        "n_valid_json": n_valid_json,
        "coverage": n_valid_json / spec.n_total if spec.n_total else NA,
        "n_missing_prediction": spec.n_total - n_prediction_files,
        "n_invalid_json": n_invalid_json,
        "Schema-nTED": schema_sum / spec.n_total if spec.n_total else NA,
        "Schema-nTED-valid": schema_sum / n_valid_json if n_valid_json else NA,
        "Value-nED": value_sum / spec.n_total if spec.n_total else NA,
        "Value-nED-valid": value_sum / n_valid_json if n_valid_json else NA,
    }


def _init_worker(index_path: str) -> None:
    global _WORKER_SAMPLES
    _WORKER_SAMPLES = load_document_samples(Path(index_path))


def _evaluate_run_worker(spec: RunSpec, pred_root: str) -> dict[str, Any]:
    if not _WORKER_SAMPLES:
        raise RuntimeError("document metric worker was not initialized")
    return evaluate_run(spec, _WORKER_SAMPLES, Path(pred_root))


def _format_value(value: Any) -> str:
    if value == NA or value is None:
        return NA
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_results(out_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    with (out_dir / "document_similarity_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in RESULT_COLUMNS})
    write_json(out_dir / "document_similarity_results_metadata.json", metadata)

    lines = [
        "# Document Similarity Results",
        "",
        "Both metrics are page-level similarities in [0, 1]; higher is better. Missing and invalid predictions score zero in the primary columns.",
        "",
        "| Run | Type | Status | Valid/Total | Schema-nTED | Schema-nTED (valid) | Value-nED | Value-nED (valid) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["run_type"]),
                    str(row["comparison_status"]),
                    f"{row['n_valid_json']}/{row['n_total']}",
                    _format_value(row["Schema-nTED"]),
                    _format_value(row["Schema-nTED-valid"]),
                    _format_value(row["Value-nED"]),
                    _format_value(row["Value-nED-valid"]),
                ]
            )
            + " |"
        )
    (out_dir / "document_similarity_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute schema tree-edit and value edit-distance similarities for existing predictions."
    )
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--main-results", default="outputs/main_exp/main_results.csv")
    parser.add_argument("--out", default="outputs/main_exp")
    parser.add_argument("--models", default="", help="Optional comma-separated run ids.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    index_path = Path(args.index)
    pred_root = Path(args.pred_root)
    main_results_path = Path(args.main_results)
    n_indexed = len(read_jsonl(index_path))
    specs = load_run_specs(main_results_path, pred_root, n_indexed)
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    if requested:
        available = {spec.model for spec in specs}
        missing = requested - available
        if missing:
            raise ValueError(f"requested run ids not found: {', '.join(sorted(missing))}")
        specs = [spec for spec in specs if spec.model in requested]

    by_model: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(index_path),),
    ) as executor:
        futures = {
            executor.submit(_evaluate_run_worker, spec, str(pred_root)): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            row = future.result()
            by_model[spec.model] = row
            print(
                f"[Document metrics] {spec.model}: "
                f"Schema-nTED={_format_value(row['Schema-nTED'])} "
                f"Value-nED={_format_value(row['Value-nED'])}"
            )

    rows = [by_model[spec.model] for spec in specs]
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_path": str(index_path),
        "pred_root": str(pred_root),
        "main_results_path": str(main_results_path),
        "n_indexed": n_indexed,
        "n_runs": len(rows),
        "dependencies": {
            "apted": version("apted"),
            "rapidfuzz": version("rapidfuzz"),
            "scipy": version("scipy"),
        },
        "primary_aggregation": "macro mean over the full run scope; missing and invalid pages are zero",
        "valid_only_diagnostic": "macro mean over valid JSON predictions only; do not use for ranking",
        "Schema-nTED": {
            "direction": "higher is better",
            "formula": "1 - APTED(schema_pred, schema_gt) / (node_count_pred + node_count_gt)",
            "tree": "root plus key nodes and ordered list-item nodes; labels include key name and object/array/scalar kind; values are excluded",
            "dict_order": "ignored by sorting stripped keys before tree construction",
            "list_order": "preserved",
            "costs": "insert=1, delete=1, rename=0 for equal labels else 2",
        },
        "Value-nED": {
            "direction": "higher is better",
            "formula": "maximum one-to-one sum of normalized Levenshtein similarities / max(pred_value_count, gt_value_count)",
            "matching": "path-independent Hungarian assignment over non-empty normalized leaf values",
            "normalization": "NFKC, lowercase, whitespace collapse, and removal of separators between digits, matching VAcc",
            "extra_missing_values": "unmatched predicted or GT values contribute zero through the denominator",
        },
    }
    write_results(Path(args.out), rows, metadata)
    print(
        "wrote document similarity results -> "
        f"{Path(args.out) / 'document_similarity_results.csv'}"
    )


if __name__ == "__main__":
    main()
