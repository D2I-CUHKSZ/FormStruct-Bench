from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from formtsr_exp.io_utils import write_json, write_jsonl

from .core import Entity, Sample, load_samples


@dataclass
class _AnswerNode:
    children: dict[str, "_AnswerNode"] = field(default_factory=dict)
    values: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the fixed SRFUND slice into the FormStruct answer schema."
    )
    parser.add_argument("--dataset-root", default="raw/srfund/extracted/dataset")
    parser.add_argument("--output-dir", default="outputs/srfund_formstruct_aligned_dataset")
    parser.add_argument("--split", default="validation_balanced")
    parser.add_argument("--unavailable-per-language", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _visible_text(entity: Entity) -> str:
    return " ".join(entity.text.split())


def answer_records(sample: Sample) -> list[tuple[tuple[str, ...], str]]:
    by_id = {entity.entity_id: entity for entity in sample.entities}
    parent_by_child = {child: parent for parent, child in sample.hierarchy_edges}
    records: list[tuple[tuple[str, ...], str]] = []
    answers = sorted(
        (entity for entity in sample.entities if entity.label == "answer"),
        key=lambda entity: (entity.bbox[1], entity.bbox[0], entity.entity_id),
    )
    for answer in answers:
        ancestry: list[Entity] = []
        seen = {answer.entity_id}
        current = answer.entity_id
        while current in parent_by_child:
            parent_id = parent_by_child[current]
            if parent_id in seen:
                raise ValueError(f"cycle in SRFUND hierarchy: {sample.sample_id}/{parent_id}")
            seen.add(parent_id)
            parent = by_id[parent_id]
            ancestry.append(parent)
            current = parent_id
        path: list[str] = []
        for entity in reversed(ancestry):
            if entity.label not in {"header", "question"}:
                continue
            name = _visible_text(entity)
            if name and (not path or path[-1] != name):
                path.append(name)
        records.append((tuple(path or ("answer",)), _visible_text(answer)))
    return records


def _render_node(node: _AnswerNode) -> Any:
    rendered = {key: _render_node(child) for key, child in node.children.items()}
    if not rendered:
        if len(node.values) == 1:
            return node.values[0]
        return list(node.values)
    if node.values:
        value_key = "value"
        suffix = 2
        while value_key in rendered:
            value_key = f"value_{suffix}"
            suffix += 1
        rendered[value_key] = node.values[0] if len(node.values) == 1 else list(node.values)
    return rendered


def sample_to_formstruct_answer(sample: Sample) -> dict[str, Any]:
    root = _AnswerNode()
    for path, value in answer_records(sample):
        node = root
        for key in path:
            node = node.children.setdefault(key, _AnswerNode())
        node.values.append(value)
    return {key: _render_node(node) for key, node in root.children.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_aligned_dataset(samples: Sequence[Sample], output_dir: Path) -> Path:
    label_dir = output_dir / "labels"
    index_path = output_dir / "index.jsonl"
    rows: list[dict[str, Any]] = []
    n_answers = 0
    n_empty_answers = 0
    for sample in samples:
        records = answer_records(sample)
        n_answers += len(records)
        n_empty_answers += sum(not value for _, value in records)
        label_path = label_dir / f"{sample.sample_id}.json"
        write_json(label_path, sample_to_formstruct_answer(sample))
        rows.append(
            {
                "sample_id": sample.sample_id,
                "template_name": f"SRFUND-{sample.language}",
                "instance_id": Path(sample.image_name).stem,
                "image_path": str(sample.image_path),
                "label_path": str(label_path),
                "language": sample.language,
                "source_split": sample.split,
                "dataset": "SRFUND",
            }
        )
    write_jsonl(index_path, rows)
    write_json(
        output_dir / "conversion_protocol.json",
        {
            "source": "SRFUND instance_annotation + relation_annotation",
            "target": "FormStruct nested answer JSON",
            "n_samples": len(samples),
            "n_answers": n_answers,
            "n_empty_answers": n_empty_answers,
            "rules": [
                "Use the unique SRFUND hierarchy parent chain for every answer.",
                "Retain non-empty header and question text as nested semantic keys.",
                "Ignore other-label ancestors because they are outside the shared schema.",
                "Represent repeated values at one semantic path as an ordered array.",
                "Represent nodes with both values and children using a value member.",
            ],
            "leakage_control": "Converted labels are used only by offline evaluation; inference receives image_path and the fixed FormStruct prompt.",
            "index_sha256": _sha256(index_path),
        },
    )
    return index_path


def main() -> None:
    args = parse_args()
    samples = load_samples(
        Path(args.dataset_root),
        args.split,
        unavailable_limit=args.unavailable_per_language,
        seed=args.seed,
    )
    index_path = write_aligned_dataset(samples, Path(args.output_dir))
    print(f"wrote {len(samples)} aligned SRFUND samples -> {index_path}")


if __name__ == "__main__":
    main()
