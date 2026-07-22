from __future__ import annotations

from pathlib import Path

from formtsr_exp.metrics import flatten_leaf_fields
from srfund_exp.build_formstruct_aligned import (
    answer_records,
    sample_to_formstruct_answer,
)
from srfund_exp.core import Entity, Sample
from srfund_exp.run_aligned_queue import completion_counts


def _sample(entities: tuple[Entity, ...], edges: set[tuple[str, str]]) -> Sample:
    return Sample(
        sample_id="en__sample",
        language="en",
        split="validation",
        image_name="sample.jpg",
        image_path=Path("sample.jpg"),
        width=1000,
        height=1000,
        entities=entities,
        links=frozenset(edges),
        hierarchy_edges=frozenset(edges),
    )


def test_converts_header_question_answer_path() -> None:
    sample = _sample(
        (
            Entity("h", "header", (0, 0, 100, 20), "Invoice"),
            Entity("q", "question", (0, 20, 50, 40), "Invoice number"),
            Entity("a", "answer", (50, 20, 100, 40), "A-123"),
        ),
        {("h", "q"), ("q", "a")},
    )
    assert answer_records(sample) == [(('Invoice', 'Invoice number'), 'A-123')]
    assert sample_to_formstruct_answer(sample) == {
        "Invoice": {"Invoice number": "A-123"}
    }


def test_repeated_values_become_an_ordered_array() -> None:
    sample = _sample(
        (
            Entity("q", "question", (0, 0, 50, 20), "Option"),
            Entity("a1", "answer", (50, 0, 100, 20), "A"),
            Entity("a2", "answer", (50, 20, 100, 40), "B"),
        ),
        {("q", "a1"), ("q", "a2")},
    )
    assert sample_to_formstruct_answer(sample) == {"Option": ["A", "B"]}


def test_parent_value_and_children_are_both_preserved() -> None:
    sample = _sample(
        (
            Entity("q1", "question", (0, 0, 50, 20), "Address"),
            Entity("a1", "answer", (50, 0, 100, 20), "Main Street"),
            Entity("q2", "question", (0, 20, 50, 40), "City"),
            Entity("a2", "answer", (50, 20, 100, 40), "Paris"),
        ),
        {("q1", "a1"), ("q1", "q2"), ("q2", "a2")},
    )
    answer = sample_to_formstruct_answer(sample)
    assert answer == {"Address": {"City": "Paris", "value": "Main Street"}}
    assert len(flatten_leaf_fields(answer)) == 2


def test_completion_counts_requires_prediction_or_explicit_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    pred_dir = output_dir / "pred" / "model"
    error_dir = output_dir / "errors"
    pred_dir.mkdir(parents=True)
    error_dir.mkdir(parents=True)
    (pred_dir / "a.json").write_text("{}", encoding="utf-8")
    (error_dir / "model.jsonl").write_text(
        '{"sample_id":"b","status":"invalid_json"}\n', encoding="utf-8"
    )
    counts = completion_counts(output_dir, "model", {"a", "b", "c"})
    assert counts == {
        "n_total": 3,
        "n_valid_json": 1,
        "n_explicit_error": 1,
        "n_attempted": 2,
        "n_pending": 1,
        "coverage": 1 / 3,
    }
