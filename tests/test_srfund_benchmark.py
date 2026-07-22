from __future__ import annotations

from pathlib import Path

import pytest

from srfund_exp.core import (
    Entity,
    Sample,
    evaluate_prediction,
    normalize_bbox,
    relation_tree_edges,
    text_similarity,
    validate_prediction,
)


def _sample() -> Sample:
    return Sample(
        sample_id="en__en_val_0",
        language="en",
        split="validation",
        image_name="en_val_0.jpg",
        image_path=Path("en_val_0.jpg"),
        width=1000,
        height=1000,
        entities=(
            Entity("1", "question", (100, 100, 300, 200), "Invoice number"),
            Entity("2", "answer", (320, 100, 500, 200), "A-123"),
        ),
        links=frozenset({("1", "2")}),
        hierarchy_edges=frozenset({("1", "2")}),
    )


def test_relation_tree_edges_ignores_named_group_nodes() -> None:
    tree = {"1": {"2": 3, "note": [4]}, "other": [5]}
    assert relation_tree_edges(tree) == frozenset({("1", "2"), ("2", "3"), ("1", "4")})


def test_bbox_normalization_and_text_similarity() -> None:
    assert normalize_bbox([10, 20, 60, 120], 100, 200) == (100.0, 100.0, 600.0, 600.0)
    assert text_similarity("  INV\u212a  12 ", "invk 12") == pytest.approx(1.0)


def test_perfect_prediction_scores_one() -> None:
    prediction = {
        "entities": [
            {"id": "e1", "label": "question", "bbox": [100, 100, 300, 200], "text": "Invoice number"},
            {"id": "e2", "label": "answer", "bbox": [320, 100, 500, 200], "text": "A-123"},
        ],
        "links": [{"source": "e1", "target": "e2"}],
    }
    valid, error = validate_prediction(prediction)
    assert valid and error is None
    row = evaluate_prediction(_sample(), prediction, model="test")
    assert row["entity_f1"] == pytest.approx(1.0)
    assert row["entity_e2e_f1"] == pytest.approx(1.0)
    assert row["entity_text_score"] == pytest.approx(1.0)
    assert row["link_f1"] == pytest.approx(1.0)
    assert row["hierarchy_f1"] == pytest.approx(1.0)


def test_missing_prediction_scores_gt_content_as_missed() -> None:
    row = evaluate_prediction(_sample(), None, model="test")
    assert row["valid_json"] is False
    assert row["entity_f1"] == 0.0
    assert row["entity_e2e_f1"] == 0.0
    assert row["link_f1"] == 0.0
    assert row["hierarchy_f1"] == 0.0


def test_duplicate_ids_and_invalid_link_endpoints_are_audited() -> None:
    prediction = {
        "entities": [
            {"id": "e1", "label": "question", "bbox": [100, 100, 300, 200], "text": "Invoice number"},
            {"id": "e1", "label": "answer", "bbox": [320, 100, 500, 200], "text": "A-123"},
        ],
        "links": [{"source": "e1", "target": "missing"}],
    }
    row = evaluate_prediction(_sample(), prediction, model="test")
    assert row["duplicate_entity_ids"] == 1
    assert row["invalid_link_endpoints"] == 1
