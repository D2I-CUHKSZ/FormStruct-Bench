from __future__ import annotations

from pathlib import Path

import pytest

from srfund_exp.core import Entity, Sample
from srfund_exp.transfer_figure import (
    CONDITIONS,
    METRICS,
    _plot,
    _plot_csv_rows,
    _summarize_transfer_pair,
    cluster_bootstrap_ci,
    evaluate_formtsr_prediction,
    evaluate_srfund_prediction,
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
            Entity("h", "header", (0, 0, 1000, 100), "Invoice"),
            Entity("q", "question", (0, 100, 400, 200), "Invoice number"),
            Entity("a", "answer", (400, 100, 800, 200), "A-123"),
        ),
        links=frozenset({("q", "a")}),
        hierarchy_edges=frozenset({("h", "q")}),
    )


def _prediction() -> dict:
    return {
        "entities": [
            {"id": "e1", "label": "header", "bbox": [0, 0, 1000, 100], "text": "Invoice"},
            {"id": "e2", "label": "question", "bbox": [0, 100, 400, 200], "text": "Invoice number"},
            {"id": "e3", "label": "answer", "bbox": [400, 100, 800, 200], "text": "A-123"},
        ],
        "links": [
            {"source": "e1", "target": "e2"},
            {"source": "e2", "target": "e3"},
        ],
    }


def test_srfund_perfect_prediction_scores_one() -> None:
    scores = evaluate_srfund_prediction(_sample(), _prediction())
    assert scores == pytest.approx(
        {"Value-nED": 1.0, "Schema-nTED": 1.0, "TSR-path": 1.0}
    )


def test_missing_predictions_score_zero() -> None:
    assert evaluate_srfund_prediction(_sample(), None) == {
        "Value-nED": 0.0,
        "Schema-nTED": 0.0,
        "TSR-path": 0.0,
    }
    assert evaluate_formtsr_prediction({"field": "value"}, None) == {
        "Value-nED": 0.0,
        "Schema-nTED": 0.0,
        "TSR-path": 0.0,
    }


def test_srfund_wrong_value_preserves_schema_but_breaks_value_and_path() -> None:
    prediction = _prediction()
    prediction["entities"][2]["text"] = "B-999"
    scores = evaluate_srfund_prediction(_sample(), prediction)
    assert scores["Schema-nTED"] == pytest.approx(1.0)
    assert scores["Value-nED"] < 1.0
    assert scores["TSR-path"] == 0.0


def test_cluster_bootstrap_is_deterministic_and_bounded() -> None:
    rows = [
        {"cluster": "a", "value": 0.0},
        {"cluster": "a", "value": 0.2},
        {"cluster": "b", "value": 0.8},
        {"cluster": "b", "value": 1.0},
    ]
    first = cluster_bootstrap_ci(
        rows, value_field="value", cluster_field="cluster", iterations=1000, seed=42
    )
    second = cluster_bootstrap_ci(
        rows, value_field="value", cluster_field="cluster", iterations=1000, seed=42
    )
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_transfer_pair_uses_paired_page_differences() -> None:
    base = [
        {"sample_id": "a", "cluster": "en", **{metric: 0.2 for metric in METRICS}},
        {"sample_id": "b", "cluster": "de", **{metric: 0.4 for metric in METRICS}},
    ]
    tuned = [
        {"sample_id": "a", "cluster": "en", **{metric: 0.3 for metric in METRICS}},
        {"sample_id": "b", "cluster": "de", **{metric: 0.5 for metric in METRICS}},
    ]
    rows = _summarize_transfer_pair(
        "qwen35",
        "Qwen3.5-9B",
        "schema_aligned",
        base,
        tuned,
        iterations=100,
        seed=42,
    )
    assert len(rows) == 3
    assert all(row["delta"] == pytest.approx(0.1) for row in rows)
    assert all(row["ci95_low"] == pytest.approx(0.1) for row in rows)
    assert all(row["ci95_high"] == pytest.approx(0.1) for row in rows)


def test_plot_writes_two_by_three_figure(tmp_path: Path) -> None:
    rows = []
    for family, model in (("qwen35", "Qwen3.5-9B"), ("qwen36", "Qwen3.6-35B-A3B")):
        for metric in METRICS:
            for index, condition in enumerate(CONDITIONS):
                mean = 0.2 + index * 0.1
                rows.append(
                    {
                        "family": family,
                        "model": model,
                        "metric": metric,
                        "condition": condition,
                        "mean": mean,
                        "ci95_low": mean - 0.02,
                        "ci95_high": mean + 0.02,
                    }
                )
    _plot(tmp_path / "figure", rows, ["qwen35", "qwen36"])
    assert (tmp_path / "figure.pdf").stat().st_size > 0
    assert (tmp_path / "figure.png").stat().st_size > 0


def test_plot_csv_has_one_row_per_bar_in_requested_order() -> None:
    condition_rows = []
    for family, model in (("qwen35", "Qwen3.5-9B"), ("qwen36", "Qwen3.6-35B-A3B")):
        for metric in METRICS:
            for condition, dataset, tuned in (
                ("pre_formtsr", "FormStruct", False),
                ("sft_formtsr", "FormStruct", True),
                ("pre_srfund", "SRFUND-Aligned", False),
                ("sft_srfund", "SRFUND-Aligned", True),
            ):
                condition_rows.append(
                    {
                        "family": family,
                        "model": model,
                        "metric": metric,
                        "condition": condition,
                        "condition_label": condition,
                        "dataset": dataset,
                        "tuned": tuned,
                        "n_samples": 10,
                        "n_clusters": 2,
                        "mean": 0.5,
                        "ci95_low": 0.4,
                        "ci95_high": 0.6,
                    }
                )
            for tuned in (False, True):
                condition_rows.append(
                    {
                        "family": family,
                        "model": model,
                        "metric": metric,
                        "condition": "sft_srfund" if tuned else "pre_srfund",
                        "condition_label": "native",
                        "dataset": "SRFUND-Native",
                        "tuned": tuned,
                        "n_samples": 10,
                        "n_clusters": 2,
                        "mean": 0.3,
                        "ci95_low": 0.2,
                        "ci95_high": 0.4,
                    }
                )
    rows = _plot_csv_rows(condition_rows, ["qwen35", "qwen36"])
    assert len(rows) == 36
    assert list(rows[0]) == [
        "model",
        "model_order",
        "metric",
        "metric_order",
        "checkpoint",
        "eval_dataset",
        "mean_score",
        "ci95_low",
        "ci95_high",
    ]
    assert [
        (row["checkpoint"], row["eval_dataset"]) for row in rows[:4]
    ] == [
        ("Pre-SFT", "FormStruct"),
        ("FormStruct-SFT", "FormStruct"),
        ("Pre-SFT", "SRFUND-Aligned"),
        ("FormStruct-SFT", "SRFUND-Aligned"),
    ]
    assert rows[0]["model_order"] == 1
    assert rows[0]["metric_order"] == 1
    assert rows[0]["mean_score"] == pytest.approx(50.0)
