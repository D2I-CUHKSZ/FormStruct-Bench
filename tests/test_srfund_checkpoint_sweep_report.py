from __future__ import annotations

from srfund_exp.transfer_figure import evaluate_formtsr_prediction
from srfund_exp.checkpoint_sweep_report import select_checkpoint


def test_select_checkpoint_uses_preregistered_tie_breaks() -> None:
    assert select_checkpoint(
        {0: 0.0, 100: 0.1, 200: 0.2, 300: 0.15},
        {0: 0.0, 100: 0.2, 200: 0.1, 300: 0.3},
    ) == 200
    assert select_checkpoint(
        {0: 0.0, 100: 0.2, 200: 0.2, 300: 0.1},
        {0: 0.0, 100: 0.1, 200: 0.3, 300: 0.4},
    ) == 200
    assert select_checkpoint(
        {0: 0.0, 100: 0.2, 200: 0.2, 300: 0.1},
        {0: 0.0, 100: 0.3, 200: 0.3, 300: 0.4},
    ) == 100


def test_invalid_prediction_scores_zero_on_every_selection_metric() -> None:
    scores = evaluate_formtsr_prediction({}, None)
    assert scores == {
        "Value-nED": 0.0,
        "Schema-nTED": 0.0,
        "TSR-path": 0.0,
    }
