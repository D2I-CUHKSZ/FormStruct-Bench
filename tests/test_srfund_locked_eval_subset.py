from __future__ import annotations

from srfund_exp.build_locked_eval_subset import select_balanced_locked_subset
from srfund_exp.core import LANGUAGES


def _rows(count: int = 8) -> list[dict[str, str]]:
    return [
        {"sample_id": f"{language}__{index}", "language": language}
        for language in LANGUAGES
        for index in range(count)
    ]


def test_locked_subset_is_balanced_deterministic_and_order_independent() -> None:
    rows = _rows()
    selected = select_balanced_locked_subset(rows, per_language=3, seed=47)
    repeated = select_balanced_locked_subset(
        list(reversed(rows)), per_language=3, seed=47
    )
    assert selected == repeated
    assert len(selected) == 3 * len(LANGUAGES)
    assert len({row["sample_id"] for row in selected}) == len(selected)
    assert all(
        sum(row["language"] == language for row in selected) == 3
        for language in LANGUAGES
    )
