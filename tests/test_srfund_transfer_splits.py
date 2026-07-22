from __future__ import annotations

from pathlib import Path

from srfund_exp.build_transfer_splits import stratified_split
from srfund_exp.core import Entity, LANGUAGES, Sample


def _sample(language: str, index: int) -> Sample:
    return Sample(
        sample_id=f"{language}__{index}",
        language=language,
        split="train",
        image_name=f"{language}_{index}.jpg",
        image_path=Path(f"{language}_{index}.jpg"),
        width=100,
        height=100,
        entities=(Entity("a", "answer", (0, 0, 10, 10), "value"),),
        links=frozenset(),
        hierarchy_edges=frozenset(),
    )


def test_stratified_split_is_balanced_disjoint_and_deterministic() -> None:
    candidates = [_sample(language, index) for language in LANGUAGES for index in range(6)]
    dev, locked = stratified_split(candidates, dev_per_language=2, seed=43)
    repeated_dev, repeated_locked = stratified_split(
        list(reversed(candidates)), dev_per_language=2, seed=43
    )
    dev_ids = {sample.sample_id for sample in dev}
    locked_ids = {sample.sample_id for sample in locked}
    assert len(dev) == 16
    assert len(locked) == 32
    assert not dev_ids & locked_ids
    assert dev_ids | locked_ids == {sample.sample_id for sample in candidates}
    assert [sample.sample_id for sample in dev] == [
        sample.sample_id for sample in repeated_dev
    ]
    assert [sample.sample_id for sample in locked] == [
        sample.sample_id for sample in repeated_locked
    ]
    assert all(sum(sample.language == language for sample in dev) == 2 for language in LANGUAGES)
