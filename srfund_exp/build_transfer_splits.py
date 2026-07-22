from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from formtsr_exp.io_utils import read_jsonl, write_json

from .build_formstruct_aligned import write_aligned_dataset
from .core import LANGUAGES, Sample, load_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed SRFUND development split and an untouched locked split."
    )
    parser.add_argument("--dataset-root", default="raw/srfund/extracted/dataset")
    parser.add_argument(
        "--diagnostic-index",
        default="outputs/srfund_formstruct_aligned_dataset/index.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="outputs/srfund_transfer_exploratory/splits"
    )
    parser.add_argument("--dev-per-language", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def _hash_order(sample: Sample, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample.sample_id}".encode("utf-8")).hexdigest()


def build_candidate_pool(dataset_root: Path, diagnostic_ids: set[str]) -> list[Sample]:
    official_train = load_samples(dataset_root, "train")
    english_unavailable = [
        sample
        for sample in load_samples(dataset_root, "all")
        if sample.language == "en" and sample.split == "unavailable"
    ]
    candidates = [
        sample
        for sample in (*official_train, *english_unavailable)
        if sample.sample_id not in diagnostic_ids
    ]
    ids = [sample.sample_id for sample in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in SRFUND transfer candidate pool")
    counts = Counter(sample.language for sample in candidates)
    if set(counts) != set(LANGUAGES):
        raise ValueError(f"candidate pool is missing languages: {counts}")
    return candidates


def stratified_split(
    candidates: list[Sample], *, dev_per_language: int, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if dev_per_language < 1:
        raise ValueError("dev_per_language must be positive")
    dev: list[Sample] = []
    locked: list[Sample] = []
    for language in LANGUAGES:
        language_samples = sorted(
            (sample for sample in candidates if sample.language == language),
            key=lambda sample: _hash_order(sample, seed),
        )
        if len(language_samples) <= dev_per_language:
            raise ValueError(
                f"not enough {language} samples for dev={dev_per_language}: "
                f"available={len(language_samples)}"
            )
        dev.extend(language_samples[:dev_per_language])
        locked.extend(language_samples[dev_per_language:])
    return dev, locked


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    diagnostic_index = Path(args.diagnostic_index)
    diagnostic_ids = {
        str(row["sample_id"]) for row in read_jsonl(diagnostic_index)
    }
    candidates = build_candidate_pool(Path(args.dataset_root), diagnostic_ids)
    dev, locked = stratified_split(
        candidates,
        dev_per_language=args.dev_per_language,
        seed=args.seed,
    )
    dev_index = write_aligned_dataset(dev, output_dir / "dev")
    locked_index = write_aligned_dataset(locked, output_dir / "locked")
    protocol = {
        "status": "preregistered_before_checkpoint_sweep",
        "purpose": {
            "dev": "checkpoint selection and exploratory trend visualization only",
            "locked": "untouched final evaluation; never use for model or hyperparameter selection",
            "diagnostic": "previously inspected 400-page slice; excluded from both new splits",
        },
        "selection_rule": (
            "Among preregistered checkpoints, maximize the unweighted mean of paired "
            "SFT gains in Value-nED, Schema-nTED, and TSR-path on dev. Break ties by "
            "TSR-path gain, then by the earlier training step."
        ),
        "seed": args.seed,
        "hash_order": "SHA256(f'{seed}:{sample_id}') within each language",
        "dev_per_language": args.dev_per_language,
        "n_diagnostic_excluded": len(diagnostic_ids),
        "n_candidates": len(candidates),
        "n_dev": len(dev),
        "n_locked": len(locked),
        "candidate_by_language": dict(sorted(Counter(s.language for s in candidates).items())),
        "dev_by_language": dict(sorted(Counter(s.language for s in dev).items())),
        "locked_by_language": dict(sorted(Counter(s.language for s in locked).items())),
        "dev_index": str(dev_index),
        "dev_index_sha256": _sha256(dev_index),
        "locked_index": str(locked_index),
        "locked_index_sha256": _sha256(locked_index),
        "diagnostic_index": str(diagnostic_index),
        "diagnostic_index_sha256": _sha256(diagnostic_index),
        "real_labels_used_for_training": False,
    }
    write_json(output_dir / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
