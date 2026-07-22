from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import make_sample_id
from .io_utils import read_json, write_json, write_jsonl


LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def _split_csv(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _find_clean_sample(clean_root: Path, template_name: str, instance_id: str) -> dict[str, str] | None:
    sample_dir = clean_root / template_name / instance_id
    image_path = sample_dir / f"{template_name}-{instance_id}.png"
    label_path = sample_dir / "answer.json"
    if image_path.exists() and label_path.exists():
        return {"image_path": str(image_path), "label_path": str(label_path)}
    return None


def _load_meta_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        meta = read_json(path)
    except Exception:
        return {"meta_parse_error": True}
    if not isinstance(meta, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("seed", "variant", "level", "metrics_before", "metrics_after", "metrics_delta", "diff_metrics"):
        if key in meta:
            summary[key] = meta[key]
    params = meta.get("params")
    if isinstance(params, dict):
        summary["params"] = {
            key: value
            for key, value in params.items()
            if key in {"operations", "level", "level_config", "contrast", "brightness", "gaussian_blur_sigma", "jpeg_quality"}
        }
    return summary


def scan_robustness_dataset(
    *,
    clean_root: Path,
    augment_root: Path,
    templates: set[str] | None = None,
    variants: set[str] | None = None,
    levels: set[str] | None = None,
    limit_base: int | None = None,
    limit_degraded: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    include_meta_summary: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean_root = clean_root.expanduser()
    augment_root = augment_root.expanduser()
    if not clean_root.exists():
        raise FileNotFoundError(f"clean data root not found: {clean_root}")
    if not augment_root.exists():
        raise FileNotFoundError(f"augment root not found: {augment_root}")

    candidates: list[tuple[str, str, str, str, Path]] = []
    skipped = Counter()
    for png_path in sorted(augment_root.rglob("*.png")):
        rel = png_path.relative_to(augment_root).parts
        if len(rel) != 5:
            skipped["non_variant_png"] += 1
            continue
        template_name, instance_id, variant, level, _filename = rel
        if templates and template_name not in templates:
            continue
        if variants and variant not in variants:
            continue
        if levels and level not in levels:
            continue
        label_path = png_path.parent / "answer.json"
        if not label_path.exists():
            skipped["missing_aug_answer"] += 1
            continue
        clean_sample = _find_clean_sample(clean_root, template_name, instance_id)
        if clean_sample is None:
            skipped["missing_clean_pair"] += 1
            continue
        candidates.append((template_name, instance_id, variant, level, png_path))

    base_pairs = sorted({(template_name, instance_id) for template_name, instance_id, _variant, _level, _path in candidates})
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(base_pairs)
    if limit_base is not None:
        base_pairs = base_pairs[:limit_base]
    selected_base_pairs = set(base_pairs)

    clean_rows: list[dict[str, Any]] = []
    for template_name, instance_id in sorted(selected_base_pairs):
        clean_sample = _find_clean_sample(clean_root, template_name, instance_id)
        if clean_sample is None:
            continue
        clean_sample_id = make_sample_id(template_name, instance_id)
        clean_rows.append(
            {
                "sample_id": clean_sample_id,
                "clean_sample_id": clean_sample_id,
                "template_name": template_name,
                "instance_id": instance_id,
                "image_path": clean_sample["image_path"],
                "label_path": clean_sample["label_path"],
                "split": "robustness_clean",
                "visual_degradation": False,
                "degradation_variant": "clean",
                "degradation_level": "clean",
            }
        )

    degraded_rows: list[dict[str, Any]] = []
    for template_name, instance_id, variant, level, png_path in candidates:
        if (template_name, instance_id) not in selected_base_pairs:
            continue
        clean_sample_id = make_sample_id(template_name, instance_id)
        sample_id = f"{clean_sample_id}__deg__{variant}__{level}"
        meta_path = png_path.parent / "augment_meta.json"
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "clean_sample_id": clean_sample_id,
            "template_name": template_name,
            "instance_id": instance_id,
            "image_path": str(png_path),
            "label_path": str(png_path.parent / "answer.json"),
            "split": "robustness_degraded",
            "visual_degradation": True,
            "degradation_variant": variant,
            "degradation_level": level,
            "augment_meta_path": str(meta_path) if meta_path.exists() else None,
        }
        if include_meta_summary:
            row["augment_meta"] = _load_meta_summary(meta_path)
        degraded_rows.append(row)

    degraded_rows.sort(
        key=lambda row: (
            str(row["template_name"]),
            str(row["instance_id"]),
            str(row["degradation_variant"]),
            LEVEL_ORDER.get(str(row["degradation_level"]), 99),
            str(row["degradation_level"]),
        )
    )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(degraded_rows)
    if limit_degraded is not None:
        degraded_rows = degraded_rows[:limit_degraded]

    by_variant_level = Counter((str(row["degradation_variant"]), str(row["degradation_level"])) for row in degraded_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clean_root": str(clean_root),
        "augment_root": str(augment_root),
        "n_clean_base_samples": len(clean_rows),
        "n_degraded_samples": len(degraded_rows),
        "templates": sorted({str(row["template_name"]) for row in clean_rows}),
        "variants": sorted({str(row["degradation_variant"]) for row in degraded_rows}),
        "levels": sorted({str(row["degradation_level"]) for row in degraded_rows}, key=lambda item: LEVEL_ORDER.get(item, 99)),
        "by_variant_level": {
            f"{variant}/{level}": count
            for (variant, level), count in sorted(
                by_variant_level.items(),
                key=lambda item: (item[0][0], LEVEL_ORDER.get(item[0][1], 99), item[0][1]),
            )
        },
        "skipped": dict(skipped),
        "limit_base": limit_base,
        "limit_degraded": limit_degraded,
        "shuffle": shuffle,
        "seed": seed,
        "sample_id_rule": "degraded sample_id = {template}__{instance}__deg__{variant}__{level}; clean_sample_id preserves the paired clean id",
    }
    return clean_rows, degraded_rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FormTSR visual robustness clean/degraded indexes.")
    parser.add_argument("--clean-data-root", default="./FormTSR/datasets")
    parser.add_argument("--augment-root", default="./FormTSR/dataset-augment")
    parser.add_argument("--out-root", default="outputs/robustness_exp")
    parser.add_argument("--clean-out", default="", help="Override clean index path.")
    parser.add_argument("--degraded-out", default="", help="Override degraded index path.")
    parser.add_argument("--metadata-out", default="", help="Override metadata path.")
    parser.add_argument("--templates", default="", help="Comma-separated template names.")
    parser.add_argument("--variants", default="", help="Comma-separated degradation variants.")
    parser.add_argument("--levels", default="", help="Comma-separated levels, e.g. low,medium,high.")
    parser.add_argument("--limit-base", type=int, default=None, help="Limit paired clean base samples.")
    parser.add_argument("--limit-degraded", type=int, default=None, help="Limit degraded rows after filtering.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-meta-summary", action="store_true", help="Do not embed compact augment_meta summary in index rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    clean_out = Path(args.clean_out) if args.clean_out else out_root / "robustness_clean_index.jsonl"
    degraded_out = Path(args.degraded_out) if args.degraded_out else out_root / "robustness_degraded_index.jsonl"
    metadata_out = Path(args.metadata_out) if args.metadata_out else out_root / "robustness_index_metadata.json"
    clean_rows, degraded_rows, metadata = scan_robustness_dataset(
        clean_root=Path(args.clean_data_root),
        augment_root=Path(args.augment_root),
        templates=_split_csv(args.templates),
        variants=_split_csv(args.variants),
        levels=_split_csv(args.levels),
        limit_base=args.limit_base,
        limit_degraded=args.limit_degraded,
        shuffle=args.shuffle,
        seed=args.seed,
        include_meta_summary=not args.no_meta_summary,
    )
    write_jsonl(clean_out, clean_rows)
    write_jsonl(degraded_out, degraded_rows)
    write_json(metadata_out, metadata)
    print(f"wrote clean robustness index: {len(clean_rows)} -> {clean_out}")
    print(f"wrote degraded robustness index: {len(degraded_rows)} -> {degraded_out}")
    print(f"metadata -> {metadata_out}")


if __name__ == "__main__":
    main()
