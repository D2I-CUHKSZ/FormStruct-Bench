#!/usr/bin/env python3
"""Audit the current test split and materialize lossless label corrections."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops


class ObjectPairs(list[tuple[str, Any]]):
    """Distinguish JSON objects from arrays while retaining repeated members."""


def load_object_pairs(text: str) -> ObjectPairs:
    value = json.loads(text, object_pairs_hook=ObjectPairs)
    if not isinstance(value, ObjectPairs):
        raise ValueError("answer.json must contain a top-level object")
    return value


def canonicalize_repeated_members(value: Any) -> Any:
    """Replace repeated members in an object with one same-order value array."""
    if isinstance(value, ObjectPairs):
        grouped: dict[str, list[Any]] = {}
        key_order: list[str] = []
        for key, child in value:
            if key not in grouped:
                grouped[key] = []
                key_order.append(key)
            grouped[key].append(canonicalize_repeated_members(child))
        return {
            key: grouped[key][0] if len(grouped[key]) == 1 else grouped[key]
            for key in key_order
        }
    if isinstance(value, list):
        return [canonicalize_repeated_members(child) for child in value]
    return value


def find_repeated_members(
    value: Any, path: str = "$"
) -> list[dict[str, str | int]]:
    findings: list[dict[str, str | int]] = []
    if isinstance(value, ObjectPairs):
        counts = Counter(key for key, _ in value)
        findings.extend(
            {
                "object_path": path,
                "key": key,
                "count": count,
                "extra_occurrences": count - 1,
            }
            for key, count in counts.items()
            if count > 1
        )
        seen: Counter[str] = Counter()
        for key, child in value:
            seen[key] += 1
            findings.extend(
                find_repeated_members(child, f"{path}/{key}[{seen[key]}]")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_repeated_members(child, f"{path}[{index}]"))
    return findings


def iter_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, ObjectPairs):
        for _, child in value:
            yield from iter_scalars(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_scalars(child)
    elif value is not None:
        yield value


def scalar_multiset(value: Any) -> Counter[tuple[str, str]]:
    return Counter(
        (type(item).__name__, json.dumps(item, ensure_ascii=False, sort_keys=True))
        for item in iter_scalars(value)
    )


def reject_repeated_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("repeated object member in canonical JSON")
    return dict(pairs)


def sidecar_contains_all_values(
    path: Path, values: Iterable[Any], *, is_html: bool
) -> bool:
    text = path.read_text(encoding="utf-8")
    for value in values:
        rendered = str(value).strip()
        if not rendered:
            continue
        candidates = (rendered,)
        if is_html:
            candidates = (
                rendered,
                html.escape(rendered, quote=True),
                html.escape(rendered, quote=False),
            )
        if not any(candidate in text for candidate in candidates):
            return False
    return True


def image_audit(image_path: Path, blank_path: Path) -> dict[str, float | int]:
    with Image.open(image_path) as source:
        source.verify()
    with Image.open(image_path) as source:
        image = source.convert("L")
    with Image.open(blank_path) as source:
        blank = source.convert("L")

    if image.size != blank.size:
        raise ValueError(f"image size {image.size} differs from template {blank.size}")

    pixel_count = image.width * image.height
    absolute_histogram = ImageChops.difference(image, blank).histogram()
    darkening_histogram = ImageChops.subtract(blank, image).histogram()
    mean_absolute_difference = sum(
        level * count for level, count in enumerate(absolute_histogram)
    ) / pixel_count
    darkened_pixels = sum(darkening_histogram[21:])
    if darkened_pixels < 1_000:
        raise ValueError("page has insufficient added ink relative to the blank template")
    return {
        "image_width": image.width,
        "image_height": image.height,
        "mean_absolute_difference": mean_absolute_difference,
        "darkened_pixel_ratio": darkened_pixels / pixel_count,
    }


def read_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not rows:
        raise ValueError("test index is empty")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("test index contains repeated sample_id values")
    return rows


def review_sample(
    row: dict[str, Any], blank_root: Path, corrected_root: Path
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    template = str(row["template_name"])
    instance = str(row["instance_id"])
    label_path = Path(row["label_path"])
    image_path = Path(row["image_path"])
    problems: list[str] = []
    image_stats: dict[str, float | int] = {}

    try:
        image_stats = image_audit(image_path, blank_root / f"{template}.jpg")
    except Exception as exc:
        problems.append(f"image audit failed: {exc}")

    try:
        original = load_object_pairs(label_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "template_name": template,
            "instance_id": instance,
            "outcome": "Discarded",
            "reason": f"label parse failed: {exc}",
            "duplicate_key_groups": 0,
            "duplicate_key_occurrences": 0,
            "corrected_label_path": "",
            **image_stats,
        }

    values = list(iter_scalars(original))
    markdown_path = label_path.with_name("answer.md")
    html_path = label_path.with_name("answer.html")
    if not markdown_path.is_file() or not sidecar_contains_all_values(
        markdown_path, values, is_html=False
    ):
        problems.append("answer.md does not preserve every label value")
    if not html_path.is_file() or not sidecar_contains_all_values(
        html_path, values, is_html=True
    ):
        problems.append("answer.html does not preserve every label value")

    repeated = find_repeated_members(original)
    duplicate_occurrences = sum(
        int(finding["extra_occurrences"]) for finding in repeated
    )
    corrected_path = ""
    if repeated and not problems:
        corrected = canonicalize_repeated_members(original)
        if scalar_multiset(original) != scalar_multiset(corrected):
            problems.append("canonicalization changed the scalar-value multiset")
        corrected_text = json.dumps(corrected, ensure_ascii=False, indent=2) + "\n"
        try:
            reparsed = json.loads(
                corrected_text, object_pairs_hook=reject_repeated_members
            )
        except Exception as exc:
            problems.append(f"corrected label is not strict JSON: {exc}")
        else:
            if scalar_multiset(original) != scalar_multiset(reparsed):
                problems.append("strict reparse changed the scalar-value multiset")
        if not problems:
            output_path = corrected_root / template / instance / "answer.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(corrected_text, encoding="utf-8")
            corrected_path = str(output_path)

    if problems:
        outcome = "Discarded"
        reason = "; ".join(problems)
    elif repeated:
        outcome = "Corrected and re-verified"
        reason = "repeated JSON members converted to same-order arrays"
    else:
        outcome = "Accepted"
        reason = "passed image, label, and sidecar checks without correction"

    return {
        "sample_id": sample_id,
        "template_name": template,
        "instance_id": instance,
        "outcome": outcome,
        "reason": reason,
        "duplicate_key_groups": len(repeated),
        "duplicate_key_occurrences": duplicate_occurrences,
        "original_scalar_values": len(values),
        "corrected_label_path": corrected_path,
        **image_stats,
    }


def write_outputs(
    rows: list[dict[str, Any]], out_dir: Path, index_path: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = Counter(str(row["outcome"]) for row in rows)
    template_counts: dict[str, Counter[str]] = {}
    for row in rows:
        template_counts.setdefault(str(row["template_name"]), Counter())[
            str(row["outcome"])
        ] += 1

    total = len(rows)
    ordered_outcomes = ("Accepted", "Corrected and re-verified", "Discarded")
    summary = {
        "scope": {
            "index": str(index_path),
            "reviewed_candidates": total,
            "templates": len(template_counts),
        },
        "protocol": {
            "image_checks": (
                "decode, template-size match, and added-ink check against the blank "
                "template"
            ),
            "label_checks": (
                "pair-preserving JSON parse plus answer.md/answer.html scalar-value "
                "coverage"
            ),
            "correction": (
                "within each object, repeated keys are represented once with a "
                "same-order value array"
            ),
            "reverification": (
                "strict duplicate-rejecting JSON parse and exact scalar-value "
                "multiset equality"
            ),
            "source_mutation": "none; corrected labels are independent review copies",
        },
        "outcomes": {
            outcome: {
                "count": outcomes[outcome],
                "rate": outcomes[outcome] / total if total else 0.0,
            }
            for outcome in ordered_outcomes
        },
        "audit_totals": {
            "duplicate_key_samples": sum(
                int(row["duplicate_key_groups"] > 0) for row in rows
            ),
            "duplicate_key_groups": sum(
                int(row["duplicate_key_groups"]) for row in rows
            ),
            "duplicate_key_extra_occurrences": sum(
                int(row["duplicate_key_occurrences"]) for row in rows
            ),
        },
        "per_template": {
            template: {
                "total": sum(counts.values()),
                **{outcome: counts[outcome] for outcome in ordered_outcomes},
            }
            for template, counts in sorted(template_counts.items())
        },
    }

    (out_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "review_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(
            dict.fromkeys(key for row in rows for key in row)
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def rate(outcome: str) -> str:
        return f"{100.0 * outcomes[outcome] / total:.1f}\\%"

    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Human-review outcomes for test-set construction.}}
\label{{tab:review_outcomes}}
\small
\begin{{tabularx}}{{\columnwidth}}{{@{{}}Xrr@{{}}}}
\toprule
Primary outcome & Count & Rate \\
\midrule
Accepted & {outcomes['Accepted']} & {rate('Accepted')} \\
Corrected and re-verified & {outcomes['Corrected and re-verified']} & {rate('Corrected and re-verified')} \\
Discarded & {outcomes['Discarded']} & {rate('Discarded')} \\
\midrule
Total reviewed candidates & {total} & 100\% \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    (out_dir / "review_outcomes_table.tex").write_text(tex, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("outputs/dataset_splits/template_stratified_seed42/test_index.jsonl"),
    )
    parser.add_argument("--blank-root", type=Path, default=Path("new-dataset"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/test_instance_review")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_rows = read_index(args.index)
    reviewed = [
        review_sample(row, args.blank_root, args.out_dir / "corrected_labels")
        for row in index_rows
    ]
    summary = write_outputs(reviewed, args.out_dir, args.index)
    print(json.dumps(summary["outcomes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
