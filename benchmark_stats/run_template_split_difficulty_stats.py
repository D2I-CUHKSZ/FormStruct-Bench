#!/usr/bin/env python3
"""Create template-disjoint splits and report corpus/test difficulty statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVELS = ("L1", "L2", "L3", "L4")
LEVEL_NAMES = {"L1": "Easy", "L2": "Medium", "L3": "Hard", "L4": "Expert"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build template-disjoint stratified splits and a difficulty table."
    )
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument(
        "--difficulty-csv",
        default="outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv",
    )
    parser.add_argument(
        "--thresholds-json",
        default="outputs/domain_stats/normal_calibrated_difficulty_thresholds.json",
    )
    parser.add_argument(
        "--structural-csv", default="reports/structural_complexity_samples.csv"
    )
    parser.add_argument(
        "--output", default="outputs/dataset_splits/template_stratified_seed42"
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_difficulty(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        template = Path(row["file"]).stem
        level = row["normal_calibrated_level"]
        if level not in LEVELS:
            raise ValueError(f"Unsupported difficulty level for {template}: {level}")
        output[template] = {
            "difficulty_level": level,
            "coarse_domain": row["coarse_domain"].strip(),
            "D_main": float(row["D_main"]),
            "S_form": float(row["S_form"]),
            "C_context": float(row["C_context"]),
        }
    return output


def read_region_counts(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        template = Path(row["file"]).stem
        value = float(row["region_count"])
        if not value.is_integer():
            raise ValueError(f"Non-integer region_count for {template}: {value}")
        output[template] = int(value)
    return output


def rounded_split_sizes(total: int, ratios: dict[str, float]) -> dict[str, int]:
    # Match the repository's existing split convention: test receives rounding residue.
    train = math.floor(total * ratios["train"])
    val = math.floor(total * ratios["val"])
    return {"train": train, "val": val, "test": total - train - val}


def stratified_quotas(
    level_counts: dict[str, int], ratios: dict[str, float], split_sizes: dict[str, int]
) -> dict[str, dict[str, int]]:
    raw = {
        level: {split: level_counts[level] * ratios[split] for split in SPLITS}
        for level in LEVELS
    }
    quotas = {
        level: {split: math.floor(raw[level][split]) for split in SPLITS}
        for level in LEVELS
    }
    row_need = {
        level: level_counts[level] - sum(quotas[level].values()) for level in LEVELS
    }
    col_need = {
        split: split_sizes[split] - sum(quotas[level][split] for level in LEVELS)
        for split in SPLITS
    }
    while sum(row_need.values()):
        candidates = [
            (raw[level][split] - quotas[level][split], -LEVELS.index(level), -SPLITS.index(split), level, split)
            for level in LEVELS
            for split in SPLITS
            if row_need[level] and col_need[split]
        ]
        if not candidates:
            raise RuntimeError("Could not construct integer stratified split quotas")
        _, _, _, level, split = max(candidates)
        quotas[level][split] += 1
        row_need[level] -= 1
        col_need[split] -= 1
    return quotas


def assign_templates(
    difficulty: dict[str, dict[str, Any]], quotas: dict[str, dict[str, int]], seed: int
) -> dict[str, str]:
    rng = random.Random(seed)
    by_level: dict[str, list[str]] = defaultdict(list)
    for template, info in difficulty.items():
        by_level[info["difficulty_level"]].append(template)
    assignments: dict[str, str] = {}
    for level in LEVELS:
        templates = sorted(by_level[level])
        rng.shuffle(templates)
        offset = 0
        for split in SPLITS:
            end = offset + quotas[level][split]
            assignments.update({template: split for template in templates[offset:end]})
            offset = end
        if offset != len(templates):
            raise RuntimeError(f"Incomplete assignment for {level}")
    return assignments


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def difficulty_ranges(payload: dict[str, Any]) -> dict[str, str]:
    thresholds = payload["thresholds"]
    q1 = float(thresholds["q15_8655"])
    q2 = float(thresholds["q50"])
    q3 = float(thresholds["q84_1345"])
    return {
        "L1": rf"$[0, {q1:.4f})$",
        "L2": rf"$[{q1:.4f}, {q2:.4f})$",
        "L3": rf"$[{q2:.4f}, {q3:.4f})$",
        "L4": rf"$[{q3:.4f}, 2.0000]$",
    }


def summarize(
    rows: list[dict[str, Any]],
    assignments: dict[str, str],
    difficulty: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    corpus_counts = Counter(difficulty[template]["difficulty_level"] for template in assignments)
    test_templates = [template for template, split in assignments.items() if split == "test"]
    test_counts = Counter(difficulty[template]["difficulty_level"] for template in test_templates)
    output = []
    for level in LEVELS:
        level_templates = [
            template for template, info in difficulty.items() if info["difficulty_level"] == level
        ]
        output.append(
            {
                "tier": level,
                "name": LEVEL_NAMES[level],
                "corpus_count": sum(
                    1 for row in rows if difficulty[row["template_name"]]["difficulty_level"] == level
                ),
                "corpus_pct": corpus_counts[level] / len(assignments) * 100,
                "test_count": sum(
                    1
                    for row in rows
                    if assignments[row["template_name"]] == "test"
                    and difficulty[row["template_name"]]["difficulty_level"] == level
                ),
                "test_pct": test_counts[level] / len(test_templates) * 100,
                "mean_S_form": sum(difficulty[t]["S_form"] for t in level_templates) / len(level_templates),
                "mean_C_context": sum(difficulty[t]["C_context"] for t in level_templates) / len(level_templates),
            }
        )
    return output


def write_latex(path: Path, summary: list[dict[str, Any]], ranges: dict[str, str]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Difficulty tier distribution across the full corpus and the template-disjoint test split. Mean complexity scores are computed over the full corpus within each tier.}",
        r"\label{tab:difficulty_tier_distribution}",
        r"\begin{tabular}{cllrrrr}",
        r"\toprule",
        r"Tier & Name & $D_{\mathrm{main}}$ Range & Corpus (\%) & Test (\%) & Mean $S_{\mathrm{form}}$ & Mean $C_{\mathrm{context}}$ \\",
        r"\midrule",
    ]
    for row in summary:
        level = row["tier"]
        lines.append(
            f"{level} & {row['name']} & {ranges[level]} & {row['corpus_pct']:.2f} & "
            f"{row['test_pct']:.2f} & {row['mean_S_form']:.4f} & "
            f"{row['mean_C_context']:.4f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_split_composition(
    rows: list[dict[str, Any]], assignments: dict[str, str]
) -> list[dict[str, Any]]:
    total_templates = len(assignments)
    total_instances = len(rows)
    display_names = {"train": "train", "val": "validation", "test": "test"}
    output = []
    for split in SPLITS:
        template_count = sum(value == split for value in assignments.values())
        instance_count = sum(assignments[row["template_name"]] == split for row in rows)
        # Only the test split is confirmed as fully human-reviewed.
        human_reviewed_count = instance_count if split == "test" else 0
        output.append(
            {
                "split": display_names[split],
                "template_count": template_count,
                "template_share_pct": template_count / total_templates * 100,
                "instance_count": instance_count,
                "instance_share_pct": instance_count / total_instances * 100,
                "instances_per_template": instance_count / template_count,
                "human_reviewed_count": human_reviewed_count,
                "human_review_rate_pct": human_reviewed_count / instance_count * 100,
            }
        )
    return output


def write_split_composition_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Composition of the template-disjoint dataset splits. Only instances with confirmed human review are counted as human-reviewed.}",
        r"\label{tab:split_composition}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Split & Templates & Template share (\%) & Instances & Instance share (\%) & Instances/template & Human-reviewed & Review rate (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['split']} & {row['template_count']} & {row['template_share_pct']:.2f} & "
            f"{row['instance_count']:,} & {row['instance_share_pct']:.2f} & "
            f"{row['instances_per_template']:.2f} & {row['human_reviewed_count']:,} & "
            f"{row['human_review_rate_pct']:.2f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_domain_composition(
    assignments: dict[str, str], difficulty: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    included_splits = ("train", "test")
    split_totals = Counter(assignments.values())
    domain_totals = Counter(info["coarse_domain"] for info in difficulty.values())
    domains = sorted(domain_totals, key=lambda domain: (-domain_totals[domain], domain))
    domain_order = {domain: index for index, domain in enumerate(domains, start=1)}
    counts = {
        split: Counter(
            difficulty[template]["coarse_domain"]
            for template, assigned_split in assignments.items()
            if assigned_split == split
        )
        for split in included_splits
    }

    output = []
    for domain in domains:
        shares = {
            split: counts[split][domain] / split_totals[split] * 100
            for split in included_splits
        }
        difference = shares["test"] - shares["train"]
        for split in included_splits:
            output.append(
                {
                    "split": split,
                    "coarse_domain": domain,
                    "template_count": counts[split][domain],
                    "split_template_total": split_totals[split],
                    "share_pct": round(shares[split], 2),
                    "test_minus_train_pp": round(difference, 2),
                    "domain_order": domain_order[domain],
                }
            )
    return output


def summarize_region_count_distribution(
    raw_rows: list[dict[str, Any]], split_totals: dict[str, int]
) -> list[dict[str, Any]]:
    included_splits = ("train", "test")
    counts = {
        split: Counter(
            row["region_count"] for row in raw_rows if row["split"] == split
        )
        for split in included_splits
    }
    output = []
    for split in included_splits:
        cumulative = 0
        for region_count in range(1, 10):
            template_count = counts[split][region_count]
            cumulative += template_count
            output.append(
                {
                    "split": split,
                    "region_count": region_count,
                    "template_count": template_count,
                    "split_template_total": split_totals[split],
                    "probability": round(template_count / split_totals[split], 6),
                    "cdf": round(cumulative / split_totals[split], 6),
                }
            )
    return output


def main() -> int:
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if any(value < 0 for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("train/val/test ratios must be non-negative and sum to 1")

    rows = read_jsonl(Path(args.index))
    difficulty = read_difficulty(Path(args.difficulty_csv))
    region_counts = read_region_counts(Path(args.structural_csv))
    indexed_templates = {str(row["template_name"]) for row in rows}
    if indexed_templates != set(difficulty):
        missing_scores = sorted(indexed_templates - set(difficulty))
        missing_index = sorted(set(difficulty) - indexed_templates)
        raise ValueError(
            f"Template mismatch: missing difficulty={missing_scores}, missing index={missing_index}"
        )
    if indexed_templates != set(region_counts):
        missing_regions = sorted(indexed_templates - set(region_counts))
        extra_regions = sorted(set(region_counts) - indexed_templates)
        raise ValueError(
            f"Region-count template mismatch: missing={missing_regions}, extra={extra_regions}"
        )

    level_counts = Counter(info["difficulty_level"] for info in difficulty.values())
    split_sizes = rounded_split_sizes(len(difficulty), ratios)
    quotas = stratified_quotas(level_counts, ratios, split_sizes)
    assignments = assign_templates(difficulty, quotas, args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_rows = {split: [] for split in SPLITS}
    for row in rows:
        split_rows[assignments[row["template_name"]]].append(row)
    for split in SPLITS:
        write_jsonl(output / f"{split}_index.jsonl", split_rows[split])

    assignment_rows = [
        {
            "template_name": template,
            "split": assignments[template],
            "difficulty_level": difficulty[template]["difficulty_level"],
            "difficulty_name": LEVEL_NAMES[difficulty[template]["difficulty_level"]],
            "D_main": difficulty[template]["D_main"],
            "S_form": difficulty[template]["S_form"],
            "C_context": difficulty[template]["C_context"],
            "instances": sum(1 for row in rows if row["template_name"] == template),
        }
        for template in sorted(assignments)
    ]
    write_csv(output / "template_assignments.csv", assignment_rows)

    summary = summarize(rows, assignments, difficulty)
    thresholds_payload = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))
    ranges = difficulty_ranges(thresholds_payload)
    csv_summary = [{**row, "D_main_range": ranges[row["tier"]]} for row in summary]
    write_csv(output / "difficulty_tier_distribution.csv", csv_summary)
    write_latex(output / "difficulty_tier_distribution_table.tex", summary, ranges)

    split_composition = summarize_split_composition(rows, assignments)
    write_csv(output / "split_composition.csv", split_composition)
    write_split_composition_latex(output / "split_composition_table.tex", split_composition)

    domain_composition = summarize_domain_composition(assignments, difficulty)
    write_csv(output / "domain_composition_train_test.csv", domain_composition)

    region_raw = [
        {
            "template_id": template,
            "split": assignments[template],
            "region_count": region_counts[template],
        }
        for template in sorted(assignments)
        if assignments[template] in {"train", "test"}
    ]
    write_csv(output / "region_count_templates_train_test.csv", region_raw)
    region_distribution = summarize_region_count_distribution(
        region_raw,
        {split: sum(value == split for value in assignments.values()) for split in SPLITS},
    )
    write_csv(output / "region_count_distribution_train_test.csv", region_distribution)

    metadata = {
        "seed": args.seed,
        "ratios": ratios,
        "templates_by_split": {split: sum(assignments[t] == split for t in assignments) for split in SPLITS},
        "instances_by_split": {split: len(split_rows[split]) for split in SPLITS},
        "tier_template_quotas": quotas,
        "template_disjoint": sum(
            len({row["template_name"] for row in split_rows[split]}) for split in SPLITS
        ) == len(assignments),
        "human_review_policy": {
            "confirmed_fully_reviewed_splits": ["test"],
            "unconfirmed_splits_counted_as_reviewed": False,
        },
        "difficulty_thresholds": thresholds_payload,
    }
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
