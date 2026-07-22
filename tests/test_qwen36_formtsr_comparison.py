from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_qwen36_formtsr_comparison import (
    _build_analysis,
    _load_aggregate_report_comparison,
    _load_relation_type_comparison,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_row(model: str, **values: str) -> dict[str, str]:
    return {
        "model": model,
        "model_id": model,
        "group": "vlm",
        "run_type": "raw",
        "comparison_status": "comparable_raw",
        "bbox_source_space": "normalized_1000",
        "sample_scope": "full_index",
        **values,
    }


class Qwen36ComparisonSupplementTest(unittest.TestCase):
    def test_aggregate_comparison_filters_and_keeps_finite_common_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tuned = root / "tuned"
            base = root / "base"
            _write_csv(
                tuned / "corrected_structure_metrics.csv",
                [
                    _aggregate_row(
                        "tuned",
                        n_total="10",
                        **{
                            "R-F1": "0.4",
                            "region_bbox_declared": "12",
                            "only_tuned": "7",
                            "not_finite": "inf",
                            "not_numeric": "NA",
                        },
                    )
                ],
            )
            _write_csv(
                base / "corrected_structure_metrics.csv",
                [
                    _aggregate_row(
                        "base",
                        n_total="10",
                        **{
                            "R-F1": "0.3",
                            "region_bbox_declared": "9",
                            "only_base": "8",
                            "not_finite": "0.2",
                            "not_numeric": "0.1",
                        },
                    )
                ],
            )
            _write_csv(
                tuned / "hierarchical_structure_metrics.csv",
                [_aggregate_row("tuned", **{"Rel-F1-micro": "0.2"})],
            )
            _write_csv(
                base / "hierarchical_structure_metrics.csv",
                [_aggregate_row("base", **{"Rel-F1-micro": "0.1"})],
            )

            rows = _load_aggregate_report_comparison(
                tuned_dir=tuned,
                tuned_model="tuned",
                base_dir=base,
                base_model="base",
            )

        by_key = {(row["report"], row["field"]): row for row in rows}
        self.assertEqual(set(by_key), {
            ("corrected_structure_metrics", "n_total"),
            ("corrected_structure_metrics", "R-F1"),
            ("corrected_structure_metrics", "region_bbox_declared"),
            ("hierarchical_structure_metrics", "Rel-F1-micro"),
        })
        self.assertEqual(by_key[("corrected_structure_metrics", "n_total")]["value_kind"], "count")
        self.assertEqual(by_key[("corrected_structure_metrics", "region_bbox_declared")]["delta"], 3)
        self.assertAlmostEqual(by_key[("corrected_structure_metrics", "R-F1")]["delta"], 0.1)

    def test_relation_type_comparison_requires_same_domain_and_gt(self) -> None:
        def relation_rows(model: str, key_value_gt: str = "10") -> list[dict[str, str]]:
            return [
                {
                    "model": model,
                    "relation_type": "key-value",
                    "TP": "2",
                    "pred": "4",
                    "GT": key_value_gt,
                    "precision": "0.5",
                    "recall": "0.2",
                    "F1": "0.285714",
                },
                {
                    "model": model,
                    "relation_type": "parent-child",
                    "TP": "1",
                    "pred": "3",
                    "GT": "5",
                    "precision": "0.333333",
                    "recall": "0.2",
                    "F1": "0.25",
                },
            ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tuned = root / "tuned"
            base = root / "base"
            _write_csv(tuned / "hierarchical_relation_type_metrics.csv", relation_rows("tuned"))
            _write_csv(base / "hierarchical_relation_type_metrics.csv", relation_rows("base"))
            rows = _load_relation_type_comparison(
                tuned_dir=tuned,
                tuned_model="tuned",
                base_dir=base,
                base_model="base",
            )
            self.assertEqual(len(rows), 12)
            f1 = next(row for row in rows if row["relation_type"] == "key-value" and row["field"] == "F1")
            self.assertEqual(f1["value_kind"], "metric")
            self.assertAlmostEqual(f1["delta"], 0.0)

            _write_csv(
                base / "hierarchical_relation_type_metrics.csv",
                relation_rows("base", key_value_gt="11"),
            )
            with self.assertRaisesRegex(ValueError, "relation GT mismatch"):
                _load_relation_type_comparison(
                    tuned_dir=tuned,
                    tuned_model="tuned",
                    base_dir=base,
                    base_model="base",
                )

    def test_analysis_includes_hierarchical_and_relation_tables(self) -> None:
        overall = [
            {
                "metric": "Rel-F1",
                "label": "Rel-F1",
                "n_paired": 2,
                "tuned_mean": 0.2,
                "base_mean": 0.1,
                "delta": 0.1,
                "ci95_low": -0.1,
                "ci95_high": 0.2,
                "wins": 1,
                "ties": 0,
                "losses": 1,
                "n_tuned_numeric": 2,
                "n_base_numeric": 2,
                "n_unpaired_numeric": 0,
            }
        ]
        aggregate = [
            {
                "report": "hierarchical_structure_metrics",
                "field": "Rel-F1-micro",
                "tuned": 0.2,
                "base": 0.1,
                "delta": 0.1,
                "value_kind": "metric",
            }
        ]
        relation = []
        for field, tuned, base, kind in (
            ("TP", 2, 1, "count"),
            ("pred", 4, 3, "count"),
            ("GT", 10, 10, "count"),
            ("precision", 0.5, 0.333333, "metric"),
            ("recall", 0.2, 0.1, "metric"),
            ("F1", 0.285714, 0.15, "metric"),
        ):
            relation.append(
                {
                    "relation_type": "key-value",
                    "field": field,
                    "tuned": tuned,
                    "base": base,
                    "delta": tuned - base,
                    "value_kind": kind,
                }
            )
        report = _build_analysis(
            index_path=Path("index.jsonl"),
            tuned_dir=Path("tuned"),
            tuned_model="tuned",
            base_dir=Path("base"),
            base_model="base",
            n_samples=2,
            n_templates=1,
            iterations=10,
            seed=42,
            overall_rows=overall,
            template_rows=[],
            notes=[],
            aggregate_rows=aggregate,
            relation_type_rows=relation,
        )
        self.assertIn("## Hierarchical aggregate diagnostics", report)
        self.assertIn("Rel-F1-micro", report)
        self.assertIn("## Relation-type diagnostics", report)
        self.assertIn("key-value", report)
        self.assertIn("0.135714", report)


if __name__ == "__main__":
    unittest.main()
