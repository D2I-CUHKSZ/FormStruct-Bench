from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from formtsr_exp.metrics import NA
from formtsr_exp.structure_ablation_metrics_report import (
    COMPARISONS,
    VARIANTS,
    load_selected_runs,
    score_variant,
    summarize_macro,
    summarize_targeted,
)


class StructureAblationMetricsReportTests(unittest.TestCase):
    def test_variant_score_uses_only_applicable_components(self) -> None:
        row = {
            "Schema-nTED": 0.6,
            "Value-nED": 0.8,
            "TSR-path": 0.4,
            "R-F1@0.5": 0.2,
            "LIG-F1": NA,
            "WAcc": 0.0,
            "Rel-F1": NA,
        }
        self.assertAlmostEqual(score_variant(row, "full_structural"), 0.4)

    def test_target_scope_includes_invalid_zero_rows(self) -> None:
        rows = []
        for valid, semantic, region in ((True, 0.6, 0.4), (False, 0.0, 0.0)):
            row = {
                "valid_json": valid,
                "region_applicable": True,
                "grid_applicable": True,
                "lig_applicable": True,
                "widget_applicable": True,
                "relation_applicable": True,
                "structural_applicable": True,
            }
            for variant in VARIANTS:
                row[f"score:{variant}"] = semantic
            row["score:semantic_region"] = region
            rows.append(row)
        run = {"model": "run", "model_id": "Model", "n_valid_json": "1"}
        targeted = summarize_targeted(rows, run)
        region_row = next(
            row for row in targeted if row["comparison"] == "region_effect_on_region_samples"
        )
        self.assertEqual(region_row["n_scope"], 2)
        self.assertAlmostEqual(region_row["score_with"], 0.2)
        self.assertAlmostEqual(region_row["score_without"], 0.3)
        self.assertAlmostEqual(region_row["delta"], -0.1)

    def test_macro_requires_common_gt_scope_count(self) -> None:
        targeted = []
        for model, delta in (("a", -0.2), ("b", -0.1)):
            for spec in COMPARISONS:
                targeted.append(
                    {
                        "model": model,
                        "comparison": spec["comparison"],
                        "scope": spec["scope"],
                        "n_scope": 10,
                        "score_with": 0.3 + delta,
                        "score_without": 0.3,
                        "delta": delta,
                        "relative_delta_pct": delta / 0.3 * 100,
                    }
                )
        macro = summarize_macro(targeted)
        self.assertEqual(len(macro), len(COMPARISONS))
        self.assertTrue(all(row["n_models"] == 2 for row in macro))
        self.assertTrue(all(row["n_scope_per_model"] == 10 for row in macro))
        self.assertTrue(all(abs(row["mean_delta"] + 0.15) < 1e-12 for row in macro))

    def test_selection_rejects_non_raw_and_non_full_runs(self) -> None:
        columns = ["model", "model_id", "n_total", "n_attempted", "n_valid_json"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "good_vllm_vlm",
                        "model_id": "Good",
                        "n_total": 10,
                        "n_attempted": 10,
                        "n_valid_json": 9,
                    }
                )
            rows = load_selected_runs(path, 10)
            self.assertEqual([row["model"] for row in rows], ["good_vllm_vlm"])

            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writerow(
                    {
                        "model": "bad_aligned_metadata",
                        "model_id": "Bad",
                        "n_total": 10,
                        "n_attempted": 10,
                        "n_valid_json": 10,
                    }
                )
            with self.assertRaisesRegex(ValueError, "non-raw"):
                load_selected_runs(path, 10)


if __name__ == "__main__":
    unittest.main()
