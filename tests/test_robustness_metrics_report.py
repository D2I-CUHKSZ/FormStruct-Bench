from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from formtsr_exp.document_similarity_report import build_schema_tree, extract_normalized_values
from formtsr_exp.metrics import NA, normalize_json
from formtsr_exp.robustness_metrics_report import (
    BaseSample,
    METRICS,
    _evaluate_prediction,
    aggregate_pairs,
    summarize_variant_macro,
)


def _pair(
    sample: str,
    *,
    variant: str,
    structure_valid: bool,
    clean_value: float,
    degraded_value: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "run-a",
        "model_id": "Model A",
        "clean_model": "run-a",
        "pairing_status": "same_run_id",
        "degraded_sample_id": sample,
        "clean_sample_id": sample.split("__deg__", 1)[0],
        "template_name": "template-a",
        "difficulty_level": "L1",
        "difficulty_name": "easy",
        "degradation_variant": variant,
        "degradation_level": "low",
        "structure_metrics_valid": structure_valid,
        "clean_valid_json": True,
        "degraded_valid_json": True,
        "clean_n_exact_match": 0,
        "degraded_n_exact_match": 0,
    }
    for metric in METRICS:
        if metric in {"R-F1@0.5", "R-F1@0.75", "LIG-F1"} and not structure_valid:
            row[f"clean_{metric}"] = NA
            row[f"degraded_{metric}"] = NA
        else:
            row[f"clean_{metric}"] = clean_value
            row[f"degraded_{metric}"] = degraded_value
    return row


class RobustnessMetricsReportTest(unittest.TestCase):
    def test_prediction_metrics_are_recomputed_against_shared_gt(self) -> None:
        gt = {"section": {"a": "x", "missing_in_clean_label": "50%"}}
        sample = BaseSample(
            sample_id="sample-a",
            template_name="template-a",
            difficulty_level="L1",
            gt=gt,
            normalized_gt=normalize_json(gt),
            schema=build_schema_tree(gt),
            values=extract_normalized_values(gt),
        )
        with tempfile.TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "sample-a.json"
            pred_path.write_text(json.dumps({"answer": gt}), encoding="utf-8")
            result = _evaluate_prediction(
                pred_path,
                sample,
                {"valid_json": True, "TSR-path": 0.0},
                {
                    "valid_json": True,
                    "R-F1": 0.25,
                    "R-F1@0.75": 0.125,
                    "LIG-F1": NA,
                },
            )
        self.assertEqual(result["Page-EM"], 1.0)
        self.assertEqual(result["Schema-nTED"], 1.0)
        self.assertEqual(result["Value-nED"], 1.0)
        self.assertEqual(result["TSR-path"], 1.0)

    def test_geometry_changing_variant_marks_spatial_metrics_na(self) -> None:
        rows = [
            _pair(
                "a__deg__perspective_skew__low",
                variant="perspective_skew",
                structure_valid=False,
                clean_value=0.8,
                degraded_value=0.5,
            ),
            _pair(
                "b__deg__perspective_skew__low",
                variant="perspective_skew",
                structure_valid=False,
                clean_value=0.6,
                degraded_value=0.3,
            ),
        ]
        summary = aggregate_pairs(rows, ("model", "degradation_variant", "degradation_level"))[0]
        self.assertEqual(summary["n_structure_applicable"], 0)
        self.assertEqual(summary["n_lig_applicable"], 0)
        self.assertEqual(summary["structure_metric_status"], "NA_geometry_changed")
        self.assertEqual(summary["R-F1@0.5_drop"], NA)
        self.assertAlmostEqual(summary["Value-nED_drop"], 0.3)

    def test_mixed_variant_summary_uses_only_geometry_preserving_spatial_rows(self) -> None:
        rows = [
            _pair(
                "a__deg__blur_noise__low",
                variant="blur_noise",
                structure_valid=True,
                clean_value=1.0,
                degraded_value=0.5,
            ),
            _pair(
                "a__deg__dilate__low",
                variant="dilate",
                structure_valid=False,
                clean_value=1.0,
                degraded_value=0.0,
            ),
        ]
        summary = aggregate_pairs(rows, ("model", "degradation_level"))[0]
        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_structure_applicable"], 1)
        self.assertEqual(summary["n_lig_applicable"], 1)
        self.assertEqual(summary["structure_metric_status"], "partial_geometry_preserving_only")
        self.assertEqual(summary["R-F1@0.5_drop"], 0.5)
        self.assertEqual(summary["Value-nED_drop"], 0.75)

    def test_variant_macro_preserves_na_spatial_metrics(self) -> None:
        rows = []
        for index in range(7):
            row: dict[str, object] = {
                "model": f"run-{index}",
                "model_id": f"Model {index}",
                "degradation_variant": "perspective_skew",
                "degradation_level": "high",
                "clean_coverage": 1.0,
                "degraded_coverage": 0.9,
            }
            for metric in METRICS:
                if metric in {"R-F1@0.5", "R-F1@0.75", "LIG-F1"}:
                    row[f"clean_{metric}"] = NA
                    row[f"degraded_{metric}"] = NA
                else:
                    row[f"clean_{metric}"] = 0.8
                    row[f"degraded_{metric}"] = 0.6
            rows.append(row)
        summary = summarize_variant_macro(rows, expected_models=7)[0]
        self.assertEqual(summary["mean_R-F1@0.5_drop"], NA)
        self.assertAlmostEqual(summary["mean_Value-nED_drop"], 0.2)
        self.assertAlmostEqual(summary["coverage_drop"], 0.1)


if __name__ == "__main__":
    unittest.main()
