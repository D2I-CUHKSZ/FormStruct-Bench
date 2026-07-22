from __future__ import annotations

import unittest

from formtsr_exp.constraint_slices import ConstraintFlag
from formtsr_exp.robustness_component_report import (
    COMPONENTS,
    annotate_pairs,
    build_contrasts,
    build_excess_drop_severity,
)
from formtsr_exp.robustness_metrics_report import METRICS


def _pair(
    *,
    model: str,
    template: str,
    clean_value: float,
    degraded_value: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "model_id": model,
        "clean_model": model,
        "pairing_status": "same_run_id",
        "template_name": template,
        "degradation_variant": "blur_noise",
        "degradation_level": "high",
        "structure_metrics_valid": True,
        "clean_valid_json": True,
        "degraded_valid_json": True,
        "clean_n_exact_match": 0,
        "degraded_n_exact_match": 0,
    }
    for metric in METRICS:
        row[f"clean_{metric}"] = clean_value
        row[f"degraded_{metric}"] = degraded_value
        row[f"{metric}_drop"] = clean_value - degraded_value
    return row


class RobustnessComponentReportTests(unittest.TestCase):
    def test_pair_labels_come_from_template_lookup(self) -> None:
        lookup = {
            "template": {
                component: ConstraintFlag(
                    present=component in {"region_local_grids", "mixed_layout"},
                    signal=1.0,
                    rule="metadata rule",
                )
                for component in COMPONENTS
            }
        }
        annotated = annotate_pairs([{"template_name": "template", "prediction_component": "wrong"}], lookup)
        self.assertEqual(len(annotated), len(COMPONENTS))
        present = {row["component"] for row in annotated if row["component_present"]}
        self.assertEqual(present, {"region_local_grids", "mixed_layout"})

    def test_contrast_is_present_drop_minus_absent_drop(self) -> None:
        base = {
            "model": "run",
            "model_id": "Model",
            "clean_model": "run",
            "pairing_status": "same_run_id",
            "degradation_variant": "blur_noise",
            "degradation_level": "high",
            "component": "region_local_grids",
            "component_name": "Region-local grids",
            "n_total": 2,
            "n_templates": 2,
        }
        present = {**base, "component_present": True}
        absent = {**base, "component_present": False}
        for metric in METRICS:
            present[f"clean_{metric}"] = 0.6
            present[f"degraded_{metric}"] = 0.4
            present[f"{metric}_drop"] = 0.2
            absent[f"clean_{metric}"] = 0.6
            absent[f"degraded_{metric}"] = 0.5
            absent[f"{metric}_drop"] = 0.1
        result = build_contrasts([present, absent])[0]
        self.assertAlmostEqual(result["Value-nED_excess_drop"], 0.1)
        self.assertEqual(result["status"], "ok")

    def test_severity_excess_uses_fixed_present_and_absent_templates(self) -> None:
        rows = []
        for model in ("a", "b"):
            with_component = _pair(
                model=model,
                template="with",
                clean_value=0.6,
                degraded_value=0.4,
            )
            with_component.update(
                {
                    "component": "region_local_grids",
                    "component_present": True,
                }
            )
            without_component = _pair(
                model=model,
                template="without",
                clean_value=0.6,
                degraded_value=0.5,
            )
            without_component.update(
                {
                    "component": "region_local_grids",
                    "component_present": False,
                }
            )
            rows.extend([with_component, without_component])
        result = build_excess_drop_severity(rows, expected_models=2)[0]
        self.assertEqual(result["n_with_templates"], 1)
        self.assertEqual(result["n_without_templates"], 1)
        self.assertAlmostEqual(result["mean_Value-nED_excess_drop"], 0.1)


if __name__ == "__main__":
    unittest.main()
