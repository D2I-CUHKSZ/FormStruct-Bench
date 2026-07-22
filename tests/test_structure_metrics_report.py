from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from formtsr_exp.page_em_report import RunSpec
from formtsr_exp.structure_metrics_report import (
    CanonicalRegion,
    StructureSample,
    evaluate_run,
    normalize_bbox,
    region_match_counts,
    resolve_bbox_space,
)


class BBoxNormalizationTest(unittest.TestCase):
    def test_supported_coordinate_spaces_share_one_result(self) -> None:
        expected = (0.1, 0.2, 0.3, 0.4)
        self.assertEqual(normalize_bbox([10, 40, 30, 80], "pixel", 100, 200), (expected, "ok"))
        self.assertEqual(normalize_bbox([100, 200, 300, 400], "normalized_1000", 1, 1), (expected, "ok"))
        self.assertEqual(normalize_bbox(expected, "normalized_1", 1, 1), (expected, "ok"))

    def test_clips_but_does_not_swap_or_repair_bad_boxes(self) -> None:
        self.assertEqual(
            normalize_bbox([-100, 100, 1200, 900], "normalized_1000", 1, 1),
            ((0.0, 0.1, 1.0, 0.9), "clipped"),
        )
        self.assertEqual(
            normalize_bbox([800, 100, 200, 900], "normalized_1000", 1, 1),
            (None, "reversed_or_degenerate"),
        )
        self.assertEqual(normalize_bbox([0, 0, 0, 1], "normalized_1", 1, 1), (None, "reversed_or_degenerate"))

    def test_manifest_inheritance_is_explicit(self) -> None:
        manifest = {
            "runs": {
                "raw": {"source_space": "normalized_1000"},
                "aligned": {"inherits": "raw"},
            }
        }
        self.assertEqual(resolve_bbox_space(manifest, "aligned"), ("normalized_1000", "raw"))
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_bbox_space(manifest, "missing")


class CanonicalRegionMatchingTest(unittest.TestCase):
    def test_generic_prediction_text_can_match_gt_value(self) -> None:
        pred = [CanonicalRegion("text", (0.1, 0.1, 0.4, 0.4))]
        gt = (CanonicalRegion("value", (0.1, 0.1, 0.4, 0.4)),)
        self.assertEqual(region_match_counts(pred, 1, gt, 0.75), (1, 1, 1))

    def test_incompatible_type_cannot_match_identical_box(self) -> None:
        pred = [CanonicalRegion("widget", (0.1, 0.1, 0.4, 0.4))]
        gt = (CanonicalRegion("label", (0.1, 0.1, 0.4, 0.4)),)
        self.assertEqual(region_match_counts(pred, 1, gt, 0.5), (0, 1, 1))


class CorrectedStructureAggregationTest(unittest.TestCase):
    def test_full_scope_penalizes_missing_page_and_invalid_box(self) -> None:
        samples = [
            StructureSample(
                "s1",
                "template",
                100,
                100,
                (CanonicalRegion("label", (0.1, 0.1, 0.3, 0.3)),),
                ((0.5, 0.5, 0.9, 0.9),),
            ),
            StructureSample(
                "s2",
                "template",
                100,
                100,
                (CanonicalRegion("label", (0.1, 0.1, 0.3, 0.3)),),
                (),
            ),
        ]
        manifest = {"runs": {"model-a": {"source_space": "normalized_1000"}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "pred" / "model-a"
            model_dir.mkdir(parents=True)
            (model_dir / "s1.json").write_text(
                json.dumps(
                    {
                        "regions": [
                            {"type": "field", "bbox": [100, 100, 300, 300]},
                            {"type": "field", "bbox": [800, 100, 200, 300]},
                        ],
                        "line_item_groups": [{"bbox": [500, 500, 900, 900]}],
                    }
                ),
                encoding="utf-8",
            )
            row = evaluate_run(
                RunSpec("model-a", "Model A", "vlm", 2),
                samples,
                root / "pred",
                manifest,
            )

        self.assertEqual(row["n_valid_json"], 1)
        self.assertEqual(row["n_missing_prediction"], 1)
        self.assertAlmostEqual(row["R-F1"], 1.0 / 3.0)
        self.assertEqual(row["R-Precision@0.5"], 0.5)
        self.assertEqual(row["R-Recall@0.5"], 0.5)
        self.assertEqual(row["R-micro-F1@0.5"], 0.5)
        self.assertEqual(row["region_bbox_declared"], 2)
        self.assertEqual(row["region_bbox_valid"], 1)
        self.assertEqual(row["region_bbox_dropped"], 1)
        self.assertEqual(row["n_lig_applicable"], 1)
        self.assertEqual(row["LIG-F1"], 1.0)


if __name__ == "__main__":
    unittest.main()
