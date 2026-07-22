from __future__ import annotations

import unittest
from pathlib import Path

from formtsr_exp.controlled_diagnostic_validation import (
    METRICS,
    PairedPageResult,
    _template_drop_map,
    build_template_plan,
    clean_prediction,
    corrupt_prediction,
    score_prediction,
)
from formtsr_exp.hierarchical_metrics_report import load_samples


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs/dataset_splits/template_stratified_seed42/test_index.jsonl"
METADATA = ROOT / "new-dataset-json"
LAYOUT = ROOT / "newdataset-layout"


class ControlledDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (INDEX, METADATA, LAYOUT)
        if any(not path.exists() for path in required):
            raise unittest.SkipTest(
                "requires the private test index and full annotation fixtures"
            )
        samples = load_samples(INDEX, METADATA, LAYOUT)
        cls.samples = {sample.template_name: sample for sample in samples}

    def _corrupt(self, template: str, error: str):
        sample = self.samples[template]
        clean = clean_prediction(sample)
        plan = build_template_plan(sample.template)
        corrupted, eligible, injected = corrupt_prediction(
            clean,
            sample,
            plan,
            error,
            1.0 - 1e-12,
            0,
        )
        self.assertGreater(eligible, 0)
        self.assertEqual(injected, eligible)
        return sample, clean, corrupted

    def test_gold_identity_is_exact_for_all_metric_families(self) -> None:
        representatives = (
            self.samples["Arabic-2"],
            self.samples["de_2"],
            self.samples["ja_6"],
            self.samples["Arabic-7"],
        )
        observed = set()
        for sample in representatives:
            scores = score_prediction(clean_prediction(sample), sample)
            for metric, score in scores.items():
                if score is not None:
                    observed.add(metric)
                    self.assertEqual(score, 1.0, (sample.sample_id, metric))
        self.assertEqual(observed, set(METRICS))

    def test_unrecoverable_relation_gt_is_excluded_before_scoring(self) -> None:
        sample = self.samples["en_7"]
        self.assertEqual(
            sample.template.audit["unrecoverable_typed_relations_excluded"], 11
        )
        self.assertEqual(
            score_prediction(clean_prediction(sample), sample)["Rel-F1"], 1.0
        )

    def test_value_corruption_is_path_and_length_preserving(self) -> None:
        sample, clean, corrupted = self._corrupt("Arabic-2", "value")
        clean_scores = score_prediction(clean, sample)
        scores = score_prediction(corrupted, sample)
        self.assertEqual(scores["Schema-nTED"], 1.0)
        self.assertLess(scores["Value-nED"], clean_scores["Value-nED"])
        self.assertLess(scores["TSR-path"], clean_scores["TSR-path"])
        for metric in METRICS[3:]:
            if scores[metric] is not None:
                self.assertEqual(scores[metric], clean_scores[metric])

    def test_hierarchy_corruption_preserves_values(self) -> None:
        sample, clean, corrupted = self._corrupt("Arabic-2", "hierarchy")
        clean_scores = score_prediction(clean, sample)
        scores = score_prediction(corrupted, sample)
        self.assertLess(scores["Schema-nTED"], 1.0)
        self.assertEqual(scores["Value-nED"], 1.0)
        self.assertLess(scores["TSR-path"], 1.0)
        self.assertEqual(scores["R-F1@0.5"], clean_scores["R-F1@0.5"])

    def test_region_corruption_has_declared_parent_dependencies(self) -> None:
        sample, clean, corrupted = self._corrupt("ja_6", "region")
        scores = score_prediction(corrupted, sample)
        self.assertLess(scores["R-F1@0.5"], 1.0)
        self.assertLess(scores["LG-GriTS-Top"], 1.0)
        self.assertLess(scores["Rel-F1"], 1.0)
        self.assertEqual(scores["Value-nED"], 1.0)
        self.assertEqual(len(corrupted.regions), len(clean.regions))

    def test_line_item_grid_widget_and_relation_targets_respond(self) -> None:
        cases = (
            ("de_2", "line_item", "LIG-F1"),
            ("ja_6", "local_grid", "LG-GriTS-Top"),
            ("Arabic-7", "widget", "WG-F1"),
            ("Arabic-2", "relation", "Rel-F1"),
        )
        for template, error, metric in cases:
            with self.subTest(error=error):
                sample, _clean, corrupted = self._corrupt(template, error)
                self.assertLess(score_prediction(corrupted, sample)[metric], 1.0)

    def test_template_macro_does_not_pool_pages(self) -> None:
        clean = {metric: 1.0 for metric in METRICS}

        def row(sample: str, template: str, corrupted_value: float) -> PairedPageResult:
            corrupted = dict(clean)
            corrupted["Value-nED"] = corrupted_value
            return PairedPageResult(
                "value",
                0.25,
                0,
                sample,
                template,
                1,
                1,
                clean,
                corrupted,
            )

        rows = [
            row("a1", "a", 0.0),
            row("b1", "b", 1.0),
            row("b2", "b", 1.0),
            row("b3", "b", 1.0),
        ]
        template_values = _template_drop_map(rows, "Value-nED")
        self.assertEqual(template_values, {"a": 100.0, "b": 0.0})
        self.assertEqual(sum(template_values.values()) / len(template_values), 50.0)


if __name__ == "__main__":
    unittest.main()
