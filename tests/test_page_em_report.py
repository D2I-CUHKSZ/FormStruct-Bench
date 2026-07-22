from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from formtsr_exp.metrics import normalize_json
from formtsr_exp.page_em_report import (
    RunSpec,
    Sample,
    classify_comparison_status,
    classify_run_type,
    evaluate_run,
    page_exact_match,
)


class PageExactMatchTest(unittest.TestCase):
    def test_normalization_and_answer_unwrap(self) -> None:
        gt = normalize_json({"section": {"field": "hello world"}})

        self.assertEqual(
            page_exact_match({"answer": {"section": {"field": "  hello   world "}}}, gt),
            1.0,
        )
        self.assertEqual(page_exact_match({"answer": {"section": {"field": "Hello world"}}}, gt), 0.0)
        self.assertEqual(
            page_exact_match({"answer": {"section": {"field": "hello world", "extra": "x"}}}, gt),
            0.0,
        )

    def test_list_order_is_significant(self) -> None:
        gt = normalize_json({"rows": ["a", "b"]})
        self.assertEqual(page_exact_match({"answer": {"rows": ["a", "b"]}}, gt), 1.0)
        self.assertEqual(page_exact_match({"answer": {"rows": ["b", "a"]}}, gt), 0.0)

    def test_run_aggregation_penalizes_invalid_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "pred" / "model-a"
            model_dir.mkdir(parents=True)
            (model_dir / "s1.json").write_text(
                json.dumps({"answer": {"field": "value"}}),
                encoding="utf-8",
            )
            (model_dir / "s2.json").write_text("not json", encoding="utf-8")
            (model_dir / "outside-index.json").write_text("{}", encoding="utf-8")
            samples = [
                Sample("s1", "labels/s1.json", normalize_json({"field": "value"})),
                Sample("s2", "labels/s2.json", normalize_json({"field": "other"})),
            ]

            row, exact_rows = evaluate_run(
                RunSpec("model-a", "Model A", "vlm", 2),
                samples,
                root / "pred",
            )

        self.assertEqual(row["n_valid_json"], 1)
        self.assertEqual(row["n_invalid_json"], 1)
        self.assertEqual(row["n_extra_prediction_files"], 1)
        self.assertEqual(row["n_exact_match"], 1)
        self.assertEqual(row["Page-EM"], 0.5)
        self.assertEqual(row["Page-EM-valid"], 1.0)
        self.assertEqual([item["sample_id"] for item in exact_rows], ["s1"])

    def test_missing_prediction_stays_in_page_em_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "pred" / "model-a"
            model_dir.mkdir(parents=True)
            (model_dir / "s1.json").write_text('{"answer":{"field":"value"}}', encoding="utf-8")
            samples = [
                Sample("s1", "labels/s1.json", normalize_json({"field": "value"})),
                Sample("s2", "labels/s2.json", normalize_json({"field": "other"})),
            ]

            row, _ = evaluate_run(RunSpec("model-a", "Model A", "vlm", 2), samples, root / "pred")

        self.assertEqual(row["n_missing_prediction"], 1)
        self.assertEqual(row["Page-EM"], 0.5)
        self.assertEqual(row["Page-EM-valid"], 1.0)

    def test_missing_model_directory_is_failed(self) -> None:
        samples = [Sample("s1", "labels/s1.json", normalize_json({"field": "value"}))]
        with tempfile.TemporaryDirectory() as tmp:
            row, exact_rows = evaluate_run(
                RunSpec("failed-model", "Failed", "vlm", 1),
                samples,
                Path(tmp) / "pred",
            )

        self.assertEqual(row["comparison_status"], "failed")
        self.assertEqual(row["n_missing_prediction"], 1)
        self.assertEqual(row["Page-EM"], 0.0)
        self.assertEqual(exact_rows, [])

    def test_run_classification(self) -> None:
        self.assertEqual(classify_run_type("model_aligned_metadata"), "aligned")
        self.assertEqual(classify_run_type("model_aligned_metadata_smoke"), "smoke")
        self.assertEqual(classify_run_type("model"), "raw")
        self.assertEqual(
            classify_comparison_status(
                run_type="raw",
                has_prediction_dir=True,
                full_scope=True,
                n_valid_json=7000,
                n_indexed=7000,
            ),
            "comparable_raw",
        )

    def test_prediction_directory_requires_matching_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pred" / "model-a").mkdir(parents=True)
            samples = [Sample("s1", "labels/s1.json", normalize_json({"field": "value"}))]
            with self.assertRaisesRegex(ValueError, "n_total=2"):
                evaluate_run(RunSpec("model-a", "Model A", "vlm", 2), samples, root / "pred")


if __name__ == "__main__":
    unittest.main()
