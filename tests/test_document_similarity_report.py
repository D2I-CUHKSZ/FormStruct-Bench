from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from formtsr_exp.document_similarity_report import (
    DocumentSample,
    build_schema_tree,
    evaluate_run,
    extract_normalized_values,
    schema_nted,
    value_ned,
)
from formtsr_exp.page_em_report import RunSpec


class SchemaNtedTest(unittest.TestCase):
    def test_dict_order_and_values_do_not_change_schema(self) -> None:
        left = build_schema_tree({"a": "one", "b": {"c": "two"}})
        right = build_schema_tree({"b": {"c": "different"}, "a": "changed"})
        self.assertEqual(schema_nted(left, right), 1.0)

    def test_key_rename_and_extra_key_have_bounded_scores(self) -> None:
        original = build_schema_tree({"a": "one"})
        renamed = build_schema_tree({"b": "one"})
        extra = build_schema_tree({"a": "one", "b": "two"})
        self.assertEqual(schema_nted(original, renamed), 0.5)
        self.assertEqual(schema_nted(extra, original), 0.8)

    def test_list_item_order_is_preserved(self) -> None:
        left = build_schema_tree({"rows": [{"a": "x"}, {"b": "y"}]})
        right = build_schema_tree({"rows": [{"b": "y"}, {"a": "x"}]})
        self.assertLess(schema_nted(left, right), 1.0)


class ValueNedTest(unittest.TestCase):
    def test_path_and_value_order_are_ignored(self) -> None:
        left = extract_normalized_values({"answer": {"a": "Alpha", "b": "Beta"}})
        right = extract_normalized_values({"x": "beta", "y": "alpha"})
        self.assertEqual(value_ned(left, right), 1.0)

    def test_soft_edit_similarity(self) -> None:
        self.assertAlmostEqual(value_ned(("hello",), ("hallo",)), 0.8, places=6)

    def test_extra_and_missing_values_are_penalized(self) -> None:
        self.assertEqual(value_ned(("same", "extra"), ("same",)), 0.5)
        self.assertEqual(value_ned(("same",), ("same", "missing")), 0.5)
        self.assertEqual(value_ned((), ()), 1.0)
        self.assertEqual(value_ned((), ("missing",)), 0.0)

    def test_missing_page_stays_in_primary_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "pred" / "model-a"
            model_dir.mkdir(parents=True)
            (model_dir / "s1.json").write_text(
                '{"answer":{"field":"value"}}',
                encoding="utf-8",
            )
            samples = [
                DocumentSample(
                    "s1",
                    build_schema_tree({"field": "value"}),
                    extract_normalized_values({"field": "value"}),
                ),
                DocumentSample(
                    "s2",
                    build_schema_tree({"field": "other"}),
                    extract_normalized_values({"field": "other"}),
                ),
            ]
            row = evaluate_run(
                RunSpec("model-a", "Model A", "vlm", 2),
                samples,
                root / "pred",
            )

        self.assertEqual(row["Schema-nTED"], 0.5)
        self.assertEqual(row["Schema-nTED-valid"], 1.0)
        self.assertEqual(row["Value-nED"], 0.5)
        self.assertEqual(row["Value-nED-valid"], 1.0)


if __name__ == "__main__":
    unittest.main()
