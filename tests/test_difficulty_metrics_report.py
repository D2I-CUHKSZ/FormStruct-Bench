from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from formtsr_exp.difficulty_metrics_report import (
    DifficultySample,
    SelectedRun,
    build_diagnostics,
    evaluate_run_by_difficulty,
    load_difficulty_samples,
    validate_against_official,
    validate_benchmark_partition,
)
from formtsr_exp.document_similarity_report import (
    build_schema_tree,
    extract_normalized_values,
)
from formtsr_exp.metrics import normalize_json


def _sample(sample_id: str, template: str, level: str, gt: dict[str, str]) -> DifficultySample:
    return DifficultySample(
        sample_id=sample_id,
        template_name=template,
        difficulty_level=level,
        normalized_gt=normalize_json(gt),
        schema=build_schema_tree(gt),
        values=extract_normalized_values(gt),
    )


class DifficultyMetricsReportTest(unittest.TestCase):
    def test_frozen_mapping_must_match_index_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels"
            labels.mkdir()
            index_path = root / "index.jsonl"
            difficulty_path = root / "difficulty.csv"
            index_rows = []
            for level in ("L1", "L2", "L3", "L4"):
                template = f"template-{level}"
                label_path = labels / f"{template}.json"
                label_path.write_text(json.dumps({"field": level}), encoding="utf-8")
                index_rows.append(
                    {
                        "sample_id": f"{template}__01",
                        "template_name": template,
                        "label_path": str(label_path),
                    }
                )
            index_path.write_text(
                "".join(json.dumps(row) + "\n" for row in index_rows),
                encoding="utf-8",
            )
            with difficulty_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["file", "normal_calibrated_level"])
                writer.writeheader()
                for level in ("L1", "L2", "L3", "L4"):
                    writer.writerow(
                        {
                            "file": f"template-{level}.json",
                            "normal_calibrated_level": level,
                        }
                    )

            samples, template_counts, sample_counts = load_difficulty_samples(
                index_path,
                difficulty_path,
            )
            self.assertEqual(len(samples), 4)
            self.assertEqual(template_counts, {level: 1 for level in ("L1", "L2", "L3", "L4")})
            self.assertEqual(sample_counts, {level: 1 for level in ("L1", "L2", "L3", "L4")})

            with difficulty_path.open("a", encoding="utf-8") as fh:
                fh.write("extra.json,L1\n")
            with self.assertRaisesRegex(ValueError, "must match the dataset index exactly"):
                load_difficulty_samples(index_path, difficulty_path)

    def test_benchmark_partition_is_fixed(self) -> None:
        validate_benchmark_partition(
            {"L1": 11, "L2": 24, "L3": 24, "L4": 11},
            {"L1": 1100, "L2": 2400, "L3": 2400, "L4": 1100},
        )
        with self.assertRaisesRegex(ValueError, "template counts changed"):
            validate_benchmark_partition(
                {"L1": 10, "L2": 25, "L3": 24, "L4": 11},
                {"L1": 1100, "L2": 2400, "L3": 2400, "L4": 1100},
            )

    def test_missing_pages_are_zero_and_lig_uses_applicable_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_root = root / "pred"
            semantic_dir = root / "semantic"
            structure_dir = root / "structure"
            model = "run-a"
            (pred_root / model).mkdir(parents=True)
            semantic_dir.mkdir()
            structure_dir.mkdir()

            samples = [
                _sample("l1-valid", "t1", "L1", {"field": "x"}),
                _sample("l1-missing", "t1", "L1", {"field": "y"}),
                _sample("l2-valid", "t2", "L2", {"field": "L2"}),
                _sample("l3-valid", "t3", "L3", {"field": "L3"}),
                _sample("l4-valid", "t4", "L4", {"field": "L4"}),
            ]
            for sample in samples:
                if sample.sample_id == "l1-missing":
                    continue
                (pred_root / model / f"{sample.sample_id}.json").write_text(
                    json.dumps({"answer": {"field": sample.values[0]}}),
                    encoding="utf-8",
                )

            semantic_rows = []
            structure_rows = []
            for sample in samples:
                valid = sample.sample_id != "l1-missing"
                semantic_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "valid_json": valid,
                        "TSR-path": 0.5 if sample.sample_id == "l1-valid" else (1.0 if valid else 0.0),
                    }
                )
                structure_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "valid_json": valid,
                        "R-F1": 0.4 if sample.sample_id == "l1-valid" else (1.0 if valid else 0.0),
                        "R-F1@0.75": 0.2 if sample.sample_id == "l1-valid" else (1.0 if valid else 0.0),
                        "LIG-F1": 0.0 if sample.sample_id == "l1-missing" else "NA",
                    }
                )
            (semantic_dir / f"{model}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in semantic_rows),
                encoding="utf-8",
            )
            (structure_dir / f"{model}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in structure_rows),
                encoding="utf-8",
            )

            run = SelectedRun(model, "Model A", {})
            rows = evaluate_run_by_difficulty(
                run,
                samples,
                pred_root,
                semantic_dir,
                structure_dir,
            )
            l1 = rows[0]
            self.assertEqual(l1["n_total"], 2)
            self.assertEqual(l1["n_valid_json"], 1)
            self.assertEqual(l1["n_exact_match"], 1)
            self.assertEqual(l1["Page-EM"], 0.5)
            self.assertEqual(l1["Schema-nTED"], 0.5)
            self.assertEqual(l1["Value-nED"], 0.5)
            self.assertEqual(l1["TSR-path"], 0.25)
            self.assertEqual(l1["R-F1@0.5"], 0.2)
            self.assertEqual(l1["R-F1@0.75"], 0.1)
            self.assertEqual(l1["n_lig_applicable"], 1)
            self.assertEqual(l1["LIG-F1"], 0.0)

    def test_rollup_reconstructs_official_page_macro_metrics(self) -> None:
        rows = []
        totals = {"L1": 1100, "L2": 2400, "L3": 2400, "L4": 1100}
        lig_totals = {"L1": 200, "L2": 600, "L3": 800, "L4": 800}
        for level in ("L1", "L2", "L3", "L4"):
            rows.append(
                {
                    "model": "run-a",
                    "model_id": "Model A",
                    "difficulty_level": level,
                    "n_total": totals[level],
                    "n_valid_json": totals[level],
                    "n_exact_match": 0,
                    "Page-EM": 0.0,
                    "Schema-nTED": 0.5,
                    "Value-nED": 0.6,
                    "TSR-path": 0.2,
                    "R-F1@0.5": 0.5,
                    "R-F1@0.75": 0.25,
                    "n_lig_applicable": lig_totals[level],
                    "LIG-F1": 0.1,
                }
            )
        official = {
            "n_total": "7000",
            "n_valid_json": "7000",
            "n_exact_match": "0",
            "n_lig_applicable": "2400",
            "Page-EM": "0.000000",
            "Schema-nTED": "0.500000",
            "Value-nED": "0.600000",
            "TSR-path": "0.2000",
            "R-F1": "0.500000",
            "R-F1@0.75": "0.250000",
            "LIG-F1": "0.100000",
        }
        validation = validate_against_official(SelectedRun("run-a", "Model A", official), rows)
        self.assertEqual(validation["n_total"], 7000)

        diagnostics = build_diagnostics(rows)
        r_f1 = next(row for row in diagnostics if row["metric"] == "R-F1@0.5")
        self.assertEqual(r_f1["L1_easy"], 0.5)
        self.assertEqual(r_f1["L1_to_L4_drop"], 0.0)


if __name__ == "__main__":
    unittest.main()
