from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from formtsr_exp.final_metrics_report import merge_results, select_best_full_runs


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FinalMetricsMergeTest(unittest.TestCase):
    def test_merges_latest_metric_sources_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {"model": "run-a", "model_id": "Model A", "group": "vlm", "n_total": "2", "n_valid_json": "1"}
            _write_csv(
                root / "main.csv",
                [{**common, "TSR-path": "0.2"}],
            )
            _write_csv(
                root / "page.csv",
                [
                    {
                        **common,
                        "run_type": "raw",
                        "comparison_status": "comparable_raw",
                        "coverage": "0.5",
                        "n_exact_match": "1",
                        "Page-EM": "0.5",
                    }
                ],
            )
            _write_csv(
                root / "document.csv",
                [{**common, "Schema-nTED": "0.6", "Value-nED": "0.7"}],
            )
            _write_csv(
                root / "structure.csv",
                [
                    {
                        **common,
                        "R-Precision@0.5": "0.8",
                        "R-Recall@0.5": "0.4",
                        "R-F1": "0.5",
                        "R-F1@0.75": "0.3",
                        "n_lig_applicable": "1",
                        "LIG-F1": "0.2",
                    }
                ],
            )
            _write_csv(
                root / "hierarchical.csv",
                [
                    {
                        **common,
                        "n_lg_gt_applicable": "1",
                        "LG-GriTS-Top": "0.7",
                        "n_wg_gt_applicable": "2",
                        "WG-F1": "0.8",
                        "n_rel_gt_applicable": "2",
                        "Rel-F1": "0.9",
                        "Rel-F1-micro": "0.85",
                        "Rel-F1-matched-endpoints": "0.95",
                    }
                ],
            )
            rows = merge_results(
                root / "main.csv",
                root / "page.csv",
                root / "document.csv",
                root / "structure.csv",
                root / "hierarchical.csv",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Page-EM"], "0.5")
        self.assertEqual(rows[0]["Schema-nTED"], "0.6")
        self.assertEqual(rows[0]["TSR-path"], "0.2")
        self.assertEqual(rows[0]["R-F1"], "0.5")
        self.assertEqual(rows[0]["LG-GriTS-Top"], "0.7")
        self.assertEqual(rows[0]["WG-F1"], "0.8")
        self.assertEqual(rows[0]["Rel-F1"], "0.9")

    def test_rejects_cross_report_coverage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(root / "main.csv", [{"model": "run-a", "n_total": "2", "n_valid_json": "1"}])
            _write_csv(
                root / "page.csv",
                [{"model": "run-a", "n_total": "2", "n_valid_json": "1"}],
            )
            _write_csv(
                root / "document.csv",
                [{"model": "run-a", "n_total": "2", "n_valid_json": "0"}],
            )
            _write_csv(
                root / "structure.csv",
                [{"model": "run-a", "n_total": "2", "n_valid_json": "1"}],
            )
            _write_csv(
                root / "hierarchical.csv",
                [{"model": "run-a", "n_total": "2", "n_valid_json": "1"}],
            )
            with self.assertRaisesRegex(ValueError, "n_valid_json mismatch"):
                merge_results(
                    root / "main.csv",
                    root / "page.csv",
                    root / "document.csv",
                    root / "structure.csv",
                    root / "hierarchical.csv",
                )

    def test_selects_best_fully_attempted_raw_run(self) -> None:
        def row(model: str, model_id: str, valid: int, schema: float) -> dict[str, str]:
            return {
                "model": model,
                "model_id": model_id,
                "run_type": "raw",
                "n_total": "2",
                "n_valid_json": str(valid),
                "Schema-nTED": str(schema),
                "Value-nED": "0.5",
                "TSR-path": "0.2",
                "R-F1": "0.1",
                "LIG-F1": "0.1",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.jsonl").write_text(
                '{"sample_id":"s1"}\n{"sample_id":"s2"}\n',
                encoding="utf-8",
            )
            for model, sample_ids in {
                "model-a-old": ("s1", "s2"),
                "model-a-best": ("s1", "s2"),
                "model-b-partial": ("s1",),
                "model-c-zero-valid": ("s1", "s2"),
            }.items():
                model_dir = root / "raw" / model
                model_dir.mkdir(parents=True)
                for sample_id in sample_ids:
                    (model_dir / f"{sample_id}.txt").write_text("raw", encoding="utf-8")

            selected = select_best_full_runs(
                [
                    row("model-a-old", "Model A", 1, 0.9),
                    row("model-a-best", "Model A", 2, 0.8),
                    row("model-b-partial", "Model B", 1, 0.9),
                    row("model-c-zero-valid", "Model C", 0, 0.0),
                ],
                root / "index.jsonl",
                root / "raw",
                root / "pred",
                root / "errors",
            )

        self.assertEqual([item["model"] for item in selected], ["model-a-best"])
        self.assertEqual(selected[0]["n_attempted"], "2")


if __name__ == "__main__":
    unittest.main()
