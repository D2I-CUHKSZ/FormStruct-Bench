from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .io_utils import ensure_dir, write_json


METRICS = [
    "TSR-path",
    "VAcc",
    "WAcc",
    "Page-EM",
    "Schema-nTED",
    "Value-nED",
    "R-F1",
    "R-F1@0.75",
    "LIG-F1",
]

STRUCTURE_METRICS = {"R-F1", "R-F1@0.75", "LIG-F1"}


def _read_csv_by_model(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {str(row["model"]): row for row in csv.DictReader(fh)}


def load_metric_values(
    main_results: Path,
    page_em_results: Path,
    document_results: Path,
    structure_results: Path,
) -> tuple[list[str], dict[str, np.ndarray]]:
    main = _read_csv_by_model(main_results)
    page = _read_csv_by_model(page_em_results)
    document = _read_csv_by_model(document_results)
    structure = _read_csv_by_model(structure_results)
    models = [
        model
        for model, row in document.items()
        if row.get("comparison_status") == "comparable_raw"
    ]
    values: dict[str, np.ndarray] = {}
    for metric in METRICS:
        source = (
            page
            if metric == "Page-EM"
            else document
            if metric in {"Schema-nTED", "Value-nED"}
            else structure
            if metric in STRUCTURE_METRICS
            else main
        )
        try:
            values[metric] = np.asarray([float(source[model][metric]) for model in models], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"could not load numeric {metric} for all comparable models") from exc
        if np.all(values[metric] == values[metric][0]):
            raise ValueError(f"cannot correlate constant metric: {metric}")
    return models, values


def compute_correlations(
    values: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[dict[str, Any]]]:
    pearson: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    spearman: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    pairs: list[dict[str, Any]] = []
    for index, metric_a in enumerate(METRICS):
        for metric_b in METRICS:
            pearson[metric_a][metric_b] = float(pearsonr(values[metric_a], values[metric_b]).statistic)
            spearman[metric_a][metric_b] = float(spearmanr(values[metric_a], values[metric_b]).statistic)
        for metric_b in METRICS[index + 1 :]:
            pearson_result = pearsonr(values[metric_a], values[metric_b])
            spearman_result = spearmanr(values[metric_a], values[metric_b])
            pairs.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "pearson_r": float(pearson_result.statistic),
                    "pearson_p": float(pearson_result.pvalue),
                    "spearman_rho": float(spearman_result.statistic),
                    "spearman_p": float(spearman_result.pvalue),
                }
            )
    pairs.sort(key=lambda row: abs(float(row["pearson_r"])), reverse=True)
    return pearson, spearman, pairs


def _write_matrix(path: Path, matrix: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", *METRICS])
        for metric in METRICS:
            writer.writerow([metric, *(f"{matrix[metric][other]:.6f}" for other in METRICS)])


def _markdown_matrix(matrix: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Metric | " + " | ".join(METRICS) + " |",
        "| --- | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | "
            + " | ".join(f"{matrix[metric][other]:.3f}" for other in METRICS)
            + " |"
        )
    return lines


def write_results(
    out_dir: Path,
    models: list[str],
    pearson: dict[str, dict[str, float]],
    spearman: dict[str, dict[str, float]],
    pairs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    _write_matrix(out_dir / "metric_correlations_model_level_pearson.csv", pearson)
    _write_matrix(out_dir / "metric_correlations_model_level_spearman.csv", spearman)
    with (out_dir / "metric_correlations_model_level_pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        columns = ["metric_a", "metric_b", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in pairs:
            writer.writerow(
                {
                    key: f"{row[key]:.6g}" if isinstance(row[key], float) else row[key]
                    for key in columns
                }
            )
    write_json(out_dir / "metric_correlations_model_level_metadata.json", metadata)

    lines = [
        "# Model-Level Metric Correlations",
        "",
        f"Scope: {len(models)} comparable raw runs. Correlations are across model-level aggregate scores, not pages.",
        "",
        "## Pearson",
        "",
        *_markdown_matrix(pearson),
        "",
        "## Spearman",
        "",
        *_markdown_matrix(spearman),
        "",
        "## Strongest Pearson Pairs",
        "",
        "| Metric A | Metric B | Pearson r | p-value | Spearman rho |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in pairs[:15]:
        lines.append(
            f"| {row['metric_a']} | {row['metric_b']} | {row['pearson_r']:.3f} | "
            f"{row['pearson_p']:.4g} | {row['spearman_rho']:.3f} |"
        )
    (out_dir / "metric_correlations_model_level.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate current model-level FormTSR metrics.")
    parser.add_argument("--main-results", default="outputs/main_exp/main_results.csv")
    parser.add_argument("--page-em-results", default="outputs/main_exp/page_em_results.csv")
    parser.add_argument("--document-results", default="outputs/main_exp/document_similarity_results.csv")
    parser.add_argument("--structure-results", default="outputs/main_exp/corrected_structure_metrics.csv")
    parser.add_argument("--out", default="outputs/main_exp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models, values = load_metric_values(
        Path(args.main_results),
        Path(args.page_em_results),
        Path(args.document_results),
        Path(args.structure_results),
    )
    pearson, spearman, pairs = compute_correlations(values)
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "model-level comparable_raw runs only",
        "n_models": len(models),
        "models": models,
        "metrics": METRICS,
        "caveats": [
            "n=9 is small; correlations are descriptive.",
            "VAcc and WAcc are retained only to diagnose redundancy with the selected reporting metrics.",
            "R-F1, R-F1@0.75, and LIG-F1 come from the corrected coordinate/type-normalized structure report.",
            "Page-EM is sparse: only two comparable raw runs have non-zero aggregate Page-EM.",
            "Across-model correlations mix metric overlap with general model-quality differences and coverage effects.",
        ],
    }
    write_results(Path(args.out), models, pearson, spearman, pairs, metadata)
    print(f"wrote metric correlations -> {Path(args.out) / 'metric_correlations_model_level.md'}")


if __name__ == "__main__":
    main()
