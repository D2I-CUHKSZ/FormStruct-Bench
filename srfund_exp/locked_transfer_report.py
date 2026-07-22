from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from formtsr_exp.io_utils import read_json, read_jsonl, write_json

from .checkpoint_sweep_report import _paired_delta_rows
from .transfer_figure import METRICS, _formtsr_rows, cluster_bootstrap_ci


RUNS = (
    ("qwen36_35b_pre_sft_locked400", "Pre-SFT", 0),
    ("qwen36_35b_sft_step100_locked400", "FormStruct-SFT@100", 100),
)
INVALID_POLICY = "invalid_or_missing_prediction_scores_zero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the frozen 400-page SRFUND transfer evaluation."
    )
    parser.add_argument(
        "--index", default="outputs/srfund_transfer_locked400/split/index.jsonl"
    )
    parser.add_argument(
        "--protocol", default="outputs/srfund_transfer_locked400/split/protocol.json"
    )
    parser.add_argument("--run-dir", default="outputs/srfund_transfer_locked400/eval")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path_base: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(str(row["metric"]), str(row["checkpoint"])): row for row in rows}
    labels = ("Pre-SFT", "FormStruct-SFT@100")
    colors = ("#73777F", "#1769AA")
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), sharey=True)
    for metric_index, metric in enumerate(METRICS):
        axis = axes[metric_index]
        metric_rows = [by_key[(metric, label)] for label in labels]
        values = [float(row["mean_score"]) for row in metric_rows]
        lows = [float(row["ci95_low"]) for row in metric_rows]
        highs = [float(row["ci95_high"]) for row in metric_rows]
        errors = np.asarray(
            [
                [max(0.0, value - low) for value, low in zip(values, lows)],
                [max(0.0, high - value) for value, high in zip(values, highs)],
            ]
        )
        axis.bar(
            np.arange(2),
            values,
            width=0.62,
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
            yerr=errors,
            error_kw={"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2.5},
        )
        axis.set_title(metric, fontsize=10.5)
        axis.set_xticks(np.arange(2), ("Pre", "SFT@100"), fontsize=8.5)
        axis.set_ylim(0.0, 100.0)
        axis.grid(axis="y", color="#D8DADD", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if metric_index == 0:
            axis.set_ylabel("Score (%)", fontsize=9.5)
    fig.suptitle("Qwen3.6-35B-A3B on frozen SRFUND-400", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.82, bottom=0.16, wspace=0.14)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    index_path = Path(args.index)
    protocol_path = Path(args.protocol)
    run_dir = Path(args.run_dir)
    index_rows = read_jsonl(index_path)
    protocol = read_json(protocol_path)
    if int(protocol.get("n_pages", -1)) != len(index_rows):
        raise ValueError("locked protocol and index page counts disagree")

    expected_ids = {str(row["sample_id"]) for row in index_rows}
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    valid_counts: dict[int, int] = {}
    for model_name, _, step in RUNS:
        metrics_path = run_dir / "per_model_metrics" / f"{model_name}.jsonl"
        attempted = read_jsonl(metrics_path) if metrics_path.is_file() else []
        if len(attempted) != len(index_rows):
            raise ValueError(
                f"incomplete locked run {model_name}: attempted={len(attempted)}, "
                f"expected={len(index_rows)}"
            )
        pred_dir = run_dir / "pred" / model_name
        valid_ids = {path.stem for path in pred_dir.glob("*.json")}
        if valid_ids - expected_ids:
            raise ValueError(f"unexpected prediction IDs for {model_name}")
        valid_counts[step] = len(valid_ids & expected_ids)
        rows_by_step[step] = _formtsr_rows(index_path, pred_dir)

    base_rows = rows_by_step[0]
    output_rows: list[dict[str, Any]] = []
    delta_summary: dict[str, dict[str, Any]] = {}
    for metric_index, metric in enumerate(METRICS, start=1):
        paired = _paired_delta_rows(base_rows, rows_by_step[100], metric)
        delta_mean = math.fsum(float(row["delta"]) for row in paired) / len(paired)
        delta_low, delta_high = cluster_bootstrap_ci(
            paired,
            value_field="delta",
            cluster_field="cluster",
            iterations=args.bootstrap_iterations,
            seed=args.seed + 5003 + metric_index,
        )
        delta_summary[metric] = {
            "mean_points": 100.0 * delta_mean,
            "ci95_low_points": 100.0 * delta_low,
            "ci95_high_points": 100.0 * delta_high,
            "significant_positive": delta_low > 0.0,
        }
        for checkpoint_order, (_, label, step) in enumerate(RUNS, start=1):
            condition_rows = rows_by_step[step]
            mean = math.fsum(float(row[metric]) for row in condition_rows) / len(condition_rows)
            low, high = cluster_bootstrap_ci(
                condition_rows,
                value_field=metric,
                cluster_field="cluster",
                iterations=args.bootstrap_iterations,
                seed=args.seed + checkpoint_order * 101 + metric_index,
            )
            output_rows.append(
                {
                    "model": "Qwen3.6-35B-A3B",
                    "model_order": 1,
                    "metric": metric,
                    "metric_order": metric_index,
                    "checkpoint": label,
                    "checkpoint_order": checkpoint_order,
                    "training_step": step,
                    "eval_dataset": "SRFUND-locked400",
                    "mean_score": 100.0 * mean,
                    "ci95_low": 100.0 * low,
                    "ci95_high": 100.0 * high,
                    "delta_vs_pre_sft": 0.0 if step == 0 else 100.0 * delta_mean,
                    "delta_ci95_low": 0.0 if step == 0 else 100.0 * delta_low,
                    "delta_ci95_high": 0.0 if step == 0 else 100.0 * delta_high,
                    "n_samples": len(index_rows),
                    "n_valid_predictions": valid_counts[step],
                    "n_clusters": len({str(row["template_name"]) for row in index_rows}),
                    "invalid_policy": INVALID_POLICY,
                    "result_status": "locked_final_test",
                    "run_model_name": RUNS[checkpoint_order - 1][0],
                }
            )

    result_path = run_dir / "locked_transfer_results.csv"
    _write_csv(result_path, output_rows)
    _plot(run_dir / "locked_transfer", output_rows)
    write_json(
        run_dir / "locked_transfer_summary.json",
        {
            "status": "locked_final_test",
            "selected_checkpoint": protocol["selected_checkpoint"],
            "selected_step": protocol["selected_step"],
            "n_pages": len(index_rows),
            "n_valid_predictions": {str(step): count for step, count in valid_counts.items()},
            "invalid_policy": INVALID_POLICY,
            "delta": delta_summary,
            "index_sha256": _sha256(index_path),
            "protocol_sha256": _sha256(protocol_path),
            "selection_source_sha256": protocol["selection_source_sha256"],
        },
    )
    print(f"wrote {len(output_rows)} rows -> {result_path}")


if __name__ == "__main__":
    main()
