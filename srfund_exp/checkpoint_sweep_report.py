from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from formtsr_exp.io_utils import read_jsonl, write_json

from .transfer_figure import METRICS, _formtsr_rows, cluster_bootstrap_ci


CHECKPOINTS = (
    ("qwen36_35b_pre_sft_dev", "Pre-SFT", 0),
    ("qwen36_35b_sft_step100_dev", "FormStruct-SFT@100", 100),
    ("qwen36_35b_sft_step200_dev", "FormStruct-SFT@200", 200),
    ("qwen36_35b_sft_final_dev", "FormStruct-SFT@final", 307),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the preregistered SRFUND-dev checkpoint sweep."
    )
    parser.add_argument(
        "--index",
        default="outputs/srfund_transfer_exploratory/splits/dev/index.jsonl",
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/srfund_transfer_exploratory/checkpoint_sweep",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Summarize only checkpoints whose per-model metrics contain the full dev set.",
    )
    return parser.parse_args()


def select_checkpoint(
    selection_scores: Mapping[int, float], tsr_deltas: Mapping[int, float]
) -> int:
    candidates = [step for step in selection_scores if step > 0]
    if not candidates:
        raise ValueError("no SFT checkpoint is available for selection")
    return max(
        candidates,
        key=lambda step: (selection_scores[step], tsr_deltas[step], -step),
    )


def _paired_delta_rows(
    base_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    base = {str(row["sample_id"]): row for row in base_rows}
    checkpoint = {str(row["sample_id"]): row for row in checkpoint_rows}
    if set(base) != set(checkpoint):
        raise ValueError("checkpoint sweep predictions do not share the same page IDs")
    return [
        {
            "sample_id": sample_id,
            "cluster": str(checkpoint[sample_id]["cluster"]),
            "delta": float(checkpoint[sample_id][metric])
            - float(base[sample_id][metric]),
        }
        for sample_id in sorted(base)
    ]


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

    checkpoint_rows = sorted(
        {str(row["checkpoint"]): row for row in rows}.values(),
        key=lambda row: int(row["checkpoint_order"]),
    )
    checkpoints = [str(row["checkpoint"]) for row in checkpoint_rows]
    palette = ("#73777F", "#D98880", "#C43C4E", "#8F2537")
    colors = palette[: len(checkpoints)]
    by_key = {
        (str(row["metric"]), str(row["checkpoint"])): row for row in rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharey=True)
    for metric_index, metric in enumerate(METRICS):
        axis = axes[metric_index]
        metric_rows = [by_key[(metric, checkpoint)] for checkpoint in checkpoints]
        values = [float(row["mean_score"]) for row in metric_rows]
        lows = [float(row["ci95_low"]) for row in metric_rows]
        highs = [float(row["ci95_high"]) for row in metric_rows]
        errors = np.asarray(
            [
                [max(0.0, value - low) for value, low in zip(values, lows)],
                [max(0.0, high - value) for value, high in zip(values, highs)],
            ]
        )
        bars = axis.bar(
            np.arange(len(checkpoints)),
            values,
            width=0.68,
            color=colors,
            edgecolor="#333333",
            linewidth=0.5,
            yerr=errors,
            error_kw={
                "ecolor": "#222222",
                "elinewidth": 0.8,
                "capsize": 2.5,
                "capthick": 0.8,
            },
        )
        for bar, row in zip(bars, metric_rows):
            if str(row["selected"]).lower() == "true":
                bar.set_linewidth(1.8)
                bar.set_edgecolor("#111111")
        axis.set_title(metric, fontsize=10.5)
        axis.set_xticks(np.arange(len(checkpoints)))
        short_labels = {
            "Pre-SFT": "Pre",
            "FormStruct-SFT@100": "100",
            "FormStruct-SFT@200": "200",
            "FormStruct-SFT@final": "Final",
        }
        axis.set_xticklabels(
            [short_labels.get(checkpoint, checkpoint) for checkpoint in checkpoints],
            fontsize=8.5,
        )
        axis.set_xlabel("Training step", fontsize=9)
        axis.set_ylim(0.0, 100.0)
        axis.grid(axis="y", color="#D8DADD", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if metric_index == 0:
            axis.set_ylabel("Score (%)", fontsize=9.5)
    fig.suptitle("Qwen3.6-35B-A3B on SRFUND-dev (exploratory)", fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.83, bottom=0.18, wspace=0.14)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    index_path = Path(args.index)
    run_dir = Path(args.run_dir)
    index_rows = read_jsonl(index_path)
    expected_ids = {str(row["sample_id"]) for row in index_rows}
    checkpoint_rows: dict[int, list[dict[str, Any]]] = {}
    valid_counts: dict[int, int] = {}
    available_checkpoints: list[tuple[str, str, int]] = []
    for checkpoint in CHECKPOINTS:
        model_name, _, step = checkpoint
        metrics_path = run_dir / "per_model_metrics" / f"{model_name}.jsonl"
        n_attempted = len(read_jsonl(metrics_path)) if metrics_path.is_file() else 0
        if n_attempted != len(index_rows):
            if args.allow_partial:
                continue
            raise ValueError(
                f"incomplete checkpoint {model_name}: attempted={n_attempted}, "
                f"expected={len(index_rows)}"
            )
        available_checkpoints.append(checkpoint)
        pred_dir = run_dir / "pred" / model_name
        valid_ids = {path.stem for path in pred_dir.glob("*.json")}
        extra_ids = valid_ids - expected_ids
        if extra_ids:
            raise ValueError(f"unexpected prediction IDs for {model_name}: {sorted(extra_ids)[:3]}")
        valid_counts[step] = len(valid_ids & expected_ids)
        checkpoint_rows[step] = _formtsr_rows(index_path, pred_dir)

    if not available_checkpoints or available_checkpoints[0][2] != 0:
        raise ValueError("a complete Pre-SFT baseline is required")
    if len(available_checkpoints) < 2:
        raise ValueError("at least one complete SFT checkpoint is required")

    base_rows = checkpoint_rows[0]
    condition_stats: dict[tuple[int, str], dict[str, float]] = {}
    delta_stats: dict[tuple[int, str], dict[str, float]] = {}
    for checkpoint_index, (_, _, step) in enumerate(available_checkpoints):
        rows = checkpoint_rows[step]
        for metric_index, metric in enumerate(METRICS):
            mean = math.fsum(float(row[metric]) for row in rows) / len(rows)
            low, high = cluster_bootstrap_ci(
                rows,
                value_field=metric,
                cluster_field="cluster",
                iterations=args.bootstrap_iterations,
                seed=args.seed + checkpoint_index * 101 + metric_index,
            )
            condition_stats[(step, metric)] = {
                "mean": mean,
                "low": low,
                "high": high,
            }
            if step == 0:
                delta_stats[(step, metric)] = {"mean": 0.0, "low": 0.0, "high": 0.0}
                continue
            paired_rows = _paired_delta_rows(base_rows, rows, metric)
            delta_low, delta_high = cluster_bootstrap_ci(
                paired_rows,
                value_field="delta",
                cluster_field="cluster",
                iterations=args.bootstrap_iterations,
                seed=args.seed + 5003 + checkpoint_index * 101 + metric_index,
            )
            delta_stats[(step, metric)] = {
                "mean": math.fsum(float(row["delta"]) for row in paired_rows)
                / len(paired_rows),
                "low": delta_low,
                "high": delta_high,
            }

    selection_scores = {
        step: math.fsum(delta_stats[(step, metric)]["mean"] for metric in METRICS)
        / len(METRICS)
        for _, _, step in available_checkpoints
    }
    tsr_deltas = {
        step: delta_stats[(step, "TSR-path")]["mean"]
        for _, _, step in available_checkpoints
    }
    selected_step = select_checkpoint(selection_scores, tsr_deltas)

    output_rows: list[dict[str, Any]] = []
    checkpoint_order = {step: index for index, (_, _, step) in enumerate(CHECKPOINTS, start=1)}
    for model_name, label, step in available_checkpoints:
        for metric_index, metric in enumerate(METRICS, start=1):
            condition = condition_stats[(step, metric)]
            delta = delta_stats[(step, metric)]
            output_rows.append(
                {
                    "model": "Qwen3.6-35B-A3B",
                    "model_order": 1,
                    "metric": metric,
                    "metric_order": metric_index,
                    "checkpoint": label,
                    "checkpoint_order": checkpoint_order[step],
                    "training_step": step,
                    "eval_dataset": "SRFUND-dev",
                    "mean_score": 100.0 * condition["mean"],
                    "ci95_low": 100.0 * condition["low"],
                    "ci95_high": 100.0 * condition["high"],
                    "delta_vs_pre_sft": 100.0 * delta["mean"],
                    "delta_ci95_low": 100.0 * delta["low"],
                    "delta_ci95_high": 100.0 * delta["high"],
                    "selection_score": 100.0 * selection_scores[step],
                    "selected": step == selected_step,
                    "n_samples": len(index_rows),
                    "n_valid_predictions": valid_counts[step],
                    "invalid_policy": "invalid_or_missing_prediction_scores_zero",
                    "n_clusters": len({str(row["template_name"]) for row in index_rows}),
                    "result_status": "exploratory_dev_only",
                    "run_model_name": model_name,
                }
            )

    result_path = run_dir / "checkpoint_sweep_results.csv"
    _write_csv(result_path, output_rows)
    _plot(run_dir / "checkpoint_sweep_trend", output_rows)
    write_json(
        run_dir / "selection.json",
        {
            "status": "exploratory_dev_only",
            "selected_step": selected_step,
            "selected_checkpoint": next(
                label for _, label, step in available_checkpoints if step == selected_step
            ),
            "selection_score_points": 100.0 * selection_scores[selected_step],
            "selection_rule": (
                "Maximize the unweighted mean paired gain across Value-nED, "
                "Schema-nTED, and TSR-path; break ties by TSR-path gain, then "
                "the earlier training step."
            ),
            "checkpoint_selection_used_locked_test": False,
            "partial": len(available_checkpoints) != len(CHECKPOINTS),
            "available_steps": [step for _, _, step in available_checkpoints],
            "n_dev": len(index_rows),
            "n_valid_predictions": {str(step): count for step, count in valid_counts.items()},
            "invalid_policy": "invalid_or_missing_prediction_scores_zero",
        },
    )
    print(f"selected FormStruct-SFT step {selected_step}")
    print(f"wrote {len(output_rows)} rows -> {result_path}")


if __name__ == "__main__":
    main()
