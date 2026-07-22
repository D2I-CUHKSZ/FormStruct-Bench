from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from formtsr_exp.config import load_config
from formtsr_exp.document_similarity_report import (
    SchemaNode,
    build_schema_tree,
    extract_normalized_values,
    schema_nted,
    value_ned,
)
from formtsr_exp.io_utils import read_json, read_jsonl, write_json
from formtsr_exp.metrics import field_accuracy, unwrap_answer

from .core import Entity, Sample, load_samples, normalize_text, validate_prediction


METRICS = ("Value-nED", "Schema-nTED", "TSR-path")
CONDITIONS = (
    "pre_formtsr",
    "sft_formtsr",
    "pre_srfund",
    "sft_srfund",
)
CONDITION_LABELS = {
    "pre_formtsr": "Pre-SFT / FormStruct",
    "sft_formtsr": "FormStruct-SFT / FormStruct",
    "pre_srfund": "Pre-SFT / SRFUND (aligned)",
    "sft_srfund": "FormStruct-SFT / SRFUND (aligned)",
}
COLORS = {
    "pre_formtsr": "#73777F",
    "sft_formtsr": "#C43C4E",
    "pre_srfund": "#8ECAE6",
    "sft_srfund": "#1769AA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the two-model FormStruct/SRFUND SFT transfer figure."
    )
    parser.add_argument("--config", default="configs/sft_transfer_figure.yaml")
    return parser.parse_args()


def _read_prediction(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def evaluate_formtsr_prediction(
    gt: Any, prediction: Mapping[str, Any] | None
) -> dict[str, float]:
    if prediction is None:
        return {metric: 0.0 for metric in METRICS}
    answer = unwrap_answer(prediction)
    path_score, _ = field_accuracy(prediction, gt)
    return {
        "Value-nED": value_ned(
            extract_normalized_values(answer), extract_normalized_values(gt)
        ),
        "Schema-nTED": schema_nted(build_schema_tree(answer), build_schema_tree(gt)),
        "TSR-path": float(path_score) if isinstance(path_score, (int, float)) else 0.0,
    }


def _valid_edges(
    entities: Sequence[Entity], edges: Iterable[tuple[str, str]]
) -> set[tuple[str, str]]:
    entity_ids = {entity.entity_id for entity in entities}
    return {
        (str(source), str(target))
        for source, target in edges
        if str(source) in entity_ids
        and str(target) in entity_ids
        and str(source) != str(target)
    }


def _entity_graph(
    entities: Sequence[Entity], edges: Iterable[tuple[str, str]]
) -> tuple[dict[str, Entity], dict[str, set[str]], dict[str, set[str]]]:
    by_id = {entity.entity_id: entity for entity in entities}
    children = {entity_id: set() for entity_id in by_id}
    parents = {entity_id: set() for entity_id in by_id}
    for source, target in _valid_edges(entities, edges):
        children[source].add(target)
        parents[target].add(source)
    return by_id, children, parents


def _canonical_graph(
    entities: Sequence[Entity], edges: Iterable[tuple[str, str]]
) -> tuple[dict[str, Entity], dict[str, set[str]], dict[str, str]]:
    by_id, _, all_parents = _entity_graph(entities, edges)

    def sort_key(entity_id: str) -> tuple[str, str, str]:
        entity = by_id[entity_id]
        return entity.label, _semantic_name(entity), entity_id

    parent_by_child: dict[str, str] = {}
    children = {entity_id: set() for entity_id in by_id}
    for child_id, parent_ids in all_parents.items():
        if not parent_ids:
            continue
        parent_id = min(parent_ids, key=sort_key)
        parent_by_child[child_id] = parent_id
        children[parent_id].add(child_id)
    return by_id, children, parent_by_child


def _semantic_name(entity: Entity) -> str:
    text = normalize_text(entity.text)
    return text if text else entity.label


def build_srfund_schema_tree(
    entities: Sequence[Entity], edges: Iterable[tuple[str, str]]
) -> SchemaNode:
    by_id, children, parent_by_child = _canonical_graph(entities, edges)

    def build(entity_id: str, ancestry: frozenset[str]) -> SchemaNode:
        entity = by_id[entity_id]
        if entity_id in ancestry:
            return SchemaNode(f"key:{entity.label}|scalar")
        descendants = [
            child_id
            for child_id in children[entity_id]
            if by_id[child_id].label != "answer"
        ]
        answers = [
            child_id
            for child_id in children[entity_id]
            if by_id[child_id].label == "answer"
        ]
        descendants.sort(key=lambda item: (by_id[item].label, _semantic_name(by_id[item]), item))
        next_ancestry = ancestry | {entity_id}
        child_nodes = [build(child_id, next_ancestry) for child_id in descendants]
        child_nodes.extend(SchemaNode("item|scalar") for _ in answers)
        if child_nodes:
            kind = "array" if answers and not descendants and len(answers) > 1 else "object"
        else:
            kind = "scalar"
        key = "answer" if entity.label == "answer" else _semantic_name(entity)
        return SchemaNode(f"key:{key}|{kind}", tuple(child_nodes))

    roots = [
        entity_id
        for entity_id, entity in by_id.items()
        if entity_id not in parent_by_child and entity.label != "answer"
    ]
    roots.sort(key=lambda item: (by_id[item].label, _semantic_name(by_id[item]), item))
    reachable: set[str] = set()

    def mark(entity_id: str) -> None:
        if entity_id in reachable:
            return
        reachable.add(entity_id)
        for child_id in children[entity_id]:
            mark(child_id)

    for root in roots:
        mark(root)
    detached = [
        entity_id
        for entity_id, entity in by_id.items()
        if entity_id not in reachable and entity.label != "answer"
    ]
    detached.sort(key=lambda item: (by_id[item].label, _semantic_name(by_id[item]), item))
    root_nodes = [build(entity_id, frozenset()) for entity_id in roots + detached]
    detached_answers = [
        entity
        for entity in entities
        if entity.label == "answer" and entity.entity_id not in parent_by_child
    ]
    root_nodes.extend(SchemaNode("key:answer|scalar") for _ in detached_answers)
    return SchemaNode("root|object", tuple(root_nodes))


def _answer_paths(
    entities: Sequence[Entity], edges: Iterable[tuple[str, str]]
) -> Counter[tuple[tuple[str, ...], str]]:
    by_id, _, parent_by_child = _canonical_graph(entities, edges)

    def parent_path(entity_id: str) -> tuple[str, ...]:
        output: list[str] = []
        seen = {entity_id}
        current = entity_id
        while current in parent_by_child:
            parent_id = parent_by_child[current]
            if parent_id in seen:
                break
            seen.add(parent_id)
            parent = by_id[parent_id]
            name = _semantic_name(parent)
            if name:
                output.append(name)
            current = parent_id
        output.reverse()
        return tuple(output)

    result: Counter[tuple[tuple[str, ...], str]] = Counter()
    for entity in entities:
        if entity.label != "answer":
            continue
        value = normalize_text(entity.text)
        if not value:
            continue
        path = parent_path(entity.entity_id)
        result[(path or ("answer",), value)] += 1
    return result


def _strict_path_score(
    pred_entities: Sequence[Entity],
    pred_edges: Iterable[tuple[str, str]],
    gt_entities: Sequence[Entity],
    gt_edges: Iterable[tuple[str, str]],
) -> float:
    pred = _answer_paths(pred_entities, pred_edges)
    gt = _answer_paths(gt_entities, gt_edges)
    total = sum(gt.values())
    if total == 0:
        return 1.0 if not pred else 0.0
    return sum((pred & gt).values()) / total


def evaluate_srfund_prediction(
    sample: Sample, prediction: Mapping[str, Any] | None
) -> dict[str, float]:
    if prediction is None:
        return {metric: 0.0 for metric in METRICS}
    valid, _ = validate_prediction(prediction)
    if not valid:
        return {metric: 0.0 for metric in METRICS}
    pred_entities = tuple(
        Entity(
            entity_id=str(raw["id"]),
            label=str(raw["label"]),
            bbox=tuple(float(value) for value in raw["bbox"]),
            text=str(raw["text"]),
        )
        for raw in prediction["entities"]
    )
    pred_edges = {
        (str(raw["source"]), str(raw["target"])) for raw in prediction["links"]
    }
    gt_edges = set(sample.links) | set(sample.hierarchy_edges)
    pred_values = tuple(
        normalize_text(entity.text)
        for entity in pred_entities
        if entity.label == "answer" and normalize_text(entity.text)
    )
    gt_values = tuple(
        normalize_text(entity.text)
        for entity in sample.entities
        if entity.label == "answer" and normalize_text(entity.text)
    )
    return {
        "Value-nED": value_ned(pred_values, gt_values),
        "Schema-nTED": schema_nted(
            build_srfund_schema_tree(pred_entities, pred_edges),
            build_srfund_schema_tree(sample.entities, gt_edges),
        ),
        "TSR-path": _strict_path_score(
            pred_entities, pred_edges, sample.entities, gt_edges
        ),
    }


def cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    cluster_field: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        clusters.setdefault(str(row[cluster_field]), []).append(float(row[value_field]))
    keys = sorted(clusters)
    if not keys:
        return 0.0, 0.0
    sums = np.asarray([math.fsum(clusters[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    chunk = 1000
    for start in range(0, iterations, chunk):
        size = min(chunk, iterations - start)
        draws = rng.integers(0, len(keys), size=(size, len(keys)))
        estimates[start : start + size] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_csv_rows(
    condition_rows: Sequence[Mapping[str, Any]],
    model_order: Sequence[str],
) -> list[dict[str, Any]]:
    model_indices = {family: index for index, family in enumerate(model_order, start=1)}
    metric_indices = {metric: index for index, metric in enumerate(METRICS, start=1)}
    dataset_indices = {"FormStruct": 1, "SRFUND-Aligned": 2, "SRFUND-Native": 3}
    sorted_rows = sorted(
        condition_rows,
        key=lambda row: (
            model_indices[str(row["family"])],
            metric_indices[str(row["metric"])],
            dataset_indices[str(row["dataset"])],
            bool(row["tuned"]),
        ),
    )
    output: list[dict[str, Any]] = []
    for row in sorted_rows:
        output.append(
            {
                "model": row["model"],
                "model_order": model_indices[str(row["family"])],
                "metric": row["metric"],
                "metric_order": metric_indices[str(row["metric"])],
                "checkpoint": "FormStruct-SFT" if bool(row["tuned"]) else "Pre-SFT",
                "eval_dataset": row["dataset"],
                "mean_score": 100.0 * float(row["mean"]),
                "ci95_low": 100.0 * float(row["ci95_low"]),
                "ci95_high": 100.0 * float(row["ci95_high"]),
            }
        )
    return output


def _formtsr_rows(index_path: Path, pred_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_row in read_jsonl(index_path):
        sample_id = str(index_row["sample_id"])
        gt = read_json(Path(str(index_row["label_path"])))
        scores = evaluate_formtsr_prediction(
            gt, _read_prediction(pred_dir / f"{sample_id}.json")
        )
        rows.append(
            {
                "sample_id": sample_id,
                "cluster": str(index_row["template_name"]),
                **scores,
            }
        )
    return rows


def _srfund_rows(samples: Sequence[Sample], pred_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        scores = evaluate_srfund_prediction(
            sample, _read_prediction(pred_dir / f"{sample.sample_id}.json")
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "cluster": sample.language,
                **scores,
            }
        )
    return rows


def _summarize_conditions(
    family: str,
    display_name: str,
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rows = condition_rows[condition]
        for metric_index, metric in enumerate(METRICS):
            mean = math.fsum(float(row[metric]) for row in rows) / len(rows)
            low, high = cluster_bootstrap_ci(
                rows,
                value_field=metric,
                cluster_field="cluster",
                iterations=iterations,
                seed=seed + metric_index + CONDITIONS.index(condition) * 101,
            )
            output.append(
                {
                    "family": family,
                    "model": display_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "dataset": "FormStruct" if condition.endswith("formtsr") else "SRFUND",
                    "tuned": condition.startswith("sft"),
                    "metric": metric,
                    "n_samples": len(rows),
                    "n_clusters": len({str(row["cluster"]) for row in rows}),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def _summarize_native_srfund_conditions(
    family: str,
    display_name: str,
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in ("pre_srfund", "sft_srfund"):
        rows = condition_rows[condition]
        for metric_index, metric in enumerate(METRICS):
            mean = math.fsum(float(row[metric]) for row in rows) / len(rows)
            low, high = cluster_bootstrap_ci(
                rows,
                value_field=metric,
                cluster_field="cluster",
                iterations=iterations,
                seed=seed + 5003 + metric_index + CONDITIONS.index(condition) * 101,
            )
            output.append(
                {
                    "family": family,
                    "model": display_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition].replace("aligned", "native schema"),
                    "dataset": "SRFUND-Native",
                    "tuned": condition.startswith("sft"),
                    "metric": metric,
                    "n_samples": len(rows),
                    "n_clusters": len({str(row["cluster"]) for row in rows}),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def _summarize_deltas(
    family: str,
    display_name: str,
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    comparisons = (
        ("in_domain", "pre_formtsr", "sft_formtsr"),
        ("transfer", "pre_srfund", "sft_srfund"),
    )
    for comparison_index, (comparison, base_name, tuned_name) in enumerate(comparisons):
        base = {str(row["sample_id"]): row for row in condition_rows[base_name]}
        tuned = {str(row["sample_id"]): row for row in condition_rows[tuned_name]}
        if set(base) != set(tuned):
            raise ValueError(f"paired sample mismatch for {family}/{comparison}")
        for metric_index, metric in enumerate(METRICS):
            delta_rows = [
                {
                    "sample_id": sample_id,
                    "cluster": tuned[sample_id]["cluster"],
                    "delta": float(tuned[sample_id][metric]) - float(base[sample_id][metric]),
                }
                for sample_id in sorted(base)
            ]
            low, high = cluster_bootstrap_ci(
                delta_rows,
                value_field="delta",
                cluster_field="cluster",
                iterations=iterations,
                seed=seed + 1009 + comparison_index * 101 + metric_index,
            )
            output.append(
                {
                    "family": family,
                    "model": display_name,
                    "comparison": comparison,
                    "metric": metric,
                    "n_paired": len(delta_rows),
                    "delta": math.fsum(row["delta"] for row in delta_rows) / len(delta_rows),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def _summarize_transfer_pair(
    family: str,
    display_name: str,
    setting: str,
    base_rows: Sequence[Mapping[str, Any]],
    tuned_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    base = {str(row["sample_id"]): row for row in base_rows}
    tuned = {str(row["sample_id"]): row for row in tuned_rows}
    if set(base) != set(tuned):
        raise ValueError(f"paired sample mismatch for {family}/{setting}")
    output: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        delta_rows = [
            {
                "sample_id": sample_id,
                "cluster": tuned[sample_id]["cluster"],
                "delta": float(tuned[sample_id][metric]) - float(base[sample_id][metric]),
            }
            for sample_id in sorted(base)
        ]
        low, high = cluster_bootstrap_ci(
            delta_rows,
            value_field="delta",
            cluster_field="cluster",
            iterations=iterations,
            seed=seed + 7001 + metric_index + (0 if setting == "schema_aligned" else 101),
        )
        output.append(
            {
                "family": family,
                "model": display_name,
                "setting": setting,
                "metric": metric,
                "n_paired": len(delta_rows),
                "delta": math.fsum(float(row["delta"]) for row in delta_rows) / len(delta_rows),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return output


def _plot(path_base: Path, rows: Sequence[Mapping[str, Any]], model_order: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    by_key = {
        (str(row["family"]), str(row["metric"]), str(row["condition"])): row
        for row in rows
    }
    fig, axes = plt.subplots(
        len(model_order),
        len(METRICS),
        figsize=(10.8, 5.7),
        sharey=True,
        constrained_layout=False,
    )
    axes_array = np.atleast_2d(axes)
    for row_index, family in enumerate(model_order):
        display_name = str(next(row["model"] for row in rows if row["family"] == family))
        for column_index, metric in enumerate(METRICS):
            axis = axes_array[row_index, column_index]
            values = [100.0 * float(by_key[(family, metric, condition)]["mean"]) for condition in CONDITIONS]
            lows = [100.0 * float(by_key[(family, metric, condition)]["ci95_low"]) for condition in CONDITIONS]
            highs = [100.0 * float(by_key[(family, metric, condition)]["ci95_high"]) for condition in CONDITIONS]
            errors = np.asarray(
                [
                    [max(0.0, value - low) for value, low in zip(values, lows)],
                    [max(0.0, high - value) for value, high in zip(values, highs)],
                ]
            )
            positions = np.arange(len(CONDITIONS))
            axis.bar(
                positions,
                values,
                width=0.72,
                color=[COLORS[condition] for condition in CONDITIONS],
                edgecolor="#333333",
                linewidth=0.45,
                yerr=errors,
                error_kw={"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2.4, "capthick": 0.8},
            )
            axis.set_ylim(0.0, 100.0)
            axis.set_xticks([])
            axis.grid(axis="y", color="#D8DADD", linewidth=0.55, alpha=0.8)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            axis.spines["left"].set_color("#777777")
            axis.spines["bottom"].set_color("#777777")
            if row_index == 0:
                axis.set_title(metric, fontsize=10.5, pad=7)
            if column_index == 0:
                axis.set_ylabel(f"{display_name}\nScore (%)", fontsize=9.5)
            axis.tick_params(axis="y", labelsize=8.5)

    legend = [
        Patch(facecolor=COLORS[condition], edgecolor="#333333", label=CONDITION_LABELS[condition])
        for condition in CONDITIONS
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.15, wspace=0.16, hspace=0.24)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_boundary(
    path_base: Path,
    rows: Sequence[Mapping[str, Any]],
    model_order: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    settings = ("schema_aligned", "native_schema")
    labels = {"schema_aligned": "Schema aligned", "native_schema": "Native schema"}
    colors = {"schema_aligned": "#2A9D8F", "native_schema": "#73777F"}
    by_key = {
        (str(row["family"]), str(row["metric"]), str(row["setting"])): row
        for row in rows
    }
    maximum = max(
        (abs(100.0 * float(row[key])) for row in rows for key in ("ci95_low", "ci95_high")),
        default=10.0,
    )
    limit = max(5.0, math.ceil(maximum / 5.0) * 5.0)
    fig, axes = plt.subplots(
        len(model_order),
        len(METRICS),
        figsize=(10.8, 5.7),
        sharey=True,
        constrained_layout=False,
    )
    axes_array = np.atleast_2d(axes)
    for row_index, family in enumerate(model_order):
        display_name = str(next(row["model"] for row in rows if row["family"] == family))
        for column_index, metric in enumerate(METRICS):
            axis = axes_array[row_index, column_index]
            values = [100.0 * float(by_key[(family, metric, setting)]["delta"]) for setting in settings]
            lows = [100.0 * float(by_key[(family, metric, setting)]["ci95_low"]) for setting in settings]
            highs = [100.0 * float(by_key[(family, metric, setting)]["ci95_high"]) for setting in settings]
            errors = np.asarray(
                [
                    [max(0.0, value - low) for value, low in zip(values, lows)],
                    [max(0.0, high - value) for value, high in zip(values, highs)],
                ]
            )
            positions = np.arange(len(settings))
            axis.bar(
                positions,
                values,
                width=0.62,
                color=[colors[setting] for setting in settings],
                edgecolor="#333333",
                linewidth=0.45,
                yerr=errors,
                error_kw={"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2.4},
            )
            axis.axhline(0.0, color="#333333", linewidth=0.8)
            axis.set_ylim(-limit, limit)
            axis.set_xticks([])
            axis.grid(axis="y", color="#D8DADD", linewidth=0.55, alpha=0.8)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if row_index == 0:
                axis.set_title(metric, fontsize=10.5, pad=7)
            if column_index == 0:
                axis.set_ylabel(f"{display_name}\nSFT gain (points)", fontsize=9.5)
    fig.legend(
        handles=[Patch(facecolor=colors[item], edgecolor="#333333", label=labels[item]) for item in settings],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.15, wspace=0.16, hspace=0.24)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Condition & Value-nED & Schema-nTED & TSR-path \\",
        r"\midrule",
    ]
    families = []
    for row in rows:
        family = str(row["family"])
        if family not in families:
            families.append(family)
    for family in families:
        model = str(next(row["model"] for row in rows if row["family"] == family))
        for condition in CONDITIONS:
            values = {
                str(row["metric"]): 100.0 * float(row["mean"])
                for row in rows
                if row["family"] == family and row["condition"] == condition
            }
            lines.append(
                f"{model} & {CONDITION_LABELS[condition]} & "
                f"{values['Value-nED']:.2f} & {values['Schema-nTED']:.2f} & "
                f"{values['TSR-path']:.2f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = Path(str(config["output_dir"]))
    iterations = int(config.get("bootstrap_iterations", 10_000))
    seed = int(config.get("seed", 42))
    if iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")

    formtsr_index = Path(str(config["formtsr_index"]))
    srfund_root = Path(str(config["srfund_root"]))
    srfund_aligned_index = Path(str(config["srfund_aligned_index"]))
    srfund_aligned_output = Path(str(config["srfund_aligned_output"]))
    srfund_native_output = Path(str(config["srfund_native_output"]))
    srfund_samples = load_samples(
        srfund_root,
        str(config.get("srfund_split", "validation_balanced")),
        unavailable_limit=int(config.get("unavailable_per_language", 50)),
        seed=seed,
    )

    condition_results: list[dict[str, Any]] = []
    native_results: list[dict[str, Any]] = []
    boundary_results: list[dict[str, Any]] = []
    model_order: list[str] = []
    for model_spec in config["models"]:
        family = str(model_spec["family"])
        display_name = str(model_spec["display_name"])
        model_order.append(family)
        condition_rows: dict[str, list[dict[str, Any]]] = {}
        condition_loaders = {
            "pre_formtsr": lambda: _formtsr_rows(
                formtsr_index, Path(str(model_spec["formtsr_base_pred_dir"]))
            ),
            "sft_formtsr": lambda: _formtsr_rows(
                formtsr_index, Path(str(model_spec["formtsr_sft_pred_dir"]))
            ),
            "pre_srfund": lambda: _formtsr_rows(
                srfund_aligned_index,
                srfund_aligned_output / "pred" / str(model_spec["srfund_aligned_base_model"]),
            ),
            "sft_srfund": lambda: _formtsr_rows(
                srfund_aligned_index,
                srfund_aligned_output / "pred" / str(model_spec["srfund_aligned_sft_model"]),
            ),
        }
        for condition, loader in condition_loaders.items():
            print(f"[{display_name}] scoring {condition}", flush=True)
            condition_rows[condition] = loader()
            print(
                f"[{display_name}] scored {condition}: {len(condition_rows[condition])} pages",
                flush=True,
            )
        model_condition_results = _summarize_conditions(
            family,
            display_name,
            condition_rows,
            iterations=iterations,
            seed=seed,
        )
        for row in model_condition_results:
            if row["dataset"] == "SRFUND":
                row["dataset"] = "SRFUND-Aligned"
        condition_results.extend(model_condition_results)
        native_condition_rows = {
            "pre_srfund": _srfund_rows(
                srfund_samples,
                srfund_native_output / "pred" / str(model_spec["srfund_native_base_model"]),
            ),
            "sft_srfund": _srfund_rows(
                srfund_samples,
                srfund_native_output / "pred" / str(model_spec["srfund_native_sft_model"]),
            ),
        }
        native_results.extend(
            _summarize_native_srfund_conditions(
                family,
                display_name,
                native_condition_rows,
                iterations=iterations,
                seed=seed,
            )
        )
        boundary_results.extend(
            _summarize_transfer_pair(
                family,
                display_name,
                "schema_aligned",
                condition_rows["pre_srfund"],
                condition_rows["sft_srfund"],
                iterations=iterations,
                seed=seed,
            )
        )
        boundary_results.extend(
            _summarize_transfer_pair(
                family,
                display_name,
                "native_schema",
                native_condition_rows["pre_srfund"],
                native_condition_rows["sft_srfund"],
                iterations=iterations,
                seed=seed,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("condition_metrics.csv", "delta_metrics.csv"):
        (output_dir / legacy_name).unlink(missing_ok=True)
    _write_csv(
        output_dir / "sft_transfer_results.csv",
        _plot_csv_rows([*condition_results, *native_results], model_order),
    )
    _plot(output_dir / "sft_transfer", condition_results, model_order)
    _plot_boundary(output_dir / "sft_transfer_boundary", boundary_results, model_order)
    _write_latex(output_dir / "sft_transfer_table.tex", condition_results)
    write_json(
        output_dir / "protocol.json",
        {
            "layout": "2 model rows x 3 metric columns; four bars per panel",
            "metrics": list(METRICS),
            "conditions": list(CONDITIONS),
            "score_scale": "0-100 in figure and CSV",
            "csv": "sft_transfer_results.csv has 36 rows: two models x three metrics x two checkpoints x FormStruct/SRFUND-Aligned/SRFUND-Native; mean and 95% CI occupy columns",
            "formtsr_scope": "full 1100-page test split; missing/invalid predictions score zero",
            "srfund_scope": "balanced 400-page validation slice; missing/invalid predictions score zero",
            "transfer_interpretation": {
                "SRFUND-Aligned": "visual/domain transfer with the FormStruct output schema held fixed",
                "SRFUND-Native": "joint visual/domain and output-schema transfer boundary",
            },
            "srfund_metric_mapping": {
                "aligned": {
                    "Value-nED": "FormStruct value multiset normalized edit similarity over the converted answer tree",
                    "Schema-nTED": "FormStruct normalized tree-edit similarity over the converted answer tree",
                    "TSR-path": "FormStruct exact normalized answer-path recall over the converted answer tree",
                },
                "native": {
                    "Value-nED": "precision-aware soft matching over entities labeled answer",
                    "Schema-nTED": "normalized tree-edit similarity over native header/question/link structure",
                    "TSR-path": "strict recall of native ancestor paths paired with exact normalized answer text",
                },
            },
            "confidence_intervals": {
                "method": "clustered percentile bootstrap of page means",
                "replicates": iterations,
                "seed": seed,
                "FormStruct_cluster": "template",
                "SRFUND_cluster": "language",
            },
        },
    )
    print(f"wrote SFT transfer results and figure to {output_dir}")


if __name__ == "__main__":
    main()
