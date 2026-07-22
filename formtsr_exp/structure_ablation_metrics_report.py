from __future__ import annotations

import argparse
import csv
import hashlib
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any

from .document_similarity_report import (
    SchemaNode,
    build_schema_tree,
    extract_normalized_values,
    schema_nted,
    value_ned,
)
from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .metrics import (
    NA,
    MatchStats,
    _relation_signature,
    extract_relations,
    field_accuracy,
    load_optional_layout_gt,
    topology_score,
    unwrap_answer,
    widget_answer_accuracy,
)
from .page_em_report import classify_run_type


COMPONENTS = (
    "Schema-nTED",
    "Value-nED",
    "TSR-path",
    "R-F1@0.5",
    "R-F1@0.75",
    "LIG-F1",
    "GlobalGrid-F1",
    "WAcc",
    "Rel-F1",
)

VARIANTS: dict[str, tuple[str, ...]] = {
    "semantic_only": ("Schema-nTED", "Value-nED", "TSR-path"),
    "semantic_region": ("Schema-nTED", "Value-nED", "TSR-path", "R-F1@0.5"),
    "semantic_region_strict": ("Schema-nTED", "Value-nED", "TSR-path", "R-F1@0.75"),
    "global_grid_struct": (
        "Schema-nTED",
        "Value-nED",
        "TSR-path",
        "R-F1@0.5",
        "GlobalGrid-F1",
    ),
    "region_local_grid_struct": (
        "Schema-nTED",
        "Value-nED",
        "TSR-path",
        "R-F1@0.5",
        "LIG-F1",
    ),
    "region_local_grid_widget": (
        "Schema-nTED",
        "Value-nED",
        "TSR-path",
        "R-F1@0.5",
        "LIG-F1",
        "WAcc",
    ),
    "region_local_grid_relation": (
        "Schema-nTED",
        "Value-nED",
        "TSR-path",
        "R-F1@0.5",
        "LIG-F1",
        "Rel-F1",
    ),
    "full_structural": (
        "Schema-nTED",
        "Value-nED",
        "TSR-path",
        "R-F1@0.5",
        "LIG-F1",
        "WAcc",
        "Rel-F1",
    ),
}

COMPARISONS = (
    {
        "comparison": "region_effect_on_region_samples",
        "scope": "region_applicable",
        "with_variant": "semantic_region",
        "without_variant": "semantic_only",
    },
    {
        "comparison": "strict_iou_075_vs_05_on_region_samples",
        "scope": "region_applicable",
        "with_variant": "semantic_region_strict",
        "without_variant": "semantic_region",
    },
    {
        "comparison": "region_local_grid_vs_global_grid_on_grid_samples",
        "scope": "grid_applicable",
        "with_variant": "region_local_grid_struct",
        "without_variant": "global_grid_struct",
    },
    {
        "comparison": "lig_effect_on_lig_samples",
        "scope": "lig_applicable",
        "with_variant": "region_local_grid_struct",
        "without_variant": "semantic_region",
    },
    {
        "comparison": "widget_answer_effect_on_widget_samples",
        "scope": "widget_applicable",
        "with_variant": "region_local_grid_widget",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "relation_effect_on_relation_samples",
        "scope": "relation_applicable",
        "with_variant": "region_local_grid_relation",
        "without_variant": "region_local_grid_struct",
    },
    {
        "comparison": "full_structural_vs_semantic_on_structural_samples",
        "scope": "structural_applicable",
        "with_variant": "full_structural",
        "without_variant": "semantic_only",
    },
)

COMPONENT_COLUMNS = [
    "model",
    "model_id",
    "component",
    "n_total",
    "n_applicable",
    "applicability",
    "mean",
]

VARIANT_COLUMNS = [
    "model",
    "model_id",
    "variant",
    "components",
    "n_total",
    "score",
]

TARGETED_COLUMNS = [
    "model",
    "model_id",
    "n_total",
    "n_valid_json",
    "coverage",
    "comparison",
    "scope",
    "n_scope",
    "with_variant",
    "without_variant",
    "score_with",
    "score_without",
    "delta",
    "relative_delta_pct",
    "interpretation",
]

MACRO_COLUMNS = [
    "comparison",
    "scope",
    "n_models",
    "n_scope_per_model",
    "mean_score_with",
    "mean_score_without",
    "mean_delta",
    "delta_std",
    "min_delta",
    "max_delta",
    "mean_relative_delta_pct",
]

PER_SAMPLE_COLUMNS = [
    "model",
    "model_id",
    "sample_id",
    "template_name",
    "valid_json",
    "region_applicable",
    "grid_applicable",
    "lig_applicable",
    "widget_applicable",
    "relation_applicable",
    "structural_applicable",
    *COMPONENTS,
    *(f"score:{variant}" for variant in VARIANTS),
]


@dataclass(slots=True)
class AblationSample:
    sample_id: str
    template_name: str
    gt_answer: Any
    schema: SchemaNode
    values: tuple[str, ...]
    structural_gt: Any | None
    global_grid_applicable: bool
    widget_applicable: bool
    relation_applicable: bool


_WORKER_SAMPLES: list[AblationSample] = []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_int(value: Any, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {field}: {value!r}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format(value: Any, places: int = 6) -> str:
    if value is None or value == NA:
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_selected_runs(path: Path, n_indexed: int, requested: set[str] | None = None) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if requested:
        rows = [row for row in rows if row.get("model") in requested]
        missing = requested - {str(row.get("model")) for row in rows}
        if missing:
            raise ValueError(f"requested runs missing from selected main table: {sorted(missing)}")
    if not rows:
        raise ValueError(f"no selected runs in {path}")

    models: set[str] = set()
    model_ids: set[str] = set()
    for row in rows:
        model = str(row.get("model") or "")
        model_id = str(row.get("model_id") or "")
        if not model or not model_id:
            raise ValueError(f"selected run is missing model/model_id: {row}")
        if model in models or model_id.casefold() in model_ids:
            raise ValueError(f"duplicate selected model: {model}/{model_id}")
        if classify_run_type(model) != "raw":
            raise ValueError(f"non-raw run in formal ablation selection: {model}")
        if _as_int(row.get("n_total"), "n_total") != n_indexed:
            raise ValueError(f"run {model} is not full-scope")
        if _as_int(row.get("n_attempted"), "n_attempted") != n_indexed:
            raise ValueError(f"run {model} was not fully attempted")
        if _as_int(row.get("n_valid_json"), "n_valid_json") <= 0:
            raise ValueError(f"run {model} has no valid predictions")
        models.add(model)
        model_ids.add(model_id.casefold())
    return rows


def _load_samples(index_path: Path, layout_root: Path) -> list[AblationSample]:
    output: list[AblationSample] = []
    seen: set[str] = set()
    for source in read_jsonl(index_path):
        sample_id = str(source["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen.add(sample_id)
        template_name = str(source["template_name"])
        gt_answer = read_json(Path(str(source["label_path"])))
        layout_gt = load_optional_layout_gt(source, layout_root)
        structural_gt = (
            gt_answer
            if isinstance(gt_answer, dict)
            and any(key in gt_answer for key in ("regions", "local_grids", "cells", "widgets", "relations"))
            else layout_gt
        )
        global_grid, _ = topology_score({}, structural_gt)
        widget_score, _ = widget_answer_accuracy({}, gt_answer, structural_gt)
        relation_applicable = bool(extract_relations(structural_gt))
        output.append(
            AblationSample(
                sample_id=sample_id,
                template_name=template_name,
                gt_answer=gt_answer,
                schema=build_schema_tree(gt_answer),
                values=extract_normalized_values(gt_answer),
                structural_gt=structural_gt,
                global_grid_applicable=_is_number(global_grid),
                widget_applicable=_is_number(widget_score),
                relation_applicable=relation_applicable,
            )
        )
    return output


def _init_worker(index_path: str, layout_root: str) -> None:
    global _WORKER_SAMPLES
    _WORKER_SAMPLES = _load_samples(Path(index_path), Path(layout_root))


def _load_corrected_rows(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = {str(row.get("sample_id")): row for row in rows if row.get("sample_id")}
    if set(by_id) != sample_ids or len(rows) != len(sample_ids):
        raise ValueError(
            f"corrected structure sample mismatch for {path}: "
            f"missing={len(sample_ids - set(by_id))}, extra={len(set(by_id) - sample_ids)}"
        )
    return by_id


def _explicit_relation_f1(pred: Any, structural_gt: Any) -> float | str:
    gt_items = extract_relations(structural_gt)
    if not gt_items:
        return NA
    pred_items = extract_relations(pred)
    gt_relations = {_relation_signature(item) for item in gt_items}
    pred_relations = {_relation_signature(item) for item in pred_items}
    return MatchStats(
        len(gt_relations & pred_relations),
        len(pred_relations),
        len(gt_relations),
    ).f1


def score_variant(row: dict[str, Any], variant: str) -> float | str:
    values = [float(row[component]) for component in VARIANTS[variant] if _is_number(row.get(component))]
    return sum(values) / len(values) if values else NA


def _corrected_component(row: dict[str, Any], score_key: str, gt_key: str) -> tuple[float | str, bool]:
    gt_count = row.get(gt_key)
    applicable = _is_number(gt_count) and float(gt_count) > 0
    if not applicable:
        return NA, False
    score = row.get(score_key)
    if not _is_number(score):
        raise ValueError(f"applicable corrected metric {score_key} is non-numeric: {score!r}")
    return float(score), True


def evaluate_model(
    run: dict[str, str],
    pred_root: Path,
    corrected_dir: Path,
) -> list[dict[str, Any]]:
    if not _WORKER_SAMPLES:
        raise RuntimeError("ablation worker was not initialized")
    model = str(run["model"])
    model_id = str(run["model_id"])
    sample_ids = {sample.sample_id for sample in _WORKER_SAMPLES}
    corrected = _load_corrected_rows(corrected_dir / f"{model}.jsonl", sample_ids)
    output: list[dict[str, Any]] = []
    n_valid = 0

    for sample in _WORKER_SAMPLES:
        pred_path = pred_root / model / f"{sample.sample_id}.json"
        pred: Any = None
        valid_json = False
        if pred_path.exists():
            try:
                pred = read_json(pred_path)
                valid_json = True
                n_valid += 1
            except Exception:
                pred = None

        corrected_row = corrected[sample.sample_id]
        corrected_valid = bool(corrected_row.get("valid_json"))
        if corrected_valid != valid_json:
            raise ValueError(
                f"validity mismatch for {model}/{sample.sample_id}: "
                f"prediction={valid_json}, corrected={corrected_valid}"
            )

        region_05, region_applicable = _corrected_component(corrected_row, "R-F1", "R-gt")
        region_075, region_075_applicable = _corrected_component(corrected_row, "R-F1@0.75", "R-gt")
        if region_applicable != region_075_applicable:
            raise ValueError(f"region threshold applicability mismatch for {model}/{sample.sample_id}")
        lig, lig_applicable = _corrected_component(corrected_row, "LIG-F1", "LIG-gt")

        if valid_json:
            answer = unwrap_answer(pred)
            schema_score = schema_nted(build_schema_tree(answer), sample.schema)
            value_score = value_ned(extract_normalized_values(answer), sample.values)
            path_score, _ = field_accuracy(pred, sample.gt_answer)
            global_grid, _ = topology_score(pred, sample.structural_gt)
            widget_score, _ = widget_answer_accuracy(pred, sample.gt_answer, sample.structural_gt)
            relation_score = _explicit_relation_f1(pred, sample.structural_gt)
        else:
            schema_score = 0.0
            value_score = 0.0
            path_score = 0.0
            global_grid = 0.0 if sample.global_grid_applicable else NA
            widget_score = 0.0 if sample.widget_applicable else NA
            relation_score = 0.0 if sample.relation_applicable else NA

        if sample.global_grid_applicable and not _is_number(global_grid):
            raise ValueError(f"missing global-grid score for {model}/{sample.sample_id}")
        if sample.widget_applicable and not _is_number(widget_score):
            raise ValueError(f"missing widget score for {model}/{sample.sample_id}")
        if sample.relation_applicable and not _is_number(relation_score):
            raise ValueError(f"missing relation score for {model}/{sample.sample_id}")
        global_grid = float(global_grid) if sample.global_grid_applicable else NA
        widget_score = float(widget_score) if sample.widget_applicable else NA
        relation_score = float(relation_score) if sample.relation_applicable else NA

        structural_applicable = any(
            (
                region_applicable,
                sample.global_grid_applicable,
                lig_applicable,
                sample.widget_applicable,
                sample.relation_applicable,
            )
        )
        row: dict[str, Any] = {
            "model": model,
            "model_id": model_id,
            "sample_id": sample.sample_id,
            "template_name": sample.template_name,
            "valid_json": valid_json,
            "region_applicable": region_applicable,
            "grid_applicable": sample.global_grid_applicable or lig_applicable,
            "lig_applicable": lig_applicable,
            "widget_applicable": sample.widget_applicable,
            "relation_applicable": sample.relation_applicable,
            "structural_applicable": structural_applicable,
            "Schema-nTED": float(schema_score),
            "Value-nED": float(value_score),
            "TSR-path": float(path_score) if _is_number(path_score) else 0.0,
            "R-F1@0.5": region_05,
            "R-F1@0.75": region_075,
            "LIG-F1": lig,
            "GlobalGrid-F1": global_grid,
            "WAcc": widget_score,
            "Rel-F1": relation_score,
        }
        for variant in VARIANTS:
            row[f"score:{variant}"] = score_variant(row, variant)
        output.append(row)

    expected_valid = _as_int(run.get("n_valid_json"), "n_valid_json")
    if n_valid != expected_valid:
        raise ValueError(f"valid count mismatch for {model}: evaluated={n_valid}, selected={expected_valid}")
    return output


def summarize_components(rows: list[dict[str, Any]], run: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for component in COMPONENTS:
        values = [float(row[component]) for row in rows if _is_number(row.get(component))]
        output.append(
            {
                "model": run["model"],
                "model_id": run["model_id"],
                "component": component,
                "n_total": len(rows),
                "n_applicable": len(values),
                "applicability": len(values) / len(rows) if rows else NA,
                "mean": sum(values) / len(values) if values else NA,
            }
        )
    return output


def summarize_variants(rows: list[dict[str, Any]], run: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant, components in VARIANTS.items():
        values = [float(row[f"score:{variant}"]) for row in rows if _is_number(row.get(f"score:{variant}"))]
        output.append(
            {
                "model": run["model"],
                "model_id": run["model_id"],
                "variant": variant,
                "components": ",".join(components),
                "n_total": len(rows),
                "score": sum(values) / len(values) if values else NA,
            }
        )
    return output


def summarize_targeted(rows: list[dict[str, Any]], run: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    n_total = len(rows)
    n_valid = _as_int(run.get("n_valid_json"), "n_valid_json")
    for spec in COMPARISONS:
        scope = str(spec["scope"])
        scoped = [row for row in rows if bool(row.get(scope))]
        with_key = f"score:{spec['with_variant']}"
        without_key = f"score:{spec['without_variant']}"
        with_values = [float(row[with_key]) for row in scoped if _is_number(row.get(with_key))]
        without_values = [float(row[without_key]) for row in scoped if _is_number(row.get(with_key)) and _is_number(row.get(without_key))]
        if len(with_values) != len(scoped) or len(without_values) != len(scoped):
            raise ValueError(f"non-numeric variant score in scope {scope} for {run['model']}")
        score_with = sum(with_values) / len(scoped) if scoped else NA
        score_without = sum(without_values) / len(scoped) if scoped else NA
        if _is_number(score_with) and _is_number(score_without):
            delta: float | str = float(score_with) - float(score_without)
            relative = delta / float(score_without) * 100.0 if float(score_without) else NA
            interpretation = "stricter_lower_score" if delta < 0 else "higher_score_with_component" if delta > 0 else "no_change"
        else:
            delta = NA
            relative = NA
            interpretation = "insufficient_data"
        output.append(
            {
                "model": run["model"],
                "model_id": run["model_id"],
                "n_total": n_total,
                "n_valid_json": n_valid,
                "coverage": n_valid / n_total if n_total else NA,
                "comparison": spec["comparison"],
                "scope": scope,
                "n_scope": len(scoped),
                "with_variant": spec["with_variant"],
                "without_variant": spec["without_variant"],
                "score_with": score_with,
                "score_without": score_without,
                "delta": delta,
                "relative_delta_pct": relative,
                "interpretation": interpretation,
            }
        )
    return output


def summarize_macro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in COMPARISONS:
        selected = [row for row in rows if row["comparison"] == spec["comparison"]]
        deltas = [float(row["delta"]) for row in selected if _is_number(row.get("delta"))]
        with_scores = [float(row["score_with"]) for row in selected if _is_number(row.get("score_with"))]
        without_scores = [float(row["score_without"]) for row in selected if _is_number(row.get("score_without"))]
        relative = [float(row["relative_delta_pct"]) for row in selected if _is_number(row.get("relative_delta_pct"))]
        scope_counts = {int(row["n_scope"]) for row in selected}
        output.append(
            {
                "comparison": spec["comparison"],
                "scope": spec["scope"],
                "n_models": len(deltas),
                "n_scope_per_model": next(iter(scope_counts)) if len(scope_counts) == 1 else NA,
                "mean_score_with": sum(with_scores) / len(with_scores) if with_scores else NA,
                "mean_score_without": sum(without_scores) / len(without_scores) if without_scores else NA,
                "mean_delta": sum(deltas) / len(deltas) if deltas else NA,
                "delta_std": pstdev(deltas) if deltas else NA,
                "min_delta": min(deltas) if deltas else NA,
                "max_delta": max(deltas) if deltas else NA,
                "mean_relative_delta_pct": sum(relative) / len(relative) if relative else NA,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in columns})


def _latex_escape(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def _write_latex(path: Path, targeted: list[dict[str, Any]], runs: list[dict[str, str]]) -> None:
    by_key = {(row["model"], row["comparison"]): row for row in targeted}
    headers = [
        "Region",
        "IoU .75-.5",
        "Local-global",
        "LIG",
        "Widget",
        "Relation",
        "Full-semantic",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evaluation-component sensitivity on the nine fully attempted raw runs. Each entry is $\Delta=\mathrm{score}_{with}-\mathrm{score}_{without}$ on a fixed GT-defined scope; negative values expose errors hidden by the simpler score.}",
        r"\label{tab:formtsr_metric_ablation_latest}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Model & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for run in runs:
        values = []
        for spec in COMPARISONS:
            row = by_key[(run["model"], spec["comparison"])]
            values.append(_format(row["delta"], 4).replace("NA", "--"))
        lines.append(_latex_escape(run["model_id"]) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(
    path: Path,
    runs: list[dict[str, str]],
    targeted: list[dict[str, Any]],
    macro: list[dict[str, Any]],
) -> None:
    labels = {
        "region_effect_on_region_samples": "+ region @0.5",
        "strict_iou_075_vs_05_on_region_samples": "IoU 0.75 vs 0.5",
        "region_local_grid_vs_global_grid_on_grid_samples": "local vs global grid",
        "lig_effect_on_lig_samples": "+ LIG",
        "widget_answer_effect_on_widget_samples": "+ widget answer",
        "relation_effect_on_relation_samples": "+ explicit relation",
        "full_structural_vs_semantic_on_structural_samples": "full vs semantic",
    }
    lines = [
        "# Latest evaluation-component ablation",
        "",
        "This is a metric-sensitivity analysis over fixed predictions, not a model, prompt, or training-component ablation. Only the nine best fully attempted raw runs from the formal main table are included.",
        "",
        "The semantic baseline is the equal-weight mean of Schema-nTED, Value-nED, and TSR-path. Page-EM is excluded because it is nearly all zero. Spatial components use the corrected coordinate-normalized per-sample scores. GlobalGrid-F1, WAcc, and explicit-only Rel-F1 remain ablation diagnostics and are not main-table metrics.",
        "",
        "Missing or invalid predictions score zero whenever the GT component is applicable. Target scopes come only from GT/template metadata, so every model uses the same denominator.",
        "",
        "## Selected runs",
        "",
        "| Model | Run | Valid/7000 |",
        "| --- | --- | ---: |",
    ]
    for run in runs:
        lines.append(f"| {run['model_id']} | `{run['model']}` | {run['n_valid_json']}/7000 |")
    lines.extend(
        [
            "",
            "## Nine-model macro",
            "",
            "| Comparison | GT scope/model | Score with | Score without | Delta | Relative delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in macro:
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[str(row["comparison"])],
                    _format(row["n_scope_per_model"], 0),
                    _format(row["mean_score_with"]),
                    _format(row["mean_score_without"]),
                    _format(row["mean_delta"]),
                    _format(row["mean_relative_delta_pct"]) + "%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-model targeted deltas",
            "",
            "A negative delta means that the added or stricter component scores below the simpler configuration on the same pages. It must not be interpreted as a causal loss in model capability.",
            "",
            "| Model | Region | IoU .75-.5 | Local-global | LIG | Widget | Relation | Full-semantic |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_key = {(row["model"], row["comparison"]): row for row in targeted}
    for run in runs:
        values = [
            _format(by_key[(run["model"], spec["comparison"])]["delta"])
            for spec in COMPARISONS
        ]
        lines.append("| " + " | ".join([run["model_id"], *values]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the latest metric-component ablation report.")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--selected-main", default="outputs/main_exp/main_experiment_results.csv")
    parser.add_argument("--corrected-dir", default="outputs/main_exp/corrected_structure_per_sample")
    parser.add_argument("--layout-root", default="newdataset-layout")
    parser.add_argument("--models", default="", help="Optional comma-separated subset of selected formal runs.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="outputs/aux_exp/structure_ablation/report_latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    index_path = Path(args.index)
    pred_root = Path(args.pred_root)
    selected_path = Path(args.selected_main)
    corrected_dir = Path(args.corrected_dir)
    layout_root = Path(args.layout_root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    n_indexed = len(read_jsonl(index_path))
    requested = {item.strip() for item in args.models.split(",") if item.strip()} or None
    runs = load_selected_runs(selected_path, n_indexed, requested)
    by_model: dict[str, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(runs)),
        initializer=_init_worker,
        initargs=(str(index_path), str(layout_root)),
    ) as executor:
        futures = {
            executor.submit(evaluate_model, run, pred_root, corrected_dir): run
            for run in runs
        }
        for future in as_completed(futures):
            run = futures[future]
            rows = future.result()
            by_model[run["model"]] = rows
            print(f"[Ablation] {run['model']}: {sum(bool(row['valid_json']) for row in rows)}/{len(rows)} valid")

    per_sample = [row for run in runs for row in by_model[run["model"]]]
    components = [row for run in runs for row in summarize_components(by_model[run["model"]], run)]
    variants = [row for run in runs for row in summarize_variants(by_model[run["model"]], run)]
    targeted = [row for run in runs for row in summarize_targeted(by_model[run["model"]], run)]
    macro = summarize_macro(targeted)

    _write_csv(out_dir / "ablation_components.csv", components, COMPONENT_COLUMNS)
    _write_csv(out_dir / "ablation_variants.csv", variants, VARIANT_COLUMNS)
    _write_csv(out_dir / "ablation_targeted_deltas.csv", targeted, TARGETED_COLUMNS)
    _write_csv(out_dir / "ablation_targeted_macro.csv", macro, MACRO_COLUMNS)
    write_jsonl(
        out_dir / "ablation_per_sample.jsonl",
        [{column: row.get(column, NA) for column in PER_SAMPLE_COLUMNS} for row in per_sample],
    )
    _write_markdown(out_dir / "ablation_results.md", runs, targeted, macro)
    _write_latex(out_dir / "ablation_results_table.tex", targeted, runs)

    scope_counts = {
        spec["scope"]: sorted(
            {
                int(row["n_scope"])
                for row in targeted
                if row["scope"] == spec["scope"]
            }
        )
        for spec in COMPARISONS
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report_type": "evaluation_metric_component_sensitivity_not_model_ablation",
        "sources": {
            "index": str(index_path),
            "pred_root": str(pred_root),
            "selected_main": str(selected_path),
            "corrected_structure_per_sample": str(corrected_dir),
            "layout_root": str(layout_root),
        },
        "source_hashes": {
            "index_sha256": _sha256(index_path),
            "selected_main_sha256": _sha256(selected_path),
        },
        "n_indexed": n_indexed,
        "n_models": len(runs),
        "n_per_sample_rows": len(per_sample),
        "selected_runs": [
            {
                "model": run["model"],
                "model_id": run["model_id"],
                "n_total": _as_int(run["n_total"], "n_total"),
                "n_attempted": _as_int(run["n_attempted"], "n_attempted"),
                "n_valid_json": _as_int(run["n_valid_json"], "n_valid_json"),
            }
            for run in runs
        ],
        "selection_policy": "read the formal main_experiment_results.csv; require unique raw runs with n_total=n_attempted=n_indexed and n_valid_json>0",
        "semantic_baseline": list(VARIANTS["semantic_only"]),
        "page_em_policy": "excluded from equal-weight ablation variants because exact-page matches are too sparse",
        "spatial_policy": "R-F1@0.5, R-F1@0.75, and LIG-F1 come from corrected coordinate-normalized per-sample reports",
        "diagnostic_components": {
            "GlobalGrid-F1": "row/column/span signature F1; not a formal main-table metric",
            "WAcc": "template-metadata widget answer-path accuracy; not a formal main-table metric",
            "Rel-F1": "explicit relation edges only; no answer-tree fallback; not a formal main-table metric",
        },
        "missing_invalid_policy": "score zero whenever the GT component is applicable",
        "scope_policy": "all targeted scopes are determined from GT/template metadata and corrected GT counts, independent of prediction validity",
        "variant_score": "equal-weight arithmetic mean over components applicable to that sample",
        "delta_rule": "mean(score_with - score_without) expressed as the difference of paired-scope means; negative values mean the added/stricter component exposes lower scores",
        "variants": {name: list(components) for name, components in VARIANTS.items()},
        "comparisons": list(COMPARISONS),
        "scope_counts": scope_counts,
    }
    write_json(out_dir / "ablation_results_metadata.json", metadata)
    print(f"wrote latest ablation report -> {out_dir}")


if __name__ == "__main__":
    main()
