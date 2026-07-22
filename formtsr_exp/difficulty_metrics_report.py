from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .document_similarity_report import (
    SchemaNode,
    build_schema_tree,
    extract_normalized_values,
    schema_nted,
    value_ned,
)
from .io_utils import ensure_dir, read_json, read_jsonl, write_json
from .metrics import NA, normalize_json, unwrap_answer
from .results import (
    DEFAULT_DIFFICULTY_CSV,
    DIFFICULTY_LEVEL_LABELS,
    DIFFICULTY_LEVEL_ORDER,
    load_difficulty_lookup,
)


REPORT_METRICS = [
    "Page-EM",
    "Schema-nTED",
    "Value-nED",
    "TSR-path",
    "R-F1@0.5",
    "R-F1@0.75",
    "LIG-F1",
]

EXPECTED_TEMPLATE_COUNTS = {"L1": 11, "L2": 24, "L3": 24, "L4": 11}
EXPECTED_SAMPLE_COUNTS = {"L1": 1100, "L2": 2400, "L3": 2400, "L4": 1100}
EXPECTED_LIG_COUNTS = {"L1": 200, "L2": 600, "L3": 800, "L4": 800}
EXPECTED_DIFFICULTY_SHA256 = "4c446da7c43909d088cc9d5de71c4621dd388ff3b47739cb4bc5ea7318921a62"
EXPECTED_INDEX_SHA256 = "d7fcca1ea45453e8acf93476b6007cd175b8987cfae9bf91123ab1fae9247489"

RESULT_COLUMNS = [
    "model",
    "model_id",
    "difficulty_level",
    "difficulty_name",
    "n_templates",
    "n_total",
    "n_valid_json",
    "coverage",
    "n_exact_match",
    *REPORT_METRICS,
    "n_lig_applicable",
]

DIAGNOSTIC_COLUMNS = [
    "model",
    "model_id",
    "metric",
    "L1_easy",
    "L2_medium",
    "L3_hard",
    "L4_expert",
    "L1_to_L4_drop",
    "relative_drop_pct",
    "adjacent_drop_count",
]


@dataclass(frozen=True, slots=True)
class SelectedRun:
    model: str
    model_id: str
    official: dict[str, str]


@dataclass(frozen=True, slots=True)
class DifficultySample:
    sample_id: str
    template_name: str
    difficulty_level: str
    normalized_gt: Any
    schema: SchemaNode
    values: tuple[str, ...]


_WORKER_SAMPLES: list[DifficultySample] = []
_WORKER_PRED_ROOT = Path()
_WORKER_SEMANTIC_DIR = Path()
_WORKER_STRUCTURE_DIR = Path()


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip() not in {"", NA, "TBD"}:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format(value: Any, places: int = 6) -> str:
    if value is None or value == NA:
        return NA
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def load_selected_runs(path: Path) -> list[SelectedRun]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no selected runs in {path}")

    seen_models: set[str] = set()
    seen_model_ids: set[str] = set()
    selected: list[SelectedRun] = []
    for row in rows:
        model = str(row.get("model") or "").strip()
        model_id = str(row.get("model_id") or "").strip()
        if not model or not model_id:
            raise ValueError(f"missing model/model_id in {path}")
        if model in seen_models or model_id.casefold() in seen_model_ids:
            raise ValueError(f"duplicate selected model in {path}: {model}/{model_id}")
        if int(str(row.get("n_total") or 0)) != int(str(row.get("n_attempted") or -1)):
            raise ValueError(f"selected run was not fully attempted: {model}")
        seen_models.add(model)
        seen_model_ids.add(model_id.casefold())
        selected.append(SelectedRun(model=model, model_id=model_id, official=row))
    return selected


def load_difficulty_samples(
    index_path: Path,
    difficulty_csv: Path,
) -> tuple[list[DifficultySample], dict[str, int], dict[str, int]]:
    index_rows = read_jsonl(index_path)
    index_templates = {str(row["template_name"]) for row in index_rows}

    csv_templates: set[str] = set()
    with difficulty_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            template_value = row.get("file") or row.get("template") or row.get("template_name")
            if not template_value:
                raise ValueError(f"difficulty row has no template identifier in {difficulty_csv}")
            template_name = Path(template_value).stem
            if template_name in csv_templates:
                raise ValueError(f"duplicate template in difficulty mapping: {template_name}")
            level = str(row.get("normal_calibrated_level") or "").strip().upper()
            if level not in DIFFICULTY_LEVEL_ORDER:
                raise ValueError(
                    f"frozen difficulty row must have an explicit L1-L4 "
                    f"normal_calibrated_level: {template_name}={level!r}"
                )
            csv_templates.add(template_name)
    if csv_templates != index_templates:
        raise ValueError(
            "frozen difficulty mapping must match the dataset index exactly: "
            f"missing={sorted(index_templates - csv_templates)}, "
            f"extra={sorted(csv_templates - index_templates)}"
        )

    difficulty_lookup = load_difficulty_lookup(
        index_rows,
        difficulty_csv=difficulty_csv,
        layout_root=None,
    )
    if set(difficulty_lookup) != index_templates:
        raise ValueError("not every frozen difficulty row has a valid L1-L4 level")
    samples: list[DifficultySample] = []
    seen: set[str] = set()
    template_sets: dict[str, set[str]] = defaultdict(set)
    sample_counts: dict[str, int] = defaultdict(int)

    for row in index_rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in index: {sample_id}")
        seen.add(sample_id)
        template_name = str(row["template_name"])
        difficulty = difficulty_lookup.get(template_name)
        if not difficulty:
            raise ValueError(f"missing difficulty mapping for template: {template_name}")
        level = str(difficulty["difficulty_level"])
        if level not in DIFFICULTY_LEVEL_ORDER:
            raise ValueError(f"unsupported difficulty level for {template_name}: {level}")
        gt = read_json(Path(str(row["label_path"])))
        samples.append(
            DifficultySample(
                sample_id=sample_id,
                template_name=template_name,
                difficulty_level=level,
                normalized_gt=normalize_json(gt),
                schema=build_schema_tree(gt),
                values=extract_normalized_values(gt),
            )
        )
        template_sets[level].add(template_name)
        sample_counts[level] += 1

    template_counts = {level: len(template_sets[level]) for level in DIFFICULTY_LEVEL_ORDER}
    level_counts = {level: sample_counts[level] for level in DIFFICULTY_LEVEL_ORDER}
    if any(not level_counts[level] for level in DIFFICULTY_LEVEL_ORDER):
        raise ValueError(f"difficulty mapping does not cover all L1-L4 levels: {level_counts}")
    return samples, template_counts, level_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(index_path: Path, difficulty_csv: Path) -> None:
    index_hash = _sha256(index_path)
    mapping_hash = _sha256(difficulty_csv)
    if index_hash != EXPECTED_INDEX_SHA256:
        raise ValueError(
            f"dataset index changed: expected SHA-256 {EXPECTED_INDEX_SHA256}, got {index_hash}"
        )
    if mapping_hash != EXPECTED_DIFFICULTY_SHA256:
        raise ValueError(
            "difficulty mapping changed: expected SHA-256 "
            f"{EXPECTED_DIFFICULTY_SHA256}, got {mapping_hash}"
        )


def validate_benchmark_partition(
    template_counts: dict[str, int],
    sample_counts: dict[str, int],
) -> None:
    if template_counts != EXPECTED_TEMPLATE_COUNTS:
        raise ValueError(
            f"frozen difficulty template counts changed: "
            f"expected={EXPECTED_TEMPLATE_COUNTS}, actual={template_counts}"
        )
    if sample_counts != EXPECTED_SAMPLE_COUNTS:
        raise ValueError(
            f"frozen difficulty sample counts changed: "
            f"expected={EXPECTED_SAMPLE_COUNTS}, actual={sample_counts}"
        )


def _load_per_sample_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"missing sample_id in {path}")
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        by_id[sample_id] = row
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"sample coverage mismatch for {path}: "
            f"missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)}"
        )
    return by_id


def _new_accumulator() -> dict[str, Any]:
    return {
        "templates": set(),
        "n_total": 0,
        "n_valid_json": 0,
        "n_exact_match": 0,
        "Schema-nTED": 0.0,
        "Value-nED": 0.0,
        "TSR-path": 0.0,
        "R-F1@0.5": 0.0,
        "R-F1@0.75": 0.0,
        "LIG-F1": 0.0,
        "n_lig_applicable": 0,
    }


def evaluate_run_by_difficulty(
    run: SelectedRun,
    samples: list[DifficultySample],
    pred_root: Path,
    semantic_metrics_dir: Path,
    structure_metrics_dir: Path,
) -> list[dict[str, Any]]:
    expected_ids = {sample.sample_id for sample in samples}
    semantic_rows = _load_per_sample_rows(
        semantic_metrics_dir / f"{run.model}.jsonl",
        expected_ids,
    )
    structure_rows = _load_per_sample_rows(
        structure_metrics_dir / f"{run.model}.jsonl",
        expected_ids,
    )
    model_pred_dir = pred_root / run.model
    if not model_pred_dir.is_dir():
        raise FileNotFoundError(model_pred_dir)

    accumulators = {level: _new_accumulator() for level in DIFFICULTY_LEVEL_ORDER}
    for sample in samples:
        accumulator = accumulators[sample.difficulty_level]
        accumulator["templates"].add(sample.template_name)
        accumulator["n_total"] += 1

        semantic = semantic_rows[sample.sample_id]
        structure = structure_rows[sample.sample_id]
        pred_path = model_pred_dir / f"{sample.sample_id}.json"
        pred: Any | None = None
        valid_json = False
        if pred_path.exists():
            try:
                pred = read_json(pred_path)
                valid_json = True
            except Exception:
                pred = None
        if bool(semantic.get("valid_json")) != valid_json:
            raise ValueError(f"semantic valid_json mismatch for {run.model}/{sample.sample_id}")
        if bool(structure.get("valid_json")) != valid_json:
            raise ValueError(f"structure valid_json mismatch for {run.model}/{sample.sample_id}")

        if valid_json:
            accumulator["n_valid_json"] += 1
            answer = unwrap_answer(pred)
            if normalize_json(answer) == sample.normalized_gt:
                accumulator["n_exact_match"] += 1
            accumulator["Schema-nTED"] += schema_nted(build_schema_tree(answer), sample.schema)
            accumulator["Value-nED"] += value_ned(extract_normalized_values(answer), sample.values)

        accumulator["TSR-path"] += _numeric(semantic.get("TSR-path")) or 0.0
        accumulator["R-F1@0.5"] += _numeric(structure.get("R-F1")) or 0.0
        accumulator["R-F1@0.75"] += _numeric(structure.get("R-F1@0.75")) or 0.0
        lig_f1 = _numeric(structure.get("LIG-F1"))
        if lig_f1 is not None:
            accumulator["n_lig_applicable"] += 1
            accumulator["LIG-F1"] += lig_f1

    rows: list[dict[str, Any]] = []
    for level in DIFFICULTY_LEVEL_ORDER:
        accumulator = accumulators[level]
        n_total = int(accumulator["n_total"])
        n_lig_applicable = int(accumulator["n_lig_applicable"])
        rows.append(
            {
                "model": run.model,
                "model_id": run.model_id,
                "difficulty_level": level,
                "difficulty_name": DIFFICULTY_LEVEL_LABELS[level],
                "n_templates": len(accumulator["templates"]),
                "n_total": n_total,
                "n_valid_json": accumulator["n_valid_json"],
                "coverage": accumulator["n_valid_json"] / n_total,
                "n_exact_match": accumulator["n_exact_match"],
                "Page-EM": accumulator["n_exact_match"] / n_total,
                "Schema-nTED": accumulator["Schema-nTED"] / n_total,
                "Value-nED": accumulator["Value-nED"] / n_total,
                "TSR-path": accumulator["TSR-path"] / n_total,
                "R-F1@0.5": accumulator["R-F1@0.5"] / n_total,
                "R-F1@0.75": accumulator["R-F1@0.75"] / n_total,
                "LIG-F1": accumulator["LIG-F1"] / n_lig_applicable
                if n_lig_applicable
                else NA,
                "n_lig_applicable": n_lig_applicable,
            }
        )
    return rows


def _init_worker(
    index_path: str,
    difficulty_csv: str,
    pred_root: str,
    semantic_metrics_dir: str,
    structure_metrics_dir: str,
) -> None:
    global _WORKER_SAMPLES
    global _WORKER_PRED_ROOT
    global _WORKER_SEMANTIC_DIR
    global _WORKER_STRUCTURE_DIR
    _WORKER_SAMPLES, _, _ = load_difficulty_samples(Path(index_path), Path(difficulty_csv))
    _WORKER_PRED_ROOT = Path(pred_root)
    _WORKER_SEMANTIC_DIR = Path(semantic_metrics_dir)
    _WORKER_STRUCTURE_DIR = Path(structure_metrics_dir)


def _evaluate_run_worker(run: SelectedRun) -> list[dict[str, Any]]:
    if not _WORKER_SAMPLES:
        raise RuntimeError("difficulty metric worker was not initialized")
    return evaluate_run_by_difficulty(
        run,
        _WORKER_SAMPLES,
        _WORKER_PRED_ROOT,
        _WORKER_SEMANTIC_DIR,
        _WORKER_STRUCTURE_DIR,
    )


def build_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    diagnostics: list[dict[str, Any]] = []
    level_columns = {
        "L1": "L1_easy",
        "L2": "L2_medium",
        "L3": "L3_hard",
        "L4": "L4_expert",
    }
    for model, model_rows in by_model.items():
        by_level = {str(row["difficulty_level"]): row for row in model_rows}
        for metric in REPORT_METRICS:
            values = {level: _numeric(by_level[level].get(metric)) for level in DIFFICULTY_LEVEL_ORDER}
            l1 = values["L1"]
            l4 = values["L4"]
            drop = (l1 - l4) if l1 is not None and l4 is not None else None
            relative_drop = (drop / l1 * 100.0) if drop is not None and l1 else None
            adjacent_drop_count = sum(
                values[left] is not None
                and values[right] is not None
                and values[left] >= values[right]
                for left, right in zip(DIFFICULTY_LEVEL_ORDER, DIFFICULTY_LEVEL_ORDER[1:])
            )
            diagnostic: dict[str, Any] = {
                "model": model,
                "model_id": model_rows[0]["model_id"],
                "metric": metric,
                "L1_to_L4_drop": drop if drop is not None else NA,
                "relative_drop_pct": relative_drop if relative_drop is not None else NA,
                "adjacent_drop_count": adjacent_drop_count,
            }
            diagnostic.update(
                {
                    level_columns[level]: values[level] if values[level] is not None else NA
                    for level in DIFFICULTY_LEVEL_ORDER
                }
            )
            diagnostics.append(diagnostic)
    return diagnostics


def _official_metric(row: dict[str, str], metric: str) -> float | None:
    if metric == "R-F1@0.5":
        return _numeric(row.get("R-F1@0.5") or row.get("R-F1"))
    return _numeric(row.get(metric))


def validate_against_official(run: SelectedRun, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = sum(int(row["n_total"]) for row in rows)
    n_valid_json = sum(int(row["n_valid_json"]) for row in rows)
    n_exact_match = sum(int(row["n_exact_match"]) for row in rows)
    if n_total != int(run.official["n_total"]):
        raise ValueError(f"n_total does not reconstruct official result for {run.model}")
    if n_valid_json != int(run.official["n_valid_json"]):
        raise ValueError(f"n_valid_json does not reconstruct official result for {run.model}")
    if n_exact_match != int(run.official["n_exact_match"]):
        raise ValueError(f"n_exact_match does not reconstruct official result for {run.model}")
    lig_counts = {
        str(row["difficulty_level"]): int(row["n_lig_applicable"])
        for row in rows
    }
    if lig_counts != EXPECTED_LIG_COUNTS:
        raise ValueError(
            f"LIG applicability partition changed for {run.model}: "
            f"expected={EXPECTED_LIG_COUNTS}, actual={lig_counts}"
        )
    if sum(lig_counts.values()) != int(run.official["n_lig_applicable"]):
        raise ValueError(f"n_lig_applicable does not reconstruct official result for {run.model}")

    differences: dict[str, float] = {}
    for metric in REPORT_METRICS:
        if metric == "LIG-F1":
            denominator = sum(int(row["n_lig_applicable"]) for row in rows)
            reconstructed = (
                sum(float(row[metric]) * int(row["n_lig_applicable"]) for row in rows) / denominator
                if denominator
                else None
            )
        else:
            reconstructed = sum(float(row[metric]) * int(row["n_total"]) for row in rows) / n_total
        official = _official_metric(run.official, metric)
        if official is None or reconstructed is None:
            continue
        difference = abs(reconstructed - official)
        differences[metric] = difference
        tolerance = 5.1e-5 if metric == "TSR-path" else 5.1e-7
        if difference > tolerance:
            raise ValueError(
                f"{metric} does not reconstruct official result for {run.model}: "
                f"difficulty={reconstructed:.9f}, official={official:.9f}"
            )
    return {
        "model": run.model,
        "n_total": n_total,
        "n_valid_json": n_valid_json,
        "n_exact_match": n_exact_match,
        "max_abs_metric_difference": max(differences.values(), default=0.0),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in columns})


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_results(
    out_dir: Path,
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    _write_csv(out_dir / "difficulty_results.csv", rows, RESULT_COLUMNS)
    _write_csv(
        out_dir / "difficulty_diagnostic_summary.csv",
        diagnostics,
        DIAGNOSTIC_COLUMNS,
    )
    write_json(out_dir / "difficulty_results_metadata.json", metadata)

    markdown = [
        "# Difficulty-stratified results",
        "",
        "Only the best fully attempted raw run per model is included. Missing or invalid predictions score zero in every full-scope metric. LIG-F1 uses only GT-applicable pages and reports its denominator separately.",
        "",
        "| Model | Level | Valid/Total | Exact pages | Schema-nTED | Value-nED | TSR-path | R-F1@0.5 | R-F1@0.75 | LIG-F1 (N) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lig = (
            f"{_format(row['LIG-F1'])} ({row['n_lig_applicable']})"
            if row["LIG-F1"] != NA
            else "NA (0)"
        )
        markdown.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["difficulty_level"]),
                    f"{row['n_valid_json']}/{row['n_total']}",
                    f"{row['n_exact_match']}/{row['n_total']}",
                    _format(row["Schema-nTED"]),
                    _format(row["Value-nED"]),
                    _format(row["TSR-path"]),
                    _format(row["R-F1@0.5"]),
                    _format(row["R-F1@0.75"]),
                    lig,
                ]
            )
            + " |"
        )
    (out_dir / "difficulty_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Level & Valid & Exact & Schema-nTED & Value-nED & TSR-path & R-F1@0.5 & R-F1@0.75 & LIG-F1 \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            " & ".join(
                [
                    _latex_escape(str(row["model_id"])),
                    str(row["difficulty_level"]),
                    _latex_escape(f"{row['n_valid_json']}/{row['n_total']}"),
                    str(row["n_exact_match"]),
                    _format(row["Schema-nTED"], 4),
                    _format(row["Value-nED"], 4),
                    _format(row["TSR-path"], 4),
                    _format(row["R-F1@0.5"], 4),
                    _format(row["R-F1@0.75"], 4),
                    _format(row["LIG-F1"], 4),
                ]
            )
            + r" \\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "difficulty_results_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    diagnostic_latex = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Metric & L1 & L2 & L3 & L4 & L1--L4 drop \\",
        r"\midrule",
    ]
    for row in diagnostics:
        diagnostic_latex.append(
            " & ".join(
                [
                    _latex_escape(str(row["model_id"])),
                    _latex_escape(str(row["metric"])),
                    _format(row["L1_easy"], 4),
                    _format(row["L2_medium"], 4),
                    _format(row["L3_hard"], 4),
                    _format(row["L4_expert"], 4),
                    _format(row["L1_to_L4_drop"], 4),
                ]
            )
            + r" \\"
        )
    diagnostic_latex.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "difficulty_diagnostic_summary_table.tex").write_text(
        "\n".join(diagnostic_latex) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean L1-L4 slices with the latest formal reporting metrics."
    )
    parser.add_argument("--main-results", default="outputs/main_exp/main_experiment_results.csv")
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--semantic-metrics-dir", default="outputs/main_exp/per_model_metrics")
    parser.add_argument(
        "--structure-metrics-dir",
        default="outputs/main_exp/corrected_structure_per_sample",
    )
    parser.add_argument("--difficulty-csv", default=str(DEFAULT_DIFFICULTY_CSV))
    parser.add_argument("--out", default="outputs/aux_exp/difficulty")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_results = Path(args.main_results)
    index_path = Path(args.index)
    pred_root = Path(args.pred_root)
    semantic_metrics_dir = Path(args.semantic_metrics_dir)
    structure_metrics_dir = Path(args.structure_metrics_dir)
    difficulty_csv = Path(args.difficulty_csv)
    out_dir = Path(args.out)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    selected_runs = load_selected_runs(main_results)
    validate_frozen_sources(index_path, difficulty_csv)
    samples, template_counts, level_counts = load_difficulty_samples(
        index_path,
        difficulty_csv,
    )
    validate_benchmark_partition(template_counts, level_counts)
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    if args.workers == 1:
        for run in selected_runs:
            rows_by_model[run.model] = evaluate_run_by_difficulty(
                run,
                samples,
                pred_root,
                semantic_metrics_dir,
                structure_metrics_dir,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(
                str(index_path),
                str(difficulty_csv),
                str(pred_root),
                str(semantic_metrics_dir),
                str(structure_metrics_dir),
            ),
        ) as executor:
            futures = {executor.submit(_evaluate_run_worker, run): run for run in selected_runs}
            for future in as_completed(futures):
                run = futures[future]
                rows_by_model[run.model] = future.result()
                print(f"[Difficulty] computed {run.model}")

    for run in selected_runs:
        run_rows = rows_by_model[run.model]
        validations.append(validate_against_official(run, run_rows))
        rows.extend(run_rows)
        print(f"[Difficulty] validated {run.model} against the official main result")

    diagnostics = build_diagnostics(rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "main_results": str(main_results),
            "index": str(index_path),
            "pred_root": str(pred_root),
            "semantic_metrics_dir": str(semantic_metrics_dir),
            "structure_metrics_dir": str(structure_metrics_dir),
            "difficulty_csv": str(difficulty_csv),
            "difficulty_csv_sha256": _sha256(difficulty_csv),
            "index_sha256": _sha256(index_path),
        },
        "selection": "the best fully attempted raw run per model from main_experiment_results.csv",
        "difficulty_mapping_policy": "frozen CSV mapping; its template set must exactly equal the dataset index, with no dynamic recalibration or layout fallback",
        "n_models": len(selected_runs),
        "n_result_rows": len(rows),
        "difficulty_level_order": DIFFICULTY_LEVEL_ORDER,
        "difficulty_level_labels": DIFFICULTY_LEVEL_LABELS,
        "template_counts_by_level": template_counts,
        "sample_counts_by_level": level_counts,
        "metric_policy": {
            "full_scope": "Page-EM, Schema-nTED, Value-nED, TSR-path, R-F1@0.5, and R-F1@0.75 use every page in the difficulty slice; missing/invalid predictions are zero",
            "LIG-F1": "page-macro F1 over GT-applicable pages only; n_lig_applicable is reported for each level",
            "R-F1@0.5": "corrected canonical-type, coordinate-normalized page-macro region F1",
            "R-F1@0.75": "the same corrected region metric at the stricter IoU threshold",
            "L1_to_L4_drop": "metric(L1) - metric(L4); positive means lower performance on expert pages",
        },
        "main_result_reconstruction": validations,
    }
    write_results(out_dir, rows, diagnostics, metadata)
    print(f"wrote clean difficulty results -> {out_dir / 'difficulty_results.csv'}")


if __name__ == "__main__":
    main()
