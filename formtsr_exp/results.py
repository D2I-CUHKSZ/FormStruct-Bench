from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json
from .metrics import NA, mean_numeric


METRIC_COLUMNS = ["TSR-path", "VAcc", "R-F1", "LIG-F1", "WAcc", "CDS"]

RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "n_total",
    "n_valid_json",
    "invalid_rate",
    "TSR-path",
    "VAcc",
    "R-F1",
    "LIG-F1",
    "WAcc",
    "CDS",
]

DIFFICULTY_LEVEL_ORDER = ["L1", "L2", "L3", "L4"]
DIFFICULTY_LEVEL_LABELS = {
    "L1": "easy",
    "L2": "medium",
    "L3": "hard",
    "L4": "expert",
}
DEFAULT_DIFFICULTY_CSV = Path("outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv")

DIFFICULTY_RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "difficulty_level",
    "difficulty_name",
    "n_total",
    "n_templates",
    "n_valid_json",
    "invalid_rate",
    *METRIC_COLUMNS,
]

DIFFICULTY_DIAGNOSTIC_COLUMNS = [
    "model",
    "model_id",
    "group",
    "metric",
    "L1_easy",
    "L2_medium",
    "L3_hard",
    "L4_expert",
    "L1_to_L4_drop",
    "relative_drop_pct",
    "adjacent_drop_count",
    "covered_levels",
]


def format_value(value: Any) -> str:
    if value == NA or value is None:
        return NA
    if value == "TBD":
        return "TBD"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize_models(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model, model_rows in sorted(by_model.items()):
        seen.add(model)
        n_total = len(model_rows)
        n_valid = sum(1 for row in model_rows if row.get("valid_json"))
        group = str(model_rows[0].get("group", ""))
        model_id = str(model_rows[0].get("model_id") or model)
        summary = {
            "model": model,
            "model_id": model_id,
            "group": group,
            "n_total": n_total,
            "n_valid_json": n_valid,
            "invalid_rate": (1 - n_valid / n_total) if n_total else NA,
            "TSR-path": mean_numeric(model_rows, "TSR-path"),
            "VAcc": mean_numeric(model_rows, "VAcc"),
            "R-F1": mean_numeric(model_rows, "R-F1"),
            "LIG-F1": mean_numeric(model_rows, "LIG-F1"),
            "WAcc": mean_numeric(model_rows, "WAcc"),
            "CDS": mean_numeric(model_rows, "CDS"),
        }
        summaries.append(summary)

    for cfg in model_configs or []:
        name = str(cfg.get("name", ""))
        if name and name not in seen:
            summaries.append(
                {
                    "model": name,
                    "model_id": cfg.get("model", name),
                    "group": cfg.get("group", cfg.get("provider", "")),
                    "n_total": 0,
                    "n_valid_json": 0,
                    "invalid_rate": "TBD",
                    "TSR-path": "TBD",
                    "VAcc": "TBD",
                    "R-F1": "TBD" if cfg.get("provider") != "traditional_tsr" else NA,
                    "LIG-F1": "TBD" if cfg.get("provider") != "traditional_tsr" else NA,
                    "WAcc": "TBD" if cfg.get("provider") != "traditional_tsr" else NA,
                    "CDS": "TBD" if cfg.get("provider") != "traditional_tsr" else NA,
                }
            )
    return summaries


def write_main_results(out_dir: Path, summaries: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    csv_path = out_dir / "main_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: format_value(row.get(key)) for key in RESULT_COLUMNS})

    tex_path = out_dir / "main_results_table.tex"
    lines = [
        "\\begin{tabular}{lllrrrrrrrrr}",
        "\\hline",
        "Run & Model ID & Group & N & Valid & Invalid & TSR-path & VAcc & R-F1 & LIG-F1 & WAcc & CDS \\\\",
        "\\hline",
    ]
    for row in summaries:
        lines.append(
            " & ".join(
                [
                    format_value(row.get("model")),
                    format_value(row.get("model_id")),
                    format_value(row.get("group")),
                    format_value(row.get("n_total")),
                    format_value(row.get("n_valid_json")),
                    format_value(row.get("invalid_rate")),
                    format_value(row.get("TSR-path")),
                    format_value(row.get("VAcc")),
                    format_value(row.get("R-F1")),
                    format_value(row.get("LIG-F1")),
                    format_value(row.get("WAcc")),
                    format_value(row.get("CDS")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "main_results_metadata.json", metadata)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _normalize_difficulty_level(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    if not text.startswith("L"):
        return None
    try:
        index = int(text[1:])
    except ValueError:
        return None
    if index <= 1:
        return "L1"
    if index >= 4:
        return "L4"
    return f"L{index}"


def _difficulty_info_from_csv_row(row: dict[str, str], source_path: Path) -> tuple[str, dict[str, Any]] | None:
    template_value = row.get("file") or row.get("template") or row.get("template_name")
    if not template_value:
        return None
    template_name = Path(template_value).stem
    level = _normalize_difficulty_level(
        row.get("normal_calibrated_level")
        or row.get("difficulty_level")
        or row.get("level")
        or row.get("original_level")
    )
    if level is None:
        return None
    info: dict[str, Any] = {
        "difficulty_level": level,
        "difficulty_name": DIFFICULTY_LEVEL_LABELS[level],
        "difficulty_source": str(source_path),
    }
    for key in ("D_main", "S_form", "C_context", "original_level", "normal_calibrated_level"):
        value = row.get(key)
        numeric = _as_float(value)
        info[key] = numeric if numeric is not None else value
    return template_name, info


def _difficulty_info_from_layout(path: Path) -> dict[str, Any] | None:
    try:
        data = read_json(path)
    except Exception:
        return None
    metadata = data.get("metadata") if isinstance(data, dict) else None
    candidates: list[dict[str, Any]] = []
    if isinstance(metadata, dict):
        for key in ("difficulty_main_v0_4", "difficulty"):
            value = metadata.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    if isinstance(data, dict):
        value = data.get("difficulty")
        if isinstance(value, dict):
            candidates.append(value)

    for item in candidates:
        level = _normalize_difficulty_level(
            item.get("difficulty_level")
            or item.get("normal_calibrated_level")
            or item.get("absolute_threshold_level")
        )
        if level is None:
            continue
        info: dict[str, Any] = {
            "difficulty_level": level,
            "difficulty_name": DIFFICULTY_LEVEL_LABELS[level],
            "difficulty_source": str(path),
        }
        for key in ("D_main", "difficulty_score"):
            value = item.get(key)
            numeric = _as_float(value)
            if numeric is not None:
                info[key] = numeric
        return info
    return None


def _template_names(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("template_name")) for row in rows if row.get("template_name")}


def load_difficulty_lookup(
    rows: list[dict[str, Any]],
    *,
    difficulty_csv: Path | None = None,
    layout_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    csv_path = difficulty_csv if difficulty_csv is not None else DEFAULT_DIFFICULTY_CSV
    if csv_path and csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                parsed = _difficulty_info_from_csv_row(row, csv_path)
                if parsed is None:
                    continue
                template_name, info = parsed
                lookup[template_name] = info

    if layout_root:
        for template_name in sorted(_template_names(rows)):
            if template_name in lookup:
                continue
            path = layout_root / f"{template_name}.json"
            if not path.exists():
                continue
            info = _difficulty_info_from_layout(path)
            if info is not None:
                lookup[template_name] = info
    return lookup


def annotate_rows_with_difficulty(
    rows: list[dict[str, Any]],
    difficulty_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    annotated: list[dict[str, Any]] = []
    missing_templates: set[str] = set()
    for row in rows:
        template_name = str(row.get("template_name") or "")
        info = difficulty_lookup.get(template_name)
        if not info:
            if template_name:
                missing_templates.add(template_name)
            continue
        copied = dict(row)
        copied.update(
            {
                "difficulty_level": info["difficulty_level"],
                "difficulty_name": info["difficulty_name"],
            }
        )
        for key in ("D_main", "S_form", "C_context", "difficulty_score"):
            if key in info:
                copied[f"difficulty_{key}"] = info[key]
        annotated.append(copied)
    return annotated, missing_templates


def summarize_models_by_difficulty(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    config_by_name = {
        str(cfg.get("name")): cfg
        for cfg in (model_configs or [])
        if isinstance(cfg, dict) and cfg.get("name")
    }
    model_names = sorted(set(by_model) | set(config_by_name))

    summaries: list[dict[str, Any]] = []
    for model in model_names:
        model_rows = by_model.get(model, [])
        cfg = config_by_name.get(model, {})
        if model_rows:
            group = str(model_rows[0].get("group", cfg.get("group", "")))
            model_id = str(model_rows[0].get("model_id") or cfg.get("model") or model)
        else:
            group = str(cfg.get("group", cfg.get("provider", "")))
            model_id = str(cfg.get("model", model))

        rows_by_level = {
            level: [row for row in model_rows if row.get("difficulty_level") == level]
            for level in DIFFICULTY_LEVEL_ORDER
        }
        for level in DIFFICULTY_LEVEL_ORDER:
            level_rows = rows_by_level[level]
            n_total = len(level_rows)
            n_valid = sum(1 for row in level_rows if row.get("valid_json"))
            row: dict[str, Any] = {
                "model": model,
                "model_id": model_id,
                "group": group,
                "difficulty_level": level,
                "difficulty_name": DIFFICULTY_LEVEL_LABELS[level],
                "n_total": n_total,
                "n_templates": len({str(item.get("template_name")) for item in level_rows if item.get("template_name")}),
                "n_valid_json": n_valid,
                "invalid_rate": (1 - n_valid / n_total) if n_total else (NA if model_rows else "TBD"),
            }
            for metric in METRIC_COLUMNS:
                row[metric] = mean_numeric(level_rows, metric) if n_total else (NA if model_rows else "TBD")
            summaries.append(row)
    return summaries


def summarize_difficulty_diagnostics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in summaries:
        by_model.setdefault(str(row["model"]), []).append(row)

    diagnostics: list[dict[str, Any]] = []
    for model, rows in sorted(by_model.items()):
        if not rows:
            continue
        rows_by_level = {str(row.get("difficulty_level")): row for row in rows}
        model_id = str(rows[0].get("model_id") or model)
        group = str(rows[0].get("group") or "")
        for metric in METRIC_COLUMNS:
            values: dict[str, float | str] = {}
            numeric_levels: list[str] = []
            for level in DIFFICULTY_LEVEL_ORDER:
                value = rows_by_level.get(level, {}).get(metric, NA)
                values[level] = value
                if isinstance(value, (int, float)):
                    numeric_levels.append(level)

            l1 = values.get("L1")
            l4 = values.get("L4")
            drop: float | str = NA
            relative_drop: float | str = NA
            if isinstance(l1, (int, float)) and isinstance(l4, (int, float)):
                drop = float(l1) - float(l4)
                relative_drop = (drop / float(l1) * 100.0) if float(l1) else NA

            adjacent_drop_count = 0
            for left, right in zip(DIFFICULTY_LEVEL_ORDER, DIFFICULTY_LEVEL_ORDER[1:]):
                left_value = values.get(left)
                right_value = values.get(right)
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)) and left_value >= right_value:
                    adjacent_drop_count += 1

            diagnostics.append(
                {
                    "model": model,
                    "model_id": model_id,
                    "group": group,
                    "metric": metric,
                    "L1_easy": values.get("L1", NA),
                    "L2_medium": values.get("L2", NA),
                    "L3_hard": values.get("L3", NA),
                    "L4_expert": values.get("L4", NA),
                    "L1_to_L4_drop": drop,
                    "relative_drop_pct": relative_drop,
                    "adjacent_drop_count": adjacent_drop_count,
                    "covered_levels": len(numeric_levels),
                }
            )
    return diagnostics


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in columns})


def _write_difficulty_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lllrrrrrrrrrr}",
        "\\hline",
        "Run & Level & Name & N & Templates & Valid & Invalid & TSR-path & VAcc & R-F1 & LIG-F1 & WAcc & CDS \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    format_value(row.get("model")),
                    format_value(row.get("difficulty_level")),
                    format_value(row.get("difficulty_name")),
                    format_value(row.get("n_total")),
                    format_value(row.get("n_templates")),
                    format_value(row.get("n_valid_json")),
                    format_value(row.get("invalid_rate")),
                    format_value(row.get("TSR-path")),
                    format_value(row.get("VAcc")),
                    format_value(row.get("R-F1")),
                    format_value(row.get("LIG-F1")),
                    format_value(row.get("WAcc")),
                    format_value(row.get("CDS")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_diagnostic_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lllrrrrrr}",
        "\\hline",
        "Run & Metric & Levels & L1 easy & L2 medium & L3 hard & L4 expert & L1--L4 drop & Relative drop \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    format_value(row.get("model")),
                    format_value(row.get("metric")),
                    format_value(row.get("covered_levels")),
                    format_value(row.get("L1_easy")),
                    format_value(row.get("L2_medium")),
                    format_value(row.get("L3_hard")),
                    format_value(row.get("L4_expert")),
                    format_value(row.get("L1_to_L4_drop")),
                    format_value(row.get("relative_drop_pct")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_difficulty_results(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    index_rows: list[dict[str, Any]] | None = None,
    difficulty_csv: Path | None = None,
    layout_root: Path | None = None,
    model_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    lookup_rows = [*(index_rows or []), *rows]
    difficulty_lookup = load_difficulty_lookup(
        lookup_rows,
        difficulty_csv=difficulty_csv,
        layout_root=layout_root,
    )
    annotated_rows, missing_templates = annotate_rows_with_difficulty(rows, difficulty_lookup)
    summaries = summarize_models_by_difficulty(annotated_rows, model_configs=model_configs)
    diagnostics = summarize_difficulty_diagnostics(summaries)

    _write_csv(out_dir / "difficulty_results.csv", summaries, DIFFICULTY_RESULT_COLUMNS)
    _write_csv(out_dir / "difficulty_diagnostic_summary.csv", diagnostics, DIFFICULTY_DIAGNOSTIC_COLUMNS)
    _write_difficulty_table_tex(out_dir / "difficulty_results_table.tex", summaries)
    _write_diagnostic_table_tex(out_dir / "difficulty_diagnostic_summary_table.tex", diagnostics)

    template_counts = Counter(info["difficulty_level"] for info in difficulty_lookup.values())
    row_counts = Counter(row["difficulty_level"] for row in annotated_rows)
    metadata = {
        "difficulty_level_order": DIFFICULTY_LEVEL_ORDER,
        "difficulty_level_labels": DIFFICULTY_LEVEL_LABELS,
        "difficulty_csv": str(difficulty_csv or DEFAULT_DIFFICULTY_CSV),
        "difficulty_csv_exists": bool((difficulty_csv or DEFAULT_DIFFICULTY_CSV).exists()),
        "layout_root": str(layout_root) if layout_root else None,
        "n_templates_with_difficulty": len(difficulty_lookup),
        "template_counts_by_level": {level: template_counts.get(level, 0) for level in DIFFICULTY_LEVEL_ORDER},
        "n_rows_input": len(rows),
        "n_rows_with_difficulty": len(annotated_rows),
        "row_counts_by_level": {level: row_counts.get(level, 0) for level in DIFFICULTY_LEVEL_ORDER},
        "missing_templates": sorted(missing_templates),
        "diagnostic_rule": "Report per-model metrics separately for calibrated L1-L4 difficulty levels; L1-to-L4 drop is L1 metric minus L4 metric, so positive values indicate degradation from easy to expert samples.",
    }
    write_json(out_dir / "difficulty_results_metadata.json", metadata)
    return {
        "summaries": summaries,
        "diagnostics": diagnostics,
        "metadata": metadata,
    }
