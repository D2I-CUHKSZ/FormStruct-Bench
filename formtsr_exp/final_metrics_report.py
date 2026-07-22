from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_jsonl, write_json
from .metrics import NA


RESULT_COLUMNS = [
    "model",
    "model_id",
    "group",
    "run_type",
    "comparison_status",
    "n_total",
    "n_valid_json",
    "coverage",
    "n_exact_match",
    "Page-EM",
    "Schema-nTED",
    "Value-nED",
    "TSR-path",
    "R-Precision@0.5",
    "R-Recall@0.5",
    "R-F1",
    "R-F1@0.75",
    "n_lig_applicable",
    "LIG-F1",
    "n_lg_gt_applicable",
    "LG-GriTS-Top",
    "n_wg_gt_applicable",
    "WG-F1",
    "n_rel_gt_applicable",
    "Rel-F1",
    "Rel-F1-micro",
    "Rel-F1-matched-endpoints",
]

CLEAN_RESULT_COLUMNS = [
    "model",
    "model_id",
    "n_total",
    "n_attempted",
    "n_valid_json",
    "coverage",
    "n_exact_match",
    "Page-EM",
    "Schema-nTED",
    "Value-nED",
    "TSR-path",
    "R-Precision@0.5",
    "R-Recall@0.5",
    "R-F1",
    "R-F1@0.75",
    "n_lig_applicable",
    "LIG-F1",
    "n_lg_gt_applicable",
    "LG-GriTS-Top",
    "n_wg_gt_applicable",
    "WG-F1",
    "n_rel_gt_applicable",
    "Rel-F1",
    "Rel-F1-micro",
    "Rel-F1-matched-endpoints",
]


def _read_by_model(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    order = [str(row.get("model") or "") for row in rows]
    if any(not model for model in order):
        raise ValueError(f"missing model value in {path}")
    if len(set(order)) != len(order):
        raise ValueError(f"duplicate model rows in {path}")
    return order, {model: row for model, row in zip(order, rows)}


def _require_same_models(reference: set[str], rows: dict[str, Any], path: Path) -> None:
    actual = set(rows)
    if actual != reference:
        missing = sorted(reference - actual)
        extra = sorted(actual - reference)
        raise ValueError(f"model mismatch for {path}: missing={missing}, extra={extra}")


def _require_equal(model: str, field: str, sources: list[tuple[str, dict[str, str]]]) -> str:
    values = {name: str(row.get(field) or "") for name, row in sources}
    if len(set(values.values())) != 1:
        raise ValueError(f"{field} mismatch for {model}: {values}")
    return next(iter(values.values()))


def merge_results(
    main_results: Path,
    page_results: Path,
    document_results: Path,
    structure_results: Path,
    hierarchical_results: Path,
) -> list[dict[str, str]]:
    page_order, page = _read_by_model(page_results)
    _, main = _read_by_model(main_results)
    _, document = _read_by_model(document_results)
    _, structure = _read_by_model(structure_results)
    _, hierarchical = _read_by_model(hierarchical_results)
    models = set(page_order)
    _require_same_models(models, main, main_results)
    _require_same_models(models, document, document_results)
    _require_same_models(models, structure, structure_results)
    _require_same_models(models, hierarchical, hierarchical_results)

    merged: list[dict[str, str]] = []
    for model in page_order:
        page_row = page[model]
        main_row = main[model]
        document_row = document[model]
        structure_row = structure[model]
        hierarchical_row = hierarchical[model]
        sources = [
            ("page", page_row),
            ("document", document_row),
            ("structure", structure_row),
            ("hierarchical", hierarchical_row),
        ]
        n_total = _require_equal(model, "n_total", sources)
        n_valid_json = _require_equal(model, "n_valid_json", sources)
        if str(main_row.get("n_total")) != n_total or str(main_row.get("n_valid_json")) != n_valid_json:
            raise ValueError(
                f"legacy main result coverage mismatch for {model}: "
                f"main={main_row.get('n_valid_json')}/{main_row.get('n_total')} "
                f"current={n_valid_json}/{n_total}"
            )
        merged.append(
            {
                "model": model,
                "model_id": page_row["model_id"],
                "group": page_row["group"],
                "run_type": page_row["run_type"],
                "comparison_status": page_row["comparison_status"],
                "n_total": n_total,
                "n_valid_json": n_valid_json,
                "coverage": page_row["coverage"],
                "n_exact_match": page_row["n_exact_match"],
                "Page-EM": page_row["Page-EM"],
                "Schema-nTED": document_row["Schema-nTED"],
                "Value-nED": document_row["Value-nED"],
                "TSR-path": main_row["TSR-path"],
                "R-Precision@0.5": structure_row["R-Precision@0.5"],
                "R-Recall@0.5": structure_row["R-Recall@0.5"],
                "R-F1": structure_row["R-F1"],
                "R-F1@0.75": structure_row["R-F1@0.75"],
                "n_lig_applicable": structure_row["n_lig_applicable"],
                "LIG-F1": structure_row["LIG-F1"],
                "n_lg_gt_applicable": hierarchical_row["n_lg_gt_applicable"],
                "LG-GriTS-Top": hierarchical_row["LG-GriTS-Top"],
                "n_wg_gt_applicable": hierarchical_row["n_wg_gt_applicable"],
                "WG-F1": hierarchical_row["WG-F1"],
                "n_rel_gt_applicable": hierarchical_row["n_rel_gt_applicable"],
                "Rel-F1": hierarchical_row["Rel-F1"],
                "Rel-F1-micro": hierarchical_row["Rel-F1-micro"],
                "Rel-F1-matched-endpoints": hierarchical_row[
                    "Rel-F1-matched-endpoints"
                ],
            }
        )
    return merged


def _numeric(value: str) -> float | None:
    if value in {"", NA, "TBD"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _prediction_attempt_ids(model: str, raw_root: Path, pred_root: Path, error_root: Path) -> set[str]:
    attempted: set[str] = set()
    for root, pattern in ((raw_root / model, "*"), (pred_root / model, "*.json")):
        if root.is_dir():
            attempted.update(path.stem for path in root.glob(pattern) if path.is_file())
    error_path = error_root / f"{model}.jsonl"
    if error_path.exists():
        with error_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                try:
                    payload = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"invalid error JSONL at {error_path}:{line_no}") from exc
                sample_id = payload.get("sample_id") if isinstance(payload, dict) else None
                if sample_id:
                    attempted.add(str(sample_id))
    return attempted


def select_best_full_runs(
    rows: list[dict[str, str]],
    index_path: Path,
    raw_root: Path,
    pred_root: Path,
    error_root: Path,
) -> list[dict[str, str]]:
    index_ids = {str(row["sample_id"]) for row in read_jsonl(index_path)}
    if not index_ids:
        raise ValueError(f"empty dataset index: {index_path}")

    candidates: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["run_type"] != "raw" or int(row["n_total"]) != len(index_ids):
            continue
        attempted = _prediction_attempt_ids(row["model"], raw_root, pred_root, error_root) & index_ids
        if attempted != index_ids or int(row["n_valid_json"]) <= 0:
            continue
        candidate = dict(row)
        candidate["n_attempted"] = str(len(attempted))
        candidates.setdefault(row["model_id"].strip().casefold(), []).append(candidate)

    def score(row: dict[str, str]) -> tuple[float, ...]:
        fields = ("n_valid_json", "Schema-nTED", "Value-nED", "TSR-path", "R-F1", "LIG-F1")
        return tuple(_numeric(row[field]) or 0.0 for field in fields)

    selected = [max(group, key=score) for group in candidates.values()]
    return sorted(selected, key=lambda row: row["model_id"].casefold())


def _format(value: str, places: int = 4) -> str:
    parsed = _numeric(value)
    return f"{parsed:.{places}f}" if parsed is not None else value


def _write_clean_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLEAN_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CLEAN_RESULT_COLUMNS} for row in rows)


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
    clean_rows: list[dict[str, str]],
    metadata: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    _write_clean_csv(out_dir / "main_experiment_results.csv", clean_rows)
    write_json(out_dir / "final_reporting_metrics_metadata.json", metadata)

    lines = [
        "# Final Reporting Metrics",
        "",
        "Only the best fully attempted raw run per model is shown below. Page-EM is displayed as exact pages / total because the decimal is sparse.",
        "",
        "| Model | Valid/Total | Exact pages | Schema-nTED | Value-nED | TSR-path | R-F1@.5 | R-F1@.75 | LIG-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in clean_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_id"],
                    f"{row['n_valid_json']}/{row['n_total']}",
                    f"{row['n_exact_match']}/{row['n_total']}",
                    _format(row["Schema-nTED"], 6),
                    _format(row["Value-nED"], 6),
                    _format(row["TSR-path"], 4),
                    _format(row["R-F1"], 6),
                    _format(row["R-F1@0.75"], 6),
                    _format(row["LIG-F1"], 6),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Hierarchical Structure Metrics",
            "",
            "Rel-F1 is the required page macro. Corpus micro, per-type, and matched-endpoint diagnostics remain in the hierarchical appendix files.",
            "",
            "| Model | LG-GriTS-Top | WG-F1 | Rel-F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in clean_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_id"],
                    _format(row["LG-GriTS-Top"], 6),
                    _format(row["WG-F1"], 6),
                    _format(row["Rel-F1"], 6),
                ]
            )
            + " |"
        )
    (out_dir / "final_reporting_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Model & Valid & Exact & Schema-nTED & Value-nED & TSR-path & R-F1@.5 & R-F1@.75 & LIG-F1 \\",
        r"\midrule",
    ]
    for row in clean_rows:
        tex.append(
            " & ".join(
                [
                    _latex_escape(row["model_id"]),
                    _latex_escape(f"{row['n_valid_json']}/{row['n_total']}"),
                    _latex_escape(f"{row['n_exact_match']}/{row['n_total']}"),
                    _format(row["Schema-nTED"]),
                    _format(row["Value-nED"]),
                    _format(row["TSR-path"]),
                    _format(row["R-F1"]),
                    _format(row["R-F1@0.75"]),
                    _format(row["LIG-F1"]),
                ]
            )
            + r" \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    tex.extend(
        [
            "",
            r"\medskip",
            "",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Model & LG-GriTS-Top & WG-F1 & Rel-F1 \\",
            r"\midrule",
        ]
    )
    for row in clean_rows:
        tex.append(
            " & ".join(
                [
                    _latex_escape(row["model_id"]),
                    _format(row["LG-GriTS-Top"]),
                    _format(row["WG-F1"]),
                    _format(row["Rel-F1"]),
                ]
            )
            + r" \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "final_reporting_metrics_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the latest FormTSR metrics into one reporting table.")
    parser.add_argument("--main-results", default="outputs/main_exp/main_results.csv")
    parser.add_argument("--page-results", default="outputs/main_exp/page_em_results.csv")
    parser.add_argument("--document-results", default="outputs/main_exp/document_similarity_results.csv")
    parser.add_argument("--structure-results", default="outputs/main_exp/corrected_structure_metrics.csv")
    parser.add_argument(
        "--hierarchical-results",
        default="outputs/main_exp/hierarchical_structure_metrics.csv",
    )
    parser.add_argument("--index", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--raw-root", default="outputs/main_exp/raw")
    parser.add_argument("--pred-root", default="outputs/main_exp/pred")
    parser.add_argument("--error-root", default="outputs/main_exp/errors")
    parser.add_argument("--out", default="outputs/main_exp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "main_results": Path(args.main_results),
        "page_results": Path(args.page_results),
        "document_results": Path(args.document_results),
        "structure_results": Path(args.structure_results),
        "hierarchical_results": Path(args.hierarchical_results),
    }
    rows = merge_results(*sources.values())
    clean_rows = select_best_full_runs(
        rows,
        Path(args.index),
        Path(args.raw_root),
        Path(args.pred_root),
        Path(args.error_root),
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {key: str(value) for key, value in sources.items()},
        "n_runs": len(rows),
        "n_comparable_raw": sum(row["comparison_status"] == "comparable_raw" for row in rows),
        "n_best_full_runs": len(clean_rows),
        "main_columns": RESULT_COLUMNS,
        "selection": "comparison_status == comparable_raw; aligned, smoke, failed, and partial runs remain only in the all-runs audit CSV",
        "clean_selection": "raw runs only; attempted sample ids from raw/pred/error must cover the full index; zero-valid runs are excluded; duplicate model_id runs are ranked by valid count then Schema-nTED, Value-nED, TSR-path, R-F1, and LIG-F1",
        "metric_policy": {
            "coverage": "always report n_valid_json / n_total",
            "Page-EM": "report exact-page count and Page-EM; missing/invalid pages are zero",
            "Schema-nTED": "full-scope document schema similarity",
            "Value-nED": "full-scope precision-aware soft value similarity",
            "TSR-path": "strict GT-path/value field recall from legacy main results",
            "R-F1": "corrected canonical type and coordinate-normalized page-macro F1 at IoU 0.5",
            "R-F1@0.75": "strict localization diagnostic",
            "LIG-F1": "corrected coordinate-normalized page-macro F1 over applicable pages",
            "LG-GriTS-Top": "page-macro local-grid topology score with frozen corrected R-F1 parent matches",
            "WG-F1": "page-macro two-level widget-group score with strict type, state, and IoU member matching",
            "Rel-F1": "page-macro directed typed-relation F1 with endpoint mappings frozen before relation scoring",
            "Rel-F1-micro": "corpus-level relation micro F1; appendix diagnostic",
            "Rel-F1-matched-endpoints": "page-macro relation F1 conditioned on matched endpoints; appendix diagnostic",
        },
        "excluded_from_main_table": {
            "VAcc": "highly redundant with Value-nED and does not penalize extra values",
            "WAcc": "widget-specific slice; report separately with its applicable denominator",
            "CDS": "legacy composite contains superseded unnormalized structure metrics",
            "valid_only_metrics": "coverage diagnostics only",
        },
    }
    write_results(Path(args.out), clean_rows, metadata)
    print(f"wrote clean main experiment CSV -> {Path(args.out) / 'main_experiment_results.csv'}")


if __name__ == "__main__":
    main()
