from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_COLUMNS = [
    "run",
    "model_id",
    "status",
    "index_pages",
    "reported_total",
    "valid_predictions",
    "coverage",
    "bbox_source_space",
    "LG-GriTS-Top",
    "LG-GriTS-Top-corpus",
    "WG-F1",
    "WG-F1-corpus",
    "Rel-F1",
    "Rel-Precision-micro",
    "Rel-Recall-micro",
    "Rel-F1-micro",
    "Rel-F1-matched-endpoints",
    "Rel-F1-matched-endpoints-micro",
]

RELATION_COLUMNS = [
    "run",
    "relation_type",
    "kind",
    "TP",
    "pred",
    "GT",
    "precision",
    "recall",
    "F1",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compact_model_rows(
    rows: list[dict[str, str]], index_pages: int
) -> list[dict[str, str | int]]:
    compact: list[dict[str, str | int]] = []
    for row in rows:
        compact.append(
            {
                "run": row["model"],
                "model_id": row["model_id"],
                "status": row["comparison_status"],
                "index_pages": index_pages,
                "reported_total": row["n_total"],
                "valid_predictions": row["n_valid_json"],
                "coverage": row["coverage"],
                "bbox_source_space": row["bbox_source_space"],
                "LG-GriTS-Top": row["LG-GriTS-Top"],
                "LG-GriTS-Top-corpus": row["LG-GriTS-Top-corpus"],
                "WG-F1": row["WG-F1"],
                "WG-F1-corpus": row["WG-F1-corpus"],
                "Rel-F1": row["Rel-F1"],
                "Rel-Precision-micro": row["Rel-Precision-micro"],
                "Rel-Recall-micro": row["Rel-Recall-micro"],
                "Rel-F1-micro": row["Rel-F1-micro"],
                "Rel-F1-matched-endpoints": row["Rel-F1-matched-endpoints"],
                "Rel-F1-matched-endpoints-micro": row[
                    "Rel-F1-matched-endpoints-micro"
                ],
            }
        )
    return compact


def compact_relation_rows(
    model_rows: list[dict[str, str | int]], raw_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str | int]], list[str]]:
    canonical_types = sorted(
        {
            row["relation_type"]
            for row in raw_rows
            if numeric(row.get("GT", "0")) > 0
        }
    )
    by_model_type = {
        (row["model"], row["relation_type"]): row for row in raw_rows
    }
    raw_by_model: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        raw_by_model.setdefault(row["model"], []).append(row)

    output: list[dict[str, str | int]] = []
    for model_row in model_rows:
        run = str(model_row["run"])
        for relation_type in canonical_types:
            row = by_model_type.get((run, relation_type))
            if row is None:
                output.append(
                    {
                        "run": run,
                        "relation_type": relation_type,
                        "kind": "canonical_not_scored",
                        "TP": "NA",
                        "pred": "NA",
                        "GT": "NA",
                        "precision": "NA",
                        "recall": "NA",
                        "F1": "NA",
                    }
                )
                continue
            output.append(
                {
                    "run": run,
                    "relation_type": relation_type,
                    "kind": "canonical",
                    "TP": row["TP"],
                    "pred": row["pred"],
                    "GT": row["GT"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "F1": row["F1"],
                }
            )

        unsupported = [
            row
            for row in raw_by_model.get(run, [])
            if row["relation_type"] not in canonical_types and numeric(row["pred"]) > 0
        ]
        if unsupported:
            tp = sum(int(numeric(row["TP"])) for row in unsupported)
            pred = sum(int(numeric(row["pred"])) for row in unsupported)
            output.append(
                {
                    "run": run,
                    "relation_type": "__unsupported_prediction_types__",
                    "kind": "unsupported_aggregate",
                    "TP": tp,
                    "pred": pred,
                    "GT": 0,
                    "precision": f"{tp / pred:.6f}" if pred else "NA",
                    "recall": "NA",
                    "F1": "NA",
                }
            )
    return output, canonical_types


def render_readme(rows: list[dict[str, str | int]], index_pages: int) -> str:
    lines = [
        "# Clean 7000-Page Hierarchical Metrics",
        "",
        f"Scope: `{index_pages}` pages from `outputs/main_exp/dataset_index.jsonl`.",
        "",
        "The table contains one row per canonical raw run. Aligned-metadata aliases and smoke duplicates are excluded. Partial and failed runs remain visible so coverage is not confused with prediction quality.",
        "",
        "| Run | Status | Valid/Index | LG-GriTS-Top | WG-F1 | Rel-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {status} | {valid_predictions}/{index_pages} | {lg} | {wg} | {rel} |".format(
                run=row["run"],
                status=row["status"],
                valid_predictions=row["valid_predictions"],
                index_pages=index_pages,
                lg=row["LG-GriTS-Top"],
                wg=row["WG-F1"],
                rel=row["Rel-F1"],
            )
        )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `model_metrics.csv`: compact per-run primary and appendix metrics.",
            "- `relation_type_metrics.csv`: canonical GT relation types; unsupported predicted types are aggregated into one diagnostic row per run.",
            "- `per_sample/`: page-level metrics for every included run.",
            "- `audit/`: the evaluator's full-width tables and metadata.",
            "- `manifest.json`: exact sources and export rules.",
            "",
            "Primary LG-GriTS-Top, WG-F1, and Rel-F1 values are page macro scores. Relation micro and matched-endpoint scores are included in `model_metrics.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir: Path = args.report_dir
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to merge into non-empty directory: {out}")

    metadata_path = report_dir / "hierarchical_structure_metrics_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index_pages = int(metadata["n_indexed"])
    full_rows = read_csv(report_dir / "hierarchical_structure_metrics.csv")
    raw_relation_rows = read_csv(
        report_dir / "hierarchical_relation_type_metrics.csv"
    )
    model_rows = compact_model_rows(full_rows, index_pages)
    relation_rows, canonical_types = compact_relation_rows(
        model_rows, raw_relation_rows
    )

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "model_metrics.csv", MODEL_COLUMNS, model_rows)
    write_csv(out / "relation_type_metrics.csv", RELATION_COLUMNS, relation_rows)
    (out / "README.md").write_text(
        render_readme(model_rows, index_pages), encoding="utf-8"
    )

    per_sample_src = report_dir / "hierarchical_structure_per_sample"
    shutil.copytree(per_sample_src, out / "per_sample")
    audit = out / "audit"
    audit.mkdir()
    shutil.copy2(
        report_dir / "hierarchical_structure_metrics.csv",
        audit / "full_model_metrics.csv",
    )
    shutil.copy2(
        report_dir / "hierarchical_relation_type_metrics.csv",
        audit / "raw_relation_type_metrics.csv",
    )
    shutil.copy2(metadata_path, audit / "evaluator_metadata.json")

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_pages": index_pages,
        "n_canonical_runs": len(model_rows),
        "canonical_relation_types": canonical_types,
        "source_report_dir": str(report_dir),
        "source_index": metadata.get("index_path"),
        "source_predictions": metadata.get("pred_root"),
        "metadata_root": metadata.get("metadata_root"),
        "layout_root": metadata.get("layout_root"),
        "selection": "Canonical raw runs explicitly selected during evaluator invocation; aligned-metadata aliases and smoke duplicates excluded.",
        "unsupported_relation_types": "Aggregated per run in relation_type_metrics.csv; unmodified rows are retained under audit/.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
