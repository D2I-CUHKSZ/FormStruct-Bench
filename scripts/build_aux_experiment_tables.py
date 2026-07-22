from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


MODEL_SPECS = [
    ("gpt_vlm", "GPT-5.5", "gpt-5.5"),
    ("Qwen3.6-35B-A3B_aligned_metadata", "Qwen3.6-35B-A3B (aligned)", "Qwen3.6-35B-A3B"),
    ("glm4_6v_flash_sglang_vlm_aligned_metadata", "GLM-4.6V-Flash (aligned)", "GLM-4.6V-Flash"),
    ("qwen3_5_9b_sglang_vlm_aligned_metadata", "Qwen3.5-9B (aligned)", "Qwen3.5-9B"),
    ("kimi_vl_a3b_vllm_vlm", "Kimi-VL-A3B-Instruct", "Kimi-VL-A3B-Instruct"),
    ("step3_vl_10b_vllm_vlm", "Step3-VL-10B", "Step3-VL-10B"),
    ("caprl_internvl3_5_8b_vllm_vlm_aligned_metadata", "CapRL-InternVL3.5-8B (aligned)", "CapRL-InternVL3.5-8B"),
    ("deepseek_vl2_vllm_vlm", "DeepSeek-VL2", "DeepSeek-VL2"),
    ("gemma4_26b_hf_vlm", "Gemma-4-26B-A4B-it", "Gemma-4-26B-A4B-it"),
    ("deepseek_ocr2_sglang_vlm", "DeepSeek-OCR-2", "DeepSeek-OCR-2"),
    ("paddleocr_vl_1_6_pipeline_sglang", "PaddleOCR-VL-1.6", "PaddleOCR-VL-1.6"),
    ("mineru2_5_pro_vllm_engine_vlm", "MinerU2.5-Pro", "MinerU2.5-Pro"),
    ("unlimited_ocr_hf_vlm", "Unlimited-OCR", "Unlimited-OCR"),
]

CONSTRAINTS = [
    ("region_local_grids", "Local grids"),
    ("widget_grouping", "Widget groups"),
    ("key_field_relations", "Key-field rel."),
    ("line_item_groups", "Line items"),
    ("mixed_layout", "Mixed layout"),
]

ABLATIONS = [
    ("region_local_grid_vs_global_grid_on_grid_samples", "Local$-$global grid"),
    ("widget_answer_effect_on_widget_answer_samples", "+Widget ans."),
    ("widget_box_effect_on_widget_box_samples", "+Widget box"),
    ("widget_full_effect_on_widget_samples", "+Widget full"),
    ("relation_effect_on_relation_samples", "+Relation"),
    ("full_structural_vs_answer_only_on_structural_samples", "Full$-$answer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact LaTeX tables for FormTSR auxiliary experiments.")
    parser.add_argument("--difficulty-dir", default="outputs/aux_exp/difficulty")
    parser.add_argument("--constraint-dir", default="outputs/aux_exp/constraints")
    parser.add_argument("--ablation-dir", default="outputs/aux_exp/structure_ablation")
    parser.add_argument("--out", default="outputs/aux_exp/latex")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "TBD"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "--"
    if abs(number) < 0.00005:
        number = 0.0
    return f"{number:.4f}"


def table_block(*, caption: str, label: str, columns: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_coverage(difficulty_rows: list[dict[str, str]]) -> dict[str, tuple[int, int]]:
    coverage: dict[str, tuple[int, int]] = {}
    for row in difficulty_rows:
        model = row["model"]
        total, valid = coverage.get(model, (0, 0))
        coverage[model] = (total + int(row["n_total"]), valid + int(row["n_valid_json"]))
    return coverage


def valid_text(coverage: dict[str, tuple[int, int]], model: str) -> str:
    total, valid = coverage.get(model, (0, 0))
    return f"{valid}/{total}" if total else "--"


def difficulty_table(rows: list[dict[str, str]], coverage: dict[str, tuple[int, int]]) -> str:
    cds = {(row["model"], row["difficulty_level"]): row["CDS"] for row in rows}
    output_rows: list[list[str]] = []
    for model, display, _model_id in MODEL_SPECS:
        levels = [cds.get((model, level)) for level in ("L1", "L2", "L3", "L4")]
        l1 = as_float(levels[0])
        l4 = as_float(levels[3])
        drop = (l1 - l4) if l1 is not None and l4 is not None else None
        output_rows.append(
            [
                latex_escape(display),
                valid_text(coverage, model),
                *[fmt(value) for value in levels],
                fmt(drop),
            ]
        )
    return table_block(
        caption=(
            "Difficulty-stratified CDS on FormTSR-Bench. $\\Delta_{1\\rightarrow4}=\\mathrm{CDS}_{L1}-"
            "\\mathrm{CDS}_{L4}$; positive values indicate degradation from easy to expert. "
            "Valid reports parsed predictions over 7000 indexed samples."
        ),
        label="tab:formtsr_difficulty_cds",
        columns="lrrrrrr",
        header=["Model", "Valid", "L1", "L2", "L3", "L4", "$\\Delta_{1\\rightarrow4}$"],
        rows=output_rows,
    )


def constraint_table(rows: list[dict[str, str]], coverage: dict[str, tuple[int, int]]) -> str:
    deltas = {
        (row["model"], row["constraint"]): row["delta"]
        for row in rows
        if row.get("metric") == "CDS"
    }
    output_rows: list[list[str]] = []
    for model, display, _model_id in MODEL_SPECS:
        output_rows.append(
            [
                latex_escape(display),
                valid_text(coverage, model),
                *[fmt(deltas.get((model, constraint))) for constraint, _name in CONSTRAINTS],
            ]
        )
    return table_block(
        caption=(
            "Constraint-sliced CDS drops. Each entry is $\\Delta(c)=\\mathbb{E}[\\mathrm{CDS}\\mid\\neg c]-"
            "\\mathbb{E}[\\mathrm{CDS}\\mid c]$; positive values indicate lower performance when the constraint is present. "
            "Visual degradation is excluded because the clean main set has no degraded positive samples."
        ),
        label="tab:formtsr_constraint_cds",
        columns="lrrrrrr",
        header=["Model", "Valid", *[name for _key, name in CONSTRAINTS]],
        rows=output_rows,
    )


def ablation_table(rows: list[dict[str, str]], coverage: dict[str, tuple[int, int]]) -> str:
    deltas = {(row["model"], row["comparison"]): row["delta"] for row in rows}
    output_rows: list[list[str]] = []
    for model, display, _model_id in MODEL_SPECS:
        output_rows.append(
            [
                latex_escape(display),
                valid_text(coverage, model),
                *[fmt(deltas.get((model, comparison))) for comparison, _name in ABLATIONS],
            ]
        )
    return table_block(
        caption=(
            "Targeted structural-ablation deltas on samples where each added dimension is applicable. "
            "$\\Delta=\\mathrm{score}_{with}-\\mathrm{score}_{without}$; negative values expose errors hidden by the simpler score. "
            "Exact per-comparison scope sizes are reported in the accompanying CSV."
        ),
        label="tab:formtsr_structural_ablation",
        columns="lrrrrrrr",
        header=["Model", "Valid", *[name for _key, name in ABLATIONS]],
        rows=output_rows,
    )


def write_manifest(path: Path, coverage: dict[str, tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "display_name", "model_id", "n_total", "n_valid_json", "coverage", "aligned"],
        )
        writer.writeheader()
        for run_id, display, model_id in MODEL_SPECS:
            total, valid = coverage.get(run_id, (0, 0))
            writer.writerow(
                {
                    "run_id": run_id,
                    "display_name": display,
                    "model_id": model_id,
                    "n_total": total,
                    "n_valid_json": valid,
                    "coverage": f"{valid / total:.4f}" if total else "NA",
                    "aligned": "aligned_metadata" in run_id,
                }
            )


def main() -> None:
    args = parse_args()
    difficulty_dir = Path(args.difficulty_dir)
    constraint_dir = Path(args.constraint_dir)
    ablation_dir = Path(args.ablation_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    difficulty_rows = read_csv(difficulty_dir / "difficulty_results.csv")
    constraint_rows = read_csv(constraint_dir / "constraint_slice_results.csv")
    ablation_rows = read_csv(ablation_dir / "structure_ablation_targeted_deltas.csv")
    coverage = build_coverage(difficulty_rows)

    expected = {model for model, _display, _model_id in MODEL_SPECS}
    present = set(coverage)
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"missing canonical models from difficulty report: {missing}")

    tables = {
        "difficulty_cds_table.tex": difficulty_table(difficulty_rows, coverage),
        "constraint_cds_drop_table.tex": constraint_table(constraint_rows, coverage),
        "structure_ablation_targeted_table.tex": ablation_table(ablation_rows, coverage),
    }
    for filename, content in tables.items():
        (out_dir / filename).write_text(content, encoding="utf-8")

    preamble = "% Requires: \\usepackage{booktabs,graphicx}\n"
    combined = preamble + "\n".join(tables.values())
    (out_dir / "auxiliary_experiments_tables.tex").write_text(combined, encoding="utf-8")
    write_manifest(out_dir / "model_selection.csv", coverage)

    print(f"wrote {len(tables)} compact LaTeX tables -> {out_dir}")
    print(f"models: {len(MODEL_SPECS)}")


if __name__ == "__main__":
    main()
