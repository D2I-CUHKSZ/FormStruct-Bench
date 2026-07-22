#!/usr/bin/env python3
"""Generate template language, script, and writing-direction statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import NamedTuple


class LanguageInfo(NamedTuple):
    language: str
    script: str
    direction: str


LANGUAGE_ORDER = [
    "Arabic",
    "Chinese",
    "Chinese--English",
    "English",
    "German",
    "Japanese",
    "Portuguese",
    "Spanish",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count template languages, scripts, and writing directions from a dataset root."
    )
    parser.add_argument("--template-root", default="FormTSR/datasets")
    parser.add_argument("--output", default="outputs/template_stats")
    return parser.parse_args()


def classify_template(template_name: str) -> LanguageInfo:
    if template_name.startswith("Arabic-"):
        return LanguageInfo("Arabic", "Arabic", "RTL")
    if template_name.startswith("zn_en_"):
        return LanguageInfo("Chinese--English", "Han + Latin", "LTR")
    if template_name.startswith("zn_"):
        return LanguageInfo("Chinese", "Han", "LTR")
    if template_name.startswith("en_"):
        return LanguageInfo("English", "Latin", "LTR")
    if template_name.startswith("de_"):
        return LanguageInfo("German", "Latin", "LTR")
    if template_name.startswith("ja_"):
        return LanguageInfo("Japanese", "Han + Hiragana + Katakana", "LTR")
    if template_name.startswith("pt_"):
        return LanguageInfo("Portuguese", "Latin", "LTR")
    if template_name.startswith("es_"):
        return LanguageInfo("Spanish", "Latin", "LTR")
    raise ValueError(f"unknown template language prefix: {template_name}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["language", "script", "direction", "templates", "pct"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]], total: int) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Template language, script, and writing-direction distribution.}",
        r"\label{tab:template_language}",
        r"\begin{tabular}{llcrr}",
        r"\toprule",
        r"Language & Script & Direction & Templates & (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['language']} & {row['script']} & {row['direction']} & "
            f"{row['templates']} & {float(row['pct']):.2f} " + r"\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & & & \textbf{{{total}}} & \textbf{{100.00}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    template_root = Path(args.template_root)
    output = Path(args.output)
    if not template_root.is_dir():
        raise SystemExit(f"template root does not exist or is not a directory: {template_root}")
    output.mkdir(parents=True, exist_ok=True)

    template_names = sorted(path.name for path in template_root.iterdir() if path.is_dir())
    classified = [(name, classify_template(name)) for name in template_names]
    counts = Counter(info for _, info in classified)
    total = len(classified)

    rows: list[dict[str, object]] = []
    for language in LANGUAGE_ORDER:
        matching = [(info, count) for info, count in counts.items() if info.language == language]
        if not matching:
            continue
        info, count = matching[0]
        rows.append(
            {
                "language": info.language,
                "script": info.script,
                "direction": info.direction,
                "templates": count,
                "pct": round(count / total * 100, 4),
            }
        )

    csv_path = output / "template_language_script_direction.csv"
    json_path = output / "template_language_script_direction.json"
    tex_path = output / "template_language_script_direction_table.tex"
    metadata_path = output / "template_language_script_direction_metadata.json"
    write_csv(csv_path, rows)
    write_latex(tex_path, rows, total)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "template_root": str(template_root),
                "template_count": total,
                "ltr_templates": sum(count for info, count in counts.items() if info.direction == "LTR"),
                "rtl_templates": sum(count for info, count in counts.items() if info.direction == "RTL"),
                "templates": [name for name, _ in classified],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"templates: {total}")
    for path in [csv_path, json_path, tex_path, metadata_path]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
