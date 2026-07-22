#!/usr/bin/env python3
"""Infer coarse and fine document domains from annotation titles."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DomainRule:
    coarse: str
    fine: str
    keywords: tuple[str, ...]


DOMAIN_RULES = [
    DomainRule(
        "Government and immigration",
        "Visa, entry, and immigration",
        (
            "visa",
            "asylum",
            "withholding of removal",
            "entry for employment",
            "proposed stay",
            "dependants",
            "travel document",
            "tourism application",
            "employment authorization",
            "签证",
            "來臺觀光",
            "来港就业",
            "居留",
            "申请信息",
        ),
    ),
    DomainRule(
        "Government and immigration",
        "Civil registration and certificates",
        (
            "certificate",
            "resident record",
            "household registration",
            "seal registration",
            "relocation",
            "reburial",
            "burial",
            "戸籍",
            "住民票",
            "印鑑",
            "改葬",
            "異動",
            "証明",
        ),
    ),
    DomainRule(
        "Government and immigration",
        "Licensing and public permits",
        (
            "license",
            "licencia",
            "permit",
            "parental rights",
            "official processing",
            "immigration authorities",
            "autoridad",
        ),
    ),
    DomainRule(
        "Education and student services",
        "Admissions and enrollment",
        (
            "admission",
            "college application",
            "application form for secondary",
            "student's particulars",
            "universal college application",
            "school most recently",
            "入学",
            "学生",
        ),
    ),
    DomainRule(
        "Education and student services",
        "Scholarship and study abroad",
        (
            "scholarship",
            "study abroad",
            "china scholarship council",
            "international office",
            "留学",
            "国家留学基金",
            "奖学金",
            "stipendien",
            "stipendium",
        ),
    ),
    DomainRule(
        "Education and student services",
        "Student aid and housing",
        (
            "student financial",
            "family economic",
            "housing application",
            "student fees",
            "家庭经济",
            "在校",
        ),
    ),
    DomainRule(
        "Employment and human resources",
        "Job application and resume",
        (
            "employment application",
            "position information",
            "work history",
            "employment history",
            "career-related",
            "professional skills",
            "job resume",
            "resume",
            "career information",
            "履歴",
            "職務",
        ),
    ),
    DomainRule(
        "Employment and human resources",
        "Staff administration",
        (
            "leave",
            "service termination",
            "reward",
            "worker",
            "employment information",
            "supervisor approval",
            "إجازة",
            "مكافأة",
            "العامل",
        ),
    ),
    DomainRule(
        "Healthcare and medical",
        "Healthcare enrollment and benefits",
        (
            "health care fund",
            "dependent enrollment",
            "medicare",
            "insured",
            "protection plans",
            "beneficiary",
            "insurance",
        ),
    ),
    DomainRule(
        "Healthcare and medical",
        "Medical examination and patient care",
        (
            "patient",
            "medical",
            "health information",
            "outpatient",
            "disease",
            "physical exam",
            "prescription",
            "saúde",
            "paciente",
            "通院",
            "体格检查",
        ),
    ),
    DomainRule(
        "Finance, banking, and utilities",
        "Credit, banking, and accounts",
        (
            "credit application",
            "bank references",
            "account opening",
            "fund account",
            "trade account",
            "financial institution",
            "bank deduction",
            "postal savings bank",
            "direct debit",
            "account transfer",
            "口座",
            "金融機関",
            "基金账号",
            "交易账号",
        ),
    ),
    DomainRule(
        "Finance, banking, and utilities",
        "Payment and settlement",
        (
            "payment",
            "payee",
            "amount",
            "settlement",
            "foreign income",
            "declaration",
            "income declaration",
            "缴",
            "支付",
            "涉外收入",
            "払込",
            "回收",
        ),
    ),
    DomainRule(
        "Finance, banking, and utilities",
        "Utility and energy services",
        (
            "electricity",
            "energy",
            "consumer",
            "consumidor",
            "conta contrato",
            "unidade consumidora",
            "ponto de entrega",
            "environment characteristic",
            "serviceart",
            "störungsbeschreibung",
            "solicitud de servicio de mantenimiento",
            "mantenimiento",
        ),
    ),
    DomainRule(
        "Business and operations",
        "Product specification and testing",
        (
            "product specification",
            "cigarette making",
            "testing application",
            "children's product",
            "sample and product",
            "product selection",
            "construction",
            "wall",
            "anchor",
            "製品",
            "採用",
        ),
    ),
    DomainRule(
        "Business and operations",
        "Document, contract, and project workflow",
        (
            "document clearance",
            "contract",
            "department review",
            "submission",
            "distribution",
            "research request",
            "project information",
            "requested by",
            "approval",
        ),
    ),
    DomainRule(
        "Business and operations",
        "Company, procurement, and organization registration",
        (
            "company",
            "enterprise",
            "procurement",
            "business type",
            "financing needs",
            "equity",
            "brand value",
            "organization",
            "organization management",
            "会社",
            "企業",
            "団体",
            "公司",
            "融资",
        ),
    ),
    DomainRule(
        "Community, religious, and nonprofit services",
        "Religious center administration",
        (
            "quran",
            "religious",
            "center information",
            "center basic information",
            "center administration",
            "committee",
            "competition",
            "reclassification",
            "local competition",
            "قرآنية",
            "القرآن",
            "المركز",
            "الحلقات",
            "مسابقة",
        ),
    ),
    DomainRule(
        "Community, religious, and nonprofit services",
        "Community club and membership",
        (
            "club",
            "member",
            "membership",
            "activity information",
            "public corporation",
            "association",
            "公益社団法人",
            "クラブ",
            "会員",
            "活動",
            "入会",
        ),
    ),
    DomainRule(
        "Service request and customer administration",
        "General service request or reservation",
        (
            "complaint",
            "suggestion",
            "request type",
            "service demand",
            "service information",
            "reservation",
            "pre-registration",
            "registration information",
            "q&a information",
            "note information",
            "備考",
            "予約",
            "登録",
            "需求",
            "طلب",
            "شكوى",
            "اقتراح",
        ),
    ),
]


SCRIPT_BY_LANGUAGE = {
    "English": "Latin",
    "Chinese": "CJK",
    "Japanese": "CJK",
    "Arabic": "Arabic",
    "German": "Latin",
    "Portuguese": "Latin",
    "Spanish": "Latin",
    "Chinese+English": "CJK+Latin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer domain labels from annotation titles.")
    parser.add_argument("--input", required=True, help="Annotation JSON file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    return parser.parse_args()


def load_samples(input_path: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.json"))
    samples: list[tuple[Path, int, dict[str, Any]]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            samples.append((path, 0, payload))
        elif isinstance(payload, list):
            samples.extend((path, idx, item) for idx, item in enumerate(payload) if isinstance(item, dict))
    return samples


def clean_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\n", " ").split())


def collect_titles(sample: dict[str, Any]) -> tuple[list[str], list[str]]:
    root_titles: list[str] = []
    section_titles: list[str] = []

    for field in sample.get("fields", []):
        if isinstance(field, dict):
            title = clean_title(field.get("original_label") or field.get("semantic_key"))
            if title:
                root_titles.append(title)

    metadata = sample.get("metadata", {})
    layout = metadata.get("layout_structure", {}) if isinstance(metadata, dict) else {}
    for section in layout.get("sections", []) if isinstance(layout, dict) else []:
        if isinstance(section, dict):
            title = clean_title(section.get("title"))
            if title:
                section_titles.append(title)

    return root_titles, section_titles


def infer_domain(title_text: str) -> tuple[str, str, str, int, list[str]]:
    lowered = title_text.lower()
    best_rule: DomainRule | None = None
    best_hits: list[str] = []
    for rule in DOMAIN_RULES:
        hits = [keyword for keyword in rule.keywords if keyword.lower() in lowered]
        if len(hits) > len(best_hits):
            best_rule = rule
            best_hits = hits

    if best_rule is None:
        return "Unclassified", "Unclassified", "low", 0, []

    confidence = "high" if len(best_hits) >= 2 else "medium"
    return best_rule.coarse, best_rule.fine, confidence, len(best_hits), best_hits


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path, index, sample in load_samples(input_path):
        metadata = sample.get("metadata", {})
        language = metadata.get("language", "") if isinstance(metadata, dict) else ""
        root_titles, section_titles = collect_titles(sample)
        title_text = " | ".join([*root_titles, *section_titles, path.stem])
        coarse, fine, confidence, match_count, matched_keywords = infer_domain(title_text)
        rows.append(
            {
                "sample_id": sample.get("id", ""),
                "file": path.name if index == 0 else f"{path.name}#{index}",
                "image": sample.get("img", ""),
                "language": language,
                "script": SCRIPT_BY_LANGUAGE.get(str(language), ""),
                "coarse_domain": coarse,
                "fine_domain": fine,
                "confidence": confidence,
                "match_count": match_count,
                "matched_keywords": "; ".join(matched_keywords),
                "root_titles": " | ".join(root_titles),
                "section_titles": " | ".join(section_titles),
            }
        )

    df = pd.DataFrame(rows)
    sample_csv = output_dir / "title_domain_sample_stats.csv"
    summary_csv = output_dir / "title_domain_summary.csv"
    summary_json = output_dir / "title_domain_summary.json"
    table_md = output_dir / "title_domain_table.md"

    df.to_csv(sample_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    if rows:
        summary = (
            df.groupby(["coarse_domain", "fine_domain"], dropna=False)
            .size()
            .reset_index(name="templates")
            .sort_values(["coarse_domain", "fine_domain"])
        )
        summary["pct"] = (summary["templates"] / len(rows) * 100).round(1)
    else:
        summary = pd.DataFrame(columns=["coarse_domain", "fine_domain", "templates", "pct"])

    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(summary.to_json(orient="records", force_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| Coarse domain | Fine domain | Templates | % |",
        "| --- | --- | ---: | ---: |",
    ]
    for record in summary.to_dict(orient="records"):
        lines.append(
            "| {coarse_domain} | {fine_domain} | {templates} | {pct:.1f}% |".format(**record)
        )
    lines.append(f"| Total |  | {len(rows)} | 100.0% |")
    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"loaded samples: {len(rows)}")
    print("output file paths:")
    for path in [sample_csv, summary_csv, summary_json, table_md]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
