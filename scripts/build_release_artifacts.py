#!/usr/bin/env python3
"""Build portable split and rights-audit manifests for the public release."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REDUNDANT_TEMPLATES = (
    "de_3",
    "de_4",
    "es_4",
    "ja_23",
    "ja_24",
    "ja_25",
    "ja_26",
    "ja_27",
    "ja_28",
    "zn_12",
)


def repository_template(template_name: str) -> str:
    if template_name.startswith("Arabic-"):
        return template_name.replace("Arabic-", "ar_", 1)
    if template_name.startswith("zn_en_"):
        return template_name.replace("zn_en_", "zh-Hant_en_", 1)
    if template_name == "zn_11":
        return "zh-Hant_11"
    if template_name.startswith("zn_"):
        return template_name.replace("zn_", "zh-Hans_", 1)
    return template_name


def portable_sample(row: dict[str, object]) -> dict[str, object]:
    template_name = str(row["template_name"])
    instance_id = str(row["instance_id"])
    repo_template = repository_template(template_name)
    return {
        "sample_id": f"{template_name}__{instance_id}",
        "template_name": template_name,
        "repository_template": repo_template,
        "instance_id": instance_id,
        "image_path": f"datasets/{repo_template}/{instance_id}/{repo_template}-{instance_id}.png",
        "label_path": f"datasets/{repo_template}/{instance_id}/answer.json",
    }


def build(source_dir: Path, output_dir: Path) -> None:
    split_dir = output_dir / "splits" / "template_stratified_seed42"
    provenance_dir = output_dir / "provenance"
    split_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    assignments: list[dict[str, str]] = []
    with (source_dir / "template_assignments.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row["repository_template"] = repository_template(row["template_name"])
            assignments.append(row)

    assignment_fields = ["template_name", "repository_template", *fieldnames[1:]]
    with (split_dir / "template_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(
            target, fieldnames=assignment_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(assignments)

    for split in ("train", "val", "test"):
        with (source_dir / f"{split}_index.jsonl").open(encoding="utf-8") as source:
            rows = [portable_sample(json.loads(line)) for line in source if line.strip()]
        with (split_dir / f"{split}_index.jsonl").open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=True) + "\n")

    shutil.copyfile(
        source_dir / "split_metadata.json", split_dir / "split_metadata.json"
    )
    with (source_dir / "split_composition.csv").open(
        encoding="utf-8", newline=""
    ) as source, (split_dir / "split_composition.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerows(csv.reader(source))

    split_by_template = {row["template_name"]: row["split"] for row in assignments}
    templates = [row["template_name"] for row in assignments] + list(REDUNDANT_TEMPLATES)
    rights_fields = [
        "template_name",
        "repository_template",
        "benchmark_scope",
        "official_split",
        "source_title",
        "source_url",
        "rightsholder",
        "source_license",
        "license_evidence_url",
        "privacy_review_status",
        "redistribution_status",
        "audit_note",
    ]
    with (provenance_dir / "template_rights.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=rights_fields, lineterminator="\n")
        writer.writeheader()
        for template_name in templates:
            canonical = template_name in split_by_template
            writer.writerow(
                {
                    "template_name": template_name,
                    "repository_template": repository_template(template_name),
                    "benchmark_scope": "canonical" if canonical else "redundant",
                    "official_split": split_by_template.get(template_name, "excluded"),
                    "source_license": "UNVERIFIED",
                    "privacy_review_status": "UNVERIFIED",
                    "redistribution_status": "UNVERIFIED_DO_NOT_REDISTRIBUTE",
                    "audit_note": (
                        "No machine-readable source, rights, or privacy evidence was "
                        "found in the release metadata as of 2026-07-23."
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_split_dir, args.output_dir)


if __name__ == "__main__":
    main()
