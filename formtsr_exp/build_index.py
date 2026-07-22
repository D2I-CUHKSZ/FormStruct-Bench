from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import scan_dataset
from .io_utils import write_jsonl
from .schema import summarize_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FormTSR-Bench dataset index.")
    parser.add_argument("--data-root", default="./FormTSR/datasets")
    parser.add_argument("--out", default="outputs/main_exp/dataset_index.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--templates", default="", help="Comma-separated template names.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema-out", default="outputs/main_exp/schema_summary.json")
    parser.add_argument("--schema-max-files", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = {item.strip() for item in args.templates.split(",") if item.strip()} or None
    rows = scan_dataset(
        Path(args.data_root),
        templates=templates,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    write_jsonl(Path(args.out), rows)
    summarize_schema(rows, Path(args.schema_out), max_files=args.schema_max_files)
    print(f"indexed {len(rows)} valid samples -> {args.out}")
    print(f"schema summary -> {args.schema_out}")


if __name__ == "__main__":
    main()
