#!/usr/bin/env python3
"""Infer page-level text direction metadata for layout JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RTL_LANGUAGES = {
    "arabic",
    "hebrew",
    "persian",
    "farsi",
    "urdu",
}

VERTICAL_RL_LANGUAGES = {
    "chinese",
    "japanese",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


def iter_nodes(value: Any) -> Any:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_nodes(child)


def collect_text_bboxes(data: dict[str, Any]) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for node in iter_nodes(data.get("fields", [])):
        if not isinstance(node, dict):
            continue
        if node.get("data_type") != "text":
            continue
        bbox = node.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            bboxes.append(bbox)
    return bboxes


def infer_line_orientation(bboxes: list[list[float]]) -> tuple[str, float, dict[str, int]]:
    """Infer whether text lines are mostly horizontal or vertical.

    Most form labels are horizontal. Single-character or nearly square boxes are
    ignored because they do not carry reliable orientation evidence.
    """

    counts = {"horizontal": 0, "vertical": 0, "ambiguous": 0}
    for x1, y1, x2, y2 in bboxes:
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width <= 0 or height <= 0:
            continue
        if width >= height * 1.35:
            counts["horizontal"] += 1
        elif height >= width * 1.35:
            counts["vertical"] += 1
        else:
            counts["ambiguous"] += 1

    decisive = counts["horizontal"] + counts["vertical"]
    if decisive == 0:
        return "horizontal", 0.5, counts

    if counts["vertical"] > counts["horizontal"]:
        orientation = "vertical"
        winning = counts["vertical"]
    else:
        orientation = "horizontal"
        winning = counts["horizontal"]

    confidence = round(winning / decisive, 4)
    return orientation, confidence, counts


def infer_reading_direction(
    language: str | None, line_orientation: str
) -> tuple[str, float, str, str | None]:
    normalized_language = language.strip().lower() if language else ""

    if line_orientation == "vertical":
        if normalized_language in VERTICAL_RL_LANGUAGES:
            return "ttb", 0.95, "vertical-rl", "rtl"
        return "ttb", 0.9, "vertical-lr", "ltr"

    if normalized_language in RTL_LANGUAGES:
        return "rtl", 0.99, "horizontal-rl", None
    return "ltr", 0.95, "horizontal-lr", None


def find_image(layout_path: Path, image_dir: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{layout_path.stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def with_text_direction_inserted(
    metadata: dict[str, Any], text_direction: dict[str, Any]
) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    inserted = False
    for key, value in metadata.items():
        if key == "text_direction":
            continue
        updated[key] = value
        if key == "language":
            updated["text_direction"] = text_direction
            inserted = True
    if not inserted:
        updated["text_direction"] = text_direction
    return updated


def process_file(layout_path: Path, image_dir: Path, dry_run: bool) -> dict[str, Any]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{layout_path}: metadata must be an object")

    language = metadata.get("language")
    if language is not None and not isinstance(language, str):
        language = str(language)

    line_orientation, orientation_confidence, bbox_counts = infer_line_orientation(
        collect_text_bboxes(data)
    )
    (
        reading_direction,
        reading_confidence,
        writing_mode,
        column_direction,
    ) = infer_reading_direction(language, line_orientation)
    image_path = find_image(layout_path, image_dir)

    text_direction = {
        "reading_direction": reading_direction,
        "line_orientation": line_orientation,
        "writing_mode": writing_mode,
        "confidence": round(min(reading_confidence, orientation_confidence), 4),
        "source": "metadata.language + text_bbox_aspect_ratio",
    }
    if column_direction is not None:
        text_direction["column_direction"] = column_direction

    if image_path is not None:
        text_direction["image"] = str(image_path)

    data["metadata"] = with_text_direction_inserted(metadata, text_direction)

    if not dry_run:
        layout_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "file": layout_path.name,
        "language": language,
        "reading_direction": reading_direction,
        "line_orientation": line_orientation,
        "writing_mode": writing_mode,
        "confidence": text_direction["confidence"],
        "image_found": image_path is not None,
        "bbox_counts": bbox_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add inferred text direction metadata to layout JSON files."
    )
    parser.add_argument("--layout-dir", default="newdataset-layout")
    parser.add_argument("--image-dir", default="new-dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir)
    image_dir = Path(args.image_dir)
    files = sorted(layout_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {layout_dir}")

    results = [process_file(path, image_dir, args.dry_run) for path in files]
    missing_images = [item["file"] for item in results if not item["image_found"]]

    direction_counts = Counter(
        (item["reading_direction"], item["line_orientation"]) for item in results
    )
    language_counts = Counter(item["language"] for item in results)

    print(f"Processed {len(results)} files")
    print(f"Dry run: {args.dry_run}")
    print(f"Languages: {dict(language_counts)}")
    print(f"Direction/orientation counts: {dict(direction_counts)}")
    if missing_images:
        print(f"Missing matching images: {missing_images}")

    for item in results:
        print(
            "{file}: language={language}, reading_direction={reading_direction}, "
            "line_orientation={line_orientation}, writing_mode={writing_mode}, "
            "confidence={confidence}, image_found={image_found}, "
            "bbox_counts={bbox_counts}".format(**item)
        )


if __name__ == "__main__":
    main()
