from __future__ import annotations

import random
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png"}


def make_sample_id(template_name: str, instance_id: str) -> str:
    return f"{template_name}__{instance_id}"


def scan_dataset(
    data_root: Path,
    *,
    templates: set[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    data_root = data_root.expanduser()
    rows: list[dict[str, Any]] = []
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    for template_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        template_name = template_dir.name
        if templates and template_name not in templates:
            continue
        for instance_dir in sorted(p for p in template_dir.iterdir() if p.is_dir()):
            instance_id = instance_dir.name
            label_path = instance_dir / "answer.json"
            expected_image = instance_dir / f"{template_name}-{instance_id}.png"
            image_path = expected_image if expected_image.exists() else None
            if image_path is None:
                pngs = sorted(p for p in instance_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                if len(pngs) == 1:
                    image_path = pngs[0]
            if image_path is None or not label_path.exists():
                continue
            rows.append(
                {
                    "sample_id": make_sample_id(template_name, instance_id),
                    "template_name": template_name,
                    "instance_id": instance_id,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def select_samples(
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
    templates: set[str] | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if not templates or row["template_name"] in templates]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    return selected
