#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into Qwen3.5 for portable serving."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument(
        "--spill-dir",
        default="",
        help="Optional second filesystem for a prefix of model shards.",
    )
    parser.add_argument(
        "--spill-max-size",
        default="0GB",
        help="Maximum bytes written to --spill-dir (for example 31GB).",
    )
    return parser.parse_args()


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", value, re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid size: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    decimal = {"b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12}
    binary = {"kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}
    multiplier = decimal.get(unit, binary.get(unit))
    if multiplier is None:
        raise ValueError(f"invalid size unit: {unit!r}")
    return int(amount * multiplier)


def save_with_optional_spill(
    model: Any,
    output: Path,
    *,
    max_shard_size: str,
    spill_dir: Path | None,
    spill_max_bytes: int,
) -> dict[str, Any]:
    if spill_dir is None or spill_max_bytes <= 0:
        model.save_pretrained(
            str(output),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        return {"spill_dir": "", "spill_max_bytes": 0, "spilled_bytes": 0, "spilled_shards": []}

    if spill_dir.exists() and any(spill_dir.iterdir()):
        raise FileExistsError(f"spill directory is not empty: {spill_dir}")
    spill_dir.mkdir(parents=True, exist_ok=True)

    import transformers.modeling_utils as modeling_utils

    original_safe_save_file = modeling_utils.safe_save_file
    spilled_bytes = 0
    spilled_shards: list[str] = []

    def routed_safe_save_file(
        tensors: dict[str, Any],
        filename: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        nonlocal spilled_bytes
        shard_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        destination = Path(filename)
        if spilled_bytes + shard_bytes <= spill_max_bytes:
            spill_path = spill_dir / destination.name
            original_safe_save_file(tensors, str(spill_path), metadata=metadata)
            destination.symlink_to(spill_path)
            spilled_bytes += shard_bytes
            spilled_shards.append(destination.name)
            return
        original_safe_save_file(tensors, filename, metadata=metadata)

    modeling_utils.safe_save_file = routed_safe_save_file
    try:
        model.save_pretrained(
            str(output),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
    finally:
        modeling_utils.safe_save_file = original_safe_save_file

    return {
        "spill_dir": str(spill_dir),
        "spill_max_bytes": spill_max_bytes,
        "spilled_bytes": spilled_bytes,
        "spilled_shards": spilled_shards,
    }


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = Path(args.base_model).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    spill_dir = Path(args.spill_dir).expanduser().resolve() if args.spill_dir else None
    spill_max_bytes = parse_size(args.spill_max_size)
    output.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(base),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tuned = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    merged = tuned.merge_and_unload(safe_merge=True)
    spill_summary = save_with_optional_spill(
        merged,
        output,
        max_shard_size=args.max_shard_size,
        spill_dir=spill_dir,
        spill_max_bytes=spill_max_bytes,
    )
    # Keep the base tokenizer/processor metadata verbatim.  Transformers 5.x
    # may serialize ``tokenizer_class=TokenizersBackend``, which older vLLM
    # releases cannot import even though the underlying tokenizer is valid.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ):
        source = base / name
        if source.exists():
            shutil.copy2(source, output / name)
    summary = {
        "base_model": str(base),
        "adapter": str(adapter),
        "output": str(output),
        "max_shard_size": args.max_shard_size,
        "dtype": "bfloat16",
        "merge": "PeftModel.merge_and_unload(safe_merge=True)",
        **spill_summary,
    }
    (output / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
