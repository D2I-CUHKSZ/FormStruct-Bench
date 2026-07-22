#!/usr/bin/env python3
"""Materialize only the base-model tensors changed by a PEFT LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
PEFT_PREFIX = "base_model.model."


@dataclass(frozen=True)
class LoRATarget:
    weight_name: str
    module_name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scale: float
    fan_in_fan_out: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for the low-rank matrix products.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Store untouched base tensors for the adapter target set.",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Optional smoke-test limit; zero materializes every target tensor.",
    )
    return parser.parse_args()


def _pattern_value(pattern: dict[str, Any], module_name: str, default: Any) -> Any:
    matches = [key for key in pattern if module_name == key or module_name.endswith(f".{key}")]
    if not matches:
        return default
    return pattern[max(matches, key=len)]


def _weight_name(adapter_key: str) -> tuple[str, str]:
    if not adapter_key.endswith(LORA_A_SUFFIX):
        raise ValueError(f"not a LoRA A key: {adapter_key}")
    module_name = adapter_key.removeprefix(PEFT_PREFIX)[: -len(LORA_A_SUFFIX)]
    return f"{module_name}.weight", module_name


def load_lora_targets(adapter_dir: Path) -> list[LoRATarget]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("use_dora") or config.get("use_bdlora") or config.get("lora_bias"):
        raise ValueError("the overlay builder currently supports vanilla, bias-free LoRA only")

    rank_default = int(config["r"])
    alpha_default = float(config["lora_alpha"])
    rank_pattern = dict(config.get("rank_pattern") or {})
    alpha_pattern = dict(config.get("alpha_pattern") or {})
    use_rslora = bool(config.get("use_rslora"))
    fan_in_fan_out = bool(config.get("fan_in_fan_out"))
    adapter_file = adapter_dir / "adapter_model.safetensors"

    targets: list[LoRATarget] = []
    with safe_open(adapter_file, framework="pt", device="cpu") as adapter:
        for key in sorted(adapter.keys()):
            if not key.endswith(LORA_A_SUFFIX):
                continue
            weight_name, module_name = _weight_name(key)
            b_key = f"{key[: -len(LORA_A_SUFFIX)]}{LORA_B_SUFFIX}"
            if b_key not in adapter.keys():
                raise KeyError(f"missing LoRA B tensor for {key}")
            rank = int(_pattern_value(rank_pattern, module_name, rank_default))
            alpha = float(_pattern_value(alpha_pattern, module_name, alpha_default))
            scale = alpha / (math.sqrt(rank) if use_rslora else rank)
            lora_a = adapter.get_tensor(key)
            lora_b = adapter.get_tensor(b_key)
            if lora_a.shape[0] != rank or lora_b.shape[1] != rank:
                raise ValueError(
                    f"rank mismatch for {module_name}: config={rank}, "
                    f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
                )
            targets.append(
                LoRATarget(
                    weight_name=weight_name,
                    module_name=module_name,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    scale=scale,
                    fan_in_fan_out=fan_in_fan_out,
                )
            )
    return targets


@torch.inference_mode()
def merge_target(base_weight: torch.Tensor, target: LoRATarget, device: str) -> torch.Tensor:
    compute_device = torch.device(device)
    base = base_weight.to(compute_device)
    lora_a = target.lora_a.to(compute_device)
    lora_b = target.lora_b.to(compute_device)
    delta = (lora_b @ lora_a) * target.scale
    if target.fan_in_fan_out:
        delta = delta.transpose(0, 1)
    if delta.shape != base.shape:
        raise ValueError(
            f"delta shape mismatch for {target.weight_name}: "
            f"delta={tuple(delta.shape)}, base={tuple(base.shape)}"
        )
    merged = base.clone()
    merged.add_(delta.to(base.dtype))
    if not torch.isfinite(merged).all():
        raise ValueError(f"non-finite merged weight: {target.weight_name}")
    return merged.cpu().contiguous()


def build_overlay(
    base_model: Path,
    adapter_dir: Path,
    *,
    device: str,
    max_tensors: int = 0,
    base_only: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index = json.loads((base_model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    targets = load_lora_targets(adapter_dir)
    if max_tensors > 0:
        targets = targets[:max_tensors]

    missing = [target.weight_name for target in targets if target.weight_name not in weight_map]
    if missing:
        raise KeyError(f"{len(missing)} adapter targets are absent from the base model: {missing[:5]}")

    by_shard: dict[str, list[LoRATarget]] = {}
    for target in targets:
        by_shard.setdefault(weight_map[target.weight_name], []).append(target)

    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_targets in sorted(by_shard.items()):
        with safe_open(base_model / shard_name, framework="pt", device="cpu") as shard:
            for target in shard_targets:
                base_weight = shard.get_tensor(target.weight_name)
                tensors[target.weight_name] = (
                    base_weight.contiguous()
                    if base_only
                    else merge_target(base_weight, target, device)
                )

    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    manifest = {
        "format": "formtsr-vllm-weight-overlay-v1",
        "base_model": str(base_model),
        "adapter": str(adapter_dir),
        "mode": "base_targets" if base_only else "merged_lora_targets",
        "device": device,
        "target_count": len(tensors),
        "tensor_bytes": total_bytes,
        "weight_names": sorted(tensors),
    }
    return tensors, manifest


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".json")
    if output.exists() and not args.force:
        raise FileExistsError(f"overlay already exists (use --force to replace): {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tensors, manifest = build_overlay(
        base_model,
        adapter_dir,
        device=args.device,
        max_tensors=args.max_tensors,
        base_only=args.base_only,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_file(
        tensors,
        temporary,
        metadata={
            "format": str(manifest["format"]),
            "base_model": str(base_model),
            "adapter": str(adapter_dir),
            "mode": str(manifest["mode"]),
        },
    )
    os.replace(temporary, output)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "weight_names"}, indent=2))


if __name__ == "__main__":
    main()
