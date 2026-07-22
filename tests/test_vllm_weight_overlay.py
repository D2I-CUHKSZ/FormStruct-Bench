from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.build_lora_weight_overlay import (
    LoRATarget,
    _pattern_value,
    _weight_name,
    merge_target,
)
from srfund_exp.vllm_weight_overlay import overlay_weights


def test_adapter_key_maps_to_hugging_face_weight() -> None:
    key = "base_model.model.model.language_model.layers.2.self_attn.q_proj.lora_A.weight"
    assert _weight_name(key) == (
        "model.language_model.layers.2.self_attn.q_proj.weight",
        "model.language_model.layers.2.self_attn.q_proj",
    )


def test_pattern_value_prefers_longest_suffix() -> None:
    pattern = {"q_proj": 8, "layers.2.self_attn.q_proj": 16}
    assert _pattern_value(pattern, "model.layers.2.self_attn.q_proj", 4) == 16
    assert _pattern_value(pattern, "model.layers.3.self_attn.q_proj", 4) == 8
    assert _pattern_value(pattern, "model.layers.3.self_attn.k_proj", 4) == 4


def test_merge_target_matches_vanilla_lora_equation() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    lora_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    lora_b = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    target = LoRATarget("layer.weight", "layer", lora_a, lora_b, 0.5, False)
    actual = merge_target(base, target, "cpu")
    assert torch.equal(actual, base + (lora_b @ lora_a) * 0.5)


def test_overlay_replaces_matching_tensor_and_requires_complete_use(tmp_path) -> None:
    path = tmp_path / "overlay.safetensors"
    replacement = torch.full((2, 3), 7, dtype=torch.bfloat16)
    save_file({"a.weight": replacement}, path)
    base = [
        ("a.weight", torch.zeros((2, 3), dtype=torch.bfloat16)),
        ("b.weight", torch.ones((1,), dtype=torch.bfloat16)),
    ]

    rows = list(overlay_weights(base, path))
    assert torch.equal(rows[0][1], replacement)
    assert torch.equal(rows[1][1], base[1][1])

    with pytest.raises(ValueError, match="not present"):
        list(overlay_weights([], path))


def test_overlay_rejects_shape_mismatch(tmp_path) -> None:
    path = tmp_path / "overlay.safetensors"
    save_file({"a.weight": torch.zeros((2, 3), dtype=torch.bfloat16)}, path)
    with pytest.raises(ValueError, match="shape mismatch"):
        list(
            overlay_weights(
                [("a.weight", torch.zeros((3, 2), dtype=torch.bfloat16))], path
            )
        )
