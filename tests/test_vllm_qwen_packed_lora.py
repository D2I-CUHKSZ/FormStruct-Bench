from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("vllm", reason="requires a patched vLLM installation")


def _layer():
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA,
    )

    layer = object.__new__(MergedColumnParallelLinearWithLoRA)
    layer.n_slices = 4
    layer.base_layer = SimpleNamespace(output_sizes=(2, 2, 4, 4))
    return layer


def test_qwen_qkvz_lora_expands_two_adapter_modules_to_four_slices() -> None:
    layer = _layer()
    qkv_a = torch.randn(3, 5)
    z_a = torch.randn(3, 5)
    qkv_b = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    z_b = torch.randn(4, 3)

    lora_a, lora_b = layer._expand_qkvz_lora_slices(
        [qkv_a, z_a], [qkv_b, z_b]
    )

    assert len(lora_a) == 4
    assert lora_a[0] is qkv_a
    assert lora_a[1] is qkv_a
    assert lora_a[2] is qkv_a
    assert lora_a[3] is z_a
    assert [part.shape for part in lora_b] == [(2, 3), (2, 3), (4, 3), (4, 3)]
    assert torch.equal(torch.cat(lora_b[:3]), qkv_b)
    assert lora_b[3] is z_b


def test_qwen_qkvz_dummy_lora_is_zero_padded_to_fused_shapes() -> None:
    layer = _layer()
    dummy_a = torch.zeros(3, 5)
    dummy_b = torch.zeros(2, 3)

    _, lora_b = layer._expand_qkvz_lora_slices(
        [dummy_a, dummy_a], [dummy_b, dummy_b]
    )

    assert [part.shape for part in lora_b] == [(2, 3), (2, 3), (4, 3), (4, 3)]
    assert all(not torch.count_nonzero(part) for part in lora_b)


def test_qwen_qkvz_nonzero_shape_mismatch_fails_closed() -> None:
    layer = _layer()
    with pytest.raises(ValueError, match="nonzero qkv LoRA"):
        layer._expand_qkvz_lora_slices(
            [torch.zeros(3, 5), torch.zeros(3, 5)],
            [torch.ones(2, 3), torch.zeros(4, 3)],
        )
