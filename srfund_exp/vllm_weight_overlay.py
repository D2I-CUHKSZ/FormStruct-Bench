"""Replace selected Hugging Face weights while vLLM loads a model.

The overlay contains fully materialized tensors for the small subset of base
weights touched by LoRA.  This preserves merged-model inference speed without
requiring another full copy of a large checkpoint on disk.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from safetensors import safe_open


LOGGER = logging.getLogger(__name__)
OVERLAY_ENV = "VLLM_WEIGHT_OVERLAY"


def overlay_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    overlay_path: str | Path,
    *,
    require_all: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield model weights, replacing tensors present in ``overlay_path``."""

    path = Path(overlay_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"weight overlay does not exist: {path}")

    with safe_open(path, framework="pt", device="cpu") as overlay:
        overlay_names = set(overlay.keys())
        used: set[str] = set()
        LOGGER.info("Loading %d replacement tensors from %s", len(overlay_names), path)

        for name, loaded_weight in weights:
            if name not in overlay_names:
                yield name, loaded_weight
                continue

            replacement = overlay.get_tensor(name)
            if replacement.shape != loaded_weight.shape:
                raise ValueError(
                    f"overlay shape mismatch for {name}: "
                    f"overlay={tuple(replacement.shape)}, base={tuple(loaded_weight.shape)}"
                )
            if replacement.dtype != loaded_weight.dtype:
                raise ValueError(
                    f"overlay dtype mismatch for {name}: "
                    f"overlay={replacement.dtype}, base={loaded_weight.dtype}"
                )
            used.add(name)
            yield name, replacement

        missing = overlay_names - used
        if require_all and missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(
                f"{len(missing)} overlay tensors were not present in the model; "
                f"first entries: {preview}"
            )


def maybe_overlay_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Apply the overlay selected by ``VLLM_WEIGHT_OVERLAY``, if configured."""

    overlay_path = os.environ.get(OVERLAY_ENV, "").strip()
    if not overlay_path:
        return weights
    return overlay_weights(weights, overlay_path)
