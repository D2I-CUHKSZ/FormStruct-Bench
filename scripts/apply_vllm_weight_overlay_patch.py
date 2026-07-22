#!/usr/bin/env python3
"""Idempotently enable FormTSR weight overlays in vLLM's Qwen3.5 loader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MARKER = "# FormTSR checkpoint sweeps store only the tensors changed by LoRA."
ORIGINAL = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
PATCHED = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FormTSR checkpoint sweeps store only the tensors changed by LoRA.
        # Keep this opt-in so ordinary vLLM model loading remains untouched.
        from srfund_exp.vllm_weight_overlay import maybe_overlay_weights

        weights = maybe_overlay_weights(weights)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''


def default_target() -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return (
        Path(sys.prefix)
        / "lib"
        / version
        / "site-packages"
        / "vllm"
        / "model_executor"
        / "models"
        / "qwen3_5.py"
    )


def apply_patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    occurrences = text.count(ORIGINAL)
    if occurrences != 1:
        raise RuntimeError(
            f"expected one Qwen3.5 multimodal load_weights block in {path}, found {occurrences}"
        )
    path.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=default_target())
    args = parser.parse_args()
    changed = apply_patch(args.target)
    print(f"{'patched' if changed else 'already patched'}: {args.target}")


if __name__ == "__main__":
    main()
