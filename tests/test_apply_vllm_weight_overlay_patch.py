from pathlib import Path

from scripts.apply_vllm_weight_overlay_patch import MARKER, ORIGINAL, apply_patch


def test_vllm_overlay_patch_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "qwen3_5.py"
    target.write_text(f"prefix\n{ORIGINAL}suffix\n", encoding="utf-8")
    assert apply_patch(target) is True
    assert MARKER in target.read_text(encoding="utf-8")
    assert apply_patch(target) is False
