#!/usr/bin/env bash
set -euo pipefail

ROOT="."
ENV_DIR="${ENV_DIR:-/path/to/bench/venvs/vllm-gemma4}"
TMPDIR="${TMPDIR:-/path/to/bench/tmp}"
UV="${UV:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/path/to/bench/uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/path/to/bench/uv-python}"
VLLM_PYTHON_VERSION="${VLLM_PYTHON_VERSION:-3.12.13}"
GEMMA4_MODEL="${GEMMA4_MODEL:-/path/to/bench/google/gemma-4-26B-A4B-it}"

mkdir -p "${ENV_DIR%/*}" "${TMPDIR}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  "${UV}" venv --python 3.12 --seed "${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"

export TMPDIR
export UV_CACHE_DIR
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR

cd "${ROOT}"

"${UV}" python install "${VLLM_PYTHON_VERSION}"

"${UV}" pip install \
  --python "${PYTHON}" \
  --link-mode copy \
  -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

"${UV}" pip install \
  --python "${PYTHON}" \
  --link-mode copy \
  backports.lzma

"${PYTHON}" - <<PY
import importlib.metadata as md
from pathlib import Path
import site

sitecustomize = '''"""Local startup fixes for the isolated Gemma4 vLLM environment."""

from __future__ import annotations

import sys


try:
    import lzma  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name == "_lzma":
        try:
            import backports.lzma as backports_lzma
        except Exception:
            raise
        sys.modules["lzma"] = backports_lzma
    else:
        raise
'''

for site_dir in site.getsitepackages():
    site_path = Path(site_dir)
    if site_path.name == "site-packages":
        (site_path / "sitecustomize.py").write_text(sitecustomize, encoding="utf-8")

for name in ["vllm", "torch", "torchvision", "transformers"]:
    try:
        print(f"{name} {md.version(name)}")
    except Exception as exc:
        print(f"{name} MISSING {exc}")

model_path = Path(${GEMMA4_MODEL@Q})
if not model_path.exists():
    raise SystemExit(f"missing Gemma4 model path: {model_path}")

from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print("gemma4 config", type(cfg).__name__, getattr(cfg, "model_type", None))

shared_libs = sorted(Path(${UV_PYTHON_INSTALL_DIR@Q}).glob(f"**/libpython3.12.so*"))
if not shared_libs:
    raise SystemExit(
        "missing libpython3.12.so from uv Python install; set VLLM_PYTHON_LIB_DIR "
        "to a directory containing libpython3.12.so.1.0"
    )
print("shared python lib", shared_libs[0])
PY
