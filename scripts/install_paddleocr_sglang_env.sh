#!/usr/bin/env bash
set -euo pipefail

ROOT="."
ENV_DIR="${ENV_DIR:-/path/to/bench/venvs/paddleocr-sglang}"
CUDA_HOME="${CUDA_HOME:-/path/to/data/cuda-12.8}"
TMPDIR="${TMPDIR:-/path/to/bench/tmp}"
UV="${UV:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/path/to/bench/uv-cache}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

mkdir -p "${ENV_DIR%/*}" "${TMPDIR}" "${UV_CACHE_DIR}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  "${UV}" venv --python 3.12 --seed "${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"

if ! ${PYTHON} -m pip --version >/dev/null 2>&1; then
  ${PYTHON} -m ensurepip --upgrade
fi

uv_install() {
  "${UV}" pip install \
    --python "${PYTHON}" \
    --default-index "${PYPI_INDEX}" \
    --link-mode copy \
    "$@"
}

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export TMPDIR
export UV_CACHE_DIR
export UV_LINK_MODE=copy
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"

cd "${ROOT}"

uv_install -U pip setuptools wheel ninja packaging

# PaddleOCR official GPU wheel index for CUDA 12.x.
uv_install \
  --index "${PADDLE_INDEX}" \
  --index-strategy unsafe-best-match \
  paddlepaddle-gpu==3.2.1
uv_install -U "paddleocr[doc-parser]>=3.7.0"

# PaddleX's SGLang plugin metadata pins torch 2.8.0 and sglang 0.5.2.
# Keep xformers no-deps so pip does not pull a CUDA 13 torch build.
uv_install --torch-backend cu126 einops "sglang[all]==0.5.2" torch==2.8.0 transformers
uv_install --torch-backend cu126 "xformers==0.0.32.post2" --no-deps
uv_install backports.lzma

# Build against the current torch/CUDA environment instead of an isolated build env.
uv_install --torch-backend cu126 "flash-attn==2.8.2" --no-build-isolation

${PYTHON} - <<'PY'
from pathlib import Path
import site

sitecustomize = '''"""Local startup fixes for the isolated PaddleOCR-VL environment."""

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

import importlib.metadata as md
import subprocess
import sys

for name in [
    "torch",
    "torchvision",
    "sglang",
    "flash-attn",
    "xformers",
    "paddlepaddle-gpu",
    "paddleocr",
    "paddlex",
]:
    try:
        print(f"{name} {md.version(name)}")
    except Exception as exc:
        print(f"{name} MISSING {exc}")

from paddlex.utils import deps

print("genai-client", deps.is_genai_client_plugin_available())
print("sglang-server", deps.is_genai_engine_plugin_available("sglang-server"))

subprocess.check_call(
    [
        sys.executable,
        "-c",
        "import lzma; import torchvision; from transformers import AutoProcessor; print('lzma/torchvision/AutoProcessor ok')",
    ]
)
PY
