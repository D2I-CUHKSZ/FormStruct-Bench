#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/path/to/data/model/Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_6_35b_a3b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

export CUDA_HOME="${CUDA_HOME:-/path/to/data/cuda-12.8}"
export PATH="$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin /usr/bin/g++-13}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-$CUDA_HOME/bin/nvcc}"
export PYTHONPATH="$ROOT_DIR/scripts/sglang_stubs${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT_DIR/.venv/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --enable-multimodal \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.92}" \
  --skip-server-warmup \
  --reasoning-parser qwen3 \
  --context-length "${CONTEXT_LENGTH:-16384}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-32768}" \
  --disable-cuda-graph \
  --attention-backend triton \
  --sampling-backend pytorch
