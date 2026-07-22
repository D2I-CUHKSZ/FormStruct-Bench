#!/usr/bin/env bash
set -euo pipefail

ROOT="."
OUT="$ROOT/outputs/srfund_transfer_exploratory/checkpoint_sweep"
LOG_DIR="$OUT/logs"
VLLM_BIN="$ROOT/.venv-vllm/bin/vllm"
VLLM_PYTHON="$ROOT/.venv-vllm/bin/python"
PYTHON_BIN="$ROOT/.venv/bin/python"
BASE_MODEL="/path/to/data/model/Qwen/Qwen3.6-35B-A3B"
SERVE_MODEL="/path/to/bench/Qwen/Qwen3.6-35B-A3B-FormTSR-Hierarchical-v2"
OVERLAY_DIR="/path/to/data/model_overlays/qwen36_formtsr_hierarchical_v2"
ADAPTER_DIR="$ROOT/outputs/qwen36_formtsr_hierarchical_v2_lora"
PORT="${PORT:-8000}"
SERVER_PID=""

mkdir -p "$LOG_DIR" "$OVERLAY_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export OMP_NUM_THREADS=1

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

"$VLLM_PYTHON" scripts/apply_vllm_weight_overlay_patch.py

ensure_overlay() {
  local step="$1"
  local overlay="$OVERLAY_DIR/checkpoint-${step}.safetensors"
  if [[ -f "$overlay" && -f "$overlay.json" ]]; then
    return
  fi
  "$PYTHON_BIN" -u scripts/build_lora_weight_overlay.py \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER_DIR/checkpoint-$step" \
    --output "$overlay" \
    --device cuda:0
}

run_checkpoint() {
  local step="$1"
  local model_name="qwen36_35b_sft_step${step}_dev"
  local served_name="qwen36_step${step}"
  local overlay="$OVERLAY_DIR/checkpoint-${step}.safetensors"
  local rpc_port="$2"
  local expected attempted ready

  expected=$(wc -l < outputs/srfund_transfer_exploratory/splits/dev/index.jsonl)
  if [[ -f "$OUT/per_model_metrics/$model_name.jsonl" ]]; then
    attempted=$(wc -l < "$OUT/per_model_metrics/$model_name.jsonl")
  else
    attempted=0
  fi
  if [[ "$attempted" -eq "$expected" ]]; then
    echo "[$model_name] already attempted $attempted/$expected pages; skipping"
    return
  fi

  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "port ${PORT} already has a vLLM-compatible server" >&2
    exit 1
  fi

  echo "[$model_name] starting from $attempted/$expected attempted pages with overlay $overlay"
  VLLM_WEIGHT_OVERLAY="$overlay" setsid "$VLLM_BIN" serve "$SERVE_MODEL" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name "$served_name" \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-backend mp \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port "$rpc_port" \
    --gpu-memory-utilization 0.95 \
    --max-model-len 24576 \
    --max-num-seqs 4 \
    --mm-processor-cache-gb 0 \
    --disable-log-stats \
    --generation-config vllm \
    >"$LOG_DIR/${model_name}_overlay_server.stdout" \
    2>"$LOG_DIR/${model_name}_overlay_server.stderr" &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$OUT/server.pid"

  ready=0
  for _ in $(seq 1 900); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[$model_name] vLLM server exited before readiness" >&2
      tail -n 160 "$LOG_DIR/${model_name}_overlay_server.stdout" >&2 || true
      tail -n 160 "$LOG_DIR/${model_name}_overlay_server.stderr" >&2 || true
      exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"$OUT/${model_name}_overlay_models.json" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[$model_name] vLLM server was not ready within 1800 seconds" >&2
    exit 1
  fi

  "$PYTHON_BIN" -u -m formtsr_exp.run_main \
    --config configs/srfund_checkpoint_sweep_dev.yaml \
    --models "$model_name" \
    --resume \
    --skip-extra-reports
  attempted=$(wc -l < "$OUT/per_model_metrics/$model_name.jsonl")
  if [[ "$attempted" -ne "$expected" ]]; then
    echo "[$model_name] stopped after only $attempted/$expected attempted pages" >&2
    exit 1
  fi
  stop_server
}

ensure_overlay 100
ensure_overlay 200
run_checkpoint 100 13612
run_checkpoint 200 13613
"$PYTHON_BIN" -u -m srfund_exp.checkpoint_sweep_report
echo "overlay checkpoint sweep completed"
