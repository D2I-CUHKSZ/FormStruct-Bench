#!/usr/bin/env bash
set -euo pipefail

ROOT="."
OUT="$ROOT/outputs/srfund_transfer_locked400/eval"
LOG_DIR="$OUT/logs"
VLLM_BIN="$ROOT/.venv-vllm/bin/vllm"
VLLM_PYTHON="$ROOT/.venv-vllm/bin/python"
PYTHON_BIN="$ROOT/.venv/bin/python"
SERVE_MODEL="/path/to/bench/Qwen/Qwen3.6-35B-A3B-FormTSR-Hierarchical-v2"
OVERLAY_DIR="/path/to/data/model_overlays/qwen36_formtsr_hierarchical_v2"
PORT="${PORT:-8000}"
SERVER_PID=""

mkdir -p "$LOG_DIR"
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

run_condition() {
  local model_name="$1"
  local served_name="$2"
  local overlay="$3"
  local rpc_port="$4"
  local expected attempted ready

  expected=$(wc -l < outputs/srfund_transfer_locked400/split/index.jsonl)
  if [[ -f "$OUT/per_model_metrics/$model_name.jsonl" ]]; then
    attempted=$(wc -l < "$OUT/per_model_metrics/$model_name.jsonl")
  else
    attempted=0
  fi
  if [[ "$attempted" -eq "$expected" ]]; then
    echo "[$model_name] already attempted $attempted/$expected pages; skipping"
    return
  fi
  if [[ ! -f "$overlay" ]]; then
    echo "missing persistent overlay: $overlay" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "port ${PORT} already has a vLLM-compatible server" >&2
    exit 1
  fi

  echo "[$model_name] resuming locked evaluation with $attempted/$expected attempted"
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
    >"$LOG_DIR/${model_name}_server.stdout" \
    2>"$LOG_DIR/${model_name}_server.stderr" &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$OUT/server.pid"

  ready=0
  for _ in $(seq 1 900); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[$model_name] vLLM server exited before readiness" >&2
      tail -n 160 "$LOG_DIR/${model_name}_server.stdout" >&2 || true
      tail -n 160 "$LOG_DIR/${model_name}_server.stderr" >&2 || true
      exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"$OUT/${model_name}_models.json" 2>/dev/null; then
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
    --config configs/srfund_locked400_qwen36_step100.yaml \
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

run_condition \
  qwen36_35b_pre_sft_locked400 \
  qwen36_locked_pre_sft \
  "$OVERLAY_DIR/pre-sft.safetensors" \
  13712
run_condition \
  qwen36_35b_sft_step100_locked400 \
  qwen36_locked_step100 \
  "$OVERLAY_DIR/checkpoint-100.safetensors" \
  13713
"$PYTHON_BIN" -u -m srfund_exp.locked_transfer_report
echo "locked 400-page transfer evaluation completed"
