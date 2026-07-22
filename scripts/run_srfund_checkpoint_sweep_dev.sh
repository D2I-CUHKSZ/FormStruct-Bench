#!/usr/bin/env bash
set -euo pipefail

ROOT="."
OUT="$ROOT/outputs/srfund_transfer_exploratory/checkpoint_sweep"
LOG_DIR="$OUT/logs"
VLLM_BIN="$ROOT/.venv-vllm/bin/vllm"
PYTHON_BIN="$ROOT/.venv/bin/python"
BASE_MODEL="/path/to/data/model/Qwen/Qwen3.6-35B-A3B"
FINAL_MODEL="/path/to/bench/Qwen/Qwen3.6-35B-A3B-FormTSR-Hierarchical-v2"
TEMP_MODEL="/tmp/qwen36_srfund_sweep_merged"
TEMP_SPILL="/dev/shm/qwen36_srfund_sweep_spill"
PORT="${PORT:-8000}"
SERVER_PID=""

mkdir -p "$LOG_DIR"
cd "$ROOT"

clear_temporary_model() {
  rm -rf "$TEMP_MODEL" "$TEMP_SPILL"
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  stop_server
  clear_temporary_model
}
trap cleanup EXIT INT TERM

if [[ ! -f outputs/srfund_transfer_exploratory/splits/dev/index.jsonl ]]; then
  "$PYTHON_BIN" -m srfund_exp.build_transfer_splits
fi

if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "port ${PORT} already has a vLLM-compatible server" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export OMP_NUM_THREADS=1

run_model() {
  local model_name="$1"
  local served_name="$2"
  local model_path="$3"
  local rpc_port="$4"
  local expected
  local completed
  expected=$(wc -l < outputs/srfund_transfer_exploratory/splits/dev/index.jsonl)
  if [[ -d "$OUT/pred/$model_name" ]]; then
    completed=$(find "$OUT/pred/$model_name" -maxdepth 1 -type f -name '*.json' | wc -l)
  else
    completed=0
  fi
  if [[ "$completed" -eq "$expected" ]]; then
    echo "[$model_name] already has $completed/$expected valid predictions; skipping"
    return
  fi

  echo "[$model_name] serving $model_path"
  setsid "$VLLM_BIN" serve "$model_path" \
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

  local ready=0
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
    --config configs/srfund_checkpoint_sweep_dev.yaml \
    --models "$model_name" \
    --resume \
    --rerun-invalid \
    --skip-extra-reports
  stop_server
}

merge_checkpoint() {
  local step="$1"
  clear_temporary_model
  echo "[step-$step] merging checkpoint to temporary storage"
  "$PYTHON_BIN" -u scripts/merge_qwen35_lora.py \
    --base-model "$BASE_MODEL" \
    --adapter "$ROOT/outputs/qwen36_formtsr_hierarchical_v2_lora/checkpoint-$step" \
    --output "$TEMP_MODEL" \
    --max-shard-size 4GB \
    --spill-dir "$TEMP_SPILL" \
    --spill-max-size 45GB \
    >"$LOG_DIR/step${step}_merge.log" 2>&1
}

# Run the readily available pair first so a truthful preliminary trend becomes
# available before the two intermediate checkpoints finish merging.
run_model qwen36_35b_pre_sft_dev qwen36_sweep_base "$BASE_MODEL" 13610
run_model qwen36_35b_sft_final_dev qwen36_final "$FINAL_MODEL" 13611
"$PYTHON_BIN" -u -m srfund_exp.checkpoint_sweep_report --allow-partial

merge_checkpoint 100
run_model qwen36_35b_sft_step100_dev qwen36_step100 "$TEMP_MODEL" 13612
clear_temporary_model

merge_checkpoint 200
run_model qwen36_35b_sft_step200_dev qwen36_step200 "$TEMP_MODEL" 13613
clear_temporary_model

"$PYTHON_BIN" -u -m srfund_exp.checkpoint_sweep_report
echo "checkpoint sweep completed"
