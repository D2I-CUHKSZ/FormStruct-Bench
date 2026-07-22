#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-.}"
cd "$ROOT_DIR"

PADDLE_MODEL="${PADDLE_MODEL:-paddleocr_vl_1_6_pipeline_sglang}"
GEMMA_MODEL="${GEMMA_MODEL:-gemma4_26b_vllm_vlm}"
OUT_DIR="${OUT_DIR:-outputs/main_exp}"
INDEX_PATH="${INDEX_PATH:-${OUT_DIR}/dataset_index.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-configs/main_experiment.yaml}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs/paddlevl_postprocess_gemma4}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
RUN_GEMMA4_SMOKE="${RUN_GEMMA4_SMOKE:-1}"
RUN_PADDLE_COMPLETION_PASS="${RUN_PADDLE_COMPLETION_PASS:-1}"

mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/orchestrator.log"
exec > >(tee -a "$LOG_PATH") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

count_files() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l
}

tmux_alive() {
  local session="$1"
  tmux has-session -t "$session" 2>/dev/null
}

paddle_tmux_alive() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Eq '^paddlevl_gpu[0-9]+(_[0-9]+)?$'
}

log_status() {
  local pred_count raw_count index_count
  pred_count="$(count_files "${OUT_DIR}/pred/${PADDLE_MODEL}")"
  raw_count="$(count_files "${OUT_DIR}/raw/${PADDLE_MODEL}")"
  index_count="$(wc -l < "$INDEX_PATH")"
  echo "[$(timestamp)] paddle raw=${raw_count} pred=${pred_count}/${index_count}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
}

stop_leftover_paddle_servers() {
  local pgids
  pgids="$(
    ps -eo pgid=,cmd= |
      awk '/paddlex.inference.genai.server --model_name PaddleOCR-VL-1.6-0.9B/ && !/awk/ {print $1}' |
      sort -u
  )"
  if [[ -z "$pgids" ]]; then
    return 0
  fi
  echo "[$(timestamp)] stopping leftover PaddleOCR-VL server process groups: ${pgids//$'\n'/ }"
  while read -r pgid; do
    [[ -n "$pgid" ]] && kill -TERM "-$pgid" 2>/dev/null || true
  done <<< "$pgids"
  sleep 10
  while read -r pgid; do
    [[ -n "$pgid" ]] && kill -KILL "-$pgid" 2>/dev/null || true
  done <<< "$pgids"
}

echo "[$(timestamp)] waiting for PaddleVL shard sessions to finish"
log_status
while paddle_tmux_alive; do
  sleep "$SLEEP_SECONDS"
  log_status
done

echo "[$(timestamp)] PaddleVL shard sessions exited"
stop_leftover_paddle_servers
log_status

index_count="$(wc -l < "$INDEX_PATH")"
pred_count="$(count_files "${OUT_DIR}/pred/${PADDLE_MODEL}")"
if [[ "$pred_count" -lt "$index_count" ]]; then
  echo "[$(timestamp)] PaddleVL predictions incomplete after shards: ${pred_count}/${index_count}"
  if [[ "$RUN_PADDLE_COMPLETION_PASS" == "1" ]]; then
    echo "[$(timestamp)] running single-process PaddleVL completion pass with --resume --rerun-invalid"
    .venv/bin/python -m formtsr_exp.run_main \
      --config "$CONFIG_PATH" \
      --index "$INDEX_PATH" \
      --models "$PADDLE_MODEL" \
      --resume \
      --rerun-invalid \
      --skip-extra-reports
    stop_leftover_paddle_servers
    log_status
    pred_count="$(count_files "${OUT_DIR}/pred/${PADDLE_MODEL}")"
  fi
fi

if [[ "$pred_count" -lt "$index_count" ]]; then
  echo "[$(timestamp)] PaddleVL predictions still incomplete: ${pred_count}/${index_count}; running evaluation with missing predictions recorded, skipping Gemma4 smoke"
  .venv/bin/python -m formtsr_exp.evaluate \
    --index "$INDEX_PATH" \
    --pred-root "${OUT_DIR}/pred" \
    --out "$OUT_DIR" \
    --config "$CONFIG_PATH" \
    --models "$PADDLE_MODEL" \
    --skip-extra-reports
  exit 2
fi

echo "[$(timestamp)] PaddleVL predictions complete; rebuilding metrics"
.venv/bin/python -m formtsr_exp.evaluate \
  --index "$INDEX_PATH" \
  --pred-root "${OUT_DIR}/pred" \
  --out "$OUT_DIR" \
  --config "$CONFIG_PATH" \
  --models "$PADDLE_MODEL" \
  --skip-extra-reports

if [[ "$RUN_GEMMA4_SMOKE" != "1" ]]; then
  echo "[$(timestamp)] RUN_GEMMA4_SMOKE=${RUN_GEMMA4_SMOKE}; skipping Gemma4 smoke"
  exit 0
fi

echo "[$(timestamp)] starting Gemma4 smoke test"
.venv/bin/python -m formtsr_exp.run_main \
  --config "$CONFIG_PATH" \
  --limit 1 \
  --models "$GEMMA_MODEL" \
  --skip-extra-reports

echo "[$(timestamp)] postprocess and Gemma4 smoke finished"
