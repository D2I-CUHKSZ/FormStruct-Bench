#!/usr/bin/env bash
set -u

cd .

KIMI_MODEL="kimi_vl_a3b_vllm_vlm"
GEMMA_MODEL="gemma4_26b_vllm_vlm"
TOTAL="${TOTAL:-7000}"
LOG_DIR="outputs/main_exp/logs"
LOG_FILE="${LOG_DIR}/gemma4_after_kimi_watcher.log"
mkdir -p "${LOG_DIR}" "outputs/main_exp/errors"

while true; do
  pred=$(find "outputs/main_exp/pred/${KIMI_MODEL}" -type f 2>/dev/null | wc -l)
  errors=0
  if [ -f "outputs/main_exp/errors/${KIMI_MODEL}.jsonl" ]; then
    errors=$(wc -l < "outputs/main_exp/errors/${KIMI_MODEL}.jsonl")
  fi
  done_count=$((pred + errors))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] kimi pred=${pred} errors=${errors} done=${done_count}/${TOTAL}" | tee -a "${LOG_FILE}"

  if [ "${done_count}" -ge "${TOTAL}" ]; then
    break
  fi
  sleep 60
done

while tmux has-session -t formtsr_kimi_vllm 2>/dev/null; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] kimi complete; waiting for formtsr_kimi_vllm session to exit" | tee -a "${LOG_FILE}"
  sleep 30
done

if tmux has-session -t formtsr_gemma4_vllm 2>/dev/null; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] formtsr_gemma4_vllm already running" | tee -a "${LOG_FILE}"
  exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting Gemma4 vLLM full run" | tee -a "${LOG_FILE}"
tmux new-session -d -s formtsr_gemma4_vllm \
  'cd .; MODEL_NAME=gemma4_26b_vllm_vlm MODEL_PROCESS_PATTERN=gemma-4-26B-A4B-it RUN_TIMEOUT_SECONDS=86400 bash scripts/run_caprl_vllm_loop.sh'
