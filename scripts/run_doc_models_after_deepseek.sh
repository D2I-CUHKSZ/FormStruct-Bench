#!/usr/bin/env bash
set -u

cd .

INDEX_PATH="${INDEX_PATH:-outputs/main_exp/dataset_index.jsonl}"
TOTAL="${TOTAL:-$(wc -l < "${INDEX_PATH}" 2>/dev/null || echo 0)}"
LOG_DIR="outputs/main_exp/logs"
LOG_FILE="${LOG_DIR}/doc_models_after_deepseek.log"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek_vl2_vllm_vlm}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

mkdir -p "${LOG_DIR}"

count_done() {
  local model_name="$1"
  local pred_dir="outputs/main_exp/pred/${model_name}"
  local error_log="outputs/main_exp/errors/${model_name}.jsonl"
  local pred=0
  local errors=0
  if [ -d "${pred_dir}" ]; then
    pred=$(find "${pred_dir}" -type f 2>/dev/null | wc -l)
  fi
  if [ -f "${error_log}" ]; then
    errors=$(wc -l < "${error_log}")
  fi
  echo $((pred + errors))
}

start_model_session() {
  local session_name="$1"
  local model_name="$2"
  local process_pattern="$3"
  local timeout_seconds="${4:-172800}"
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${session_name} already running" | tee -a "${LOG_FILE}"
    return
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${model_name} in ${session_name}" | tee -a "${LOG_FILE}"
  tmux new-session -d -s "${session_name}" \
    "cd .; MODEL_NAME=${model_name} MODEL_PROCESS_PATTERN=${process_pattern} RUN_TIMEOUT_SECONDS=${timeout_seconds} SKIP_EXTRA_REPORTS=1 bash scripts/run_caprl_vllm_loop.sh"
}

wait_model_done() {
  local model_name="$1"
  local label="$2"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${label}" | tee -a "${LOG_FILE}"
  while true; do
    local done_count
    done_count=$(count_done "${model_name}")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${model_name} done=${done_count}/${TOTAL}" | tee -a "${LOG_FILE}"
    if [ "${done_count}" -ge "${TOTAL}" ]; then
      break
    fi
    sleep "${SLEEP_SECONDS}"
  done
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${DEEPSEEK_MODEL}; total=${TOTAL}" | tee -a "${LOG_FILE}"
while true; do
  done_count=$(count_done "${DEEPSEEK_MODEL}")
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${DEEPSEEK_MODEL} done=${done_count}/${TOTAL}" | tee -a "${LOG_FILE}"
  if [ "${done_count}" -ge "${TOTAL}" ]; then
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

start_model_session "formtsr_paddleocr_vl" "paddleocr_vl_1_6_pipeline_sglang" "PaddleOCR-VL|paddlex|sglang" "172800"
wait_model_done "paddleocr_vl_1_6_pipeline_sglang" "PaddleOCR-VL"

start_model_session "formtsr_unlimited_ocr" "unlimited_ocr_hf_vlm" "unlimited_ocr_hf_vlm"
wait_model_done "unlimited_ocr_hf_vlm" "UnlimitedOCR"

start_model_session "formtsr_mineru2_5_pro" "mineru2_5_pro_hf_vlm" "mineru2_5_pro_hf_vlm"
wait_model_done "mineru2_5_pro_hf_vlm" "MinerU2.5-Pro"

if [ -x "./.venv-vllm/bin/vllm" ]; then
  start_model_session "formtsr_gemma4_vllm" "gemma4_26b_vllm_vlm" "gemma-4-26B-A4B-it" "172800"
  wait_model_done "gemma4_26b_vllm_vlm" "Gemma4"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gemma4_26b_vllm_vlm not started: ./.venv-vllm/bin/vllm is missing" | tee -a "${LOG_FILE}"
fi
