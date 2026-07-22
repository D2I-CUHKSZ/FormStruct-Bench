#!/usr/bin/env bash
set -u

cd .

MODEL_NAME="${MODEL_NAME:-caprl_internvl3_5_8b_vllm_vlm}"
CONFIG="${CONFIG:-configs/main_experiment.yaml}"
INDEX_PATH="${INDEX_PATH:-outputs/main_exp/dataset_index.jsonl}"
RUNNER_PYTHON="${RUNNER_PYTHON:-.venv/bin/python}"
MODEL_PROCESS_PATTERN="${MODEL_PROCESS_PATTERN:-CapRL-InternVL3.5-8B}"
LOG_DIR="outputs/main_exp/logs"
LOOP_LOG="${LOG_DIR}/${MODEL_NAME}_loop.log"
RUN_LOG="${LOG_DIR}/${MODEL_NAME}_run.log"
PRED_DIR="outputs/main_exp/pred/${MODEL_NAME}"
ERROR_LOG="outputs/main_exp/errors/${MODEL_NAME}.jsonl"
TOTAL="${TOTAL:-$(wc -l < "${INDEX_PATH}" 2>/dev/null || echo 0)}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-86400}"
SKIP_EXTRA_REPORTS="${SKIP_EXTRA_REPORTS:-1}"

extra_report_args=()
case "${SKIP_EXTRA_REPORTS}" in
  1|true|TRUE|yes|YES|on|ON)
    extra_report_args+=(--skip-extra-reports)
    ;;
esac

mkdir -p "${LOG_DIR}" "${PRED_DIR}" "outputs/main_exp/errors"

prev_done=-1
stalls=0

cleanup_server() {
  ps -ef | awk -v pat="${MODEL_PROCESS_PATTERN}" '($0 ~ "vllm serve" && $0 ~ pat) || $0 ~ "VLLM::EngineCore" || $0 ~ "EngineCore_DP" || $0 ~ "ApiServer_" {if ($0 !~ /awk/) print $2}' | xargs -r kill -TERM
  sleep 2
  ps -ef | awk -v pat="${MODEL_PROCESS_PATTERN}" '($0 ~ "vllm serve" && $0 ~ pat) || $0 ~ "VLLM::EngineCore" || $0 ~ "EngineCore_DP" || $0 ~ "ApiServer_" {if ($0 !~ /awk/) print $2}' | xargs -r kill -KILL
}

while true; do
  cleanup_server
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] run_main start" | tee -a "${LOOP_LOG}"
  timeout --signal=TERM "${RUN_TIMEOUT_SECONDS}" "${RUNNER_PYTHON}" -m formtsr_exp.run_main \
    --config "${CONFIG}" \
    --models "${MODEL_NAME}" \
    --resume \
    "${extra_report_args[@]}" >> "${RUN_LOG}" 2>&1 || true
  cleanup_server

  pred=$(find "${PRED_DIR}" -type f 2>/dev/null | wc -l)
  errors=0
  if [ -f "${ERROR_LOG}" ]; then
    errors=$(wc -l < "${ERROR_LOG}")
  fi
  done_count=$((pred + errors))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pred=${pred} errors=${errors} done=${done_count}" | tee -a "${LOOP_LOG}"

  if [ "${done_count}" -ge "${TOTAL}" ]; then
    break
  fi
  if [ "${done_count}" -eq "${prev_done}" ]; then
    stalls=$((stalls + 1))
  else
    stalls=0
  fi
  if [ "${stalls}" -ge 3 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stalled; exiting" | tee -a "${LOOP_LOG}"
    break
  fi

  prev_done="${done_count}"
  sleep 10
done
