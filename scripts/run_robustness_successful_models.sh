#!/usr/bin/env bash
set -uo pipefail

cd .

MODELS="${MODELS:-Qwen3.6-35B-A3B,qwen3_5_9b_sglang_vlm,glm4_6v_flash_sglang_vlm,caprl_internvl3_5_8b_vllm_vlm,deepseek_vl2_vllm_vlm,kimi_vl_a3b_vllm_vlm,step3_vl_10b_vllm_vlm}"
CONFIG="${CONFIG:-configs/main_experiment.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/robustness_exp}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] building robustness indexes"
.venv/bin/python -m formtsr_exp.build_robustness_index \
  --clean-data-root ./FormTSR/datasets \
  --augment-root ./FormTSR/dataset-augment \
  --out-root "${OUT_ROOT}" \
  --no-meta-summary

IFS=',' read -r -a MODEL_ARRAY <<< "${MODELS}"

for MODEL in "${MODEL_ARRAY[@]}"; do
  MODEL="$(echo "${MODEL}" | xargs)"
  if [[ -z "${MODEL}" ]]; then
    continue
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] robustness degraded: ${MODEL}"
  if ! .venv/bin/python -m formtsr_exp.run_main \
    --config "${CONFIG}" \
    --index "${OUT_ROOT}/robustness_degraded_index.jsonl" \
    --out-dir "${OUT_ROOT}/degraded" \
    --models "${MODEL}" \
    --resume \
    --rerun-invalid \
    --skip-extra-reports; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] degraded failed for ${MODEL}; continuing with next model" >&2
  fi

done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] robustness inference complete; generate report_latest with the current reporters"
