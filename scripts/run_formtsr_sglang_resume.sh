#!/usr/bin/env bash
set -euo pipefail

cd .
mkdir -p outputs/main_exp/logs

exec .venv/bin/python -m formtsr_exp.run_main \
  --config configs/main_experiment.yaml \
  --models Qwen3.6-35B-A3B \
  --resume \
  "$@"
