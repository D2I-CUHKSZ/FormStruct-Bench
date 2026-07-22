#!/usr/bin/env bash
set -euo pipefail

ROOT="."
cd "$ROOT"

SRFUND_PID_FILE="outputs/srfund_qwen_transfer_benchmark/queue.pid"
SRFUND_STATUS="outputs/srfund_qwen_transfer_benchmark/status/qwen36_35b_a3b_base.json"

echo "[$(date -u +%FT%TZ)] waiting for SRFUND four-condition queue"
while ! rg -q '"state": "complete"' "$SRFUND_STATUS" 2>/dev/null; do
  if [[ -f "$SRFUND_PID_FILE" ]]; then
    srfund_pid="$(<"$SRFUND_PID_FILE")"
    if ! kill -0 "$srfund_pid" 2>/dev/null; then
      echo "SRFUND queue exited before final status became complete" >&2
      exit 1
    fi
  fi
  sleep 30
done

if [[ -f "$SRFUND_PID_FILE" ]]; then
  srfund_pid="$(<"$SRFUND_PID_FILE")"
  while kill -0 "$srfund_pid" 2>/dev/null; do
    sleep 5
  done
fi
echo "[$(date -u +%FT%TZ)] generating transfer metrics and figure"
.venv/bin/python -m srfund_exp.transfer_figure \
  --config configs/sft_transfer_figure.yaml

echo "[$(date -u +%FT%TZ)] complete"
