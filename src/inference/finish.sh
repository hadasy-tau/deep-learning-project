#!/bin/zsh
# Runs once the lean runner has exited: one more lean pass (it re-sends every
# subset chunk without a successful row, i.e. the errored ones), merge,
# coverage, validate, and upload ONLY if validation passes.  Idempotent.
set -e
cd "$(dirname "$0")/../.."
. .venv/bin/activate
echo "[$(date +%H:%M:%S)] waiting for the runner to exit"
until ! pgrep -f "src/inference/run(_lean)?.py" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] runner done; retry pass over errored rows"
(ulimit -n 4096; python src/inference/run_lean.py --workers-a 32 --workers-b 24 2>&1 | tail -3)
echo "[$(date +%H:%M:%S)] merge"
python src/inference/merge.py
echo "[$(date +%H:%M:%S)] coverage"
python src/inference/coverage.py
echo "[$(date +%H:%M:%S)] validate"
if python src/inference/validate_final.py; then
  echo "[$(date +%H:%M:%S)] VALIDATION PASSED -> upload"
  python src/inference/merge.py --upload
  echo "[$(date +%H:%M:%S)] FINISHED OK"
else
  echo "[$(date +%H:%M:%S)] VALIDATION FAILED -> NOT uploading"; exit 1
fi
