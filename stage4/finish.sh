#!/bin/zsh
# Runs once both arms have exited: retry any errored rows, merge, validate, and
# upload ONLY if validation passes.  Idempotent; safe to re-run.
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
echo "[$(date +%H:%M:%S)] waiting for runners to exit"
until ! pgrep -f "stage4/run.py --arm" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] runners done; retrying errored rows (transient network) on each arm"
(ulimit -n 4096; python stage4/run.py --arm A --run full_A --chunk-ids stage4/subset_stage1_chunks.txt --retry-failed --workers 32 2>&1 | tail -2)
(ulimit -n 4096; python stage4/run.py --arm B --run full_B --endpoint fifhkwzcjy7zey --chunk-ids stage4/subset_stage1_chunks.txt --retry-failed --workers 20 2>&1 | tail -2)
echo "[$(date +%H:%M:%S)] merge"
python stage4/merge.py
echo "[$(date +%H:%M:%S)] coverage"
python stage4/coverage.py
echo "[$(date +%H:%M:%S)] validate"
if python stage4/validate_final.py; then
  echo "[$(date +%H:%M:%S)] VALIDATION PASSED -> upload"
  python stage4/merge.py --upload
  echo "[$(date +%H:%M:%S)] FINISHED OK"
else
  echo "[$(date +%H:%M:%S)] VALIDATION FAILED -> NOT uploading"; exit 1
fi
