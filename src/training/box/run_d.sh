#!/bin/bash
# Run D: the missing cell of the 2x2 -- semi-verbatim targets with dev-loss selection -- after the next_runs chain ($1).
cd /workspace/deep-learning-project && source /workspace/env.sh
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date)] waiting for pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date)] run D: verbatim targets + dev-loss selection, lr 3e-4"
python src/training/run_panel.py --arm B --budgets 80 --seeds 0 --lrs 3e-4 --targets verbatim --eval-batch 64 --control-folds 2 --control-budget 80
echo "[$(date)] run D exit $?; summary + backup"
python src/training/run_panel.py --summary > /dev/null
python src/training/backup.py --repo Dolevabudi/knesset-committees-adapters --once --logs /workspace/deep-learning-project/src/training/outputs/run_next.log /workspace/deep-learning-project/src/training/outputs/run_d.log
echo "[$(date)] RUN D DONE"
