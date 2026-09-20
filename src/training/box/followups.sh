#!/bin/bash
# Follow-ups to the seed-0 run (docs/training_run2.md): wait for it, then
#   1. the site axis: LoRA on the encoder only, 80 min, all speakers, + its control
#   2. the lr axis: 1e-4 and 3e-4 at 80 min, all speakers, + their controls
cd /workspace/deep-learning-project && source /workspace/env.sh
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date)] waiting for the seed-0 run (pid $1)"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date)] seed-0 run finished; starting follow-up 1 (encoder-only site)"
python src/training/run_panel.py --arm B --sites encoder --budgets 80 --seeds 0 --eval-batch 64 --control-folds 2 --control-budget 80
echo "[$(date)] follow-up 1 exit $?; starting follow-up 2 (lr 1e-4, 3e-4)"
python src/training/run_panel.py --arm B --budgets 80 --seeds 0 --lrs 1e-4 3e-4 --eval-batch 64 --control-folds 2 --control-budget 80
echo "[$(date)] follow-up 2 exit $?; summary + backup"
python src/training/run_panel.py --summary > /dev/null
python src/training/backup.py --repo Dolevabudi/knesset-committees-adapters --once --logs /workspace/deep-learning-project/src/training/outputs/run_seed0.log /workspace/deep-learning-project/src/training/outputs/run_followups.log
echo "[$(date)] FOLLOWUPS DONE"
