#!/bin/bash
# A record of the 2026-09-20 session.  --select forgiven no longer exists (the forgiven-shared
# count is withdrawn, docs/personalization_research.md § 1.5): runs A and B cannot be rerun as written.
# docs/archive/training_next.md items 1, 2(A2 alone) and 4: wait for the lr sweep chain ($1) and the
# train/dev transcription ($2), build the semi-verbatim targets, then three runs, each with
# its two-fold control, then the summary and a backup.
cd /workspace/deep-learning-project && source /workspace/env.sh
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[$(date)] waiting for pids $1 $2"
while kill -0 "$1" 2>/dev/null || kill -0 "$2" 2>/dev/null; do sleep 60; done
echo "[$(date)] building targets"
python src/training/run_panel.py --build-targets 2>&1 | sed 's/^/targets: /'
echo "[$(date)] run A: verbatim targets + selection on dev forgiven WER, lr 3e-4"
python src/training/run_panel.py --arm B --budgets 80 --seeds 0 --lrs 3e-4 --targets verbatim --select forgiven --eval-batch 64 --control-folds 2 --control-budget 80
echo "[$(date)] run A exit $?; run B: protocol targets + selection on dev forgiven WER, lr 3e-4"
python src/training/run_panel.py --arm B --budgets 80 --seeds 0 --lrs 3e-4 --select forgiven --eval-batch 64 --control-folds 2 --control-budget 80
echo "[$(date)] run B exit $?; run C: decoder-MLP site, lr 3e-4"
python src/training/run_panel.py --arm B --sites decoder_mlp --budgets 80 --seeds 0 --lrs 3e-4 --eval-batch 64 --control-folds 2 --control-budget 80
echo "[$(date)] run C exit $?; summary + backup"
python src/training/run_panel.py --summary > /dev/null
python src/training/backup.py --repo knesset-asr/knesset-committees-adapters --once --logs /workspace/deep-learning-project/src/training/outputs/run_seed0.log /workspace/deep-learning-project/src/training/outputs/run_followups.log /workspace/deep-learning-project/src/training/outputs/run_next.log
echo "[$(date)] NEXT RUNS DONE"
