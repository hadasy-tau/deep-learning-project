#!/usr/bin/env bash
# Run 4 on a GPU pod (docs/training_plan_v4.md).  All the data work happened before the pod
# (word scores, plans, gate, and data_v4.sh extracting the audio into panel-hq), so this only
# boots and runs the queue (jobs_v4.py).
#
#   bash src/training/box/pod_v4.sh boot      # deps, model weights, run 4's audio (panel-hq), run 3's results -- in parallel
#   bash src/training/box/pod_v4.sh run       # backup loop + one queue worker per GPU (SLOTS per GPU), until the queue is empty
#   bash src/training/box/pod_v4.sh status    # the queue: done / running / failed, elapsed, cost, remaining
#   bash src/training/box/pod_v4.sh finish    # results.csv, the data curve, last backup, verified file by file
#   bash src/training/box/pod_v4.sh all       # boot -> run -> finish, then stop the pod (STOP_POD=0 to keep it)
#
# The queue starts with the overfit check and the base transcriptions; the lr decision at 360
# minutes (v4_decide.py pick) is a job too, so 720/1440 minutes start the moment it is made.
# More workers per GPU: SLOTS=2 (job peak is ~21 GB of 80); `jobs_v4.py worker --gpu k` can
# also be started by hand while the queue runs, if nvidia-smi shows a GPU underused.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source src/training/box/env.sh

PY=${PY:-python}
PRIOR_REPO=knesset-asr/knesset-committees-v3-results     # run 3: the 5/20/80-minute cells of 23558 and 30752
RESULTS_REPO=${RESULTS_REPO:-knesset-asr/knesset-committees-v4-results}
OUT=src/training/outputs; LOGS=$OUT/logs; mkdir -p "$LOGS" "$OUT/results"
J="$PY src/training/box/jobs_v4.py"
detect_gpus() { local n; n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') || true; echo "${n:-0}"; }
GPUS=${GPUS:-$(detect_gpus)}; [ "${GPUS:-0}" -ge 1 ] 2>/dev/null || GPUS=1
SLOTS=${SLOTS:-1}

say() { echo "[$(date '+%F %T')] $*"; }

stage_boot() {
    local t0=$SECONDS
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    pip install -q -r src/training/requirements.txt > "$LOGS/pip.log" 2>&1 &
    local p_pip=$!
    $PY -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 'not logged in to HuggingFace: run  hf auth login  (typed, never pasted into a chat)')"
    # 14 packs of FLAC clips, decoded back to WAV in parallel: the 44,033 WAVs one by one run at ~5 files/s (box/packs_v4.py)
    $PY src/training/box/packs_v4.py fetch > "$LOGS/boot_audio.log" 2>&1 &
    local p_audio=$!
    $PY -c "from huggingface_hub import snapshot_download as s; s('ivrit-ai/whisper-large-v3'); print('model weights: cached')" > "$LOGS/boot_model.log" 2>&1 &
    local p_model=$!
    # run 3's results for the speakers whose 5/20/80-minute points come from it: own cells, their controls, base caches
    $PY - "$PRIOR_REPO" <<'EOF' > "$LOGS/boot_prior.log" 2>&1 &
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type='dataset', local_dir='src/training/outputs',
                  allow_patterns=['results/*23558*', 'results/*30752*'], max_workers=16)
print('run 3 results: downloaded')
EOF
    local p_prior=$!
    wait $p_pip || { say "pip failed: $LOGS/pip.log"; exit 1; }
    $PY -c "import torch, transformers, peft; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"
    wait $p_model || { say "model download failed: $LOGS/boot_model.log"; exit 1; }
    wait $p_prior || { say "run 3 results failed: $LOGS/boot_prior.log"; exit 1; }
    wait $p_audio || { say "audio failed: $LOGS/boot_audio.log"; exit 1; }
    $PY src/training/materialize.py verify --plan src/training/panel_plan_v4.parquet
    $PY src/training/materialize.py verify --plan src/training/panel_test07_v4.parquet
    $PY src/common.py > /dev/null && $PY src/evaluation/evaluate.py && $PY src/training/run_panel.py --self-check \
        && $PY src/training/box/v4_decide.py --self-check && $J --self-check
    $J build
    say "boot: $(( (SECONDS - t0) / 60 )) min; $GPUS GPU(s) x $SLOTS slot(s)"
}

stage_backup() {
    if pgrep -f "backup.py --repo $RESULTS_REPO" >/dev/null; then say "backup already running"; return; fi
    setsid nohup $PY src/training/backup.py --repo "$RESULTS_REPO" --every 30 \
        --logs "$LOGS/*.log" "$OUT/queue_v4.json" "$OUT/recipe_v4.json" "$OUT/tuning_v4.csv" > "$LOGS/backup.out" 2>&1 < /dev/null &
    say "backup to $RESULTS_REPO every 30 min"
}

stage_run() {
    stage_backup
    local pids=() g k
    for ((g = 0; g < GPUS; g++)); do
        for ((k = 0; k < SLOTS; k++)); do
            $J worker --gpu $g > "$LOGS/worker.gpu$g.$k.log" 2>&1 &
            pids+=($!)
            sleep 5                                  # staggered: model loads do not all hit the disk at once
        done
    done
    say "run: ${#pids[@]} worker(s); progress: bash src/training/box/pod_v4.sh status"
    for p in "${pids[@]}"; do wait "$p" || true; done
    $J status --gpus "$GPUS"
    if $PY -c "import json, sys; J = json.load(open('$OUT/queue_v4.json'))['jobs']; sys.exit(0 if all(j['status'] == 'done' for j in J) else 1)"; then
        say "run: every job done"
    else
        say "run: NOT every job done -- see status above; rerun 'run' to retry (finished work is skipped)"; return 1
    fi
}

stage_finish() {
    $PY src/training/run_panel.py --summary > "$LOGS/summary.log" 2>&1 && tail -n 5 "$LOGS/summary.log"
    $PY src/training/run_panel.py --curve | tee "$LOGS/curve.log"
    $PY src/training/backup.py --repo "$RESULTS_REPO" --once --logs "$LOGS/*.log" "$OUT/queue_v4.json" "$OUT/recipe_v4.json" "$OUT/tuning_v4.csv" "$OUT/curve_v4.csv"
    # verified: every result JSON and adapter file here is in the backup with the same size
    $PY - "$RESULTS_REPO" <<'EOF'
import glob, os, sys
from huggingface_hub import HfApi
repo = sys.argv[1]; remote = {f.path: f.size for f in HfApi().list_repo_tree(repo, repo_type='dataset', recursive=True) if hasattr(f, 'size')}
local = {('results/' + os.path.relpath(p, 'src/training/outputs/results')): os.path.getsize(p) for p in glob.glob('src/training/outputs/results/*.json')}
local.update({('runs/' + os.path.relpath(p, 'src/training/runs')): os.path.getsize(p) for p in glob.glob('src/training/runs/*/adapter_model.safetensors') + glob.glob('src/training/runs/*/train_meta.json')})
bad = [k for k, v in local.items() if remote.get(k) != v]
print(f'backup check: {len(local) - len(bad)}/{len(local)} files match' + (f'; missing or different: {bad[:10]}' if bad else ''))
sys.exit(1 if bad else 0)
EOF
}

stop_pod() {
    [ "${STOP_POD:-1}" = 1 ] || { say "STOP_POD=0: the pod keeps running (and billing)"; return; }
    export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=') && runpodctl stop pod "$RUNPOD_POD_ID"
}

run() { say "== $1"; "stage_$1" 2>&1 | tee -a "$LOGS/pod_v4_$1.log"; }

case "${1:-}" in
    boot|run|finish|backup) run "$1" ;;
    status) $J status --gpus "$GPUS" ;;
    all)
        run boot; run run; run finish && { say "all: done, backup verified -- stopping the pod"; stop_pod; } ;;
    *) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 1 ;;
esac
