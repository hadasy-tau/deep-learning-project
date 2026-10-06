#!/usr/bin/env bash
# Plan v3 on a GPU pod, stage by stage.  docs/archive/pod_runbook_v3.md is the order, the stop rules
# and what to report; docs/training_plan_v3.md is why.  Every stage is idempotent: a scored
# cell, a finished adapter, a tuning run and a cached base transcription are all skipped on a
# rerun, so after a crash or a preemption run the same stage again.
#
#   bash src/training/box/pod_v3.sh setup      # deps, GPU and HF login checks, model weights, self-checks
#   bash src/training/box/pod_v3.sh data       # panel audio from HF, fill any gap from the corpus, verify both plans
#   bash src/training/box/pod_v3.sh backup     # mirror results to $RESULTS_REPO every 30 min (background)
#   bash src/training/box/pod_v3.sh sanity     # overfit check + one timed tuning run (price the plan)
#   bash src/training/box/pod_v3.sh tune       # stages A, B, C on validation only, each picked by v3_decide.py
#   bash src/training/box/pod_v3.sh base       # base WER on the high-quality test vs the error map (hard stop)
#   bash src/training/box/pod_v3.sh final      # 5 / 20 / 80 min, seed 0, controls at every budget, both tests
#   bash src/training/box/pod_v3.sh seeds      # seeds 1 and 2 at 80 min, own adapters and their controls
#   bash src/training/box/pod_v3.sh summary    # outputs/results.csv, one last backup
#   bash src/training/box/pod_v3.sh all        # sanity -> tune -> base -> final -> seeds -> summary
#
# The panel is the 12 speakers of panel_plan_v2.parquet (30601 deferred, docs/training_plan_v3.md § 1).
#
# Several GPUs: GPUS=N (default: every GPU the pod has).  The work is hundreds of independent
# ~3-minute jobs, so N GPUs finish in about 1/N of the time at the same GPU-hours, i.e. the same
# price.  Jobs are split so that no two processes ever write the same file: tuning by speaker,
# own adapters by speaker group, controls by (seed, budget) after every own adapter is done --
# a control is always trained on ALL the panel's speakers, never on one worker's share.
# Each GPU runs its jobs one after another and logs to outputs/logs/<stage>.gpu<k>.log.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source src/training/box/env.sh

PY=${PY:-python}
V2=src/training/panel_plan_v2.parquet
T07=src/training/panel_test07.parquet
PANEL_REPO=knesset-asr/knesset-committees-panel-hq
RESULTS_REPO=${RESULTS_REPO:-knesset-asr/knesset-committees-v3-results}
TUNE_SPK="30685 23558 30718 30859"
OUT=src/training/outputs; LOGS=$OUT/logs; mkdir -p "$LOGS"
RP="$PY src/training/run_panel.py"
D="$PY src/training/box/v3_decide.py"
STEP="--plan $V2 --max-steps 400 --eval-batch 64"        # step mode, early stopping fixed at 4 validations
detect_gpus() { local n; n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') || true; echo "${n:-0}"; }   # 0 without nvidia-smi
GPUS=${GPUS:-$(detect_gpus)}; [ "${GPUS:-0}" -ge 1 ] 2>/dev/null || GPUS=1

say() { echo "[$(date '+%F %T')] $*"; }

speakers() {   # every speaker of the plan
    $PY -c "import pandas as pd; print(' '.join(str(s) for s in sorted(pd.read_parquet('$V2').speaker_id.unique())))"
}

group() {      # group <i> <n> <items...>: the i-th of n round-robin shares of the items
    local i=$1 n=$2 k=0 out=(); shift 2
    for x in "$@"; do (( k % n == i )) && out+=("$x"); k=$((k + 1)); done
    echo "${out[*]:-}"
}

on_gpus() {    # on_gpus <label> <job>...: job j goes to GPU j mod $GPUS; each GPU runs its jobs in order
    local label=$1; shift
    local jobs=("$@") pids=() g fail=0
    for ((g = 0; g < GPUS && g < ${#jobs[@]}; g++)); do
        local chain="set -e"
        for ((j = g; j < ${#jobs[@]}; j += GPUS)); do chain+="; ${jobs[$j]}"; done
        CUDA_VISIBLE_DEVICES=$g bash -c "$chain" > "$LOGS/$label.gpu$g.log" 2>&1 &
        pids+=($!)
    done
    say "$label: ${#jobs[@]} job(s) on ${#pids[@]} GPU(s); logs $LOGS/$label.gpu*.log"
    for g in "${!pids[@]}"; do
        if wait "${pids[$g]}"; then say "$label: GPU $g done"; else say "$label: GPU $g FAILED -- $LOGS/$label.gpu$g.log"; fail=1; fi
    done
    return $fail
}

stage_setup() {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    pip install -q -r src/training/requirements.txt
    # a RunPod PyTorch template ships torch + torchaudio for its CUDA; requirements must not have replaced them
    $PY -c "import torch, torchaudio; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"
    $PY -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 'not logged in to HuggingFace: run  hf auth login  (typed, never pasted into a chat)')"
    $PY -c "from huggingface_hub import HfApi; HfApi().list_repo_files('$PANEL_REPO', repo_type='dataset'); print('HF read access: OK')"
    # both arms' weights once, here: parallel workers must not race to download them into one cache
    $PY -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('ivrit-ai/whisper-large-v3', 'openai/whisper-large-v3')]; print('model weights: cached')"
    $PY src/common.py >/dev/null && $PY src/evaluation/evaluate.py && $D --self-check
    say "setup: $GPUS GPU(s) will be used"
}

stage_data() {
    $PY src/training/materialize.py download --repo "$PANEL_REPO"
    # only clips not on disk are pulled: nothing if the dataset is complete, otherwise the gap (up to 163 shards, ~1 h)
    $PY src/training/materialize.py extract --plan "$V2" --prefetch 3
    $PY src/training/materialize.py extract --plan "$T07" --prefetch 3
    $PY src/training/materialize.py verify --plan "$V2"
    $PY src/training/materialize.py verify --plan "$T07"
    say "data: both plans verified"
}

stage_backup() {
    if pgrep -f "backup.py --repo $RESULTS_REPO" >/dev/null; then say "backup already running"; return; fi
    # the log pattern is quoted: backup.py expands it on every pass, so per-GPU logs written later are included
    setsid nohup $PY src/training/backup.py --repo "$RESULTS_REPO" --every 30 \
        --logs "$LOGS/*.log" "$OUT/recipe_v3.json" > "$LOGS/backup.out" 2>&1 < /dev/null &
    say "backup to $RESULTS_REPO every 30 min (log: $LOGS/backup.out)"
}

stage_sanity() {
    CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF'
import sys, pandas as pd; sys.path.insert(0, 'src/training'); import train
P = pd.read_parquet('src/training/panel_plan_v2.parquet')
train.overfit_check(P, 'src/training/outputs/panel_audio', speaker=30685, arm='B')
EOF
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES=0 $RP --tune $STEP --speakers 30685 --budgets 80 --seeds 0 --lrs 3e-4    # one of stage A's runs, so it is reused
    say "sanity: one tuning run took $(( (SECONDS - t0) / 60 )) min; the whole plan is about 250 of these in GPU time, split over $GPUS GPU(s)"
}

stage_tune() {
    for st in A B C; do
        local jobs=() s g5 g80
        g5=$($D grid $st 5) || exit 1; g80=$($D grid $st 80) || exit 1
        for s in $TUNE_SPK; do                            # one job per tuning speaker: 4 GPUs run a stage at once
            jobs+=("$RP --tune $STEP --speakers $s --seeds 0 --budgets 5 $g5; $RP --tune $STEP --speakers $s --seeds 0 --budgets 80 $g80")
        done
        on_gpus "tune$st" "${jobs[@]}"
        $D pick $st                                       # after every worker: the rule needs all four speakers
    done
    say "tune: recipe -> $OUT/recipe_v3.json"
}

stage_base() {
    local spk jobs=() i share; spk=$(speakers) || exit 1
    for ((i = 0; i < GPUS; i++)); do
        share=$(group $i $GPUS $spk); [ -n "$share" ] && jobs+=("$D base $share")
    done
    on_gpus base "${jobs[@]}"
    $D base $spk                                          # every speaker in the run, from the caches: the one table
}

own_jobs() {   # own_jobs <seed> <budgets...>: the own-adapter cells, one job per speaker share
    local seed=$1; shift
    local spk i b f; spk=$(speakers) || return 1
    for ((i = 0; i < GPUS; i++)); do
        local share chain=""; share=$(group $i $GPUS $spk); [ -n "$share" ] || continue
        for b in "$@"; do
            f=$($D flags $b) || return 1                  # refuses unless tuning picked A, B and C; never fall back to a default rate
            chain+="${chain:+; }$RP $STEP --seeds $seed --test07-plan $T07 --speakers $share --budgets $b $f"
        done
        printf '%s\n' "$chain"
    done
}

control_job() {   # control_job <seed> <budget>: the 2-fold control over ALL the panel's speakers
    local f spk; f=$($D flags $2) || return 1; spk=$(speakers) || return 1
    echo "$RP $STEP --seeds $1 --test07-plan $T07 --speakers $spk --budgets $2 $f --control-only --control-folds 2 --control-budgets $2"
}

stage_final() {
    # every job line is built through a checked assignment: bash does not stop on an error inside $(...)
    local jobs=() line out c5 c20 c80
    out=$(own_jobs 0 5 20 80) || exit 1
    while IFS= read -r line; do [ -n "$line" ] && jobs+=("$line"); done <<< "$out"
    c5=$(control_job 0 5) || exit 1; c20=$(control_job 0 20) || exit 1; c80=$(control_job 0 80) || exit 1
    on_gpus final_own "${jobs[@]}"
    # controls only now: every speaker's base and arm-A caches exist, so no two jobs write one file
    on_gpus final_ctrl "$c5" "$c20" "$c80"
}

stage_seeds() {
    local jobs=() line out s c1 c2
    for s in 1 2; do
        out=$(own_jobs $s 80) || exit 1
        while IFS= read -r line; do [ -n "$line" ] && jobs+=("$line"); done <<< "$out"
    done
    c1=$(control_job 1 80) || exit 1; c2=$(control_job 2 80) || exit 1
    on_gpus seeds_own "${jobs[@]}"
    # run_control trains its control at the FIRST seed it is given, so one job per seed
    on_gpus seeds_ctrl "$c1" "$c2"
}

stage_summary() {
    $RP --summary
    $PY src/training/backup.py --repo "$RESULTS_REPO" --once --logs "$LOGS/*.log" "$OUT/recipe_v3.json"
}

run() { say "== $1"; "stage_$1" 2>&1 | tee -a "$LOGS/$1.log"; }       # pipefail: the stage's failure is run's

case "${1:-}" in
    setup|data|backup|sanity|tune|base|final|seeds|summary) run "$1" ;;
    all)
        run sanity; run tune; run base; run final; run seeds; run summary
        say "all: done -- docs/archive/pod_runbook_v3.md § What to report" ;;
    *) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 1 ;;     # the header comment is the usage
esac
