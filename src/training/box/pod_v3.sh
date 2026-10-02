#!/usr/bin/env bash
# Plan v3 on a GPU pod, stage by stage.  docs/pod_runbook_v3.md is the order, the stop rules
# and what to report; docs/training_plan_v3.md is why.  Every stage is idempotent: a scored
# cell, a finished adapter, a tuning run and a cached base transcription are all skipped on a
# rerun, so after a crash or a preemption run the same stage again.
#
#   bash src/training/box/pod_v3.sh setup      # deps, ffmpeg, GPU and HF login checks, self-checks
#   bash src/training/box/pod_v3.sh data       # panel audio from HF, fill any gap from the corpus, verify both plans
#   bash src/training/box/pod_v3.sh backup     # mirror results to $RESULTS_REPO every 30 min (background)
#   bash src/training/box/pod_v3.sh gate       # the audio gate on 30601 -> outputs/gate_30601.txt
#   bash src/training/box/pod_v3.sh sanity     # overfit check + one timed tuning run (price the plan)
#   bash src/training/box/pod_v3.sh tune       # stages A, B, C on validation only, each picked by v3_decide.py
#   bash src/training/box/pod_v3.sh base       # base WER on the high-quality test vs the error map (hard stop)
#   bash src/training/box/pod_v3.sh final      # 5 / 20 / 80 min, seed 0, controls at every budget, both tests
#   bash src/training/box/pod_v3.sh seeds      # seeds 1 and 2 at 80 min, own adapters and their controls
#   bash src/training/box/pod_v3.sh summary    # outputs/results.csv, one last backup
#   bash src/training/box/pod_v3.sh all        # sanity -> tune (gate alongside) -> base -> final -> seeds -> summary
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
GATE_FILE=$OUT/gate_30601.txt
RP="$PY src/training/run_panel.py"
D="$PY src/training/box/v3_decide.py"
STEP="--plan $V2 --max-steps 400 --eval-batch 64"        # step mode, early stopping fixed at 4 validations

say() { echo "[$(date '+%F %T')] $*"; }

speakers() {   # every plan speaker, minus 30601 if the audio gate failed him
    $PY -c "import pandas as pd; print(' '.join(str(s) for s in sorted(pd.read_parquet('$V2').speaker_id.unique())))" |
        { if grep -qx FAIL "$GATE_FILE" 2>/dev/null; then tr ' ' '\n' | grep -vx 30601 | paste -sd' ' -; else cat; fi; }
}

stage_setup() {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
    pip install -q -r src/training/requirements.txt
    # a RunPod PyTorch template ships torch + torchaudio for its CUDA; requirements must not have replaced them
    $PY -c "import torch, torchaudio; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
    $PY -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 'not logged in to HuggingFace: run  hf auth login  (typed, never pasted into a chat)')"
    $PY -c "from huggingface_hub import HfApi; HfApi().list_repo_files('$PANEL_REPO', repo_type='dataset'); print('HF read access: OK')"
    $PY src/common.py >/dev/null && $PY src/evaluation/evaluate.py && $D --self-check
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
    setsid nohup $PY src/training/backup.py --repo "$RESULTS_REPO" --every 30 \
        --logs "$LOGS/sanity.log" "$LOGS/tune.log" "$LOGS/gate.log" "$LOGS/base.log" "$LOGS/final.log" "$LOGS/seeds.log" "$OUT/recipe_v3.json" "$GATE_FILE" \
        > "$LOGS/backup.log" 2>&1 < /dev/null &
    say "backup to $RESULTS_REPO every 30 min (log: $LOGS/backup.log)"
}

stage_gate() {
    local seg=src/preprocessing/speaker_index/outputs
    [ -f $seg/segments.parquet ] || $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download('knesset-asr/knesset-committees-speakers', 'segments.parquet', repo_type='dataset', local_dir='$seg')"
    # validate_audio.py reads the token from the environment; taken from the login, never printed
    HF_TOKEN=$($PY -c "from huggingface_hub import get_token; print(get_token())") \
        $PY src/preprocessing/speaker_index/validate_audio.py --speakers 30601 --per-speaker 40 --others 60 --tag _30601
    if $D gate; then echo PASS > "$GATE_FILE"; else echo FAIL > "$GATE_FILE"; fi
    say "gate: $(cat "$GATE_FILE")"
}

stage_sanity() {
    $PY - <<'EOF'
import sys, pandas as pd; sys.path.insert(0, 'src/training'); import train
P = pd.read_parquet('src/training/panel_plan_v2.parquet')
train.overfit_check(P, 'src/training/outputs/panel_audio', speaker=30685, arm='B')
EOF
    local t0=$SECONDS
    $RP --tune $STEP --speakers 30685 --budgets 80 --seeds 0 --lrs 3e-4      # one of stage A's runs, so it is reused
    say "sanity: one tuning run took $(( (SECONDS - t0) / 60 )) min; the whole plan is about 70-90 of these"
}

stage_tune() {
    for st in A B C; do
        for b in 5 80; do
            local g; g=$($D grid $st $b)     # an assignment, so a failure here stops the stage
            $RP --tune $STEP --speakers $TUNE_SPK --budgets $b --seeds 0 $g
        done
        $D pick $st
    done
    say "tune: recipe -> $OUT/recipe_v3.json"
}

stage_base() { $D base; }

stage_final() {
    [ -f "$GATE_FILE" ] || { say "final: the audio gate has not run (stage gate)"; exit 1; }
    local spk; spk=$(speakers)
    for b in 5 20 80; do          # each budget with its own recipe and its own budget-matched control
        local f; f=$($D flags $b)     # refuses unless tuning picked A, B and C; never fall back to a default rate
        $RP $STEP --seeds 0 --forgiven --test07-plan "$T07" --speakers $spk --budgets $b $f \
            --control-folds 2 --control-budgets $b
    done
}

stage_seeds() {
    local spk; spk=$(speakers)
    local f; f=$($D flags 80)
    for s in 1 2; do              # run_control trains its control at the FIRST seed given, so one call per seed
        $RP $STEP --seeds $s --forgiven --test07-plan "$T07" --speakers $spk --budgets 80 $f \
            --control-folds 2 --control-budgets 80
    done
}

stage_summary() {
    $RP --summary
    $PY src/training/backup.py --repo "$RESULTS_REPO" --once --logs "$LOGS"/*.log "$OUT/recipe_v3.json" "$GATE_FILE"
}

run() { say "== $1"; "stage_$1" 2>&1 | tee -a "$LOGS/$1.log"; }       # pipefail: the stage's failure is run's

case "${1:-}" in
    setup|data|backup|gate|sanity|tune|base|final|seeds|summary) run "$1" ;;
    all)
        [ -f "$GATE_FILE" ] || { run gate & GATE_PID=$!; }      # independent of tuning; final waits for it
        run sanity; run tune
        if [ -n "${GATE_PID:-}" ]; then wait "$GATE_PID" || say "gate crashed: see $LOGS/gate.log; final will refuse until it has a verdict"; fi
        run base; run final; run seeds; run summary
        say "all: done -- docs/pod_runbook_v3.md § What to report" ;;
    *) sed -n '2,22p' "$0"; exit 1 ;;
esac
