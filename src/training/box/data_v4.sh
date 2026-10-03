#!/usr/bin/env bash
# Run 4's audio, cut before any GPU is rented (docs/training_plan_v4.md).  Runs on any machine
# with the repo and an HF login; run 4's was cut on the laptop, which already held plan v2's
# clips under src/training/outputs/panel_audio/.
#
#   bash src/training/box/data_v4.sh data      # extract both v4 plans (only clips not on disk), verify with audio
#   bash src/training/box/data_v4.sh upload    # add the new WAVs and both plans to $REPO; check every file arrived
#   bash src/training/box/data_v4.sh all       # data -> upload
#
# The clips go into knesset-committees-panel-hq beside plan v2's, which do not change: the plans
# (panel_plan_v2 / _v4, panel_test07 / _v4) say which run used which clips.  Both datasets and
# the source corpus are public.  extract skips WAVs on disk; upload_large_folder resumes.
set -euo pipefail
cd "$(dirname "$0")/../../.."

PY=${PY:-python}
V4=src/training/panel_plan_v4.parquet
T07=src/training/panel_test07_v4.parquet
REPO=${REPO:-knesset-asr/knesset-committees-panel-hq}
REVISION=839622c1dbc7509f8250c4a0eaa5881c629e848f        # the corpus revision the index (and so every plan) was built at
OUT=src/training/outputs; LOGS=$OUT/logs; mkdir -p "$LOGS"

say() { echo "[$(date '+%F %T')] $*"; }

stage_data() {
    $PY -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 'not logged in to HuggingFace: run  hf auth login  (typed, never pasted into a chat)')"
    mkdir -p src/inference/cache && echo "\"$REVISION\"" > src/inference/cache/revision.json
    $PY src/training/materialize.py extract --plan "$V4" --prefetch 4
    $PY src/training/materialize.py extract --plan "$T07" --prefetch 4
    $PY src/training/materialize.py verify --plan "$V4"
    $PY src/training/materialize.py verify --plan "$T07"
    say "data: both v4 plans verified, with audio"
}

stage_upload() {
    $PY - "$REPO" "$V4" "$T07" <<'PYEOF'
import os, sys
import pandas as pd
from huggingface_hub import HfApi
repo, v4, t07 = sys.argv[1:4]; api = HfApi()
# the folder must hold only clips the v4 plans name, so nothing else lying there is published
P = pd.concat([pd.read_parquet(v4), pd.read_parquet(t07)]).drop_duplicates('chunk_id')
root = 'src/training/outputs/panel_audio'
local = {os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs if f.endswith('.wav')}
extra = local - set(P.filename)
if extra: sys.exit(f'{len(extra)} wavs under {root} are in no v4 plan (e.g. {sorted(extra)[:3]}): move them out first')
api.upload_large_folder(repo_id=repo, repo_type='dataset', folder_path='src/training/outputs', allow_patterns=['panel_audio/**'])
for p in (v4, t07):
    api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=repo, repo_type='dataset',
                    commit_message=f'run 4: {os.path.basename(p)}')
remote = {f.path: f.size for f in api.list_repo_tree(repo, repo_type='dataset', recursive=True) if hasattr(f, 'size')}
bad = [f for f in P.filename if remote.get('panel_audio/' + f) != os.path.getsize(os.path.join('src/training/outputs/panel_audio', f))]
print(f'{len(P) - len(bad):,}/{len(P):,} clips in {repo} with the right size')
sys.exit(f'missing or different: {bad[:5]}' if bad else 0)
PYEOF
    say "upload: every v4 clip in $REPO"
}

run() { say "== $1"; "stage_$1" 2>&1 | tee -a "$LOGS/data_v4_$1.log"; }

case "${1:-}" in
    data|upload) run "$1" ;;
    all) run data; run upload; say "all: done" ;;
    *) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 1 ;;
esac
