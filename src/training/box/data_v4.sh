#!/usr/bin/env bash
# Run 4's audio, cut on a CPU pod so no GPU waits for it (docs/training_plan_v4.md).
#
#   bash src/training/box/data_v4.sh setup     # deps, HF login check, the corpus revision pinned
#   bash src/training/box/data_v4.sh data      # panel-hq's clips, then extract the rest of both v4 plans, verify with audio
#   bash src/training/box/data_v4.sh pack      # one uncompressed tar per speaker under outputs/panel_tars/
#   bash src/training/box/data_v4.sh upload    # the tars + both plans to $OUT_REPO (private)
#   bash src/training/box/data_v4.sh all       # setup -> data -> pack -> upload
#
# The GPU pod then downloads ~15 tars instead of ~70,000 WAVs (pod_v4.sh boot): a snapshot of
# that many small files is slow and rate-limited, a few large files run at full bandwidth.
# Everything is idempotent: extract skips WAVs on disk, pack skips a speaker whose tar is
# newer than its WAVs, upload_large_folder resumes.
set -euo pipefail
cd "$(dirname "$0")/../../.."

PY=${PY:-python}
V4=src/training/panel_plan_v4.parquet
T07=src/training/panel_test07_v4.parquet
SRC_REPO=knesset-asr/knesset-committees-panel-hq          # run 3's clips: a head start on extract
OUT_REPO=${OUT_REPO:-knesset-asr/knesset-committees-panel-v4}
REVISION=839622c1dbc7509f8250c4a0eaa5881c629e848f        # the corpus revision the index (and so every plan) was built at
OUT=src/training/outputs; LOGS=$OUT/logs; TARS=$OUT/panel_tars; mkdir -p "$LOGS"
export HF_HUB_CACHE=${HF_HUB_CACHE:-/root/hfcache/hub} HF_XET_CACHE=${HF_XET_CACHE:-/root/hfcache/xet}

say() { echo "[$(date '+%F %T')] $*"; }

stage_setup() {
    pip install -q pandas pyarrow numpy soundfile huggingface_hub duckdb requests
    $PY -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 'not logged in to HuggingFace: run  hf auth login  (typed, never pasted into a chat)')"
    mkdir -p src/inference/cache && echo "\"$REVISION\"" > src/inference/cache/revision.json
    say "setup: corpus pinned at $REVISION; $(nproc) CPUs, $(df -h --output=avail . | tail -1) free"
}

stage_data() {
    $PY src/training/materialize.py download --repo "$SRC_REPO"
    $PY src/training/materialize.py extract --plan "$V4" --prefetch 4
    $PY src/training/materialize.py extract --plan "$T07" --prefetch 4
    $PY src/training/materialize.py verify --plan "$V4"
    $PY src/training/materialize.py verify --plan "$T07"
    say "data: both v4 plans verified, with audio"
}

stage_pack() {
    mkdir -p "$TARS"
    $PY - "$V4" "$T07" "$TARS" <<'EOF'
import os, sys, tarfile, pandas as pd
v4, t07, tars = sys.argv[1:4]; audio = 'src/training/outputs/panel_audio'
P = pd.concat([pd.read_parquet(v4), pd.read_parquet(t07)]).drop_duplicates('chunk_id')
for sid, g in P.groupby('speaker_id'):
    path = os.path.join(tars, f'{sid}.tar'); files = sorted(set(g.filename))
    if os.path.exists(path) and os.path.getmtime(path) >= max(os.path.getmtime(os.path.join(audio, f)) for f in files):
        continue
    with tarfile.open(path + '.part', 'w') as tf:
        for f in files: tf.add(os.path.join(audio, f), arcname=f)
    os.replace(path + '.part', path)
    print(f'{sid}: {len(files):,} wavs, {os.path.getsize(path) / 2**30:.2f} GB', flush=True)
EOF
    cp "$V4" "$T07" "$TARS/"
    say "pack: $(ls "$TARS"/*.tar | wc -l) tars, $(du -sh "$TARS" | cut -f1)"
}

stage_upload() {
    $PY - "$OUT_REPO" "$TARS" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1:3]
api = HfApi(); api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type='dataset')
card = """# Knesset Committees Panel v4\n\nRun 4's audio (docs/training_plan_v4.md in hadasy-tau/deep-learning-project): one tar per\nspeaker of 16 kHz mono WAVs (`<speaker_id>/<chunk_id>.wav`), the clips of panel_plan_v4.parquet\nand panel_test07_v4.parquet, cut from knesset-asr/knesset-committees-chunks. Private.\n"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo='README.md', repo_id=repo, repo_type='dataset')
print('uploaded', repo)
EOF
    $PY - "$OUT_REPO" "$TARS" <<'EOF'
import os, sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1:3]
remote = {f.path: f.size for f in HfApi().list_repo_tree(repo, repo_type='dataset') if hasattr(f, 'size')}
bad = [f for f in os.listdir(folder) if remote.get(f) != os.path.getsize(os.path.join(folder, f))]
sys.exit(f'upload incomplete: {bad}' if bad else 0)
EOF
    say "upload: every tar and plan in $OUT_REPO, sizes match"
}

run() { say "== $1"; "stage_$1" 2>&1 | tee -a "$LOGS/data_v4_$1.log"; }

case "${1:-}" in
    setup|data|pack|upload) run "$1" ;;
    all) run setup; run data; run pack; run upload; say "all: done -- stop this pod" ;;
    *) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 1 ;;
esac
