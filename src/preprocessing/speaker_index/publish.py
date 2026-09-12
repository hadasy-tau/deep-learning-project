"""Stage 3 publish: push the index (never audio) to a private HuggingFace dataset.

Decisions (docs/speaker_index_plan.md): index only; Dolevabudi/knesset-committees-speakers;
private until Step 4 passes; licence CC-BY-SA 4.0 inherited from the Knesset
Corpus; credits ivrit.ai and the Knesset Corpus.

The write token is read from HF_TOKEN, else the repo-level cache/hf_write_token (mode
600, git-ignored).  It is never written into any published file.  Run with
dry_run=True first: it prints what would be uploaded and touches nothing.
"""
import os
import pandas as pd
from huggingface_hub import HfApi

from roster import OUT

# Repo-level cache/, shared with src/inference/providers.py -- the token is the
# repo's, not the speaker-index builder's.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'cache')

REPO   = 'Dolevabudi/knesset-committees-speakers'
FILES  = ['segments.parquet', 'manifest.csv', 'mk_metadata.csv', 'mk_stints.csv', 'mk_name_variants.csv',
          'alias_names.csv', 'alias_et.csv', 'build_sessions.csv', 'match_report.txt', 'holdout_report.txt']

def write_token():
    t = os.environ.get('HF_TOKEN')
    p = os.path.join(CACHE, 'hf_write_token')
    if not t and os.path.exists(p):
        t = open(p).read().strip()
    assert t, f'no write token: set HF_TOKEN or create {os.path.join(CACHE, "hf_write_token")}'
    return t

def card(df):
    mk = df[df.label == 'mk']
    h = df.groupby('label').duration_s.sum().div(3600).round(1)
    return f"""---
license: cc-by-sa-4.0
language:
- he
task_categories:
- automatic-speech-recognition
- audio-classification
tags:
- knesset
- hebrew
- speaker-identification
- parliament
pretty_name: Knesset Committees Speakers
size_categories:
- 1M<n<10M
---

# Knesset Committees Speakers

An index that attaches a verified Knesset member identity, and through it
demographics, to the committee audio in
[`ivrit-ai/knesset-committees`](https://huggingface.co/datasets/ivrit-ai/knesset-committees).
**No audio is included.** Each row names a span (`session`, `start`, `end`) of that
dataset's `audio.m4a`; `filename` follows the VoxKnesset convention
`{{speaker_id}}_{{session}}_{{start_ms}}_{{end_ms}}.wav` so the same tooling applies.

`speaker_id` is the Knesset's official `PersonID` -- the same id space as the
[Knesset Corpus](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus) and
[VoxKnesset](https://huggingface.co/datasets/ivrit-ai/VoxKnesset).

## How identities were assigned

Two independent text sources had to agree before an id was written, and the person
had to hold a seat on the recording date (Knesset ODATA). Sessions with a Knesset
Corpus protocol (through 2024-03-26) use text alignment between the corpus
sentences and ivrit.ai's protocol text (`match_method = A`); later sessions use
exact-string matching of the protocol's speaker line against seated MKs (`B1`)
or a pure, >=3-observation alias learned from the aligned sessions (`B2`); the
editor's picker id is used only as a veto, never to assign. No fuzzy matching is
used anywhere. Anything that did not pass is kept with `label = unresolved` and a
`reason`; guests are `non_mk`; former MKs speaking as guests are `former_mk`,
unless the roster says they had died or were past 95 on the day -- then
`unresolved` with `reason = deceased` / `implausible_age`.

## Contents

| label | hours | rows |
|---|---|---|
""" + '\n'.join(f'| {k} | {h[k]} | {(df.label == k).sum():,} |' for k in h.index) + f"""

{mk.speaker_id.nunique()} MKs with `label = mk`, Knessets {int(df.knesset.min())}-{int(df.knesset.max())},
sessions {df.session_date.min()} to {df.session_date.max()}.

Columns: `filename, speaker_id, session, start, end, duration_s, reference_text, quality,
label, label_path, match_method, reason, raw_name, local_speaker_id, et_id,
gold_speaker_id, is_chairman, knesset, committee_name, session_date, speaker_name, gender (0 f / 1 m),
age, date_of_birth, place_of_birth, year_of_aliya, religion, nationality,
religious_orientation`. `quality` is the median word alignment probability; it is
stored, not filtered on.

Side tables: `manifest.csv` (one row per session directory), `mk_metadata.csv`,
`mk_stints.csv`, `mk_name_variants.csv`, `alias_names.csv`, `alias_et.csv`,
`build_sessions.csv` (per-session placement, verification and any error -- the
record of which sessions were excluded and why), `match_report.txt`,
`holdout_report.txt`.

## Sources and licence

Audio, protocol text and alignment: ivrit.ai, `ivrit-ai/knesset-committees` (ivrit.ai
licence). Sentence-level speaker attribution and demographics: Goldin, Howell,
Ordan, Rabinovich, Wintner, *The Knesset Corpus* (Lang Resources & Evaluation, 2025),
CC-BY-SA 4.0. MK service dates: Knesset ODATA. This index is released under
CC-BY-SA 4.0.
"""

def publish(dry_run=True, private=True):
    api = HfApi(token=write_token())
    df = pd.read_parquet(os.path.join(OUT, 'segments.parquet'))
    readme = os.path.join(OUT, 'README.md')
    open(readme, 'w', encoding='utf-8').write(card(df))
    paths = [os.path.join(OUT, f) for f in FILES + ['README.md'] if os.path.exists(os.path.join(OUT, f))]
    print(('DRY RUN -- would upload' if dry_run else 'uploading') + f' to {REPO} (private={private}):')
    for p in paths:
        print(f'  {os.path.basename(p):24s} {os.path.getsize(p)/1e6:8.1f} MB')
    if dry_run:
        return
    api.create_repo(REPO, repo_type='dataset', private=private, exist_ok=True)
    for p in paths:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=REPO, repo_type='dataset')
    print(f'done: https://huggingface.co/datasets/{REPO}')

if __name__ == '__main__':
    publish(dry_run=True)
