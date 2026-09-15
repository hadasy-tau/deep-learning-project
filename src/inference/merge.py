"""Stage 4 merge: assemble the run JSONLs into the dataset Stage 1 reads.

    python src/inference/merge.py            # -> outputs/inference.parquet + coverage; dry-run the upload
    python src/inference/merge.py --upload   # push to Dolevabudi/knesset-committees-inference

One row per (chunk_id, arm) with the hypothesis, the reference, timing and
the chunk's metadata; plus a wide table with one row per chunk carrying
hyp_A and hyp_B side by side for paired scoring.  Later runs re-run this and
the parquet grows; nothing is ever re-transcribed because run.py resumes from
the same JSONLs.
"""
import glob, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'outputs')
REPO = 'Dolevabudi/knesset-committees-inference'
KEEP = ['chunk_id','arm','provider','model','speaker_id','session','session_date','knesset','duration_s','quality',
        'reference','hypothesis','language','latency_s','exec_s','queue_s','error','ts']

def load_runs():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, '*.jsonl'))):
        if os.path.basename(p).startswith(('verify', 'e2e')): continue
        for line in open(p, encoding='utf-8'):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            rows.append({k: r.get(k) for k in KEEP})
    df = pd.DataFrame(rows)
    # Arm A ran twice: first letting Whisper auto-detect the language (language
    # null), then with Hebrew forced.  They are different experiments, so they
    # are kept apart as arms 'A' (forced, the one Stage 1 scores) and 'A_auto'.
    df.loc[(df.arm == 'A') & df.language.isna(), 'arm'] = 'A_auto'
    # keep the latest successful row per (chunk, arm); an errored row only if no success exists
    df['ok'] = df.error.isna()
    df = df.sort_values(['chunk_id','arm','ok','ts']).drop_duplicates(['chunk_id','arm'], keep='last').drop(columns='ok')
    return df

def wide(long):
    ok = long[long.error.isna()]
    a = ok[ok.arm=='A'].set_index('chunk_id'); b = ok[ok.arm=='B'].set_index('chunk_id'); aa = ok[ok.arm=='A_auto'].set_index('chunk_id')
    meta = ok.drop_duplicates('chunk_id').set_index('chunk_id')[['speaker_id','session','session_date','knesset','duration_s','quality','reference']]
    w = meta.join(a[['hypothesis','model','latency_s']].rename(columns=lambda c: c+'_A'), how='left') \
            .join(b[['hypothesis','model','latency_s','exec_s']].rename(columns=lambda c: c+'_B'), how='left') \
            .join(aa[['hypothesis']].rename(columns={'hypothesis': 'hypothesis_A_auto'}), how='left')
    w['has_A'] = w.hypothesis_A.notna(); w['has_B'] = w.hypothesis_B.notna()
    return w.reset_index()

def card(w, long):
    """Dataset card: what the files are, how they were produced, what to read for Stage 1."""
    ws = w[w.in_stage1_subset]
    return f"""---
license: cc-by-sa-4.0
language:
- he
task_categories:
- automatic-speech-recognition
tags:
- knesset
- hebrew
- whisper
- evaluation
pretty_name: Knesset Committees Inference
---

# Knesset Committees Inference

Transcriptions of Knesset committee audio by two models, with the protocol
reference alongside, for the personalised-ASR study (Stage 1: per-speaker
WER, general model vs Hebrew fine-tune). Audio and references come from
[`Hadasy/knesset-committees-chunks`](https://huggingface.co/datasets/Hadasy/knesset-committees-chunks);
speaker identities from
[`Dolevabudi/knesset-committees-speakers`](https://huggingface.co/datasets/Dolevabudi/knesset-committees-speakers).
No audio is included.

| arm | model | served by | language |
|---|---|---|---|
| A | `openai/whisper-large-v3` | HF Inference (deepinfra) | forced `he` |
| B | `ivrit-ai/whisper-large-v3-turbo-ct2` | RunPod serverless (faster-whisper) | forced `he` |
| A_auto | `openai/whisper-large-v3` | same as A | auto-detected (see below) |

## Files

- `inference.parquet` -- one row per chunk, `hypothesis_A` and `hypothesis_B`
  side by side with `reference`, `speaker_id`, `session`, `session_date`,
  `knesset`, `duration_s`, `quality`; `in_stage1_subset` marks the evaluated
  subset. {len(ws):,} subset chunks, {ws.duration_s.sum()/3600:.0f} h, {ws.speaker_id.nunique()} speakers, all with both hypotheses.
- `inference_long.parquet` -- one row per (chunk, arm) with timing (`latency_s`,
  provider `exec_s`/`queue_s`) and any error; {len(long):,} rows.
- `coverage.parquet` -- one row per chunk of the whole corpus ({1204617:,}):
  `in_subset`, `done_A`, `done_B`, `done_A_auto`, errors. The record of what was
  and was not transcribed.

## The subset

One hour per MK (alignment quality >= 0.5), drawn round-robin over sessions so
it spans the speaker's tenure. Four speakers have under six minutes of usable
audio in total and correspondingly few chunks (`speaker_id` 12944, 30684,
30698, 30846); filter by hours before per-speaker statistics.

## Language forcing and `hypothesis_A_auto`

The first Arm A run let Whisper detect the language. On chunks under 3 s it
guessed wrong a quarter of the time (Portuguese, Arabic, Russian...), 6.7 %
of all chunks. Arm B is always told the language, so the comparison was
unfair; Arm A was re-run with `language='he'`. The re-run is `hypothesis_A`
(arm `A`); the auto-detect run is kept as `hypothesis_A_auto` (arm `A_auto`
in the long table) because the detection-failure rate is itself a finding.

## Scoring as validated

`normalize_he` (src/common.py) then word-level Levenshtein, corpus WER on
the subset: **A 0.417, B 0.324**. Empty hypotheses: A 0.30 %, B 0. Hebrew
script: A 99.7 %, B 99.99 %.

## Licence

References and speaker attribution derive from the Knesset Corpus (CC-BY-SA
4.0); audio from ivrit.ai. This dataset is released under CC-BY-SA 4.0.
"""

if __name__ == '__main__':
    long = load_runs()
    w = wide(long)
    sub = set(pd.read_parquet(os.path.join(HERE, 'subset_stage1.parquet')).chunk_id)
    w['in_stage1_subset'] = w.chunk_id.isin(sub)
    long.to_parquet(os.path.join(OUT, 'inference_long.parquet'), index=False)
    w.to_parquet(os.path.join(OUT, 'inference.parquet'), index=False)
    h = lambda m: w.loc[m, 'duration_s'].sum()/3600
    print(f'long: {len(long):,} rows | wide: {len(w):,} chunks, {h(w.index):.0f} h')
    print(f'  A: {w.has_A.sum():,} ({h(w.has_A):.0f} h) | B: {w.has_B.sum():,} ({h(w.has_B):.0f} h) | paired: {(w.has_A&w.has_B).sum():,} ({h(w.has_A&w.has_B):.0f} h)')
    print(f'  Stage-1 subset paired: {(w.has_A&w.has_B&w.in_stage1_subset).sum():,}/{len(sub):,} | speakers paired: {w[w.has_A&w.has_B].speaker_id.nunique()}')
    print(f'  errors left: A {int((long[long.arm=="A"].error.notna()).sum())}, B {int((long[long.arm=="B"].error.notna()).sum())}')
    if '--upload' in sys.argv:
        from huggingface_hub import HfApi
        import providers as P
        api = HfApi(token=P.secret('HF_WRITE_TOKEN', 'hf_write_token'))
        api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
        open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8').write(card(w, long))
        api.upload_file(path_or_fileobj=os.path.join(OUT, 'README.md'), path_in_repo='README.md', repo_id=REPO, repo_type='dataset'); print('  uploaded README.md')
        for f in ['inference.parquet', 'inference_long.parquet', 'coverage.parquet']:
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO, repo_type='dataset'); print('  uploaded', f)
        print(f'https://huggingface.co/datasets/{REPO}')
