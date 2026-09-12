"""Stage 4 merge: assemble the run JSONLs into the dataset Stage 1 reads.

    python stage4/merge.py            # -> outputs/inference.parquet + coverage; dry-run the upload
    python stage4/merge.py --upload   # push to Dolevabudi/knesset-committees-inference

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
        'reference','hypothesis','latency_s','exec_s','queue_s','error','ts']

def load_runs():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, '*.jsonl'))):
        if os.path.basename(p).startswith(('verify', 'e2e')): continue
        for line in open(p, encoding='utf-8'):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            rows.append({k: r.get(k) for k in KEEP})
    df = pd.DataFrame(rows)
    # keep the latest successful row per (chunk, arm); an errored row only if no success exists
    df['ok'] = df.error.isna()
    df = df.sort_values(['chunk_id','arm','ok','ts']).drop_duplicates(['chunk_id','arm'], keep='last').drop(columns='ok')
    return df

def wide(long):
    ok = long[long.error.isna()]
    a = ok[ok.arm=='A'].set_index('chunk_id'); b = ok[ok.arm=='B'].set_index('chunk_id')
    meta = ok.drop_duplicates('chunk_id').set_index('chunk_id')[['speaker_id','session','session_date','knesset','duration_s','quality','reference']]
    w = meta.join(a[['hypothesis','model','latency_s']].rename(columns=lambda c: c+'_A'), how='left') \
            .join(b[['hypothesis','model','latency_s','exec_s']].rename(columns=lambda c: c+'_B'), how='left')
    w['has_A'] = w.hypothesis_A.notna(); w['has_B'] = w.hypothesis_B.notna()
    return w.reset_index()

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
        for f in ['inference.parquet', 'inference_long.parquet', 'coverage.parquet']:
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO, repo_type='dataset'); print('  uploaded', f)
        print(f'https://huggingface.co/datasets/{REPO}')
