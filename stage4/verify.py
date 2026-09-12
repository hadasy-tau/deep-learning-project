"""Stage 4 verify: score a run's JSONL against the protocol reference and check
the pipeline itself behaved.

    python stage4/verify.py outputs/A_whisper-large-v3.jsonl outputs/B_whisper-large-v3-turbo-ct2.jsonl

Scoring is Stage 1's, verbatim: normalize_he() from stage2/pipeline.py, then
word- and character-level Levenshtein (rapidfuzz), corpus WER = sum(errors) /
sum(ref words).  Pipeline checks: every requested chunk has exactly one row,
no error, a non-empty hypothesis, latency within reason, and -- when both arms
are given -- the two runs cover the identical chunk set so A vs B is paired.
"""
import json, os, sys
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stage2'))
from pipeline import normalize_he

def load(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    df = pd.DataFrame(rows)
    # an all-None `error` column is dropped by the DataFrame constructor on
    # some pandas versions; a run with zero failures must still have the column
    for c in ('error', 'hypothesis'):
        if c not in df: df[c] = None
    return df.drop(columns=['raw'], errors='ignore')

def score(df):
    df = df.copy()
    df['ref_n'] = df.reference.map(normalize_he); df['hyp_n'] = df.hypothesis.fillna('').map(normalize_he)
    df['n_words'] = df.ref_n.str.split().str.len(); df['n_chars'] = df.ref_n.str.len()
    df['werr'] = [Levenshtein.distance(r.split(), h.split()) for r, h in zip(df.ref_n, df.hyp_n)]
    df['cerr'] = [Levenshtein.distance(r, h) for r, h in zip(df.ref_n, df.hyp_n)]
    df['wer'] = df.werr / df.n_words.clip(lower=1)
    return df

def report(df, name):
    ok = df[df.error.isna()]
    print(f'=== {name}: {df.arm.iloc[0]} {df.model.iloc[0]} via {df.provider.iloc[0]} ===')
    print(f'  rows {len(df)} | ok {len(ok)} | failed {int(df.error.notna().sum())} | duplicate chunk_ids {int(df.chunk_id.duplicated().sum())}')
    print(f'  empty hypotheses: {int((ok.hypothesis.fillna("").str.strip()=="").sum())}')
    print(f'  latency s: median {ok.latency_s.median():.2f}  p95 {ok.latency_s.quantile(.95):.2f}  max {ok.latency_s.max():.2f}')
    if 'exec_s' in ok and ok.exec_s.notna().any():
        print(f'  provider exec s: median {ok.exec_s.median():.2f}  queue s: median {ok.queue_s.median():.2f}  '
              f'(GPU realtime factor {ok.duration_s.sum()/ok.exec_s.sum():.1f}x)')
    s = score(ok)
    print(f'  corpus WER {s.werr.sum()/s.n_words.sum():.4f}   CER {s.cerr.sum()/s.n_chars.sum():.4f}   '
          f'(over {s.n_words.sum()} ref words, {s.duration_s.sum():.0f} s audio)')
    print(f'  per-chunk WER: median {s.wer.median():.3f}  max {s.wer.max():.3f}')
    return s

def checks(dfs, expected_ids=None):
    fails = []
    def chk(name, cond, detail=''):
        print(f'  [{"OK " if cond else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
        if not cond: fails.append(name)
    for name, df in dfs.items():
        chk(f'{name}: no errors', df.error.isna().all(), str(df.error.dropna().head(2).tolist())[:160])
        chk(f'{name}: chunk_id unique', df.chunk_id.is_unique)
        chk(f'{name}: non-empty hypothesis', (df.hypothesis.fillna('').str.strip() != '').all())
        chk(f'{name}: hypothesis is Hebrew', df.hypothesis.fillna('').str.contains(r'[א-ת]').all())
        chk(f'{name}: reference present', df.reference.notna().all())
        if expected_ids is not None:
            chk(f'{name}: exactly the requested chunks', set(df.chunk_id) == set(expected_ids), f'{len(set(df.chunk_id)^set(expected_ids))} differ')
    if len(dfs) == 2:
        a, b = dfs.values()
        chk('A and B cover the same chunks (paired)', set(a.chunk_id) == set(b.chunk_id))
        chk('A and B used different models', a.model.iloc[0] != b.model.iloc[0], f'{a.model.iloc[0]} vs {b.model.iloc[0]}')
    return fails

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='one or two run JSONL files')
    ap.add_argument('--expect', default=None, help='file of chunk_ids every run must cover exactly')
    a = ap.parse_args()
    paths = a.paths
    exp = [l.strip() for l in open(a.expect) if l.strip()] if a.expect else None
    dfs = {os.path.basename(p): load(p) for p in paths}
    scored = {k: report(v, k) for k, v in dfs.items()}
    print('\n=== pipeline checks ===')
    fails = checks(dfs, exp)
    if len(scored) == 2:
        (ka, a), (kb, b) = scored.items()
        m = a.merge(b, on='chunk_id', suffixes=('_A', '_B'))
        print('\n=== paired, per chunk ===')
        print(m[['chunk_id', 'speaker_id_A', 'duration_s_A', 'wer_A', 'wer_B', 'latency_s_A', 'latency_s_B']]
              .rename(columns={'speaker_id_A': 'speaker', 'duration_s_A': 'dur_s'}).round(3).to_string(index=False))
        print(f'\nA better on {int((m.wer_A < m.wer_B).sum())}, B better on {int((m.wer_B < m.wer_A).sum())}, tie {int((m.wer_A == m.wer_B).sum())}')
    print('\nRESULT:', 'ALL CHECKS PASSED' if not fails else f'FAILED: {fails}')
    sys.exit(0 if not fails else 1)
