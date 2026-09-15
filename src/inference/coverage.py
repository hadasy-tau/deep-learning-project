"""Stage 4 coverage: which chunks have been transcribed, by which arm, and
which have not -- the record of what inference has and has not been run on.

    python src/inference/coverage.py                # print the table, write coverage.parquet

Reads every outputs/*.jsonl, joins to the full index, and writes
outputs/coverage.parquet with one row per chunk in the corpus:
  chunk_id, speaker_id, session, duration_s, in_subset, done_A, done_B, done_A_auto, error_A, error_B
(done_A is the forced-Hebrew Arm A run; done_A_auto the earlier auto-detect run)
so a later run (Stage 2 wanting more of a speaker) can select exactly the
chunks that still need an arm, and nothing is ever sent twice.
"""
import glob, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

def results(arm):
    ok, err = set(), {}
    for p in glob.glob(os.path.join(OUT, '*.jsonl')):
        if os.path.basename(p).startswith(('verify', 'e2e')): continue
        for line in open(p, encoding='utf-8'):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            # Arm A: only the forced-Hebrew run counts (rows carry language='he');
            # the earlier auto-detect rows are 'A_auto' (see merge.py)
            a = r.get('arm')
            if a == 'A' and r.get('language') != 'he': a = 'A_auto'
            if a != arm: continue
            if r.get('error') is None: ok.add(r['chunk_id']); err.pop(r['chunk_id'], None)
            elif r['chunk_id'] not in ok: err[r['chunk_id']] = r['error'][:80]
    return ok, err

if __name__ == '__main__':
    idx = D.index()[['chunk_id', 'speaker_id', 'session', 'duration_s']]
    sub = set(pd.read_parquet(os.path.join(os.path.dirname(OUT), 'subset_stage1.parquet')).chunk_id) \
          if os.path.exists(os.path.join(os.path.dirname(OUT), 'subset_stage1.parquet')) else set()
    okA, errA = results('A'); okB, errB = results('B'); okAA, _ = results('A_auto')
    c = idx.assign(in_subset=idx.chunk_id.isin(sub), done_A=idx.chunk_id.isin(okA), done_B=idx.chunk_id.isin(okB), done_A_auto=idx.chunk_id.isin(okAA),
                   error_A=idx.chunk_id.map(errA), error_B=idx.chunk_id.map(errB))
    c.to_parquet(os.path.join(OUT, 'coverage.parquet'), index=False)
    h = lambda m: c.loc[m, 'duration_s'].sum() / 3600
    print(f'corpus: {len(c):,} chunks, {h(c.index):,.0f} h   |   Stage-1 subset: {c.in_subset.sum():,} chunks, {h(c.in_subset):.0f} h')
    for arm, done, err in (('A', c.done_A, c.error_A.notna()), ('B', c.done_B, c.error_B.notna())):
        print(f'  arm {arm}: done {done.sum():>7,} ({h(done):6.1f} h)  |  in subset done {(done & c.in_subset).sum():>6,}/{c.in_subset.sum():,}'
              f'  |  outside subset done {(done & ~c.in_subset).sum():>6,}  |  errors pending {err.sum()}')
    both = c.done_A & c.done_B
    print(f'  BOTH arms: {both.sum():,} chunks ({h(both):.1f} h) -> paired A-vs-B ready;  subset paired {(both & c.in_subset).sum():,}/{c.in_subset.sum():,}')
    print(f'wrote {OUT}/coverage.parquet')
