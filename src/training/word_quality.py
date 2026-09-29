"""Word-level alignment scores for the panel's candidate train/dev chunks.

A chunk's `quality` is a median over its words' alignment probabilities, so a
stretch of badly aligned words -- protocol text that does not match the audio
there -- can hide inside a good median.  The raw ivrit-ai/knesset-committees
sessions keep every protocol word with its time span and probability
(transcript.refined.json, read by speaker_index/protocol.py).  This module
takes, for each candidate chunk, the words whose midpoint falls inside the
chunk's span and summarises them:

  n_aligned   words found in the span (should equal the reference's word count)
  low_share   share of those words with probability < LOW_P
  low_run     longest run of consecutive words below LOW_P
  min_p       the lowest word probability

Candidates are chunks at quality >= MIN_QUALITY in sessions that are NOT the
speaker's personal-test sessions (panel_plan.parquet's test is kept as it is),
newest session first, until RAW_TARGET_MIN minutes of candidates -- enough for
dev + the 80-minute budget after the word rule removes some.

The rule itself (word_ok) lives here so plan and verify share it; its
threshold MAX_LOW_SHARE was fixed from `report()` before any training
(docs/training_plan_v3.md).

    python src/training/word_quality.py            # build word_quality.parquet, print the report
    python src/training/word_quality.py --check    # self-checks only
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'word_quality.parquet')
PANEL = os.path.join(ROOT, 'src', 'evaluation', 'outputs', 'committees_panel.csv')
PLAN_V1 = os.path.join(HERE, 'panel_plan.parquet')
INDEX = os.path.join(ROOT, 'src', 'inference', 'cache', 'index.parquet')

MIN_QUALITY = 0.95      # ivrit.ai's "high quality" line for a segment's median word probability
LOW_P = 0.5             # a word below this is clearly misaligned (half the words of a 0.95 chunk sit below 0.95)
MAX_RUN = 3             # a run of this many low words in a row drops the chunk
MAX_LOW_SHARE = 0.10    # at most this share of low words; fixed from report() (see the module docstring)
MAX_COUNT_GAP = 0.2     # |aligned words - reference words| / reference words above this: the span is not verifiable
RAW_TARGET_MIN = 200    # candidate minutes per speaker (>= 0.95), before the word rule
EXCLUDE_SESSIONS = {30843: {2235355}}   # the audio gate: this dev session is mostly another voice (docs/training_run2.md)


def span(chunk_id):
    """{speaker}_{session}_{start_ms}_{end_ms}.flac -> (session, start_s, end_s)."""
    _, sess, a, b = chunk_id.rsplit('.', 1)[0].split('_')
    return int(sess), int(a) / 1000, int(b) / 1000


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0; best = max(best, cur)
    return best


def summarise(probs, n_ref):
    p = np.asarray([x for x in probs if x is not None], dtype=float)
    if not len(p):
        return dict(n_aligned=0, n_ref=n_ref, low_share=1.0, low_run=0, min_p=0.0)
    low = p < LOW_P
    return dict(n_aligned=int(len(p)), n_ref=n_ref, low_share=float(low.mean()), low_run=longest_run(low), min_p=float(p.min()))


def words_in(words, start, end, pad=0.05):
    """Words whose time midpoint lies inside [start, end] (+pad): a chunk is cut on
    word boundaries, so its own words are exactly these."""
    return [w for w in words if start - pad <= (w['start'] + w['end']) / 2 <= end + pad]


def word_ok(W, max_low_share=MAX_LOW_SHARE):
    """The rule, on a frame with word_quality's columns."""
    gap = (W.n_aligned - W.n_ref).abs() / W.n_ref.clip(lower=1)
    return (gap <= MAX_COUNT_GAP) & (W.low_share <= max_low_share) & (W.low_run < MAX_RUN)


def candidates(index_path=INDEX, panel_path=PANEL, plan_v1=PLAN_V1, target_min=RAW_TARGET_MIN):
    panel = pd.read_csv(panel_path, index_col=0); core = panel[~panel.profile.str.contains('alt')].index
    idx = pd.read_parquet(index_path, columns=['chunk_id', 'speaker_id', 'session', 'session_date', 'duration_s', 'quality', 'text'])
    idx = idx[idx.speaker_id.isin(core) & (idx.quality >= MIN_QUALITY)]
    P1 = pd.read_parquet(plan_v1)
    out = []
    for sid, d in idx.groupby('speaker_id'):
        test = P1[(P1.speaker_id == sid) & (P1.part == 'test')]
        first_test = test.session_date.min()
        d = d[~d.session.isin(set(test.session_id)) & (d.session_date <= first_test) & ~d.session.isin(EXCLUDE_SESSIONS.get(int(sid), set()))]
        per = d.groupby('session').agg(date=('session_date', 'first'), mins=('duration_s', lambda s: s.sum() / 60)).sort_values(['date'], ascending=False)
        n = int(np.searchsorted(per.mins.cumsum().values, target_min)) + 1
        out.append(d[d.session.isin(per.index[:n])])
    return pd.concat(out, ignore_index=True)


def _one_session(session, rows):
    sys.path.insert(0, os.path.join(ROOT, 'src', 'preprocessing', 'speaker_index'))
    import protocol as PR
    s = PR.load_session(int(session), keep=False)
    words = [w for seg in s['segments'] for w in seg['words']]
    res = []
    for r in rows.itertuples():
        _, a, b = span(r.chunk_id)
        W = words_in(words, a, b)
        res.append(dict(chunk_id=r.chunk_id, speaker_id=int(r.speaker_id), session=int(session),
                        **summarise([w.get('probability') for w in W], len(str(r.text).split()))))
    return res


def build(out=OUT, workers=6):
    C = candidates()
    done = pd.read_parquet(out) if os.path.exists(out) else pd.DataFrame(columns=['chunk_id', 'session'])
    todo = C[~C.chunk_id.isin(set(done.chunk_id))]
    groups = list(todo.groupby('session'))
    print(f'{len(C):,} candidate chunks ({C.duration_s.sum()/3600:.1f} h) over {C.session.nunique()} sessions; {len(groups)} sessions to fetch', flush=True)
    rows, t0, failed = [], time.time(), []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_one_session, s, g): s for s, g in groups}
        for i, f in enumerate(as_completed(futs), 1):
            try: rows += f.result()
            except Exception as e: failed.append((futs[f], repr(e)[:100]))
            if i % 25 == 0 or i == len(groups):
                print(f'  {i}/{len(groups)} sessions, {time.time()-t0:.0f}s', flush=True)
                pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out, index=False)   # checkpoint
    W = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    W.to_parquet(out, index=False)
    if failed: print(f'{len(failed)} sessions failed (re-run to retry): {failed[:3]}')
    return W


def report(W=None, index_path=INDEX, shares=(0.0, 0.05, 0.10, 0.15, 0.20)):
    """Minutes that survive per speaker under each candidate MAX_LOW_SHARE: the
    table the threshold is chosen from (strictest value leaving every speaker
    >= 15 min dev + 80 min train)."""
    W = pd.read_parquet(OUT) if W is None else W
    dur = pd.read_parquet(index_path, columns=['chunk_id', 'duration_s']).set_index('chunk_id').duration_s
    W = W.assign(mins=W.chunk_id.map(dur) / 60)
    gap = (W.n_aligned - W.n_ref).abs() / W.n_ref.clip(lower=1)
    print(f'{len(W):,} chunks; word count matches the reference within {MAX_COUNT_GAP:.0%} for {(gap <= MAX_COUNT_GAP).mean():.1%}; '
          f'median low_share {W.low_share.median():.3f}; runs of >= {MAX_RUN} low words in {(W.low_run >= MAX_RUN).mean():.1%}')
    t = pd.DataFrame({f'<= {s:.2f}': W[word_ok(W, s)].groupby('speaker_id').mins.sum() for s in shares})
    t.insert(0, 'all >= 0.95', W.groupby('speaker_id').mins.sum())
    print('minutes kept per speaker (candidates only; need >= 95):'); print(t.round(0).to_string())
    return t


def _self_checks():
    assert span('23558_2238216_40_18960.flac') == (2238216, 0.04, 18.96)
    assert longest_run([True, True, False, True, True, True, False]) == 3 and longest_run([]) == 0
    s = summarise([0.9, 0.2, 0.3, 0.1, 0.95, None], 5)
    assert s['n_aligned'] == 5 and abs(s['low_share'] - 0.6) < 1e-9 and s['low_run'] == 3 and s['min_p'] == 0.1
    ws = [dict(start=0.0, end=0.4), dict(start=0.5, end=1.0), dict(start=1.9, end=2.3)]
    assert len(words_in(ws, 0.5, 2.0)) == 1 and len(words_in(ws, 0.0, 2.3)) == 3
    W = pd.DataFrame(dict(n_aligned=[10, 10, 10, 10], n_ref=[10, 10, 10, 20], low_share=[0.0, 0.3, 0.1, 0.0], low_run=[0, 1, 3, 0]))
    assert list(word_ok(W, 0.10)) == [True, False, False, False]    # clean; too many low; a run of 3; half the words unaccounted for
    print('word_quality.py: OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--report', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    _self_checks()
    if a.check: sys.exit(0)
    W = pd.read_parquet(OUT) if a.report else build(workers=a.workers)
    report(W)
