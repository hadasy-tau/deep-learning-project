"""The audio gate for run 4's two new speakers, on their own plan chunks (docs/training_plan_v4.md).

speaker_index/validate_audio.py embeds segments with ECAPA-TDNN and flags a segment that is
closer to another speaker's centroid than to its own.  Run 2 ran it on the panel's own
chunks; 23641 and 23635 were never gated.  This samples their plan-v4 chunks per split
(test, dev, train) and, as the reference population a contaminated chunk could sit near,
the other speakers recorded in the same sessions -- the chair and the MKs in the room, who
are the voices that leak into a committee chunk.  It needs no GPU (CPU ECAPA is fast enough
for a few hundred segments) and reads the audio spans straight from
ivrit-ai/knesset-committees with ffmpeg, as validate_audio does.

    python src/training/box/gate_v4.py                   # embed, flag, write the report
    python src/training/box/gate_v4.py --self-check

Writes src/preprocessing/speaker_index/outputs/audio_check_run4.parquet and
audio_report_run4.txt.  The stop rule: more than MAX_FLAGGED of a speaker's sampled chunks
flagged, or a test or dev session with a flagged chunk, means that session goes into
word_quality.EXCLUDE_SESSIONS and plan-v4 is rebuilt before any GPU time is spent.
"""
import argparse, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TRAINING = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(TRAINING, '..', '..'))
SI = os.path.join(ROOT, 'src', 'preprocessing', 'speaker_index')
sys.path.insert(0, TRAINING); sys.path.insert(0, SI)
PLAN_V4 = os.path.join(TRAINING, 'panel_plan_v4.parquet')
INDEX = os.path.join(ROOT, 'src', 'inference', 'cache', 'index.parquet')
OUT = os.path.join(SI, 'outputs')
NEW = [23641, 23635]
PER_PART, REF_SPEAKERS, PER_REF, MIN_S = 15, 30, 12, 3.0
MAX_FLAGGED = 0.10


def sample(P, idx, seed=0):
    """15 chunks per split per new speaker, one per session where the split has enough
    sessions; then the speakers most often present in those sessions, 12 chunks each."""
    import word_quality as WQ
    rng = np.random.default_rng(seed); rows = []
    for s in NEW:
        for part in ('test', 'dev', 'train'):
            g = P[(P.speaker_id == s) & (P.part == part) & (P.duration_s >= MIN_S)]
            g = g.sample(frac=1, random_state=seed).drop_duplicates('session_id') if g.session_id.nunique() >= PER_PART else g.sample(frac=1, random_state=seed)
            rows.append(g.head(PER_PART).assign(role='new', part=part))
    S = pd.concat(rows)
    sess = set(S.session_id)
    co = idx[idx.session.isin(sess) & ~idx.speaker_id.isin(NEW) & (idx.quality >= 0.7) & (idx.duration_s >= MIN_S)]
    refs = co.groupby('speaker_id').session.nunique().sort_values(ascending=False).head(REF_SPEAKERS).index
    pool = idx[idx.speaker_id.isin(refs) & (idx.quality >= 0.7) & (idx.duration_s >= MIN_S)]
    R = pool.sample(frac=1, random_state=seed).groupby('speaker_id').head(PER_REF).assign(role='ref', part='ref')
    R = R.rename(columns={'session': 'session_id'})
    out = pd.concat([S[['chunk_id', 'speaker_id', 'session_id', 'part', 'role', 'duration_s']],
                     R[['chunk_id', 'speaker_id', 'session_id', 'part', 'role', 'duration_s']]], ignore_index=True)
    sp = out.chunk_id.map(WQ.span)
    out['session'] = sp.map(lambda x: x[0]); out['start'] = sp.map(lambda x: x[1]); out['end'] = sp.map(lambda x: x[2])
    return out


def verdict(res):
    """Per new speaker: share flagged (closer to another centroid than to its own), the
    flagged test/dev sessions, and whether the speaker passes."""
    rows = []
    for s in NEW:
        r = res[(res.speaker_id == s) & (res.role == 'new')]; f = r[r.margin < 0]
        bad = sorted(set(f[f.part.isin(['test', 'dev'])].session_id.astype(int)))
        rows.append(dict(speaker=s, sampled=len(r), flagged=len(f), share=len(f) / max(len(r), 1),
                         flagged_by_part=f.part.value_counts().to_dict(), test_dev_sessions_flagged=bad,
                         ok=bool(len(f) / max(len(r), 1) <= MAX_FLAGGED and not bad)))
    return pd.DataFrame(rows)


def run():
    import validate_audio as VA
    from huggingface_hub import get_token
    P = pd.read_parquet(PLAN_V4)
    idx = pd.read_parquet(INDEX, columns=['chunk_id', 'speaker_id', 'session', 'duration_s', 'quality'])
    rows = sample(P, idx)
    print(f'{len(rows)} segments: {int((rows.role == "new").sum())} of {NEW}, {rows[rows.role == "ref"].speaker_id.nunique()} reference speakers', flush=True)
    token = get_token()                                      # passed to ffmpeg's header by validate_audio, never printed
    # fetch every span in parallel first (each is a seek into an hours-long m4a over HTTP, ~4 s,
    # and validate_audio fetches them one by one); embed then reads from memory
    from concurrent.futures import ThreadPoolExecutor
    def get(r):
        try: return (r.session, r.start, r.end), VA.fetch_audio(r.session, r.start, r.end, token)
        except Exception: return (r.session, r.start, r.end), None
    with ThreadPoolExecutor(12) as ex: cache = dict(ex.map(get, rows.itertuples()))
    print(f'fetched {sum(v is not None for v in cache.values())}/{len(cache)} spans', flush=True)
    fetch = VA.fetch_audio
    def cached(session, start, end, tok):
        b = cache.get((session, start, end))
        if b is None: raise RuntimeError('span not fetched')
        return b
    VA.fetch_audio = cached
    try: E, keep = VA.embed(rows, token)
    finally: VA.fetch_audio = fetch
    rows = rows.loc[keep].reset_index(drop=True)
    res = pd.concat([rows, VA.contamination(E, rows.speaker_id.values).drop(columns='speaker_id')], axis=1)
    V = verdict(res)
    os.makedirs(OUT, exist_ok=True)
    res.to_parquet(os.path.join(OUT, 'audio_check_run4.parquet'), index=False)
    flagged = res[(res.role == 'new') & (res.margin < 0)].sort_values('margin')
    text = (V.to_string(index=False) + '\n\nreference population: ' + f'{res[res.role == "ref"].speaker_id.nunique()} speakers from the same sessions, '
            f'{(res[res.role == "ref"].margin < 0).mean():.1%} of their segments flagged\n\n'
            + (flagged[['speaker_id', 'part', 'session_id', 'chunk_id', 'own', 'nearest_other', 'nearest_id', 'margin']].to_string(index=False) if len(flagged) else 'no new-speaker chunk flagged'))
    open(os.path.join(OUT, 'audio_report_run4.txt'), 'w', encoding='utf-8').write(text + '\n')
    print(text)
    print('\nPASS' if V.ok.all() else '\nSTOP: add the flagged sessions to word_quality.EXCLUDE_SESSIONS and rebuild plan-v4')
    return V


def _self_check():
    res = pd.DataFrame(dict(speaker_id=[23641] * 20 + [23635] * 20, role='new',
                            part=(['test'] * 5 + ['dev'] * 5 + ['train'] * 10) * 2, session_id=list(range(40)),
                            margin=[0.3] * 19 + [-0.1] + [0.3] * 2 + [-0.2] + [0.3] * 17))
    V = verdict(res).set_index('speaker')
    assert V.at[23641, 'ok'] and V.at[23641, 'flagged'] == 1                  # one train chunk: 5 %, no test/dev session
    assert not V.at[23635, 'ok'] and V.at[23635, 'test_dev_sessions_flagged'] == [22]   # a test chunk flagged
    print('gate_v4 self-check: OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--self-check', action='store_true')
    a = ap.parse_args()
    if a.self_check: _self_check(); sys.exit(0)
    _self_check(); sys.exit(0 if run().ok.all() else 1)
