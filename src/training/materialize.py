"""Materialize the panel: the audio training and evaluation read, as WAV files on disk.

train.py and evaluate.py open audio with common.read_wav(filename, start, end):
16 kHz 16-bit mono WAV files in a folder, named by a table.  The corpus
(Hadasy/knesset-committees-chunks) holds its audio as FLAC bytes inside 410
parquet shards of ~770 MB.  This module bridges the two, in two commands that
run on different machines:

  plan      laptop, seconds.  For every panel speaker (src/evaluation/outputs/
            committees_panel.csv) take the NEWEST sessions at quality >= 0.7 until
            they cover personal-test + dev + the 80-minute budget; split them
            session-disjoint by DATE (test = newest, then dev, rest train); order
            train latest-session-first so nested budgets are prefixes.  Writes
            src/training/panel_plan.parquet -- one row per chunk, small, committed,
            so the choice is reviewable and the extraction is deterministic.
  extract   any machine with a fast link.  Downloads each shard the plan needs,
            pulls only the plan's chunks (DuckDB, 1.75 GB peak), decodes FLAC ->
            WAV under outputs/panel_audio/<speaker_id>/<chunk_id>.wav, deletes the
            shard.  ~80 GB pass through, ~3 GB stay.  Resumable: existing WAVs are
            skipped, so a dropped link costs one shard.
  verify    the split invariants (no session on two sides, test/dev/train sizes,
            nested budgets are prefixes) and every WAV present and readable.
  upload / download   the WAV folder to / from a private HF dataset, so the GPU
            box fetches 3 GB instead of re-pulling 80 GB.  upload is never run
            unasked.

The plan's `session` column is a DATE-SORTABLE key, 'YYYY-MM-DD_<session id>',
because train.py's take_budget orders train by `session` descending and
docs/committees_handoff.md warns that session ids and dates diverge per speaker;
`session_id` keeps the number.  `start`/`end` are offsets INSIDE the chunk's own
WAV (0 and its duration), which is what read_wav slices by.

    python src/training/materialize.py plan
    python src/training/materialize.py extract [--max-shards N] [--prefetch 2]
    python src/training/materialize.py verify
"""
import argparse, io, os, re, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(ROOT, 'src', 'inference'))
from common import make_splits, read_wav, SR

PANEL  = os.path.join(ROOT, 'src', 'evaluation', 'outputs', 'committees_panel.csv')
INDEX  = os.path.join(ROOT, 'src', 'inference', 'cache', 'index.parquet')
SUBSET = os.path.join(ROOT, 'src', 'inference', 'subset_stage1.parquet')
PLAN   = os.path.join(HERE, 'panel_plan.parquet')
AUDIO  = os.path.join(HERE, 'outputs', 'panel_audio')          # git-ignored (src/training/outputs/)
DL     = os.path.join(HERE, 'outputs', '_shards')

MIN_QUALITY, DEV_MIN, BUDGET_MIN, TEST_FLOOR_MIN = 0.7, 15, 80, 45

def log(msg):
    print(time.strftime('[%H:%M:%S] ') + msg, flush=True)

# ---- plan ---------------------------------------------------------------------
def _offset_s(chunk_id):
    m = re.match(r'^\d+_\d+_(\d+)_(\d+)\.flac$', chunk_id)
    return int(m.group(1)) / 1000 if m else 0.0

def plan(panel_path=PANEL, index_path=INDEX, out=PLAN, budget_min=BUDGET_MIN, dev_min=DEV_MIN):
    panel = pd.read_csv(panel_path, index_col=0); core = panel[~panel.profile.str.contains('alt')]
    test_min = {sid: max(float(r.personal_test_h_recommended) * 60, TEST_FLOOR_MIN) for sid, r in core.iterrows()}
    idx = pd.read_parquet(index_path, columns=['chunk_id', 'speaker_id', 'session', 'session_date', 'knesset', 'duration_s', 'quality', 'text', 'shard'])
    idx = idx[idx.speaker_id.isin(core.index) & (idx.quality >= MIN_QUALITY)].copy()
    sub_ids = set(pd.read_parquet(SUBSET, columns=['chunk_id']).chunk_id)
    def fill(mins, want):
        """make_splits takes the sessions whose cumulative minutes stay <= the target,
        i.e. it stops BEFORE the session that crosses it.  Sessions here are long
        (often 20-40 min), so a 45-minute target can yield a 22-minute test set.
        Return the cumulative at the first session that reaches `want`, so the
        crossing session is inside; a hair less than the next cumulative, so the
        one after it is not."""
        cum = np.cumsum(mins)
        k = int(np.argmax(cum >= want)) if (cum >= want).any() else len(cum) - 1
        return float(cum[k]) + 1e-6
    rows = []
    for sid, d in idx.groupby('speaker_id'):
        per = d.groupby('session').agg(date=('session_date', 'first'), dur=('duration_s', 'sum')).sort_values(['date', 'session'], ascending=False)
        mins = (per.dur / 60).values
        take = []
        for n in range(1, len(per) + 1):                 # newest sessions first; keep adding until train has the budget
            take = list(per.index[:n])
            t_min = fill(mins[:n], test_min[sid])
            rest = mins[:n][np.cumsum(mins[:n]) > t_min - 1e-6]      # sessions not in test, newest first
            d_min = fill(rest, dev_min) if len(rest) else dev_min
            sel = d[d.session.isin(take)].copy(); sel['session_id'] = sel.session
            sel['session'] = sel.session_date + '_' + sel.session_id.astype(str)
            sp = make_splits(sel, test_min={sid: t_min}, dev_min=d_min)
            if sp[sp.part == 'train'].duration_s.sum() / 60 >= budget_min: break
        rows.append(sp)
    P = pd.concat(rows, ignore_index=True)
    P['session_offset_s'] = P.chunk_id.map(_offset_s)
    P = P.sort_values(['speaker_id', 'session', 'session_offset_s'], ascending=[True, False, True], kind='stable').reset_index(drop=True)
    P['filename'] = P.speaker_id.astype(str) + '/' + P.chunk_id.str.replace('.flac', '.wav', regex=False)
    P['start'], P['end'] = 0.0, P.duration_s
    P['in_subset'] = P.chunk_id.isin(sub_ids)
    tr = P[P.part == 'train']
    P['train_order'] = np.nan; P.loc[tr.index, 'train_order'] = tr.groupby('speaker_id').cumcount().astype(float)
    P['budget_cum_min'] = np.nan; P.loc[tr.index, 'budget_cum_min'] = tr.groupby('speaker_id').duration_s.cumsum() / 60
    cols = ['chunk_id', 'speaker_id', 'session', 'session_id', 'session_date', 'knesset', 'session_offset_s', 'duration_s', 'quality',
            'text', 'shard', 'part', 'filename', 'start', 'end', 'in_subset', 'train_order', 'budget_cum_min']
    P = P[cols]; P.to_parquet(out, index=False)
    log(f'plan: {len(P):,} chunks, {P.duration_s.sum()/3600:.1f} h, {P.speaker_id.nunique()} speakers, {P.shard.nunique()} shards -> {out}')
    return P

def plan_summary(P):
    t = P.groupby(['speaker_id', 'part']).duration_s.sum().div(60).unstack().round(1)
    t['sessions'] = P.groupby('speaker_id').session.nunique(); t['chunks'] = P.groupby('speaker_id').size()
    t['test_in_subset'] = P[(P.part == 'test')].groupby('speaker_id').in_subset.mean().round(2)
    return t[['test', 'dev', 'train', 'sessions', 'chunks', 'test_in_subset']]

# ---- extract ------------------------------------------------------------------
def _fetch(shard, local_dir):
    import data as D
    from huggingface_hub import hf_hub_download
    for attempt in range(6):
        try:
            return hf_hub_download(D.REPO, f'data/{shard}', repo_type='dataset', revision=D.revision(), local_dir=local_dir)
        except Exception as e:
            log(f'  {shard}: download attempt {attempt+1} failed ({e!r}'[:140] + ')'); time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'{shard}: download failed 6 times')

def _write_wav(path, flac_bytes):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(flac_bytes), dtype='int16')
    if data.ndim > 1: data = data[:, 0]
    assert sr == SR, f'{path}: {sr} Hz, expected {SR}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, SR, subtype='PCM_16')
    return len(data) / SR

def extract(plan_path=PLAN, audio_dir=AUDIO, max_shards=None, prefetch=2, shards=None):
    from run_lean import shard_rows
    P = pd.read_parquet(plan_path)
    P['path'] = P.filename.map(lambda f: os.path.join(audio_dir, f))
    todo = P[~P.path.map(os.path.exists)]
    by_shard = {s: g for s, g in todo.groupby('shard')}
    order = sorted(by_shard) if shards is None else [s for s in shards if s in by_shard]
    if max_shards: order = order[:max_shards]
    log(f'extract: {len(P):,} chunks in the plan, {len(P)-len(todo):,} already on disk, {len(todo):,} to write from {len(order)} shards')
    os.makedirs(DL, exist_ok=True); pool = ThreadPoolExecutor(prefetch)
    ahead = {s: pool.submit(_fetch, s, DL) for s in order[:prefetch]}
    written = 0; t0 = time.time()
    for k, shard in enumerate(order, 1):
        path = ahead.pop(shard).result()
        nxt = k - 1 + prefetch
        if nxt < len(order) and order[nxt] not in ahead: ahead[order[nxt]] = pool.submit(_fetch, order[nxt], DL)
        g = by_shard[shard]; want = set(g.chunk_id)
        blobs = shard_rows(path, want)
        missing = want - set(blobs)
        if missing: log(f'  {shard}: {len(missing)} plan chunks not in the shard (!)')
        for r in g.itertuples():
            if r.chunk_id in blobs:
                dur = _write_wav(r.path, blobs[r.chunk_id]); written += 1
                if abs(dur - r.duration_s) > 0.05: log(f'  {r.chunk_id}: wav {dur:.2f}s vs index {r.duration_s:.2f}s')
        del blobs
        try: os.remove(path)
        except OSError: pass
        if k % 5 == 0 or k == len(order):
            el = (time.time() - t0) / 60; log(f'  {k}/{len(order)} shards | {written:,} wavs written | {el:.0f} min, eta {el / k * (len(order) - k):.0f} min')
    pool.shutdown(); shutil.rmtree(DL, ignore_errors=True)
    log(f'extract done: {written:,} wavs written to {audio_dir}')
    return written

# ---- verify -------------------------------------------------------------------
def verify(plan_path=PLAN, audio_dir=AUDIO, budget_min=BUDGET_MIN, check_audio=True, sample=200):
    P = pd.read_parquet(plan_path); ok = True
    def chk(cond, msg):
        nonlocal ok; ok &= bool(cond); print(f'  [{"OK " if cond else "FAIL"}] {msg}')
    print('=== plan ===')
    for sid, g in P.groupby('speaker_id'):
        by = {p: set(x.session) for p, x in g.groupby('part')}
        chk(not (by.get('train', set()) & by.get('test', set())) and not (by.get('train', set()) & by.get('dev', set())) and not (by.get('dev', set()) & by.get('test', set())),
            f'{sid}: no session on two sides')
        mins = g.groupby('part').duration_s.sum() / 60
        chk(mins.get('test', 0) >= TEST_FLOOR_MIN * 0.9 and mins.get('train', 0) >= budget_min, f'{sid}: test {mins.get("test", 0):.0f} min, dev {mins.get("dev", 0):.0f}, train {mins.get("train", 0):.0f}')
        dates = g.groupby('part').session_date.agg(['min', 'max'])
        chk(dates.loc['train', 'max'] <= dates.loc['test', 'min'], f'{sid}: train ends {dates.loc["train", "max"]} before test starts {dates.loc["test", "min"]}')
        tr = g[g.part == 'train'].sort_values('train_order')
        chk((tr.budget_cum_min.diff().fillna(tr.budget_cum_min).round(6) == (tr.duration_s / 60).round(6)).all() and (tr.session.values[:-1] >= tr.session.values[1:]).all(),
            f'{sid}: train order is latest-session-first; nested budgets are prefixes')
    chk((P.start == 0).all() and np.allclose(P.end, P.duration_s), 'start/end are offsets inside each chunk wav')
    if not check_audio:
        return ok
    print('=== audio ===')
    have = P.filename.map(lambda f: os.path.exists(os.path.join(audio_dir, f)))
    chk(have.all(), f'{int(have.sum()):,} of {len(P):,} wavs present under {audio_dir}')
    if have.any():
        rng = np.random.default_rng(0); pick = P[have].sample(min(sample, int(have.sum())), random_state=0)
        bad = 0
        for r in pick.itertuples():
            w = read_wav(os.path.join(audio_dir, r.filename), r.start, r.end)
            bad += abs(len(w) / SR - r.duration_s) > 0.05 or float(np.abs(w).max()) == 0.0
        chk(bad == 0, f'{len(pick)} sampled wavs read by common.read_wav with the right duration and non-silent audio ({bad} bad)')
    print('RESULT:', 'ALL CHECKS PASSED' if ok else 'FAILED')
    return ok

# ---- move the folder ----------------------------------------------------------
def upload(repo, audio_dir=AUDIO, plan_path=PLAN):
    from huggingface_hub import HfApi
    api = HfApi(); api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=plan_path, path_in_repo='panel_plan.parquet', repo_id=repo, repo_type='dataset')
    api.upload_folder(folder_path=audio_dir, path_in_repo='panel_audio', repo_id=repo, repo_type='dataset')
    log(f'uploaded plan + wavs to {repo}')

def download(repo, audio_dir=AUDIO):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type='dataset', allow_patterns=['panel_audio/*', 'panel_plan.parquet'], local_dir=os.path.dirname(audio_dir))
    log(f'downloaded {repo} under {os.path.dirname(audio_dir)}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan'); e = sub.add_parser('extract'); e.add_argument('--max-shards', type=int); e.add_argument('--prefetch', type=int, default=2); e.add_argument('--shards', nargs='+')
    v = sub.add_parser('verify'); v.add_argument('--no-audio', action='store_true')
    for c in ('upload', 'download'): sub.add_parser(c).add_argument('--repo', required=True)
    a = ap.parse_args()
    if a.cmd == 'plan': print(plan_summary(plan()).to_string())
    elif a.cmd == 'extract': extract(max_shards=a.max_shards, prefetch=a.prefetch, shards=a.shards)
    elif a.cmd == 'verify': sys.exit(0 if verify(check_audio=not a.no_audio) else 1)
    elif a.cmd == 'upload': upload(a.repo)
    elif a.cmd == 'download': download(a.repo)
