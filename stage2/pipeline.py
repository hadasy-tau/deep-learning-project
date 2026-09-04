"""Stage 2 base: index -> chunk -> split.  Text only, no GPU, no audio.

Three facts from Step 0 shape this file:
  * The session id is in the filename: speaker_session_start_end.wav
  * The public ct2 dump carries reference_text + segments_json + demographics,
    so gated VoxKnesset is needed for WAVEFORMS ONLY.  There is no `index`
    stage over 275 GB; there is a read_parquet.
  * Only the ct2 dump (model B) has segments_json, so chunk boundaries are
    B-derived.  Accepted: boundary-only bias, see the plan's circularity note.
"""
import glob, json, os, re, unicodedata, wave
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein

MAX_S = 30.0

# Recording-level QC, matching stage1_basic.ipynb exactly. A rate filter cannot
# work here: a segment is a contiguous span of session time and so contains the
# speaker's pauses, and because a rate is a ratio its allowance shrinks toward
# zero for short references. Stage 1 replaced wpm 30-350 (which dropped 309
# recordings, ~246 of them legitimate) with a flat allowance for audio the
# reference cannot account for -- 14 recordings, and it still catches the
# 4.2-hour file carrying a 126-word reference.
SLOW_WPM          = 40            # generous; the corpus averages ~108
SEC_PER_WORD      = 60 / SLOW_WPM
MAX_UNEXPLAINED_S = 300

# Chunk-level guard. NOT a speaking-rate filter: it flags spans where our own
# linear word-time interpolation compressed too much text, not speech the model
# found hard. There is no lower bound on purpose -- a 30 s chunk holding one
# word ("thank you", then a pause) is legitimate speech, which is the case
# Stage 1 established.
#
# 250, not the 350 this started at. Over the 4-speaker panel's 17,998 chunks the
# rate is p50 114, p99 181, max 340 -- so 350 removed nothing at all while
# alignment fell off a cliff well below it:
#
#     <=150 wpm   16783 chunks (93.2%)   mean alignment 0.904
#      >200 wpm      87 chunks ( 0.5%)   mean alignment 0.554
#      >250 wpm      30 chunks ( 0.2%)   mean alignment 0.367
#
# One inspected case at 256 wpm carried a 128-word label over audio holding 69
# words -- the reference slice was about twice too long for its window, which is
# an interpolation artifact and not a fast speaker. 250 drops 0.17% of chunks.
#
# This does not contradict the never-filter-on-alignment rule below: that rule
# protects audio the MODEL found hard, whereas this removes text the audio
# cannot physically contain. The caveat is that chunk wpm derives from B's
# timings, so it is less model-independent than Stage 1's recording-level wpm.
MAX_CHUNK_WPM = 250
SR = 16000
REPO_B = 'Dolevabudi/voxknesset-whisper-large-v3-ct2-inference'

def read_wav(path, start=0.0, end=None):
    """VoxKnesset audio is 16 kHz 16-bit mono WAV, so stdlib `wave` is enough --
    no soundfile/librosa/torchaudio dependency. Returns float32 in [-1, 1]."""
    with wave.open(path, 'rb') as w:
        assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (SR, 2, 1), \
            f'unexpected wav format in {path}'
        w.setpos(min(int(start * SR), w.getnframes()))
        n = w.getnframes() - w.tell() if end is None else int((end - start) * SR)
        raw = w.readframes(max(n, 0))
    return np.frombuffer(raw, '<i2').astype(np.float32) / 32768.0

# ---- from Stage 1, verbatim.  Do not re-derive. ---------------------------
_niqqud = re.compile(r'[֑-ׇ]')
_quotes = re.compile(r'[׳״"\'‘’“”]')
_dashes = re.compile(r'[־\-‐-―_/\\]')
_punct  = re.compile(r'[^\w\s֐-׿]', flags=re.UNICODE)

def normalize_he(s):
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = _niqqud.sub('', s)
    s = _quotes.sub(' ', s)      # -> space, see the note above
    s = _dashes.sub(' ', s)
    s = _punct.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# ---- index ----------------------------------------------------------------
def load_index(local_only=True):
    """Every non-audio field we need, from the public dump."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(REPO_B, repo_type='dataset', allow_patterns=['data/*'],
                          local_files_only=local_only)
    files = [f for f in glob.glob(os.path.join(p, 'data', '*.parquet'))
             if 'results_10' not in os.path.basename(f)]          # 10-row pilot
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # speaker_id in the dump is unreliable (one row is a corrupted int64);
    # the filename is authoritative for both speaker and session.
    parts = df.filename.str.extract(r'^(\d+)_(\d+)_(\d+)_(\d+)\.wav$')
    assert parts[0].notna().all(), 'unparseable filenames'
    df['speaker_id'] = parts[0].astype(int)
    df['session']    = parts[1].astype(int)
    df = df.drop_duplicates('filename').reset_index(drop=True)

    # Same QC as Stage 1, on the same unit, so the two stages agree about which
    # recordings exist at all. Without it Stage 2 would chunk the recordings
    # Stage 1 threw away.
    n_words = df.reference_text.map(lambda t: len(normalize_he(t).split()))
    keep = (df.duration_s - n_words * SEC_PER_WORD) <= MAX_UNEXPLAINED_S
    if not keep.all():
        print(f'QC: dropping {(~keep).sum()} recording(s) with more than '
              f'{MAX_UNEXPLAINED_S}s of audio the reference cannot account for')
    return df[keep].reset_index(drop=True)

# ---- materialize ----------------------------------------------------------
# Gated VoxKnesset is needed for waveforms only.  `audio.path` == filename
# (Step 0 check 3), and parquet stores struct fields as separate columns, so
# the shard index reads paths without pulling a single sample of audio.
REPO_AUDIO = 'datasets/ivrit-ai/VoxKnesset/data/*.parquet'
HERE = os.path.dirname(os.path.abspath(__file__))

def shard_index(cache=os.path.join(HERE, 'shard_index.csv')):
    """filename -> (shard, row_group).  ~12 min once, then cached."""
    if os.path.exists(cache):
        return pd.read_csv(cache)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs, rows = HfFileSystem(), []
    for sh in sorted(fs.glob(REPO_AUDIO)):
        pf = pq.ParquetFile(fs.open(sh))
        md = pf.metadata
        rg = np.repeat(np.arange(md.num_row_groups),
                       [md.row_group(i).num_rows for i in range(md.num_row_groups)])
        paths = [x['path'] for x in pf.read(columns=['audio.path']).column('audio').to_pylist()]
        rows += [(p, os.path.basename(sh), int(g)) for p, g in zip(paths, rg)]
        print(f'  {os.path.basename(sh)}: {len(paths)}', flush=True)
    df = pd.DataFrame(rows, columns=['filename', 'shard', 'row_group'])
    df.to_csv(cache, index=False)
    return df

def materialize(filenames, out_dir):
    """Write the wanted .wav files to out_dir.  Reads only the row groups they live in."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    want = set(filenames)
    idx = shard_index()
    idx = idx[idx.filename.isin(want)]
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for shard, g in idx.groupby('shard'):
        todo = {f for f in g.filename if not os.path.exists(os.path.join(out_dir, f))}
        if not todo:
            continue
        pf = pq.ParquetFile(fs.open(f'datasets/ivrit-ai/VoxKnesset/data/{shard}'))
        for rg in sorted(g[g.filename.isin(todo)].row_group.unique()):
            for a in pf.read_row_group(int(rg), columns=['audio']).column('audio').to_pylist():
                if a['path'] in todo:
                    with open(os.path.join(out_dir, a['path']), 'wb') as fh:
                        fh.write(a['bytes'])
                    n += 1
        print(f'  {shard}: {n} written', flush=True)
    return n

# ---- chunk ----------------------------------------------------------------
def _hyp_word_times(segments):
    """Flat hyp word list + a time for each, interpolated inside its segment."""
    words, times = [], []
    for s in segments:
        w = normalize_he(s.get('text', '')).split()
        if not w:
            continue
        # ponytail: linear interpolation inside the segment, no word timestamps
        # available. Cuts land within a word or two; upgrade to stable-ts word
        # timings only if chunk audio is audibly clipped.
        t = np.linspace(s['start'], s['end'], len(w), endpoint=False)
        words += w
        times += list(t)
    return words, np.array(times)

def chunk_recording(reference_text, segments_json, duration_s, max_s=MAX_S):
    """-> list of {start, end, text, score}: reference sliced into <=max_s pieces."""
    ref = normalize_he(reference_text).split()
    try:
        segs = json.loads(segments_json) or []
    except (TypeError, ValueError):
        segs = []
    if not ref:
        return []

    hyp, hyp_t = _hyp_word_times(segs)
    # Anchor ref words to time via the words the two transcripts agree on.
    anchors_i, anchors_t = [], []
    if hyp:
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
            if tag == 'equal':
                anchors_i += list(range(i1, i2))
                anchors_t += list(hyp_t[j1:j2])
    if len(anchors_i) < 2:                      # no usable agreement: spread evenly
        ref_t = np.linspace(0, duration_s, len(ref), endpoint=False)
    else:
        ref_t = np.interp(np.arange(len(ref)), anchors_i, anchors_t)
    ref_t = np.maximum.accumulate(ref_t)        # time must not run backwards

    out, start_i = [], 0
    while start_i < len(ref):
        lo = float(ref_t[start_i])
        # last word whose start still fits in the window; always take >= 1 word
        i  = max(int(np.searchsorted(ref_t, lo + max_s, side='right')), start_i + 1)
        hi = float(min(ref_t[i] if i < len(ref) else duration_s, lo + max_s, duration_s))
        text = ' '.join(ref[start_i:i])
        if text and hi > lo:
            said = ' '.join(w for w, t in zip(hyp, hyp_t) if lo <= t < hi)
            out.append(dict(start=lo, end=hi, text=text,
                            score=Levenshtein.normalized_similarity(text.split(),
                                                                    said.split())))
        start_i = i
    return out

def chunk_all(df, max_s=MAX_S):
    rows = []
    for r in df.itertuples():
        for c in chunk_recording(r.reference_text, r.segments_json, r.duration_s, max_s):
            rows.append(dict(filename=r.filename, speaker_id=r.speaker_id,
                             session=r.session, age=r.age, **c))
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c['duration_s'] = c.end - c.start
    c['n_words']    = c.text.str.split().str.len()
    c['wpm']        = c.n_words / (c.duration_s / 60)
    # Keep every chunk regardless of `score` (see plan: filtering on alignment
    # quality deletes the hard cases). Recording-level QC already ran in
    # load_index; all that is left here is the interpolation-artifact guard.
    return c[c.wpm <= MAX_CHUNK_WPM].reset_index(drop=True)

# ---- split ----------------------------------------------------------------
def make_splits(chunks, test_min, dev_min=15):
    """Session-disjoint per speaker: latest sessions -> test, then dev, rest train.

    test_min: minutes of personal-test, scalar or {speaker_id: minutes}.
    Sized per speaker from stage0_gate -- a flat 45 min leaves low-WER
    speakers unable to resolve anything short of a halving of WER.
    """
    def one(spk, g):
        want = test_min[spk] if isinstance(test_min, dict) else test_min
        dur  = g.groupby('session').duration_s.sum().sort_index(ascending=False)
        cum  = dur.cumsum() / 60
        test = set(dur.index[cum <= want]) or {dur.index[0]}
        rest = dur.drop(index=list(test))
        dev  = set(rest.index[rest.cumsum() / 60 <= dev_min]) or set(rest.index[:1])
        return g.session.map(lambda s: 'test' if s in test else 'dev' if s in dev else 'train')

    out = chunks.copy()
    if out.empty:
        out['part'] = pd.Series(dtype=object)
        return out
    # Concatenated explicitly rather than through groupby.apply: apply infers
    # its own return shape, and on pandas 3 a SINGLE group comes back as a
    # 1xN DataFrame rather than a Series -- so a one-speaker call (which is
    # exactly what the smoke-test notebook does) failed where the four-speaker
    # panel worked. Nothing here should depend on how many speakers there are.
    parts = [one(spk, g) for spk, g in out.groupby('speaker_id')]
    out['part'] = pd.concat(parts).reindex(out.index)
    return out


def budget_order(train_chunks):
    """Train chunks ordered latest session first, so D4's nested budgets
    (1,2,5,10,20,40,80 min) are prefixes of one ordering and nested by
    construction.  Latest-first keeps train temporally closest to dev/test."""
    return train_chunks.sort_values(['session', 'start'], ascending=[False, True])

def files_needed(splits, train_min=80):
    """Recordings we must download: dev + test + the largest training budget.
    Materializing by split rather than by speaker is the difference between
    ~22 h and the panel's full 136 h of audio."""
    keep = [splits[splits.part.isin(['dev', 'test'])]]
    for _, g in splits[splits.part == 'train'].groupby('speaker_id'):
        g = budget_order(g)
        keep.append(g[g.duration_s.cumsum() / 60 <= train_min])
    return sorted(set(pd.concat(keep).filename))

# ---- self-check -----------------------------------------------------------
if __name__ == '__main__':
    idx = load_index()
    print(f'index: {len(idx):,} recordings, {idx.speaker_id.nunique()} speakers')

    sample = idx[idx.speaker_id.isin([11835, 528, 1057, 4416])]

    # no reference word may be lost or reordered by chunking (plan section 08)
    for r in sample.head(300).itertuples():
        cs = chunk_recording(r.reference_text, r.segments_json, r.duration_s)
        assert ' '.join(c['text'] for c in cs) == normalize_he(r.reference_text), r.filename

    ch = chunk_all(sample)
    print(f'chunks: {len(ch):,} from {len(sample):,} recordings')

    assert (ch.duration_s <= MAX_S + 1e-6).all(), 'chunk longer than the Whisper window'
    assert ch.start.ge(0).all() and (ch.end > ch.start).all()
    for f, g in ch.groupby('filename'):
        g = g.sort_values('start')
        assert (g.start.values[1:] >= g.end.values[:-1] - 1e-6).all(), f'overlap in {f}'

    sp = make_splits(ch, test_min=45)
    for s, g in sp.groupby('speaker_id'):
        by = {p: set(x.session) for p, x in g.groupby('part')}
        for a, b in (('train', 'test'), ('train', 'dev'), ('dev', 'test')):
            assert not (by.get(a, set()) & by.get(b, set())), f'{a}/{b} session overlap, spk {s}'
    print('splits: session-disjoint OK')
    print(sp.groupby(['speaker_id', 'part']).duration_s.sum().div(60).round(1).to_string())
    print(f'\nalignment score: median {ch.score.median():.3f}  '
          f'p10 {ch.score.quantile(.1):.3f}  (kept unfiltered, by design)')

    # Only if some audio has been materialized: chunk times come from the dump's
    # duration_s, so a disagreement with the real waveform would slice past EOF.
    have = os.path.isdir('audio') and os.listdir('audio')
    if have:
        got = ch[ch.filename.isin(os.listdir('audio'))]
        for r in got.itertuples():
            with wave.open(os.path.join('audio', r.filename), 'rb') as w:
                real = w.getnframes() / w.getframerate()
            assert r.end <= real + 0.05, f'{r.filename}: chunk ends past EOF'
            assert len(read_wav(os.path.join('audio', r.filename), r.start, r.end)) > 0
        print(f'audio: {got.filename.nunique()} file(s) checked, '
              f'{len(got)} chunk slices in bounds')
