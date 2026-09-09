"""Preprocessing: build a corpus of <= 30 s audio chunks, each holding exactly
one speaker, from the Knesset committees recordings.

Two inputs, one output.  `Dolevabudi/knesset-committees-speakers` publishes an
*index* -- one row per aligned segment, carrying a global Knesset PersonID and
demographics, but no audio at all (its `filename` column names files that do not
exist).  `ivrit-ai/knesset-committees` holds the audio, one `audio.m4a` per
session.  This module joins them: it walks the index, streams each session's
m4a, decodes it once, cuts the chunks out by sample index, and throws the m4a
away.

Deliberately blind to what consumes it.  No speaker selection, no hours budget,
no train/dev/test split, no consumer-specific column names.  Those are filters
over a corpus, not properties of one.

Modelled on ivrit-ai/asr-training/create_dataset.py (`generate_slices`,
`merge_slice_segments`), with one necessary change: their slicer packs segments
into fixed 30 s windows regardless of who is speaking, so a slice can straddle a
speaker change.  Fine for a training corpus, fatal for per-speaker measurement.
Here the turn boundary is a hard wall.

Facts that shape this file (all measured 2026-09-08 against the live data):
  * The index is 5,158,763 rows.  Assigned rows (`speaker_id > 0`, i.e. label in
    {mk, former_mk}) are 2,295,972 / 3,397 h / 339 speakers / 10,889 sessions.
  * A turn is a maximal run of one speaker -- same session, same
    `local_speaker_id`, same `speaker_id` -- and it MUST be derived over all
    rows.  An unresolved guest speaking between two of X's segments breaks X's
    turn; deriving runs from the assigned subset alone merges across them.
  * 935,366 assigned runs -> ~1,190,302 chunks, ~4,109 h of emitted audio, over
    339 speakers and 10,889 sessions.  No run holds two speakers (checked).
  * Segment durations are already within [1.0, 30.0] s, with exactly one
    exception among assigned rows (117.96 s at quality 0.0).  Nothing here needs
    to split a segment; the one outlier is skipped rather than truncated.
  * m4a is 100 kbps modal (45 MB per audio-hour); median session 86 MB, max
    411 MB.  ~612 GB moves across a full run -- cumulatively, never resident.
  * `int(round(t * SR))`, never `int(t * SR)`: index times carry 2 decimals and
    `int(976.62 * 16000)` truncates one sample low.
  * ffmpeg is not a declared dependency of this repo and may be absent.  Only
    the build path may require it; planning and self-checks must not.

Decisions:
  * FLAC, lossless.  The source is already 100 kbps AAC; a second lossy
    generation would put a codec confound inside any error measurement, and mp3
    specifically would favour the model trained on ivrit.ai's own mp3 pipeline.
  * Quality is stored, never filtered.  Every chunk carries its alignment
    confidence so a consumer can set its own floor and measure what it cost.
  * A chunk's audio ends at its last packed segment, not at seek + 30 s.
    ivrit.ai always grab a full window; here that would run past the turn
    boundary into the next speaker and break the one-speaker guarantee.
"""
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import get_token, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
CACHE = os.path.join(HERE, 'cache')
PARTS = os.path.join(OUT, 'parts')
M4A_DIR = os.path.join(CACHE, 'm4a')

INDEX_REPO = 'Dolevabudi/knesset-committees-speakers'
AUDIO_REPO = 'ivrit-ai/knesset-committees'

SR = 16000                 # Whisper's input rate; the only rate this module emits
MAX_S = 30.0               # Whisper's window, and ivrit.ai's slice_length
MIN_S = 1.0                # the index's own floor; a chunk can never be shorter
MERGE_GAP = 0.3            # ivrit.ai's merge_below_gap_threshold
WORKERS = 4                # network- and RAM-bound.  stage3 uses 6 for 3 MB JSONs;
                           # here each worker holds a whole decoded session in RAM.
MAX_SESSION_S = 8 * 3600   # decode guard: 8 h of PCM is 921 MB in one worker
BYTES_PER_S = SR * 2       # 32,000 B/s of 16 kHz 16-bit mono
FLAC_RATIO = 0.55          # measured typical FLAC ratio for 16 kHz mono speech
MB_PER_S = 0.0125          # 100 kbps, the modal m4a encode; only a fallback for
                           # estimate_cost() when session_bytes() is cold

# Columns needed to find turn boundaries and describe a chunk.  reference_text is
# 319 MB of the index's 520 MB, so it is fetched separately and only for the rows
# that survive turn selection.
LIGHT_COLS = ['filename', 'speaker_id', 'local_speaker_id', 'session', 'start', 'end', 'duration_s',
              'quality', 'label', 'speaker_name', 'knesset', 'committee_name', 'session_date',
              'gender', 'age', 'date_of_birth', 'place_of_birth', 'year_of_aliya',
              'religion', 'nationality', 'religious_orientation']

DEMO = ['gender', 'age', 'date_of_birth', 'place_of_birth', 'year_of_aliya',
        'religion', 'nationality', 'religious_orientation']

# Explicit output schema.  Without it pyarrow infers per part and the parts
# disagree: year_of_aliya is '1950' for some speakers and null for others, so a
# part of only-null values infers double and fails to concatenate with a part
# that has strings.  `audio` is excluded -- it is a struct of raw bytes and is
# reattached after coercion.
#
# Only speaker_id and session are plain int64: they identify the row and cannot
# be absent.  Every other integer is nullable Int64, so one malformed row raises
# where it is built rather than taking down the whole part at cast time.
SCHEMA = {'chunk_id': 'string', 'speaker_id': 'int64', 'speaker_name': 'string',
          'session': 'int64', 'session_date': 'string', 'knesset': 'Int64',
          'committee_name': 'string', 'seek': 'float64', 'duration_s': 'float64',
          'abs_start': 'float64', 'abs_end': 'float64', 'text': 'string',
          'transcript': 'string', 'prev_transcript': 'string', 'has_prev': 'boolean',
          'has_timestamps': 'boolean', 'quality': 'float64', 'n_words': 'Int64',
          'n_segments': 'Int64', 'gender': 'Int64', 'age': 'float64',
          'date_of_birth': 'string', 'place_of_birth': 'string', 'year_of_aliya': 'string',
          'religion': 'string', 'nationality': 'string', 'religious_orientation': 'string'}


# ---- small helpers --------------------------------------------------------
def _cached(name, fn):
    """JSON cache beside the module.  Copied from stage3/manifest.py rather than
    imported: stage3 exists only on branch pr12, so importing it would make this
    module unloadable on main."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    val = fn()
    json.dump(val, open(path, 'w', encoding='utf-8'))
    return val


def coerce(df):
    """One dtype per column, so every part file shares a schema.  `audio` rides
    through untouched."""
    audio = df['audio'] if 'audio' in df else None
    if df.empty:
        out = pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})
        out['audio'] = pd.Series(dtype='object')
        return out
    for c, t in SCHEMA.items():
        if c not in df:
            df[c] = None
        if t == 'string':
            df[c] = df[c].astype('string')
        elif t == 'boolean':
            df[c] = df[c].astype('boolean')
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(t)
    out = df[list(SCHEMA)]
    if audio is not None:
        out = out.assign(audio=audio)
    return out


# ---- the index ------------------------------------------------------------
def index_path():
    """The published index, downloaded once (520 MB) and cached by huggingface_hub.
    Reading it locally beats streaming: the file has only 5 row groups, so a
    per-session `filters=` read would pull a ~100 MB row group each time."""
    return hf_hub_download(INDEX_REPO, 'segments.parquet', repo_type='dataset',
                           token=get_token())


def load_index(columns=None, with_text=False):
    """-> DataFrame of the whole index, projected.  Strings come back as
    categories; without that cast the frame is several GB."""
    cols = list(columns or LIGHT_COLS)
    if with_text and 'reference_text' not in cols:
        cols.append('reference_text')
    tbl = pq.read_table(index_path(), columns=cols)
    tbl = tbl.cast(pa.schema([f.with_type(pa.dictionary(pa.int32(), pa.string()))
                              if pa.types.is_large_string(f.type) or pa.types.is_string(f.type)
                              else f for f in tbl.schema]))
    return tbl.to_pandas()


def attach_text(rows):
    """Add `reference_text` to a subset of index rows, keyed on `filename`.

    Read one row group at a time and keep only the wanted rows: the text column
    is 319 MB of the index's 520 MB, and a build only ever needs the fraction of
    it belonging to the sessions in hand.  The index's filenames are unique, so
    the merge cannot fan out."""
    want = set(rows.filename)
    pf = pq.ParquetFile(index_path())
    got = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=['filename', 'reference_text']).to_pandas()
        t['filename'] = t.filename.astype(str)
        got.append(t[t.filename.isin(want)])
    text = pd.concat(got, ignore_index=True)
    out = rows.assign(filename=rows.filename.astype(str)).merge(text, on='filename', how='left')
    assert len(out) == len(rows), 'filename is not unique in the index'
    return out


def add_turns(df):
    """Label maximal runs of one speaker, contiguous and homogeneously resolved.

    Computed over EVERY row, including unresolved and non-MK ones: a guest
    speaking between two of X's segments ends X's turn, and deriving runs from
    the assigned subset alone merges straight across them.

    The run also breaks when `speaker_id` changes, which matters more than it
    looks.  49,679 turns contain a mix of assigned and unassigned segments from
    the SAME local speaker -- stage3 labels per word and Path A only assigns
    when the protocol's gold speaker, the name and the seat all agree, so a
    `name_disagree` segment stays unresolved mid-turn.  Without this break the
    tiling runs straight across that gap and the chunk's audio carries speech
    whose attribution was deliberately refused, with no reference text covering
    it.  Measured: the break excludes 205 h of exactly that audio.
    """
    df = df.sort_values(['session', 'start'], kind='mergesort').reset_index(drop=True)
    brk = ((df.local_speaker_id != df.local_speaker_id.shift())
           | (df.session != df.session.shift())
           | (df.speaker_id != df.speaker_id.shift()))
    return df.assign(turn=brk.cumsum())


def assigned(df):
    """Rows whose speaker resolved to a Knesset PersonID (label mk / former_mk).
    speaker_id > 0 is the same set and does not need the label column."""
    return df[df.speaker_id > 0]


# ---- slicing --------------------------------------------------------------
def _ts(seconds):
    """Whisper timestamp token, rounded to the model's 0.02 s grid."""
    if not 0 <= seconds <= MAX_S + 1e-9:
        raise ValueError(f'timestamp out of range: {seconds}')
    return f'<|{0.02 * round(seconds / 0.02):.2f}|>'


def _merge(segs, gap=MERGE_GAP):
    """ivrit.ai's merge_slice_segments: fold a segment into its predecessor when
    the silence between them is shorter than `gap`.  Fewer, longer timestamp
    spans; the audio is untouched."""
    out = []
    for s in segs:
        if out and s['start'] - out[-1]['end'] < gap:
            prev = out[-1]
            prev['end'] = s['end']
            prev['text'] = f"{prev['text']} {s['text']}".strip()
            prev['parts'] = prev['parts'] + s['parts']
        else:
            out.append(dict(s, parts=list(s['parts'])))
    return out


def slice_turn(segs, max_s=MAX_S):
    """One speaker's turn -> a list of <= max_s chunks.

    ivrit.ai's generate_slices, with their restart semantics kept: a window runs
    from `seek`, packs whole segments, and closes early when the next segment
    would cross its edge, so the following window starts exactly where the last
    packed segment ended.  Windows therefore tile the turn back-to-back, with no
    gaps and no overlap.

    `segs` must be one speaker's segments, sorted by start.  Each is a dict with
    start, end, text, quality.
    """
    hi = segs[-1]['end']
    out, i, seek = [], 0, segs[0]['start']
    while i < len(segs) and seek < hi:
        # A segment longer than a whole window can never be packed.  Measured:
        # exactly one assigned segment in the corpus needs this (117.96 s, at
        # quality 0.0), so it is a live path -- rare, but not dead code.
        if segs[i]['end'] - segs[i]['start'] > max_s:
            i += 1
            seek = segs[i]['start'] if i < len(segs) else hi
            continue

        edge = min(seek + max_s, hi)
        picked = []
        while i < len(segs) and segs[i]['start'] < edge:
            if segs[i]['end'] <= edge:
                picked.append(segs[i])
                i += 1
            else:
                break                       # crosses the edge; this window closes here
        if not picked:
            # The only candidate starts inside the window and ends past it.
            # Restart the window at its start rather than emitting an empty one.
            # This always advances, because a segment starting exactly at `seek`
            # and shorter than max_s would have fitted.
            seek = segs[i]['start']
            continue

        # The chunk ends at its last packed segment, NOT at seek + max_s.  A full
        # window would run past the turn into the next speaker's audio.
        out.append({'seek': seek, 'end': picked[-1]['end'],
                    'segments': _merge([dict(s, parts=[s]) for s in picked])})
        seek = picked[-1]['end']
    return out


def _weighted_median(values, weights):
    """Duration-weighted median.  ivrit.ai take the median over every word
    probability in the slice; the index publishes one median per segment, so
    this weights those by segment duration -- an approximation of their
    definition, and the closest one the index supports."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    if w.sum() <= 0:
        return float(np.median(v)) if len(v) else 0.0
    if len(v) == 1:
        return float(v[0])
    cum = np.cumsum(w) - 0.5 * w            # weight midpoints
    return float(np.interp(0.5 * w.sum(), cum, v))


def chunk_rows(turn_df, pcm, session_meta):
    """One turn -> chunk dicts with audio attached.  Returns (rows, dropped)."""
    segs = [{'start': float(r.start), 'end': float(r.end), 'quality': float(r.quality),
             'text': ' '.join(str(r.reference_text or '').split())}
            for r in turn_df.itertuples()]
    segs = [s for s in segs if s['text']]
    if not segs:
        return [], 0

    first = turn_df.iloc[0]
    # Cheap, and it is the invariant the corpus is built to hold.  add_turns
    # breaks runs on speaker_id, so a violation here means the caller grouped by
    # something other than `turn`.
    assert turn_df.speaker_id.nunique() == 1, f'run spans {turn_df.speaker_id.nunique()} speakers'
    rows, dropped, prev_transcript = [], 0, ''
    for ch in slice_turn(segs):
        seek, end = ch['seek'], ch['end']
        i0, i1 = int(round(seek * SR)), int(round(end * SR))
        if i1 > len(pcm) + int(0.05 * SR):
            # The m4a is shorter than the alignment claims.  Never emit an index
            # row for audio that does not exist.
            dropped += 1
            prev_transcript = ''
            continue
        clip = pcm[i0:min(i1, len(pcm))]
        if len(clip) < MIN_S * SR:
            dropped += 1
            prev_transcript = ''
            continue

        parts = ch['segments']
        transcript = ''.join(f"{_ts(s['start'] - seek)}{s['text']}{_ts(s['end'] - seek)}"
                             for s in parts)
        text = ' '.join(s['text'] for s in parts)
        flat = [p for s in parts for p in s['parts']]
        quality = _weighted_median([p['quality'] for p in flat],
                                   [p['end'] - p['start'] for p in flat])
        dur = len(clip) / SR
        rows.append({
            'audio': {'bytes': encode_flac(clip),
                      'path': _chunk_id(first.speaker_id, first.session, seek, end)},
            'chunk_id': _chunk_id(first.speaker_id, first.session, seek, end),
            'speaker_id': int(first.speaker_id), 'speaker_name': first.speaker_name,
            'session': int(first.session), 'seek': seek, 'duration_s': dur,
            'abs_start': seek, 'abs_end': seek + dur,
            'text': text, 'transcript': transcript, 'prev_transcript': prev_transcript,
            'has_prev': bool(prev_transcript), 'has_timestamps': True,
            'quality': quality, 'n_words': len(text.split()), 'n_segments': len(flat),
            **{k: first[k] for k in DEMO}, **session_meta,
        })
        prev_transcript = transcript
    return rows, dropped


def _chunk_id(speaker_id, session, start, end):
    """{speaker_id}_{session}_{start_ms}_{end_ms}.flac -- the index's own 4-field
    convention, with a suffix that matches the bytes we actually store."""
    return f'{int(speaker_id)}_{int(session)}_{int(round(start * 1000))}_{int(round(end * 1000))}.flac'


# ---- audio ----------------------------------------------------------------
def ffmpeg_path():
    """ffmpeg is not a declared dependency of this repo, so find it or say how."""
    for cand in (os.environ.get('FFMPEG'), shutil.which('ffmpeg')):
        if cand and (os.path.exists(cand) or shutil.which(cand)):
            return cand
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        'ffmpeg not found.  Install one of:\n'
        '  winget install Gyan.FFmpeg\n'
        '  conda install -c conda-forge ffmpeg\n'
        '  pip install imageio-ffmpeg\n'
        'or point $FFMPEG at the binary.')


def _get(url, dest, tries=6, token=None):
    """Stream to `dest`, verifying the transfer completed.

    The retry shape is stage3/protocol.py::_dl's -- capped jittered backoff, 404
    fails fast -- but the body streams to disk instead of into memory, because
    an m4a is 86 MB at the median and 411 MB at the worst.

    Content-Length is verified before the rename.  A stalled CDN connection
    produces a SHORT BUT SUCCESSFUL download, and ffmpeg decodes a truncated m4a
    happily; every chunk after the cut would then be silence or the wrong audio,
    with nothing to signal it.  Verifying turns that into a loud retryable error.
    """
    token = token or get_token()
    tmp = dest + '.part'
    last = None
    for i in range(tries):
        try:
            with requests.get(url, headers={'Authorization': f'Bearer {token}'},
                              stream=True, timeout=300) as r:
                if r.status_code == 404:
                    r.raise_for_status()
                if r.ok:
                    want = int(r.headers.get('Content-Length', 0))
                    n = 0
                    with open(tmp, 'wb') as fh:
                        for block in r.iter_content(1 << 20):
                            fh.write(block)
                            n += len(block)
                    if want and n != want:
                        raise IOError(f'truncated: got {n} of {want} bytes')
                    os.replace(tmp, dest)
                    return dest
                last = requests.HTTPError(f'{r.status_code} for {url}')
        except (requests.RequestException, IOError) as e:
            last = e
        if i < tries - 1:
            time.sleep(min(2 ** i, 30) * (1 + random.random()))
    raise last


def fetch_m4a(session, dest_dir=M4A_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f'{session}.m4a')
    if os.path.exists(dest):
        return dest
    url = f'https://huggingface.co/datasets/{AUDIO_REPO}/resolve/main/{session}/audio.m4a'
    return _get(url, dest)


def decode_session(path):
    """The whole session, once -> int16 mono 16 kHz.

    `-f s16le`, not `-f wav`: a WAV written to a pipe carries an unknown size
    field, whereas raw PCM needs no header parsing and `len(pcm) / SR` *is* the
    duration.  `-map 0:a:0 -vn` because some of these m4a carry cover art.
    `-nostdin` so a background run cannot eat the notebook's stdin.
    """
    cmd = [ffmpeg_path(), '-nostdin', '-v', 'error', '-i', path,
           '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SR),
           '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, '<i2')
    if len(pcm) > MAX_SESSION_S * SR:
        raise RuntimeError(f'{path}: {len(pcm)/SR/3600:.1f} h exceeds MAX_SESSION_S')
    return pcm


_HAVE_SF = None


def encode_flac(pcm):
    """int16 mono 16 kHz -> FLAC bytes.  soundfile when it is installed, ffmpeg
    otherwise, so the module has no hard dependency beyond the one it already
    needs for decoding."""
    global _HAVE_SF
    if _HAVE_SF is None:
        try:
            import soundfile            # noqa: F401
            _HAVE_SF = True
        except ImportError:
            _HAVE_SF = False
    if _HAVE_SF:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, pcm, SR, format='FLAC', subtype='PCM_16')
        return buf.getvalue()
    cmd = [ffmpeg_path(), '-nostdin', '-v', 'error', '-f', 's16le', '-ar', str(SR),
           '-ac', '1', '-i', 'pipe:0', '-f', 'flac', 'pipe:1']
    return subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, check=True).stdout


def decode_flac(blob):
    """FLAC bytes -> int16 array.  Used by the self-checks and verify_session."""
    global _HAVE_SF
    if _HAVE_SF is None:
        encode_flac(np.zeros(1, '<i2'))
    if _HAVE_SF:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(blob), dtype='int16')
        assert sr == SR, f'unexpected rate {sr}'
        return data
    cmd = [ffmpeg_path(), '-nostdin', '-v', 'error', '-i', 'pipe:0',
           '-f', 's16le', '-ac', '1', '-ar', str(SR), 'pipe:1']
    raw = subprocess.run(cmd, input=blob, capture_output=True, check=True).stdout
    return np.frombuffer(raw, '<i2')


# ---- cost -----------------------------------------------------------------
def session_bytes(sessions, workers=8):
    """True m4a sizes, without transferring them.  HF answers a HEAD with a 302
    and an `x-linked-size` header giving the real LFS size; following the
    redirect would start the download."""
    sessions = [int(s) for s in sessions]
    known = _cached('session_bytes.json', dict) if os.path.exists(
        os.path.join(CACHE, 'session_bytes.json')) else {}
    todo = [s for s in sessions if str(s) not in known]
    if todo:
        token = get_token()

        def head(s):
            url = f'https://huggingface.co/datasets/{AUDIO_REPO}/resolve/main/{s}/audio.m4a'
            r = requests.head(url, headers={'Authorization': f'Bearer {token}'},
                              allow_redirects=False, timeout=60)
            return s, int(r.headers.get('x-linked-size') or r.headers.get('Content-Length') or 0)

        with ThreadPoolExecutor(workers) as ex:
            for s, n in ex.map(head, todo):
                known[str(s)] = n
        os.makedirs(CACHE, exist_ok=True)
        json.dump(known, open(os.path.join(CACHE, 'session_bytes.json'), 'w'))
    return {s: known.get(str(s), 0) for s in sessions}


def estimate_cost(turns_df, sessions=None, use_head=False):
    """What a build over `turns_df` will move, produce, and need free."""
    sessions = sorted(set(sessions if sessions is not None else turns_df.session.unique()))
    span = (turns_df.groupby('turn').end.max() - turns_df.groupby('turn').start.min()).sum()
    if use_head:
        transfer = sum(session_bytes(sessions).values())
    else:
        # max(end) per session under-counts an m4a that carries more audio than
        # the alignment covers; use_head=True is exact.
        transfer = float(turns_df.groupby('session').end.max().sum()) * MB_PER_S * 1e6
    out_bytes = span * BYTES_PER_S * FLAC_RATIO
    free = shutil.disk_usage(HERE).free
    need = out_bytes + WORKERS * 512e6
    return {'sessions': len(sessions), 'transfer_gb': transfer / 1e9,
            'out_gb': out_bytes / 1e9, 'emitted_hours': span / 3600,
            'n_chunks': int(np.ceil(span / MAX_S)), 'free_disk_gb': free / 1e9,
            'ok': free > need}


# ---- build ----------------------------------------------------------------
def build_session(session, rows, keep_m4a=False):
    """Fetch, decode, cut, encode, delete.  Returns (DataFrame, stats dict).

    The m4a is removed as soon as it has been decoded, so peak transient disk is
    WORKERS x one file rather than the corpus.
    """
    t0 = time.time()
    st = {'session': int(session), 'n_turns': int(rows.turn.nunique()), 'n_chunks': 0,
          'n_dropped': 0, 'decoded_s': None, 'max_end_s': float(rows.end.max()),
          'bytes': None, 'seconds': None, 'error': None}
    path = None
    try:
        path = fetch_m4a(session)
        st['bytes'] = os.path.getsize(path)
        pcm = decode_session(path)
        st['decoded_s'] = len(pcm) / SR
        meta = {'session_date': rows.iloc[0].session_date, 'knesset': rows.iloc[0].knesset,
                'committee_name': rows.iloc[0].committee_name}
        out, dropped = [], 0
        for _, turn_df in rows.groupby('turn', sort=True):
            r, d = chunk_rows(turn_df, pcm, meta)
            out += r
            dropped += d
        st['n_chunks'], st['n_dropped'] = len(out), dropped
        df = coerce(pd.DataFrame(out))
    finally:
        if path and os.path.exists(path) and not keep_m4a:
            os.remove(path)
    st['seconds'] = round(time.time() - t0, 1)
    return df, st


def build(turns_df, sessions=None, workers=WORKERS, keep_m4a=False):
    """Build every session in `turns_df`, threaded and resumable.

    One part file per session, not stage3's batch of 400: a batch here costs tens
    of GB of transfer, so losing one to an interruption is unacceptable.  A
    session whose part file exists is skipped, so a partial run is a valid corpus
    of the sessions it finished.
    """
    os.makedirs(PARTS, exist_ok=True)
    by_session = {int(s): g for s, g in turns_df.groupby('session', sort=True)}
    todo = sorted(by_session) if sessions is None else [int(s) for s in sessions]
    stats, t0, fresh = [], time.time(), 0

    def one(sid):
        pp = os.path.join(PARTS, f'{sid}.parquet')
        sp = os.path.join(PARTS, f'{sid}_stats.json')
        if os.path.exists(pp) and os.path.exists(sp):
            return sid, json.load(open(sp, encoding='utf-8')), True
        df, st = build_session(sid, by_session[sid], keep_m4a=keep_m4a)
        df.to_parquet(pp, index=False)
        json.dump(st, open(sp, 'w', encoding='utf-8'))
        return sid, st, False

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, s): s for s in todo}
        for k, f in enumerate(as_completed(futs), 1):
            sid = futs[f]
            try:
                sid, st, resumed = f.result()
            except Exception as e:
                # A session that fails must never sink the run; it is recorded and
                # retried on the next pass, since no part file was written.
                stats.append({'session': sid, 'error': repr(e)[:200]})
                print(f'  [{sid}] FAILED {repr(e)[:120]}', flush=True)
                continue
            stats.append(st)
            fresh += 0 if resumed else 1
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t0
                # ETA over sessions actually fetched: resumed ones cost no time,
                # and dividing by k makes the first estimate after a resume absurd.
                eta = el / max(fresh, 1) * (len(todo) - k) / 60
                print(f'  {k}/{len(todo)} sessions, {sum(s.get("n_chunks") or 0 for s in stats):,} '
                      f'chunks, {el/60:.1f} min, eta {eta:.0f} min', flush=True)
    sdf = pd.DataFrame(stats)
    os.makedirs(OUT, exist_ok=True)
    sdf.to_csv(os.path.join(OUT, 'build_sessions.csv'), index=False)
    return sdf


def assemble(shard_mb=500, out_dir=None):
    """Concatenate the per-session parts into shards, grouped by knesset."""
    out_dir = out_dir or os.path.join(OUT, 'data')
    os.makedirs(out_dir, exist_ok=True)
    parts = sorted(f for f in os.listdir(PARTS) if f.endswith('.parquet'))
    buf, n, shard, written = [], 0, 0, []
    for f in parts:
        p = os.path.join(PARTS, f)
        buf.append(pd.read_parquet(p))
        n += os.path.getsize(p)
        if n >= shard_mb * 1e6:
            written.append(_flush(buf, out_dir, shard))
            buf, n, shard = [], 0, shard + 1
    if buf:
        written.append(_flush(buf, out_dir, shard))
    return written


def _flush(buf, out_dir, shard):
    path = os.path.join(out_dir, f'chunks-{shard:05d}.parquet')
    pd.concat(buf, ignore_index=True).to_parquet(path, index=False)
    print(f'  wrote {path}', flush=True)
    return path


def verify_session(session, turns_df, n=3, out_dir=None):
    """Cut n chunks and write them where they can be listened to.

    The only check that catches a time-origin offset between the m4a and the
    alignment: plausible-looking chunks of the wrong words satisfy every
    assertion in this module.  Run it before committing to a full build.
    """
    out_dir = out_dir or os.path.join(OUT, 'listen')
    os.makedirs(out_dir, exist_ok=True)
    rows = turns_df[turns_df.session == int(session)]
    df, st = build_session(int(session), rows, keep_m4a=False)
    for _, r in df.head(n).iterrows():
        path = os.path.join(out_dir, r.chunk_id)
        with open(path, 'wb') as fh:
            fh.write(r.audio['bytes'])
        print(f'{r.chunk_id}  {r.duration_s:5.2f}s  q={r.quality:.3f}  {r.speaker_name}')
        print(f'    {r.text[:150]}')
    return df, st


# ---- self-checks ----------------------------------------------------------
def _check_slicing():
    # A turn of six 4 s segments with 1 s gaps spans 29 s -> exactly one chunk,
    # ending at the last segment rather than at seek + 30.
    segs = [{'start': i * 5.0, 'end': i * 5.0 + 4.0, 'text': f't{i}', 'quality': 0.9}
            for i in range(6)]
    ch = slice_turn(segs)
    assert len(ch) == 1, ch
    assert (ch[0]['seek'], ch[0]['end']) == (0.0, 29.0), ch[0]

    # Ten of them span 49 s -> two chunks, back-to-back, neither over 30 s.
    segs = [{'start': i * 5.0, 'end': i * 5.0 + 4.0, 'text': f't{i}', 'quality': 0.9}
            for i in range(10)]
    ch = slice_turn(segs)
    assert len(ch) == 2, len(ch)
    assert ch[1]['seek'] == ch[0]['end'], 'windows must tile back-to-back'
    for c in ch:
        assert c['end'] - c['seek'] <= MAX_S + 1e-9
        assert c['end'] <= segs[-1]['end'], 'a chunk may never run past its turn'

    # Every segment is packed exactly once -- no loss, no duplication.
    packed = [p for c in ch for s in c['segments'] for p in s['parts']]
    assert len(packed) == len(segs), (len(packed), len(segs))

    # Gaps under MERGE_GAP fold into one timestamp span; wider ones do not.
    tight = [{'start': 0.0, 'end': 2.0, 'text': 'a', 'quality': 0.9},
             {'start': 2.1, 'end': 4.0, 'text': 'b', 'quality': 0.9}]
    assert len(slice_turn(tight)[0]['segments']) == 1
    wide = [{'start': 0.0, 'end': 2.0, 'text': 'a', 'quality': 0.9},
            {'start': 3.0, 'end': 4.0, 'text': 'b', 'quality': 0.9}]
    assert len(slice_turn(wide)[0]['segments']) == 2

    # A segment too long to ever fit is skipped, not emitted truncated.
    long = [{'start': 0.0, 'end': 40.0, 'text': 'x', 'quality': 0.9},
            {'start': 41.0, 'end': 45.0, 'text': 'y', 'quality': 0.9}]
    ch = slice_turn(long)
    assert len(ch) == 1 and ch[0]['segments'][0]['text'] == 'y'


def _check_tokens():
    assert _ts(0) == '<|0.00|>'
    assert _ts(4.163) == '<|4.16|>'
    assert _ts(30.0) == '<|30.00|>'
    try:
        _ts(30.5)
        raise AssertionError('should have rejected an out-of-range timestamp')
    except ValueError:
        pass


def _check_ids():
    # The 4-field convention round-trips: this is what makes chunk_id a key.
    cid = _chunk_id(4405, 2073683, 14.6, 16.94)
    assert cid == '4405_2073683_14600_16940.flac', cid
    spk, sess, a, b = cid[:-5].split('_')
    assert (int(spk), int(sess), int(a) / 1000, int(b) / 1000) == (4405, 2073683, 14.6, 16.94)
    # Index times carry 2 decimals, and 2.01 * 16000 is 32159.999999999996, so
    # int() loses a sample.  Measured: 1,179 of the first 200,000 two-decimal
    # timestamps (0.6%) truncate low this way, which is why every conversion in
    # this module rounds.
    assert int(2.01 * SR) == 32159 and int(round(2.01 * SR)) == 32160
    assert sum(1 for i in range(200_000) if int(i / 100 * SR) != round(i / 100 * SR)) == 1179


def _check_audio_roundtrip():
    rng = np.random.default_rng(0)
    pcm = (rng.normal(0, 3000, 2 * SR)).astype('<i2')
    blob = encode_flac(pcm)
    back = decode_flac(blob)
    assert len(back) == len(pcm), (len(back), len(pcm))
    assert np.array_equal(back, pcm), 'FLAC must be lossless'
    assert len(pcm) == round(2.0 * SR)


def _check_coerce():
    empty = coerce(pd.DataFrame())
    assert list(empty.columns)[:-1] == list(SCHEMA)
    # The year_of_aliya trap: an all-null part must still carry a string column,
    # or it infers double and refuses to concatenate with a part that has values.
    a = coerce(pd.DataFrame([{'chunk_id': 'x', 'speaker_id': 1, 'session': 1, 'year_of_aliya': None}]))
    b = coerce(pd.DataFrame([{'chunk_id': 'y', 'speaker_id': 2, 'session': 2, 'year_of_aliya': '1950'}]))
    assert str(a.year_of_aliya.dtype) == 'string' and str(b.year_of_aliya.dtype) == 'string'
    assert len(pd.concat([a, b], ignore_index=True)) == 2


def _check_index(fetch=False):
    """Assert the corpus numbers this module was designed against.

    Skipped unless the index is already cached, because it is 520 MB and a
    self-check should not start a download nobody asked for.  Run
    `python preprocessing.py --fetch` to pull it and check for real."""
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(INDEX_REPO, 'segments.parquet', repo_type='dataset')
    if not isinstance(hit, str):
        if not fetch:
            print('  index not cached; skipping corpus-level checks '
                  '(pass --fetch to download it, 520 MB)')
            return None
        print('  downloading the index (520 MB) ...', flush=True)
        index_path()
    df = add_turns(load_index())
    assert len(df) == 5_158_763, len(df)
    a = assigned(df)
    assert len(a) == 2_295_972, len(a)
    assert a.speaker_id.nunique() == 339, a.speaker_id.nunique()
    assert a.session.nunique() == 10_889, a.session.nunique()
    assert a.turn.nunique() == 935_366, a.turn.nunique()
    assert abs(a.duration_s.sum() / 3600 - 3397) < 2
    assert a.duration_s.min() >= MIN_S - 1e-9, a.duration_s.min()
    # Exactly one assigned segment in 2,295,972 breaks the 30 s ceiling: speaker
    # 30118 in session 2193222, 117.96 s, quality 0.0 -- an alignment that scored
    # nothing.  slice_turn skips any segment too long to fit a window, so it is
    # handled rather than truncated, but it is a live path and not a guard.
    over = a[a.duration_s > MAX_S]
    assert len(over) == 1 and over.iloc[0].quality == 0.0, over[['session', 'duration_s', 'quality']]
    assert a[a.quality >= 0.7].duration_s.max() <= MAX_S + 1e-9

    # The guarantee the whole module exists for: no run holds two speakers.
    g = a.groupby('turn', sort=False).speaker_id
    assert int((g.max() != g.min()).sum()) == 0, 'a run holds more than one speaker'
    return a


if __name__ == '__main__':
    _check_slicing()
    _check_tokens()
    _check_ids()
    _check_coerce()
    try:
        _check_audio_roundtrip()
        audio_ok = 'FLAC round-trip exact'
    except RuntimeError as e:
        audio_ok = f'audio checks skipped ({str(e).splitlines()[0]})'
    a = _check_index(fetch='--fetch' in sys.argv)
    corpus = (f'{len(a):,} assigned segments, {a.turn.nunique():,} turns, '
              f'{a.duration_s.sum()/3600:,.0f} h, {a.speaker_id.nunique()} speakers'
              if a is not None else 'corpus checks skipped (index not cached)')
    print(f'preprocessing self-checks OK: slicing, tokens, ids, schema; {audio_ok}; {corpus}')
