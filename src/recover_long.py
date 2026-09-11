"""Recover the sessions the whole-session decoder had to skip.

`decode_session` holds a whole session as PCM, so `MAX_SESSION_S` caps it at
8 h -- 12 h of audio is 1.35 GB in one worker, doubled transiently by ffmpeg's
captured stdout, against ~4 GB free.  Seven sessions of 10,905 exceed it.

The fix is not a bigger ceiling but a different decoder: ask ffmpeg for one
turn's span at a time (`-ss`/`-t` against the local file), so peak memory is set
by the longest TURN (<= 30 s of audio in a chunk, a few minutes in a turn)
rather than by the recording.  Same technique as stage3/validate_audio.py, which
is the right tool for a handful of outliers and the wrong one for 10,905
sessions -- it costs one ffmpeg process per turn instead of one per session.

    python recover_long.py           # builds and ships them as one more shard
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import preprocessing as P

# The seven, from the build log.  Passed explicitly rather than rediscovered:
# a failure list is evidence, and it should not change under us on a re-run.
LONG_SESSIONS = [2141001, 2162642, 2166970, 2169555, 2194576, 2208911, 2230295]


def decode_span(path, start, end, pad=0.0):
    """One span of a local media file -> int16 mono 16 kHz.

    `-ss` BEFORE `-i` seeks without decoding everything up to the mark, which is
    what keeps this cheap on an 11 h file.  On a local file that seek is exact
    enough for our purposes; the returned length is what the caller trusts, not
    the requested duration.
    """
    dur = max(end - start + pad, 0.0)
    cmd = [P.ffmpeg_path(), '-nostdin', '-v', 'error',
           '-ss', f'{max(start - pad, 0):.3f}', '-i', path, '-t', f'{dur:.3f}',
           '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(P.SR),
           '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, '<i2')


def build_long_session(session, rows, keep_m4a=False):
    """Same output as build_session, decoding per turn instead of per session."""
    t0 = time.time()
    st = {'session': int(session), 'n_turns': int(rows.turn.nunique()), 'n_chunks': 0,
          'n_dropped': 0, 'decoded_s': None, 'max_end_s': float(rows.end.max()),
          'bytes': None, 'seconds': None, 'error': None}
    path = None
    try:
        path = P.fetch_m4a(session)
        st['bytes'] = os.path.getsize(path)
        meta = {'session_date': rows.iloc[0].session_date, 'knesset': rows.iloc[0].knesset,
                'committee_name': rows.iloc[0].committee_name}
        out, dropped, decoded = [], 0, 0.0
        for _, turn_df in rows.groupby('turn', sort=True):
            lo = float(turn_df.start.min())
            hi = float(turn_df.end.max())
            pcm = decode_span(path, lo, hi)
            decoded += len(pcm) / P.SR
            # chunk_rows indexes a session-absolute array, so shift this turn's
            # times to be relative to the span we actually decoded.
            shifted = turn_df.assign(start=turn_df.start - lo, end=turn_df.end - lo)
            r, d = P.chunk_rows(shifted, pcm, meta)
            for row in r:                       # restore absolute provenance
                row['abs_start'] += lo
                row['abs_end'] += lo
                row['chunk_id'] = P._chunk_id(row['speaker_id'], row['session'],
                                              row['abs_start'], row['abs_end'])
                row['audio']['path'] = row['chunk_id']
            out += r
            dropped += d
        st['decoded_s'], st['n_chunks'], st['n_dropped'] = decoded, len(out), dropped
        df = P.coerce(pd.DataFrame(out))
    finally:
        if path and os.path.exists(path) and not keep_m4a:
            os.remove(path)
    st['seconds'] = round(time.time() - t0, 1)
    return df, st


if __name__ == '__main__':
    a = P.assigned(P.add_turns(P.load_index()))
    done = P.uploaded_sessions()
    todo = [s for s in LONG_SESSIONS if s not in done]
    print(f'{len(todo)} of {len(LONG_SESSIONS)} long sessions still to recover', flush=True)
    if not todo:
        raise SystemExit('nothing to do')

    rows = P.attach_text(a[a.session.isin(todo)])
    os.makedirs(P.PARTS, exist_ok=True)
    stats, built = [], []
    for sid in todo:
        try:
            df, st = build_long_session(sid, rows[rows.session == sid])
            df.to_parquet(os.path.join(P.PARTS, f'{sid}.parquet'), index=False)
            json.dump(st, open(os.path.join(P.PARTS, f'{sid}_stats.json'), 'w', encoding='utf-8'))
            built.append(sid)
            print(f'  [{sid}] {st["n_chunks"]:,} chunks from {st["decoded_s"]/3600:.1f} h '
                  f'decoded, {st["seconds"]:.0f}s', flush=True)
        except Exception as e:
            stats.append({'session': sid, 'error': repr(e)[:200]})
            print(f'  [{sid}] FAILED {repr(e)[:140]}', flush=True)
    if built:
        idx = P.next_shard_index(P.CHUNKS_REPO)
        P.flush_shard(P.CHUNKS_REPO, idx, built)
        print(f'recovered {len(built)} sessions as shard {idx:05d}', flush=True)
