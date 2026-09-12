"""Full build driver: stream the whole committees corpus to the Hub.

    python src/preprocessing/chunk_corpus/run_build.py    # start, or resume where it stopped

Resumable at session granularity via the uploaded ledger, so an interrupt costs
at most the sessions of the shard that had not yet been pushed.  Safe to kill
and restart; safe to run for days.

Why blocks.  This machine has ~4 GB of RAM free, and `reference_text` for all
2,297,627 assigned rows plus a per-session groupby of them would sit at the edge
of that before a single session is decoded.  So the light index stays resident
(~200 MB) and the text is attached one block of sessions at a time.  The cost is
re-reading the index's text column once per block -- local disk, ~1 min -- which
over a 40 h run is noise.

WORKERS is 3 rather than the module default of 4 for the same reason: each
worker holds a decoded session (up to 333 MB) and ffmpeg's captured stdout
briefly doubles that, so 4 workers can spike past 2.7 GB.
"""
import gc
import time

import chunks as P

BLOCK = 600          # sessions per text-attach pass; 19 blocks over the corpus
WORKERS = 3
SHARD_MB = 500

if __name__ == '__main__':
    t0 = time.time()
    print(f'index revision {P.index_revision()[:8]} -> {P.CHUNKS_REPO}', flush=True)

    a = P.assigned(P.add_turns(P.load_index()))
    gc.collect()
    sessions = sorted(a.session.unique())
    done = P.uploaded_sessions()
    todo = [s for s in sessions if s not in done]
    print(f'{len(a):,} assigned segments, {a.turn.nunique():,} runs, '
          f'{a.speaker_id.nunique()} speakers', flush=True)
    print(f'{len(todo):,} sessions to build, {len(done):,} already uploaded', flush=True)

    for i in range(0, len(todo), BLOCK):
        block = todo[i:i + BLOCK]
        n = i + len(block)
        print(f'\n=== block {i // BLOCK + 1}/{-(-len(todo) // BLOCK)}: '
              f'sessions {i + 1}-{n} of {len(todo)} ===', flush=True)
        rows = P.attach_text(a[a.session.isin(block)])
        P.stream_build(rows, repo=P.CHUNKS_REPO, sessions=block,
                       workers=WORKERS, shard_mb=SHARD_MB)
        del rows
        gc.collect()
        el = (time.time() - t0) / 3600
        print(f'--- {n}/{len(todo)} sessions, {el:.1f} h elapsed, '
              f'eta {el / max(n, 1) * (len(todo) - n):.1f} h ---', flush=True)

    print(f'\nDONE in {(time.time() - t0) / 3600:.1f} h -> {P.CHUNKS_REPO}', flush=True)
