"""Stage 4 data: iterate Hadas's chunk corpus without holding it.

Hadas/knesset-committees-chunks is 410 parquet shards, ~275 GB, one row group
each, FLAC bytes inline.  Two facts shape the reader:
  * The metadata columns of all 410 shards fit in memory (1.2 M rows, ~150 MB)
    and were already pulled once into stage3/cache/hadas_chunks_meta.parquet.
    Selection (speakers, sessions, quality, limit, resume) happens on that
    table; audio is fetched only for the rows that survive.
  * A shard's audio column is read whole (one row group), so selecting rows
    across shards means reading each touched shard once, not once per row.
Pinned to one dataset revision so a run cannot straddle a re-upload.
"""
import io, os, json
import pandas as pd
import requests
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, get_token

class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests.

    pyarrow reads a parquet file's footer, then exactly the byte ranges of the
    columns it was asked for.  Through HfFileSystem that took 23 s per shard and
    stalled outright under 8 threads (an fsspec block-cache sits between pyarrow
    and the socket).  Plain Range requests: 5 HTTP calls and 16 s for the same
    shard, nothing to hang.  Only the metadata columns (0.2% of a shard) are ever
    fetched this way; audio is read whole per shard by audio_iter()."""
    def __init__(self, url, token, timeout=120, tries=4):
        super().__init__()
        self.h = {'Authorization': f'Bearer {token}'}; self.timeout, self.tries = timeout, tries
        self.pos = 0; self.calls = 0
        r = self._req('head', url); self.size = int(r.headers['Content-Length']); self.url = r.url
    def _req(self, method, url, **kw):
        import time as _t
        for i in range(self.tries):
            try:
                r = getattr(requests, method)(url, headers={**self.h, **kw.pop('headers', {})}, allow_redirects=True,
                                              timeout=self.timeout, stream=(method == 'get'), **kw)
                if r.status_code in (429, 500, 502, 503, 504): raise requests.HTTPError(f'HTTP {r.status_code}')
                if method == 'get' and 'Range' in (kw.get('headers') or {}) and r.status_code == 200:
                    r.close(); raise requests.HTTPError('Range ignored (HTTP 200)')
                r.raise_for_status()
                if method == 'get': r.content            # materialise the (small) ranged body
                return r
            except requests.RequestException as e:
                err = e; _t.sleep(min(2 ** i, 15))
        raise err
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n == -1: n = self.size - self.pos
        if n <= 0: return b''
        r = self._req('get', self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'})
        if r.status_code != 206:
            # The CDN occasionally answers a Range request with the whole file
            # (HTTP 200).  Handing pyarrow 597 MB when it asked for 32 KB fails
            # as "File too short"; treat it as a transient and retry the range.
            raise requests.HTTPError(f'Range ignored: HTTP {r.status_code} for {len(r.content):,} B')
        b = r.content; self.pos += len(b); self.calls += 1; return b
    def readinto(self, buf):
        b = self.read(len(buf)); buf[:len(b)] = b; return len(b)

def shard_url(name):
    return f'https://huggingface.co/datasets/{REPO}/resolve/{revision()}/data/{name}'

HERE   = os.path.dirname(os.path.abspath(__file__))
CACHE  = os.path.join(HERE, 'cache')
REPO   = 'Hadasy/knesset-committees-chunks'
META   = os.path.join(os.path.dirname(HERE), 'stage3', 'cache', 'hadas_chunks_meta.parquet')
META_COLS = ['chunk_id', 'speaker_id', 'speaker_name', 'session', 'session_date', 'knesset',
             'committee_name', 'duration_s', 'quality', 'n_words', 'text', 'gender', 'age']

def _cached(name, fn):
    os.makedirs(CACHE, exist_ok=True); p = os.path.join(CACHE, name)
    if os.path.exists(p):
        return json.load(open(p))
    v = fn(); json.dump(v, open(p, 'w')); return v

def revision():
    """Dataset commit sha, pinned on first use (delete cache/revision.json to move)."""
    return _cached('revision.json', lambda: HfApi(token=get_token()).dataset_info(REPO).sha)

def shard_files():
    fs = HfFileSystem(token=get_token())
    return sorted(fs.glob(f'datasets/{REPO}@{revision()}/data/*.parquet'))

def _shard_of(chunk_id_to_shard, chunk_ids):
    return [chunk_id_to_shard[c] for c in chunk_ids]

def _read_meta(name):
    """Metadata columns of one shard via Range requests (retries inside RangeFile)."""
    f = RangeFile(shard_url(name), get_token())
    t = pq.ParquetFile(f).read(columns=META_COLS).to_pandas()
    t['shard'] = name; return t

def index(refresh=False, workers=8):
    """Metadata for every chunk (all META_COLS, incl. the reference text) plus
    which shard holds it.  Built once, then read from cache.

    Checkpointed per shard under cache/index_parts/: a 30-minute pull over 410
    remote files died at shard 364 on a DNS blip and, because the result was
    only written at the end, lost all of it.  Now each shard's metadata lands
    on disk as it arrives and a restart re-reads only what is missing."""
    p = os.path.join(CACHE, 'index.parquet')
    if os.path.exists(p) and not refresh:
        df = pd.read_parquet(p)
        missing = [c for c in META_COLS + ['shard'] if c not in df.columns]
        if not missing:
            return df
        print(f'index cache lacks {missing}; rebuilding', flush=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parts_dir = os.path.join(CACHE, 'index_parts'); os.makedirs(parts_dir, exist_ok=True)
    names = [os.path.basename(f) for f in shard_files()]
    have = {n for n in names if os.path.exists(os.path.join(parts_dir, n))}
    todo = [n for n in names if n not in have]
    print(f'index: {len(have)} shards cached, {len(todo)} to fetch', flush=True)
    def fetch(n):
        t = _read_meta(n); t.to_parquet(os.path.join(parts_dir, n), index=False); return n
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, n): n for n in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:                 # one bad shard must not sink the run
                failed.append((futs[f], repr(e)[:120]))
            if i % 25 == 0: print(f'  index: {len(have) + i}/{len(names)} shards', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} shard(s) failed; re-run to retry them: {failed[:3]}')
    df = pd.concat([pd.read_parquet(os.path.join(parts_dir, n)) for n in names], ignore_index=True)
    assert df.chunk_id.is_unique, 'duplicate chunk_id across shards'
    assert len(df) > 1_000_000 and df.shard.nunique() == len(names), (len(df), df.shard.nunique())
    df.to_parquet(p, index=False)
    return df

def select(df, speakers=None, sessions=None, min_quality=None, min_s=None, max_s=None, limit=None, seed=0, chunk_ids=None):
    """Filters over the index.  `limit` samples uniformly (seeded) after filtering."""
    m = pd.Series(True, index=df.index)
    if chunk_ids is not None: m &= df.chunk_id.isin(set(chunk_ids))
    if speakers is not None:  m &= df.speaker_id.isin(set(speakers))
    if sessions is not None:  m &= df.session.isin(set(sessions))
    if min_quality is not None: m &= df.quality >= min_quality
    if min_s is not None: m &= df.duration_s >= min_s
    if max_s is not None: m &= df.duration_s <= max_s
    out = df[m]
    if limit is not None and len(out) > limit:
        out = out.sample(limit, random_state=seed)
    return out.sort_values(['shard', 'chunk_id'])

def _read_shard_audio(path, want, fs):
    with fs.open(path) as fh:
        t = pq.ParquetFile(fh).read(columns=['chunk_id', 'audio'])
    ids = t.column('chunk_id').to_pylist(); aud = t.column('audio').to_pylist()
    return {i: a['bytes'] for i, a in zip(ids, aud) if i in want}

def audio_iter(rows, fs=None, prefetch=2):
    """Yield (row, flac_bytes) for the selected rows, one shard at a time, in
    shard order.  rows must carry `shard` and `chunk_id`.

    The next `prefetch` shards are fetched on a background thread while the
    current one is being consumed.  Without this, a 128-thread request pool sat
    idle for up to 20 s at every shard boundary waiting on a 600 MB download --
    measured as 18 rows/s against a 22 rows/s ceiling on the live Arm A run."""
    import queue, threading
    fs = fs or HfFileSystem(token=get_token())
    groups = [(shard, g) for shard, g in rows.groupby('shard', sort=True)]
    q = queue.Queue(maxsize=prefetch)
    def producer():
        for shard, g in groups:
            try:
                q.put((shard, g, _read_shard_audio(f'datasets/{REPO}@{revision()}/data/{shard}', set(g.chunk_id), fs), None))
            except Exception as e:
                q.put((shard, g, None, e))
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is None: return
        shard, g, blob, err = item
        if err is not None:
            raise err
        for r in g.itertuples(index=False):
            b = blob.get(r.chunk_id)
            if b is None:
                raise KeyError(f'{r.chunk_id} not found in {shard}')
            yield r, b

def flac_duration(b):
    import soundfile as sf
    i = sf.info(io.BytesIO(b)); return i.frames / i.samplerate

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    df = index()
    print(f'index: {len(df):,} chunks, {df.speaker_id.nunique()} speakers, {df.shard.nunique()} shards, revision {revision()[:8]}')
    assert len(df) == 1_204_617 and df.shard.nunique() == 410, (len(df), df.shard.nunique())
    s = select(df, min_quality=0.7, min_s=5, max_s=25, limit=3, seed=1)
    assert len(s) == 3 and (s.quality >= 0.7).all()
    n = 0
    for r, b in audio_iter(s):
        d = flac_duration(b); assert abs(d - r.duration_s) < 0.05, (r.chunk_id, d, r.duration_s); n += 1
    assert n == 3
    print('data.py self-checks OK: 3 chunks fetched, durations match')
