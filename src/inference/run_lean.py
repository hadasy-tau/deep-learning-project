"""Stage 4 lean runner: both arms from one memory-bounded process.

Built for a 17 GB laptop with a few GB to spare.  What the earlier design cost,
measured (2026-09-12): reading a shard whole inflates the 600 MB audio column
to 4.8-5.1 GB peak, three concurrent extractions were ~15 GB and the OS killed
everything; and every runner held the whole 1.2 M-row index (544 MB) when it
needed the 66 k-row subset (32 MB).  Reading the same shard over HfFileSystem
with iter_batches still peaked at 4.0 GB (fsspec buffers the body); from a
local file it is 1.35 GB, the floor for a one-row-group shard.

Here:
  * one process; only the subset's index rows are resident
  * each shard is fetched to DISK (the 53 shards the extractor already cached
    are used as-is; the rest via hf_hub_download, two prefetched ahead), read
    with iter_batches, then deleted -- one download per shard for the whole job
  * a shard's rows go to BOTH arms (B round-robin over the RunPod endpoints);
    in-flight audio is bounded per pool, so RAM ~ 1.4 GB read + tens of MB
  * same JSONLs, resume and provider layer as run.py; finish.sh applies after

    python src/inference/run_lean.py --max-shards 1            # smoke
    python src/inference/run_lean.py                           # everything left
"""
import argparse, gc, glob, json, os, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D, providers as P
from run import Runner, OUT, log, _wait_some
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
# Arrow's default pool (jemalloc/mimalloc) keeps freed shard memory around; the
# runner's floor crept from 0.9 to 2.7 GB in an hour and spiked to 4.7 GB per
# read.  The system pool plus an explicit release after each shard hands it back.
pa.set_memory_pool(pa.system_memory_pool())
DL = os.path.join(D.CACHE, 'lean_dl')                      # transient shard downloads, deleted after use

def slim_index(subset_ids):
    """The subset's rows of cache/index.parquet, read in 100 k-row batches so the
    full 544 MB table is never resident."""
    want = pa.array(sorted(subset_ids)); parts = []
    for b in pq.ParquetFile(os.path.join(D.CACHE, 'index.parquet')).iter_batches(batch_size=100_000):
        m = pc.is_in(b.column('chunk_id'), value_set=want)
        if pc.any(m).as_py(): parts.append(b.filter(m))
    return pa.Table.from_batches(parts).to_pandas()

def remaining(sub_idx):
    """subset rows still needed by A and by B, from every JSONL on disk."""
    A, B = set(), set()
    for p in glob.glob(os.path.join(OUT, '*.jsonl')):
        if os.path.basename(p).startswith(('verify', 'e2e')): continue
        for line in open(p, encoding='utf-8'):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if r.get('error') is not None: continue
            if r['arm'] == 'B': B.add(r['chunk_id'])
            elif r.get('language') == 'he': A.add(r['chunk_id'])   # auto-detect rows do not count
    return sub_idx[~sub_idx.chunk_id.isin(A)], sub_idx[~sub_idx.chunk_id.isin(B)]

def cached_path(shard):
    hits = glob.glob(os.path.join(D.CACHE, 'audio', '*', shard))
    return hits[0] if hits else None

def fetch(shard):
    """Local path of the shard: the extractor's cache if it has it, else a fresh
    download under DL (returned path is deleted by the caller)."""
    c = cached_path(shard)
    if c: return c, False
    for attempt in range(6):
        try:
            return hf_hub_download(D.REPO, f'data/{shard}', repo_type='dataset', revision=D.revision(), local_dir=DL), True
        except Exception as e:
            log(f'  {shard}: download attempt {attempt+1} failed ({e!r}'[:140] + ')'); time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'{shard}: download failed 6 times')

def shard_rows(path, want, batch=48):
    """{chunk_id: flac bytes} for `want`.

    DuckDB first: it streams the single 770 MB row group under a memory cap
    (measured 1.75 GB peak, 1 s).  Arrow's iter_batches on the same file peaks
    at ~4x the file size whatever the pre-buffer/mmap settings (3 GB, 25 s),
    so it is only the fallback."""
    try:
        import duckdb
        con = duckdb.connect()
        con.execute("SET memory_limit='2500MB'; SET threads=1; SET preserve_insertion_order=false")
        con.register('want', pd.DataFrame({'chunk_id': sorted(want)}))
        cur = con.execute("SELECT p.chunk_id, p.audio['bytes'] FROM read_parquet(?) p "
                          "WHERE p.chunk_id IN (SELECT chunk_id FROM want)", [path])
        out = {}
        while True:
            rows = cur.fetchmany(64)
            if not rows: break
            for cid, b in rows: out[cid] = bytes(b)
        con.close(); return out
    except Exception as e:
        log(f'  duckdb read failed ({e!r}'[:120] + '); falling back to arrow')
    pf = pq.ParquetFile(path); out = {}
    for b in pf.iter_batches(batch_size=batch, columns=['chunk_id', 'audio']):
        ids = b.column('chunk_id').to_pylist()
        hit = [i for i, c in enumerate(ids) if c in want]
        if hit:
            aud = b.column('audio').to_pylist()
            # the dataset stores audio as {bytes, path}; the extractor's cache as bare bytes
            for i in hit: out[ids[i]] = aud[i]['bytes'] if isinstance(aud[i], dict) else aud[i]
        if len(out) == len(want): break
    return out

class Lean:
    def __init__(self, workers_a, workers_b, endpoints):
        self.A = Runner(P.make('A'), OUT / 'full_A.jsonl', workers=workers_a)
        self.Bs = [Runner(P.make('B', endpoint_id=e), OUT / f'full_B{"" if i == 0 else i + 1}.jsonl', workers=workers_b)
                   for i, e in enumerate(endpoints)]
        self.all = [self.A] + self.Bs
        for r in self.all:
            r.out.parent.mkdir(parents=True, exist_ok=True); r.fh = r.out.open('a', encoding='utf-8')
        self.pools = {r: ThreadPoolExecutor(r.workers) for r in self.all}
        self.futs = {r: set() for r in self.all}

    def submit(self, r, row, audio):
        """Submit to r's pool, first blocking until its backlog is under 3x workers."""
        if len(self.futs[r]) >= r.workers * 3:
            _, self.futs[r] = _wait_some(self.futs[r], r.workers)
        self.futs[r].add(self.pools[r].submit(r.one, row, audio))

    def run(self, need_a, need_b, max_shards=None, only=None, prefetch=2):
        shards = sorted(set(need_a.shard) | set(need_b.shard))
        if only: shards = [s for s in shards if s in set(only)]
        shards = shards[:max_shards]
        na, nb = len(need_a), len(need_b); t0 = time.time(); ga = gb = 0
        n_cached = sum(cached_path(s) is not None for s in shards)
        log(f'lean: {len(shards)} shards ({n_cached} already on disk); A needs {na:,} rows, B needs {nb:,}')
        os.makedirs(DL, exist_ok=True)
        pre = ThreadPoolExecutor(prefetch); ahead = {s: pre.submit(fetch, s) for s in shards[:prefetch]}
        for k, shard in enumerate(shards, 1):
            nxt = k - 1 + prefetch                                    # keep `prefetch` downloads in flight (disk only)
            if nxt < len(shards) and shards[nxt] not in ahead: ahead[shards[nxt]] = pre.submit(fetch, shards[nxt])
            if any(r.stop.is_set() for r in self.all): log('a runner asked to stop (billing/quota); ending'); break
            ra = need_a[need_a.shard == shard]; rb = need_b[need_b.shard == shard]
            want = set(ra.chunk_id) | set(rb.chunk_id)
            try:
                path, temp = ahead.pop(shard).result()
                blob = shard_rows(path, want)
            except Exception as e:
                log(f'  {shard}: skipped ({e!r}'[:160] + ')'); continue
            if temp: os.remove(path)                     # only this shard; the next ones are still downloading here
            missing = want - blob.keys()
            if missing: log(f'  {shard}: {len(missing)} wanted chunks not in shard -- left for coverage')
            for r in ra.itertuples(index=False):
                if r.chunk_id in blob: self.submit(self.A, r, blob[r.chunk_id])
            for j, r in enumerate(rb.itertuples(index=False)):
                if r.chunk_id in blob: self.submit(self.Bs[j % len(self.Bs)], r, blob[r.chunk_id])
            del blob; gc.collect(); pa.default_memory_pool().release_unused()
            ga += len(ra); gb += len(rb)
            if k % 5 == 0 or k == len(shards):
                el = time.time() - t0; b_exec = sum(x.stats.get('exec_s', 0.0) for x in self.Bs)
                log(f'  {k}/{len(shards)} shards | sent A {ga:,}/{na:,}  B {gb:,}/{nb:,} | written '
                    f'{" ".join(f"{x.prov.arm}{i}:{x.stats["ok"]}/{x.stats["failed"]}err" for i, x in enumerate(self.all))} | '
                    f'{el/60:.0f} min, eta {el/k*(len(shards)-k)/60:.0f} min | '
                    f'A ${self.A.stats["audio_s"]/60*0.00045:.2f}  B gpu ${b_exec/3600*0.25:.2f}')
        pre.shutdown(); shutil.rmtree(DL, ignore_errors=True)
        for r in self.all: self.pools[r].shutdown(wait=True); r.fh.close()
        log('written: ' + ', '.join(f'{r.out.name} ok {r.stats["ok"]:,} failed {r.stats["failed"]:,}' for r in self.all))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers-a', type=int, default=32); ap.add_argument('--workers-b', type=int, default=16)
    ap.add_argument('--endpoints', nargs='+', default=['fifhkwzcjy7zey', 'q5fuot57ltygcs'])
    ap.add_argument('--max-shards', type=int, default=None); ap.add_argument('--shards', nargs='+', default=None)
    ap.add_argument('--prefetch', type=int, default=2, help='shard downloads kept in flight (disk only, ~770 MB each)')
    a = ap.parse_args()
    sub = pd.read_parquet(os.path.join(HERE, 'subset_stage1.parquet'))
    sub_idx = slim_index(set(sub.chunk_id))
    assert len(sub_idx) == len(sub), (len(sub_idx), len(sub))
    need_a, need_b = remaining(sub_idx)
    log(f'resident index: {len(sub_idx):,} rows, {sub_idx.memory_usage(deep=True).sum()/1e6:.0f} MB')
    L = Lean(a.workers_a, a.workers_b, a.endpoints)
    L.run(need_a, need_b, a.max_shards, a.shards, a.prefetch)
    log('LEAN DONE')
