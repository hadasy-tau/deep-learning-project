"""Stage 4 runner: transcribe the committee chunks with one arm, resumably.

    python src/inference/run.py --arm A --limit 10                 # smoke
    python src/inference/run.py --arm B --speakers 526 4405        # a few speakers
    python src/inference/run.py --arm A                            # everything (~3,840 h)

Design, inherited from the VoxKnesset arm-A driver that preceded it
(stage1/inference_openai_whisper_large_v3/full_run.py, removed with the rest of
the VoxKnesset path):
  * one JSONL per (arm, run) under outputs/, appended and flushed per row --
    crash-safe; a torn last line is skipped on resume
  * resume by chunk_id: rows already written without error are not re-sent;
    --retry-failed re-sends the ones that errored
  * a thread pool over network calls (the work is I/O bound); a global cooldown
    on 429 so 8 workers do not hammer independently; billing errors stop the run
  * the reference text rides along in every row, so scoring needs no join
  * every N minutes the JSONL files are mirrored to a private HF dataset repo
    (--upload-repo); off by default for smoke runs

The record written per chunk:
  chunk_id, arm, provider, model, speaker_id, session, session_date, knesset,
  duration_s, quality, reference, hypothesis, latency_s, error, ts
plus provider-specific extras under `raw` (RunPod exec/delay ms, job id).
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

def _wait_some(futs, k):
    """Block until at least k of `futs` finish; return (count finished, remaining)."""
    finished = 0
    while finished < k and futs:
        done, futs = wait(futs, return_when=FIRST_COMPLETED); finished += len(done)
    return finished, set(futs)
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import providers as P

OUT = Path(os.path.dirname(os.path.abspath(__file__))) / 'outputs'
RETRY_DELAYS = [2, 8, 30]

def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

def load_done(path, include_failed):
    done = set()
    if not path.exists(): return done
    for line in path.open(encoding='utf-8'):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if include_failed or r.get('error') is None:
            done.add(r['chunk_id'])
    return done

class Runner:
    def __init__(self, prov, out_path, workers=8, max_attempts=4, tag=None):
        self.prov, self.out, self.workers, self.max_attempts, self.tag = prov, out_path, workers, max_attempts, tag
        self.lock = threading.Lock(); self.gate = threading.Event(); self.gate.set()
        self.stop = threading.Event(); self.stats = dict(ok=0, failed=0, audio_s=0.0, t0=time.time())
        self.fh = None

    def one(self, row, audio):
        text = err = raw = None; t0 = time.perf_counter(); attempt = 0
        while attempt < self.max_attempts and not self.stop.is_set():
            self.gate.wait(timeout=120)
            try:
                res = self.prov.transcribe(audio); text, raw, err = res.text, res.raw, None; break
            except P.TransientError as e:
                err = f'{type(e).__name__}: {e}'
                if '429' in str(e):
                    with self.lock:
                        if self.gate.is_set():
                            self.gate.clear(); log('429: global 90 s cooldown'); threading.Timer(90, self.gate.set).start()
                    continue
                attempt += 1; time.sleep(RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)])
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
                if any(k in err.lower() for k in ('402', 'credit', 'payment', 'quota', 'insufficient')):
                    log(f'BILLING/QUOTA ERROR -- stopping: {err[:200]}'); self.stop.set()
                break
        rec = dict(chunk_id=row.chunk_id, arm=self.prov.arm, provider=self.prov.name, model=self.prov.model,
                   speaker_id=int(row.speaker_id), session=int(row.session), session_date=row.session_date,
                   knesset=int(row.knesset), duration_s=float(row.duration_s), quality=float(row.quality),
                   reference=row.text, hypothesis=text, latency_s=round(time.perf_counter() - t0, 3),
                   # provider-side timings as first-class columns: latency_s is
                   # our wall clock incl. queue and polling; these are what the
                   # provider itself reports, and the only fair cross-arm speed
                   exec_s=(raw or {}).get('exec_ms', None) and round(raw['exec_ms'] / 1000, 3),
                   queue_s=(raw or {}).get('delay_ms', None) and round(raw['delay_ms'] / 1000, 3),
                   language=(raw or {}).get('language'),      # None = the provider's auto-detect
                   error=err, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), raw=raw)
        with self.lock:
            self.fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); self.fh.flush()
            self.stats['ok' if err is None else 'failed'] += 1
            if err is None:
                self.stats['audio_s'] += float(row.duration_s)
                self.stats['exec_s'] = self.stats.get('exec_s', 0.0) + ((raw or {}).get('exec_ms') or 0) / 1000
        return rec

    def run(self, rows, log_every=500):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.out.open('a', encoding='utf-8')
        n = len(rows); last_logged = 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = set()
            for row, audio in D.audio_iter(rows, tag=self.tag):
                if self.stop.is_set(): break
                futs.add(ex.submit(self.one, row, audio))
                if len(futs) >= self.workers * 4:          # bound in-flight audio in RAM
                    done_now, futs = _wait_some(futs, self.workers)
                # progress is counted from rows actually WRITTEN (stats), not from
                # futures collected: the two drift by a full in-flight window, and a
                # check on the collected count re-fired on every loop iteration
                written = self.stats['ok'] + self.stats['failed']
                if written - last_logged >= log_every:
                    last_logged = written - (written % log_every); self.progress(written, n)
            for f in as_completed(futs): pass
        self.fh.close(); self.progress(self.stats['ok'] + self.stats['failed'], n, final=True)

    # Live cost, from what each provider actually bills (docs/inference.md):
    #   A  deepinfra   $0.00045 per audio-minute, no per-call fee
    #   B  RunPod      GPU worker-seconds; exec_s is the provider's own figure per job,
    #                  priced at the AMPERE_16 pool's A4000 rate; idle is not visible here
    PRICE = {'A': ('audio', 0.00045 / 60), 'B': ('exec', 0.25 / 3600)}

    def progress(self, done, n, final=False):
        s = self.stats; el = time.time() - s['t0']; rate = s['audio_s'] / el if el else 0
        basis, unit = self.PRICE[self.prov.arm]
        spent = (s['audio_s'] if basis == 'audio' else s.get('exec_s', 0.0)) * unit
        frac = done / n if n else 0
        eta_h = (el / frac - el) / 3600 if frac else float('nan')
        proj = spent / frac if frac else float('nan')
        log(f'{"done" if final else "progress"}: {done:,}/{n:,} sent ({frac:.1%}), ok {s["ok"]:,}, failed {s["failed"]:,} | '
            f'{s["audio_s"]/3600:.2f} audio-h in {el/3600:.2f} h ({rate:.1f}x realtime) | '
            f'spent ~${spent:.2f}, projected ~${proj:.0f} total, eta {eta_h:.1f} h')

def upload(repo, files, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token); api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f'results/{f.name}', repo_id=repo, repo_type='dataset')
    log(f'uploaded {len(files)} file(s) -> {repo}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A', 'B'])
    ap.add_argument('--audio-cache', default=None, metavar='TAG', help='read audio from cache/audio/TAG (built by data.build_audio_cache); extract on the fly if missing')
    ap.add_argument('--endpoint', default=None, help='Arm B: RunPod endpoint id (default: RUNPOD_ENDPOINT_ID / cache)')
    ap.add_argument('--shard-mod', type=int, nargs=2, metavar=('K', 'N'), default=None,
                    help='take only chunks whose stable hash %% N == K, to split one corpus across N runners')
    ap.add_argument('--run', default=None, help='run name; default <arm>_<model short>')
    ap.add_argument('--speakers', type=int, nargs='*'); ap.add_argument('--sessions', type=int, nargs='*')
    ap.add_argument('--chunk-ids', type=str, default=None, help='file with one chunk_id per line')
    ap.add_argument('--min-quality', type=float); ap.add_argument('--min-s', type=float); ap.add_argument('--max-s', type=float)
    ap.add_argument('--limit', type=int); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--upload-repo', default=None, help='e.g. Dolevabudi/knesset-committees-inference')
    ap.add_argument('--upload-every-min', type=float, default=15)
    a = ap.parse_args()

    prov = P.make(a.arm, **({'endpoint_id': a.endpoint} if a.endpoint else {}))
    # Concurrency = requests in flight, not GPUs.  Arm B's endpoint has 3 GPU
    # workers doing ~0.8 s per chunk; a queue of ~12 keeps them busy while the
    # other 9 requests sit in submit/poll/transfer.  Arm A: 8 gave 8.5x realtime.
    workers = a.workers or (8 if a.arm == 'A' else 12)
    run = a.run or f'{a.arm}_{prov.model.split("/")[-1]}'
    out = OUT / f'{run}.jsonl'
    log(f'arm {a.arm}: {prov.model} via {prov.name} -> {out}  (workers {workers})')
    if a.arm == 'B':
        h = prov.health(); log(f'runpod health: workers {h.get("workers")} jobs {h.get("jobs")}')

    df = D.index()
    ids = [l.strip() for l in open(a.chunk_ids) if l.strip()] if a.chunk_ids else None
    rows = D.select(df, speakers=a.speakers, sessions=a.sessions, min_quality=a.min_quality,
                    min_s=a.min_s, max_s=a.max_s, limit=a.limit, seed=a.seed, chunk_ids=ids)
    if a.shard_mod:
        import zlib
        k, n_ = a.shard_mod
        rows = rows[rows.chunk_id.map(lambda c: zlib.crc32(c.encode()) % n_) == k]
    done = load_done(out, include_failed=not a.retry_failed)
    rows = rows[~rows.chunk_id.isin(done)]
    log(f'{len(rows):,} chunks to send ({len(done):,} already done), {rows.duration_s.sum()/3600:.2f} h of audio, '
        f'{rows.speaker_id.nunique()} speakers, {rows.shard.nunique()} shards')
    if rows.empty: return

    stop_upl = threading.Event()
    if a.upload_repo:
        tok = P.secret('HF_WRITE_TOKEN', 'hf_write_token')
        def loop():
            while not stop_upl.wait(a.upload_every_min * 60):
                try: upload(a.upload_repo, [out], tok)
                except Exception as e: log(f'upload failed: {e}')
        threading.Thread(target=loop, daemon=True).start()

    Runner(prov, out, workers=workers, tag=a.audio_cache).run(rows)
    stop_upl.set()
    if a.upload_repo:
        upload(a.upload_repo, [out], tok)

if __name__ == '__main__':
    main()
