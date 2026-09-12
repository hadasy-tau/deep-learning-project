"""Stage 4 runner: transcribe the committee chunks with one arm, resumably.

    python stage4/run.py --arm A --limit 10                 # smoke
    python stage4/run.py --arm B --speakers 526 4405        # a few speakers
    python stage4/run.py --arm A                            # everything (~3,840 h)

Design, inherited from stage1/inference_openai_whisper_large_v3/full_run.py:
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    def __init__(self, prov, out_path, workers=8, max_attempts=4):
        self.prov, self.out, self.workers, self.max_attempts = prov, out_path, workers, max_attempts
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
                   error=err, ts=time.strftime('%Y-%m-%dT%H:%M:%S'), raw=raw)
        with self.lock:
            self.fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); self.fh.flush()
            self.stats['ok' if err is None else 'failed'] += 1
            if err is None: self.stats['audio_s'] += float(row.duration_s)
        return rec

    def run(self, rows, log_every=25):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.out.open('a', encoding='utf-8')
        n = len(rows); done = 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = []
            for row, audio in D.audio_iter(rows):
                if self.stop.is_set(): break
                futs.append(ex.submit(self.one, row, audio))
                if len(futs) >= self.workers * 4:          # bound in-flight audio in RAM
                    for f in as_completed(futs[:self.workers]):
                        done += 1
                    futs = [f for f in futs if not f.done()]
                if done and done % log_every == 0: self.progress(done, n)
            for f in as_completed(futs): done += 1
        self.fh.close(); self.progress(done, n, final=True)

    def progress(self, done, n, final=False):
        s = self.stats; el = time.time() - s['t0']; rate = s['audio_s'] / el if el else 0
        log(f'{"done" if final else "progress"}: {done}/{n} sent, ok {s["ok"]}, failed {s["failed"]}, '
            f'{s["audio_s"]/60:.1f} audio-min in {el/60:.1f} min ({rate:.1f}x realtime)')

def upload(repo, files, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token); api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f'results/{f.name}', repo_id=repo, repo_type='dataset')
    log(f'uploaded {len(files)} file(s) -> {repo}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A', 'B'])
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

    prov = P.make(a.arm)
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

    Runner(prov, out, workers=workers).run(rows)
    stop_upl.set()
    if a.upload_repo:
        upload(a.upload_repo, [out], tok)

if __name__ == '__main__':
    main()
