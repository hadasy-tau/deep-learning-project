#!/usr/bin/env python3
"""Full VoxKnesset inference run: openai/whisper-large-v3 via HF Inference Providers.

Streams the dataset (no download), transcribes every segment, writes JSONL
incrementally (crash-safe, resumable), and periodically uploads results to a
HF dataset repo.

Env vars:
  HF_TOKEN        - inference token (billed account)
  HF_DATA_TOKEN   - dataset read token (account with VoxKnesset license accepted)
  HF_WRITE_TOKEN  - token with repo.write for the results repo

Usage:
  python full_run.py --workers 80 --provider deepinfra
  (re-running resumes: already-transcribed filenames are skipped)
"""

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gc

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, InferenceClient

DATASET = "ivrit-ai/VoxKnesset"
SPLITS = {"train": 49_433, "test": 17_237}
# parallel shard streams per split. Keep LOW: each stream holds a decompressed
# ~400-570MB parquet row group (100 rows) in RAM, and one stream already yields
# ~10 samples/s — enough to feed all inference workers.
PRODUCERS = {"train": 1, "test": 1}
FOOTPRINT_LIMIT_GB = 14  # watchdog: clean restart (exit 3) past this. NOTE: must
# watch phys footprint (top), not RSS - macOS compresses leaked pages so RSS lies.
# Legit working set is ~6-8GB (two decoding row groups + in-flight audio + refs),
# so the limit must sit well above that; 14GB leaves headroom on a 32GB machine.
SHARD_FILES = {
    "train": [f"data/train-{i:05d}-of-00050.parquet" for i in range(50)],
    "test": [f"data/test-{i:05d}-of-00020.parquet" for i in range(20)],
}
PRICE_PER_AUDIO_MIN = 0.00045  # deepinfra whisper-large-v3
RETRY_DELAYS = [2, 8, 30]
UPLOAD_EVERY_SECONDS = 15 * 60
QUEUE_MAX = 32  # bounds producer prefetch (audio bytes waiting for a worker)

stop_event = threading.Event()  # abort everything (billing error / shutdown)
producer_stop = threading.Event()  # stop fetching new data (limit reached)
watchdog_tripped = threading.Event()  # memory limit hit -> drain in-flight, exit 3
rate_gate = threading.Event()  # cleared during a global 429 cooldown; workers wait
rate_gate.set()
rate_lock = threading.Lock()
write_lock = threading.Lock()
stats_lock = threading.Lock()
stats = {"ok": 0, "failed": 0, "audio_seconds": 0.0, "started": None}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def audio_duration_seconds(raw: bytes) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(io.BytesIO(raw))
        return info.frames / info.samplerate
    except Exception:
        return None


def load_done(out_dir: Path, include_failed: bool = False) -> set[str]:
    """Filenames to skip. include_failed=True also skips rows that failed —
    used for the main pass so scattered failure-holes don't force a ~400MB
    row-group re-download every cycle; a final --retry-failed pass sweeps them."""
    done = set()
    for f in out_dir.glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if include_failed or r.get("error") is None:
                    done.add(r["filename"])
            except json.JSONDecodeError:
                continue  # torn line from a crash; that sample will be redone
    return done


def load_references(token: str) -> dict[str, str]:
    """transcripts.parquet at repo root: filename -> reference text (~66k rows)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    with fs.open(f"datasets/{DATASET}/transcripts.parquet") as f:
        table = pq.read_table(f)
    return dict(zip(table.column("filename").to_pylist(), table.column("text").to_pylist()))


NAMES_MAP_PATH = "names_map.jsonl"  # per file: filenames grouped by row group (static, built once)


def ensure_names_map(token: str) -> dict[str, list[list[str]]]:
    """file path -> [[filenames of rg0], [rg1], ...]. Remote-scanning all 70 files
    every cycle retained ~12GB of allocator pages and starved the watchdog budget,
    so the (static) map is built once and cached on disk."""
    names_map: dict[str, list[list[str]]] = {}
    map_file = Path(NAMES_MAP_PATH)
    if map_file.exists():
        for line in map_file.open(encoding="utf-8"):
            rec = json.loads(line)
            names_map[rec["file"]] = rec["rg_names"]
    missing = [p for split in SPLITS for p in SHARD_FILES[split] if p not in names_map]
    if not missing:
        return names_map
    log(f"building names map for {len(missing)} files (one-time) ...")
    fs = HfFileSystem(token=token)
    with map_file.open("a", encoding="utf-8") as out:
        for path in missing:
            with fs.open(f"datasets/{DATASET}/{path}") as f:
                pf = pq.ParquetFile(f)
                names = [os.path.basename(r["path"] or "") for r in pf.read(columns=["audio.path"]).column(0).to_pylist()]
                rg_names, offset = [], 0
                for rg in range(pf.num_row_groups):
                    n = pf.metadata.row_group(rg).num_rows
                    rg_names.append(names[offset:offset + n])
                    offset += n
            names_map[path] = rg_names
            out.write(json.dumps({"file": path, "rg_names": rg_names}) + "\n")
            out.flush()
    log("names map ready")
    return names_map


def producer(split: str, shard_idx: int, n_shards: int, token: str, done: set[str],
             q: queue.Queue, names_map: dict[str, list[list[str]]]):
    """Reads parquet row groups directly with pyarrow (the `datasets` streaming
    path leaked ~27MB/sample). Memory is released back to the OS per row group."""
    try:
        fs = HfFileSystem(token=token)
        for path in SHARD_FILES[split][shard_idx::n_shards]:
            if stop_event.is_set() or producer_stop.is_set():
                return
            rg_pending = [any(nm not in done for nm in rg) for rg in names_map[path]]
            if not any(rg_pending):
                continue
            log(f"{path}: {sum(rg_pending)}/{len(rg_pending)} row groups pending")
            with fs.open(f"datasets/{DATASET}/{path}") as f:
                pf = pq.ParquetFile(f)
                for rg in range(pf.num_row_groups):
                    if stop_event.is_set() or producer_stop.is_set():
                        return
                    if not rg_pending[rg]:
                        continue
                    rows = pf.read_row_group(rg).to_pylist()
                    for j in range(len(rows)):
                        row, rows[j] = rows[j], None  # free consumed refs early
                        audio = row.pop("audio")
                        filename = os.path.basename(audio.get("path") or "")
                        if filename in done:
                            continue
                        item = {"split": split, "filename": filename,
                                "audio_bytes": audio["bytes"], "metadata": row}
                        while not (stop_event.is_set() or producer_stop.is_set()):
                            try:
                                q.put(item, timeout=5)
                                break
                            except queue.Full:
                                continue
                        item = row = audio = None
                    del rows
                    pa.default_memory_pool().release_unused()
                    gc.collect()
    except Exception as e:
        log(f"PRODUCER ERROR {split}/{shard_idx}: {type(e).__name__}: {e}")
    finally:
        q.put(None)  # this producer is finished


def is_billing_error(err: Exception) -> bool:
    s = str(err).lower()
    return "402" in s or "credit" in s or "payment" in s or "quota" in s


def transcribe(client: InferenceClient, model: str, item: dict, refs: dict[str, str], out_files: dict):
    text, error = None, None
    t0 = time.perf_counter()
    deadline = time.time() + 20 * 60  # a sample survives up to 20min of 429 storms
    attempt = 0
    while time.time() < deadline:
        if stop_event.is_set():
            return
        rate_gate.wait(timeout=60)  # everyone pauses during a global cooldown
        try:
            out = client.automatic_speech_recognition(audio=item["audio_bytes"], model=model)
            text, error = out.text, None
            break
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if is_billing_error(e):
                log(f"BILLING ERROR - stopping run: {error}")
                stop_event.set()
                return
            if "429" in error:
                # one global cooldown instead of 64 independent hammers
                with rate_lock:
                    if rate_gate.is_set():
                        rate_gate.clear()
                        log("429 storm: global 120s cooldown")
                        threading.Timer(120, rate_gate.set).start()
                continue  # storm waits don't consume regular attempts
            attempt += 1
            if attempt >= 4:
                break
            time.sleep(RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)])
    latency = time.perf_counter() - t0
    if error is not None and stats["failed"] < 5:  # sample first few errors to the log
        log(f"SAMPLE ERROR {item['filename']}: {error[:200]}")
    dur = audio_duration_seconds(item["audio_bytes"])
    record = {
        "filename": item["filename"],
        "split": item["split"],
        "audio_seconds": round(dur, 2) if dur else None,
        "latency_seconds": round(latency, 2),
        "transcription": text,
        "reference": refs.get(item["filename"]),
        "error": error,
        **item["metadata"],
    }
    with write_lock:
        f = out_files[item["split"]]
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
    with stats_lock:
        stats["ok" if error is None else "failed"] += 1
        if dur:
            stats["audio_seconds"] += dur


def uploader_loop(api: HfApi, repo_id: str, out_dir: Path):
    while not stop_event.wait(UPLOAD_EVERY_SECONDS):
        upload_results(api, repo_id, out_dir)


def upload_results(api: HfApi, repo_id: str, out_dir: Path):
    for f in sorted(out_dir.glob("*.jsonl")):
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"data/{f.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"checkpoint {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
        except Exception as e:
            log(f"UPLOAD ERROR {f.name}: {type(e).__name__}: {e}")
    log(f"uploaded checkpoints -> {repo_id}")


def watchdog_loop(out_files: dict):
    """Restart cleanly (exit 3, wrapper relaunches + resumes) before memory
    pressure can freeze the machine."""
    import subprocess

    pid = os.getpid()
    ticks = 0
    while not stop_event.wait(30):
        try:
            out = subprocess.check_output(
                ["top", "-l", "1", "-pid", str(pid), "-stats", "mem"], text=True
            )
            val = out.strip().splitlines()[-1].strip().rstrip("+-")
            mult = {"K": 1 / 1048576, "M": 1 / 1024, "G": 1.0}
            fp_gb = float(val[:-1]) * mult[val[-1]] if val[-1] in mult else 0.0
        except Exception:
            continue
        ticks += 1
        if ticks % 10 == 0:
            log(f"memory footprint: {fp_gb:.1f}GB")
        if fp_gb > FOOTPRINT_LIMIT_GB:
            # Do NOT hard-exit here: that re-orphans long samples mid-inference
            # every cycle (they become permanent "holes"). Stop producers, let
            # in-flight work drain, then main exits with rc 3 for the wrapper.
            log(f"MEMORY WATCHDOG: footprint={fp_gb:.1f}GB > {FOOTPRINT_LIMIT_GB}GB, draining for clean restart")
            watchdog_tripped.set()
            producer_stop.set()
            return


def reporter_loop(total_todo: int):
    while not stop_event.wait(60):
        with stats_lock:
            done = stats["ok"] + stats["failed"]
            mins = stats["audio_seconds"] / 60
        elapsed = time.time() - stats["started"]
        rate = done / elapsed if elapsed else 0
        eta_h = (total_todo - done) / rate / 3600 if rate else float("inf")
        log(
            f"progress {done}/{total_todo} ok={stats['ok']} failed={stats['failed']} "
            f"rate={rate * 60:.0f}/min eta={eta_h:.1f}h est_cost=${mins * PRICE_PER_AUDIO_MIN:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--provider", default="deepinfra")
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--repo", default="Dolevabudi/voxknesset-whisper-large-v3-baseline")
    parser.add_argument("--out", default="results_full")
    parser.add_argument("--limit", type=int, default=0, help="stop after N new samples (0 = all)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="final sweep: re-attempt rows whose stored result is an error")
    args = parser.parse_args()

    inference_token = os.environ["HF_TOKEN"]
    data_token = os.environ.get("HF_DATA_TOKEN", inference_token)
    write_token = os.environ.get("HF_WRITE_TOKEN", inference_token)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    done = load_done(out_dir, include_failed=not args.retry_failed)
    total = sum(SPLITS.values())
    todo = total - len(done)
    if args.limit:
        todo = min(todo, args.limit)
    log(f"total={total} already_done={len(done)} todo={todo}")
    if todo == 0:
        log("nothing to do")
        return 0

    log("loading reference transcripts ...")
    refs = load_references(data_token)
    log(f"loaded {len(refs)} references")
    names_map = ensure_names_map(data_token)

    api = HfApi(token=write_token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    out_files = {s: (out_dir / f"{s}.jsonl").open("a", encoding="utf-8") for s in SPLITS}
    client = InferenceClient(provider=args.provider, api_key=inference_token)
    q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
    stats["started"] = time.time()

    n_producers = 0
    for split, n in PRODUCERS.items():
        for i in range(n):
            threading.Thread(target=producer, args=(split, i, n, data_token, done, q, names_map), daemon=True).start()
            n_producers += 1

    threading.Thread(target=uploader_loop, args=(api, args.repo, out_dir), daemon=True).start()
    threading.Thread(target=reporter_loop, args=(todo,), daemon=True).start()
    threading.Thread(target=watchdog_loop, args=(out_files,), daemon=True).start()

    submitted = 0
    finished_producers = 0
    # cap in-flight samples: the pool's internal queue is unbounded, and audio
    # bytes are large — without this the process OOMs when fetch outpaces inference
    in_flight = threading.Semaphore(args.workers + 16)

    def run_one(item):
        try:
            transcribe(client, args.model, item, refs, out_files)
        finally:
            in_flight.release()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while finished_producers < n_producers and not stop_event.is_set():
            item = q.get()
            if item is None:
                finished_producers += 1
                continue
            in_flight.acquire()
            pool.submit(run_one, item)
            submitted += 1
            if args.limit and submitted >= args.limit:
                producer_stop.set()  # stop producers from downloading more data
                break
        pool.shutdown(wait=True)

    stop_event.set()
    for f in out_files.values():
        f.close()
    upload_results(api, args.repo, out_dir)

    with stats_lock:
        mins = stats["audio_seconds"] / 60
        tag = "CYCLE-END (watchdog, drained cleanly)" if watchdog_tripped.is_set() else "DONE"
        log(
            f"{tag} ok={stats['ok']} failed={stats['failed']} "
            f"audio_hours={mins / 60:.1f} est_cost=${mins * PRICE_PER_AUDIO_MIN:.2f} "
            f"wall_hours={(time.time() - stats['started']) / 3600:.2f}"
        )
    return 3 if watchdog_tripped.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
