# Stage 4 — inference over the committees chunk corpus

Transcribes [`Hadasy/knesset-committees-chunks`](https://huggingface.co/datasets/Hadasy/knesset-committees-chunks)
(1,204,617 chunks, 3,840 h, 330 identified speakers) with the two arms, writing one
JSONL row per chunk with the protocol reference alongside the hypothesis.

| Arm | Model | Served by | Why this checkpoint |
|---|---|---|---|
| **A** | `openai/whisper-large-v3` | HF Inference Providers → `deepinfra` | Same provider and price as Stage 1, so Arm A is comparable across corpora |
| **B** | `ivrit-ai/whisper-large-v3-turbo-ct2` | RunPod serverless, ivrit.ai's own worker image | The checkpoint ivrit.ai recommend for inference; `ivrit-ai/whisper-large-v3` is the training checkpoint |

Neither model has seen committee audio: Arm B's card lists only plenum data, and
`ivrit-ai/knesset-committees` postdates it by fourteen months. That is the reason
this stage exists — see the top-level README.

## Files

| file | holds |
|---|---|
| `providers.py` | one `transcribe(bytes) -> Result` over both back ends; the exact payload/response contracts, probed live |
| `data.py` | pinned dataset revision, metadata index, per-shard audio fetch, selection filters |
| `run.py` | resumable threaded runner: JSONL append, resume by `chunk_id`, 429 cooldown, billing stop, periodic HF mirror |
| `verify.py` | Stage 1's WER/CER scoring (`normalize_he` + Levenshtein) plus pipeline invariants; pairs A vs B |
| `verify_chunks.txt` | the fixed 10-chunk sample (one per speaker, 5–25 s, quality ≥ 0.7) both arms are verified on |
| `subset_stage1.parquet` | the evaluated subset: 65,990 chunks, 1 h per MK (quality ≥ 0.5), round-robin over sessions |
| `run_lean.py` | the runner that did the full subset: one process, one shard at a time streamed with DuckDB, both arms fed from it. Built for a 17 GB laptop; `run.py`'s design, without the memory |
| `merge.py` | JSONLs → `inference.parquet` (wide: `hypothesis_A`, `hypothesis_B`, `hypothesis_A_auto`), `inference_long.parquet`; `--upload` publishes them with a dataset card |
| `coverage.py` | one row per corpus chunk: which arm has it, any pending error — the record of what ran |
| `validate_final.py` | the acceptance test: coverage, identity against the index, hypotheses, sampled audio, scoring. Exits non-zero on any failure |
| `finish.sh` | after the runner: retry pass, merge, coverage, validate, upload **only on pass** |

## Secrets

Read from the environment, else `the repo-level cache/` (mode 600, git-ignored). Never
written to any output.

```
HF_INFERENCE_TOKEN   hf_inference_token    Arm A   (scope: inference.serverless.write)
RUNPOD_API_KEY       runpod_api_key        Arm B
RUNPOD_ENDPOINT_ID   runpod_endpoint_id    Arm B
HF_WRITE_TOKEN       hf_write_token        only with --upload-repo
```

## Run

Verification (10 fixed chunks, both arms, then the invariants):

```bash
python src/inference/providers.py                       # offline self-checks
python src/inference/data.py                            # index + 3 real fetches
python src/inference/run.py --arm A --chunk-ids src/inference/verify_chunks.txt
python src/inference/run.py --arm B --chunk-ids src/inference/verify_chunks.txt
python src/inference/verify.py src/inference/outputs/A_*.jsonl src/inference/outputs/B_*.jsonl --expect src/inference/verify_chunks.txt
```

The subset run, as it was actually done (resumable; re-running sends only what
has no successful row yet):

```bash
ulimit -n 4096
python src/inference/run_lean.py --workers-a 96 --workers-b 24 --prefetch 2   # both arms
zsh src/inference/finish.sh          # waits, retries, merges, validates, uploads on pass
```

`run_lean.py` reads shards with DuckDB (`pip install duckdb`); without it, it falls
back to Arrow at roughly twice the peak memory.

`run.py` is the per-arm runner with filters (`--speakers`, `--sessions`,
`--min-quality`, `--min-s`, `--max-s`, `--limit`, `--chunk-ids FILE`); it resumes,
and `--retry-failed` re-sends errored rows. It reads whole shards, so on a small
machine prefer `run_lean.py`.

## Cost and time

The subset (230 h per arm) cost about $12 per arm: Arm A twice (auto-detect, then with
Hebrew forced), Arm B on 10 RunPod workers for about 4 h including idle. Wall time was
set by pulling 410 shards of 770 MB to a laptop, not by either provider; the numbers are
in `docs/inference.md` § Result.

### Full corpus (3,840 h), if ever needed

Derived in `docs/inference.md` § Cost; summary:

- **Arm A**, deepinfra: $0.00045 / audio-minute, no per-call fee, no HF markup → **~$104**.
- **Arm B**, RunPod: bills GPU worker-seconds, not calls. Warm, the GPU runs 23× realtime
  GPU-bound: speed = worker count (~14× realtime each). 10 workers: $5.80/h × ~26 h ≈ **$150**.
  Fewer workers cost the same and take longer. See the doc.
  Raise the endpoint's `idleTimeout` (5 s → 60–120 s) so workers stay warm between bursts.

Neither provider charges per call, so the median 7.8 s chunk does not inflate cost — it
inflates wall time, which is why the runner keeps many jobs in flight. Run a sustained pilot
(a few hundred chunks, both arms) and read `exec_s` / `queue_s` before a full run.

## Output row

```
chunk_id, arm, provider, model, speaker_id, session, session_date, knesset,
duration_s, quality, reference, hypothesis, language, latency_s, error, ts, raw
```
`raw` carries provider extras (RunPod `exec_ms`, `delay_ms`, `job_id`, word timings).

