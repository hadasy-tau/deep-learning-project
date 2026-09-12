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

## Secrets

Read from the environment, else `stage3/cache/` (mode 600, git-ignored). Never
written to any output.

```
HF_INFERENCE_TOKEN   hf_inference_token    Arm A   (scope: inference.serverless.write)
RUNPOD_API_KEY       runpod_api_key        Arm B
RUNPOD_ENDPOINT_ID   runpod_endpoint_id    Arm B
HF_WRITE_TOKEN       hf_write_token        only with --upload-repo
```

## Run

```bash
python stage4/providers.py                       # offline self-checks
python stage4/data.py                            # index + 3 real fetches
python stage4/run.py --arm A --limit 10          # smoke
python stage4/run.py --arm B --chunk-ids stage4/verify_chunks.txt
python stage4/verify.py stage4/outputs/A_*.jsonl stage4/outputs/B_*.jsonl --expect stage4/verify_chunks.txt

python stage4/run.py --arm A --upload-repo Dolevabudi/knesset-committees-inference   # everything
python stage4/run.py --arm B --upload-repo Dolevabudi/knesset-committees-inference
```

Filters: `--speakers`, `--sessions`, `--min-quality`, `--min-s`, `--max-s`, `--limit`,
`--chunk-ids FILE`. Re-running resumes; `--retry-failed` re-sends errored rows.

## Cost and time, full corpus (3,840 h)

Derived in `docs/stage4-inference.md` § Cost; summary:

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
duration_s, quality, reference, hypothesis, latency_s, error, ts, raw
```
`raw` carries provider extras (RunPod `exec_ms`, `delay_ms`, `job_id`, word timings).
