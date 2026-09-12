# Stage 4 — inference over the committees chunk corpus

How the two ASR arms are run over `Hadasy/knesset-committees-chunks`, what each
piece of `stage4/` does, what was verified, and what it costs. Written after the
pipeline was built and verified live (2026-09-11); every number below was
measured, and the section on cost corrects an earlier estimate.

## Why this stage exists

Stage 1 measured Arm A and Arm B on VoxKnesset. That comparison is contaminated:
Arm B (`ivrit-ai/whisper-large-v3`) was trained for three epochs on
`ivrit-ai/knesset-plenums-whisper-training`, and VoxKnesset is cut from the same
plenum audio. Arm B had seen the test waveforms.

The committees corpus is clean by construction: it is not in Arm B's training
data, `ivrit-ai/knesset-committees` was published fourteen months after the
model, and no `knesset-committees-whisper-training` exists. Stage 4 re-runs both
arms on it.

## Inputs

| dataset | what it is | how Stage 4 uses it |
|---|---|---|
| `Hadasy/knesset-committees-chunks` | 1,204,617 single-speaker chunks ≤ 30 s, FLAC inline, protocol text, verified `speaker_id` (Knesset PersonID) and demographics. 410 shards, 275 GB. | the audio and the reference text |
| `Dolevabudi/knesset-committees-speakers` (`segments.parquet`) | the Stage 3 index the chunks were cut from; carries `label` (`mk` / `former_mk`) | optional join to keep only sitting MKs — not needed for Stage 1's error map |

The chunk corpus is pinned to one dataset revision for the life of a run
(`stage4/cache/revision.json`), so a re-upload cannot change the data under a
running job.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │ data.py                                       │
                    │  index()      metadata of all 1.2 M chunks     │
                    │               (HTTP Range reads, 0.2 % of      │
                    │               each shard; cached once)         │
                    │  select()     --speakers/--sessions/--limit …  │
                    │  audio_iter() FLAC bytes, one shard at a time  │
                    └──────────────┬───────────────────────────────┘
                                   │ (row, audio bytes)
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ run.py  Runner                                                     │
   │  ThreadPool(N in flight) ──► providers.transcribe(bytes) ──► Result │
   │  per row: append JSON line, flush           (crash-safe)           │
   │  resume by chunk_id       429 → one global cooldown                │
   │  --retry-failed           402/quota → stop the run                 │
   │  --upload-repo            mirror JSONL to HF every N min           │
   └───────────────┬───────────────────────────────────────────────────┘
                   │
       ┌───────────┴─────────────┐
       ▼                         ▼
 providers.HFProvider      providers.RunPodProvider
 Arm A                     Arm B
 openai/whisper-large-v3   ivrit-ai/whisper-large-v3-turbo-ct2
 HF Inference Providers    RunPod serverless, ivrit.ai's worker image
 provider = deepinfra      POST /run → poll /status → parse segments
                                   │
                                   ▼
                     outputs/<run>.jsonl  (one row per chunk)
                                   │
                                   ▼
                     verify.py  WER/CER (Stage 1 scoring), invariants,
                                paired A-vs-B table
```

### `providers.py` — one call shape over two back ends

`transcribe(audio_bytes) -> Result(text, latency_s, raw)`. The two contracts were
probed live before the client was written and are recorded in the module
docstring; they differ from what the docs and the `ivrit` package's convenience
API suggest:

- **RunPod** wants `input.transcribe_args.blob` (base64 FLAC), not `input.data`;
  sending the latter fails with `transcribe_args field not provided`. The
  response is `output[0].result`, a list of batches of `{type, data}` entries;
  only `type == 'segments'` carries text, and `type == 'progress'` entries are
  interleaved and must be skipped. The client submits to `/run` and polls
  `/status` rather than blocking on `/runsync`, so many jobs can sit in the
  endpoint's queue at once (see *throughput*).
- **HF** is `InferenceClient(provider='deepinfra').automatic_speech_recognition`,
  which for this model returns text only (`chunks=None`). Same provider and
  price Stage 1 used, so Arm A is comparable across corpora.

Arm B is the **turbo-ct2** checkpoint because that is what ivrit.ai serve for
inference; `ivrit-ai/whisper-large-v3` is the training checkpoint.

Transient failures (429, 5xx, queue timeout, network) raise `TransientError`
and are retried; anything else is recorded as the row's `error` and the run
moves on.

### `data.py` — read 275 GB without holding it

Metadata for all chunks (id, speaker, session, duration, quality, reference
text, …) is ~150 MB and is pulled once into `cache/index.parquet`. Selection
happens on that table; audio is fetched only for the rows that survive, one
shard at a time in shard order, so each touched shard is read once.

Metadata is read over plain HTTP Range requests (`RangeFile`), not
`HfFileSystem`: pyarrow asks for the footer and then exactly the column byte
ranges it needs, and the fsspec block cache that `HfFileSystem` puts in between
made this take 23 s per shard and stall outright under 8 threads. Range requests:
5 HTTP calls, 16 s, nothing to hang.

### `run.py` — the runner

Inherited from `stage1/inference_openai_whisper_large_v3/full_run.py`:

- one JSONL per `(arm, run)`, appended and flushed per row; a torn last line
  from a crash is skipped on resume
- resume by `chunk_id`: rows without `error` are never re-sent; `--retry-failed`
  re-sends the ones that errored
- a thread pool over network calls (the work is I/O-bound); one global 90 s
  cooldown on 429 so N workers do not retry independently; billing/quota errors
  stop the run
- the protocol reference rides in every row, so scoring needs no join
- `--upload-repo` mirrors the JSONL to a private HF dataset every N minutes

Filters: `--speakers`, `--sessions`, `--chunk-ids FILE`, `--min-quality`,
`--min-s`, `--max-s`, `--limit` (seeded uniform sample).

### `verify.py` — scoring and invariants

Stage 1's scoring, unchanged: `normalize_he()` from `stage2/pipeline.py`, then
word- and character-level Levenshtein (rapidfuzz); corpus WER = Σerrors /
Σreference words. Invariants: no errors, unique `chunk_id`, non-empty and Hebrew
hypotheses, reference present, exactly the requested chunk set, and — with two
runs — identical coverage so A vs B is paired.

## Output row

```
chunk_id, arm, provider, model,
speaker_id, session, session_date, knesset, duration_s, quality,
reference, hypothesis,
latency_s      our wall clock: submit → result, incl. queue and polling
exec_s         provider-reported GPU time (RunPod only)
queue_s        provider-reported queue wait (RunPod only)
error, ts, raw (provider extras: job id, worker, word timings)
```

`latency_s` cannot compare the two arms' speed; `exec_s` is the fair number.

## Verification

Ten fixed chunks (`stage4/verify_chunks.txt`: one per speaker, 5–25 s,
alignment quality ≥ 0.7), sent through the real pipeline to both arms:

| | Arm A | Arm B |
|---|---|---|
| calls | 10/10 ok | 10/10 ok |
| corpus WER / CER | 0.520 / 0.389 | 0.389 / 0.307 |
| per-chunk, B vs A | — | B better on 9, tie on 1 |

All 14 pipeline invariants passed. The HF mirror was exercised once against
`Dolevabudi/knesset-committees-inference` (private).

The WER level is the committees register, not the pipeline: the reference is a
cleaned stenographic protocol, while both models faithfully transcribe the
repetitions and cross-talk on the tape. Stage 1's plenum medians were 0.20 (A) /
0.10 (B). One chunk, both models:

```
REF: באולם ליד דנים כעת בחוק הנאמנות בתרבות עלזה שיש מי שבא מכיוון היוצרים, ואומר: פה אתם רוצים לעשות בדיוק אותו דבר.
B:   באולם ליד, דנים עכשיו, באולם ליד, דנים עכשיו בחוק הנאמנות בתרבות על זה שיש מי שבא מכיוון היוצרים ואומר, מונעים מאיתנו. פה אתם רוצים לעשות בדיוק אותו ד
```

## Cost — corrected

The first estimate in this stage's README (~$240 for Arm B) was wrong, and
the question "is there a per-call overhead, and do short chunks cost more?"
is exactly the right one. Here is what each provider actually bills, from
the measured runs and the providers' own price pages.

### Arm A — deepinfra via HF

deepinfra's price for `openai/whisper-large-v3` is `type: input_length`,
**0.00075 ¢ per input second** (= $0.00045 / audio-minute). HF adds no markup
(its pricing page: "no markup from Hugging Face"). There is no per-call fee and
no minimum duration in the published price, so a 2 s chunk costs 2 s.

**Full corpus: 3,840.6 h × 60 × $0.00045 = $104.** This number stands.

Whether the 38.8 % of chunks under 5 s are rounded up is not stated on the
price page; if they were rounded to a 5 s minimum it would add at most
~$4 (they hold 8.7 % of the audio). Not material.

### Arm B — RunPod serverless

RunPod bills **worker-seconds while a worker is running**, at the GPU's hourly
rate (`AMPERE_16` pool; RTX A4000 is $0.25/h secure, $0.17/h community). It does
not bill per call or per audio second. What matters is therefore GPU seconds
per audio second — and that has to be measured *warm*, because a worker that
just started spends its first call loading the model.

From the 10-chunk run, splitting warm from cold:

| | calls | GPU s / audio s |
|---|---|---|
| warm (exec < 2 s) | 7 | **0.044** — 23× realtime |
| cold (model load inside exec) | 3 | 0.232 |

The earlier estimate used ~0.25 GPU-s per audio-s, i.e. it assumed *every* call
was cold. That is what a 10-call burst against `workersMin 0` /
`idleTimeout 5 s` looks like, and it is not steady state.

**Full corpus, warm: 3,840.6 h × 0.044 = 169 GPU-hours ≈ $42 on an A4000
(secure), $29 community.** Roughly a sixth of the earlier figure.

Two things still push the real bill above $42:

1. **Idle time.** RunPod bills a worker from start until it scales down
   (`idleTimeout` after its last job). With the queue kept full this is small;
   with gaps between batches it is not. Keep the queue full (the runner's
   `--workers` sets in-flight jobs, default 12 for three GPUs) and the idle
   share stays under ~10 %.
2. **Cold starts.** Each scale-up pays one model load (~3 s of exec on the
   sample). At `idleTimeout 5 s` the endpoint scales to zero between every
   pause. Raising it to 60–120 s in the RunPod console costs a few idle
   seconds per pause and removes the reloads.

**Working estimate for Arm B: $45–60**, not $240. The account balance at
verification time was $4.41 with an $80 spend limit, so a top-up is still
needed, but a much smaller one.

### Short chunks

Neither provider charges per call, so the median 7.8 s chunk length does not
inflate cost directly. It inflates *wall time*: every chunk pays the same
submit → queue → transfer round trip regardless of length, and on RunPod that
overhead (~2–5 s) exceeds the GPU time (~0.8 s) for a typical chunk. That is
why the runner keeps many jobs in flight rather than one per GPU — the
round-trips overlap, the GPU stays busy, and the idle share (which *is*
billed) stays small.

### Wall time

Measured throughput on the 10-chunk sample is cold-start dominated and is not a
steady-state number. The GPU-side rate is 23× realtime per worker, so three warm
workers process the corpus in ~56 GPU-hours of work spread over roughly 2–3 days
of wall time at high queue occupancy. Arm A at 8 in flight gave 3.6× realtime on
the sample; Stage 1 reached ~10 samples/s at 80 workers on the same provider, so
`--workers` can go well above 8 once a pilot shows no 429s.

**Run a sustained pilot first** — a few hundred chunks of one speaker, both
arms — and read `exec_s` / `queue_s` from the JSONL. That gives the real
throughput and the real idle share before either full run is committed.

## Run

```bash
python stage4/providers.py                                     # offline self-checks
python stage4/data.py                                          # build/refresh the index; 3 real fetches
python stage4/run.py --arm A --chunk-ids stage4/verify_chunks.txt
python stage4/run.py --arm B --chunk-ids stage4/verify_chunks.txt
python stage4/verify.py stage4/outputs/A_*.jsonl stage4/outputs/B_*.jsonl --expect stage4/verify_chunks.txt

python stage4/run.py --arm A --upload-repo Dolevabudi/knesset-committees-inference   # full
python stage4/run.py --arm B --upload-repo Dolevabudi/knesset-committees-inference
```

Secrets: `HF_INFERENCE_TOKEN`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, and
`HF_WRITE_TOKEN` (mirror only) from the environment, else `stage3/cache/`
(mode 600, git-ignored). Nothing here writes a key to any file.
