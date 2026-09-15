# Stage 4 — inference over the committees chunk corpus

How the two ASR arms are run over `Hadasy/knesset-committees-chunks`, what each
piece of `src/inference/` does, what was verified, and what it costs. Written after the
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
(`src/inference/cache/revision.json`), so a re-upload cannot change the data under a
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
  price Stage 1 used, so Arm A is comparable across corpora. The call passes
  `extra_body={'language': 'he'}` -- see *Language* below.

### Language — forced on both arms

Arm B is always called with `language='he'`. The first full Arm A run let
Whisper auto-detect, and the final validation caught it: 6.7% of A's
hypotheses were not Hebrew at all (Portuguese, Polish, Arabic, Russian) and
207 were empty. The failures sit on the short chunks -- 25% of chunks under
3 s, 0.3% over 20 s -- where a second of audio is not enough to detect the
language. A 1.0 s chunk whose reference is `נכון.` came back as ` نحن`; with
the language forced it is ` נכון`. Scoring that against a Hebrew fine-tune
would charge the general model for language detection, not transcription.

Arm A was therefore re-run over the whole subset with Hebrew forced (about
$6, 2.5 h). Every row now records `language` (`'he'`, or null for the
auto-detect run). The two runs are kept apart: `arm = 'A'` is the forced
run and is what Stage 1 scores; the auto-detect rows are `arm = 'A_auto'`
in the long table and `hypothesis_A_auto` in the wide one, so the
detection-failure rate is itself available as a finding.

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

Inherited from `the VoxKnesset arm-A driver that preceded it (since removed)`:

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

Stage 1's scoring, unchanged: `normalize_he()` from `src/common.py`, then
word- and character-level Levenshtein (rapidfuzz); corpus WER = Σerrors /
Σreference words. Invariants: no errors, unique `chunk_id`, non-empty and Hebrew
hypotheses, reference present, exactly the requested chunk set, and — with two
runs — identical coverage so A vs B is paired.

## Output row

```
chunk_id, arm, provider, model,
speaker_id, session, session_date, knesset, duration_s, quality,
reference, hypothesis,
language       'he' when the model was told the language; null = auto-detect
latency_s      our wall clock: submit → result, incl. queue and polling
exec_s         provider-reported GPU time (RunPod only)
queue_s        provider-reported queue wait (RunPod only)
error, ts, raw (provider extras: job id, worker, word timings)
```

`latency_s` cannot compare the two arms' speed; `exec_s` is the fair number.

## Verification

Ten fixed chunks (`src/inference/verify_chunks.txt`: one per speaker, 5–25 s,
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

**Measured on the live run (2026-09-12).** Two things the 10-chunk sample could
not show:

*Arm B is dispatch-bound, then quota-bound.* Each worker runs a chunk in ~0.6 s
of GPU time, yet a single endpoint fed 80 in-flight jobs kept only 3–7 running
with 70 queued: RunPod's dispatcher hands work to a few workers at a time
regardless of queue depth. A second endpoint (its own dispatcher) helps — but
the account's **worker quota is 10 across all endpoints** (`saveEndpoint`
refuses more), so two endpoints split the same ten GPUs, they do not add any.
Throughput per worker ~7–8× realtime under dispatch overhead → ~75× for ten.

*Arm A is client-bound, not provider-bound.* deepinfra answered a 6 s chunk in
1.45 s at every concurrency tried and never returned a 429. Throughput scaled
with in-flight requests — 8 → 47×, 64 → 128×, 128 → 200× realtime — and the only
failures were local: a socket limit (`ulimit -n 4096`) and two dropped
connections, retried by `--retry-failed`.

*The corpus reader was the third bottleneck.* A 600 MB shard downloads in
~30 s; the request pool drained a shard's rows in ~9 s and then sat idle. Fixed
by prefetching the next two shards on a background thread. A **scattered**
selection (1 h per speaker drawn across all 410 shards) still pulls 26 MB per
audio-minute against 1.1 MB for shard-sequential reading — 24× read
amplification — and three runs each pull every shard. Acceptable for a 230 h
subset; for anything larger, select whole shards or share one download across
arms.

### Stage-1 subset (what was actually run)

Stage 1 needs a per-speaker WER with a tight CI, on every speaker — not every
hour each speaker ever spoke. stage0's power analysis puts the per-speaker CI
half-width near 0.04 WER at 45 min. The subset takes **1 h per MK** (quality
≥ 0.5) drawn round-robin over sessions so it spans the speaker's tenure:
**65,990 chunks, 230 h, 267 speakers, 8,217 sessions**, 238 speakers with ≥ 8
sessions for Stage 2's session-disjoint splits. 6 % of the corpus; the rest is
untouched and selectable later via `coverage.parquet`.

Measured on the subset: Arm A ~170× realtime → ~1.1 h, ~$5; Arm B ~130×
combined → ~1.5 h, ~$15 at $8.70/h. **Both arms in under two hours, under $20.**

*Arm A is client-bound, not provider-bound.* deepinfra answered a 6 s chunk in
1.45 s at every concurrency tried and never returned a 429. Throughput scaled
almost linearly with in-flight requests — 8 → 47×, 64 → 128×, 128 → 142×
realtime — and the first failure at 128 was a local socket limit (`Bad file
descriptor`), fixed with `ulimit -n 4096`. **128 in flight: ~24 h, $94–104.**

### Result (2026-09-13)

The subset is complete on both arms and published as
`Dolevabudi/knesset-committees-inference` (private): 65,990/65,990 chunks with
both hypotheses, 267 speakers, 0 unrecovered errors, all 22 acceptance checks
of `validate_final.py` passed. Corpus WER on the subset, Stage 1's method:
**A 0.417, B 0.324** (B better by 22 %). Empty hypotheses A 0.30 % / B 0;
Hebrew script A 99.7 % / B 99.99 %.

What it actually cost and took, against the estimates above: Arm B ran in
about 4 h of wall time over two RunPod endpoints (10 GPU workers, the account
quota), roughly $12 including idle; Arm A, run twice (auto-detect, then with
Hebrew forced), about $12 on deepinfra. Wall time was dominated not by either
provider but by pulling 410 shards of 770 MB to a laptop: 12 s per shard on a
30 MB/s link, 18 min per shard when the link fell to 0.7 MB/s overnight. The
memory work (`run_lean.py`) was what let the run share a 17 GB machine at all.

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
python src/inference/providers.py                                     # offline self-checks
python src/inference/data.py                                          # build/refresh the index; 3 real fetches
python src/inference/run.py --arm A --chunk-ids src/inference/verify_chunks.txt
python src/inference/run.py --arm B --chunk-ids src/inference/verify_chunks.txt
python src/inference/verify.py src/inference/outputs/A_*.jsonl src/inference/outputs/B_*.jsonl --expect src/inference/verify_chunks.txt

ulimit -n 4096
python src/inference/run_lean.py --workers-a 96 --workers-b 24 --prefetch 2   # the subset, both arms
zsh src/inference/finish.sh          # retry pass, merge, coverage, validate, upload only on pass
```

`run.py --arm A|B` is the per-arm runner with filters; it reads whole shards and
was what the verification and the first third of the subset ran on. `run_lean.py`
is what finished it (see § Result).

Secrets: `HF_INFERENCE_TOKEN`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, and
`HF_WRITE_TOKEN` (mirror only) from the environment, else `the repo-level cache/`
(mode 600, git-ignored). Nothing here writes a key to any file.
