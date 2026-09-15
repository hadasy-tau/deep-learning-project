# Handoff: inference on the Knesset committees chunk corpus

Read this before touching anything. It is written for a fresh Claude session with no memory of how the corpus was built.

## The goal

Run Whisper inference with **both** models on the committees corpus, then do **per-speaker and per-subgroup evaluation** the way Stage 1 did for VoxKnesset:

- **A**: `openai/whisper-large-v3` (general)
- **B**: `ivrit-ai/whisper-large-v3` (Hebrew fine-tune)

For each speaker and subgroup, measure WER/CER under both models and the adaptation gain (A minus B). Stage 2 later fine-tunes on the poorly served speakers.

**Why committees and not VoxKnesset.** `ivrit-ai/whisper-large-v3` was fine-tuned on `knesset-plenums-whisper-training`, about 4,700 h of plenum audio. VoxKnesset also comes from the plenums, so evaluating model B there measures train-on-test. The committees recordings are not in that training mix, so they are held out for both models. **Do not use VoxKnesset audio or its inference dumps.** That was the user's explicit decision.

One caveat that remains: the *speakers* overlap with the plenum training data. Recordings are held out; voices are not. The Stage 1 "prior exposure" covariate still applies.

## Where everything is

**Git.** Repo `hadasy-tau/deep-learning-project`, branch `main`. The corpus builder is `src/preprocessing/chunk_corpus/`; the speaker index it consumes is `src/preprocessing/speaker_index/`.

**Importing.** Each folder is self-contained and imports its siblings flat, so put the folder itself on the path: `sys.path.insert(0, 'src/preprocessing/chunk_corpus')`, then `import chunks`. Anything shared lives in `src/common.py`, reached by putting `src/` on the path.

(The module used to be `preprocessing.py` inside a folder of the same name, so `import preprocessing` from `src/` bound the folder as an empty namespace package and failed later with an `AttributeError` rather than an `ImportError`. Renaming it to `chunks.py` closed that; `src/common.py`'s self-check asserts it stays closed.)

| path | what |
|---|---|
| `src/preprocessing/chunk_corpus/chunks.py` | the corpus builder. Its module docstring records every measured fact and decision. `SCHEMA` defines the output columns. |
| `src/preprocessing/chunk_corpus/run_build.py`, `src/preprocessing/chunk_corpus/recover_long.py` | how the corpus was built. Do not rerun them. |
| `src/preprocessing/chunk_corpus/outputs/uploaded.json` | manifest: the 10,905 session ids in the corpus |
| `docs/chunk_corpus_build.html` | the build record: design, numbers, failures, open issues |
| `src/preprocessing/chunk_corpus/make_ear_check.py`, `src/preprocessing/chunk_corpus/ear_check_template.html` | the listening-check page generator |
| `src/evaluation/evaluate.py` | **reuse it.** `load(arm)`, `transcribe_short`, `score`, `wer`, `paired_bootstrap` |
| `src/common.py` | **reuse it.** `normalize_he` (Stage 1 Hebrew normalisation, never re-derive it), `make_splits` |
| `notebooks/speaker_error_map.ipynb` | the Stage 1 analysis, as run on VoxKnesset: per-speaker table, gain, subgroups |
| `src/evaluation/error_map.py`, `notebooks/committees_error_map.ipynb` | **that analysis, done on this corpus.** `docs/error_map.md` is the record; the adaptations below are already made there |
| `src/inference/` | the inference pipeline itself, both arms, over this corpus. `docs/inference.md` is its design; `src/inference/README.md` its runbook |
| ~~`stage1/CONTEXT.md`~~ | **gone.** Stage 1's notes on why normalisation and QC are what they are were never committed and are no longer in the working tree. What survives of them is the note above `normalize_he` in `src/common.py` |
| ~~`stage1/inference_openai_whisper_large_v3/full_run.py`~~ | removed with the VoxKnesset path. `src/inference/run.py` inherits its design and says so |
| `src/preprocessing/speaker_index/` (on `main`) | how speaker identities were resolved. `docs/speaker_index_plan.md` explains it. |

**Published pages.**
- Build record: https://claude.ai/code/artifact/823fabc9-4002-4e06-9002-24e0d40dc5fb
- Ear check (35 clips, shared verdicts): https://claude.ai/code/artifact/a775c7c4-1fa5-4d45-8889-f1449dad971b

**HuggingFace.**
- **`Hadasy/knesset-committees-chunks`** is the corpus. It is **private**, so every read needs `huggingface_hub.get_token()` or `HF_TOKEN`.
- `Dolevabudi/knesset-committees-speakers` is the speaker index it was built from, pinned to revision `56b19714`.
- `ivrit-ai/knesset-committees` is the source audio. You should not need it.

## The corpus

- **1,202,046 chunks, 410 shards** (`data/chunks-00000.parquet` … `data/chunks-00409.parquet`), 274.5 GB
- **330 speakers, 10,905 sessions, ~4,111 h of audio.** Every chunk is ≤30 s and holds **exactly one speaker**.
- Each chunk tiles a speaker's turn back to back and never crosses a speaker change (ivrit.ai's slicer with one extra rule).

**Columns** (`SCHEMA` in `src/preprocessing/chunk_corpus/chunks.py`):

```
audio            struct {bytes: FLAC 16 kHz mono int16, path: chunk_id}
chunk_id         {speaker_id}_{session}_{start_ms}_{end_ms}.flac
speaker_id  speaker_name  session  session_date  knesset  committee_name
seek  duration_s  abs_start  abs_end
text             raw protocol text: the reference, NOT normalised
transcript       ivrit.ai training format with <|0.00|> timestamp tokens
prev_transcript  has_prev  has_timestamps
quality          alignment confidence (median word probability), 0..1
n_words  n_segments
gender  age  date_of_birth  place_of_birth  year_of_aliya  religion  nationality  religious_orientation
```

**Reading it.**
- There is no dataset card or declared `Audio` feature. The `audio` column is a plain struct of FLAC bytes. Decode each chunk with `soundfile.read(io.BytesIO(row['audio']['bytes']), dtype='int16')`, or reuse `decode_flac` in `src/preprocessing/chunk_corpus/chunks.py`. It is lossless FLAC by design, so there is no codec confound in WER.
- Shards are about 500 MB to 1.3 GB each. Read them one at a time with `columns=[...]` projection, and never all 274 GB at once. To check audio lengths, use `duration_s`; don't decode.

**Quality.** Nothing was filtered at build time. The user wanted a general-purpose corpus, so any filter is the consumer's decision.
- 11.3% of assigned audio has segment quality below 0.7.
- Turns with median quality ≥ 0.7 come to 3,483 h, 324 speakers and 8,853 sessions.
- `quality ≈ 0` usually means the protocol text doesn't match the audio. One pilot chunk had 14.7 s of audio against about 350 words. Filter on `quality`, and consider a words-per-second bound too.

## The integration gap you must close first

`src/evaluation/evaluate.py::transcribe_short(model, proc, chunks, audio_dir, ...)` reads audio through `read_wav(os.path.join(audio_dir, r.filename), r.start, r.end)`, which expects **WAV files on disk**. This corpus is **FLAC bytes inside parquet**, so `transcribe_short` cannot consume it unchanged.

Write a bytes-based twin: decode FLAC → float32 in [-1, 1] → `proc.feature_extractor(..., sampling_rate=16000)` → `model.generate(..., language='he', task='transcribe')`. Keep everything else in `evaluate.py` identical, especially `score()`, which stores error **counts**, never rates. Don't materialise WAV files, since that would mean a 4,111 h extraction.

`evaluate.py`'s `ARMS` is already `{'A': 'openai/whisper-large-v3', 'B': 'ivrit-ai/whisper-large-v3'}` (transformers checkpoints, fp16 on CUDA).

## Decisions that were open here, and how they were settled

All four were live questions when this was written. `src/inference/` answered them; `docs/inference.md` carries the reasoning and the measured costs.

1. **Inference stack.** Not one stack after all. Arm A goes through HF Inference Providers (deepinfra) — the same provider and price Stage 1 used, so A is comparable across corpora — and arm B through RunPod serverless on ivrit.ai's own worker image, running `ivrit-ai/whisper-large-v3-turbo-ct2`, the checkpoint they recommend for inference.
2. **How much of the corpus.** A 1 h/speaker subset, pinned in `src/inference/subset_stage1.parquet` (65,990 chunks), rather than all 3,840 h.
3. **Quality floor.** ≥ 0.7, as recommended.
4. **Where to run.** Neither arm needs a local GPU, so the MX230 stopped mattering. `src/inference/finish.sh` drives the run from a laptop.

## Adapting the Stage 1 analysis (`notebooks/speaker_error_map.ipynb`)

It inner-merges two frames on `filename`: an A-side with `filename, speaker_id, split, audio_seconds, reference, transcription, error, age, gender, speaker_place_of_birth, speaker_year_of_aliya, speaker_religion, speaker_nationality, speaker_religious_orientation`, and a B-side with `filename, model_transcription`.

Mismatches to handle in an adapter, not by editing the corpus:
- `chunk_id` → `filename`, `text` → `reference`, `duration_s` → `audio_seconds`
- Demographics have no `speaker_` prefix here. Map `place_of_birth` → `speaker_place_of_birth`, and so on.
- `gender` is 0/1 Int64. Religion, nationality and orientation are raw Hebrew strings mapped by the notebook's dicts.
- The notebook's QC cell (`unexplained_s = duration_s - n_words*1.5 > 300`) **can never fire on ≤30 s chunks.** Replace it with a `quality` filter.
- The notebook's `ORIGIN` dict is missing three birthplaces in this corpus: `אוסטריה`, `דנמרק`, `האימפריה האוסטרו-הונגרית` (one speaker each). Without them those speakers silently become `Unknown`.
- Keep the Stage 1 normalisation exactly. Geresh and gershayim map to a **space**, not nothing — worth ~5.7% relative WER on B. The note above `normalize_he` in `src/common.py` records why.
- There is no `split` column. The corpus is unsplit. Use `src/common.py::make_splits`, which is session-disjoint per speaker, if needed, and sort by `session_date`, not session id (they diverge per speaker).

## Open problems to keep in mind, not fix

- **Speaker labels are verified against the protocol, not the voice.** The ECAPA speaker-embedding check (`src/preprocessing/speaker_index/validate_audio.py`, Step 4.2 of `docs/speaker_index_plan.md`) has never been run and needs a GPU. Committee cross-talk is heavy; the protocol records who held the floor, not who was audible. If per-speaker WER looks anomalous for one speaker, suspect labels before suspecting the model.
- **The ear check was spot-checked, not swept.** A time-origin offset between audio and alignment would give fluent but wrong-text chunks for a whole session. It would show up as that session's WER near 1.0 under **both** models.

## Conventions in this repo

- Plain functions driven from notebooks. Every module ends with `# ---- self-checks ---` and `if __name__ == '__main__':` hard asserts on real measured values.
- Comments explain *why* a number is what it is, usually with the measurement behind it.
- Store error **counts**, aggregate micro (Σerrors / Σwords). Never average per-chunk WER, since chunks are short and that is badly biased.
- Pin HuggingFace revisions for anything that runs long. The speaker index changed mid-session once and withdrew 9 speakers.
- Read tokens with `huggingface_hub.get_token()`. Never print or embed them.
- Streaming or long jobs: resumable per unit, per-unit errors recorded and not fatal, **retries with jittered backoff on both download and upload** (a HuggingFace 503 once killed a 29-hour job).
- The user decides commits, pushes and PRs. Ask before doing any of them. `gh` isn't installed.

## What not to do

- Don't rebuild or modify the corpus. It is done and verified.
- Don't read `src/preprocessing/chunk_corpus/ear_check.html`. It is ~7 MB of base64 audio, gitignored, and will flood the context.
- Don't read the build logs under the session temp directory. They're huge and irrelevant now.
- Don't use VoxKnesset.
