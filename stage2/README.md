# Stage 2 — personalizing Hebrew ASR to one speaker

Three files. Plain functions, no CLI, no config system. Drive it from a notebook.
Repo-wide layout and data sources are in the [root README](../README.md).

| File | Plan stage | Needs |
|---|---|---|
| `pipeline.py` | 01 index · 02 materialize · 03 chunk · 04 split | nothing (chunking is text-only); HF token for `materialize` |
| `train.py` | 06 train | GPU |
| `evaluate.py` | 05 baseline · 07 eval · 08 stats | GPU for transcription, nothing for scoring |

`../stage0/` holds Step 0: the filename/session checks and `stage0_gate`.

## What Step 0 changed

Three findings shape this code, all in `../stage0/`:

1. **The session id is in the filename** — `speaker_session_start_end.wav`. Splits are
   genuinely session-disjoint; no `age` proxy, no year-disjoint fallback.
2. **VoxKnesset is needed for waveforms only.** The public ct2 dump carries
   `reference_text`, `segments_json` and every demographic column, so there is no
   275 GB column-projection scan — `load_index()` is a `read_parquet`.
3. **Personal-test is sized per speaker, not a flat 45 min.** 40 min at WER .19 but
   ~7 h at WER .06 for the same 10% relative resolution. See `stage0_gate_curve.csv`.

## Order

```python
import pipeline as P, train, evaluate as E

idx = P.load_index()                                  # 66,670 recordings
panel = idx[idx.speaker_id.isin([11835, 528, 1057, 4416])]
ch = P.chunk_all(panel)                               # <=30 s chunks, text-only

TEST = {11835: 40, 528: 108, 1057: 366, 4416: 432}    # minutes, from stage0_gate
sp = P.make_splits(ch, test_min=TEST)

P.materialize(P.files_needed(sp), 'audio/')           # ~2.6 GB, needs HF_TOKEN
```

`files_needed()` takes dev + test + the largest D4 budget rung only. Materializing by
split rather than by speaker is 2.6 GB instead of 15.7 GB.

Then, on a GPU box:

```python
train.overfit_check(sp, 'audio/', speaker=11835)      # must reach ~0 loss first
train.train_cell(sp, 'audio/', 'runs/', speaker=11835, arm='B',
                 site='both', method='lora', budget=30, rank=8, lr=1e-3)
```

## Before believing any result

`evaluate.like_for_like()` is a hard gate. Stage 1 measured **long-form** WER over whole
recordings, so personal-test must be re-scored long-form with the base model and reproduce
Stage 1 *on those same recordings*. Comparing Stage-1 long-form against Stage-2 chunk-level
WER would fail for reasons that have nothing to do with a bug.

The target is not the speaker's published `wer_B` — that covers all their recordings, and
personal-test is only the latest sessions, so the two differ on set membership alone.
`stage1_wer(index, filenames)` recomputes it over exactly the recordings being re-scored,
off the dump's `model_transcription` column (which *is* Stage 1's arm-B output):

```python
test_files = sp[sp.part == 'test'].filename.unique()
hyps = E.transcribe_long(model, proc, test_files, 'audio/')
refs = idx.set_index('filename').reference_text.reindex(test_files)
E.like_for_like(E.score(refs, hyps), idx, test_files)
```

What is left between the two numbers is runtime only — Stage 1 decoded through CTranslate2,
this decodes through transformers — which is what `tol` absorbs.

## Deviations from the plan, and why

- **`forced_decoder_ids` is gone.** transformers 5 dropped it from `GenerationConfig`;
  `generate(language='he', task='transcribe')` is the mechanism now.
- **No `soundfile`/`librosa`.** The audio is 16 kHz 16-bit mono WAV, which stdlib `wave`
  reads in three lines.
- **No `cells.csv` manifest.** It exists in the plan only to drive several GPUs in
  parallel. Add it when there are several GPUs.
- **Chunking cuts inside hypothesis segments.** 22% of them exceed 30 s (max 112 s), so
  reference words get interpolated timestamps and cuts land every 30 s. One code path
  instead of pack-vs-split.

## Deliberate simplifications

- Word times are linearly interpolated inside hypothesis segments (no word-level
  timestamps in the dump). Cuts land within a word or two. Upgrade to `stable-ts` only if
  chunk audio turns out audibly clipped — measured alignment is 0.923 median.
- `dev` takes whole sessions, so it overshoots when a speaker's first spare session is
  large (speaker 11835 gets 39 min, not 15).
- Alignment score is computed and stored but **never filtered on** — filtering it would
  delete exactly the hard cases and let arm B curate the test set both arms are judged on.
  `wpm` (30–350) is model-independent, so that filter stays.

## Self-checks

```bash
python pipeline.py     # index, chunk, split; asserts <=30 s, no overlap, text conserved
python evaluate.py     # score() and paired_bootstrap() on synthetic cases
```
