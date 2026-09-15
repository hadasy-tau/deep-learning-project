# Adaptation — the design for stages 6 and 7

How one speaker's audio is turned into an adapter, and how the adapter is judged. This is
the design `src/training/train.py`, `src/evaluation/evaluate.py` and `src/common.py` implement;
the decisions are numbered D1–D7 because the code cites them by those names. It descends from
the Stage 2 plan written against VoxKnesset (`stage2/plan.html`, removed with that path) and is
restated here for the committees corpus, with what the error map (`error_map.md`) changed.

**The question.** Once a population-level Hebrew fine-tune has taken part of the error, is
there anything speaker-specific left — and how many minutes of one person's voice does it take
to get it? Either answer is a contribution: a small personal gain on top of arm B is a real
negative result, and arm A still yields the full recipe curve.

## What the error map changed

| the VoxKnesset plan assumed | measured on the committees (`error_map.md`) | consequence |
|---|---|---|
| the population fine-tune removes ~50 % of error for everyone | it removes a median 24 %, and 11–14 % for the least-helped speakers | there is a residual worth adapting to; the "who gains least" list exists (`committees_adaptation_candidates.csv`) |
| arm B had seen the test speakers *and* the test audio | recordings are held out; voices are not | the "prior exposure" covariate stays; the audio contamination is gone |
| chunking, alignment and a quality score had to be built | the corpus ships ≤ 30 s chunks with `quality` | the chunking stage is gone; the split and budget code remain |
| subgroup rules were a guess | demographic rules explain < 10 % of the variance; speaking rate is the strongest gain rule | the sharing axis starts from the candidate list and acoustic similarity, not from demographics |
| half of B's residual error might be speaker-specific | 53 % of B's errors are insertions against a cleaned protocol | a WER floor no adaptation crosses; the S/D/I split is the detector for style-learning |

## Choosing the panel

**Do not select on noise.** A speaker's measured WER is true WER plus sampling noise. Picking
the worst decile by measured WER selects a mixture of hard speakers and unlucky samples, and the
unlucky ones improve on re-measurement with no fine-tune at all. The variable selected on and
the variable the outcome is measured on must not share noise. The session-disjoint split (D2)
gives this for free: define the split first, rank speakers on their *train-side* sessions, and
leave personal-test untouched until final scoring.

**Entry requirements.** At least 20 chunks in the error map (its reliability floor), at least
3 h of corpus audio at quality ≥ 0.7 and at least 8 sessions, so an 80-minute training budget
plus disjoint dev and test is possible. Personal-test is sized per speaker, not a flat number:
resolution tracks WER, so a low-WER speaker needs far more test audio to resolve the same
relative change (`make_splits` takes `test_min` per speaker for this reason).

**Profiles.** Four kinds of speaker, so the recipe is read across the range rather than at one
point:

| profile | what they test |
|---|---|
| high `wer_B`, low `gain_rel` | the population fine-tune failed them |
| high `wer_B`, normal `gain_rel` | simply hard |
| median `wer_B` | the typical speaker |
| low `wer_B` | headroom — is anything left? |

The error map adds a filter the plan did not have: prefer speakers whose quality-filter
footprint is below the median. The footprint predicts WER (rho 0.6–0.7) and nothing in the
text says whether that is acoustics or labels; the audio gate
(`src/preprocessing/speaker_index/validate_audio.py`) on the panel is the cheapest insurance
before GPU time.

## Decisions

**D1 — Two arms.** A `openai/whisper-large-v3` is the positive control: if the recipe shows
nothing on B, A says whether nothing is left or the pipeline is broken. A's gain is mostly
Hebrew and Knesset-style learning, so only Δpersonal − Δdomain is comparable across arms, and A
runs the reference configuration plus the budget axis only. B `ivrit-ai/whisper-large-v3` is
the target. Note the checkpoints: inference (stage 3) served B as the `turbo-ct2` checkpoint
ivrit.ai recommend; training and local evaluation use the transformers checkpoint. Comparisons
therefore stay inside one engine — A versus B over the providers, base versus tuned locally —
with the scoring shared.

**D2 — Session-disjoint splits, by date.** Personalization needs train and test from the same
person. Each speaker's sessions are ordered by `session_date` (not session id; they diverge),
the latest become personal-test, the next dev (~15 min), the rest train. `common.make_splits`.
Never split randomly within a session: same room, mic and topic. Run the random split once as
a deliberate contrast and report the gap — it is how much of a naive gain is session
memorization.

**D3 — The cross-speaker control is mandatory.** "Train on S, WER drops" conflates speaker
acoustics, protocol-style adaptation, session memorization and topic drift. A budget-matched
2×2 separates them:

| | eval on S | eval on others |
|---|---|---|
| trained on S | Δpersonal | transfer |
| trained on others, same minutes | Δdomain | floor |

personalization effect = Δpersonal − Δdomain. The transfer cell is the same quantity read the
other way: r = Δ(others)/Δ(S) near 1 means a generic adapter (deployable, not personal), near 0
means genuinely personal. Report both gains, not the ratio.

**D4 — Recipe axes, one factor at a time.** Reference configuration: LoRA r = 8, α = 2r,
q/v projections, encoder + decoder, lr 1e-3, 30-minute budget, early stop on personal-dev.

| axis | levels | question |
|---|---|---|
| site | encoder · decoder self-attn · decoder cross-attn · both · all-linear | acoustic or lexical? Self-attention is language modelling, cross-attention the acoustic→text mapping; `SITES` in train.py keeps them separable |
| method | LoRA · DoRA · (IA)³ · full fine-tune | which family suits minutes of one voice; `DEFAULT_LR` gives full FT 1e-5, adapters 1e-3 |
| budget | 1, 2, 5, 10, 20, 40, 80 min, nested | the recipe curve; `take_budget` makes every rung a prefix of one ordering (`budget_order`, latest session first) |
| capacity | rank 1, 4, 8, 16, 32 | rank as a function of budget |
| optimization | lr {3e-4, 1e-3, 3e-3} | a recipe needing per-speaker tuning is not a recipe; report the spread |
| sharing | personal-only · subgroup-only · subgroup-then-personal | does personalization add anything beyond group membership |

The site axis carries a prediction that differs between arms, stated before running: B has
absorbed 4,700 h of plenums, so its residual should be acoustic (encoder-side); A has learned
neither Hebrew formatting nor Knesset vocabulary, so its gain should sit decoder-side. Start the
sharing axis at one speaker and one definition; both are config parameters.

**D5 — Report the cost.** Every adapter is also scored on other Knesset speakers and on an
out-of-domain Hebrew set. Track runaway decodes (`score` flags them). α-scaling at inference
traces the personal-versus-generic frontier from one run for free; KL-to-base is the costlier
alternative and only worth it if α-scaling's frontier is worse.

**D6 — Regularization.** Weight decay, early stopping on dev, checkpoint averaging. Their optimal
strength as a function of minutes is part of the recipe. Augmentation is an ablation, not a
default: it smears the acoustics personalization is trying to learn.

**D7 — Short-form primary, long-form secondary.** Training is on ≤ 30 s chunks and so is the
primary score (`transcribe_short`): it matches the training distribution and gives the paired
bootstrap many items. Long-form on whole personal-test recordings (`transcribe_long`) is
deployment realism; the gap between the two is where hallucination and repetition show.

## The pipeline, against the code

| step | what | where |
|---|---|---|
| select the panel | from the error map, train-side only | `src/evaluation/outputs/committees_adaptation_candidates.csv`; the rules above |
| gate the labels | ECAPA centroid check on the panel | `src/preprocessing/speaker_index/validate_audio.py` (GPU; never run) |
| materialize | pull the panel's chunks at quality ≥ 0.7 from the corpus shards to WAV, with `filename, text, session, session_date, duration_s` | **not written.** `train.py` and `evaluate.py` read `filename, start, end` WAVs; the corpus is FLAC inside parquet (`committees_handoff.md` § the integration gap). The lean runner's shard reader is the extraction half |
| split | session-disjoint by date, test sized per speaker, nested budgets | `common.make_splits`, `common.budget_order` |
| baseline | both base models on personal-test, short- and long-form; error counts stored | `evaluate.load`, `transcribe_short/long`, `score` |
| train | one cell → one adapter, idempotent | `train.train_cell`; `overfit_check` first |
| evaluate | every adapter on personal-test, cross-speaker, out-of-domain; paired bootstrap and Wilcoxon over chunks | `evaluate.score`, `evaluate.paired_bootstrap` |
| statistics | recipe curves, mixed effects across the panel | notebook, to be written |

## Fine-tuning gotchas encoded in `train.py`

- `language='he'`, `task='transcribe'` on the generation config; `forced_decoder_ids` no longer
  exists in transformers 5, and `suppress_tokens = []` keeps the loss off tokens never forced.
- The tokenizer prepends `<|startoftranscript|>` and `shift_tokens_right` prepends
  `decoder_start_token_id`, the same id; `collate` strips one copy or the decoder trains on a
  doubled start token.
- Pad labels are masked with −100; the special-token prefix is identical between train and eval,
  and the timestamp decision (`predict_timestamps`) is held constant across every cell.
- Gradient checkpointing needs `use_cache = False` and, under PEFT, `enable_input_require_grads()`.
- `target_modules` takes a regex, which is the only way to keep decoder self-attention apart from
  cross-attention (`SITES`); without it the site axis cannot be run.
- bf16 where the card has it, fp16 with a loss scaler where it does not; the adapter learning rate
  of 1e-3 is where fp16 trips the scaler.

## Compute

`whisper-large-v3` is 1.55 B parameters. LoRA is comfortable on a 24 GB card with bf16 (L4, A10G)
and feasible on a 16 GB T4 with fp16 and checkpointing (the proof-of-concept path `train.py`
keeps). Full fine-tuning needs 40 GB (A100) with 8-bit Adam and checkpointing; below that the
method axis degenerates into a comparison of LoRA variants and the encoder-versus-decoder
question loses its most informative arm. A wide sweep on `whisper-large-v3-turbo` is about 5×
cheaper and worth it for everything except the site axis, whose decoder conclusions may not
transfer from a four-layer decoder.

## Scope

- **Must land.** Panel of 4–12 speakers · both arms (A on the reference configuration and the
  budget axis) · budget axis × site reduced to {encoder, decoder cross-attn, both, full FT} ·
  cross-speaker control · session-disjoint versus random contrast · short-form primary with
  long-form on personal-test · paired-bootstrap statistics with the S/D/I split.
- **If time.** Rank × budget · the method axis · decoder self-attn as its own level · the
  sharing axis beyond one cell, with acoustic (ECAPA) similarity as the first definition ·
  α-scaling frontier · augmentation and KL ablations · out-of-domain probe.
- **Out.** All 267 speakers · prompt and prefix tuning · LM fusion · speaker-embedding
  conditioning · algorithmic hyperparameter search.

## Verification, before the grid

- Split integrity: zero session overlap between train, dev and test; budgets strict subsets;
  the control pool shares no speaker with the panel (`common.py` self-checks cover the first
  two).
- Baseline reproduction: the base models' short-form WER on each panel speaker's personal-test
  must agree with the error map's `wer_A`/`wer_B` for that speaker within its CI. If it does not,
  materialization or scoring changed the metric and nothing downstream is valid.
- Training sanity: `overfit_check` drives 20 examples to near-zero loss.
- Statistics: paired bootstrap over chunks, plus the S/D/I breakdown. **A gain that comes mostly
  from fewer insertions is protocol style, not the speaker, and does not count.**
- End-to-end smoke: one speaker × one budget × one method, all steps, before committing to the
  grid.

## Risks that outlive the design

- **Non-verbatim references.** The protocol is not what was said; over half of B's residual
  errors are insertions. There is a floor, and part of any gain is style. The cross-speaker
  control and the S/D/I split exist to bound it.
- **Labels.** Verified against the protocol, not the voice. The audio gate has not run.
- **The quality filter is model-derived.** The corpus's `quality` comes from an aligner in the
  Whisper family, so filtering on it can remove exactly the audio the aligner found hard. The
  error map keeps it because below 0.7 the reference demonstrably does not match the audio
  (WER ≈ 1 under both arms). Report personal-test results with and without the filter.
- **The null result.** A small arm-B effect is plausible. The design survives it if the panel
  could have detected an effect, which is what per-speaker test sizing and the bootstrap CI
  establish.
