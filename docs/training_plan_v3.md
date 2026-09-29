# Training plan v3: cleaner labels, tuned hyperparameters, augmentation

Written 2026-09-29, after reading the second run (`docs/training_run2.md`). The project
narrows to three things that decide whether the personalization result can be trusted.
Everything else from `docs/training_next.md` is parked:

- the looping-output decoding fix;
- extra seeds;
- similar-speaker groups;
- the word-for-word test set.

**Rules for the whole plan:**
- **The test set is read once, at the end.** Every choice is made on validation (the `dev` split in the code).
- `results.csv` keeps its format. Tuning results go to `outputs/tuning.csv`.

## 1. Labels: train on clips whose protocol matches the audio

**The problem.** The reference is the edited Knesset protocol. A model trained on it learns to drop the words the stenographer dropped (`training_run2.md` § Results).

**ivrit.ai's advice** (their researchers, 2026-09): train only on segments whose alignment `quality` is high. That is the median of the aligner's per-word probabilities. They use 0.65 as "fit for training" and 0.95 as "high quality". The scale matches ours (0–1, confirmed).

**Measured on the panel** (words both base models heard that the protocol lacks, per 100 protocol words):

| quality | clips | omitted words per 100 |
|---|---|---|
| 0.70–0.80 | 595 | 28.5 |
| 0.80–0.90 | 1,036 | 18.3 |
| 0.90–0.95 | 980 | 13.4 |
| ≥ 0.95 | 2,566 | 6.8 |

**A median can hide a bad stretch.** A clip whose median is 0.95 has, by definition, half its words below 0.95. A few of those can be clearly misaligned (probability near 0), meaning the protocol text doesn't match the audio there, without moving the median. The raw `ivrit-ai/knesset-committees` sessions keep every word's probability (`transcript.refined.json`). So a second, word-level rule is added.

**The rule for train and validation** (`src/training/word_quality.py`, function `word_ok`):
1. the clip's `quality` is ≥ 0.95;
2. at most `MAX_LOW_SHARE` of its words have probability < 0.5, with the value fixed from `word_quality.py --report` before any training;
3. no run of 3 or more such words in a row;
4. the words found in the clip's time span account for its reference text, within 20%.

**The data plan** (`materialize.py plan-v2` → `panel_plan_v2.parquet`):
- **test:** the v1 plan's test clips, unchanged. The cached base transcriptions still apply, and results stay comparable with the second run. Reported two ways: all test clips (quality ≥ 0.7, `results.csv`) and the test clips with quality ≥ 0.95 (`results_q95.csv`, re-scored from the saved transcriptions, no GPU).
- **validation:** the newest non-test sessions, filtered clips only, about 15 minutes. Filtering validation as well means choosing the best checkpoint no longer rewards the stenographer's style. 30843's session 2235355 is excluded, because the audio gate found it is mostly another voice.
- **train:** older sessions, filtered clips only, newest first, at least 80 minutes, so the 5/20/80 budgets still nest.

About half of each speaker's audio passes the filter, so the plan reaches further back than v1. The new clips must be extracted before training (§ Runbook).

## 2. Hyperparameters

**Step mode** (`train_cell(max_steps=…)`, `run_panel.py --max-steps`):

| setting | value | why |
|---|---|---|
| training length | `max_steps = 400` for every budget | Equal optimisation for 5 and 80 minutes; the budgets differ only in how much distinct audio they hold. 400 is about 8 epochs at 80 minutes. |
| validation | loss every 20 steps | An epoch was too coarse: the second run's best epoch was always 1 or 2. |
| early stopping | stop after `patience` validations without improvement; restore the best | `patience` is tuned. |
| schedule | constant after 20 warmup steps | With linear decay the rate depends on `max_steps`, and a run stopped early never reaches its low-rate phase. |

**What is tuned, one setting at a time from the reference** (lr 3e-4, rank 8, dropout 0, no augmentation):

| round | setting | values |
|---|---|---|
| 1 | learning rate | 1e-4, 3e-4, 1e-3 |
| 2 | LoRA rank (alpha = 2 × rank) | 4, 8, 16 |
| 3 | Whisper `dropout` | 0, 0.1 |
| 4 | augmentation (§ 3) | none, +SpecAugment, +tempo, +noise |
| — | patience | 2, 4, 8 validations, computed from the logged curves with no extra runs |

**Why Whisper's `dropout`, and not the other two:**
- `dropout` acts on every block's output, so it regularises everything the adapter's changes flow through.
- `attention_dropout` zeroes attention weights, including the cross-attention between audio and text, which is exactly where the adapters sit.
- `activation_dropout` acts only inside the feed-forward layers, where there are no adapters.
- LoRA's own dropout stays at 0.05.

**Keeping the cost down:**
- Tune at the two ends only, 5 and 80 minutes. If they agree, 20 minutes uses the same value; if they disagree, only that setting is run at 20 minutes.
- Tune on 6 speakers, one or more per profile: 30685 (S1), 23558 (S2), 30718 (S3), 30868 (S3), 30859 (S4), 30752 (C).
- Tuning runs never transcribe the test set (`--tune`).

The total is about 108 short runs, roughly 5 A100 hours.

**How a setting is judged:**
- For each speaker, the relative drop in validation loss from the untuned model to the kept checkpoint (`rel_drop` in `tuning.csv`).
- A setting replaces the reference only if it raises the mean over the 6 speakers by at least 1% **and** helps at least 4 of the 6. Otherwise the simpler setting stays.

## 3. Augmentation

**Principle:** add variety without changing who is speaking.
- Training clips only; validation and test stay clean.
- Random every time a clip is read.
- Seeded from the run's seed.
- The control adapters get exactly the same augmentation.

| order | technique | setting | how |
|---|---|---|---|
| 1 | SpecAugment | time masks: probability 0.05, length 10; frequency masks: probability 0.05, length 10 | Whisper's config (`apply_spec_augment`), active in training mode only |
| 2 | tempo | rate 0.9–1.1 on half the clips, pitch unchanged | `audiomentations.TimeStretch` in `ChunkDataset`. Slowing is capped so no clip passes Whisper's 30 s window. |
| 3 | coloured noise | 10–25 dB signal-to-noise ratio on half the clips | `audiomentations.AddColorNoise` in `ChunkDataset` |
| — | not used: pitch shift, vocal-tract perturbation, resampling "speed" perturbation | | they change the voice |

## 4. The final run

- **Cells:** the chosen recipe at 5, 20 and 80 minutes, all 11 speakers, seed 0.
- **Controls:** at each budget (`--control-folds 2 --control-budgets 5 20 80`).
- **Metrics:** standard and forgiven WER on the full test set (`results.csv`) and on its ≥ 0.95 part (`results_q95.csv`).
- **Personalization** is taken against the control of the same budget and recipe (`run_panel._table`).

## Runbook

```bash
# laptop, no GPU
python src/training/word_quality.py                  # word scores for the candidate clips + the report
#   set MAX_LOW_SHARE in word_quality.py from the report (strictest value leaving every speaker >= 95 min)
python src/training/materialize.py plan-v2           # panel_plan_v2.parquet
python src/training/materialize.py verify --plan src/training/panel_plan_v2.parquet --no-audio

# any machine with a fast link: only the clips not already on disk are pulled
python src/training/materialize.py extract --plan src/training/panel_plan_v2.parquet
python src/training/materialize.py verify  --plan src/training/panel_plan_v2.parquet
python src/training/materialize.py upload  --plan src/training/panel_plan_v2.parquet --repo Dolevabudi/knesset-committees-panel

# GPU: tuning, validation only (round 1 shown; later rounds change one flag)
S="30685 23558 30718 30868 30859 30752"; V2=src/training/panel_plan_v2.parquet
python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0 --lrs 1e-4 3e-4 1e-3
python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0 --lrs <best> --ranks 4 16
python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0 --lrs <best> --ranks <best> --dropouts 0.1
python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0 --lrs <best> --ranks <best> \
    --augments specaug specaug+tempo specaug+tempo+noise
python src/training/run_panel.py --tuning-report     # outputs/tuning.csv + the comparison table, per patience

# GPU: the final run, test read once
python src/training/run_panel.py --plan $V2 --max-steps 400 --patience <best> --budgets 5 20 80 --seeds 0 \
    --lrs <best> --ranks <best> [--dropouts 0.1] [--augments <best>] --control-folds 2 --control-budgets 5 20 80
```
