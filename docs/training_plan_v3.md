# Training plan v3: high-quality data, lean hyperparameter tuning, augmentation

Written 2026-09-29 and revised 2026-10-01, after the second run (`docs/training_run2.md`).
The project narrows to three things that decide whether the personalization result can be
trusted. Parked from `docs/training_next.md`:

- the looping-output decoding fix;
- extra seeds;
- similar-speaker groups;
- the word-for-word test set.

**Rules for the whole plan:**
- **High-quality data only**, for test, validation and train alike (§ 1).
- **The test set is read once, at the end.** Every choice is made on validation (the `dev` split in the code).
- `results.csv` keeps its format. New rows carry `hq` in their cell name. Tuning results go to `outputs/tuning.csv`.

## 1. Labels: only clips whose protocol matches the audio

**The problem.** The reference is the edited Knesset protocol. A model trained on it learns to drop the words the stenographer dropped (`training_run2.md` § Results).

**ivrit.ai's advice** (their researchers, 2026-09): use only segments with a high alignment `quality`. That is the median of the aligner's per-word probabilities. They use 0.65 as "fit for training" and 0.95 as "high quality". The scale matches ours (0–1, confirmed).

**Measured on the panel** (words both base models heard that the protocol lacks, per 100 protocol words):

| quality | clips | omitted words per 100 |
|---|---|---|
| 0.70–0.80 | 595 | 28.5 |
| 0.80–0.90 | 1,036 | 18.3 |
| 0.90–0.95 | 980 | 13.4 |
| ≥ 0.95 | 2,566 | 6.8 |

**A median can hide a bad stretch.** A clip whose median is 0.95 has, by definition, half its words below 0.95. A few of those can be clearly misaligned (probability near 0), meaning the protocol text doesn't match the audio there, without moving the median. The raw `ivrit-ai/knesset-committees` sessions keep every word's probability (`transcript.refined.json`), so a word-level rule is added.

**The rule, for every split** (`src/training/word_quality.py`, function `word_ok`):
1. the clip's `quality` is ≥ 0.95;
2. at most 10% of its words have probability < 0.5. Fixed from `word_quality.py --report`: the strictest value that leaves every speaker 140 minutes. At 5%, the smallest speaker falls short.
3. no run of 3 or more such words in a row;
4. the words found in the clip's time span account for its reference text, within 20%.

**The data plan** (`materialize.py plan-v2` → `panel_plan_v2.parquet`), per speaker and session-disjoint by date:
- **test:** the newest sessions, filtered clips only, until 45 minutes (v1's floor);
- **validation:** the next sessions, until 15 minutes;
- **train:** older sessions, newest first, at least 80 minutes, so the 5/20/80 budgets nest.

**Panel change.** 30843 has only about 124 high-quality minutes, short of the 140 needed. Her S2 alternate, **556 חיים כץ**, takes her place (`word_quality.PANEL_SWAPS`). His caveat is 15 h of plenum exposure, the reason he was an alternate. Her validation session 2235355, flagged by the audio gate, stays excluded in case she returns.

**Consequences:**
- The test clips differ from the second run's, so results aren't directly comparable with it.
- The base models transcribe the new test sets once. That's cached as `base_*_hq.json`, beside the second run's caches, never over them.
- Most of the plan's clips are new and must be extracted first (§ Runbook).

## 2. Hyperparameters: lean tuning on validation

**Step mode** (`train_cell(max_steps=…)`, `run_panel.py --max-steps`):

| setting | value | why |
|---|---|---|
| training length | at most `max_steps = 400` for every budget | Equal optimisation room for 5 and 80 minutes. |
| validation | loss every 20 steps | An epoch was too coarse: the second run's best epoch was always 1 or 2. |
| early stopping | **fixed**: stop after 4 validations (80 steps) without improvement, restore the best | Not tuned, to keep the number of runs down. Runs typically stop near step 150–200. |
| schedule | constant after 20 warmup steps | With linear decay the rate depends on `max_steps`, and a run stopped early never reaches its low-rate phase. |

**What is tuned, in two stages:**

| stage | settings | values | combinations |
|---|---|---|---|
| A | learning rate (rank 8, dropout 0) | 1e-4, 3e-4, 1e-3 | 3 |
| B | LoRA rank × Whisper `dropout`, at stage A's rate | {8, 16} × {0, 0.1} | 4, one already run in A |
| C | augmentation (§ 3), at the A + B winner | SpecAugment; SpecAugment + tempo | 2 |

Rate is tuned first because it mattered most in the second run. With alpha = 2 × rank, the adapter's scale doesn't change with rank, so rate and rank interact weakly.

**Why Whisper's `dropout`, and not the other two:**
- `dropout` acts on every block's output, so it regularises everything the adapter's changes flow through.
- `attention_dropout` zeroes attention weights, including the cross-attention between audio and text, which is exactly where the adapters sit.
- `activation_dropout` acts only inside the feed-forward layers, where there are no adapters.
- LoRA's own dropout stays at 0.05.

**Who and where:**
- **4 tuning speakers,** one per profile: 30685 (S1), 23558 (S2), 30718 (S3), 30859 (S4).
- **2 budgets,** the ends: 5 and 80 minutes. If they pick the same value, 20 minutes uses it too. If they disagree, 20 minutes takes the value of the nearer end (80), and the disagreement is reported.
- `--tune` runs never transcribe the test set.

**How a setting is judged:**
- For each speaker, the relative drop in validation loss from the untuned model to the best checkpoint (`rel_drop` in `tuning.csv`).
- A setting replaces the current best only if it raises the mean over the 4 speakers by at least 1% **and** helps at least 3 of the 4. Otherwise the simpler setting stays.

## 3. Augmentation

**Principle:** add variety without changing who is speaking.
- Training clips only; validation and test stay clean.
- Random every time a clip is read.
- Seeded from the run's seed.
- The control adapters get exactly the same augmentation.

| variant | what | how |
|---|---|---|
| SpecAugment | hide short random stretches of time (probability 0.05, length 10) and frequency (probability 0.05, length 10) in the model's input | Whisper's config (`apply_spec_augment`), active in training mode only |
| + tempo | rate 0.9–1.1 on half the clips, pitch unchanged | `audiomentations.TimeStretch` in `ChunkDataset`. Slowing is capped so no clip passes Whisper's 30 s window. |
| (+ noise, implemented, not in the plan) | coloured noise at a 10–25 dB signal-to-noise ratio | `--augments specaug+tempo+noise`. Left out because committee audio is already noisy, and it saves runs. |
| not used | pitch shift, vocal-tract perturbation, resampling "speed" perturbation | they change the voice |

## 4. The final run, and where the random speaker group comes in

- **Cells:** the chosen recipe at 5, 20 and 80 minutes, all 11 speakers, seed 0.
- **The control:** a random group of the *other* panel speakers, in 2 folds, trained on the same number of minutes and evaluated on every speaker it never heard. It runs at **each** budget (`--control-folds 2 --control-budgets 5 20 80`). Personalization = the own adapter's gain − the control's gain, on the same test clips (`run_panel._table`). This is the comparison that tells "learned this voice" apart from "learned committee Hebrew".
- **Not in this plan:** comparing *similar*-speaker groups against random groups of the same size (`training_next.md`, "sharing"). It can follow on the chosen recipe: about 44 short runs.
- **Metrics:** standard WER (and CER) on the high-quality test set, with the error-type split and the style flag. Forgiven-shared WER is off (`--forgiven` turns it back on). On high-quality clips the protocol matches the audio, which is what it was a workaround for.

## 5. What every run collects, for the analysis notebook

Nothing is computed only for display. Every number a plot could need is written to disk as the run goes, and `backup.py` mirrors it to HuggingFace every 30 minutes.

| file | one per | holds |
|---|---|---|
| `outputs/results.csv` (from `outputs/results/<cell>.json`) | scored cell or control evaluation | the settings, train minutes, steps and best checkpoint, test size, base and tuned WER and CER, the gain with its 95% interval and p-values, the substitution/deletion/insertion shares and looping outputs, the style flag, the control's gain and **personalization** = own gain − control's gain |
| `outputs/results/<cell>.hyps.json`, `base_*_hq.json` | cell; speaker | every test clip's transcription by the tuned and the base model, for re-scoring and per-clip analysis without a GPU |
| `runs/<cell>/train_meta.json`, `runs_tune/<cell>/train_meta.json` | trained adapter (final run and tuning) | `settings` (speaker, budget, lr, rank, alpha, LoRA dropout, Whisper dropout, augmentation, batch, schedule, seed); `train_log`: the **training loss**, gradient norm and learning rate every 5 steps; `evals`: the **validation loss** every 20 steps; `base_eval_loss` (the untuned model's); `best_step`, `global_step`, `stopped_early`; `trainable_params`, `train_runtime_s`, `peak_gpu_mem_gb`; the train and validation clip IDs |
| `outputs/tuning.csv` | tuning run | the settings, untuned and best validation loss, `rel_drop` (the tuning criterion), best and stop steps |

## Cost (A100 at $1.9–2.7/h; about 3 minutes per early-stopped run)

| step | runs | GPU time |
|---|---|---|
| A. learning rate: 3 × 2 budgets × 4 speakers | 24 | about 1.2 h |
| B. rank × dropout: 3 new × 2 × 4 | 24 | about 1.2 h |
| C. augmentation: 2 × 2 × 4 | 16 | about 0.8 h |
| final: 33 cells + 6 control trainings + 33 control evaluations + base transcriptions | — | about 2–2.5 h |
| **total** | | **about 5–6 h, $10–16** |

## Runbook

```bash
# laptop, no GPU
python src/training/word_quality.py                  # word scores for the candidate clips + the report
python src/training/materialize.py plan-v2           # panel_plan_v2.parquet
python src/training/materialize.py verify --plan src/training/panel_plan_v2.parquet --no-audio

# any machine with a fast link (no GPU): only clips not already on disk are pulled
python src/training/materialize.py extract --plan src/training/panel_plan_v2.parquet
python src/training/materialize.py verify  --plan src/training/panel_plan_v2.parquet
python src/training/materialize.py upload  --plan src/training/panel_plan_v2.parquet --repo Dolevabudi/knesset-committees-panel

# GPU: tuning, validation only
S="30685 23558 30718 30859"; V2=src/training/panel_plan_v2.parquet
T="python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0"
$T --lrs 1e-4 3e-4 1e-3                                          # A
$T --lrs <A> --ranks 8 16 --dropouts 0 0.1                       # B (the rank-8, dropout-0 cell is skipped: already run)
$T --lrs <A> --ranks <B> --dropouts <B> --augments specaug specaug+tempo     # C
python src/training/run_panel.py --tuning-report                 # outputs/tuning.csv + the comparison table

# GPU: the final run, test read once
python src/training/run_panel.py --plan $V2 --max-steps 400 --budgets 5 20 80 --seeds 0 \
    --lrs <A> --ranks <B> --dropouts <B> --augments <C or none> --control-folds 2 --control-budgets 5 20 80
```

## Handoff: continuing from here (written 2026-10-01)

For whoever runs the training next, a person or a fresh Claude session on a GPU box. Read § 1–5 above first: they carry the decisions. This section is only the *how*, in order.

### Where things stand
- **Code:** PR [hadasy-tau/deep-learning-project#23](https://github.com/hadasy-tau/deep-learning-project/pull/23) (`training-plan-v3` → `main`). Train from `main` once it's merged, otherwise from the branch. It includes Dolev's second run.
- **Data plan:** `src/training/panel_plan_v2.parquet` (6,462 clips, 27.9 h, 11 speakers), built and verified. It's committed, along with `word_quality.parquet`.
- **Audio:** extracted on Hadas's laptop on 2026-10-01 into `src/training/outputs/panel_audio/<speaker>/<clip>.wav` (git-ignored): all 6,462 WAVs, 3.1 GB, 46 minutes for 186 shards at about 80 Mbit/s. `verify` passed every check, including 200 sampled WAVs with the right duration and non-silent audio. Nothing has been trained on it yet.
- **Access:** `Dolevabudi/knesset-committees-panel` and `Dolevabudi/knesset-committees-adapters` are Dolev's private datasets and aren't readable from Hadas's HF account. The corpus `Hadasy/knesset-committees-chunks` is Hadas's.

### 1. Get the audio onto the GPU box, one of two ways
- **(a) From a private HF dataset**, if the laptop's WAVs were uploaded (3.1 GB):
  ```bash
  python src/training/materialize.py download --repo <that dataset>      # fetches panel_audio/ and panel_plan*.parquet
  ```
  The upload, from the laptop: `python src/training/materialize.py upload --plan src/training/panel_plan_v2.parquet --repo <dataset>`.
- **(b) Extract on the box itself.** It needs read access to `Hadasy/knesset-committees-chunks`. 186 shards pass through, about 125 GB; each is deleted after use, and the run is resumable. That's under an hour on a datacenter link, CPU only.
  ```bash
  python src/training/materialize.py extract --plan src/training/panel_plan_v2.parquet --prefetch 3
  ```

Either way, it must pass before anything else:
```bash
python src/training/materialize.py verify --plan src/training/panel_plan_v2.parquet    # splits, filters, every WAV readable
```

### 2. Set up the box
- **GPU:** one A100 (40 or 80 GB). LoRA also fits a 24 GB card, about 2× slower.
- **Disk:** about 10 GB for audio, adapters and model weights.
```bash
git clone https://github.com/hadasy-tau/deep-learning-project.git && cd deep-learning-project   # (git checkout training-plan-v3 if not merged)
pip install -r src/training/requirements.txt         # torch: the box's CUDA wheel; audiomentations is new in v3
hf auth login                                        # typed interactively; never paste a token into a chat or a file
source src/training/box/env.sh                       # RunPod: caches on the container disk, BLAS thread cap
# a pod can be preempted: mirror results every 30 min to a private dataset YOU can write to
nohup python src/training/backup.py --repo <your-private-dataset> --every 30 > backup.log 2>&1 &
```

### 3. Two sanity checks (about 15 minutes)
```bash
python - <<'EOF'                                     # the loop learns: 20 clips driven to near-zero loss
import sys, pandas as pd; sys.path.insert(0, 'src/training'); import train
P = pd.read_parquet('src/training/panel_plan_v2.parquet')
train.overfit_check(P, 'src/training/outputs/panel_audio', speaker=30685, arm='B')
EOF
V2=src/training/panel_plan_v2.parquet
python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers 30685 --budgets 80 --seeds 0 --lrs 3e-4
python src/training/run_panel.py --tuning-report
```
- **First check:** the overfit check's loss must fall steeply. If it doesn't, stop.
- **Second check:** the single tuning run should take about 2–4 minutes, and its `rel_drop` should be positive. Multiply its time by about 70 to price the whole plan, and tell Hadas before spending it.

### 4. Tuning: validation only, three stages
```bash
S="30685 23558 30718 30859"
T="python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0 --eval-batch 64"
$T --lrs 1e-4 3e-4 1e-3                                          # A: 24 runs
python src/training/run_panel.py --tuning-report                 # pick A by the rule below
$T --lrs <A> --ranks 8 16 --dropouts 0 0.1                       # B: 24 runs (rank 8 / dropout 0 is skipped: run in A)
python src/training/run_panel.py --tuning-report
$T --lrs <A> --ranks <B> --dropouts <B> --augments specaug specaug+tempo     # C: 16 runs
python src/training/run_panel.py --tuning-report
```
- **How to choose** (§ 2), separately for each budget, from `tuning.csv`: a setting replaces the current best only if it raises the mean `rel_drop` over the 4 speakers by at least 1% (absolute) **and** helps at least 3 of the 4. Otherwise keep the simpler setting: lower rank, no dropout, no augmentation.
- **If 5 and 80 minutes disagree,** 20 minutes takes 80's value. Write the disagreement down; it's a finding.
- **Never** look at test WER while choosing. `--tune` doesn't compute it.

### 5. The final run: test set used once
```bash
python src/training/run_panel.py --plan $V2 --max-steps 400 --budgets 5 20 80 --seeds 0 --eval-batch 64 \
    --lrs <A> --ranks <B> --dropouts <B> --augments <C or none> --control-folds 2 --control-budgets 5 20 80
python src/training/run_panel.py --summary                       # outputs/results.csv
```
- **Time:** about 2–2.5 hours. It's idempotent, so kill and rerun it freely.
- **If 5 and 80 minutes chose different settings,** run the command once per budget group with that group's flags, adding `--control-only` for the second group's controls only where needed. Each recipe gets its own controls through its cell name.
- **Base transcriptions** of the new high-quality test set are made once per speaker and cached as `outputs/results/base_B_<speaker>_…_hq.json`, next to the second run's caches, never over them.

### 6. Reading the results
- **New rows** are the ones with `hq` in `cell`.
- **The headline column** is `personalization_rel`, the own adapter's gain minus the control's gain at the same budget. A gain whose interval (`ci_lo`, `ci_hi`) crosses zero isn't a result.
- **`style_not_speaker`** marks gains that are mostly fewer insertions. On high-quality clips these should be rare. If they aren't, say so.
- **Base-WER sanity check:** the base WER on the high-quality test set should come out *below* each speaker's `wer_B` in `src/evaluation/outputs/committees_speaker_performance.csv`, because these clips are cleaner. If it comes out more than 0.10 *above*, materialization or scoring is broken: stop.

### 7. What to bring back
The backup loop already mirrors everything to your private dataset:
- `outputs/results.csv`, `outputs/results/*.json` and `*.hyps.json`;
- `outputs/tuning.csv`;
- `runs/` (adapters + `train_meta.json`) and `runs_tune/` (`train_meta.json`);
- the run logs, passed with `backup.py --logs`.

§ 5 describes what each file holds, for the analysis notebook.

### Gotchas
- **On a Linux box,** DataLoader workers are forked and nothing special is needed. On Windows, `train_cell(num_workers=0)` is required: the collator is a lambda, and spawned workers re-import the script.
- **`--dropouts 0` names the cell the same as no dropout flag**, on purpose. That's how stage B skips the stage-A cell.
- **The second run's rows in `results.csv`** are a different data plan and test set; don't pool them with the `hq` rows.
