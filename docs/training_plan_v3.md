# Training plan v3: high-quality data, lean hyperparameter tuning, augmentation

Written 2026-09-29 and revised 2026-10-01, after the second run (`docs/training_run2.md`).
**Run on 2026-10-02: `docs/training_run3.md` has the results and conclusions.**
The project narrows to three things that decide whether the personalization result can be
trusted. Parked from `docs/training_next.md`:

- similar-speaker groups;
- the word-for-word test set.

**Rules for the whole plan:**
- **High-quality data only** for training and validation (§ 1). The test is scored twice: on the high-quality clips, and on every clip of the same test sessions at quality ≥ 0.7 (§ 4, *Two test sets*).
- **The test set is read once, at the end.** Every choice is made on validation (the `dev` split in the code).
- **Every transcription goes through the loop guard** (added 2026-10-02, `evaluate.py`): a hypothesis whose text compresses more than 3.0 (zlib, Whisper's own test, with the threshold measured on the second run's 58,720 hypotheses) is decoded again with no 6-token n-gram repeated, and the new one is kept if it compresses less. Identical for base, own adapter and control; caches record their decoding (`decode`), so a pre-guard cache is redone rather than mixed in. First parked, then brought in: in the second run one looping chunk (~200 errors) decided whole cells.
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

**Panel change, revised 2026-10-02: 12 speakers** (`word_quality.PANEL_ADD`).
- **30843 stays.** The first version swapped 30843 out: about 120 high-quality minutes, short of the 140 needed. But 30843 is one of the three speakers where the second run found a personal effect, and the hard speakers are the project's question. So 30843 keeps the 80-minute budget and gives up test and dev minutes instead: **30 min test, 10 min dev** (`materialize.SPLIT_MIN`). The validation session 2235355, flagged by the audio gate, stays excluded.
  - The ≥ 0.7 test (§ 4) gives 30843 91 minutes.
- **556 חיים כץ stays too,** as an extra S2 speaker. The caveat is 15 h of plenum exposure, the reason he was an alternate.
- **30601 ינון אזולאי is deferred** (decided 2026-10-02, after being added the same day). "Hard and left behind", from outside the original panel: WER_B 0.455, higher than anyone on the panel, and the fine-tune removes only 18% of arm A's error; 44 corpus hours, 146 minutes passing the word rule. He is left for a later run, if one is needed:
  - his quality-filter footprint is high (21% of the audio below 0.7, against 7–14% for most of the panel), so he would need the audio gate first (`validate_audio.py --speakers 30601 --per-speaker 40 --others 60`);
  - his plenum exposure has not been measured.
  - To bring him back: add him to `word_quality.PANEL_ADD`, rebuild `plan-v2` and `plan-test07`, extract. His word scores are kept in `word_quality.parquet`.
- **The other speakers' data is unchanged,** row for row, in `panel_plan_v2.parquet` and `panel_test07.parquet`: both plans are rebuilt per speaker, and the rebuild without 30601 equals the 13-speaker plans with his rows removed, in every row and column (checked 2026-10-02). `word_quality.parquet` gained rows for 30601 (kept) and 82 chunks from 30843's four newest sessions that had never been scored.
- **The panel, 12 speakers:** S1 30831, 30685, 30701 (the fine-tune helped least); S2 23558, 30843, 556 (hard under both models); S3 30813, 30718, 30868 (typical difficulty, gain below the median); S4 30859 (low WER: is any headroom left?); C 30752, 30777 (high-gain controls). Plan v2: 6,887 clips, 30.0 h.

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

- **Cells:** the chosen recipe at 5, 20 and 80 minutes, all 12 speakers, seed 0.
- **Seeds 1 and 2 at 80 minutes** (added 2026-10-02), for the own adapters and their controls alike. The second run's personal effect was significant for only 1–4 of 11 speakers per recipe, the size seed-to-seed variation can reach; three seeds say whether a speaker's effect is the speaker or the seed. One `run_panel.py` call per seed, because `run_control` trains its control at the first seed it is given; each seed's control folds are a different shuffle. 5 and 20 minutes stay at one seed.
- **The forgiven-shared count is withdrawn** (2026-10-03, docs/personalization_research.md § 1.5). It was on in the run (`--forgiven`) on the assumption that a word both models produce and the protocol lacks was spoken; on human-corrected committee clips that held for only about half of them. Its `*_f` columns in the run's results are not evidence, and the option is removed from the code.
- **The control:** a random group of the *other* panel speakers, in 2 folds, trained on the same number of minutes and evaluated on every speaker it never heard. It runs at **each** budget (`--control-folds 2 --control-budgets 5 20 80`). Personalization = the own adapter's gain − the control's gain, on the same test clips (`run_panel._table`). This is the comparison that tells "learned this voice" apart from "learned committee Hebrew".
  - **No shared meetings** (fixed 2026-10-02). A committee meeting often has several panel members in it, so a trainer's session can be the very meeting an evaluated speaker is tested on. In the first version, the 80-minute control trained on 2 of 30843's test meetings (about 4 of the 30 test minutes) and 1 of 30831's. The control's pool now excludes every session of the evaluated speakers' test (both test sets) and dev.
- **Not in this plan:** comparing *similar*-speaker groups against random groups of the same size (`training_next.md`, "sharing"). It can follow on the chosen recipe: about 44 short runs.
- **Metrics:** standard WER (and CER) on the high-quality test set, with the error-type split and the style flag. (The forgiven-shared count that ran beside it is withdrawn; see *Cells* above.)

**Two test sets** (added 2026-10-02). Quality comes from a Whisper-family aligner, so the ≥ 0.95 filter keeps the clips a Whisper model already finds easy. On the second run's test sets it kept only 31–40% of the speech of the hardest speakers (23558, 30701, 30843), the speakers the project is about, against 64% overall. `docs/adaptation_plan.md` asks for results with and without the filter. So every cell and every control evaluation is also scored on a second test set:
- **`panel_test07.parquet`** (`materialize.py plan-test07`): the same test sessions as the high-quality test, every clip at quality ≥ 0.7, no word rule. Built 2026-10-02 at corpus revision `839622c1`; the index reproduces the eleven-speaker `panel_plan_v2.parquet` row for row. 12 speakers: 6,158 clips, 20.3 h; the high-quality test is 46% of it. Per speaker it keeps 26% (23558), 31% (556), 33% (30843) and 37% (30701) of the test speech at ≥ 0.95, and 53–67% for the rest. Beyond plan v2's test it adds 4,085 clips (10.9 h). 0.7 is the corpus floor: below it the protocol demonstrably does not match the audio, and the WER measures the labels, not the model.
- The high-quality test is a subset of it (`hq` = True), so the two differ only in the filter. Session-disjoint from train and validation by construction; `verify` checks it.
- Its columns are `test07.*` in `results.csv`. (The run also scored it with the forgiven-shared count, now withdrawn.) Between 0.7 and 0.8 the protocol differs most from the speech, so read gains on that band against a human reference, not the protocol alone.
- **Per quality band** (`test07.bands.q070`, `q080`, `q090`, `q095`): the gain and the personalization in each of 0.7–0.8, 0.8–0.9, 0.9–0.95 and ≥ 0.95. This answers whether the adapter helps on the hard clips the filter drops.
- Training and tuning are unchanged: still high-quality only.

## 5. What every run collects, for the analysis notebook

Nothing is computed only for display. Every number a plot could need is written to disk as the run goes, and `backup.py` mirrors it to HuggingFace every 30 minutes.

| file | one per | holds |
|---|---|---|
| `outputs/results.csv` (from `outputs/results/<cell>.json`) | scored cell or control evaluation | the settings, train minutes, steps and best checkpoint, test size, base and tuned WER and CER, the gain with its 95% interval and p-values, the substitution/deletion/insertion shares and looping outputs, the style flag, the control's gain and **personalization** = own gain − control's gain, with its own paired bootstrap (`personalization_ci_lo`, `_ci_hi`, `_p`; `_f` and `test07.*` alike) |
| `outputs/results/<cell>.hyps.json`, `base_*_hq.json` | cell; speaker | every test clip's transcription by the tuned and the base model, for re-scoring and per-clip analysis without a GPU |
| `outputs/results/<cell>.test07.hyps.json`, `base_{A,B}_*_test07.json` | cell; speaker | the same on the ≥ 0.7 test set (arm A's cache served the forgiven count, withdrawn) |
| `runs/<cell>/train_meta.json`, `runs_tune/<cell>/train_meta.json` | trained adapter (final run and tuning) | `settings` (speaker, budget, lr, rank, alpha, LoRA dropout, Whisper dropout, augmentation, batch, schedule, seed); `train_log`: the **training loss**, gradient norm and learning rate every 5 steps; `evals`: the **validation loss** every 20 steps; `base_eval_loss` (the untuned model's); `best_step`, `global_step`, `stopped_early`; `trainable_params`, `train_runtime_s`, `peak_gpu_mem_gb`; the train and validation clip IDs |
| `outputs/tuning.csv` | tuning run | the settings, untuned and best validation loss, `rel_drop` (the tuning criterion), best and stop steps |

## Cost (A100 at $1.9–2.7/h; about 3 minutes per early-stopped run)

| step | runs | GPU time |
|---|---|---|
| A. learning rate: 3 × 2 budgets × 4 speakers | 24 | about 1.2 h |
| B. rank × dropout: 3 new × 2 × 4 | 24 | about 1.2 h |
| C. augmentation: 2 × 2 × 4 | 16 | about 0.8 h |
| final: 36 cells (12 speakers) + 6 control trainings + 36 control evaluations + base transcriptions | — | about 2.3–2.8 h |
| the ≥ 0.7 test: the clips outside the high-quality test (10.9 h) for 72 adapters, plus both base models on all 20.3 h once | — | about 1.3–1.7 h |
| arm A on the high-quality test, once per speaker (`--forgiven`, as run; the count is now withdrawn and the option removed) | — | about 0.2 h |
| seeds 1 and 2 at 80 min: 24 cells + 4 control trainings + 24 control evaluations, both tests | — | about 2.5 h |
| **total** | | **about 10–11 h, $16–18 at A100 SXM's $1.59/h** |

Those are GPU-hours. The jobs are independent, so `src/training/box/pod_v3.sh` spreads them over every GPU of the pod: on 4× A100 SXM ($1.59/h each on 2026-10-02) the run takes about 3 hours for about $19, the 12 speakers splitting 3 per GPU (`docs/pod_runbook_v3.md` § How many GPUs).

## Runbook

```bash
# laptop, no GPU
python src/training/word_quality.py                  # word scores for the candidate clips + the report
python src/training/materialize.py plan-v2           # panel_plan_v2.parquet
python src/training/materialize.py verify --plan src/training/panel_plan_v2.parquet --no-audio
python src/training/materialize.py plan-test07         # panel_test07.parquet: the same test sessions at quality >= 0.7
python src/training/materialize.py verify --plan src/training/panel_test07.parquet --no-audio

# any machine with a fast link (no GPU): only clips not already on disk are pulled
python src/training/materialize.py extract --plan src/training/panel_plan_v2.parquet
python src/training/materialize.py verify  --plan src/training/panel_plan_v2.parquet
python src/training/materialize.py upload  --plan src/training/panel_plan_v2.parquet --repo knesset-asr/knesset-committees-panel

# GPU: tuning, validation only
S="30685 23558 30718 30859"; V2=src/training/panel_plan_v2.parquet
T="python src/training/run_panel.py --tune --plan $V2 --max-steps 400 --speakers $S --budgets 5 80 --seeds 0"
$T --lrs 1e-4 3e-4 1e-3                                          # A
$T --lrs <A> --ranks 8 16 --dropouts 0 0.1                       # B (the rank-8, dropout-0 cell is skipped: already run)
$T --lrs <A> --ranks <B> --dropouts <B> --augments specaug specaug+tempo     # C
python src/training/run_panel.py --tuning-report                 # outputs/tuning.csv + the comparison table

# GPU: the final run, test read once
python src/training/run_panel.py --plan $V2 --max-steps 400 --budgets 5 20 80 --seeds 0 \
    --lrs <A> --ranks <B> --dropouts <B> --augments <C or none> --control-folds 2 --control-budgets 5 20 80 \
    --test07-plan src/training/panel_test07.parquet
```

## Handoff: continuing from here (written 2026-10-01)

For whoever runs the training next, a person or a fresh Claude session on a GPU box. Read § 1–5 above first: they carry the decisions. This section is only the *how*, in order.

**On the pod, follow `docs/pod_runbook_v3.md`** (added 2026-10-02): `src/training/box/pod_v3.sh` runs the steps below stage by stage, and `src/training/box/v3_decide.py` applies the tuning rule and the base-WER check by rule. What follows is the manual equivalent.

### Where things stand
- **Code:** PR [hadasy-tau/deep-learning-project#23](https://github.com/hadasy-tau/deep-learning-project/pull/23) (`training-plan-v3` → `main`). Train from `main` once it's merged, otherwise from the branch. It includes Dolev's second run.
- **Data plan:** `src/training/panel_plan_v2.parquet`, built and verified, and committed along with `word_quality.parquet` and `panel_test07.parquet`.
  - The first version (6,462 clips, 27.9 h, 11 speakers) was rebuilt on 2026-10-02 with 13 speakers, then with 12 (§ 1, 30601 deferred): 6,887 clips, 30.0 h.
- **Audio:** the first version's 6,462 WAVs were extracted on Hadas's laptop on 2026-10-01 into `src/training/outputs/panel_audio/<speaker>/<clip>.wav` (git-ignored): 3.1 GB, 46 minutes for 186 shards at about 80 Mbit/s, and `verify` passed every check. They are in `knesset-asr/knesset-committees-panel-hq`.
  - **Completed on 2026-10-02:** 30843's clips and the ≥ 0.7 test's extra ones were extracted and uploaded to `panel-hq` (10,380 of the 10,972 WAVs both plans need were there by 15:40; the last ≥ 0.7 test clips were uploading). Step 1 fetches whatever is still missing; only clips not on disk are pulled.
  - Nothing has been trained on any of it yet.
- **Access:** every project dataset now lives in the HF organization `knesset-asr` (moved 2026-10 from the personal accounts; the old names redirect): `knesset-committees-chunks` (the corpus), `-speakers`, `-inference`, `-panel` (the v1 panel audio, ≥ 0.7), `-adapters` (the second run's adapters and the 462 per-experiment result files, removed from git) and `-panel-hq` (the plan-v3 audio).

### 1. Get the audio onto the GPU box, one of two ways
- **(a) From HuggingFace (recommended).** The laptop's WAVs are uploaded to the private dataset **`knesset-asr/knesset-committees-panel-hq`**, in the project's HF organization `knesset-asr` (Hadas and Dolev are members): all 6,462 WAVs, 3.0 GB, plus `panel_plan_v2.parquet` and a card. It needs an HF token with read access to the `knesset-asr` organization. A fine-grained token scoped only to a personal account gets a 404. This takes a few minutes:
  ```bash
  python src/training/materialize.py download --repo knesset-asr/knesset-committees-panel-hq   # -> src/training/outputs/panel_audio/
  ```
  It is private because the audio comes from ivrit.ai's gated `ivrit-ai/knesset-committees`. Making it public would hand the audio to people who never accepted ivrit.ai's terms, so ask them first.
- **(b) Extract on the box itself.** It needs read access to `knesset-asr/knesset-committees-chunks`. 186 shards pass through, about 125 GB; each is deleted after use, and the run is resumable. That's under an hour on a datacenter link, CPU only.
  ```bash
  python src/training/materialize.py extract --plan src/training/panel_plan_v2.parquet --prefetch 3
  ```

Then, either way, any clip not in `panel-hq` yet (§ 1, § 4 *Two test sets*). Extraction pulls only the clips not already on disk:
```bash
python src/training/materialize.py extract --plan src/training/panel_plan_v2.parquet --prefetch 3
python src/training/materialize.py extract --plan src/training/panel_test07.parquet --prefetch 3
```

Either way, both must pass before anything else:
```bash
python src/training/materialize.py verify --plan src/training/panel_plan_v2.parquet    # splits, filters, every WAV readable
python src/training/materialize.py verify --plan src/training/panel_test07.parquet    # same test sessions as v2, v2's test is the hq subset
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
nohup python src/training/backup.py --repo knesset-asr/<a new private dataset, e.g. knesset-committees-v3-results> --every 30 > backup.log 2>&1 &
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
    --lrs <A> --ranks <B> --dropouts <B> --augments <C or none> --control-folds 2 --control-budgets 5 20 80 \
    --test07-plan src/training/panel_test07.parquet
python src/training/run_panel.py --summary                       # outputs/results.csv
```
- **Time:** about 2–2.5 hours. It's idempotent, so kill and rerun it freely.
- **If 5 and 80 minutes chose different settings,** run the command once per budget group with that group's flags, adding `--control-only` for the second group's controls only where needed. Each recipe gets its own controls through its cell name.
- **Base transcriptions** of the new high-quality test set are made once per speaker and cached as `outputs/results/base_B_<speaker>_…_hq.json`, next to the second run's caches, never over them.

### 6. Reading the results
- **New rows** are the ones with `hq` in `cell`.
- **The headline column** is `personalization_rel`, the own adapter's gain minus the control's gain at the same budget. Its interval is `personalization_ci_lo`, `personalization_ci_hi` (absolute WER) with `personalization_p`: a paired bootstrap of the control's hypotheses against the own adapter's on the same clips. A personalization whose interval crosses zero isn't a result, and neither is a gain whose (`ci_lo`, `ci_hi`) does. For scale: in the second run, re-scored this way, 1–4 of 11 speakers per recipe had a significant personal effect at 80 minutes, mostly 30843 and 23558.
- **Report both test sets side by side:** `personalization_rel` (≥ 0.95) and `test07.personalization_rel` (≥ 0.7), `test07.bands.*` says on which clips it was earned.
- **If `--test07-plan` was missed**, rerun the same command with it: scored cells are not retrained; their ≥ 0.7 scores are added from the saved adapters.
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
