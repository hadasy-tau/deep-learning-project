# The third training run (plan v3): log, results and conclusions

Run on 2026-10-02 on one RunPod pod with 4× A100 SXM 80 GB (US-MD-1, $6.36/h), driven by
`src/training/box/pod_v3.sh` from `main` at `74f4f17`. `docs/training_plan_v3.md` is the plan and
its reasons; `docs/pod_runbook_v3.md` is how the pod was run; `docs/training_run2.md` is the
previous run, which every comparison below refers to. Everything the run produced is in the
HuggingFace dataset `knesset-asr/knesset-committees-v3-results` (private): `results.csv` (144
rows: 72 cells, 72 control evaluations), `results/` (528 files: one JSON and one hypotheses file
per cell and test set, the base and arm-A caches, the train/dev transcriptions), `runs/` (84
adapters), `runs_tune/` (64 tuning curves), `tuning.csv`, `logs/` (every stage's log, per GPU,
plus `recipe_v3.json` and `targets_verbatim.parquet`). Checked complete against the pod before it
was stopped; a second copy of everything but the audio sits on Dolev's laptop. Git keeps the three
small tables: `src/training/outputs/results_v3.csv`, `tuning_v3.csv` and `recipe_v3.json`.

> **Revised 2026-10-03: the forgiven-shared count is withdrawn.** The run also scored every cell
> under a count that did not charge an inserted word arm A also produced (`*_f` columns in the
> result files). It rests on the assumption that two models agreeing against the protocol means
> the protocol is wrong; on human-corrected committee clips that held only about half the time
> (`docs/personalization_research.md` § 1.5). Every forgiven number is removed from this report, and the conclusions that
> leaned on it (the "stenographer" reading, the semi-verbatim recipe) are restated below. The
> semi-verbatim targets rest on the same assumption.

**The sentence.** On clean data, with the recipe tuned on validation only and the control run at
every budget and seed, a personal adapter buys **2–3 points of relative WER over an adapter trained
on anyone else's committee audio**, stably across three seeds, significantly for about a quarter of
the speaker-seed pairs and never significantly negative. Everything else in the 12–15 % gain is
domain. How much of any of it is the speaker's speech rather than agreement with the protocol,
which itself differs from what was said by ~18 % of its words, only a human verbatim test can say.

## What changed from the second run

| | second run (2026-09-20) | this run |
|---|---|---|
| speakers | 11 | **12** (30843 kept with a 30-minute test, 556 added; 30601 deferred) |
| data | quality ≥ 0.7 | **≥ 0.95 plus the word rule**, every split (`panel_plan_v2.parquet`: 6,887 clips, 30.0 h) |
| test | one set | **two**: the high-quality clips (45 min a speaker) and every clip of the same sessions at ≥ 0.7 (`panel_test07.parquet`: 6,158 clips, 20.3 h; the clean test is 46 % of it), scored per quality band |
| training length | 8 passes, best epoch | **step mode**: ≤ 400 steps, validation every 20, stop after 4 without improvement |
| recipe | hand-picked | **tuned on validation only**, by rule (`v3_decide.py`) |
| control | 80 min only, one seed | **every budget, every seed**, and never trained on a meeting its speakers are tested on (`74f4f17`) |
| seeds | 0 | **0, 1, 2 at 80 minutes** (own adapters and controls) |
| decoding | greedy | greedy with the **loop guard** (`evaluate.py`: re-decode a hypothesis whose zlib ratio > 3.0 with no repeated 6-gram) |
| headline statistic | personalization as a point estimate | **paired bootstrap on personalization itself** (`personalization_ci_*`, `personalization_p`) |

## Timeline (UTC)

| time | stage | wall |
|---|---|---|
| 13:16–13:24 | setup, data (10,972 WAVs from `panel-hq`, verify 61 OK / 0 FAIL), backup loop, sanity | 8 min |
| 13:25–14:01 | tuning A, B, C: 64 runs, 4 GPUs | 36 min |
| 14:01–14:03 | base-WER check: PASS | 1.5 min |
| 14:03–14:53 | final run, seed 0: 36 cells (34 min), then 3 controls (16 min) | 50 min |
| 14:53–15:27 | seeds 1 and 2 at 80 min: 24 cells (19 min), 2 controls (15 min) | 34 min |
| 15:27 | summary, backup: 120 rows | — |
| 15:27–~16:10 | the semi-verbatim extra (§ Extra) | ~40 min |

The main run took **2 h 03 min** of wall time, about 8 GPU-hours. The GPUs ran at 70–100 %
with 25 GB each; per-process training speed (1.78–2.11 steps/s at batch 8) equalled a lone GPU's,
so four processes did not slow one another. CPU load stayed at 6 of 64 cores. The only idle GPU
time was in the control phases (3 jobs, then 2, on 4 GPUs), about 20 minutes in all.

Sanity: the overfit check drove 20 clips from loss 0.267 to 0.0005 in 30 steps; one 80-minute
tuning run took 2 minutes end to end.

## Tuning (validation only; 4 speakers, 5 and 80 minutes)

The criterion is `rel_drop`, the relative drop in validation loss from the untuned model to the
best checkpoint, averaged over 30685, 23558, 30718 and 30859. A candidate replaces the simpler
incumbent only if it gains ≥ 0.01 and helps ≥ 3 of the 4.

| budget | lr 1e-4 | lr 3e-4 | lr 1e-3 | rank 16 | dropout 0.1 | SpecAugment | + tempo |
|---|---|---|---|---|---|---|---|
| 5 min | **0.294** (3/4) | 0.266 | 0.269 | 0.281 | 0.019 | 0.298 | 0.300 |
| 80 min | 0.434 | **0.432** | 0.429 | 0.432 | 0.077 | 0.430 | 0.431 |

(Rank, dropout and augmentation at each budget's chosen rate.)

- **The learning rate matters only at 5 minutes.** At 80 minutes the three rates sit within
  0.5 points of each other: early stopping on validation loss absorbs the difference (best
  checkpoint at step 145 for 1e-4, 75 for 3e-4, 40 for 1e-3). At 5 minutes 1e-4 gains 2.8
  points on 3 of 4 speakers. So the recipe's rate is **1e-4 at 5 minutes, 3e-4 at 20 and 80**:
  the 5/80 disagreement the plan anticipated.
- **Whisper's `dropout` 0.1 is destructive**: the validation drop collapses to 2–8 % at both
  budgets, for every speaker. Off.
- **Rank 16 equals rank 8** at 80 minutes (0.4321 vs 0.4321) and is worse at 5. Rank 8.
- **Augmentation does nothing**: +0.4–0.6 points at 5 minutes (below the rule), −0.1–0.2 at 80.
  Off, as the September augmentation research predicted for LoRA on whisper-large-v3.

Final recipe: LoRA r = 8, α = 16, on q/v at both sites; lr as above; no Whisper dropout, no
augmentation; ≤ 400 steps with early stopping (the cells stopped at a median of 120–160 steps,
the best checkpoint at a median of 60).

## The base-WER check (PASS)

Base B on each speaker's high-quality test against the error map's `wer_B` (one hour per speaker
across 2019–2025, quality ≥ 0.7):

| speaker | 556 | 23558 | 30685 | 30701 | 30718 | 30752 | 30777 | 30813 | 30831 | 30843 | 30859 | 30868 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base WER, HQ test | .191 | .177 | .180 | .142 | .116 | .117 | .128 | .104 | .142 | .174 | .105 | .121 |
| error map | .376 | .390 | .344 | .345 | .275 | .228 | .198 | .287 | .328 | .361 | .218 | .306 |

Every speaker scores far *below* the error map, as clean clips must. Medians: **0.135** on the
high-quality test, **0.244** on the ≥ 0.7 test. The clean test
has half the error of the second run's, so every relative gain below is measured against a
smaller base.

## Results

Relative change in WER, positive = better. "Own" is the speaker's adapter; "control" is the adapter
trained on the same minutes of the *other* fold's speakers, never on a meeting the speaker is
tested on; **personalization = own − control** on the same test chunks, with its own paired
bootstrap. Medians over the 12 speakers.

### Seed 0, all budgets

| budget | own, HQ | control, HQ | **personalization, HQ** | sig. (of 12) | own, ≥0.7 | control, ≥0.7 | **personalization, ≥0.7** | sig. |
|---|---|---|---|---|---|---|---|---|
| 5 min | +4.3 % | +0.3 % | **+3.1 %** | 4 | +4.2 % | +6.7 % | **−1.9 %** | 2 |
| 20 min | +7.5 % | +7.8 % | **+1.6 %** | 3 | +9.7 % | +10.9 % | **+1.4 %** | 5 |
| 80 min | +12.0 % | +10.1 % | **+2.6 %** | 2 | +15.3 % | +14.3 % | **+2.3 %** | 5 |

"sig." counts speakers with `personalization_p < 0.05` and a positive effect; no speaker is
significantly negative at any budget.

### 80 minutes, three seeds

| seed | own, HQ | control, HQ | personalization, HQ | sig. | personalization, ≥0.7 | sig. |
|---|---|---|---|---|---|---|
| 0 | +12.0 % | +10.1 % | +2.6 % | 2 | +2.3 % | 5 |
| 1 | +11.1 % | +9.7 % | +3.3 % | 3 | +2.2 % | 6 |
| 2 | +12.1 % | +10.4 % | +1.0 % | 3 | +1.7 % | 5 |
| **pooled (36 speaker-seeds)** | +11.6 % | +10.2 % | **+2.4 %** | **8 of 36** | **+2.2 %** | **16 of 36** |

### 80 minutes, per speaker, personalization across seeds 0 / 1 / 2 (* = p < 0.05)

| speaker | profile | HQ test | ≥ 0.7 test | reading |
|---|---|---|---|---|
| 30813 | S3 | +10.1* +11.3* +10.4* | +4.5* +5.5* +7.3* | the control *hurts* her on the clean test (−10, −8, −5 %) while her own adapter does little (0, +3.5, +5.7 %); on the ≥ 0.7 test her own adapter gains +6–7.5 % against the control's +0–2 % |
| 23558 | S2 | +6.2* +3.9 +1.4 | **+9.6* +10.1* +9.2*** | the one large, reproducible personal effect: own +29 % vs control +20 % on the ≥ 0.7 test in every seed |
| 30701 | S1 | +5.8 +5.1* +1.0 | +6.1* +4.7* +3.0* | positive in every seed and test |
| 30752 | C | +2.8 +2.0 +8.0* | +4.0* +5.0* +5.2* | a control speaker with a consistent small effect |
| 556 | S2 | +3.4 +0.3 +2.5 | +2.2* +2.3* +5.2* | small, consistent on the ≥ 0.7 test |
| 30868 | S3 | +2.3 +5.3* +9.6* | +2.1 +2.4* +2.2 | positive throughout, noisy on the clean test |
| 30843 | S2 | +5.4 +4.5 +0.9 | +2.5 +1.7 +0.5 | positive, never significant; her test is 30 minutes |
| 30777 | C | +2.4 +4.5 −0.4 | +0.1 +2.1 −1.3 | nothing |
| 30685 | S1 | −1.5 −0.4 −2.5 | +2.3 +1.9 +1.3 | nothing on the clean test |
| 30859 | S4 | −1.8 −1.8 +0.5 | 0.0 −1.8 +0.2 | nothing: no headroom at WER 0.105 |
| 30718 | S3 | −1.4 +2.6 −4.4 | −2.6 −0.2 −0.6 | nothing |
| 30831 | S1 | −3.3 −6.7 −1.2 | −1.5 +0.5 −1.2 | nothing |

Positive in all three seeds on both tests: 556, 23558, 30701, 30752, 30813, 30843, 30868.
Significant in at least two seeds on the ≥ 0.7 test: 556, 23558, 30701, 30752, 30813.

### By alignment-quality band of the ≥ 0.7 test (80 minutes, all seeds, medians)

| band | minutes | base WER | own | control | personalization |
|---|---|---|---|---|---|
| 0.70–0.80 | 4 | 0.624 | +20.5 % | +20.4 % | +0.5 % |
| 0.80–0.90 | 11 | 0.447 | +19.7 % | +19.6 % | +4.0 % |
| 0.90–0.95 | 14 | 0.304 | +15.7 % | +14.6 % | +2.5 % |
| ≥ 0.95 | 63 | 0.163 | +14.0 % | +11.6 % | +2.3 % |

### The style flag, the error mix and the loops

- **`style_not_speaker` fires in 35 of 36 own cells at 80 minutes** (9/12 at 5, 10/12 at 20), and
  in 33 of 36 control evaluations. Insertions fall from 49 % to 34 % of the errors while deletions
  double, 10 % → 21 %. The adapters write fewer words the protocol lacks and drop more words it
  has. Whether the words they stopped writing were spoken (the adapter learned the protocol's
  omissions) or not (it stopped hallucinating) cannot be read off the protocol: on 23
  human-corrected committee clips about half of the insertions were real speech
  ({N}).
- **Loops.** Base B produces 132 runaway transcriptions over the twelve ≥ 0.7 test sets (9 on the
  clean one); the 80-minute adapters produce 46 (3). The loop guard re-decoded 90 chunks across
  the run (base 2, final 59, seeds 12 batches) and every re-decode compressed less than the
  original and was kept. Part of the gain on the hard clips is loop suppression.
- **Timing.** Cells took a median of 129–157 s at 20 and 80 minutes (train + both test sets);
  251 s at 5 minutes, where the once-per-speaker base transcriptions land (arm A's too, for the
  withdrawn forgiven count).
- **The 5-minute control is nearly untrained**: 13–22 chunks for 140–160 steps at lr 1e-4. It
  gains +0.3 % on the clean test and +6.7 % on the ≥ 0.7 test, which is why the 5-minute
  personalization has opposite signs on the two tests. Read the 5-minute row as noise around
  zero.

## Conclusions

1. **The personal effect is +2–3 points of relative WER at 80 minutes, and it is stable.** Three
   seeds give medians of +2.6 / +3.3 / +1.0 % on the clean test and +2.3 / +2.2 / +1.7 % on the
   ≥ 0.7 test; pooled, 8 of 36 speaker-seeds are significant on the clean test and 16 of 36 on
   the ≥ 0.7 test, and none is significantly negative. The second run's +2 to +5 points, measured
   on noisier data without a per-budget control or a bootstrap on the difference, holds up.
2. **Of the 12–15 % an adapter removes, 10–14 points are domain.** Anyone's 80 minutes of
   committee audio buys them. The control at every budget is what makes this readable; without it
   every cell would read as a 12 % personalization.
3. **The adapters shift the error mix toward the protocol's.** The style flag fires in 35 of 36
   cells: insertions fall, deletions double. That is consistent with learning the protocol's
   omissions, but the protocol cannot show it, and the forgiven count that was meant to is
   withdrawn. The filter changed the *base* (half the error of run 2's test); whether it changed
   what the adapter learns needs a human reference.
4. **For whom.** The effect concentrates where the second run found it: the hardest speakers
   (23558 reproducibly, +9–10 % on the ≥ 0.7 test in every seed; 30701; 556) and 30813, where
   the clean-test "personalization" is the control hurting her and the ≥ 0.7 test shows a real
   own-adapter gain. The S1 speakers, where the Hebrew fine-tune had failed and the most room was
   expected, show no personal component in any seed (30831 and 30685), and neither does the
   low-WER speaker 30859. 30752, a control speaker, shows a small consistent effect.
5. **Where the gain lives.** The gain against the protocol is largest on the hard clips (+20 % at
   quality 0.7–0.9) and +14 % on the clean ones; the control matches it in every band.
   Personalization is ~0 on the worst band and +2–4 % elsewhere, and does not grow on the
   cleanest clips.
6. **Budget.** Personalization is +3 / +1.6 / +2.6 % at 5 / 20 / 80 minutes on the clean test,
   noisy at 5 minutes (the control there is nearly untrained). The curve is shallow; the second
   run's reading stands.
7. **Seeds were worth it.** The seed-to-seed SD of a speaker's personalization is 2.4 points, the
   size of the effect; a single-seed ranking of speakers is not reliable, and the per-speaker
   claims above are made only where all three seeds agree.
8. **The recipe.** The learning rate matters only at 5 minutes; Whisper dropout destroys the
   adapter; rank and augmentation change nothing. Early stopping on validation loss decides how
   much optimisation a cell gets (best checkpoint at a median of 60 steps) and makes the rate
   irrelevant at 80 minutes.

## Extra: semi-verbatim targets at 80 minutes

Added during the run as the remedy for conclusion 3. **It rests on the withdrawn assumption**:
a word goes back when both base models produced it, and only about half of such words were spoken
on the human-corrected clips (`docs/personalization_research.md` § 1.5), so these targets add roughly as many unspoken words as
spoken ones. The second run's run D, the protocol text with the words both base models produced put back
(`targets.py`), trained with the same recipe, seed 0, with its own two-fold control. Both arms
transcribed every train and dev chunk once (12 minutes on 4 GPUs); the targets cover all 4,814
train/dev chunks, change 53 % of them and put back **6.5 words per 100** (10.7 on the second run's
≥ 0.7 data: the clean filter removed four in ten of the omissions, not all). 12 cells + control:
31 minutes.

| 80 min, seed 0 | own, HQ | control, HQ | personalization, HQ (sig +/−) | own, ≥0.7 | personalization, ≥0.7 (sig +) | runaways, ≥0.7 |
|---|---|---|---|---|---|---|
| protocol targets | +12.0 % | +10.1 % | +2.6 % (2 / 0) | +15.3 % | +2.3 % (5) | 44 → 15 |
| semi-verbatim | +5.8 % | +6.3 % | +1.1 % (1 / 1) | +7.1 % | +2.5 % (6) | 44 → 30 |

- **The error mix stops moving toward the protocol's.** The insertion share stays put (49 → 46 %,
  deletions 10 → 11 %, against 49 → 34 % and 10 → 21 % under protocol targets), and the gain
  against the protocol halves, as it must when the targets contain words the protocol lacks.
  Whether that is closer to the speech is not measurable here (see the warning above).
- **The personal component is about the same.** On the ≥ 0.7 test +2.5 %, significant for 6 of 12
  speakers (5 under protocol targets). On the clean test it shrinks to +1.1 %: there the control
  gains as much as the own adapter (+6.3 vs +5.8 %) and 30831 is significantly negative (own
  +5.4 %, control +11.2 %).
- **For whom, again:** 23558 (+6.5 % clean, +8.5 % ≥ 0.7, both p < 0.001), 30868 (+4.3 / +4.1 %),
  and now 30859 (+3.3 / +2.9 %) and 30685 (+4.3 % on ≥ 0.7, p < 0.001), who showed nothing under
  protocol targets. 30831 and 30843 are negative: for them "more committee Hebrew" beats "their
  own 80 minutes" under either objective.
- **The style flag is not a useful instrument when the gain is small.** It still fires in 10 of
  12 semi-verbatim cells although the insertion share barely moves, because it is the *share* of
  a now-small improvement.
- **Loops.** The semi-verbatim adapters suppress fewer runaways (44 → 30 against 44 → 15), but,
  unlike run 2 where they *added* loops, none above the base: the loop guard holds.

## What to do next

- **Build a human verbatim test** before the next training run (`docs/personalization_research.md`
  § 4, P0): every number above is measured against the protocol, which is ~18 % off the speech.
- **Do not adopt the semi-verbatim target** on this evidence: its case rested on the forgiven count.
  Re-score both recipes on the human test first; the hypotheses are saved.
- **The sharing axis** (similar-speaker groups against random ones, `training_next.md`) is the
  natural next experiment: the control already shows that a random group buys 10 points; the
  question is whether a *similar* group closes the remaining 2–3.
- **30601** stays deferred; bringing him in needs the audio gate first (`training_plan_v3.md` § 1).

## Operational notes

- A fresh RunPod PyTorch pod (torch 2.8.0+cu128, 4× A100 SXM) could not reach GitHub's default
  addresses; `140.82.121.4` worked (`git -c http.curloptResolve=github.com:443:140.82.121.4`).
  The system Python is PEP-668 locked; a venv with `--system-site-packages` kept the template's
  torch. HuggingFace put the token under `/workspace/.cache/huggingface` (the template's
  `HF_HOME`), which non-interactive SSH shells do not inherit.
- The backup loop uploaded every 30 minutes and once at the end; the pod was stopped from inside
  with `runpodctl` using the pod-scoped key in PID 1's environment.
- Nothing in `run_panel.py`, `train.py` or `evaluate.py` needed changing during the run.
