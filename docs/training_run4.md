# Training run 4: the personal-data curve

Run on 2026-10-04. The plan is `docs/training_plan_v4.md`, amended the same day (`docs/STATUS.md`).
A dated record: where the project stands is `docs/STATUS.md`.

## The question, and the answer

Does the personal effect grow with more personal data? Run 3's curve was extended from 80 minutes
to 360 minutes for four speakers and to 1,440 minutes (24 h) for two. Every point has the
budget-matched control.

**Personalization does not grow with data.** The speakers' own adapters keep improving, but the
control, trained on the same amount of other speakers' committee audio, improves just as fast.
On the ≥ 0.7 test, personalization at the top budget is the same as at 80 minutes or lower for all
four speakers. The one significant change is a decrease (23558).

## How it ran

| pass | pod | jobs | time | cost |
|---|---|---|---|---|
| 1 | 4× A100 SXM, 05:50–07:12 UTC | 45 of 57: everything up to 360 min; the 720/1,440 jobs were put on hold | ~1.4 h | ~$9 |
| 2 | 4× A100 SXM, 13:25–14:20 UTC | the 12 held jobs with the amendment, plus 3 r = 32 jobs | ~0.9 h | ~$6 |

- **Boot:** about 2 minutes. The 44,033 clips came as 14 FLAC packs, downloaded in 1.1 min and
  decoded in 0.6 min (`box/packs_v4.py`). A snapshot of the WAVs one by one had measured
  ~5 files/s, which would have meant ~2.3 h of idle GPUs.
- **Checks:**
  - The overfit check passed (0.45 → 0.0002).
  - The base WER on the high-quality test is within the rule for all four speakers; 23558's is
    0.177, run 3's value exactly.
  - The backup was verified file by file after both passes (326/326, then 338/338), with a laptop
    copy of all 876 files.
  - Both pods were stopped or terminated by the scripts.
- **First pass closed early:** the 12 held jobs were read by the queue as "nothing left" and the
  run closed itself. Fixed (`jobs_v4._finished`), and resumed from the backup on a new pod
  (`pod_v4.sh resume`).

## The recipe as run

- **The rest of run 3's recipe**, unchanged: r = 8 on q/v, no dropout, no augmentation, patience 4.
- **lr 3e-4 at every budget:** the lr check at 360 minutes kept it (1e-3 was worse for 4 of 4
  speakers, mean `rel_drop` 0.381 against 0.391).
- **Early stopping at 360 minutes** came after about 1.3 passes over the data.
- **The amendment for 720 and 1,440 minutes** (`--min-epochs 1`): no stopping before one full pass.
  - The best checkpoint still came at about 0.84–0.9 of a pass, with training stopped at
    1.02–1.11 passes. So validation genuinely stops improving within the first pass; this is no
    longer the patience cutting the run short.
- **The capacity check:** r = 32 at 1,440 minutes, seed 0, own adapter and control.

## Results (≥ 0.7 test; seeds 0–2 at 80 and at the top budget, seed 0 in between)

Personalization (own gain − control gain, relative to base WER), with the own adapter's gain in
brackets:

| speaker | 80 | 160 | 360 | 720 | 1,440 | Δ top − 80 [95 % CI] |
|---|---|---|---|---|---|---|
| 23558 ביטן | +9.7 % (29.4) | +6.3 % (29.1) | +6.0 % (32.9) | +5.4 % (33.7) | +4.1 % (34.3) | **−5.6 [−7.7, −3.6]** |
| 23641 אשר | +3.3 % (17.4) | +3.6 % (19.9) | +2.2 % (21.8) | +3.0 % (22.8) | +2.3 % (23.4) | −1.1 [−2.8, +0.5] |
| 30752 טאהא | +4.7 % (23.3) | −3.9 % (20.8) | +1.7 % (26.9) | — | — | −3.0 [−6.1, +0.2] |
| 23635 תמנו | +0.4 % (15.0) | +0.2 % (18.2) | −0.1 % (19.5) | — | — | −0.6 [−3.0, +1.8] |

- **The high-quality test** says the same with wider intervals: 23558 +3.9 → +4.8 % (Δ +0.9
  [−3.7, +6.2]), 23641 +5.6 → +2.0 % (Δ −3.6 [−8.0, +1.0]).
- **The plan's rule** (fixed in advance, ≥ 0.7 test) says "undecided". No speaker shows "data
  limits" (Δ ≥ +2 with its interval above zero). And "data does not limit" requires every interval
  to hold zero, which 23558's does not: it lies entirely below zero, the opposite of what more data
  was supposed to do.
- **Read plainly:** across 4 speakers and up to 24 hours, more personal data never raised
  personalization above its 80-minute level, and for the one speaker with a large personal effect
  it lowered it.

## The capacity check (seed 0, 1,440 minutes)

| speaker | test | r = 8 | r = 32 |
|---|---|---|---|
| 23558 | ≥ 0.7 | +2.7 % | +4.4 % |
| 23558 | HQ | +7.1 % | +2.9 % |
| 23641 | ≥ 0.7 | +1.6 % | +2.3 % |
| 23641 | HQ | +1.7 % | +6.7 % |

- **No consistent gain from a bigger adapter.** The differences go both ways, are within one
  seed's noise (2–3 points in run 3), and no r = 32 cell comes near 23558's +9.7 % at 80 minutes.
- **So the adapter's size does not explain the flat curve.**
- This is one seed. It closes the objection only as far as one seed can.

## What it means for the project

- **Ruled out:** more personal data does not unlock personalization in this setup, and the two
  obvious objections were tested (a full pass over the data, a 4× larger adapter).
  - The adapter does gain from data, but so does any adapter trained on the same minutes of other
    committee speech. The gain is domain, not voice.
- **Not ruled out:**
  - **Everything here is scored against the edited protocol** (~18 % off the speech). If the
    personal effect lives in words the protocol does not record, this metric cannot see it. Every
    hypothesis is saved, so the whole run can be re-scored on a human verbatim test without a GPU
    (STATUS § Next 1).
  - A different objective (KL to the base model, a personal LM) is a separate question.
- **For whom:** 23558 keeps the largest personal effect at every budget, but it is largest at
  small budgets (+9.9 % at 5 min, +9.7 % at 80). 23635 has none at any budget.

## Where things are

- **Results, adapters, hypotheses, logs:** `knesset-asr/knesset-committees-v4-results` (private).
  The laptop copy is `src/training/outputs/v4_backup/` (git-ignored).
- **Tables:** `curve_v4.csv` (per speaker and budget, with Δ and its bootstrap),
  `capacity_v4.csv`, `results.csv`, `recipe_v4.json`, `tuning_v4.csv`, `queue_v4.json`. Copied
  into `src/training/outputs/` as `curve_v4.csv` and `results_v4.csv`.
