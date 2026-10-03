# Training plan v4: the personal-data curve

Written 2026-10-03, before any GPU time. A dated record: where the project stands is
`docs/STATUS.md`.

## The question, and the rule that answers it

Run 3 measured personalization of +2–3 % relative WER at 80 minutes, and +9–10 % for 23558 on
the ≥ 0.7 test in every seed. STATUS § Next 2 asks whether the personal effect is limited by the
amount of personal data. This run extends run 3's curve, the same experiment, from 80 minutes to
360 minutes for four speakers and to 1,440 minutes (24 h) for two of them, with the budget-matched
control at every point.

- **Measure:** `personalization_rel` = (own adapter's gain − control's gain) / base WER, on the same
  test chunks, on the high-quality test and the ≥ 0.7 test. At 80 minutes and at each speaker's top
  budget it is the mean over seeds 0–2. The points in between run seed 0 only and give the curve's
  shape.
- **Delta** = personalization at the top budget − at 80 minutes, per speaker. Its interval comes
  from a paired bootstrap over test chunks, one resample for every seed and both budgets at once, so
  the shared test set is held fixed (`run_panel.py --curve` → `outputs/curve_v4.csv`).
- **Decision, fixed now, on the ≥ 0.7 test:**
  - **"Data limits":** Delta ≥ +2 points with its interval above zero for at least 2 of the 4
    speakers.
  - **"Data does not limit":** every speaker's interval holds zero and its upper end is below +2.
  - **Otherwise "undecided".**
- **Secondary reading:** the own and the control curves apart. If both rise, more data buys more
  domain. If only the own one rises, the adapter is learning the voice.
- **The long branch** (23558, 23641): if the curve keeps rising through 720 and 1,440 minutes,
  there is a case for going further.
- **Known limits:**
  - Everything is scored against the protocol, which differs from the speech by about 18 % of its
    words, so the protocol may hide an improvement. Every hypothesis is saved, so the run can be
    re-scored on a human verbatim test (STATUS § Next 1).
  - The long trains reach back to 2023. They stay inside 2023–2025, never 2018–2020.

## Speakers and budgets

| speaker | source | budgets (minutes) | seeds |
|---|---|---|---|
| 23558 דוד ביטן | panel (S2) | 5/20/80 from run 3; + 160, 360, **720, 1,440** | 160, 720: 0; 360, 1,440: 0–2 |
| 23641 יעקב אשר | new | 20, 80, 160, 360, **720, 1,440** | 20, 160, 720: 0; 80, 360, 1,440: 0–2 |
| 30752 וליד טאהא | panel (C) | 5/20/80 from run 3; + 160, 360 | 160: 0; 360: 0–2 |
| 23635 פנינה תמנו | new | 20, 80, 160, 360 | 20, 160: 0; 80, 360: 0–2 |

**Why these four.** 23558 has the one reproducible personal effect. 23641 is the project's
question in one speaker: Haredi, WER_B 0.41 (94th percentile), the fine-tune removes only 17 % of
arm A's error, 1.3 h of plenum exposure. 30752 is an Arab speaker (Hebrew as a second language)
with a significant effect in every seed. 23635 is Ethiopia-born, the only woman, with an accent no
one else represents. 30859 was dropped: zero effect at 80 minutes and a base WER of 0.106, so an
empty result there could not tell "no effect" from "unmeasurable". Each budget is a separate
training run on a nested subset (`take_budget`, newest session first). A checkpoint taken partway
through a larger run is an unfinished model trained on more data, not a model trained on less.

## Same experiment as run 3

- **Data:** the same rules (quality ≥ 0.95, the word rule, session-disjoint by date).
  `panel_plan_v4.parquet` keeps plan v2's test and dev chunk for chunk for every panel speaker, and
  v2's train is the head of v4's train (`materialize.check_v4_extends_v2`, asserted when the plan
  is built and by `verify`). `panel_test07_v4.parquet` holds `panel_test07`'s panel rows unchanged.
- **The control is run 3's definition:** `control_folds` over the 12 panel speakers at the seed.
  The control trains on the fold that does not hold the evaluated speaker, on the same minutes,
  round-robin, newest first, and every test and dev session of the fold's speakers is kept out of
  its pool.
  - 23641 and 23635 are not in the panel, so they never enter a fold or a pool. They are attached
    to 23558's fold and evaluated on that fold's adapter, whose pool also leaves out their
    sessions. That adapter is labelled `ctrl<k>of2x` (`run_panel.fold_plan`).
  - At 20 and 80 minutes a copy of 23558's fold control is trained for the new speakers, because
    run 3's controls did not leave out their sessions. At 160 minutes and up, one adapter serves
    23558 and the new speakers together.
- **The recipe:** run 3's, from `recipe_v3.json` — LoRA r = 8 on q/v, no Whisper dropout, no
  augmentation, patience 4, a validation every 20 steps, lr 3e-4 at 20 and 80 minutes.
  - **The step cap grows with the data:** 400 to 80 minutes (unchanged; early stopping ends those
    runs near step 160), 1,200 at 160 and 360, 3,000 at 720 and 1,440 (about 4.3 epochs).
  - **One check, on validation only:** 3e-4 against 1e-3 at 360 minutes over the four speakers,
    with run 3's rule (mean `rel_drop` +0.01 and 3 of 4 helped; `box/v4_decide.py`). 720 and 1,440
    take 360's rate. 160 takes 360's if it equals 80's, and 80's otherwise, so 160 is always 3e-4.
  - The check is run as real cells: the 360-minute seed-0 cells at both rates. The rule reads only
    their validation curves. The losing rate's cells get no control, so they never enter a number.
- Every difference from run 3 is in the cell name (`ms1200`, `ms3000`, the lr), and a control is
  joined to an own cell only when the recipe is the same.

## Data, checked before the run (2026-10-03)

`word_quality.py --run4` scored 2,295 more sessions (63,136 candidate chunks, 244 h). Minutes that
pass the word rule, and the plan built from them:

| speaker | passing | plan v4: test / dev / train |
|---|---|---|
| 23558 | 1,603 | 46 / 16 / 1,452 |
| 23641 | 1,770 | 48 / 17 / 1,440 |
| 30752 | 698 | 46 / 15 / 637 |
| 23635 | 516 | 54 / 16 / 372 |

- **Train for the rest of the panel** is filled up to 800 minutes (556, 30813 and 30843 hold less).
  Their train is the pool the large controls draw from. A 1,440-minute control's pool, after its
  held-out sessions, holds 2,365–3,068 minutes in every seed's fold of 23558.
- **Fallback rule:** a speaker with fewer than 1,440 train minutes takes 1,200 as the top budget
  (`jobs_v4.top_budget`). Neither needed it.
- **Audio gate:** `box/gate_v4.py` was run on 23641 and 23635 before the plan was closed (the
  result is in `docs/STATUS.md`). It sampled 15 chunks per split per speaker, with the 30 speakers
  most often heard in the same sessions as the reference population.

## Where it runs, and what it costs

All the data work happens off the GPU:

1. **Laptop:** the word scores, the plans and the gate. Then `box/data_v4.sh` extracts the 33,061
   new clips (137 h) from the corpus. That means reading 387 of its 410 shards, about 300 GB, to
   keep about 16 GB. It verifies the audio and adds the WAVs and both v4 plans to
   `knesset-asr/knesset-committees-panel-hq`, beside plan v2's clips, which do not change.
2. **A GPU pod** (4× A100 SXM 80 GB, Secure cloud) runs `box/pod_v4.sh`:
   - **boot**, about 15 minutes: deps, weights, the audio and run 3's results, in parallel.
   - **run:** one worker per GPU on a shared queue (`box/jobs_v4.py`). The queue starts with the
     overfit check, then the base transcriptions, then the lr check. The 720 and 1,440-minute jobs
     start the moment the lr is decided.
   - **finish:** the summary, the curve, and the backup to `knesset-asr/knesset-committees-v4-results`
     verified file by file. Then the pod stops itself.

`jobs_v4.py dry-run`: 57 jobs, about 8.4 GPU-hours. That is about 2.4 h of wall time on 4 GPUs,
about $15 at $1.59 per GPU-hour.

**Stop rules:**
- boot takes more than 25 minutes;
- a 1,440-minute job takes more than 45 minutes;
- the projected cost goes over $25;
- the overfit check's loss does not fall;
- base WER on the high-quality test is more than 0.10 above the error map (`v4_decide.py base`);
- `verify` fails.

Each one means stop and report.
