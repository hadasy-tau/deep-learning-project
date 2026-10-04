# Where the project stands

**Updated 2026-10-04.** This is the one document that changes as the project moves. Every other
document is either a stable reference (the README, `design.md`, `committees_handoff.md`,
`adaptation_plan.md`, `pod_runbook_v3.md`, the folder READMEs) or a dated record that is not
edited afterwards (the build records, `error_map.md`, the training plans and run logs,
`personalization_research.md`). When something here changes, change it here; when a record turns
out to be wrong, add one line at its top pointing here, and list it under § Withdrawn.

## The pipeline

| stage | state | record |
|---|---|---|
| 1. Speaker index | done, published (`knesset-asr/knesset-committees-speakers`). Text gate passed; the audio gate has run on the adaptation panel only (5.4 % foreign-voice chunks) | `speaker_index_plan.md` |
| 2. Chunk corpus | done: ~1.2 M chunks ≤ 30 s, 330 speakers (`-chunks`) | `chunk_corpus_build.html` |
| 3. Inference, both arms | done: 65,990 chunks, 267 speakers (`-inference`). Corpus WER A 0.417, B 0.324; after the quality ≥ 0.7 filter 0.387 / 0.292 | `inference.md` |
| 4–5. Error map, panel | done. The fine-tune helps every speaker (median 24 % of A's error) but rescues no one in particular. Panel: 12 speakers (the 11 of `committees_panel.csv` plus 556; 30601 deferred) | `error_map.md`, `adaptation_plan.md` |
| 6–7. Adaptation | four runs. Run 2 (2026-09-20, 11 speakers, ≥ 0.7 data); run 3 (2026-10-02, plan v3: 12 speakers, ≥ 0.95 data, tuned on validation by rule, the control at every budget, seeds 0–2 at 80 min); run 4 (2026-10-04, the personal-data curve to 360 / 1,440 min: more data does not raise personalization) | `training_run2.md`, `training_run3.md`, `training_run4.md` |

## What the runs found

- **Personalization is +2–3 points of relative WER at 80 minutes**, stable over three seeds (run 3:
  +2.6 / +3.3 / +1.0 % on the high-quality test), significant for about a quarter of the
  speaker-seed pairs, never significantly negative. The rest of the 12–15 % an adapter removes
  is domain: an adapter on anyone else's committee audio buys it.
- **For whom:** the hardest speakers (23558 reproducibly, +9–10 % on the ≥ 0.7 test in every seed;
  30701, 556) and 30813; nothing for the S1 speakers the fine-tune failed (30831, 30685).
- **The recipe:** LoRA r = 8 on q/v, lr 1e-4 at 5 min and 3e-4 at 20/80, no Whisper dropout, no
  augmentation, early stopping on validation loss. Site, rank and augmentation change nothing.
- **The reference is noisy:** on 23 human-corrected committee clips the protocol differs from the
  speech by ~18 % of its words, and about 40 % of B's measured WER there is the protocol.
  Whether the personal effect is larger against what was actually said is not known.

## Next (from `personalization_research.md` § 4)

1. **A human verbatim test** of 15–20 minutes per speaker, corrected from the protocol, blind to the
   models, by ivrit.ai's guidelines (`ivrit-ai/eval-forced-alignment`), 10 % double-annotated.
   Re-score runs 2–3 on it without a GPU. Success, fixed in advance: median personalization ≥ 5 %
   with the interval above zero for half the speakers in two of three seeds.
2. **More personal data: run 4, done 2026-10-04** (`training_plan_v4.md`, `training_run4.md`). Ruled out as
   the route to personalization in this setup; re-score on the human test (1) before closing it for good.
   - **Design:** run 3's curve is extended to 360 minutes for 23558, 30752 and two new speakers,
     23641 יעקב אשר (Haredi, hard, the fine-tune barely helped) and 23635 פנינה תמנו
     (Ethiopia-born). 23558 and 23641 go on to 1,440 minutes.
   - **Same experiment as run 3:** the same data rules, test sets, folds and control definition.
   - **The decision rule is fixed in advance** (Delta of personalization, top budget − 80 minutes,
     on the ≥ 0.7 test).
   - **Audio gate on 23641 and 23635** (`box/gate_v4.py`, 2026-10-03): 0 of 45 sampled chunks flagged
     for each, against 3.1 % for the 30 speakers heard in the same sessions. Nothing excluded.
   - **Prepared:** the word scores (2,295 more sessions), `panel_plan_v4.parquet` and
     `panel_test07_v4.parquet`, both verified, and the queue, about 8.4 GPU-hours.
   - **Audio ready (2026-10-04):** all 44,033 clips of both v4 plans are extracted and verified, and
     they are in `-panel-hq` as WAVs. They are also there as 14 lossless FLAC packs (`panel_packs/`,
     11.7 GB), which is what the pod fetches: 44k small files download at about 5 files/s.
   - **First pass, 2026-10-04** (4× A100, about 1.4 h, about $9; `knesset-asr/knesset-committees-v4-results`):
     - 45 of 57 jobs ran, everything up to 360 minutes, with the backup verified (326/326).
     - The lr check kept 3e-4: 1e-3 was worse for 4 of 4 speakers.
     - **Personalization does not grow from 80 to 360 minutes; it shrinks.** On the ≥ 0.7 test
       23558 goes +9.7 → +6.0 % (Δ −3.7 [−5.6, −1.9]), 23641 +3.3 → +2.2, 30752 +4.7 → +1.7,
       23635 +0.4 → −0.1. The own adapters keep improving (23558 +29 → +33 %), but the control
       improves as fast. The rule's verdict is "undecided": no speaker shows "data limits", and
       23558's interval lies below zero.
   - **Amendment before the 720/1,440 jobs (2026-10-04):**
     - At 360 minutes early stopping came after about 1.3 passes. At 1,440 the fixed patience would
       stop before half a pass, so those budgets would not really be seen. So 720 and 1,440 get
       `--min-epochs 1`: no stopping before one full pass, and the best checkpoint is still chosen
       on validation.
     - The capacity check is added back: r = 32 at 1,440, seed 0, own and control.
     - The aim is to rule out "more data" with the two objections closed.
     - The 12 jobs had been put on hold to decide this, and the queue read the hold as the end and
       closed the run (fixed: `jobs_v4._finished`).
   - **Second pass, same day** (new pod, about 0.9 h, about $6): the 12 held jobs plus the r = 32 check.
     The backup was verified (338/338), a laptop copy was made (876 files), and the pod was
     terminated. **Run 4 is done: `docs/training_run4.md`.**
   - **Result:** personalization does not grow with personal data, up to 1,440 minutes.
     - On the ≥ 0.7 test from 80 minutes to the top budget: 23558 +9.7 → +4.1 % (Δ −5.6 [−7.7, −3.6]),
       23641 +3.3 → +2.3, 30752 +4.7 → +1.7, 23635 +0.4 → −0.1.
     - The own adapters improve (23558 +29 → +34 %), but the control improves as fast.
     - A full pass over the data (the best checkpoint at about 0.85 of a pass) and r = 32 (no
       consistent gain, one seed) do not change it.
     - The rule says "undecided" only because 23558's interval lies below zero.
   - **Added 2026-10-04: 5 minutes for 23641 and 23635** (1× H100, about 20 min, about $1), at run 3's
     5-minute recipe (lr 1e-4), with a copy of 23558's fold control, so all four curves start at 5. The
     verdict and every Δ are unchanged (they compare 80 with the top budget). Backup verified; laptop
     copy 918/918; pod terminated. At 5 minutes the controls are weak on the clean test (−5 and −10 %), so
     the 5-minute personalization there (+14, +10) is mostly a poor control, not a strong own adapter.
3. **KL regularization to the base model.**
4. Later: a personal text LM against an other-speakers LM; similar-speaker groups; speaker-
   conditioned models. Not recommended now: test-time adaptation, in-context learning, prompting.

## Open questions and decisions pending

- 30601 is deferred; adding him back needs the audio gate on him first.
- The `knesset-asr` datasets are public and ungated since 2026-10-02 (except
  `knesset-committees-v3-results`); the audio is cut from `ivrit-ai/knesset-committees`, which is
  gated under the ivrit.ai licence. Whether it should stay public is undecided.
- `run_panel.py` and `materialize.py` still default `--plan` to plan v1; `pod_v3.sh` passes it.
- `speaker_index/validate_audio.py` passes the HF token to ffmpeg on the command line, so it is
  visible in a process listing while the gate runs. It should go through an environment variable
  or a header file instead.

## Withdrawn — do not rely on these

| claim | where it appeared | why withdrawn |
|---|---|---|
| The forgiven-shared WER (an inserted word arm A also produced is not charged) as evidence | error map, runs 2–3, plan v3 | on human-corrected clips, two models agreeing against the protocol meant the protocol was wrong only about half the time (`personalization_research.md` § 1.5); removed from the code |
| "73 % of B's insertions are shared with A, so they were spoken" | error map, handoff, research | the 73 % stands; about half of such words were spoken (19 of 36) |
| The "attributable error" analysis (35 %, adapters add 10–16 %, 54 % of personal fixes are conventions) | `personalization_research.md` § 1.1–1.2 | built on the same arm-A proxy |
| The semi-verbatim targets as the recipe; "the adapter learns the stenographer" | runs 2–3 | both rested on the forgiven count |
| "~140 high-quality minutes per speaker" as a data limit | plan v3 | a cap of the 250 candidate minutes scored, not of the corpus |

## Where things are

- Code and docs: `main` of `hadasy-tau/deep-learning-project`. Superseded documents: `docs/archive/`.
- Run 2's adapters and per-cell results: `knesset-asr/knesset-committees-adapters`. Run 3's:
  `knesset-asr/knesset-committees-v3-results` (private); its tables are also in
  `src/training/outputs/` (`results_v3.csv`, `tuning_v3.csv`, `recipe_v3.json`).
- The panel audio: `knesset-asr/knesset-committees-panel` (plan v1); `-panel-hq` (plans v2 and v4,
  all clips, plus `panel_packs/`, one FLAC tar per speaker).
- Run 4's results, adapters and hypotheses: `knesset-asr/knesset-committees-v4-results` (private; laptop
  copy `src/training/outputs/v4_backup/`). Its tables are in `src/training/outputs/` (`results_v4.csv`,
  `curve_v4.csv`, `capacity_v4.csv`, `recipe_v4.json`, `tuning_v4.csv`).
