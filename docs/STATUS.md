# Where the project stands

**Updated 2026-10-06.** This is the one document that changes as the project moves. Every other
document is either a stable reference (the README, `design.md`, `committees_handoff.md`,
`adaptation_plan.md`, `paper/README.md`, the folder READMEs) or a dated record that is not edited
afterwards (the build records, `error_map.md`, `inference.md`, the training plans and run logs).
When something here changes, change it here. When a record turns out to be wrong, add one line at
its top pointing here, and list it under § Withdrawn.

## Now

The experiments are done (runs 2–4). The work now is the paper, in `paper/`. Its guide, with the
editorial decisions and where every number comes from, is [`paper/README.md`](../paper/README.md).

- **2026-10-07:** a paragraph-by-paragraph edit of the whole paper. Everything except Limitations
  is one PR, and the Limitations rewrite is a second PR, for Hadas's review, with a PDF that shows its
  changes (`paper/Limitations_changes_for_review.pdf`).

## The pipeline

| stage | state | record |
|---|---|---|
| 1. Speaker index | done, published (`knesset-asr/knesset-committees-speakers`). Text gate passed. The audio gate ran on the adaptation panel only (5.4 % foreign-voice chunks) and on 23641 and 23635 (0 of 45 flagged each) | `speaker_index_plan.md` |
| 2. Chunk corpus | done: ~1.2 M chunks ≤ 30 s, 330 speakers (`-chunks`) | `chunk_corpus_build.html` |
| 3. Inference, both arms | done: 65,990 chunks, 267 speakers (`-inference`). Corpus WER A 0.417, B 0.324. After the quality ≥ 0.7 filter 0.387 / 0.292 | `inference.md` |
| 4–5. Error map, panel | done. The fine-tune helps every speaker (median 24 % of A's error) but rescues no one in particular. Panel: 12 speakers (the 11 of `committees_panel.csv` plus 556, 30601 deferred) | `error_map.md`, `adaptation_plan.md` |
| 6–7. Adaptation | done, three runs. Run 2 (2026-09-20, 11 speakers, ≥ 0.7 data). Run 3 (2026-10-02, plan v3: 12 speakers, ≥ 0.95 data, tuned on validation by rule, the control at every budget, seeds 0–2 at 80 min). Run 4 (2026-10-04, plan v4: the personal-data curve to 360 and 1,440 min for 4 speakers) | `training_run2.md`, `training_run3.md`, `training_run4.md` |

## What the runs found

- **Personalization is +2–3 points of relative WER at 80 minutes**, stable over three seeds (run 3:
  +2.6 / +3.3 / +1.0 % on the high-quality test), significant for about a quarter of the
  speaker-seed pairs, never significantly negative at 80 minutes. The rest of the 12–15 % an adapter removes
  is domain: an adapter on anyone else's committee audio buys it.
- **For whom:** the hardest speakers (23558 reproducibly, +9–10 % on the ≥ 0.7 test in every seed,
  then 30701 and 556) and 30813. Nothing for the S1 speakers the fine-tune failed (30831, 30685).
- **More personal data does not raise personalization** (run 4, up to 1,440 minutes). The own
  adapters keep improving (23558 +29 → +34 % on the ≥ 0.7 test), but the control improves as fast.
  For 23558 personalization falls, +9.7 → +4.1 % (Δ −5.6 [−7.7, −3.6]). A full pass over the data
  and a rank-32 adapter (one seed) do not change it. The rule fixed in advance reads "undecided",
  only because 23558's interval lies below zero.
- **The recipe:** LoRA r = 8 on q/v, lr 1e-4 at 5 min and 3e-4 from 20 min on, no Whisper dropout,
  no augmentation, early stopping on validation loss. Rank, augmentation and (at 80 min) the
  learning rate change nothing.
- **Every WER is measured against the edited protocol**, not a verbatim transcript. The gains are
  gains in agreement with the protocol.

## Possible further experiments (none planned)

- KL regularization to the base model.
- Similar-speaker groups against random ones (the sharing axis).
- A personal text LM against an other-speakers LM.
- Speaker-conditioned models.
- Not recommended: test-time adaptation, in-context learning, prompting.

## Open questions and decisions pending

- 30601 is deferred. Adding him back needs the audio gate on him first.
- The `knesset-asr` datasets are public and ungated since 2026-10-02, except the two results
  datasets (`-v3-results`, `-v4-results`, private). The audio is cut from `ivrit-ai/knesset-committees`,
  which is gated under the ivrit.ai licence. Whether it should stay public is undecided.
- `run_panel.py` and `materialize.py` still default `--plan` to plan v1. `pod_v3.sh` passes it.
- `speaker_index/validate_audio.py` passes the HF token to ffmpeg on the command line, so it is
  visible in a process listing while the gate runs. It should go through an environment variable
  or a header file instead.

## Withdrawn — do not rely on these

| claim | where it appeared | why withdrawn |
|---|---|---|
| The forgiven-shared WER (an inserted word arm A also produced is not charged) as evidence | error map, runs 2–3, plan v3 | it assumes that two models agreeing against the protocol means the protocol is wrong, an assumption too strong to rest a result on. Removed from the code |
| "73 % of B's insertions are shared with A, so they were spoken" | error map, handoff, research | the 73 % is a count. That the words were spoken rests on the same assumption |
| The "attributable error" analysis (35 %, adapters add 10–16 %, 54 % of personal fixes are conventions) | `archive/personalization_research.md` § 1.1–1.2 | built on the same arm-A proxy |
| The semi-verbatim targets as the recipe, and "the adapter learns the stenographer" | runs 2–3 | both rested on the forgiven count |
| "~140 high-quality minutes per speaker" as a data limit | plan v3 | a cap of the 250 candidate minutes scored, not of the corpus |
| "No speaker is significantly negative at any budget" | `training_run3.md` § Results | true at 80 minutes only. At 5 and 20 minutes (seed 0) some speakers are significantly negative: 30813 at 5 on the high-quality test, and 30752, 30813, 556 at 5 and 30831, 30843 at 20 on the ≥ 0.7 test |
| "At 1,440 minutes the best checkpoint came at about 0.84–0.9 of a pass", validation stops improving "within the first pass" | `training_run4.md` § The recipe as run | true of the controls only. The own adapters peak after 0.93–1.11 passes (all runs: 0.83–1.11, stopped at 1.02–1.23). The conclusion stands: about one pass, far below the step cap |

## Where things are

- Code, docs and the paper: `main` of `hadasy-tau/deep-learning-project`. Superseded documents:
  `docs/archive/`.
- Run 2's adapters and per-cell results: `knesset-asr/knesset-committees-adapters`. Run 3's:
  `knesset-asr/knesset-committees-v3-results` (private). Its tables are also in
  `src/training/outputs/` (`results_v3.csv`, `tuning_v3.csv`, `recipe_v3.json`).
- Run 4's results, adapters and hypotheses: `knesset-asr/knesset-committees-v4-results` (private,
  with a git-ignored laptop copy in `src/training/outputs/v4_backup/`). Its tables are in
  `src/training/outputs/` (`results_v4.csv`, `curve_v4.csv`, `capacity_v4.csv`, `recipe_v4.json`,
  `tuning_v4.csv`).
- The panel audio: `knesset-asr/knesset-committees-panel` (plan v1), `-panel-hq` (plans v2 and v4,
  all clips, plus `panel_packs/`, one FLAC tar per speaker).
