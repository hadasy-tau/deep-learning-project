# Where the project stands

**Updated 2026-10-03.** This is the one document that changes as the project moves. Every other
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
| 6–7. Adaptation | three runs. Run 2 (2026-09-20, 11 speakers, ≥ 0.7 data) and run 3 (2026-10-02, plan v3: 12 speakers, ≥ 0.95 data, tuned on validation by rule, the control at every budget, seeds 0–2 at 80 min) | `training_run2.md`, `training_run3.md` |

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
2. **More personal data:** 80 → 320 → 1,200 minutes per speaker with the matched control. The
   corpus holds a median of ~15 h per panel speaker at quality ≥ 0.95, ~23 h at ≥ 0.7.
3. **KL regularization to the base model.**
4. Later: a personal text LM against an other-speakers LM; similar-speaker groups; speaker-
   conditioned models. Not recommended now: test-time adaptation, in-context learning, prompting.

## Open questions and decisions pending

- 30601 is deferred; adding him back needs the audio gate on him first.
- The `knesset-asr` datasets are public and ungated since 2026-10-02 (except
  `knesset-committees-v3-results`); the audio is cut from `ivrit-ai/knesset-committees`, which is
  gated under the ivrit.ai licence. Whether it should stay public is undecided.
- `run_panel.py` and `materialize.py` still default `--plan` to plan v1; `pod_v3.sh` passes it.

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
- The panel audio: `knesset-asr/knesset-committees-panel` (plan v1), `-panel-hq` (plan v3, complete).
