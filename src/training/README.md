# `src/training/` — the adaptation code

> This file describes the code. **To run training, follow `docs/pod_runbook_v3.md`**; the plan
> and its reasons are `docs/training_plan_v3.md`; where the project stands is `docs/STATUS.md`.
> The design is `docs/adaptation_plan.md`. The panel is `src/evaluation/outputs/committees_panel.csv`
> plus 556 (`word_quality.PANEL_ADD`).

| file | holds |
|---|---|
| `materialize.py` | `plan`, `plan-v2`, `plan-test07` (laptop): which chunks, which split — writes the plan tables. `extract` (any fast link): shards → WAVs under `outputs/panel_audio/`. `verify`, `upload`, `download` |
| `panel_plan.parquet` | plan v1, used by run 2: 8,113 chunks, 28.8 h, 11 speakers, 111 shards; per speaker ≥ 45 min test (the newest sessions), ~15 min dev, ≥ 80 min train, session-disjoint by date, train ordered latest-first so budgets nest |
| `train.py` | one cell → one adapter (`train_cell`), `overfit_check` |
| `run_panel.py` | cells → adapters → scored results (`outputs/results/*.json`, `outputs/results.csv`); WER per cell; `--control-folds K` for the cross-speaker control (D3) |
| `targets.py` | semi-verbatim training targets: the protocol text with the words both base models produced put back (`--build-targets`). Rests on the withdrawn two-model-agreement assumption (docs/STATUS.md § Withdrawn) |
| `word_quality.py` | per-word alignment scores for the candidate train/dev clips (from the raw ivrit.ai sessions) and the word rule `word_ok`; writes `word_quality.parquet` |
| `panel_plan_v2.parquet` | the high-quality plan: test, dev and train all at quality ≥ 0.95 passing the word rule; 12 speakers: the panel, plus 556 (30601 deferred); 30843 with a 30-minute test (`materialize.py plan-v2`) |
| `panel_test07.parquet` | the second test set: plan v2's test sessions, every chunk at quality ≥ 0.7, no word rule; v2's test is its `hq` subset (`materialize.py plan-test07`, scored with `run_panel.py --test07-plan`) |
| `backup.py` | copies `outputs/results/` and `runs/` (adapters only, no trainer checkpoints) to a private HF dataset on a loop |
| `box/` | `pod_v3.sh` (the plan-v3 pod session, stage by stage, on every GPU), `v3_decide.py` (the tuning rule and the base-WER check), `env.sh` (caches on the container disk, thread cap), and run 2's detached chains (`followups.sh`, `next_runs.sh`, `run_d.sh`, historical) |
| `requirements.txt` | the stack; pin torch to the box's CUDA build |

## The machine

LoRA on `whisper-large-v3` is comfortable on 24 GB with bf16 (L4, A10G). Full fine-tuning
(the method axis) needs 40 GB (A100) with 8-bit Adam. Every run so far is LoRA only, so
24 GB is enough. Disk: 10 GB for the audio and adapters, plus 2 GB per shard in flight if
you extract on the box. Network: extraction pulls ~85 GB once.

## Speed, measured on an A100 80 GB (2026-09-20)

The first run took ~30 min a cell; the second takes ~1.5 / 2 / 4.7 min for the 5 / 20 / 80-minute
budgets, about 2 h for 33 cells plus the control. What changed, in order of weight:

- **BLAS threads.** Log-mel extraction is a numpy STFT; with the default pool (one thread per
  core, 255 on that box) it took 1.4 s a chunk, 8 ms with four. Through `__getitem__` and the
  per-epoch dev eval that was most of the 30 minutes. `train.py` and `run_panel.py` cap
  `OMP_NUM_THREADS` at 8 before numpy loads; the eval path computes the log-mel on the GPU.
- **bf16 base weights, no gradient checkpointing** for adapters on a card ≥ 40 GB: 823 → 518 ms
  a step at batch 8, 21 GB peak. Full fine-tuning keeps fp32 weights and checkpointing.
- **Generation at batch 64** with GPU features: the 224-chunk test set in 24 s against 138 s at
  batch 8. `--eval-batch 32` on a 24 GB card.
- Trainer checkpoints are deleted once the best epoch is restored: the adapter is 21 MB, the
  checkpoints were ~3× that each, and the GPU box's volume had a 10 GB quota.

`run_panel.py` is idempotent: a cell with a result JSON is skipped, a cell with a finished
adapter is not retrained, and the base model's transcription of each speaker's test set is
cached once. Kill it and rerun it freely.

## What a result row holds

`run_panel.py --summary` writes `outputs/results.csv`, one row per cell or control evaluation:

- `wer_base`, `wer_tuned`, `delta_abs`, `delta_rel` with `ci_lo`, `ci_hi`, `p_boot`: WER against
  the protocol on the speaker's test chunks, base vs tuned on the same chunks (paired bootstrap).
- `control = True` rows: the speaker scored on an adapter trained on the same minutes of the
  *other* fold's speakers (`--control-folds K`), never on a meeting the speaker is tested on.
- `personalization_rel` / `_abs` = own gain − control gain, with its own paired bootstrap
  (`personalization_ci_lo`, `_ci_hi`, `personalization_p`). This is the headline.
- `test07.*`: the same on the quality ≥ 0.7 test set (`--test07-plan`), with `test07.bands.*` per
  alignment-quality band.
- `improvement_from_insertions` and `style_not_speaker` (> 0.5): a gain made mostly of fewer
  insertions is flagged, not counted as personalization (`docs/adaptation_plan.md` § Verification).
- Older result files (runs 2–3) also carry `*_f` columns, the forgiven-shared count; it is
  withdrawn (`docs/STATUS.md` § Withdrawn) and the summary drops them.

## Not yet

- Long-form scoring on whole personal-test recordings (D7's secondary protocol): the chunks
  are on disk, the recordings are not.
- The sharing axis (similar-speaker groups against random ones) and the method axis (full
  fine-tuning, DoRA, IA3): `run_panel.py` takes `--methods`, but no run has used anything but
  LoRA. The site (`--sites encoder`, `decoder_mlp`) and rank (8, 16) axes were varied in runs 2
  and 3; the cross-speaker control (`--control-folds`) has run in every run since run 2.
