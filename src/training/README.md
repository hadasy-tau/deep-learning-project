# Adaptation — the runbook

> **Current plan: `docs/training_plan_v3.md`** — high-quality data only, for test, validation and
> train (`word_quality.py`, `materialize.py plan-v2`), step-mode training with fixed early stopping,
> lean validation-only tuning (`run_panel.py --tune`, `--tuning-report`) and augmentation. Its
> runbook supersedes the session below, which is how the second run was driven.
> **On the GPU pod: `docs/pod_runbook_v3.md`**, which drives plan v3 through `box/pod_v3.sh <stage>`,
> with the tuning rule and the base-WER check applied by `box/v3_decide.py`.

What is in this folder, how run 2 was driven, and what to bring back. For a new run, **read
`docs/pod_runbook_v3.md` first**. `docs/training_handoff.md` is historical (runs 1–2) but keeps
the two rules for reading results. The design is `docs/adaptation_plan.md`; the panel is
`src/evaluation/outputs/committees_panel.csv` plus 556 (`word_quality.PANEL_ADD`); the audio
plan v3 uses is described by `panel_plan_v2.parquet` and `panel_test07.parquet` (run 2 used
`panel_plan.parquet`): one row per chunk, with who, which session, which split, which shard.

| file | holds |
|---|---|
| `materialize.py` | `plan` (laptop): which chunks, which split — writes `panel_plan.parquet`. `extract` (any fast link): shards → WAVs under `outputs/panel_audio/`. `verify`, `upload`, `download` |
| `panel_plan.parquet` | 8,113 chunks, 28.8 h, 11 speakers, 111 shards; per speaker ≥ 45 min test (the newest sessions), ~15 min dev, ≥ 80 min train, session-disjoint by date, train ordered latest-first so budgets nest |
| `train.py` | one cell → one adapter (`train_cell`), `overfit_check` |
| `run_panel.py` | cells → adapters → scored results (`outputs/results/*.json`, `outputs/results.csv`); WER per cell; `--control-folds K` for the cross-speaker control (D3) |
| `targets.py` | semi-verbatim training targets: the protocol text with the words both base models produced put back (`--build-targets`). Rests on the withdrawn two-model-agreement assumption (docs/personalization_research.md § 1.5) |
| `word_quality.py` | per-word alignment scores for the candidate train/dev clips (from the raw ivrit.ai sessions) and the word rule `word_ok`; writes `word_quality.parquet` |
| `panel_plan_v2.parquet` | the high-quality plan: test, dev and train all at quality ≥ 0.95 passing the word rule; 12 speakers: the panel, plus 556 (30601 deferred); 30843 with a 30-minute test (`materialize.py plan-v2`) |
| `panel_test07.parquet` | the second test set: plan v2's test sessions, every chunk at quality ≥ 0.7, no word rule; v2's test is its `hq` subset (`materialize.py plan-test07`, scored with `run_panel.py --test07-plan`) |
| `backup.py` | copies `outputs/results/` and `runs/` (adapters only, no trainer checkpoints) to a private HF dataset on a loop |
| `box/` | the GPU box's `env.sh` (caches on the container disk, thread cap) and the detached run chains of the 2026-09-20 session |
| `requirements.txt` | the stack; pin torch to the box's CUDA build |

## The machine

LoRA on `whisper-large-v3` is comfortable on 24 GB with bf16 (L4, A10G). Full fine-tuning
(the method axis) needs 40 GB (A100) with 8-bit Adam. Every run so far is LoRA only, so
24 GB is enough. Disk: 10 GB for the audio and adapters, plus 2 GB per shard in flight if
you extract on the box. Network: extraction pulls ~85 GB once.

## The session (run 2, historical)

Run 2 (2026-09-20) was driven by hand on `panel_plan.parquet`, 11 speakers at quality ≥ 0.7,
with `run_panel.py --arm B --budgets 5 20 80 --seeds 0 --control-folds 2 --control-budget 80`
and the follow-up chains in `box/` (`followups.sh`, `next_runs.sh`, `run_d.sh`); the full
record is `docs/training_run2.md`. A new run follows `docs/pod_runbook_v3.md`, which drives
`box/pod_v3.sh`. The flags below still exist, except `--select forgiven`.

Recipe options added on 2026-09-20 (`docs/training_run2.md` for what each did):
`--lrs 3e-4` (the rate to use; 1e-3 restores epoch 1 everywhere), `--sites decoder_mlp`,
`--targets verbatim` (needs `--transcribe-parts` once, then `--build-targets`),
`--select forgiven` (removed with the forgiven count), `--control-folds K` (one control per recipe).

Seeds 1 and 2 only if seed 0 shows an effect (`docs/training_handoff.md` § Before the second
run, point 5). Passes are fixed at 8 whatever the budget, so steps scale with the budget:
`train_steps` is in every result row (point 4).

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

## Reading the results

`wer_base` / `wer_tuned` are WER against the protocol. The second and third runs also carried
a forgiven-shared count (`*_f`: an inserted word arm A also produced was not charged); it is
**withdrawn and removed** — on human-corrected committee clips, two models agreeing against the
protocol meant the protocol was wrong only about half the time (docs/personalization_research.md § 1.5). The
summary drops `*_f` columns from older result files.

With `--control-folds K`, each speaker is also evaluated on an adapter trained for the same
budget on a pool of the *other* panel speakers (rows with `control = True`; K = 2 leaves no
speaker evaluated on their own audio, one extra training run over a single adapter). The
summary joins it per speaker: `personalization_abs = delta_abs − control_delta_abs`, the D3
quantity. Only the 80-minute row is budget-matched to the control.

Each cell reports `wer_base` and `wer_tuned` on the speaker's personal-test chunks
(short-form, D7's primary protocol), the paired-bootstrap CI and p-value over chunks, and
the S/D/I split for both. The summary adds `improvement_from_insertions`: the share of the
WER improvement that came from fewer insertions. **Above 0.5 the row is flagged
`style_not_speaker`** — the adapter learned the protocol's tidying, not the voice — and does
not count as a personalization gain. This is the rule from `docs/adaptation_plan.md`
§ Verification.

The base-model sanity check is built in: `wer_base` for a speaker should sit within a few
points of their `wer_B` in `src/evaluation/outputs/committees_speaker_performance.csv`. It
will not match exactly — the error map used ivrit.ai's `turbo-ct2` serving checkpoint on a
different hour of the same speaker, and the sessions here are the newest — but a gap above
0.10 means materialization or scoring changed the metric, and nothing downstream is valid.

## What to bring back

`src/training/outputs/results.csv`, `outputs/results/*.json` (the per-cell numbers and
hypotheses), and the adapters under `runs/` (small: LoRA r=8 is a few MB each). The audio
and the shards stay on the box.

## Not yet

- Long-form scoring on whole personal-test recordings (D7's secondary protocol): the chunks
  are on disk, the recordings are not.
- The sharing axis (similar-speaker groups against random ones) and the method axis (full
  fine-tuning, DoRA, IA3): `run_panel.py` takes `--methods`, but no run has used anything but
  LoRA. The site (`--sites encoder`, `decoder_mlp`) and rank (8, 16) axes were varied in runs 2
  and 3; the cross-speaker control (`--control-folds`) has run in every run since run 2.
