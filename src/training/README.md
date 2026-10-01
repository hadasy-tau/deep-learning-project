# Adaptation — the runbook

> **Current plan: `docs/training_plan_v3.md`** — high-quality data only, for test, validation and
> train (`word_quality.py`, `materialize.py plan-v2`), step-mode training with fixed early stopping,
> lean validation-only tuning (`run_panel.py --tune`, `--tuning-report`) and augmentation. Its
> runbook supersedes the session below, which is how the second run was driven.

What to do on the GPU box, in order, and what to bring back. **Read
`docs/training_handoff.md` first** — it carries the decisions, the panel and the two rules for
reading the results. The design is
`docs/adaptation_plan.md`; the panel is `src/evaluation/outputs/committees_panel.csv`;
the audio it needs is described by `panel_plan.parquet` (one row per chunk: who, which
session, which split, which shard).

| file | holds |
|---|---|
| `materialize.py` | `plan` (laptop): which chunks, which split — writes `panel_plan.parquet`. `extract` (any fast link): shards → WAVs under `outputs/panel_audio/`. `verify`, `upload`, `download` |
| `panel_plan.parquet` | 8,113 chunks, 28.8 h, 11 speakers, 111 shards; per speaker ≥ 45 min test (the newest sessions), ~15 min dev, ≥ 80 min train, session-disjoint by date, train ordered latest-first so budgets nest |
| `train.py` | one cell → one adapter (`train_cell`), `overfit_check` |
| `run_panel.py` | cells → adapters → scored results (`outputs/results/*.json`, `outputs/results.csv`); standard and forgiven-shared WER per cell; `--control-folds K` for the cross-speaker control (D3) |
| `targets.py` | semi-verbatim training targets: the protocol text with the words both base models produced put back (`--build-targets`); `forgiven_score` for training-side selection |
| `word_quality.py` | per-word alignment scores for the candidate train/dev clips (from the raw ivrit.ai sessions) and the word rule `word_ok`; writes `word_quality.parquet` |
| `panel_plan_v2.parquet` | the high-quality plan: test, dev and train all at quality ≥ 0.95 passing the word rule; 30843 replaced by her alternate 556 (`materialize.py plan-v2`) |
| `backup.py` | copies `outputs/results/` and `runs/` (adapters only, no trainer checkpoints) to a private HF dataset on a loop |
| `box/` | the GPU box's `env.sh` (caches on the container disk, thread cap) and the detached run chains of the 2026-09-20 session |
| `requirements.txt` | the stack; pin torch to the box's CUDA build |

## The machine

LoRA on `whisper-large-v3` is comfortable on 24 GB with bf16 (L4, A10G). Full fine-tuning
(the method axis) needs 40 GB (A100) with 8-bit Adam. The first experiment is LoRA only, so
24 GB is enough. Disk: 10 GB for the audio and adapters, plus 2 GB per shard in flight if
you extract on the box. Network: extraction pulls ~85 GB once.

## The session

```bash
git clone https://github.com/hadasy-tau/deep-learning-project.git && cd deep-learning-project
pip install -r src/training/requirements.txt          # torch: use the CUDA wheel for the box
huggingface-cli login                                 # read access to the chunk corpus

# 1. the audio: EITHER fetch the folder the laptop extracted (3 GB) ...
python src/training/materialize.py download --repo Dolevabudi/knesset-committees-panel
# ... OR extract it here (85 GB pass through, ~45 min on a fast link, resumable)
python src/training/materialize.py extract
python src/training/materialize.py verify             # splits + every wav present and readable

# 2. the labels: the audio gate on the panel (ECAPA), never run before
pip install speechbrain torchaudio
python src/preprocessing/speaker_index/validate_audio.py --per-speaker 12 --min-segments 8
#    a panel speaker whose segments fail leaves the panel: swap in the alternate from the panel file

# 3. the training loop learns at all (20 examples to near-zero loss, ~10 min)
python - <<'PY'
import sys, pandas as pd; sys.path.insert(0, 'src/training'); import train
P = pd.read_parquet('src/training/panel_plan.parquet')
train.overfit_check(P, 'src/training/outputs/panel_audio', speaker=30831, arm='B')
PY

# 4. the experiment: arm B, LoRA at both sites, budgets 5 / 20 / 80 min, one seed (33 cells),
#    then the cross-speaker control in two folds; detached, so it survives the shell
nohup python src/training/backup.py --repo <user>/knesset-committees-adapters --every 30 &
setsid nohup python src/training/run_panel.py --arm B --budgets 5 20 80 --seeds 0 --eval-batch 64 \
    --control-folds 2 --control-budget 80 > src/training/outputs/run_seed0.log 2>&1 < /dev/null &
python src/training/run_panel.py --summary            # outputs/results.csv
```

Recipe options added on 2026-09-20 (`docs/training_run2.md` for what each did):
`--lrs 3e-4` (the rate to use; 1e-3 restores epoch 1 everywhere), `--sites decoder_mlp`,
`--targets verbatim` (needs `--transcribe-parts` once, then `--build-targets`),
`--select forgiven` (checkpoint by dev forgiven-shared WER; measured worse than dev loss at
15-minute dev sets), `--control-folds K` (one control per recipe).

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

Every cell carries two counts. `wer_base` / `wer_tuned` are standard WER; `wer_base_f` /
`wer_tuned_f` are forgiven-shared WER (`error_analysis.forgiven_counts`, `docs/error_map.md`
§ The same map, protocol-aware): an inserted word that arm A also produced at that chunk is
not charged, because two models hearing the same absent word is speech the protocol dropped.
Arm A's transcription of each speaker's test set is cached once (`outputs/results/base_A_*`).
A gain that survives the forgiven count is more likely the voice; one that appears only under
the standard count is the adapter learning the stenographer's omissions. In the profiled cell
(30831, 20 min) standard WER fell 12 % while forgiven WER rose 16 % — the deletion share
went from 12 % to 33 % of errors.

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
§ Verification, and `docs/error_map.md` § The same map, protocol-aware is why it exists.

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
- The cross-speaker control (D3): `run_panel.py` trains on the speaker only; the
  budget-matched "trained on others" arm is the next driver.
- The sharing axis, and the method / site / rank axes: `run_panel.py` takes them as arguments
  (`--sites`, `--methods`, `--ranks`, `--lrs`) but the first experiment does not use them.
