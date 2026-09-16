# Adaptation — the runbook

What to do on the GPU box, in order, and what to bring back. The design is
`docs/adaptation_plan.md`; the panel is `src/evaluation/outputs/committees_panel.csv`;
the audio it needs is described by `panel_plan.parquet` (one row per chunk: who, which
session, which split, which shard).

| file | holds |
|---|---|
| `materialize.py` | `plan` (laptop): which chunks, which split — writes `panel_plan.parquet`. `extract` (any fast link): shards → WAVs under `outputs/panel_audio/`. `verify`, `upload`, `download` |
| `panel_plan.parquet` | 8,113 chunks, 28.8 h, 11 speakers, 111 shards; per speaker ≥ 45 min test (the newest sessions), ~15 min dev, ≥ 80 min train, session-disjoint by date, train ordered latest-first so budgets nest |
| `train.py` | one cell → one adapter (`train_cell`), `overfit_check` |
| `run_panel.py` | cells → adapters → scored results (`outputs/results/*.json`, `outputs/results.csv`) |
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

# 4. the first experiment: arm B, LoRA at both sites, budgets 5 / 20 / 80 min, three seeds (99 cells)
python src/training/run_panel.py --arm B --budgets 5 20 80 --seeds 0 1 2
python src/training/run_panel.py --summary            # outputs/results.csv
```

`run_panel.py` is idempotent: a cell with a result JSON is skipped, a cell with a finished
adapter is not retrained, and the base model's transcription of each speaker's test set is
cached once. Kill it and rerun it freely.

## Reading the results

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
