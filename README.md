# Toward Personalized Hebrew ASR

Adapting a general-purpose Hebrew ASR model to an individual speaker, on the Knesset
**committees** corpus — ~15,500 h of committee audio with the official protocol
force-aligned to it.

**The question.** Once a population-level Hebrew fine-tune has taken part of the error, is
there anything speaker-specific left — for whom, and how many minutes of one person's voice
does it take to get it?

Two models, the comparison the whole project rests on:

| Arm | Model | Role |
|---|---|---|
| **A** | `openai/whisper-large-v3` | General multilingual. Positive control — makes a null result on B interpretable |
| **B** | `ivrit-ai/whisper-large-v3` | Hebrew fine-tune. The real adaptation target |

Inference over the corpus ran B as `ivrit-ai/whisper-large-v3-turbo-ct2`, the checkpoint
ivrit.ai serve; training adapts `ivrit-ai/whisper-large-v3` (`src/inference/README.md`).

**Why committees and not plenums.** Arm B was fine-tuned on ~4,700 h of *plenum* audio.
[VoxKnesset](https://huggingface.co/datasets/ivrit-ai/VoxKnesset) is also plenum speech, so
measuring B there measures train-on-test. The committee recordings are not in that training
mix, so they are held out for both arms. The project ran on VoxKnesset through its first
stages and moved off it; nothing here reads it any more.

One caveat survives the move: the *speakers* overlap with B's training data. Recordings are
held out, voices are not.

## The pipeline

Each step reads what the one before it published. Results, numbers and what is still open
live in the step's document, not here.

| Step | What it does | Code | Document |
|---|---|---|---|
| Speaker identity | Resolves each protocol speaker name to the Knesset's `PersonID`, exactly or not at all, giving an index of `(session, start, end)` spans | `src/preprocessing/speaker_index/` | [`speaker_index_plan.md`](docs/speaker_index_plan.md) |
| Chunk corpus | Index × audio → chunks of ≤ 30 s, one speaker each, protocol text as reference. Nothing filtered at build time | `src/preprocessing/chunk_corpus/` | [`chunk_corpus_build.html`](docs/chunk_corpus_build.html) |
| Inference | Both arms over a 1 h-per-speaker subset of the corpus | `src/inference/` | [`inference.md`](docs/inference.md), runbook [`src/inference/README.md`](src/inference/README.md) |
| Error map | Per-speaker WER, CER and gain (A − B) with CIs; subgroup rules; who to adapt | `src/evaluation/`, `notebooks/` | [`error_map.md`](docs/error_map.md) |
| Adaptation design | The decisions the training code implements (D1–D7), the panel and how it was chosen | `src/common.py`, `src/evaluation/evaluate.py` | [`adaptation_plan.md`](docs/adaptation_plan.md) |
| Training | One LoRA adapter per speaker and budget, scored against base B and a cross-speaker control | `src/training/` | [`training_run2.md`](docs/training_run2.md) (the second run), [`training_plan_v3.md`](docs/training_plan_v3.md) (the current plan) |

## What to read

- **New to the project:** [`docs/design.md`](docs/design.md) — the whole flow on one page.
- **Current work:** [`docs/training_plan_v3.md`](docs/training_plan_v3.md).
- **Running training on a GPU pod:** [`docs/pod_runbook_v3.md`](docs/pod_runbook_v3.md).
- **Working with the corpus itself:** [`docs/committees_handoff.md`](docs/committees_handoff.md) —
  how to read it and what is known to be wrong with it.

Records of earlier sessions, kept for the reasoning they carry:
[`training_handoff.md`](docs/training_handoff.md) (the setup of the first GPU runs) and
[`training_next.md`](docs/training_next.md) (the options weighed after the second run, which
plan v3 chose from).

## Structure

```
src/
  common.py                    normalize_he, read_wav, make_splits, budget_order.
                               Hebrew normalisation is Stage 1's, verbatim -- never re-derive it

  preprocessing/
    speaker_index/             who is speaking, and when. Knesset ODATA roster, protocol
                               parsing, strict name agreement, per-word labelling ->
                               an index of (session, start, end) spans carrying a PersonID.
                               validate_audio.py is the audio gate: does the voice match the id
    chunk_corpus/              that index + the audio -> <=30 s chunks, one speaker each

  inference/                   both arms over the chunk corpus. Arm A via HF Inference
                               Providers (deepinfra), arm B via RunPod serverless on
                               ivrit.ai's own worker image. Resumable, cost-bounded

  evaluation/evaluate.py       transcribe (with the loop guard), count errors, paired bootstrap
  evaluation/error_map.py      the per-speaker error map: WER/CER and gain per speaker with CIs,
                               subgroup viability and separation, who to adapt
  evaluation/error_analysis.py what the errors are; the protocol-aware ("forgiven") count
  evaluation/outputs/          the error map's tables (committees_*), the panel
                               (committees_panel.csv), and speaker_performance.csv -- the
                               VoxKnesset per-speaker table, kept as the plenum reference point

  training/                    materialize.py: which chunks go in which split (panel_plan*.parquet)
                               and their audio as WAVs; word_quality.py: the word-level label rule;
                               train.py: one cell -> one adapter; run_panel.py: cells -> scored
                               results, with the control. README.md there lists every file
    box/                       the GPU pod: env.sh, pod_v3.sh (plan v3, staged) and v3_decide.py

notebooks/                     explore_committees.ipynb (the corpus with speakers attached);
                               committees_error_map.ipynb (the error map in full) and
                               stage1_error_map.ipynb (its condensed version)
docs/                          one document per step (table above); figures/ holds the
                               notebooks' plots
cache/                         git-ignored. Secrets (mode 600) read by inference/providers.py
                               and speaker_index/publish.py
```

Each folder is self-contained and imports its siblings flat, so run things by path
(`python src/inference/run.py`) or put the folder on `sys.path`. Anything shared lives in
`src/common.py`, reached by putting `src/` on the path.

## Data

Everything the project reads and writes lives on HuggingFace, except the Knesset's own
[ODATA API](https://knesset.gov.il/Odata/ParliamentInfo.svc/) (public, no key), which the
speaker index uses for MK service dates.

**Read — the sources**

| Dataset | Access | Holds |
|---|---|---|
| [`ivrit-ai/knesset-committees`](https://huggingface.co/datasets/ivrit-ai/knesset-committees) | gated | ~15,500 h of committee audio, one `audio.m4a` per session, with the official protocol force-aligned to it and raw speaker names |
| [`HaifaCLGroup/KnessetCorpus`](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus) | public | Every Knesset protocol split into sentences with a `speaker_id`, plus a 1,137-person MK roster with demographics. Runs to 2024-03-26 |

Gated means an accepted licence on the dataset page and an `HF_TOKEN` in the environment.

**Write — what this repo produces**, all in the HF organization
[`knesset-asr`](https://huggingface.co/knesset-asr). The audio in them is cut from the gated
`ivrit-ai/knesset-committees`, whose licence applies.

| Dataset | Holds |
|---|---|
| [`knesset-committees-speakers`](https://huggingface.co/datasets/knesset-asr/knesset-committees-speakers) | An **index, not audio**: 5,158,763 rows naming a `(session, start, end)` span, each carrying a verified Knesset `PersonID` and its demographics. 3,345 h of identified MK speech, 268 speakers, Knessets 20–25 |
| [`knesset-committees-chunks`](https://huggingface.co/datasets/knesset-asr/knesset-committees-chunks) | The corpus itself: ~1.2 M chunks of ≤ 30 s, 330 speakers, 410 parquet shards, FLAC inline. Exact totals in `docs/committees_handoff.md` |
| [`knesset-committees-inference`](https://huggingface.co/datasets/knesset-asr/knesset-committees-inference) | Both models' transcriptions of the 1 h-per-speaker subset (65,990 chunks, 267 speakers) beside the protocol reference: `inference.parquet`, `inference_long.parquet`, `coverage.parquet` |
| [`knesset-committees-panel`](https://huggingface.co/datasets/knesset-asr/knesset-committees-panel) | The adaptation panel's audio as WAVs, plan v1 (`panel_plan.parquet`, quality ≥ 0.7) |
| [`knesset-committees-panel-hq`](https://huggingface.co/datasets/knesset-asr/knesset-committees-panel-hq) | The same for plan v2 (`panel_plan_v2.parquet`, high-quality clips only) |
| [`knesset-committees-adapters`](https://huggingface.co/datasets/knesset-asr/knesset-committees-adapters) | Training runs mirrored from the GPU pod: adapters, per-cell results, logs |

`speaker_id` throughout is the Knesset's official `PersonID` — the same id space as
KnessetCorpus — so the tables join directly.

## Self-checks

No GPU and no network beyond the cached data:

```bash
python src/common.py                                   # Hebrew normalisation, splits
python src/evaluation/evaluate.py                      # scoring, paired bootstrap
python src/evaluation/error_map.py                     # pooling, eta squared, bootstrap CI
python src/inference/providers.py                      # both provider contracts, offline
python src/preprocessing/speaker_index/roster.py       # PersonID uniqueness, active_on
python src/preprocessing/speaker_index/names.py        # cleaning rules, strict agreement
```
