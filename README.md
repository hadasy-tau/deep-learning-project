# Toward Personalized Hebrew ASR

Adapting a general-purpose Hebrew ASR model to an individual speaker, on the Knesset
**committees** corpus — ~15,500 h of committee audio with the official protocol
force-aligned to it.

Two models, the comparison the whole project rests on:

| Arm | Model | Role |
|---|---|---|
| **A** | `openai/whisper-large-v3` | General multilingual. Positive control — makes a null result on B interpretable |
| **B** | `ivrit-ai/whisper-large-v3` | Hebrew fine-tune. The real adaptation target |

**Why committees and not plenums.** Arm B was fine-tuned on ~4,700 h of *plenum* audio.
[VoxKnesset](https://huggingface.co/datasets/ivrit-ai/VoxKnesset) is also plenum speech, so
measuring B there measures train-on-test. The committee recordings are not in that training
mix, so they are held out for both arms. The project ran on VoxKnesset through its first
stages and moved off it; nothing here reads it any more.

One caveat survives the move: the *speakers* overlap with B's training data. Recordings are
held out, voices are not.

## Structure

```
src/
  common.py                    normalize_he, read_wav, make_splits, budget_order.
                               Hebrew normalisation is Stage 1's, verbatim -- never re-derive it

  preprocessing/
    speaker_index/             who is speaking, and when. Knesset ODATA roster, protocol
                               parsing, strict name agreement, per-word labelling ->
                               an index of (session, start, end) spans carrying a PersonID
    chunk_corpus/              that index + the audio -> <=30 s chunks, one speaker each

  inference/                   both arms over the chunk corpus. Arm A via HF Inference
                               Providers (deepinfra), arm B via RunPod serverless on
                               ivrit.ai's own worker image. Resumable, cost-bounded

  training/                    materialize.py: the panel's audio from the corpus shards to WAVs
                               (panel_plan.parquet says which chunks, which split);
                               train.py: one cell -> one adapter (speaker, arm, site, method,
                               budget, rank, lr, seed); run_panel.py: cells -> scored results.
                               README.md there is the GPU session, in order
  evaluation/evaluate.py       transcribe, count errors, paired bootstrap
  evaluation/error_map.py      the per-speaker error map over the committees inference:
                               WER/CER and gain per speaker with CIs, subgroup viability and
                               separation, who to adapt
  evaluation/outputs/          committees_*.csv -- that error map (speaker_performance.csv is
                               the VoxKnesset one Stage 1 produced, kept as the reference point)

notebooks/                     explore_committees.ipynb, committees_error_map(_v2).ipynb (the
                               Stage 1 analysis on the committees), training_run3_figures.ipynb
                               and training_run4_figures.ipynb (the paper's Stage 2 figures)
paper/                         the paper: LaTeX source, figures, the compiled PDF.
                               README.md there is the guide for editing it
docs/                          STATUS.md where it all stands, the one document kept current.
                               design.md + design.html: the one-page overview, end to end.
                               The build records: committees_handoff.md, speaker_index_plan.md,
                               chunk_corpus_build.html, inference.md, error_map.md.
                               adaptation_plan.md: the design. The training plans
                               (training_plan_v3.md, training_plan_v4.md) and the runs
                               (training_run2.md, training_run3.md, training_run4.md).
                               docs/archive/ is history, not instructions
cache/                         git-ignored. Secrets (mode 600) read by inference/providers.py
                               and speaker_index/publish.py
```

Each folder is self-contained and imports its siblings flat, so run things by path
(`python src/inference/run.py`) or put the folder on `sys.path`. Anything shared lives in
`src/common.py`, reached by putting `src/` on the path.

**Working with the corpus itself:** read [`docs/committees_handoff.md`](docs/committees_handoff.md)
first: what the corpus is, how to read it, what is known to be wrong with it.

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

**Write — what this repo produces**

| Dataset | Holds |
|---|---|
| [`knesset-asr/knesset-committees-speakers`](https://huggingface.co/datasets/knesset-asr/knesset-committees-speakers) | An **index, not audio**: 5,163,124 rows naming a `(session, start, end)` span, each carrying a verified Knesset `PersonID` and its demographics. 3,345 h of identified MK speech, 268 sitting MKs (330 PersonIDs with the former MKs heard as guests, `label = former_mk`), Knessets 20–25 |
| [`knesset-asr/knesset-committees-chunks`](https://huggingface.co/datasets/knesset-asr/knesset-committees-chunks) | The corpus itself: 1,204,617 chunks of ≤30 s, 3,840.6 h, 330 speakers, 10,905 sessions, 410 parquet shards, FLAC inline |
| [`knesset-asr/knesset-committees-inference`](https://huggingface.co/datasets/knesset-asr/knesset-committees-inference) | Both models' transcriptions of the Stage-1 subset -- 65,990 chunks, 230 h, 267 speakers, 1 h per MK -- beside the protocol reference. `inference.parquet` (one row per chunk, `hypothesis_A`/`hypothesis_B`/`hypothesis_A_auto`), `inference_long.parquet` (per chunk and arm, with timing and errors), `coverage.parquet` (every corpus chunk: what ran on which arm). Validated end to end; corpus WER A 0.417, B 0.324 |

All `knesset-asr` datasets are public and ungated since 2026-10-02, except the run-3 results
(`knesset-committees-v3-results`, private). Whether the audio should stay public is open: it is
cut from `ivrit-ai/knesset-committees`, which is gated under the ivrit.ai licence.

`speaker_id` throughout is the Knesset's official `PersonID` — the same id space as
KnessetCorpus — so the tables join directly.

## State

Where the project stands, what the runs found, what comes next and which earlier claims are
withdrawn: **[`docs/STATUS.md`](docs/STATUS.md)**, the one document kept current. The rest of
`docs/` is reference or dated record; `docs/archive/` is history, not instructions.

Self-checks, no GPU and no network beyond the cached data:

```bash
python src/common.py                                   # Hebrew normalisation, splits
python src/evaluation/evaluate.py                      # scoring, paired bootstrap
python src/evaluation/error_map.py                     # pooling, eta squared, bootstrap CI
python src/inference/providers.py                      # both provider contracts, offline
python src/preprocessing/speaker_index/roster.py       # PersonID uniqueness, active_on
python src/preprocessing/speaker_index/names.py        # cleaning rules, strict agreement
```
