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

notebooks/                     explore_committees.ipynb, speaker_error_map.ipynb (Stage 1 on
                               VoxKnesset), committees_error_map.ipynb (the same on the committees)
docs/                          design.md + design.html are the one-page overview: the
                               whole flow, end to end. Then the build records:
                               committees_handoff.md, speaker_index_plan.md,
                               chunk_corpus_build.html, inference.md, error_map.md, and
                               adaptation_plan.md for what comes next
cache/                         git-ignored. Secrets (mode 600) read by inference/providers.py
                               and speaker_index/publish.py
```

Each folder is self-contained and imports its siblings flat, so run things by path
(`python src/inference/run.py`) or put the folder on `sys.path`. Anything shared lives in
`src/common.py`, reached by putting `src/` on the path.

**Read [`docs/committees_handoff.md`](docs/committees_handoff.md) first.** It is the
orientation document: what the corpus is, how to read it, what is known to be wrong with it.

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
| [`Dolevabudi/knesset-committees-speakers`](https://huggingface.co/datasets/Dolevabudi/knesset-committees-speakers) | An **index, not audio**: 5,158,763 rows naming a `(session, start, end)` span, each carrying a verified Knesset `PersonID` and its demographics. 3,345 h of identified MK speech, 268 speakers, Knessets 20–25 |
| `Hadasy/knesset-committees-chunks` | **Private.** The corpus itself: ~1.2 M chunks of ≤30 s, 330 speakers, 410 parquet shards, FLAC inline. Exact totals in `docs/committees_handoff.md` |
| [`Dolevabudi/knesset-committees-inference`](https://huggingface.co/datasets/Dolevabudi/knesset-committees-inference) | **Private.** Both models' transcriptions of the Stage-1 subset -- 65,990 chunks, 230 h, 267 speakers, 1 h per MK -- beside the protocol reference. `inference.parquet` (one row per chunk, `hypothesis_A`/`hypothesis_B`/`hypothesis_A_auto`), `inference_long.parquet` (per chunk and arm, with timing and errors), `coverage.parquet` (every corpus chunk: what ran on which arm). Validated end to end; corpus WER A 0.417, B 0.324 |

`speaker_id` throughout is the Knesset's official `PersonID` — the same id space as
KnessetCorpus — so the tables join directly.

## State

- **Speaker index** — built and published. Text validation passed with zero cross-person
  errors over 55.4 h held out. Its audio gate (`speaker_index/validate_audio.py`) is written
  but has never run; it needs a GPU, and it is the check that would catch both text sources
  copying the same wrong speaker header.
- **Chunk corpus** — built and uploaded. Nothing was filtered at build time, deliberately:
  filtering is the consumer's decision. Filter on `quality` (≥ 0.7 is the recommendation).
- **Inference** — done. Both arms verified against a fixed 10-chunk sample, then run over a
  1 h/speaker subset (65,990 chunks, pinned in `src/inference/subset_stage1.parquet`):
  every subset chunk has both hypotheses, the 22 acceptance checks of
  `src/inference/validate_final.py` pass, and the result is published as
  `Dolevabudi/knesset-committees-inference`. Corpus WER A 0.417, B 0.324. Arm A is run with
  the language forced to Hebrew, as B always was; its first, auto-detect run is kept as
  `hypothesis_A_auto` because Whisper mis-detected 6.7 % of chunks (mostly under 3 s).
- **Error map** — done (`docs/error_map.md`). After the quality ≥ 0.7 filter: 58,180 chunks,
  217 h, 267 speakers; corpus WER A 0.387, B 0.292. Every speaker is helped by the fine-tune,
  median 24 % of A's error removed (52 % on the plenums). Corpus hours do not predict WER; the
  share of a speaker's chunks the filter removed does (rho ≈ 0.6–0.7), which is the labels
  question. Subgroup rules separate gain (speaking rate strongest, then religion, nationality,
  age) or difficulty (religious orientation, gender), never both — the reverse of VoxKnesset.
  `committees_adaptation_candidates.csv` is the input to the sharing axis. Beyond Stage 1: B's
  insertions are real speech the protocol left out (73 % also heard by A), the digit problem is
  1 % of errors, the per-speaker ranking is reliable (split-half 0.75–0.85), and 2018–19
  sessions and the Finance committee are the hard conditions. Under a protocol-aware count that
  stops charging for added words both models heard, B's advantage is 37 % rather than 24 %, and
  the per-speaker conclusions hold (ranking Spearman 0.96, 35 of 40 candidates the same).
- **Adaptation** — ready for the GPU. The panel is chosen (11 speakers, `docs/adaptation_plan.md`
  § The panel), its audio is planned and materialized (`src/training/materialize.py`: 8,113
  chunks, 28.8 h, session-disjoint by date, ≥ 45 min test / ~15 dev / ≥ 80 train each), the
  driver runs cells to scored results with the paired bootstrap and the S/D/I rule
  (`src/training/run_panel.py`), and the whole path was smoke-tested on CPU with a tiny Whisper.
  Never run on the real models: `src/training/README.md` is the session, in order — the audio
  gate first, then `overfit_check`, then arm B / LoRA / three budgets / three seeds.

Self-checks, no GPU and no network beyond the cached data:

```bash
python src/common.py                                   # Hebrew normalisation, splits
python src/evaluation/evaluate.py                      # scoring, paired bootstrap
python src/evaluation/error_map.py                     # pooling, eta squared, bootstrap CI
python src/inference/providers.py                      # both provider contracts, offline
python src/preprocessing/speaker_index/roster.py       # PersonID uniqueness, active_on
python src/preprocessing/speaker_index/names.py        # cleaning rules, strict agreement
```
