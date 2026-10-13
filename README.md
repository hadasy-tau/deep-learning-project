# Toward Personalized Hebrew ASR

Adapting a general-purpose Hebrew ASR model to an individual speaker, on the Knesset
**committees** corpus — ~15,500 h of committee audio with the official protocol
force-aligned to it.

Two models, the comparison the whole project rests on:

| Arm | Model | Role |
|---|---|---|
| **A** | `openai/whisper-large-v3` | General multilingual. Positive control — makes a null result on B interpretable |
| **B** | `ivrit-ai/whisper-large-v3` | Hebrew fine-tune. The real adaptation target. The error map and the published transcriptions use `ivrit-ai/whisper-large-v3-turbo-ct2`, a separate Hebrew fine-tune of `whisper-large-v3-turbo` |

**Why committees and not plenums.** Arm B was fine-tuned on ~4,700 h of *plenum* audio.
[VoxKnesset](https://huggingface.co/datasets/ivrit-ai/VoxKnesset) is also plenum speech, so
measuring B there measures train-on-test. The committee recordings are not in that training
mix, so they are held out for both arms. The project ran on VoxKnesset through its first
stages and moved off it; nothing here reads it any more.

One caveat survives the move: the *speakers* overlap with B's training data. Recordings are
held out, voices are not.

To check the paper's results, go to [Reproduce the paper](#reproduce-the-paper): a laptop,
no GPU and no token, about 15 minutes, most of it a one-time download of the chunk index.

## How the project fits together

Eight steps, each feeding the next. Steps 1, 2, 3 and 6 need the gated source audio, API keys
or a GPU, and their outputs are published on Hugging Face. Steps 4, 7 and 8, and step 5's data
plans, rerun on a laptop from those outputs.

| # | Step | Code | Output |
|---|---|---|---|
| 1 | Speaker index: who speaks when, with a verified Knesset `PersonID` | `src/preprocessing/speaker_index/` | `knesset-asr/knesset-committees-speakers` |
| 2 | Chunk corpus: the audio cut into chunks of up to 30 s, one speaker each | `src/preprocessing/chunk_corpus/` | `knesset-asr/knesset-committees-chunks` |
| 3 | Inference: both arms on 1 h per speaker | `src/inference/` | `knesset-asr/knesset-committees-inference` |
| 4 | Error map: WER and gain per speaker and per group (paper 4.1) | `src/evaluation/`, `notebooks/committees_error_map.ipynb` | `src/evaluation/outputs/` |
| 5 | Panel data: the speakers picked for adaptation, their train, dev and test chunks | `src/training/word_quality.py`, `src/training/materialize.py` | `src/training/panel_*.parquet`, `knesset-asr/knesset-committees-panel-hq` |
| 6 | Training: an own adapter per speaker and budget, and the control | `src/training/` | `knesset-asr/knesset-committees-v3-results`, `-v4-results`, `src/training/outputs/` |
| 7 | Personalization analysis (paper 3.6 and 4.2) | `notebooks/personalization.ipynb`, `notebooks/personalization_beyond_80_minutes.ipynb` | the paper's figures |
| 8 | The paper | `paper/` | `paper/Toward_Personalized_Hebrew_ASR.pdf` |

## Structure

```
src/
  common.py                    normalize_he, read_wav, make_splits, budget_order.
                               Hebrew normalisation is Stage 1's, verbatim -- never re-derive it

  preprocessing/
    speaker_index/             step 1. Who is speaking, and when: the Knesset ODATA roster
                               (roster.py), the sessions (manifest.py), protocol parsing
                               (protocol.py), strict name agreement (names.py), per-word
                               labelling (label.py, Paths A and B) -> an index of
                               (session, start, end) spans carrying a PersonID (segments.py).
                               run_pilot.py and run_full.py build it, validate.py and
                               validate_audio.py check it, publish.py uploads it
    chunk_corpus/              step 2. That index + the audio -> <=30 s chunks, one speaker
                               each (chunks.py). run_build.py builds the corpus, recover_long.py
                               adds the seven sessions longer than 8 h

  inference/                   step 3. Both arms over the chunk corpus. Arm A via HF Inference
                               Providers (deepinfra), arm B via RunPod serverless on ivrit.ai's
                               own worker image (providers.py). run.py and run_lean.py send the
                               chunks, resumable and cost-bounded. merge.py, coverage.py and
                               validate_final.py assemble and check the dataset. data.py reads
                               the corpus and builds the chunk index

  evaluation/evaluate.py       transcribe, count errors, paired bootstrap
  evaluation/error_map.py      step 4. WER/CER and gain per speaker with CIs, subgroup viability
                               and separation, who to adapt
  evaluation/error_analysis.py the rest of the error map: conditions (year, committee, length,
                               rate), error content, insertions, split-half reliability
  evaluation/outputs/          committees_*.csv/json -- the error map's tables.
                               committees_panel.csv is the panel picked from it.
                               speaker_performance.csv is the VoxKnesset table Stage 1 produced,
                               kept as the reference point

  training/                    step 5: word_quality.py (the word-level alignment rule) and
                               materialize.py (the data plans, panel_plan_v2/_v4.parquet and
                               panel_test07/_v4.parquet, and the panel's audio as WAVs).
                               Step 6: train.py (one cell -> one adapter), run_panel.py (cells
                               -> scored results, the control, the summaries), targets.py
                               (verbatim targets), backup.py (results to HF during a run)
  training/box/                the GPU sessions: pod_v3.sh (run 3) and pod_v4.sh (run 4), the
                               rules that make their decisions (v3_decide.py, v4_decide.py), and
                               run 4's audio, voice check and job queue (data_v4.sh, packs_v4.py,
                               gate_v4.py, jobs_v4.py)
  training/outputs/            the run tables the paper reads (git keeps only these)

notebooks/                     committees_error_map.ipynb (step 4), personalization.ipynb (run 3)
                               and personalization_beyond_80_minutes.ipynb (run 4): every figure
                               in the paper. explore_committees.ipynb is a tour of the speaker
                               index, and needs access to the gated source
paper/                         the paper: LaTeX source, figures, the compiled PDF and the
                               same folder zipped for Overleaf
requirements.txt               what reproducing the paper needs
cache/                         git-ignored. Secrets (mode 600) read by inference/providers.py
                               and speaker_index/publish.py
```

Each folder is self-contained and imports its siblings flat, so run things by path
(`python src/inference/run.py`) from the repository root, or put the folder on `sys.path`.
Anything shared lives in `src/common.py`, reached by putting `src/` on the path. Every module
opens with a docstring that says what it does and how to run it.

### Terms used in the code

| Term | Meaning |
|---|---|
| arm A, arm B | the two models above |
| Stage 1 | the per-speaker error map, first on VoxKnesset, then on the committees (step 4). `subset_stage1.parquet` is its 1 h per speaker subset |
| Stage 2 | personalization, the adapters (steps 5 to 7). The paper's `stage2_*` figures |
| Stage 3, Stage 4 | the speaker index (step 1) and the inference (step 3) |
| run 2, run 3, run 4 | the GPU training runs. Run 3 is the main experiment (5, 20 and 80 minutes), run 4 goes up to 1,440 minutes. The paper does not use run 2 |
| plan v2, plan v4 | the data plans of runs 3 and 4. Plan v3 is run 3's training plan, applied by `box/v3_decide.py` |
| clean test, full test | the two test sets: chunks at quality ≥ 0.95 that pass the word rule (`hq`), and the same sessions at quality ≥ 0.7 (`test07`) |
| cell | one adapter: speaker, arm, site, method, budget, rank, learning rate, seed |
| control | an adapter trained on the same amount of audio from other panel speakers. Personalization is the own adapter's gain minus the control's, on the same test chunks |

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
| [`knesset-asr/knesset-committees-panel-hq`](https://huggingface.co/datasets/knesset-asr/knesset-committees-panel-hq) | The panel's audio for runs 3 and 4 as WAVs, the plans that say which run used which clip, and one pack per speaker for the GPU pod |
| [`knesset-asr/knesset-committees-v3-results`](https://huggingface.co/datasets/knesset-asr/knesset-committees-v3-results) | Run 3: one result per cell, the adapters, training curves (`train_meta.json`) and logs |
| [`knesset-asr/knesset-committees-v4-results`](https://huggingface.co/datasets/knesset-asr/knesset-committees-v4-results) | Run 4: the same, plus its queue and its learning-rate decision |
| [`knesset-asr/knesset-committees-panel`](https://huggingface.co/datasets/knesset-asr/knesset-committees-panel), [`-adapters`](https://huggingface.co/datasets/knesset-asr/knesset-committees-adapters) | Run 2's audio and results. The paper does not use them |

All `knesset-asr` datasets are public and ungated, the run results included. Whether the audio
should stay public is open: it is
cut from `ivrit-ai/knesset-committees`, which is gated under the ivrit.ai licence.

`speaker_id` throughout is the Knesset's official `PersonID` — the same id space as
KnessetCorpus — so the tables join directly.

## Reproduce the paper

Every number, table and figure in the paper comes from the published Hugging Face datasets and
the tables in git. None of this needs a GPU or a token. Checked from a fresh clone on
Python 3.12 and 3.14.

1. Install the analysis dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Build the chunk index once, from the repository root. It is the metadata of the corpus's 410
   shards (no audio), about 160 MB and git-ignored, and the error map reads gender, age, quality
   and committee from it. A few minutes to half an hour, depending on the connection:

   ```bash
   python src/inference/data.py --index
   ```

3. Run each notebook top to bottom (Run All in Jupyter, or
   `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`). The first run
   downloads the inference and speaker tables (a few minutes), and after that each notebook takes
   under a minute:

   | notebook | reproduces |
   |---|---|
   | `committees_error_map.ipynb` | Section 4.1 and Appendices A and D.1: per-speaker WER, the gain from the Hebrew fine-tune, reliability, groups |
   | `personalization.ipynb` | Sections 3.6 and 4.2 and Appendix C: own adapter, control and personal margin at 5, 20 and 80 minutes, tuning |
   | `personalization_beyond_80_minutes.ipynb` | Section 4.2 beyond 80 minutes |

   The figures, by the file the paper includes from `paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/`:

   | Figure | file | notebook |
   |---|---|---|
   | 1 | `error_map_hist.png` | `committees_error_map.ipynb` |
   | 2 | `group_rate_band.png` | `committees_error_map.ipynb` |
   | 3 | `stage2_budget_own_control_personalization.png` | `personalization.ipynb` |
   | 4 | `stage2_best_model_per_speaker.png` | `personalization.ipynb` |
   | 5 | `stage2_data_curve_24h.png` | `personalization_beyond_80_minutes.ipynb` |
   | 6 | `error_map_gain.png` | `committees_error_map.ipynb` |
   | 7 | `error_map_reliability.png` | `committees_error_map.ipynb` |
   | 8 | `conditions_by_year.png` | `committees_error_map.ipynb` |
   | 9 | `stage2_wer_vs_minutes_median.png` | `personalization.ipynb` |
   | 10 | `stage2_data_curve_6h.png` | `personalization_beyond_80_minutes.ipynb` |
   | 11 | `group_religion.png`, `group_nationality.png`, `group_age_band.png`, `group_gender.png`, `group_origin.png`, `group_orientation.png` | `committees_error_map.ipynb` |
   | 12 | `stage2_personalization_per_speaker_seeds.png` | `personalization.ipynb` |
   | 13 | `stage2_winning_learning_curves.png` | `personalization.ipynb` |

   The notebooks write Figures 1 and 4 to 10 straight into that folder, each at a fixed size, so
   a rerun leaves the paper's layout as it is. Figures 2, 3, 11, 12 and 13 are drawn in the notebooks, and the paper keeps the
   copies exported earlier. `personalization.ipynb` also checks the tables in git against the
   run's own copy in `knesset-asr/knesset-committees-v3-results` and reads Figure 13's training
   curves from it.

4. Optional, the tables in git. These rewrite `src/evaluation/outputs/` and
   `src/training/panel_*.parquet`, and come out the same as what git holds:

   ```bash
   python src/evaluation/error_map.py --run                 # committees_speaker_performance.csv, _subgroups, _summary.json, ...
   python src/evaluation/error_analysis.py --run            # committees_conditions.csv, _error_content.csv, _checks.json
   python src/training/materialize.py plan-v2               # run 3's data plan
   python src/training/materialize.py plan-test07           # its full test
   python src/training/materialize.py plan-v4               # run 4's data plan
   python src/training/materialize.py plan-test07 --plan src/training/panel_plan_v4.parquet
   ```

5. Build the paper (the commands under [Running](#running)).

Table 1 lists the training settings in `src/training/train.py` and
`src/training/outputs/recipe_v3.json`. Table 2 is `src/training/outputs/capacity_v4.csv`. The
training runs themselves need a GPU (under Running). What they wrote is in
`src/training/outputs/`: `results_v3.csv`, `tuning_v3.csv` and `recipe_v3.json` for run 3,
`results_v4.csv`, `curve_v4.csv` and `capacity_v4.csv` for run 4. The rest of each run (every
cell's result, the adapters, the logs, run 4's `recipe_v4.json`) is in its results dataset.

## Self-checks

Every module ends with self-checks under `if __name__ == '__main__':`. These run with
`requirements.txt` alone, no GPU and no token:

```bash
python src/common.py                                   # Hebrew normalisation, splits, budgets
python src/evaluation/evaluate.py                      # scoring, paired bootstrap, loop guard
python src/evaluation/error_map.py                     # pooling, eta squared, bootstrap CI
python src/evaluation/error_analysis.py                # insertions, conditions, reliability
python src/inference/providers.py                      # both provider contracts, offline
python src/preprocessing/speaker_index/names.py        # cleaning rules, strict agreement
python src/preprocessing/chunk_corpus/chunks.py        # slicing, ids, schema, FLAC round trip
python src/training/targets.py                         # verbatim targets
python src/training/run_panel.py --self-check          # control folds, data curve
python src/training/word_quality.py --check            # the word rule
python src/training/box/v3_decide.py --self-check      # run 3's decision rules
python src/training/box/v4_decide.py --self-check      # run 4's decision rules
python src/training/box/jobs_v4.py --self-check        # run 4's job queue
python src/training/box/gate_v4.py --self-check        # run 4's voice check
```

`src/training/train.py` and `src/training/box/packs_v4.py` need `src/training/requirements.txt`.
`roster.py`, `manifest.py`, `protocol.py`, `segments.py` and `label.py` under `speaker_index/`
check against the live sources (Knesset ODATA, the gated committees dataset).

## Running

The full pipeline, for rebuilding a step rather than checking the paper. Secrets come from the
environment, else from the git-ignored `cache/` (mode 600), and are never written to any output:
`HF_TOKEN` or `hf auth login` (the gated source, and the uploads of steps 1 and 2),
`HF_INFERENCE_TOKEN` (arm A), `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` (arm B), `HF_WRITE_TOKEN`
(the inference upload). The datasets the steps upload to are named in the scripts (a `REPO`
constant, `--repo` or `RESULTS_REPO`), so point them at your own namespace first.

**Speaker index** (step 1). Reads the gated source, KnessetCorpus and Knesset ODATA:

```bash
python src/preprocessing/speaker_index/roster.py       # who was an MK, when, with demographics
python src/preprocessing/speaker_index/manifest.py     # every committee session and its labelling path
python src/preprocessing/speaker_index/run_pilot.py    # a pilot slice, learns the alias tables
python src/preprocessing/speaker_index/validate.py     # how often Path B would put the wrong person on a word
python src/preprocessing/speaker_index/run_full.py     # every session, resumable, then the index
python src/preprocessing/speaker_index/publish.py      # dry run: prints what publish(dry_run=False) uploads
```

**Chunk corpus** (step 2). Needs `ffmpeg`, the gated source and a write token. Resumable, about
40 hours:

```bash
python src/preprocessing/chunk_corpus/run_build.py     # stream every session to the Hub
python src/preprocessing/chunk_corpus/recover_long.py  # the seven sessions over 8 h, as one more shard
```

**Inference** (step 3), both arms over the chunk corpus. First the 10 fixed verification chunks,
then the subset as it was run (resumable: a rerun sends only what has no successful row yet).
`run_lean.py --endpoints` takes your own RunPod endpoint ids:

```bash
pip install -r src/inference/requirements.txt
python src/inference/run.py --arm A --chunk-ids src/inference/verify_chunks.txt
python src/inference/run.py --arm B --chunk-ids src/inference/verify_chunks.txt
python src/inference/verify.py src/inference/outputs/A_*.jsonl src/inference/outputs/B_*.jsonl --expect src/inference/verify_chunks.txt
python src/inference/run_lean.py --workers-a 96 --workers-b 24 --prefetch 2
zsh src/inference/finish.sh          # retry pass, merge, coverage, validate, upload only on pass
```

**Panel data** (step 5). The plans are in git (step 4 of Reproduce rebuilds them).
`word_quality.py` scores the candidate chunks word by word from the gated source, and
`materialize.py` cuts the panel's audio (`extract`, then `verify`). The GPU sessions download
that audio from `knesset-asr/knesset-committees-panel-hq` instead.

**Training** (step 6), on a GPU pod with `src/training/requirements.txt` and `hf auth login`.
Each script runs in stages, and its header comment lists them. Set `RESULTS_REPO` to a dataset
you can write to, since both back up there during the run (the defaults are the project's own
results datasets):

```bash
export RESULTS_REPO=<your-namespace>/<dataset>
bash src/training/box/pod_v3.sh setup  # run 3: deps, weights, checks
bash src/training/box/pod_v3.sh data   # the panel's audio
bash src/training/box/pod_v3.sh backup # results to RESULTS_REPO every 30 min, in the background
bash src/training/box/pod_v3.sh all    # tuning, 5, 20 and 80 minutes, controls, seeds, outputs/results.csv
bash src/training/box/pod_v4.sh all    # run 4: beyond 80 minutes (boot, run, finish)
```

A run writes `src/training/outputs/results.csv` (and `tuning.csv`). The copies in git are named
after the run: `results_v3.csv`, `tuning_v3.csv`, `results_v4.csv`.

**Paper**. After a change to the folder, rebuild the PDF and the Overleaf zip in the same commit:

```bash
cd paper/Toward_Personalized_Hebrew_ASR_overleaf && latexmk -pdf -outdir=/tmp/paper acl_latex.tex && cp /tmp/paper/acl_latex.pdf ../Toward_Personalized_Hebrew_ASR.pdf
cd paper/Toward_Personalized_Hebrew_ASR_overleaf && rm -f ../Toward_Personalized_Hebrew_ASR_overleaf.zip && zip -q -r -X ../Toward_Personalized_Hebrew_ASR_overleaf.zip . -x '.DS_Store'
```

## Licence

The code is under the MIT licence (`LICENSE`). The data keeps its sources' licences: the audio
and the protocol text come from `ivrit-ai/knesset-committees` under the ivrit.ai licence, and
the speaker attribution and demographics from the Knesset Corpus under CC BY-SA 4.0.
