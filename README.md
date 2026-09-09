# Toward Personalized Hebrew ASR

Adapting a general-purpose Hebrew ASR model to an individual speaker, on
[VoxKnesset](https://huggingface.co/datasets/ivrit-ai/VoxKnesset) — 2,300 h of Knesset
speech from 393 identified speakers.

Two models of identical architecture, throughout:

| Arm | Model | Role |
|---|---|---|
| **A** | `openai/whisper-large-v3` | General multilingual. Positive control — makes a null result on B interpretable |
| **B** | `ivrit-ai/whisper-large-v3` | Hebrew fine-tune. The real adaptation target |

## Structure

```
stage0/                  Step 0 — cheap checks that gate the Stage 2 design
  check_filenames.py       filename is speaker_session_start_end.wav (66,294 segments)
  check_sessions.py        confirms it is a real session key, and finer than `age`
  stage0_gate.py           statistical power: smallest detectable WER change
  explore.ipynb            look up a wav or a speaker, and listen to the audio
  outputs/                 gate curves, required test sizes, eligible speakers

stage1/                  Stage 1 — the speaker-level error map (complete)
  stage1_basic.ipynb       the paired-segment scoring path; defines normalize_he()
  outputs/                 segment_metrics.csv.gz, speaker_error_map.csv, figures/

stage2/                  Stage 2 — personalize to one speaker
  pipeline.py              index -> materialize -> chunk -> split
  train.py                 one cell (speaker, arm, site, method, budget, rank, lr, seed)
  evaluate.py              transcribe, count errors, paired bootstrap
  README.md                run order and design notes

stage3/                  Stage 3 — speaker identity for the committees corpus (complete)
  roster.py                ODATA pull, active_on(date), demographics join
  manifest.py              session inventory, corpus-shard join by protocol filename
  names.py                 cleaning rules, name variants, strict agree(), alias tables
  protocol.py              shard fetch, verbatim placement, per-word char offsets
  label.py                 Path A and Path B word labelling
  segments.py              turn-aware cutting, quality, output schema
  run_pilot.py / run_full.py   pilot slice, then all 11,127 sessions
  validate.py              holdout simulation and regression
  validate_audio.py        Step 4.2 embedding gate (written, unrun — needs a GPU)
  publish.py               upload the index to HF
  plan.md                  the design and the measurements behind it
```

Not in the repo: `~/.claude/plans/c-users-hadas-downloads-final-project-p-rippling-steele.md`
is the Stage 2 design and the source of truth.

## Data

Everything the project reads and writes lives on HuggingFace, except the Knesset's
own [ODATA API](https://knesset.gov.il/Odata/ParliamentInfo.svc/) (public, no key),
which Stage 3 uses for MK service dates.

**Read — the source corpora**

| Dataset | Access | Holds |
|---|---|---|
| [`ivrit-ai/VoxKnesset`](https://huggingface.co/datasets/ivrit-ai/VoxKnesset) | gated | 2,307 h of **plenum** speech, 393 MKs, 2009–2025. Waveforms plus age/gender/birthplace. Stage 1 and 2 run on this |
| [`ivrit-ai/knesset-committees`](https://huggingface.co/datasets/ivrit-ai/knesset-committees) | gated | ~15,500 h of **committee** audio, one `audio.m4a` per session, with the official protocol text force-aligned to it and raw speaker names. Stage 3's audio and text source |
| [`HaifaCLGroup/KnessetCorpus`](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus) | public | Every Knesset protocol split into sentences with a `speaker_id`, plus a 1,137-person MK roster with demographics. Protocols run to 2024-03-26. Stage 3's identity source |

Gated means an accepted licence on the dataset page and an `HF_TOKEN` in the
environment. Read access is enough; Stage 3 needs write only at `publish.py`.

**Read — our own Stage 1 inference dumps**

| Dataset | Access | Holds |
|---|---|---|
| [`Dolevabudi/voxknesset-whisper-large-v3-ct2-inference`](https://huggingface.co/datasets/Dolevabudi/voxknesset-whisper-large-v3-ct2-inference) | public | Arm B transcriptions, and with them `reference_text`, `segments_json` and demographics — this is what `stage2/pipeline.py:load_index()` reads |
| [`Dolevabudi/voxknesset-whisper-large-v3-baseline`](https://huggingface.co/datasets/Dolevabudi/voxknesset-whisper-large-v3-baseline) | public | Arm A transcriptions (no segmentation) |

These public dumps carry every non-audio field, so gated access to VoxKnesset is
needed only at `materialize`, not to index or chunk.

**Write — what Stage 3 produces**

| Dataset | Holds |
|---|---|
| [`Dolevabudi/knesset-committees-speakers`](https://huggingface.co/datasets/Dolevabudi/knesset-committees-speakers) | An **index, not audio**: 5,158,763 rows naming a `(session, start, end)` span of `ivrit-ai/knesset-committees`, each carrying a verified Knesset `PersonID` and its demographics. 3,345 h of identified MK speech from 268 speakers, Knessets 20–25. Plus the roster, alias tables, per-session manifest and match reports it was built from |

`speaker_id` there is the Knesset's official `PersonID` — the same id space as
KnessetCorpus and VoxKnesset — so the three join directly.

## State

Stage 1 is complete. Step 0 is complete — all four checks returned, and two of them
changed the Stage 2 design:

- **Session ids exist**, encoded in the filename, so splits are genuinely
  session-disjoint rather than resting on `age` as a proxy.
- **Personal-test must be sized per speaker.** A flat 45 min resolves a 9% relative
  gain for the worst speaker but only a 58% one for the best — resolution tracks WER,
  because a low-WER speaker has few errors left to remove.

Stage 2's data path (index, chunk, split, materialize) is implemented and verified on
real data. Training and evaluation are written but unrun — they need a GPU.

Stage 3 is built and published. Its text validation passed with zero cross-person
errors over 55.4 h held out; its audio gate (`validate_audio.py`) is written but
unrun, and is the check that would catch both text sources copying the same wrong
speaker header.

Self-checks, no GPU or network required beyond the cached dumps:

```bash
python stage2/pipeline.py     # chunk/split integrity, text conservation, audio bounds
python stage2/evaluate.py     # scoring and paired-bootstrap correctness
python stage3/roster.py       # PersonID uniqueness, active_on, homonym hard-fail
python stage3/names.py        # cleaning rules and the strict-agreement cases
python stage3/segments.py     # piece bounds, output schema, demographics join
```
