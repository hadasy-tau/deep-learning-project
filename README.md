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
```

Not in the repo: `~/.claude/plans/c-users-hadas-downloads-final-project-p-rippling-steele.md`
is the Stage 2 design and the source of truth.

## Data

| Source | Gated | Holds |
|---|---|---|
| `Dolevabudi/...-ct2-inference` | no | `reference_text`, `segments_json`, demographics — the index |
| `Dolevabudi/...-baseline` | no | Arm A transcriptions (no segmentation) |
| `ivrit-ai/VoxKnesset` | **yes** | **waveforms only** — needs an accepted licence and `HF_TOKEN` |

The public dumps carry every non-audio field, so gated access is needed only at
`materialize`, not to index or chunk.

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

Self-checks, no GPU or network required beyond the cached dumps:

```bash
python stage2/pipeline.py     # chunk/split integrity, text conservation, audio bounds
python stage2/evaluate.py     # scoring and paired-bootstrap correctness
```
