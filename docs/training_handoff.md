# Handoff: training the per-speaker adapters on a GPU box

Read this before touching anything. It is written for a fresh Claude session on a rented GPU
machine, with no memory of how any of this was built. Everything before this stage is done and
verified; your job is the first training experiment and nothing else.

`src/training/README.md` is the command list. This document is the *why*, the decisions already
taken, and the rules for reading what comes out. Where they disagree, this one wins.

## What the project asks

Does adapting a Hebrew ASR model to one individual speaker help, and for whom? Two models:

| Arm | Model | Role |
|---|---|---|
| **A** | `openai/whisper-large-v3` | general multilingual. The positive control — without it a null result on B is unreadable |
| **B** | `ivrit-ai/whisper-large-v3` | Hebrew fine-tune. The real adaptation target, and the only arm in this first run |

The corpus is Knesset **committee** audio, chosen because arm B was fine-tuned on Knesset
*plenum* audio — measuring it on plenums would be measuring train-on-test. Committee recordings
postdate the model. The speakers overlap with B's training data; the recordings do not.

## Where things stand

Done, published, and not to be redone:

- **Speaker index** — 5.16 M `(session, start, end)` spans carrying a verified Knesset PersonID
  (`Dolevabudi/knesset-committees-speakers`).
- **Chunk corpus** — 1.2 M single-speaker chunks of ≤ 30 s with the protocol text as reference
  (`Hadasy/knesset-committees-chunks`, private, 410 parquet shards).
- **Inference** — both arms over a 1 h/speaker subset: 65,990 chunks, 267 speakers, validated by
  22 acceptance checks (`Dolevabudi/knesset-committees-inference`). Corpus WER A 0.387, B 0.292
  after the quality filter. See `docs/inference.md`.
- **Error map** — per-speaker WER, gain and subgroup analysis over that inference
  (`docs/error_map.md`, `src/evaluation/`). Every speaker is helped by the fine-tune, median 24 %
  of A's error removed.
- **The panel** — 11 speakers chosen for adaptation, with 4 alternates
  (`docs/adaptation_plan.md` § The panel, `src/evaluation/outputs/committees_panel.csv`).
- **The panel's audio** — materialized as WAVs and published as
  `Dolevabudi/knesset-committees-panel` (private, 3.3 GB, 8,113 files). The table that describes
  it is `src/training/panel_plan.parquet`, committed.

Not done — this is you:

- The **audio gate** has never run. Labels are verified against the protocol text, never against
  the voice.
- **Training has never run** on the real models. `train.py`, `evaluate.py` and `run_panel.py` were
  smoke-tested end to end on CPU with `openai/whisper-tiny`, which caught one real bug
  (transformers 5 dropped `warmup_ratio`). Expect the large model to surface more seams.

## The panel

Eleven speakers. `test` is their newest sessions, `train` the older ones, session-disjoint by
date, and `train` is ordered latest-session-first so the nested budgets (5, 20, 80 min) are
prefixes of one ordering. Minutes below are what `panel_plan.parquet` holds.

| speaker_id | profile | name | test | dev | train | WER_B | gain | filter footprint |
|---|---|---|---|---|---|---|---|---|
| 30831 | S1 | אליהו דלל | 45 | 15 | 82 | 0.328 | 14 % | 0.12 |
| 30685 | S1 | אורית פרקש הכהן | 51 | 20 | 82 | 0.344 | 13 % | 0.11 |
| 30701 | S1 | אופיר כץ | 51 | 16 | 81 | 0.345 | 20 % | 0.10 |
| 30843 | S2 | יאסר חוג'יראת | 47 | 39 | 94 | 0.361 | 22 % | 0.09 |
| 23558 | S2 | דוד ביטן | 49 | 16 | 83 | 0.390 | 26 % | **0.15** |
| 30813 | S3 | אבתיסאם מראענה | 61 | 15 | 80 | 0.287 | 15 % | 0.05 |
| 30718 | S3 | עידית סילמן | 83 | 15 | 82 | 0.275 | 16 % | 0.07 |
| 30868 | S3 | יונתן מישרקי | 47 | 18 | 82 | 0.306 | 21 % | 0.10 |
| 30859 | S4 | צביקה פוגל | 60 | 19 | 82 | 0.218 | 26 % | 0.06 |
| 30752 | C | וליד טאהא | 46 | 19 | 81 | 0.228 | 37 % | 0.05 |
| 30777 | C | משה טור פז | 63 | 17 | 86 | 0.199 | 30 % | 0.07 |

Profiles: **S1** the fine-tune failed them (lowest gain — the most room for personalization),
**S2** simply hard, **S3** typical and left behind, **S4** headroom check (low WER — is anything
left?), **C** controls the fine-tune already served well. Alternates, in case the gate removes
someone: 30808 אפרת רייטן (S1), 556 חיים כץ (S2), 30695 יואב סגלוביץ' (S3), 30807 גלעד קריב (S4).

**דוד ביטן (23558) is the one to watch.** His quality-filter footprint is above the pool median,
and under a protocol-aware count he sits at the 75th percentile of benefit rather than the 59th —
part of his apparent difficulty is the protocol, not his voice. If the audio gate flags him,
replace him with חיים כץ and say so.

## The machine

LoRA on `whisper-large-v3` (1.55 B params) fits comfortably in **24 GB with bf16** (L4, A10G).
Full fine-tuning — not in this run — needs 40 GB (A100) with 8-bit Adam. Disk: 10 GB. If you hit
out-of-memory, use `--batch 4 --grad-accum 2` rather than shrinking the model.

Secrets are **not** in the repo (`cache/` is git-ignored, and tokens must never be printed or
committed). The box needs a HuggingFace token with read access to the private datasets, set with
`huggingface-cli login` or `HF_TOKEN`. The RunPod key in the user's `cache/` is scoped to
serverless and returns 403 on pod creation — launch pods from RunPod's web console.

## The session, in order

Full commands are in `src/training/README.md`. The order matters:

1. **Install and fetch the audio.** `materialize.py download --repo Dolevabudi/knesset-committees-panel`
   then `materialize.py verify`. Verify must pass: 45 split checks plus every one of the 8,113
   WAVs present and readable. If it fails, stop and report — do not re-extract from the shards
   (that is 85 GB and was already done).
2. **`overfit_check`** — 20 examples driven to near-zero loss, about 10 minutes. If the loss does
   not fall, nothing downstream is worth running.
3. **One real cell, timed**: `run_panel.py --speakers 30831 --budgets 20 --seeds 0`. Read the
   base-WER check below before doing anything else. Multiply its wall time by 99 to price the
   full run, and tell the user the number before spending it.
4. **The full experiment**: `run_panel.py --arm B --budgets 5 20 80 --seeds 0 1 2` (99 cells),
   then `--summary`.
5. **The audio gate**, ideally started in the background during step 4 — it is independent of
   training and only changes how you *interpret* a speaker's result. It needs `ffmpeg`,
   `speechbrain`, `torchaudio`, gated read access to `ivrit-ai/knesset-committees`, and
   `segments.parquet` (521 MB) from `Dolevabudi/knesset-committees-speakers` placed in
   `src/preprocessing/speaker_index/outputs/`. It range-seeks each span over HTTP rather than
   downloading sessions, so it is bandwidth-cheap but makes thousands of small ffmpeg calls;
   budget two to four hours. Output: `outputs/audio_check.parquet` and `audio_report.txt`.

`run_panel.py` is idempotent — a scored cell is skipped, a finished adapter is not retrained, and
each speaker's base transcription is cached. Kill and rerun it freely.

## How to read the results — the two rules that decide everything

**Rule 1: the base-WER sanity check is a hard abort.** For each speaker, `wer_base` in the cell's
JSON should land within about 0.10 of their `wer_B` in
`src/evaluation/outputs/committees_speaker_performance.csv`:

```
30831: 0.328   30685: 0.344   30701: 0.345   30843: 0.361   23558: 0.390   30813: 0.287
30718: 0.275   30868: 0.306   30859: 0.218   30752: 0.228   30777: 0.198
```

It will not match exactly — the error map used ivrit.ai's `turbo-ct2` serving checkpoint on a
different hour of the same speaker, and these are the newest sessions. A gap above 0.10 means
materialization or scoring changed the metric, and **every number after it is invalid**. Stop and
report rather than continuing.

**Rule 2: a gain made of fewer insertions is not personalization.** This is the most important
thing in this document and the easiest to get wrong.

The reference text is a *cleaned stenographic protocol*, not a verbatim transcript. People say
"I, I mean, you know" and the stenographer writes the tidy version. Both models transcribe what
was actually said, so every extra word counts against them. We measured this: **73 % of the words
model B adds also appear in model A's transcription of the same chunk** — they were spoken, and
the protocol dropped them. Over half of B's errors are insertions.

So an adapter can lower WER simply by learning to omit filler, which is learning the
stenographer's habits, not the speaker's voice. `run_panel.py --summary` computes
`improvement_from_insertions` (the share of the WER improvement that came from fewer insertions)
and flags `style_not_speaker` when it exceeds 0.5. **A flagged cell does not count as a
personalization gain.** Report it as style-learning and say so plainly.

Beyond those: each cell reports a paired bootstrap over test chunks (`ci_lo`, `ci_hi`, `p_boot`)
comparing base and tuned on the *same* chunks. A `delta_rel` whose interval spans zero is not a
result, however large it looks.

## Open problems — know them, do not fix them

- **Labels are verified against the protocol, not the voice.** Committee cross-talk is heavy and
  the protocol records who held the floor, not who was audible. The error map found that how badly
  a speaker's chunks align predicts their WER (rho 0.6–0.7) — which could be acoustics or could be
  mislabelling. The audio gate is the only check that can tell. If one speaker's result looks
  anomalous, suspect the labels before the model.
- **A WER floor exists** that no adaptation can cross, because of the protocol's register. See
  Rule 2.
- **This run is arm B only, personal-only, LoRA only.** The cross-speaker control (does training on
  *anyone's* audio buy the same thing?), the site and method axes, and the sharing axis are all
  designed in `docs/adaptation_plan.md` (decisions D1–D7) and deliberately out of scope here.
  `run_panel.py` accepts `--sites`, `--methods`, `--ranks`, `--lrs` for later.
- **Long-form scoring** (D7's secondary protocol, whole recordings) is not possible here: the
  chunks are on disk, the full recordings are not.

## Conventions in this repo

- Plain functions driven from scripts and notebooks. Every module ends with `# ---- self-checks`
  and `if __name__ == '__main__':` with hard asserts on real measured values. Run them.
- Store error **counts**, never rates; aggregate micro (Σerrors / Σwords). Averaging per-chunk WER
  is badly biased at 30 seconds a chunk.
- `normalize_he` in `src/common.py` is Stage 1's Hebrew normalisation, verbatim. **Never
  re-derive it.** Geresh and gershayim map to a *space*, not nothing — worth ~5.7 % relative WER.
- Comments explain *why* a number is what it is, with the measurement behind it.
- Pin HuggingFace revisions for anything long-running.
- Read tokens with `huggingface_hub.get_token()` or the environment. Never print, echo or commit
  one. Secret-scan every diff before pushing.
- The user decides commits, pushes and PRs. Ask before any of them. Keep PRs small — one commit
  per logical step, not per edit.

## What not to do

- Don't rebuild the corpus, the speaker index, or the inference. They are done and verified.
- Don't re-extract the panel audio from the shards; download the published dataset.
- Don't change `normalize_he`, or the scoring in `evaluate.py`.
- Don't filter training data on alignment `quality` beyond the ≥ 0.7 already applied. Quality comes
  from a Whisper-family aligner, so filtering harder removes exactly the audio the model finds
  difficult — which is the thing under study.
- Don't let a cell's result stand on a WER drop alone. Rules 1 and 2 both have to pass.
- Don't spend the full 99-cell run before reporting the one-cell timing to the user.

## What to bring back

`src/training/outputs/results.csv`, `outputs/results/*.json` (per-cell numbers and hypotheses),
the adapters under `runs/` (a few MB each at LoRA r=8), and the audio gate's
`audio_report.txt`. The WAVs and any downloaded shards stay on the box.
