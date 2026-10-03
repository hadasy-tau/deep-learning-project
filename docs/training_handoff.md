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
  (`knesset-asr/knesset-committees-speakers`).
- **Chunk corpus** — 1.2 M single-speaker chunks of ≤ 30 s with the protocol text as reference
  (`knesset-asr/knesset-committees-chunks`, private, 410 parquet shards).
- **Inference** — both arms over a 1 h/speaker subset: 65,990 chunks, 267 speakers, validated by
  22 acceptance checks (`knesset-asr/knesset-committees-inference`). Corpus WER A 0.387, B 0.292
  after the quality filter. See `docs/inference.md`.
- **Error map** — per-speaker WER, gain and subgroup analysis over that inference
  (`docs/error_map.md`, `src/evaluation/`). Every speaker is helped by the fine-tune, median 24 %
  of A's error removed.
- **The panel** — 11 speakers chosen for adaptation, with 4 alternates
  (`docs/adaptation_plan.md` § The panel, `src/evaluation/outputs/committees_panel.csv`).
- **The panel's audio** — materialized as WAVs and published as
  `knesset-asr/knesset-committees-panel` (private, 3.3 GB, 8,113 files). The table that describes
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

1. **Install and fetch the audio.** `materialize.py download --repo knesset-asr/knesset-committees-panel`
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
   `segments.parquet` (521 MB) from `knesset-asr/knesset-committees-speakers` placed in
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

## Before the second run — what to change (written 2026-09-20, after the first run started)

The first run launched as designed. Watching it exposed six things. The first three change
what the result *means* and should be done before more GPU hours are spent; the last three are
judgement calls to settle and state. Nothing already trained is wasted: adapters stay valid,
and the driver skips finished cells on a restart.

**1. Profile one cell before anything else.** The box measured about 30 minutes per cell. The
arithmetic says ten: the 80-minute budget is 372 chunks × 8 passes ÷ batch 8 ≈ 370 steps, well
under ten minutes on an A100, and transcribing 223 test chunks is about a minute. Time the
phases separately — model load, training, dev eval per epoch, test transcription, scoring.
Usual suspects: generation not batched or with a large `max_new_tokens`, something in fp32,
the model reloaded more than once per cell. Fix what you find. This could halve the bill.

**2. Score the test set under the protocol-aware count as well.** The training target is the
cleaned protocol, so the loss *rewards* learning to drop filler the way a stenographer does —
precisely the behaviour `style_not_speaker` flags. The flag catches it afterward; the objective
causes it. Mitigation: transcribe each speaker's test chunks with arm A once (cache it like the
base-B transcription), then compute forgiven-shared WER (`src/evaluation/error_analysis.py::
forgiven_counts`, see `docs/error_map.md` § The same map, protocol-aware) for base and tuned
beside standard WER, and put both in `results.csv`. A gain that survives when shared insertions
are forgiven is more likely the voice. *(2026-10-03: withdrawn. On human-corrected committee clips
two models agreeing against the protocol meant the protocol was wrong only about half the time;
the forgiven count is removed from the code. `docs/personalization_research.md` § 1.5.)*

**3. Add the cross-speaker control (design decision D3) to this run.** Without it, "the adapter
helped speaker X" cannot be told from "80 minutes of any committee audio teaches the model
committee Hebrew". Train one adapter on a budget-matched 80 minutes pooled from the *other*
speakers (the ten other panel members, or non-panel speakers from the corpus), evaluate it on
every panel speaker's test set, and report personalization = Δ(own adapter) − Δ(others'
adapter). One training run and eleven evaluations. It is the single addition that makes the
word "personalization" defensible.

**4. Steps versus passes — decide and state it.** Eight passes regardless of budget gives the
80-minute cell 23× the optimisation of the 5-minute cell (≈370 steps against 16), so the
budget curve confounds "more audio" with "more compute". Either fix the step count and cycle
small budgets more often, or keep passes fixed and report steps per cell. Both are defensible;
leaving it implicit is not.

**5. One seed first.** `--seeds 0` is 33 cells. Stopping the 99-cell run and restarting with
one seed loses only the cell in progress. Add seeds 1 and 2 only if seed 0 shows an effect;
they tighten the intervals, they do not change the answer.

**6. State the checkpoint seam.** The error map's B was ivrit.ai's `turbo-ct2` serving
checkpoint (4 decoder layers, faster-whisper); the adapters train on the full
`whisper-large-v3` fine-tune (32 decoder layers). That is why the base-WER check allows a
0.10 gap, and why the adaptation gain and the error-map gain are not on one scale. Say so in
the write-up.

**Status (2026-09-20, second run).** Points 1–3 are done and 4–5 decided: the profile found
BLAS thread oversubscription in the log-mel extraction (1.4 s a chunk) and gradient
checkpointing on an 80 GB card, and a cell now takes 1.5–4.7 minutes (`src/training/README.md`
§ Speed); every cell carries forgiven-shared WER beside standard WER (since withdrawn); the control runs in two
folds (`--control-folds 2`), so no speaker is evaluated on an adapter that saw them; passes
stay fixed at 8 and `train_steps` is reported per cell; the run is `--seeds 0` plus the
control. Point 6 stays a write-up item. The session's results and every decision are in `docs/training_run2.md`; what to do next is `docs/training_next.md`.

**Operational, learned the hard way.** This pod has no persistent volume, and RunPod can
preempt a pod on its own: a preemption after 30 hours loses everything. Run a background loop
that copies `src/training/outputs/results/` and `src/training/runs/` to a private HuggingFace
dataset every 30 minutes; they are megabytes. And any HuggingFace token that was pasted into a
chat is in a transcript — revoke it and issue a read-only replacement, given through
`hf auth login` typed interactively.

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
