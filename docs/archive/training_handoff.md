# Handoff: training the per-speaker adapters on a GPU box

> **Historical (run 1 and run 2, 2026-09-20). Do not follow it for a new run.** The later plans
> are `docs/training_plan_v3.md` and `docs/training_plan_v4.md`, and they take precedence wherever
> this file differs. Since it was written: the audio gate has run (run 2:
> 5.4 % of the panel's chunks carry another voice), training has run three times
> (`docs/training_run2.md`; run 3 in `docs/training_run3.md`), the panel is 12 speakers on the
> quality ≥ 0.95 data of plan v3, and the project datasets are public except
> `knesset-committees-v3-results`. The sections on the state, the panel, the machine, the session,
> the open problems, what not to do and what to bring back were removed on 2026-10-03 because they
> described that first run and contradicted the current plan (git history has them). What remains:
> the question, the two rules for reading results, the changes made before run 2 (which
> `docs/training_run2.md` refers to by number), and the repo's conventions.

It was written before run 1 for a fresh Claude session on a rented GPU machine, as the *why*
beside the command list in `src/training/README.md`: the decisions taken and the rules for
reading what comes out.

## What the project asks

Does adapting a Hebrew ASR model to one individual speaker help, and for whom? Two models:

| Arm | Model | Role |
|---|---|---|
| **A** | `openai/whisper-large-v3` | general multilingual. The positive control — without it a null result on B is unreadable |
| **B** | `ivrit-ai/whisper-large-v3` | Hebrew fine-tune. The real adaptation target, and the only arm in this first run |

The corpus is Knesset **committee** audio, chosen because arm B was fine-tuned on Knesset
*plenum* audio — measuring it on plenums would be measuring train-on-test. Committee recordings
postdate the model. The speakers overlap with B's training data; the recordings do not.

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
model B adds also appear in model A's transcription of the same chunk** — read at the time as
proof that they were spoken and the protocol dropped them. *(2026-10-03: that reading is
withdrawn, `docs/STATUS.md` § Withdrawn.)* Over half of B's errors are insertions.

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
are forgiven is more likely the voice. *(2026-10-03: withdrawn. Two models agreeing against the protocol is too weak a sign that
the protocol is wrong, and the forgiven count is removed from the code. `docs/STATUS.md` § Withdrawn.)*

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
control. Point 6 stays a write-up item. The session's results and every decision are in `docs/training_run2.md`; what to do next is `docs/archive/training_next.md`.

**Operational, learned the hard way.** This pod has no persistent volume, and RunPod can
preempt a pod on its own: a preemption after 30 hours loses everything. Run a background loop
that copies `src/training/outputs/results/` and `src/training/runs/` to a private HuggingFace
dataset every 30 minutes; they are megabytes. And any HuggingFace token that was pasted into a
chat is in a transcript — revoke it and issue a read-only replacement, given through
`hf auth login` typed interactively.

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
