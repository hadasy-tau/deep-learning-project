# The second training run: log and decisions

> **Note (2026-10-03): the forgiven-shared count is withdrawn.** Every "forgiven" or "protocol-aware" number below assumed that a word two models produce and the protocol lacks was spoken; on human-corrected committee clips only about half were (`docs/personalization_research.md` § 1.5). They are kept as a record of what was measured, not as evidence. The standard (protocol) counts stand.

Written on the GPU box on 2026-09-20, as the run went. `docs/archive/training_handoff.md` is the
plan and the rules for reading results; this file is what was decided on the box, why, and
what came out. Everything here was measured on one RunPod A100 SXM 80 GB pod (EU-RO-1).

## Timeline

| time (UTC) | what |
|---|---|
| 10:19 | panel audio download started (3.3 GB, 8,113 WAVs); model weights prefetched |
| 10:20 | `/workspace` hit "disk quota exceeded" at ~7.5 GB — see § The box |
| 10:41 | `materialize.py verify`: all 45 split checks and 8,113 WAVs pass |
| 10:43 | `overfit_check`: loss 0.58 → 0.0005 in 60 steps, 0.5 min |
| 10:44 | profiled cell 30831 / 20 min: 3.1 min end to end (first run: ~30 min a cell) |
| 10:48 | seed-0 run launched, detached: 33 cells + the control in two folds |
| 11:18 | follow-ups queued to start when the seed-0 run ends: encoder-only site, then lr 1e-4 / 3e-4 |

Progress per cell is in `src/training/outputs/run_seed0.log` and `run_followups.log`; every
30 minutes `src/training/backup.py` mirrors results and adapters to the private dataset
`knesset-asr/knesset-committees-adapters` (public since 2026-10-02).

## The box

`/workspace` (the persistent volume) reports petabytes free and refuses writes past about
10 GB. Probed by writing 1 GB files until failure. Consequences:

- Model weights and the HuggingFace xet chunk cache live on the container disk (100 GB, not
  persistent, re-downloadable in minutes): `/workspace/env.sh` sets `HF_HUB_CACHE` and
  `HF_XET_CACHE` to `/root/hfcache` and is sourced from `~/.bashrc`. The token stays under
  `HF_HOME` on the volume.
- Audio, results and adapters stay in the repo on the volume. Trainer checkpoints are deleted
  once the best epoch is restored (they carried optimizer state, ~3× the 21 MB adapter each,
  ~7 GB over a run).
- The RunPod API key in the environment is unauthorized for pod queries, so the hourly rate
  could not be read; the projections below assume $1.9–2.7/h.

## Why the first run took 30 minutes a cell

Profiled phase by phase before any cell was trained (`docs/archive/training_handoff.md` § Before the
second run, point 1). In order of weight:

1. **BLAS thread oversubscription.** Whisper's log-mel extraction is a numpy STFT and a mel
   matmul. With the default pool (one thread per core, 255 on this box) it took **1.4 s per
   chunk**; with four threads, 8 ms; on the GPU, 10 ms for a batch of eight. Training called it
   once per example per epoch and the dev eval once per dev chunk per epoch, so an 80-minute
   cell spent about an hour in it. Fix: `train.py` and `run_panel.py` cap `OMP_NUM_THREADS`
   (and MKL/OpenBLAS) at 8 before numpy loads; `evaluate.transcribe_short` computes the log-mel
   on the device; the training dataloader gets four workers.
2. **Gradient checkpointing and fp32-cast weights.** transformers 5 loads the checkpoint at
   fp32 by default, and the LoRA path enabled checkpointing unconditionally. On an 80 GB card
   neither is needed: base weights in bf16 (PEFT keeps the adapter in fp32) without
   checkpointing take **518 ms a step** at batch 8 and 21 GB peak, against 823 ms. Full
   fine-tuning keeps fp32 weights and checkpointing; cards under 40 GB keep checkpointing.
3. **Generation batch.** The 224-chunk test set took 138 s at batch 8, 42 s at 32, 24 s at 64.
   The run uses `--eval-batch 64`; 32 is the default for a 24 GB card.

Measured after the fixes: 5 / 20 / 80-minute cells take about 2.7–3.7 / 2.2–2.8 / 4.6–6.3
minutes including the once-per-speaker base transcriptions. Projection for the seed-0 run:
about 2.5 h, $5–7.

## Decisions

**D-run2-1: forgiven-shared WER in every cell.** The training target is the cleaned
protocol, so the loss rewards dropping filler the way a stenographer does. Each cell now also
scores base and tuned under the protocol-aware count (`error_analysis.forgiven_counts`,
reused, not re-derived): an inserted word that arm A also produced at that chunk is not
charged. Arm A's transcription of each speaker's test set is cached once beside arm B's
(`outputs/results/base_A_*`). Columns `wer_base_f`, `wer_tuned_f`, `delta_rel_f`, `ci_*_f`,
`p_boot_f`, and the S/D/I/Ish shares. Cost: about 35 s per speaker, once.

**D-run2-2: the cross-speaker control in two folds, not one adapter.** The handoff asked for
one adapter trained on 80 minutes pooled from the other speakers and evaluated on all eleven.
A single adapter cannot be clean for all eleven: either it saw each speaker (~7 minutes of
them, one eleventh of the pool) or it excludes one and cannot be evaluated on the rest. Two
folds (6 and 5 speakers, a fixed shuffle) cost one extra ~5-minute training run and leave no
speaker evaluated on an adapter that heard them. The pool is round-robin over each speaker's
own budget ordering, so every speaker contributes 13–22 minutes and no voice dominates. The
summary joins it per speaker as `personalization_abs = delta_abs − control_delta_abs`. Only the
80-minute rows are budget-matched; the 5 and 20-minute rows read as "less audio of the
speaker versus 80 minutes of everyone else". Each sweep recipe (site, learning rate) gets its
own control, so the comparison is always like for like.

**D-run2-3: passes fixed at 8, steps reported.** Point 4 of the handoff. The alternative
(fixed steps, small budgets cycled more) was not taken because early stopping on dev loss
already decides how much optimisation each cell gets — see the observation below.
`train_steps` and `best_eval_loss` are in every result row, `train_meta.json` in every run
directory holds the per-epoch dev-loss curve.

**D-run2-4: one seed.** `--seeds 0` (33 cells). Seeds 1 and 2 only if seed 0 leaves an
effect in doubt; the first cells show large, significant effects in both directions.

**D-run2-5: Rule 1 read with the scorer verified.** Speaker 30685 came in at base WER 0.233
against the error map's 0.344, a gap of 0.111, just over the hard-abort line. Before continuing:
our `evaluate.score` was run on the error map's own arm-B hypotheses for all eleven speakers and
reproduced the published WER to three decimals (23558: 0.401 vs 0.390, the quality-filter
boundary). So the metric is intact. The error map's hour is one chunk per session across
2019–2025 with per-session WER ranging from 0 to above 1; our test set is the newest sessions
only. A 0.11 gap in the easier direction is session sampling. The run continued. Four speakers
(30752, 30813, 30843, 30868) have 46 test chunks that were also in the inference hour; a
chunk-level comparison on those is the stronger version of this check. First result, 30752:
14 shared chunks, reference text identical in the plan and the inference table, WER 0.217 for
the error map's turbo-ct2 hypotheses against 0.249 for ours on the same audio, 2 of 14
hypotheses identical. A 0.03 gap on identical chunks is the checkpoint seam (D-run2-6), not
materialization or scoring. 30813 (speaker-level gap 0.126, base 0.161 against 0.287): 14
shared chunks, identical references, 0.088 for the error map's hypotheses against 0.076 for
ours — the same audio scores the same; her three test sessions (June 2022) are easier than
her hour's average. 30843: 8 shared chunks, 0.330 against 0.330. 30868: 10 shared chunks,
0.281 against 0.286. On the 46 chunks the two runs share, the error map's checkpoint and ours
score the same within 0.03; every speaker-level gap is which sessions the test set holds.

**D-run2-6: the checkpoint seam, stated.** The error map's arm B was ivrit.ai's `turbo-ct2`
serving checkpoint (4 decoder layers); the adapters train on the full `ivrit-ai/whisper-large-v3`
(32 decoder layers), revision `766847c9`. Base WER here and the error map's WER_B are not on
one scale; the adaptation gain is measured against this run's own base, on the same chunks.

## What the cells showed while running

**The objective learns the stenographer, and the forgiven count sees it.** The first
profiled cell (30831, 20 min): standard WER 0.274 → 0.241 (−12 %, p 0.001) while forgiven
WER 0.145 → 0.169 (+16 %, p < 0.001); the deletion share of errors went from 12 % to 33 % and
insertions from 57 % to 36 %. The adapter learned to omit. `style_not_speaker` fired. The same
pattern in 30685 at 20 and 80 minutes. Not every speaker: 23558 gains 26–36 % standard and
8–31 % forgiven, both significant at 20 and 80 minutes; 30701 gains 25 % standard, 14–18 %
forgiven (not significant).

**Early stopping restores epoch 1 or 2 in every 20 and 80-minute cell.** Dev loss bottoms
after the first or second pass and climbs to the end (30685 / 80 min: 0.331 → 0.569). At
lr 1e-3 the recipe overshoots; each 80-minute adapter is effectively one pass. Hence the two
follow-ups.

## Follow-ups (queued to run after the seed-0 run, same detached pattern)

1. **The site axis: encoder-only LoRA at 80 minutes, all speakers, with its own two-fold
   control.** The decoder is where a model learns to drop filler; the encoder can adapt to a
   voice but has far less room to learn omission. If the encoder-only adapter's gain survives
   the forgiven count where the both-sites adapter's does not, the split is acoustic versus
   lexical, which is what D4's site axis was for. About 11 × 5 min + 15 min.
2. **The learning-rate axis: 1e-4 and 3e-4 at 80 minutes, all speakers, each with its
   control.** Tells whether the 80-minute adapters trail the 20-minute ones (30685) because of
   optimisation rather than data, and settles the passes-versus-steps question empirically.
   About 22 × 5 min + 30 min.

Not done, deliberately: selecting the epoch by dev WER (or dev forgiven WER) instead of dev
loss. It changes what every cell means and belongs in a run of its own.

## Results: the seed-0 run (finished 12:55 UTC, 33 cells + 11 control evaluations, 2.1 h GPU)

`src/training/outputs/results.csv`, one row per cell and per control evaluation; the per-cell
hypotheses beside them. Rule 1 holds for all eleven speakers (§ D-run2-5). Medians over the
eleven speakers, relative change in WER (positive = better), with the number of speakers
whose paired bootstrap is significant at p < 0.05:

| budget | standard WER | forgiven-shared WER | worse under forgiven | `style_not_speaker` | own − control (median) |
|---|---|---|---|---|---|
| 5 min | +6 % (4/11 sig) | −5 % (4/11 sig, all worse) | 9/11 | 9/11 | −9 pts |
| 20 min | +12 % (6/11) | −2 % | 6/11 | 10/11 | −5 pts |
| 80 min | +17 % (8/11) | +5 % (3/11) | 5/11 | 9/11 | +2 pts |
| control (80 min of others) | +15 % (8/11) | +2 % | 6/11 | — | — |

**1. Standard WER falls, and the fall is mostly not the voice.** The 80-minute personal
adapters remove a median 17 % of the base error, significant for eight speakers. An adapter
trained on 80 minutes of *other* panel members removes 15 %, significant for eight. The
personalization effect, own minus control on the same test chunks, has a median of +2 points
at 80 minutes and is negative at 5 and 20. Reading speaker by speaker at 80 minutes: the own
adapter beats the control by 5–9 points for 23558, 30752, 30843, by 2–4 for 30831 and 30868,
matches it for 30859, and trails it for 30701, 30777, 30718 and 30685 (−18: the own adapter
hurts this speaker, the control helps). 30813 is the odd case where the control hurts (−22 %,
not significant) and the own adapter is flat. What "80 minutes of committee Hebrew" teaches,
any committee member's audio teaches.

**2. Under the protocol-aware count the gain largely disappears.** Forgiven-shared WER, which
does not charge an inserted word that arm A also heard, moves by a median of +5 % at 80
minutes (three speakers significant) and is *worse* for five of eleven. At 5 minutes it is
worse for nine of eleven, four of them significantly. `style_not_speaker` fires for nine or
ten of eleven cells at every budget: the adapters learn the stenographer's omissions first.
Insertion share of the tuned model's errors falls at every budget while deletions rise.
The three speakers whose gain survives the forgiven count at 80 minutes are 23558 (+31 %,
p < 0.001), 30752 (+8 %, p 0.046) and 30701 (+21 %, p 0.069); 23558 is the one speaker with
a large, significant gain on both counts at every budget — and his control also gains 21 %
forgiven, so most of his gain is domain too.

**3. Runaway decodes.** The base produces 23 runaway transcriptions over the eleven test sets;
the 80-minute adapters produce 12. A runaway is charged as dozens of insertions, so part of the
standard-WER gain is the adapter suppressing loops, which the forgiven count also forgives when
arm A looped alike.

**4. Two low-WER speakers are hurt by their own 80 minutes.** 30685 (−7 % standard, −54 %
forgiven) and 30813 (−9 %, −34 %); both had base WER well below their error-map value because
their newest sessions are easy. Their dev loss curves bottom at epoch 1 and the best checkpoint
still degrades the test set: at lr 1e-3 even one pass over-fits the protocol register of the
training sessions. This is what the learning-rate follow-up tests.

**5. The budget curve is real but shallow.** Standard WER improves monotonically with budget
for eight speakers; at 5 minutes half the cells are noise (16–40 steps). Under the forgiven
count only the 80-minute rows show a positive median.

**Reading.** On this evidence, LoRA on q/v at both sites with the protocol as target is a
committee-domain adapter that also learns to write like the stenographer; the personal
component is small and, for most speakers, inside the bootstrap noise. The design's D3
control was the thing that made this readable — without it every 80-minute row would have
been reported as a 17 % personalization gain. The follow-ups ask whether the site or the
learning rate changes the picture.

## Results: follow-up 1, the site axis (finished 14:05 UTC, 11 cells + 11 control evaluations, 1.2 h)

LoRA on the encoder's q/v only, 80 minutes, lr 1e-3, against the both-sites cells above.
Medians over eleven speakers:

| site | standard WER | forgiven-shared | worse under forgiven | style flag | own − control | control, standard / forgiven |
|---|---|---|---|---|---|---|
| both (seed-0 run) | +17 % (8/11 sig) | +5 % (3/11) | 5/11 | 9/11 | +2 pts (forgiven +2) | +15 % / +2 % |
| encoder only | +16 % (7/11 sig) | +4 % (3/11) | 5/11 | 10/11 | +3 pts (forgiven +1) | +12 % / +3 % |

**The site does not change the picture.** The encoder alone recovers essentially the whole
both-sites gain on the standard count and the same small, mostly non-significant gain under
the forgiven count; its control, trained on other speakers, gains as much again. So the
"stenographer" effect is not something the decoder learns on its own: an encoder-side
adapter reaches it too, through the cross-attention it feeds. The hypothesis that
encoder-only would isolate an acoustic gain is not supported.

What did change, speaker by speaker: 30685 goes from −7 % / −54 % (both sites) to +16 % /
0 % — the encoder-only adapter does not damage her; 30752 goes the other way (+18 % / +8 % to
−6 % / −25 %, both non-significant, one runaway); 30843's forgiven gain becomes significant
(+9 %, p 0.002). Runaways: 9 against 12. Elsewhere the two sites agree within a few points.
The one robust personal effect stays 23558 (+32 % / +23 %, both significant, control +29 % /
+23 %: again mostly domain).

**A note on single runaway chunks.** 30752's two losing cells (encoder-only at 1e-3, both
sites at 1e-4) are one chunk each: a 7-word reference where the tuned model looped for about
200 errors. Without that chunk both cells are gains of the same size as his other cells
(595 and 578 errors against the base's 792). The paired bootstrap already prices this in —
that is why those rows have p ≈ 0.8 — and `runaway` in the S/D/I block counts them. Read a
large, non-significant loss as "one loop", not as damage to the speaker.

## Results: follow-up 2, the learning-rate axis (finished 16:42 UTC, 22 cells + 22 control evaluations, 2.8 h)

Both sites, 80 minutes, 8 passes, best epoch restored by dev loss; medians over eleven speakers:

| lr | standard WER | forgiven-shared | worse under forgiven | control std / forgiven | own − control std / forgiven | best epoch | runaways |
|---|---|---|---|---|---|---|---|
| 1e-3 (seed-0 run) | +17.2 % (8/11 sig) | +4.7 % (3/11 sig) | 5/11 | +15.0 % / +1.6 % | +2.2 / +2.0 pts | 1 for all | 12 |
| 3e-4 | +17.7 % (9/11 sig) | **+11.6 %** (2/11 sig) | 4/11 | +12.5 % / +4.3 % | **+3.6 / +2.6 pts** | 2 for all | 10 |
| 1e-4 | +16.7 % (8/11 sig) | +5.5 % (2/11 sig) | 4/11 | +14.0 % / +1.8 % | +1.8 / +3.2 pts | 3–5 | 9 |

**The rate changes how much of the gain survives the forgiven count, not the standard
gain.** All three rates remove about 17 % of standard WER. At 3e-4 the forgiven median more
than doubles (4.7 → 11.6 %), the two speakers the high rate damaged recover (30685: −7 % → +15 %
standard, −54 % → −4 % forgiven), and the best epoch is the second for every speaker. At
1e-4 the curve bottoms at epochs 3–5 and the forgiven gain is back near the 1e-3 level: too
little optimisation now. **3e-4 is the recipe's rate from here on.**

**The control moves less than the personal adapters.** The 3e-4 control removes 12.5 % standard
and 4.3 % forgiven; own minus control widens to +3.6 / +2.6 points. Still small, still mostly
domain, but the direction is consistent: the lower rate lets the personal adapter keep more of
the voice-specific part and less of the stenographer.

**Two speakers are rate-independent.** 30813 is worse at every rate and site (and the control
hurts her too: −18 to −22 %), so her problem is not optimisation — the audio gate, running as
this was written, is the check. 30752's 3e-4 and 1e-4 cells are the single-loop artefact
described above.

**On passes versus steps (handoff point 4), now answered empirically.** With 8 passes and
dev-loss selection, the effective training is 1 pass at 1e-3, 2 at 3e-4, 3–5 at 1e-4. Fixing
the step count would not have removed the confound; the selection criterion is what sets the
optimisation each cell gets, and the follow-up on selection by dev forgiven WER (next section)
is where that is addressed.

## The audio gate, run for the first time (17:20 UTC)

`validate_audio.py` with a `--speakers` focus (the eleven panel members at 40 segments each,
plus 60 random other MK speakers at 12 as the reference population; 1,127 segments, ECAPA
embeddings, ~1 h with the GPU shared). Output: `src/preprocessing/speaker_index/outputs/
audio_check_panel.parquet`, `audio_report_panel.txt`.

**Corpus level: 4.6 % of segments are closer to another speaker's centroid than their own**
(52 of 1,127; the reference population alone 3.8 %), against the gate's 0.5 % design
threshold. Committee cross-talk is real and the protocol's floor-holder label is wrong for a
few percent of spans. That is a property of the corpus, not of the panel.

**Panel level.** Flags per speaker (of 31–40 sampled): 23558 7, 30859 5, 30843 4, 30701 3,
30718 / 30752 / 30831 2, and **zero for 30685, 30777, 30813 and 30868**. The flagged
segments have own-similarity near 0 — a different voice entirely, not a noisy one.

- **30813 is clean.** 18 of her 40 sampled segments fall in the panel's sessions, none flagged,
  and her own-centroid similarity (0.91) is the tightest of the panel. Her being hurt by every
  adapter is not mislabelling. The targets table above offers the alternative: her protocol is
  the most condensed of the panel (17 words put back per 100), so her adapters had the most
  stenographer to learn.
- **30843's dev set has another voice in it.** Three of the four flagged segments in panel
  sessions are in her dev session 2235355 (own-similarity 0.09–0.13), one in train session
  2232234. Dev is what early stopping and the forgiven-WER selection read; her dev set is also
  the panel's largest (39 min). Her cells stand — the test set is untouched — but her dev
  signal is noisier than the others'.
- **23558's flags are all outside the panel's sessions** (2018–2020 sessions; the panel uses
  2025–2026). His label quality over the corpus is the worst of the eleven, which fits the
  handoff's "the one to watch"; his panel material is not sampled by this pass.

The first pass sampled the longest segments per Knesset, so only 66 of the 425 panel segments
fall in the panel's own sessions. A second pass over the panel's *own* train, dev and test
chunks (15 per part per speaker, plus 40 reference speakers) is running as this is written;
its result is `audio_report_panel_chunks.txt`, read here.

**Second pass, the panel's own chunks (18:20 UTC).** 388 panel chunks (15 per split per
speaker, minus thin Knessets) and 465 reference segments from 40 other MK speakers.
Panel chunks flagged: **21 of 388 (5.4 %)**; reference 3.4 %. Per speaker, flagged of ~36
sampled, by split:

| speaker | flagged | where | reading |
|---|---|---|---|
| 30843 | **8** | 7 in dev session 2235355, 1 train | her dev set is substantially another voice (own-similarity 0.15–0.19, nearest centroid 30808, the alternate who chaired): early stopping and the forgiven-WER selection read a noisy signal for her |
| 30777 | 3 | 2 test, 1 dev | one test chunk is clearly someone else (own 0.06) |
| 30868 | 2 | 2 test | one test chunk is 30859's voice (similarity 0.81 to his centroid, 0.23 to own) |
| 23558, 30752 | 2 each | dev/train, dev/test | marginal (own 0.28–0.49) |
| 30685, 30701, 30831, 30859 | 1 each | mixed | marginal |
| **30718, 30813** | **0** | — | clean |

So the panel's labels are about 95 % right at the chunk level, cross-talk accounts for the
rest, and it is not evenly spread: 30843's dev session is the one materially contaminated
set, and 30868 and 30777 carry a couple of foreign-voice test chunks each (out of 202 and
268), which lowers their measurable ceiling slightly but does not move a corpus-level WER.
30813 is clean on both passes; her results are about her protocol, not her labels. For the
write-up: report the 5.4 % and name 30843's dev set; if her selection-based cells look
erratic, that is why.

## Results: follow-ups 3–5, the objective, the selection criterion, the decoder MLPs

Queued after the sweep (`src/training/box/next_runs.sh`): semi-verbatim targets with
selection on dev forgiven WER; selection alone; LoRA on the decoder's `fc1`/`fc2`. All at
3e-4, 80 minutes, with two-fold controls. Filled in as they finish.

**The targets** (`src/training/targets.py`, `outputs/targets_verbatim.parquet`; both arms'
transcriptions of every train and dev chunk cached under `outputs/results/hyps_*`): the
protocol text with the words arms A and B both produced at the same place put back, nothing
removed. Over the 5,177 train and dev chunks, 58 % change and 10.7 words go back per 100
reference words; per speaker the rate runs from 4 (30859) to 17 (30813) per 100 — and 30813, whose protocol is the most condensed, is the speaker every adapter has hurt so far: her training targets were the most stenographer-like of the panel. The words
put back are the ones a stenographer drops: אני (574), לא (565), זה (478), אז (405), את (284), גם (213), מה (211), אבל (207), כן (195), בעצם (190), אנחנו (178), באמת (162), יש (144), הוא (130).

**Run A: semi-verbatim targets + selection on dev forgiven WER (finished 19:10 UTC, 11 cells +
11 control evaluations, 2.5 h).** Against the best protocol recipe (3e-4, dev-loss selection):

| recipe, 3e-4, 80 min | standard WER | forgiven-shared | worse under forgiven | control std / forgiven | own − control std / forgiven | runaways |
|---|---|---|---|---|---|---|
| protocol targets, dev-loss selection | +17.7 % (9/11 sig) | +11.6 % (2/11 sig) | 4/11 | +12.5 % / +4.3 % | +3.6 / +2.6 pts | 10 |
| semi-verbatim targets, dev-forgiven selection | +7.7 % (6/11 sig) | +6.9 % (**4/11 sig**) | 3/11 | +6.8 % / +2.6 % | **+0.5 / +4.7 pts** | 17 |

**What changed.** The standard gain halves, as it must: a model trained to write what was
said is scored against a protocol that did not. The standard and forgiven gains now coincide
(7.7 vs 6.9 %) where the protocol recipe had a 6-point gap — the omission incentive is gone
from the objective. The forgiven gain is *not* higher than the protocol recipe's median (6.9
vs 11.6 %), but it is significant for four speakers instead of two, no speaker is damaged
(30813: −1 %, from −31 %), and the personal share of it — own minus the control trained the
same way — rises from +2.6 to **+4.7 points**, the largest personalization number of the
session. Per speaker the forgiven personal effect is now positive for 30718 (+15), 30868
(+17), 30843 (+11), 30701 (+10), 23558 (+6), 30831 (+5); negative for 30752 (the loop), 30813,
30777, 30685.

**The cost.** Runaway decodes rise from 10 to 17 over the eleven test sets: the targets contain
repetitions and fillers, and a model trained toward them loops more readily. That is the
decode-hygiene item (archive/training_next.md § D) becoming necessary rather than optional.

**Reading.** The semi-verbatim objective does what it was built for — it removes the
stenographer from what the adapter learns — and it moves the personal component up, but it
does not lift the ceiling: what an 80-minute personal adapter can buy over "80 minutes of
anyone" under an honest count is about five points of relative WER, concentrated in the
same speakers as before. Run B (selection alone) separates how much of this is the targets and
how much the selection criterion.

**Run B: protocol targets + selection on dev forgiven WER (finished 21:35 UTC).** The
separating experiment. Same rate, same targets as the sweep's best recipe; only the
checkpoint rule differs.

| recipe, 3e-4, 80 min | standard WER | forgiven-shared | worse under forgiven | control std / forgiven | own − control std / forgiven |
|---|---|---|---|---|---|
| protocol, dev-loss selection | +17.7 % (9/11 sig) | +11.6 % (2/11 sig) | 4/11 | +12.5 % / +4.3 % | +3.6 / +2.6 pts |
| protocol, dev-forgiven selection | +15.1 % (7/11 sig) | **+2.0 %** (3/11 sig) | 5/11 | +12.8 % / +4.7 % | +3.3 / **−0.7** pts |
| semi-verbatim, dev-forgiven selection | +7.7 % (6/11 sig) | +6.9 % (4/11 sig) | 3/11 | +6.8 % / +2.6 % | +0.5 / +4.7 pts |

**Selecting on dev forgiven WER is worse than selecting on dev loss.** With protocol targets it
drops the forgiven median from 11.6 to 2.0 % and the personal share to zero. The reason is
plain in the epochs it picks — 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3 against the loss's uniform 2 —
and in the dev sets: 15 minutes is 40–120 chunks, too few for a generation-based WER to rank
epochs that differ by a point or two; the loss, averaged over every token, is the steadier
signal. The idea (archive/training_next.md § A2) was sound and is now measured: not at this dev size.

**Therefore run A's gain in the personal component is the targets', and run A was
handicapped by its selection rule.** The clean reading of the semi-verbatim objective needs
the missing cell of the 2 × 2 — semi-verbatim targets with dev-loss selection — which is
queued as run D after the decoder-MLP run.

**Run C: LoRA on the decoder's MLPs (`fc1`/`fc2`), protocol targets, dev-loss selection
(finished 22:35 UTC, 11 cells + 11 control evaluations, 1 h).** The placement the survey
singled out (Müller-Eberstein et al.: decoder MLP > attention for atypical speakers).

| recipe, 3e-4, 80 min | standard WER | forgiven-shared | control std / forgiven | own − control std / forgiven | runaways |
|---|---|---|---|---|---|
| q/v at both sites | +17.7 % (9/11 sig) | +11.6 % (2/11 sig) | +12.5 % / +4.3 % | +3.6 / +2.6 pts | 10 |
| decoder `fc1`/`fc2` | +13.4 % (9/11 sig) | +2.0 % (2/11 sig) | +13.1 % / −3.7 % | +0.8 / +3.3 pts | 7 |

**Not better.** The decoder-MLP adapters remove less standard error, almost nothing under the
forgiven count, and their control is as strong on the standard count as the personal
adapters — the same domain story. The one place the site looked different, 23558 (own +30 %
against control +17 %), does not generalise: 30777's and 30831's controls match or beat
their own adapters. Fewer runaways (7) and cells of three minutes are the only advantages.
Site, then, has now been varied three ways (q/v both, q/v encoder, decoder MLP) with the
same answer: under this objective the adapter site does not separate the voice from the
protocol.

**Run D: semi-verbatim targets with dev-loss selection** — the missing cell — runs last; its
row completes the table in § Conclusion.

**Run D: semi-verbatim targets with dev-loss selection (finished 23:35 UTC, 11 cells + 11
control evaluations, 1.2 h).** The missing cell.

| recipe, 3e-4, 80 min, q/v both sites | standard WER | forgiven-shared | worse under forgiven | control std / forgiven | own − control std / forgiven | runaways |
|---|---|---|---|---|---|---|
| protocol targets, dev-loss selection | +17.7 % (9/11 sig) | +11.6 % (2/11 sig) | 4/11 | +12.5 % / +4.3 % | +3.6 / +2.6 pts | 10 |
| protocol targets, dev-forgiven selection | +15.1 % (7/11) | +2.0 % (3/11) | 5/11 | +12.8 % / +4.7 % | +3.3 / −0.7 pts | 10 |
| semi-verbatim, dev-forgiven selection | +7.7 % (6/11) | +6.9 % (4/11) | 3/11 | +6.8 % / +2.6 % | +0.5 / +4.7 pts | 17 |
| **semi-verbatim, dev-loss selection** | +9.3 % (6/11) | **+11.0 %** (3/11) | **1/11** | +7.4 % / +4.9 % | +2.5 / +2.6 pts | 15 |

**The semi-verbatim objective reaches the same forgiven ceiling as the protocol one and
stops hurting anyone.** Median forgiven gain 11.0 % against 11.6 %; positive for ten of
eleven speakers instead of seven; only one speaker worse (30752's loop) instead of four;
30813 goes from −31 % to **+16 %** and 30859 gets his first significant forgiven gain
(+7.5 %, p < 0.001). Standard and forgiven now agree (9.3 vs 11.0 %), so the number reported
is the number earned. The personal share is unchanged at +2.6 points. What it costs: runaway
decodes, 15 against 10.

## Conclusion (2026-09-20, 23:40 UTC; 242 result rows, 88 control evaluations, ~13 GPU hours)

> Conclusion 2 ("learns the stenographer"), the semi-verbatim clauses of 3 and 4, and the write-up sentence below rested on the forgiven count or on targets built from the same two-model agreement; they are withdrawn as evidence (`docs/personalization_research.md` § 1.5). Conclusion 1, the rest of 3 and 4 (who gains against the protocol; rate, site, budget), 5 and 6 stand.

1. **Adapting to one speaker helps by about 17 % of WER at 80 minutes, and about 15 of
   those 17 points are bought equally well by anyone's committee audio.** The D3 control,
   run for every recipe, is the finding. Personal minus control is +2 to +5 points across
   every recipe tried; it is never zero and never large.
2. **Under a protocol-aware count, the recipe as designed learns the stenographer.** Three
   quarters of the standard gain is fewer insertions of words that were said; the style flag
   fires in nine or ten cells of eleven at every budget. The semi-verbatim objective removes
   that incentive: the standard and forgiven gains coincide, no speaker is damaged, and the
   forgiven gain (11 %) is as large as the best protocol recipe's. The ceiling did not move;
   the honesty of the number did.
3. **For whom.** The personal effect concentrates in the hardest speakers (23558, 30843,
   30701; base WER 0.32–0.44) and in one typical speaker (30868). The S1 speakers, where the
   fine-tune had failed and the most room was expected, show no personal component. The
   low-WER speakers gain the domain effect only, and two of them are hurt by the protocol
   objective and not by the semi-verbatim one.
4. **The recipe.** LoRA r=8 on q/v at both sites, lr 3e-4 (1e-3 overshoots by epoch 1;
   1e-4 undershoots), 8 passes with the best epoch by dev loss (dev forgiven WER is worse at
   15-minute dev sets), semi-verbatim targets. Site does not matter (three placements, one
   answer). Budget matters shallowly; 5 minutes is noise.
5. **The labels.** The audio gate's first run: 5 % of the panel's chunks carry another voice,
   30843's dev session most of all; 30813 and 30718 are clean. Report it; it moves no
   corpus-level number.
6. **Two things to fix before the numbers go in a paper.** Runaway decodes decide individual
   cells (one chunk, ~200 errors) and rise under semi-verbatim targets: decode with Whisper's
   compression-ratio fallback or a repeat-n-gram guard, re-cache both arms, report the
   repeated-n-gram rate. And seed 0 only: seeds 1–2 at 80 minutes for the two objectives
   would put intervals on the "for whom" claim.

The write-up's table is the one above this section; its sentence is: *LoRA with the
protocol as target is a committee-domain adapter that also learns the stenographer's habits;
with a semi-verbatim target it is an honest committee-domain adapter; the personal component
is a few points, concentrated in the speakers the base model finds hardest.*

## How to resume on a fresh pod

The pod was stopped after this session; the network volume (`/workspace`) persists, the
container disk does not. On a new pod attached to the same volume:

```bash
cd /workspace/deep-learning-project && source src/training/box/env.sh     # caches to the container disk, thread cap
hf auth login                                                               # if the token under HF_HOME is gone
python src/training/materialize.py verify                                   # the audio is on the volume; should pass
nohup python src/training/backup.py --repo knesset-asr/knesset-committees-adapters --every 30 &
```

Model weights re-download on first use (~6 GB, minutes). `runs/` (the adapters, 1.3 GB) and
`outputs/results/` are on the volume and mirrored in the backup dataset; `run_panel.py` skips
every scored cell, so re-launching any of this session's commands is a no-op, and a new
sweep (`docs/archive/training_next.md`) only trains what is new. The detached-launch pattern:

```bash
setsid nohup python src/training/run_panel.py <args> > src/training/outputs/<name>.log 2>&1 < /dev/null &
```

Committed with this session: the code changes (`train.py`, `evaluate.py`, `run_panel.py`,
`backup.py`, `box/`), `outputs/results.csv` and `outputs/results/*.json` (force-added; the
folder is otherwise git-ignored), the run logs, this document, and `archive/training_next.md`.
