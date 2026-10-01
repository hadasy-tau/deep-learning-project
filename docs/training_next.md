# What to try next: an investigation into improving the adaptation

Written 2026-09-20 at the end of the second run (`docs/training_run2.md`), before the pod was
stopped. It sets out what the runs permit and rule out, what the literature says, and a
ranked plan with costs, so the next session can queue work the way this one did.

## What the evidence fixes

Six facts from `src/training/outputs/results.csv` (66 own cells and 22–44 control
evaluations, seed 0, arm B):

1. **The gain is mostly domain.** At 80 minutes the personal adapters remove a median 17 % of
   base WER; an adapter trained on 80 minutes of *other* panel speakers removes 15 % on the
   same chunks. Own minus control: +2 points median.
2. **The gain is mostly the stenographer.** 34 % of all base errors are insertions that arm A
   also produced at that chunk, i.e. words spoken but not in the protocol. Under the count
   that forgives them the median gain is +5 % and `style_not_speaker` fires in nine or ten
   cells of eleven at every budget. Deletions rise as insertions fall.
3. **Site does not separate acoustic from lexical.** Encoder-only LoRA reproduces the
   both-sites numbers, own and control alike.
4. **Learning rate 1e-3 is too high for this data.** Best epoch is epoch 1 in all thirteen
   80-minute cells at 1e-3, epoch 2 in all nine at 3e-4, epochs 3–5 at 1e-4. Lower rates fix
   the two speakers the high rate damaged and slightly improve the forgiven numbers; they do
   not change the reading.
5. **Where personalization shows, it is in the hardest speakers.** Own beats control by 5–9
   points for 23558 and 30843 (profile S2, base WER 0.44 and 0.32) and 30752; it is zero or
   negative for the low-WER S3/S4/C speakers and for the S1 speakers, where the most room had
   been expected.
6. **Runaway decodes decide individual cells.** One 7-word chunk looping for ~200 errors turns
   a 17 % gain into a 10 % loss with p ≈ 0.8. The bootstrap discounts it; the point estimate
   does not.

What this rules out: more sites, ranks, DoRA/IA3, full fine-tuning — none of them addresses
facts 1 and 2, and fact 3 says the site axis is not where the answer is. What it points at:
the *target*, the *selection criterion*, and the *speakers*.

## Approaches, assessed

### A. Change what the model is trained toward

**A1. Pseudo-verbatim targets.** For each training chunk, transcribe with arms A and B (the
base models, no adapter), align both to the protocol text, and reinsert into the reference the
words both models produced at the same position — the shared-insertion logic of
`error_analysis.forgiven_counts`, applied to the train side. The model then trains toward what
was probably said, not what the stenographer kept. Fact 2 puts the ceiling at a third of the
base error. Cost: arm A and B over the train sets (915 min of audio, ~25 min of GPU), a
~60-line target builder reusing the alignment code, then the 80-minute cells with their
control (~1.5 h). *Risk:* the reinserted words are model output; two correlated models can
agree on a hallucination. Mitigation: only insert where both agree *and* the word is a known
filler/repetition pattern or appears in the speaker's other chunks; report the share of
tokens changed. **Highest expected effect on the omission problem; medium cost.**

**A2. Select checkpoints on dev forgiven WER, not dev loss.** The loss is the quantity that
rewards omission; selecting on it picks the most stenographer-like epoch. Generation over a
15-minute dev set costs ~10 s at batch 64. Implementation: a `compute_metrics` with
`predict_with_generate` in `train.py`, arm A's dev transcription cached per speaker. Cost:
small code change, then rerun 80-minute cells (~1.3 h). Effect: bounded by how much the
epochs differ; fact 4 says they differ a lot at 1e-3 and little at 1e-4. **Small cost, likely
moderate effect; combine with A1.**

**A3. Loss masking of protocol-dropped regions.** Instead of changing the target, do not
back-propagate through reference positions adjacent to a forgiven insertion. Same alignment
work as A1, more intrusive in the collator, and it silences the model where the truth is
unknown rather than teaching it. Prefer A1.

**A4. Filler-aware label smoothing / insertion-tolerant loss.** No tooling in the repo, no
clean formulation for seq2seq CE; skip unless the literature reports it working.

### B. Change how adaptation is done

**B1. Self-training on the speaker's own audio (pseudo-labels from the base model).** Labels
are the base model's own transcriptions, so the adapter *cannot* learn the stenographer's
omissions — the target contains everything the model heard. Plain self-training gives little
gradient; the useful variants add augmentation (SpecAugment, speed perturbation, noise) so the
model learns invariance for this voice (noisy-student style), or use a stronger teacher (beam
search, or the A+B agreement). This is the one method whose objective is orthogonal to fact 2
by construction. Cost: pseudo-labels for the train sets (~25 min GPU), augmentation in
`ChunkDataset` (~40 lines), 80-minute cells with control (~1.5 h). Effect on the *personal*
component: unknown; it measures acoustics alone, which is what the project wants to know.
**Medium cost, the most informative negative or positive result available.**

**B2. Whisper text prompting as zero-training adaptation.** `prompt_ids` / the previous-text
context can steer vocabulary (names, committee terms) at decode time. It does not adapt to a
voice and it is known to increase hallucination in some settings. Useful as a cheap *lexical*
control: if prompting recovers a chunk of the domain gain, the adapters' gain is lexical.
Cost: small, decode-only, ~15 min for eleven speakers.

**B3. Speaker conditioning (x-vector / embedding prefix).** Requires training a conditioning
path across many speakers before it can be evaluated per speaker; the panel is eleven
speakers. Out of scope for this corpus size.

**B4. Two-stage: generic committee adapter, then personal on top.** Fact 1 says a generic
adapter is the deployable product. Train one on several hours of *non-panel* committee
speakers, evaluate on the panel, then train the personal adapters starting from it (stacked
LoRA or merged base) and measure the residual. Cost: non-panel audio must be extracted from
the shards (~45 min per 10 h through `materialize.py extract` with a wider plan), then one
training (~15 min) and the personal cells (~1.5 h). **The strongest practical deliverable;
medium-large cost because of the data.**

### C. Change the recipe within LoRA

**C1. Learning rate 3e-4 as the default**, best epoch 2 everywhere, no speaker damaged. Zero
cost; already measured.

**C2. Rank.** With ~370 steps and 80 minutes, r=8 (3.9 M parameters) is not the constraint;
r=4 would be a regularisation test, r=16 a capacity test. Low priority; one hour each.

**C3. Steps versus passes.** With selection by dev metric (A2) this becomes moot: fix passes
at 6, let selection choose.

### D. Decoding and the runaway problem

**D1. Cap `max_new_tokens` by audio duration** (e.g. 8 tokens per second of audio + 20)
and set `no_repeat_ngram_size` or a repetition penalty. This bounds the damage a loop does to
a chunk (fact 6) without changing the scorer. *Caveat:* it changes hypotheses for base and
tuned alike, so base transcriptions must be regenerated; do it for both arms, cache
separately (a `decode` tag in the cache name). Cost: small; ~15 min to re-transcribe the
eleven test sets per arm.

**D2. Report a "trimmed" WER** beside the standard one — errors capped at 3× the reference
length per chunk — as a diagnostic only. Not a replacement for the count.

### E. The speakers and the labels

**E1. The audio gate.** Never run. 30813 is worse at every site and rate and her base WER is
far below her error-map value; both fit mislabelled chunks. It needs `ffmpeg`, `speechbrain`,
`torchaudio`, gated access to `ivrit-ai/knesset-committees` and `segments.parquet` (521 MB).
CPU only, two to four hours, runs beside anything else.

**E2. Seeds 1 and 2 for the 80-minute cells only** (22 cells, ~2 h) to put intervals on the
S2 finding; and add the S2 alternate חיים כץ (556) to test "personalization pays for hard
speakers" on a third speaker (audio must be materialized: one speaker, ~10 shards).

**E3. The random-split contrast (D2).** Train and test on chunks from the *same* sessions,
once, to measure how much of a naive gain is session memorisation. One run of eleven cells.
Useful for the write-up's "why session-disjoint" paragraph; does not change the answer.

## The plan, ranked (revised after the survey)

| # | what | cost | expected effect | motivated by |
|---|---|---|---|---|
| 1 | **A1 + A2**: semi-verbatim targets (reinsert words arms A and B agree on) and checkpoint selection on dev forgiven WER; 80 min, lr 3e-4, 11 cells + control | ~2.5 h GPU | removes the cause of fact 2; may expose the acoustic personal component | Pakhomov 2001; McNamara 2412.07937; hesitation tagging 2506.04076 |
| 2 | **Recipe hygiene, once, before any paper table**: sample timestamps (40–50 %) and previous-text (50 %) into training as ivrit-ai did, mask prompt tokens from the loss, lr 3e-4 with weight decay 0.05, early stop on dev loss + dev WER; decode with Whisper's compression-ratio / logprob fallback and a repeated-n-gram rewind guard, re-cache both arms; report the repeated-5-gram rate beside `runaway` | small code, ~1 h GPU to re-run 80-min cells + re-cache | fixes facts 4 and 6; no change to the personal component expected | ivrit-ai train-whisper.py and blog; Radford 2212.04356 Table 7; CrisperWhisper 2.0 |
| 3 | **E1 audio gate** on the panel, CPU, in the background of 1–2 | 0 GPU | explains or clears 30813; validates labels | handoff § open problems; ParlaSpeech 2409.15397 practice |
| 4 | **Site ablation at the decoder MLPs** (`fc1`/`fc2`, LoRA r=8), 80 min, 11 cells + control | ~1.3 h | tests whether speaker information lands where q/v did not look | Müller-Eberstein 2406.04240 |
| 5 | **B1 self-training with augmentation** (pseudo-labels from the base, SpecAugment / speed perturbation) | ~2 h | isolates the acoustic component by construction; the literature has no speaker-specific evidence for typical speakers, so a null is informative | Interspeech 2025 noisy-student on dysarthric speech |
| 6 | **B4 generic committee adapter from non-panel speakers, then personal on top**; optionally x-vector conditioning of the decoder on the pooled adapter | ~3 h + extraction | the deployable result; conditioning gains are predicted only for the hardest speakers | kNN-Whisper 2410.18850; x-vector 2505.12991 |
| 7 | **E2** seeds 1–2 at 80 min and a third S2 speaker (556) | ~3 h | intervals on the "for whom" claim | stutter/dysarthria literature: gains scale with baseline WER |
| 8 | **B2 / kNN zero-training bounds**: `prompt_ids` with the speaker's earlier protocol text | 0.3 h | bounds the lexical share of the ~2-point personal component | Peng 2305.11095; 2410.18850 |

**Not adopted from the survey, and why.** *Segment-level agreement filtering* (drop training chunks
whose protocol-vs-model CER is worst) is standard corpus practice, but the handoff's rule
against filtering harder than quality ≥ 0.7 applies to it for the same reason: the chunks it
removes are the ones the model finds hard, which is the object of study. Use it only as a
diagnostic contrast, never as the main recipe. *Rank* changes: the survey's small-data results
(r=8 best at tens of files) agree with ours. *Style/coverage tokens* (Reverb) are the principled
long-term fix but need parallel verbatim targets, which item 1 would produce first.

**Status after the session (2026-09-20, evening).** Items 1, 3 and 4 were executed on the
same pod, plus the missing cell of item 1's 2 × 2 (semi-verbatim targets with dev-loss
selection) and the selection criterion on its own; every result is in
`docs/training_run2.md` § Results: follow-ups 3–5 and § Conclusion. In one line each:
semi-verbatim targets remove the omission incentive and raise the personal share of the
forgiven gain to about five points, at the cost of more runaway decodes; selecting on dev
forgiven WER is *worse* than dev loss at 15-minute dev sets; the decoder-MLP site is not
better than q/v; the audio gate finds 5 % foreign-voice chunks, concentrated in 30843's dev
session, and clears 30813. Items 2 (decode hygiene, now necessary because of the runaways),
5, 6, 7 and 8 remain.

Items 1–3 are one session of about four GPU hours. Item 2 should be done before item 1's
cells are scored, so that every table in the paper uses one decode configuration.
Everything runs through `run_panel.py` with new flags (`--targets verbatim`, `--select
forgiven`, `--sites decoder_mlp`, `--decode whisper`, `--self-train`) so the idempotent,
resumable, backed-up pattern of this session carries over.

## Literature survey

Produced by a research agent from web search during the session (2026-09-20); claims marked as from abstracts or blogs are so marked. Citations are as the agent reported them and should be checked before they go into a paper.

