# The per-speaker error map — committees corpus

Stage 1's analysis, re-done on the corpus the project moved to. Written from the
run of 2026-09-15; every number is measured. The code is
`src/evaluation/error_map.py` (plain functions, self-checks at the bottom); the
narrative and figures are `notebooks/committees_error_map.ipynb`; the tables are
under `src/evaluation/outputs/committees_*`.

## What it answers

1. How well does each model do **per speaker** — WER and CER, counts pooled (Σerrors / Σwords).
2. Each speaker's **adaptation gain**, `WER_A − WER_B`: how much the Hebrew fine-tune already
   bought that voice. Design step 5 picks the speakers with the *least* gain.
3. Whether the amount of speech predicts recognition (it does not) and how much of the
   per-speaker spread is measurement noise.
4. Which **subgroup rules** are viable for Stage 2's sharing axis, and which actually separate
   speakers.

## The data

The Stage-1 subset of `Hadasy/knesset-committees-chunks` transcribed by both arms,
`Dolevabudi/knesset-committees-inference` (`docs/inference.md` § Result): one hour per MK at
alignment quality ≥ 0.5, round-robin over sessions — **65,990 chunks, 230 h, 267 speakers**,
every chunk carrying both hypotheses. Arm A is `openai/whisper-large-v3` with the language forced
to Hebrew; Arm B is `ivrit-ai/whisper-large-v3-turbo-ct2`. Demographics come from
`Dolevabudi/knesset-committees-speakers`; corpus hours per speaker from the inference index.

## What changed from the VoxKnesset version, and why

| Stage 1 on VoxKnesset | here | why |
|---|---|---|
| filter: a flat 300 s of audio the reference cannot account for | filter: alignment `quality ≥ 0.7` | the old rule cannot fire on ≤ 30 s chunks; the failure it caught — reference text that does not match the audio — is what `quality` measures. Below 0.7 the WER is ≈ 1.0 under both arms |
| `hours` = speech in the dump | volume axis = **corpus hours** at quality ≥ 0.7 | the subset caps every speaker at 1 h, so subset hours are what survived the filter, not volume |
| no per-speaker CI | 95 % bootstrap CI (chunks resampled) on WER and gain | "who gains least" is a ranking; its noise has to be visible |
| — | S/D/I split, runaway flag, flagged sessions, language-detection table | `evaluate.score` gives them for free; the last two are the handoff's warnings as numbers |

Everything else is Stage 1's: the normalisation (`src/common.py`, verbatim), the scoring, the
subgroup rules and their thresholds (≥ 3 speakers, ≥ 5 h with the largest member held out;
separation on speakers with ≥ 20 chunks), eta squared against a 2,000-permutation null.

## Results

### The filter

| | chunks | hours | ref words | WER A | WER B | CER A | CER B |
|---|---|---|---|---|---|---|---|
| before filter | 65,990 | 230.1 | 1,610,207 | 0.4173 | 0.3243 | 0.2662 | 0.2286 |
| after filter (quality ≥ 0.7) | 58,180 | 217.3 | 1,528,937 | **0.3868** | **0.2924** | 0.2414 | 0.2031 |
| the dropped chunks | 7,810 | 12.8 | 81,270 | 0.9915 | 0.9242 | 0.7527 | 0.7297 |

Unlike the plenums, where fourteen segments could not move the average, here the filter moves
the corpus WER by three points on both arms: 12 % of the chunks hold 5 % of the reference words
and 12–14 % of the errors, and score a WER of ≈ 1 — errors equal to reference words,
which is what a reference that does not match its audio looks like. Both arms pay it equally,
so the *difference* between them barely changes (0.093 → 0.094). By quality bucket the WER is
1.06 / 1.01 (A / B) under 0.6, 0.95 / 0.87 at 0.6–0.7, and 0.32 / 0.22 above 0.9, where 60 % of
the chunks sit.

### Overall

A 0.387 WER, B 0.292 — B better by 24 %. Roughly double the plenum level (0.207 / 0.098): the
reference is a cleaned stenographic protocol, both models transcribe the repetitions and
cross-talk on the tape. The error mix differs between arms: **53 % of B's word errors are
insertions against 38 % for A** (substitutions 36 % vs 49 %). B transcribes more of what the
protocol left out; its errors are the register's, which is why the comparison is a gain and not an
absolute WER. Runaway decodes (hypothesis > 3× the reference words): 306 chunks for A, 420 for B,
reported and left in.

### Per speaker

267 speakers, 261 with ≥ 20 chunks after the filter. Median per-speaker WER 0.378 (A) and 0.291
(B); interquartile ranges 0.33–0.45 and 0.25–0.33. Median 95 % CI half-width ±0.035 on WER_B and
±0.017 on the gain.

**Every speaker is helped by the fine-tune** (267 / 267; the gain's CI excludes zero for 265).
Median relative gain **24 %** of the general model's error removed — against 52 % on the plenums.
WER_A and WER_B rank speakers almost identically (Spearman 0.90): a speaker who is hard for the
general model is hard for the fine-tune. Absolute gain tracks difficulty (Spearman +0.63 with
WER_A: more error, more to remove); relative gain does not (+0.07), so the ranking is on it. Largest relative gains among reliable speakers run to
58 % (משולם נהרי, סופה לנדבר, עמיר פרץ), the smallest to 11 % (אופיר אקוניס, אביר קארה, אוסנת הילה
מארק, משה פסל), and the smallest-gain speakers are not the hardest ones — their WER_A sits near
the median; the fine-tune simply did little for them.

### WER against the amount of data

**The level is flat**: corpus hours do not predict either model's WER (Spearman −0.07 for A,
+0.06 for B). **The spread is measurement noise**: the SD of WER_A falls from 0.21 among speakers
with 6–20 chunks to 0.08 with 151–400, and from 0.15 among speakers with under an hour of corpus
speech to 0.06–0.10 above ten hours.

**The one real correlation is with the filter.** The share of a speaker's chunks the quality
filter removed predicts their WER on the *kept* chunks with rho +0.61 (A) and +0.68 (B). Speakers
whose speech aligns badly are also harder to recognise. Nothing in the text can say whether that
is acoustics (cross-talk, microphones) or labels (the protocol naming the wrong speaker). The
audio gate in `src/preprocessing/speaker_index/validate_audio.py` is the check, and it has not
run; until it does, an anomalous speaker is a labelling question first.

### Subgroups

**Viability.** Every genuine group clears 3 speakers and 5 corpus hours with its largest member
held out, including the smallest: Bedouin (5 speakers, 27 h), Europe (5, 68 h), Christian (3, 76 h).
Only the metadata-missing cells fail (`Unknown` ×3 speakers, `Unrecorded` orientation, the single
Sunni entry). Data is not the binding constraint.

**Separation.** Eta squared on the reliable speakers in genuine groups, against each rule's own
permutation null:

| rule | groups | eta² gain_rel | p | eta² wer_B | p | separates |
|---|---|---|---|---|---|---|
| speaking rate tertile | 3 | **0.092** | <0.001 | 0.001 | 0.90 | gain |
| religion | 4 | 0.072 | 0.002 | 0.009 | 0.51 | gain |
| nationality | 4 | 0.071 | 0.003 | 0.009 | 0.48 | gain |
| age quartile | 4 | 0.050 | 0.004 | 0.009 | 0.53 | gain |
| country of origin | 6 | 0.037 | 0.09 | 0.035 | 0.11 | neither |
| gender | 2 | 0.020 | 0.02 | 0.027 | 0.008 | both, weakly |
| religious orientation | 3 | 0.002 | 0.88 | **0.062** | 0.002 | difficulty |

The two outcomes come apart, **the other way round from VoxKnesset**. There, demographic
membership predicted who the models find difficult and only speaking rate predicted who benefits.
Here speaking rate, religion, nationality and age predict the *gain* and not the difficulty;
religious orientation and gender predict the difficulty and not the gain. Speaking rate is the one
rule that separated gain on both corpora, and it is the strongest here: pooled, slow speakers
have 27 % of A's error removed, fast speakers 22 %. Arab, Muslim and Bedouin speakers gain more
than Jewish speakers (29–30 % vs 24 %) with no difference in how hard they are for B — the
fine-tune closes a gap the general model has on them. Haredi speakers are the hardest group under
both arms (B 0.334 vs 0.285 secular) and gain no more than anyone.

The best rule explains 9 % of the between-speaker variance in gain and 6 % in `wer_B`; 90 % of
the variation is within every group. Real, weak, and a different set of rules than the plenums
produced — which says the plenum result was partly a property of that corpus.

### Who to adapt

`committees_adaptation_candidates.csv`: reliable speakers by relative gain ascending, with the CI.
The bottom of the list — gains of 11–14 % where the median is 24 % — is where a personal
adapter has the most room; 70 speakers sit below the median gain with their whole CI; whether a low gain is the voice or the labels is the open question
above. Speaking-rate and demographic membership are weak predictors of it, so the candidate list,
not a rule, is the input to Stage 2.

### Two things to distrust

- **14 sessions** with ≥ 5 chunks score WER > 0.9 under *both* arms after the filter
  (`committees_flagged_sessions.csv`), one of them 49 chunks (session 2077130). The handoff's
  time-origin-offset signature. Listed for a listening check, not removed.
- **Language detection.** Arm A's first run let Whisper detect the language: 7.1 % of chunks came
  back in another script, a quarter of those under 3 s. That run is kept as `hypothesis_A_auto`;
  everything above uses the forced-Hebrew run.

## What this version leaves out

- Only one filter. Chunks whose reference is *short* for the audio survive it: after the filter,
  chunks under 50 words per minute (1.3 % of chunks, 3 h) score a WER of 2.0 under both arms —
  speech the protocol condensed, every word an insertion. A small share of the words; left in.
- No check of the transcription side (runaway decodes are reported, not filtered).
- No covariate model; rules are examined one at a time.
- The labels, as above.
- The digit problem: the protocol writes `שמונה`, Whisper writes `8`.

## Files

| file | holds |
|---|---|
| `src/evaluation/outputs/committees_speaker_performance.csv` | one row per speaker: Stage 1's columns, CIs, S/D/I shares, runaway counts, demographics, corpus hours |
| `committees_qc_effect.csv` | the filter's before / after / dropped table |
| `committees_subgroups.csv` | viability table for every rule and group |
| `committees_subgroup_rank.csv` | eta squared, permutation null and p for every rule |
| `committees_adaptation_candidates.csv` | reliable speakers by relative gain, ascending, with CI |
| `committees_flagged_sessions.csv` | sessions failing under both arms |
| `committees_summary.json` | the headline numbers |
| `docs/figures/error_map/*.png` | the notebook's figures |

```bash
python src/evaluation/error_map.py          # self-checks
python src/evaluation/error_map.py --run    # every table, ~10 s
```
