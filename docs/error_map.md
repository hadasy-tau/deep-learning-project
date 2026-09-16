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

## Beyond Stage 1 — what the inference also tells

Eight checks Stage 1 did not make, over the filtered subset (`src/evaluation/error_analysis.py`;
notebook § 11; tables `committees_conditions.csv`, `committees_error_content.csv`,
`committees_checks.json`).

**What the errors are.** The digit problem Stage 1 feared is negligible here: substitutions
involving a digit are 1 % of errors, and WER with numeric tokens removed from both sides is
unchanged (A 0.387 → 0.387, B 0.292 → 0.293). What dominates instead is orthography and clitics:
the most frequent substitutions on both arms are אני↔ואני, הכול↔הכל, לכן↔ולכן, זו↔זה, הזאת↔הזו,
כול↔כל — a conjunction the speaker said and the stenographer dropped (or the reverse), full
versus defective spelling, and the stenographer's synonym (כאן for the spoken פה). Substitutions
one character apart are 35 % of A's substitutions and 42 % of B's, 15–17 % of all errors. The
top insertions and deletions are function and discourse words (אני, לא, זה, את, תודה, אז, אבל,
כן, באמת): the filler the protocol cleans away. A normalisation note for the report: `1,000`
splits into the tokens `1` and `000` under the frozen normaliser, which is where `000→אלף` in
the substitution list comes from.

**B's insertions are real speech, not hallucination.** 73 % of the words B inserts also appear
in A's hypothesis of the same chunk (236,530 inserted words). Two independent models hearing
the same absent word is the protocol not writing it down. Looping decodes (a 3-gram repeated
three or more times) are 0.8 % of chunks for A and 1.1 % for B; runaway decodes 0.5 % and
0.7 %; empty hypotheses 0.2 % and 0.

**Conditions.** By year, 2018–2019 sessions score 6–10 points worse than 2020 onward on both
arms (A 0.45–0.49 against 0.35–0.39; B 0.32–0.37 against 0.27–0.29). By room, the Finance
committee is the hardest of the large committees (B 0.374) and the Status of Women committee
the easiest (0.212). Both effects are on top of the speaker, and both are why splits are by
date (D2) and why a panel speaker's held-out sessions were checked against the train side.

**The per-speaker ranking is reliable.** Split-half over odd and even sessions, Spearman-Brown
corrected: WER_A 0.85, WER_B 0.78, absolute gain 0.85, relative gain 0.75 (254 speakers).
Most of the between-speaker spread is real at one hour per speaker; the subgroup rules'
eta squared of a few percent is therefore a weak effect on a reliable outcome, not noise.

**Chunk level.** Median chunk WER 0.385 (A) and 0.262 (B); 7.5 % of chunks are perfect under B,
2.9 % under A; 5.6 % of chunks fail (WER ≥ 1) under both arms; the two arms' error counts
correlate at 0.87 across chunks.

**Language forcing.** On the filtered subset, forcing Hebrew moved A's corpus WER from 0.392 to
0.387, almost all of it on chunks under 3 s (0.638 → 0.573).

**The plenums.** 212 speakers have 20 or more segments on both corpora. Difficulty transfers:
committees and plenum WER correlate at 0.51 (A) and 0.42 (B), relative gain at 0.34. The level
does not: per speaker, committee WER_B is 3.2 times the plenum figure. The plenum number was
measured on audio arm B had trained on, in the cleaner register; the committee number is the
honest one.

## The same map, protocol-aware

§ Beyond Stage 1 showed that B's insertions are mostly the protocol's omissions. This section
re-runs the whole map under one alternative count, **forgiven-shared WER**: an inserted word
that the *other* model also produced at that chunk is not charged; substitutions, deletions and
the denominator are unchanged; an insertion only one model produces is still charged. Standard
WER stays the headline; this is the second column. Code: `error_analysis.forgiven_counts`,
`error_map.run(scoring='forgiven')`, `error_map.compare`; notebook § 12; tables
`committees_*_forgiven.*` and `committees_scoring_comparison.json`.

| | A | B | B's advantage | median gain per speaker | speakers hurt |
|---|---|---|---|---|---|
| standard WER | 0.387 | 0.292 | 24 % | 24 % | 0 |
| forgiven-shared WER | 0.282 | 0.179 | **37 %** | 36 % | 0 |

**What moved: the level, and B's advantage.** The two models fail differently — A's errors are
more often genuine mishearings, B's more often the protocol's omissions — so charging both
equally for the protocol flatters A. Under the protocol-aware count the Hebrew fine-tune removes
over a third of the general model's error, not a quarter.

**What held: everything about individual speakers.** Every reliable speaker is still helped and
none hurt. The per-speaker ranking barely moves: Spearman between the two counts is 0.94 on
WER_A, 0.92 on WER_B, 0.99 on absolute gain, 0.96 on relative gain; 35 of the bottom-40
adaptation candidates are the same. Every subgroup rule keeps its verdict on gain (speaking rate
strongest on both counts, eta² 0.09); country of origin picks up a weak difficulty verdict it did
not have; the effects stay small.

**What to watch.** One panel speaker, דוד ביטן, moves from the 59th to the 75th percentile of
benefit: part of his apparent difficulty is the protocol. Read together with the standard
results, the panel stands and his role is relabelled — hard, and already well served — with the
audio gate as the check that could remove him (`adaptation_plan.md` § The panel).

**The caveat.** A and B are both Whisper large-v3 descendants, so a shared insertion is strong
evidence of a protocol omission, not proof: correlated models can hallucinate alike. The
measure is a diagnostic beside the standard one, not a replacement.

## What this version leaves out

- **The protocol question is not settled.** A human verbatim transcription of 30–60 minutes,
  stratified by speaker and chunk length, scored against both models, would say how much of the
  residual error is the protocol and how much the model, and would calibrate the forgiven-shared
  count. Deferred, deliberately.
- Only one filter. Chunks whose reference is *short* for the audio survive it: after the filter,
  chunks under 50 words per minute (1.3 % of chunks, 3 h) score a WER of 2.0 under both arms —
  speech the protocol condensed, every word an insertion. A small share of the words; left in.
- No check of the transcription side (runaway decodes are reported, not filtered).
- No covariate model; rules are examined one at a time.
- The labels, as above.
- The digit problem, which § Beyond Stage 1 now measures at 1 % of errors.

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
| `committees_conditions.csv` | corpus WER by year, committee, chunk length, speaking rate |
| `committees_error_content.csv` | top substitution pairs, insertions and deletions per arm |
| `committees_checks.json` | insertion agreement, hallucination rates, language forcing, split-half reliability, cross-corpus |
| `committees_*_forgiven.csv`, `committees_summary_forgiven.json` | the six tables above under the protocol-aware count |
| `committees_scoring_comparison.json` | standard vs protocol-aware: corpus, gain, ranking correlations, candidate overlap, rule verdicts |
| `docs/figures/error_map/*.png` | the notebook's figures |

```bash
python src/evaluation/error_map.py          # self-checks
python src/evaluation/error_map.py --run    # every table, ~10 s
```
