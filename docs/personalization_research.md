# How to make personalization work: diagnosis, literature, and what to test next

Written 2026-10-03, after the third training run (`docs/training_run3.md`). **Revised the same
day after checking against a human reference (§ 1.5): the arm-A agreement proxy used in § 1.1–1.2
did not survive it. The claims built on it are marked.** The question: the
project's goal is a per-speaker model that recognises *this* speaker better than a general one.
Run 3 found a personal effect of only +2–3 points of relative WER. Is that the ceiling, or the
method? This document checks our own data first, then the literature, and ends with a ranked,
testable plan. Every number from our data below was computed on run 3's saved hypotheses
(`knesset-asr/knesset-committees-v3-results`); every literature number was checked against the
paper itself, not a summary, unless marked otherwise.

## 1. What our own data says first

### 1.1 Most of the remaining "errors" are the reference, not the voice

Base B on the high-quality test (12 speakers, WER 0.139):

| finding | number | what it means |
|---|---|---|
| error mix | substitutions 40 %, deletions 10 %, **insertions 49 %** | half the errors are words the model wrote and the protocol did not |
| insertions that arm A also produces (Stage 1) | 73 % | read as spoken words the stenographer dropped; § 1.5 found about half were (19 of 36) |
| **substitutions and deletions that arm A makes identically** | **66 %** of 4,796 | two models with different training data agreeing on the same "error" points to the reference's conventions |
| substitutions that are near-misses (one character, or a prefix letter) | 58 % (44 % on the ≥ 0.7 test) | spelling and morphology conventions, not mishearings |
| errors on a word only this speaker uses (in their train text, in no other panel speaker's) | **1.0 %** | personal vocabulary is not where the errors are |
| errors *attributable to the model* (not made identically by arm A) | **35 %** (median over speakers) | ~~the room any model change can really work in~~ **not supported (§ 1.5)**: agreement with arm A does not tell a protocol error from a model error |

### 1.2 What the adapters actually learn (built on the arm-A proxy: see § 1.5 before relying on it)

- **More than half of what the personal adapter fixes and the control does not (54 %) are errors
  arm A also makes**, i.e. the adapter learns *how the protocol writes this speaker*, not their
  voice. The rest: 21 % real substitutions, 20 % near-misses, 6 % deletions. Net personal fixes
  over three seeds at 80 minutes: 291 errors (995 fixed only by the own adapter, 704 only by the
  control).
- **On the attributable errors, fine-tuning makes things worse.** At 80 minutes, over three
  seeds: own adapter **−10.4 %**, control **−15.6 %** (more errors than the base). The entire
  standard-WER gain (+11.6 %) comes from matching the protocol's conventions, while the adapters
  add errors arm A does not make, most likely dropping words the protocol keeps (deletions double,
  10 % → 21 % of errors). The semi-verbatim recipe: own −4.6 %, control −0.5 %.
- **Personalization on the attributable errors is +4.3 % (median) but spread from −16 to +26 %**:
  30843 +26 %, 30813 +23 %, 30868 +13 %, 30701 +11 %, while 30685, 30718, 30831 are −14 to −16 %.
  For some speakers the personal adapter clearly learns something the control does not; for
  others it learns the wrong thing. The averages hide both.

The attributable-error measure uses arm A as a second reference. It is a proxy with known
biases (arm A is weaker; an error both models make for acoustic reasons is counted as a label
convention). It is a reason to build a better reference (§ 4, P0), not a replacement for one.

### 1.5 Checked against a human reference: the protocol is noisy, but the proxy does not find the noise

`ivrit-ai/eval-forced-alignment` (gated, accessible with our token) holds 72 human-corrected Hebrew
clips, 52 from Knesset committees (7.4 min, 954 words), transcribed by ivrit.ai's guidelines:
numbers as words, spoken words added, unspoken words removed, no fillers. None of its sessions is
a panel test session (four are panel train/dev sessions). For 23 clips the protocol span could be
recovered from our corpus index (the others are non-MK speakers our index does not hold); arms A
and B transcribed those 23 clips locally (338 protocol words, 387 gold words).

| | vs the protocol | vs the human transcript |
|---|---|---|
| base B | 0.281 | **0.165** |
| base A | 0.393 | 0.295 |
| the protocol itself | — | **0.183** |

- **The reference problem is real and large.** The protocol differs from what was said by 18 % of
  its words, and about 40 % of B's measured WER on these clips is the protocol, not the model.
- **The arm-A agreement proxy does not detect it.** Of B's errors against the protocol, the human
  transcript sides with the model in **45 % of the errors arm A shares and 49 % of those it does
  not**: agreement with A carries almost no information about who is wrong. Shared substitutions
  are mostly real model errors (the gold sides with the model in 6 of 20); insertions are about
  half real speech either way (19 of 36 shared, 12 of 25 not).
- **Consequences.** The "35 % attributable", the "adapters add 10–16 % attributable errors" and
  the "54 % of personal fixes are conventions" figures in § 1.1–1.2 rest on that proxy and are
  withdrawn as evidence. What stands: the protocol is a noisy reference; run 3's standard gains
  partly measure agreement with it; only a human reference can say how much.
- Limits: 23 clips, one annotator each, number normalisation differs between gold (words) and
  protocol/hypotheses (digits), which counts against the models in both columns.

### 1.3 Two assumptions checked

- **"Model B already heard these voices in the plenum, so there is nothing left to personalize"
  is not supported.** Plenum hours per speaker in VoxKnesset (0.5–15 h) correlate *positively*
  with personalization on the ≥ 0.7 test (Spearman +0.55, p 0.07, n = 12).
- **"Only ~140 high-quality minutes exist per speaker" is wrong.** That figure came from
  `word_quality.RAW_TARGET_MIN = 250`: only 250 candidate minutes per speaker were ever scored
  for the word rule. The corpus has a median of **23 hours per panel speaker at quality ≥ 0.7 and
  ~15 hours at ≥ 0.95** (23558: 124 h; 30843, the smallest: 7 h). The adapters trained on 80
  minutes. **The data-size axis has not been tested beyond 6 % of what exists.**

### 1.4 Implications before reading any paper

1. The training signal is the problem as much as the method: the protocol differs from the
   speech by ~18 % of its words (§ 1.5), and a loss on it rewards matching those differences. A
   cleverer adapter trained on the same targets will mostly learn the same thing.
2. The measurement is the other problem: against a reference that is itself 18 % off, a modest
   acoustic effect can be invisible, and no automatic proxy we tried separates the two (§ 1.5).
3. The largest untested lever is data: 80 minutes against 15–23 hours.

## 2. What the literature says (checked)

**Personalization gains scale with how badly the base model fits the speaker.** For atypical
speech (dysarthria, deaf speech, severe impairment) gains are very large: hypernetwork-generated
adapters give 75 % relative WER reduction with 0.1 % of the parameters (Apple, TACL 2024, atypical
speech); Perceiver-Prompt speaker prompts on a LoRA-tuned Whisper give 13 % relative CER
reduction overall and 51 % on the most severe speakers (Interspeech 2024); Project Euphonia's
personalized models reduce WER by 26–35 %. For typical speakers the evidence is thinner and the
gains smaller.

**Typical-speaker numbers in the literature include the domain effect.** Speaker adaptation of a
~278 M-parameter Whisper on LibriSpeech/TED-LIUM gives ~15 % relative (quantised-model papers,
arXiv 2408.03979; Interspeech 2024 SAML; IEEE TASLP 2025 SAMD). None of the typical-speaker papers
we checked trains a budget-matched adapter on *other* speakers' data, so their gains include what
our control showed to be 10–14 points of domain. Our design is stricter than the literature's;
our +2–3 points is not directly comparable to their 15–25 %.

**Personal data scales log-linearly, to 20 hours.** Weninger et al. (Nuance, Interspeech 2019):
seq2seq ASR, 35 dictation speakers, base trained on 7.6 k hours; WER falls log-linearly with
adaptation data from 5 minutes to 20 hours, **14 % relative at 2 h and 25 % at 20 h**, using
KL-divergence regularization to the base model (β 0.6–0.8), which also made gains "more uniform
(no degradations for any speaker)". Adapting all parameters beat decoder-only (14.3 vs 12.3 %).
Caveats: no other-speaker control, and dictation domains carry speaker-specific vocabulary. Their
adaptation labels were "pseudo truth" (ASR output plus user corrections), not an edited record.
Our own adapters' standard gains (5 → 20 → 80 minutes: +4.3 / +7.5 / +12 %) fit the same
log-linear shape.

**KL regularization to the base model** is the classic remedy for small-data adaptation that
damages some speakers (Yu et al. 2013; Weninger 2019). It is the direct answer to § 1.2: an
adapter pulled toward the base distribution cannot freely add deletions the base would not make.

**Test-time adaptation helps weak models, not strong ones.** SUTA/SGEM on children's speech
(Interspeech 2025): −3 points absolute on an off-the-shelf wav2vec2 (30.5 → 27.5 % WER) but only
−0.7 on the fine-tuned large model, which got *worse* for 11 of 91 speakers; the authors note TTA
"may be less capable … for highly accurate models". SUTA was designed for CTC. STAR (NeurIPS 2024)
adapts Whisper to *domains* (noise, accents) from < 1 h unlabeled audio, 13.5 % average across 14
domains; it is unsupervised domain adaptation, not personalization, but its pseudo-label quality
indicator is reusable.

**In-context learning works where the language is supported.** Phi-4-multimodal with 12
same-speaker utterances (~50 s) in context: 19.7 % relative on English, strongest for low-resource
varieties and when context and target speaker match (arXiv 2505.14887). Phi-4-multimodal's speech
input does not cover Hebrew. Whisper's speech-based in-context learning (SICL, 2023) gives large
gains on *isolated words* in Chinese dialects, but the examples and the test must share Whisper's
30-second window: unusable for our ≤ 30 s chunks.

**Text prompting is weak evidence.** Whisper "may not understand textual prompts in a
human-expected way" and better topic adherence does not guarantee lower WER (arXiv 2406.05806); a
preregistered ablation found no detectable WER change from prompt-level context on a production
corpus (arXiv 2608.28875, GPT-4o-transcribe). Our diagnosis adds that only 1 % of errors fall on
speaker-unique words.

**External language models help less as the base gets stronger.** Whisper-LM (2025): n-gram fusion
gains up to 51 % for Basque, but 0–7 % for large Catalan models and degradations for large
Spanish models out of distribution. Hebrew with ivrit.ai's 4,700-hour fine-tune is closer to the
second case. And an LM trained on protocol text teaches protocol conventions, the thing § 1.2
already shows the adapters learning.

**The label problem has dedicated methods.** Style-controlled ASR with verbatim/intended decoder
tokens (arXiv 2607.18934) needs *paired* verbatim and clean transcripts of the same audio, which
we do not have. A token-level tolerant CTC loss that skips reference tokens the acoustics do not
support gives 9.5 % relative across 19 languages (arXiv 2609.30160); it is CTC-only, but the idea
transfers to attention models as loss masking. Multi-reference, style-agnostic evaluation shows
that standard WER "significantly over-estimates" contentful errors of strong ASR systems (arXiv
2412.07937), exactly what § 1.1 measures for us.

**Speaker-conditioned models** (speaker prompts, mixtures of LoRA experts routed by speaker,
hypernetworks) are the current research frontier, with results on dysarthric, elderly and
quantised-model settings. They are evidence that zero-shot speaker conditioning works where speaker
variation is large; nothing we found shows it for typical native speakers.

## 3. What this means

- **The literature does not promise large personal gains for typical native speakers on a strong,
  in-language model.** Large gains come with atypical speech or a weak base. Our +2–3 points is
  not obviously a failure of method.
- **But our setup leaves three things untested that the literature says matter**: the amount of
  personal data (80 minutes of 15–23 hours), regularization against new errors, and a training
  signal that is not dominated by the reference's conventions.
- **And we cannot currently see an acoustic effect if one exists**, because the reference is the
  protocol. Any next experiment without a better reference risks repeating run 3's ambiguity.

## 4. Ranked plan (what to test, why, and what result would count)

Each step keeps run 3's design: the budget-matched control at every budget, the paired bootstrap
on personalization, three seeds where a claim depends on it.

| | step | why (evidence) | expected | cost | risk |
|---|---|---|---|---|---|
| **P0** | **A verbatim gold test**: the team transcribes 10 minutes per speaker of the existing test clips word for word (≈ 2 h of audio, Hebrew speakers in the team). Re-score run 3's saved hypotheses on it (no GPU). Report "attributable" WER beside it. | § 1.1–1.2: two thirds of the errors are reference conventions; 2412.07937 | tells whether run 3's personal effect is 2 % or 10 % on real errors; decides everything below | ~15–20 person-hours, $0 compute | inter-annotator spelling choices; agree on conventions first |
| **P1** | **Clean the loss, not just the data** — *weakened by § 1.5*: masking by A/B consensus would mask real model errors about as often as protocol errors. Keep the idea (token-level tolerance, 2609.30160) but choose the mask by a criterion validated on P0 (e.g. aligner word probability, or B-vs-protocol disagreement weighted by B's confidence), and test it against plain semi-verbatim targets. | § 1.5; token-level tolerance 2609.30160 | unknown until validated | small code change in `train.py` (labels = −100 on masked tokens) | the mask criterion is the whole question |
| **P2** | **KL regularization to the base** (β 0.5–0.8; the base distribution comes free from PEFT's `disable_adapter()`). | Weninger 2019, Yu 2013; § 1.2 shows adapters adding errors and hurting some speakers | fewer new deletions; no speaker hurt; small personal gain kept | ~20 lines; one extra forward pass | too strong a β erases the personal effect too; tune β on validation |
| **P3** | **Scale personal data: 80 → 320 → 1,200 minutes**, with the matched control at each budget, on P1+P2 targets. Score more candidate minutes (`RAW_TARGET_MIN`) or use the ≥ 0.7 data with P1 masking. | Weninger: log-linear to 20 h; § 1.3: 15–23 h exist per speaker | the key unknown: does the *personal* share grow once domain saturates? | data prep on the laptop; ~2 GPU-h on 4 GPUs (raise `max_steps` for large budgets) | the control grows too; only the difference counts |
| P4 | Personal text LM: shallow fusion with an n-gram LM from each MK's KnessetCorpus protocols, against an LM of other MKs' text of equal size. Exclude each speaker's dev and test sessions by ID (556, 30718 and 30813 have test sessions before the corpus's 2024-03-26 end). | Weninger: +2.4 % from speaker LM on top of adaptation; Whisper-LM | small; partly conventions again | beam search with fusion in `evaluate.py` | leakage; convention learning |
| P5 | Similar-speaker groups (the sharing axis): an adapter on the 3 most similar speakers (ECAPA distance) against 3 random ones. | `training_next.md`; personal data is scarce for some speakers (30843: 7 h) | in between control and own | ~44 short runs | — |
| P6 | Speaker-conditioned model: one LoRA trained on all speakers with a learned per-speaker prompt (Perceiver-Prompt style) or speaker-routed LoRA experts. | dysarthric/elderly/quantised results | unknown for typical speakers | a research project | — |

**Not recommended now**, with reasons: test-time adaptation (helps weak models, hurt 12 % of
speakers on a fine-tuned one); Whisper speech in-context learning (30-second window);
Phi-4-style in-context learning (no Hebrew speech); text prompting (null results, 1 % of errors
on personal vocabulary); more augmentation or rank (both measured flat in run 3).

### 4.1 What would count as success (fixed before running)

On the P0 gold test, at the best budget: **median personalization ≥ 5 % relative, with the
bootstrap interval above zero for at least half the speakers, in at least two of three seeds**,
and no speaker significantly worse than the control. A gain that appears on the protocol reference
but not on the gold test is convention learning, whatever its size.

### 4.2 A concrete run 4

1. P0 on the laptop: pick 10 minutes per speaker from the high-quality test (fixed seed), the team
   transcribes verbatim, re-score run 3 (base, own, control, both recipes).
2. Implement P1 (masked labels) and P2 (KL) in `train.py`, each behind a flag, with self-checks.
3. Score more candidate minutes per speaker; build budgets of 320 and 1,200 minutes (nested, as
   now).
4. On a 4× A100 pod: 12 speakers × {80, 320, 1,200} min × {P1, P1+P2}, controls at each budget,
   seed 0; seeds 1–2 for the best cell. About 4–5 GPU-hours, ~$10–15.
5. Read the result on the gold test first, then on the protocol and attributable counts.

## 5. Limits of this review

- The literature search covered speaker adaptation, test-time adaptation, in-context learning,
  prompting, LM fusion, label-noise and style methods, and speaker-conditioned models, from 2019
  to 2026. It is not exhaustive. Hebrew-specific personalization work did not come up.
- Several numbers above come from abstracts (Euphonia's 26–35 %, the hypernetwork's 75 %,
  SAML/SAMD's figures, Whisper-LM's headline). Weninger 2019, the TTA child-speech paper and the
  quantised-Whisper table were read in full.
- The attributable-error analysis was a proxy built on arm A; checked on 23 human-transcribed
  committee clips it did not discriminate (§ 1.5). That check is itself small (one annotator per
  clip). P0 exists to replace both.

## Sources

- Weninger et al., *Listen, Attend, Spell and Adapt: Speaker Adapted Sequence-to-Sequence ASR*, Interspeech 2019 — https://www.isca-archive.org/interspeech_2019/weninger19b_interspeech.pdf
- Shi et al., *Examining Test-Time Adaptation for Personalized Child Speech Recognition*, Interspeech 2025 — https://www.isca-archive.org/interspeech_2025/shi25h_interspeech.pdf
- *Speaker Adaptation for Quantised End-to-End ASR Models*, arXiv 2408.03979 — https://arxiv.org/abs/2408.03979
- *SAML: Speaker Adaptive Mixture of LoRA Experts*, Interspeech 2024 — https://arxiv.org/abs/2406.19706
- *Hypernetworks for Personalizing ASR to Atypical Speech*, TACL 2024 — https://arxiv.org/abs/2406.04240
- *Perceiver-Prompt: Flexible Speaker Adaptation in Whisper*, Interspeech 2024 — https://www.isca-archive.org/interspeech_2024/jiang24b_interspeech.pdf
- Project Euphonia — https://research.google/blog/project-euphonias-personalized-speech-recognition-for-non-standard-speech/
- *STAR: Self-Taught Recognizer*, NeurIPS 2024 — https://arxiv.org/abs/2405.14161
- *In-Context Learning Boosts Speech Recognition…*, arXiv 2505.14887 — https://arxiv.org/abs/2505.14887
- *Can Whisper perform speech-based in-context learning?*, arXiv 2309.07081 — https://arxiv.org/abs/2309.07081
- *Do Prompts Really Prompt? Exploring the Prompt Understanding Capability of Whisper*, arXiv 2406.05806 — https://arxiv.org/abs/2406.05806
- *No Detectable Change in Side-Level WER from Prompt-Level Context*, arXiv 2608.28875 — https://arxiv.org/abs/2608.28875
- *Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages*, arXiv 2503.23542 — https://arxiv.org/abs/2503.23542
- *Activating Controllable Verbatim ASR…*, arXiv 2607.18934 — https://arxiv.org/abs/2607.18934
- *A Training Criterion with Token-Level Tolerance to Transcription Ambiguity*, arXiv 2609.30160 — https://arxiv.org/abs/2609.30160
- *Style-agnostic evaluation of ASR using multiple reference transcripts*, arXiv 2412.07937 — https://arxiv.org/abs/2412.07937
- ivrit.ai, *Hebrew Forced Alignment Evaluation* (human-corrected clips) — https://huggingface.co/datasets/ivrit-ai/eval-forced-alignment
- Yu et al., *KL-divergence regularized deep neural network adaptation*, ICASSP 2013 — https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/0007893.pdf
