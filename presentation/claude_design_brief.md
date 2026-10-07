# Brief for Claude Design: the project presentation

## The task

Design a 16:9 slide deck for a 15-minute oral presentation of a deep learning course project at
Tel Aviv University. Two students present, about 7 minutes each. The deck will be exported to
PowerPoint (PPTX) and submitted with the paper.

The deck supports the talk. It does not replace it. The examiners grade how well we explain the
work in our own words, and they lower the grade if we read text from the screen. So the slides
carry figures, diagrams, results and examples, and very little text.

Paper: *Toward Personalized Hebrew ASR: Adapting a General Model to a Single Speaker*, by Dolev
Abudi and Hadas Yonat (attached as a PDF). Every fact on a slide must come from this paper or from
a paper it cites. Do not invent numbers, names or claims. If something is missing, leave a visible
placeholder marked `[TODO]`.

## Rules for every slide

1. Short sentences. At most about 25 words of body text per slide, numbers excluded.
2. No invented data. Use only the numbers given in this brief or in the attached paper.
3. Both names, small, in the footer of every slide: `Dolev Abudi · Hadas Yonat`.
4. No slide numbers.
5. Where a slide uses a fact from a paper, put a short reference in the footer, for example
   `Radford et al., 2023` or `Our paper, §4.2, Fig. 4`.
6. If a slide becomes crowded, split it or move the detail to a backup slide.
7. 12 main slides at most. Backup slides come after the last main slide, under a divider titled
   "Backup", and are shown only when a question needs them.
8. Titles state the point of the slide ("The fine-tune helps everyone, but the gaps stay"), not
   only its topic.
9. Use the attached figures as they are. Do not redraw them with different numbers. Crop or
   enlarge them so the labels can be read from the back of a room.
10. Speaker notes are keywords only, never full sentences, so we cannot read them aloud. Each slide's
    notes include a line "Why / alternatives" with the decision behind the slide.
11. Slides in English. We speak in Hebrew.

## Words to use

Use the paper's terms and no others: *segment* (not chunk or clip), *session* (not meeting), *arm A*
(general Whisper) and *arm B* (the Hebrew fine-tune), *own adapter*, *control* (control adapter),
*personalization* (own gain minus control gain), *clean test* (alignment quality ≥ 0.95) and
*full test* (quality ≥ 0.7), *alignment quality score*.

Never use or mention: a "forgiven" or "protocol-aware" WER, two models agreeing as proof that the
protocol is wrong, or any human transcription or human check. These are not part of the work.

## Visual style

- Clean, academic and calm. White or very light background, dark text, generous white space.
- Neutral chrome: a dark navy (`#1E2A3A`) for titles, a mid gray (`#6B7280`) for footers and
  secondary text. No gradients, no stock photos, no decorative icons.
- Data colors must match the attached figures, because the figures cannot be recolored:
  - Experiment 1 slides (models, inference, speaker-level analysis): arm A blue `#1F77B4`, arm B
    orange `#FF7F0E`.
  - Experiment 2 slides (adaptation and results): own adapter blue `#0072B2`, control orange
    `#D55E00`, personalization green `#009E73`.
  - In diagrams we draw ourselves, use the same colors for the same things.
- One sans-serif typeface. Large numbers for the key results.

## Attachments

The paper and the figures are attached to this message. They are also public at these links:

- The paper: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR.pdf
- Slide 7: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/error_map_hist.png
- Slide 7: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/group_rate_band.png
- Slide 9: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/stage2_winning_learning_curves_compact.png
- Slide 10: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/stage2_budget_own_control_personalization.png
- Slide 11: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/stage2_data_curve_24h.png
- Backup B3: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/stage2_personalization_per_speaker_seeds.png
- Backup B2: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/stage2_tuning_scores.png
- Backup B4: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/error_map_gain.png
- Backup B4: https://raw.githubusercontent.com/hadasy-tau/deep-learning-project/main/paper/Toward_Personalized_Hebrew_ASR_overleaf/figures/error_map_reliability.png

## The slides

Times are speaking times. They add up to 14 minutes, which leaves one minute of slack.
Dolev presents slide 1 and slides 8–12. Hadas presents slides 2–7. Each speaks about 7 minutes.

---

### 1. Title (Dolev, 15 s)

- **On screen:** the paper title, *Toward Personalized Hebrew ASR: Adapting a General Model to a
  Single Speaker*. Below it: Dolev Abudi, Hadas Yonat, Tel Aviv University.
- **Visual:** none, or a quiet waveform line as the only decoration.
- **Notes:** who we are, the one-sentence question: can a Hebrew ASR model be adapted to one
  speaker?

---

### 2. Goal: three questions (Hadas, 1 min)

- **Title:** "ASR is judged on average. People use it one at a time."
- **On screen:** three numbered questions.
  1. Does a Hebrew fine-tune serve all speakers equally?
  2. What does a speaker's own speech add over the same amount of other speakers' speech?
  3. Does that personal gain grow with more of the speaker's data?
- **Visual:** a small icon-free illustration of one average bar against many per-speaker bars, or
  nothing beyond the three questions.
- **Footer:** Our paper, §1.
- **Notes:** averages hide the speakers a model serves poorly. Deployments that serve one person,
  e.g. a Knesset member transcribed every day. Why / alternatives: personalization means adapting a
  general model with a small amount of one person's speech.

---

### 3. Project flow (Hadas, 45 s)

- **Title:** "Two experiments, one pipeline"
- **Visual:** a left-to-right flow diagram with six boxes:
  Data preprocessing → Inference (arm A, arm B) → Speaker-level analysis → Choosing 12 speakers →
  Adaptation (own adapter vs. control) → Evaluation.
  A bracket under the first three boxes: "Experiment 1: speaker-level analysis. Led by Hadas
  Yonat". A bracket under the last three: "Experiment 2: personalization. Led by Dolev Abudi".
  One small line under both: "Designed together, decisions made together".
- **Footer:** Our paper, §3.1 and Appendix "Division of Work".
- **Notes:** the map of the talk. The output of experiment 1 (who is served poorly) chooses the
  speakers for experiment 2.

---

### 4. The models (Hadas, 1 min)

- **Title:** "One architecture, two models"
- **Visual:** a simple Whisper diagram: audio (≤ 30 s) → log-Mel spectrogram → Transformer encoder →
  Transformer decoder → Hebrew text. Next to it, two labeled cards:
  - **Arm A**, `openai/whisper-large-v3`: general, multilingual. Blue.
  - **Arm B**, `ivrit-ai/whisper-large-v3`: the same model fine-tuned on ~4,700 h of Hebrew
    *plenary* speech. Orange.
- **On screen, small:** encoder–decoder Transformer, 32 + 32 layers, ~1.55B parameters. Every
  adapter in this work is trained on top of arm B.
- **Footer:** Radford et al., 2023 · Marmor et al., 2025 · Our paper, §3.3.
- **Notes:** why two arms: arm A is the starting point the fine-tune is measured from. The gain
  A → B, per speaker, is what experiment 1 maps. Why committees: B was trained on plenary sessions, so testing B
  on plenary audio would be testing on its training data. Committee recordings are new to both
  arms. The speakers are not new (see Limitations).

---

### 5. Data preprocessing (Hadas, 1.5 min)

- **Title:** "From 15,466 hours of sessions to labeled 30-second segments"
- **Visual:** a three-step pipeline with an input and an output.
  - **Input:** `ivrit-ai/knesset-committees`: 13,177 sessions, 15,466 h, Nov 2018 to Jan 2026,
    protocols force-aligned to the audio.
  - **Step 1, speaker ID:** protocol names matched exactly to the Knesset roster, only within
    the member's term. Unresolved names dropped. Age, gender and origin joined.
  - **Step 2, segments:** each speaker turn cut into segments of ≤ 30 s that never cross a
    speaker change. Reference = the aligned protocol text.
  - **Step 3, alignment quality score (0–1):** the median word probability from the aligner.
    Below 0.7: excluded. Training and validation: ≥ 0.95.
  - **Output:** `knesset-asr/knesset-committees-chunks`: 1,204,617 segments, 3,841 h, 330
    speakers, 10,905 sessions.
- **Footer:** Our paper, §3.2, §3.4, Appendix A · Marmor et al., 2026 (VoxKnesset procedure).
- **Notes:** why 30 s: Whisper's input limit. Why verified speakers: a speaker-level analysis
  needs every segment tied to one person. Why a quality score: the protocol is stenographic, not
  verbatim, so a low score means text and audio disagree. 0.95 follows ivrit.ai's recommendation.

---

### 6. Inference (Hadas, 1 min 15 s)

- **Title:** "Both models transcribe the same 230 hours"
- **Visual, left:** two columns, one per arm.
  - **Arm A:** Hugging Face Inference Providers.
  - **Arm B:** RunPod endpoint with ivrit.ai's inference checkpoint (`whisper-large-v3-turbo-ct2`).
  - Both told the language is Hebrew.
  - Under them: 1 h per speaker, spread over their sessions → 65,990 segments, 230 h, 267
    speakers → `knesset-asr/knesset-committees-inference`.
- **Visual, right:** one real example. A play button for the audio, then three lines: the protocol
  reference, arm A's output, arm B's output, with the wrong words highlighted.
  `[TODO: audio example, a real segment from knesset-committees-inference, to be chosen]`
- **Big numbers:** corpus WER, arm A 0.417, arm B 0.324.
- **Footer:** Our paper, §3.5, Appendix A.
- **Notes:** WER is measured against the protocol, an edited record, not a word-for-word transcript.
  So part of every error is protocol vs. audio, not a recognition mistake. Why paid APIs: no
  GPU of our own, cost-bounded and resumable. Arm B's served checkpoint has 4 decoder layers
  instead of 32.

---

### 7. Speaker-level analysis (Hadas, 1.5 min)

- **Title:** "The fine-tune helps everyone, but the gaps stay"
- **On screen, top:** Question: does the Hebrew fine-tune serve all speakers equally, and does a
  speaker's group predict who it serves well?
- **Visual:** `error_map_hist.png`, large. Beside it, `group_rate_band.png`, small.
- **Key facts, as three short lines:**
  - All 261 speakers improve under arm B, every one beyond noise.
  - Per-speaker spread shrinks only in proportion (CV 0.25 → 0.23). Speakers keep their order
    (Spearman ρ = 0.90).
  - Groups explain little. The clearest: slow speakers gain 27 %, fast 22 %.
- **Footer:** Our paper, §4.1, Fig. 1–2. 267 speakers, 58,180 segments after the quality filter,
  WER 0.387 → 0.292.
- **Notes:** method: WER per speaker under A and B, gain and relative gain, 95 % bootstrap interval
  per speaker, seven groupings (gender, nationality, religion, orientation, origin, age, speaking
  rate). Statistics over the 261 speakers with ≥ 20 segments. Variation within a group is larger
  than between groups → the group does not tell you how well the model serves you → adapt to the
  individual. Why / alternatives: a group-level fix (e.g. a model per group) would not help, which
  is why we went to single speakers.

---

### 8. Own adapter vs. control (Dolev, 1.5 min)

- **Title:** "Separating the voice from the room"
- **Visual, left: the 12 speakers.** A small grid of 12 tiles grouped by profile, no names except
  three examples:
  - Helped least by the fine-tune (3)
  - Hard under both models (3), e.g. David Bitan
  - Typical (3), e.g. Idit Silman
  - Low WER (1)
  - Served well by the fine-tune (2), e.g. Walid Taha
- **Visual, right: the control design.** The 12 speakers split at random into two folds of 6.
  Speaker X's **own adapter** (blue) trains on X's minutes. The **control** (orange) trains on the
  same number of minutes from the 6 speakers of the other fold. Both are scored on the same test
  segments of X.
- **Formula, large:** personalization = own gain − control gain (green).
- **Small line:** 5, 20 and 80 minutes of training audio. The newest sessions are held out for
  testing.
- **Footer:** Our paper, §3.1, §3.6.
- **Notes:** an adapter on a speaker's audio learns two things: the speaker, and committee speech in
  general (rooms, microphones, vocabulary, protocol style). Only the control can tell them apart.
  A gain without a control is mostly domain. Why the newest sessions for testing: speakers were
  ranked on earlier sessions, so we do not select on noise. Train, validation and test share no
  session. Two test sets: clean (≥ 0.95, median 47 min per speaker) and full (≥ 0.7, median 90
  min), because a 0.95 test filter favours easy speech.

---

### 9. LoRA and the training recipe (Dolev, 1 min 15 s)

- **Title:** "A small adapter, a few minutes on one GPU"
- **Visual, left:** a LoRA diagram: a frozen weight matrix W, plus a trainable low-rank path B·A
  (rank 8) added to it. Placed on the query and value projections of every attention layer.
- **On screen, right, as a short table:**
  - Rank 8, on q and v → 3.9M trainable parameters
  - Learning rate 1e-4 at 5 min, 3e-4 at 20 and 80 min
  - ≤ 400 steps, early stopping on validation loss
  - Augmentation: tested, no gain
- **Visual, bottom:** `stage2_winning_learning_curves_compact.png`, small.
- **Footer:** Hu et al., 2022 · Our paper, §3.6, Appendix B, Fig. 7.
- **Notes:** tuning: 64 runs, 4 speakers (one per profile), validation loss only, test never
  touched. A change replaced the default only if it helped clearly (+0.01 on average and for 3 of 4
  speakers). Rank 16 = no help. SpecAugment and tempo ±10 % = below the threshold. No pitch or
  speed change, it would change the voice. Whisper's own dropout 0.1 prevents learning. Labels =
  protocol text. Validation loss bottoms out at step 60–100, about two passes, 38–54 % below arm B.
  Why LoRA and not full fine-tuning: little data per speaker, cheap, one small adapter per person.

---

### 10. Results: 5, 20 and 80 minutes (Dolev, 2 min)

- **Title:** "Most of the gain is shared. The personal part is 2–3 %."
- **Visual:** `stage2_budget_own_control_personalization.png`, full width.
- **Big numbers, at 80 minutes (clean test, seed 0):** own adapter +12.0 %, control +10.1 %.
  Under them: personalization +2.4 % (median over 36 speaker–seed pairs).
- **One line:** the own adapter beats arm B in 35 of 36 speaker–seed pairs on the clean test.
- **Footer:** Our paper, §4.2, Fig. 3. Relative WER change over arm B. Full test: own +15.3 %,
  control +14.3 %, personalization +2.2 %.
- **Notes:** how to read the plot: each dot a speaker, bar = median, blue own, orange control,
  green the difference. At 5 min both are unreliable. The control rises almost as fast as the
  own adapter, so extra audio buys mostly non-personal gain. Per speaker: 5 speakers significant
  in all three seeds on the full test, led by 23558 (David Bitan) at +9 to +10 %. 6 never. Profile
  does not predict it. Per-speaker claims only where all three seeds agree.

---

### 11. Beyond 80 minutes (Dolev, 1 min 15 s)

- **Title:** "Up to 24 hours: the own adapter improves, the personal margin does not"
- **Visual:** `stage2_data_curve_24h.png`, large.
- **Key facts, two lines:**
  - Speaker 23558, full test, 80 min → 24 h: own +29.4 % → +34.3 %, control +19.7 % → +30.2 %.
  - Personalization at 24 h: +4.1 % (23558), +2.3 % (23641).
- **Footer:** Our paper, §4.2, Fig. 6, Appendix B.
- **Notes:** why this test: maybe 80 minutes was too little to learn a voice. Two speakers up to
  1,440 min: 23558 (largest personal effect) and 23641 (one of the highest WERs, outside the 12).
  Each budget has its own control. Validation loss stops improving after about one pass. Rank 32
  does not help. Two more speakers up to 360 min, same pattern. Personalization shrinks as a share
  of the gain, for 23558 from a third to an eighth, but does not vanish.

---

### 12. Conclusions (Dolev, 45 s)

- **Title:** "A claim of personalization needs a control"
- **On screen, four short lines:**
  - The Hebrew fine-tune lowers every speaker's error, but leaves them as unequal.
  - 80 minutes of own speech: −12 % WER. Others' speech: −10 %. Personal: 2–3 %.
  - More data → more gain, but not more personal gain.
  - Adaptation is cheap: about two passes, a few minutes on a standard GPU.
- **Small line:** next: similar rather than random speakers for the control, objectives that
  separate the voice from the setting.
- **Footer:** Our paper, §5. Code: github.com/hadasy-tau/deep-learning-project
- **Notes:** the takeaway in one sentence. Then thanks and questions.

---

## Backup slides

After a divider slide titled "Backup". Same style, same footer.

- **B1. Hyperparameters:** the full table from Appendix B (Table "Training hyperparameters").
- **B2. Tuning:** `stage2_tuning_scores.png` and the four tuning findings from Appendix B.
- **B3. Per speaker and seed:** `stage2_personalization_per_speaker_seeds.png`.
- **B4. Gain per speaker and reliability:** `error_map_gain.png`, `error_map_reliability.png`
  (a speaker's two alternating halves of sessions agree: reliability 0.78 for WER under arm B and
  0.75 for relative gain, Appendix D).
- **B5. Quality filters:** the two thresholds (0.7 and 0.95), the four training-data conditions,
  and why there are two test sets (Appendix A).
- **B6. Limitations:** WER against the protocol, prior exposure of arm B to these voices, test
  speech mostly from committees also seen in training (about 75–82 %), a random control group
  (Limitations section).

## References for the footers

- Radford et al., 2023. *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML.
- Marmor et al., 2025. *Building an Accurate Open-Source Hebrew ASR System through Crowdsourcing.*
  Interspeech.
- Marmor et al., 2026. *VoxKnesset: A Large-Scale Longitudinal Hebrew Speech Dataset for Aging
  Speaker Modeling.* arXiv:2603.01270.
- Hu et al., 2022. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Abudi and Yonat, 2026. *Toward Personalized Hebrew ASR: Adapting a General Model to a Single
  Speaker.* (our paper)
