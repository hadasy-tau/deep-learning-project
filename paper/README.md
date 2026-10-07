# The paper

*Toward Personalized Hebrew ASR: Adapting a General Model to a Single Speaker*, ACL format,
5 pages plus references and appendices. Read this before editing the paper. It holds the
decisions about what the paper says, the words it uses, and where each number comes from.

## Files

| file | what |
|---|---|
| `Toward_Personalized_Hebrew_ASR_overleaf/acl_latex.tex` | **the source.** The version in this repo is the one that counts |
| `Toward_Personalized_Hebrew_ASR_overleaf/custom.bib`, `figures/` | references and figures |
| `Toward_Personalized_Hebrew_ASR.pdf` | the compiled paper |
| `Toward_Personalized_Hebrew_ASR_overleaf.zip` | the same folder zipped, for uploading to Overleaf |
| `Appendix_changes_for_review.pdf` | the 2026-10-07 appendix edits marked against the version before them, for review |

The PDF and the zip are built from the folder. After a change to the folder, rebuild both in the
same commit:

```bash
cd paper/Toward_Personalized_Hebrew_ASR_overleaf && latexmk -pdf -outdir=/tmp/paper acl_latex.tex && cp /tmp/paper/acl_latex.pdf ../Toward_Personalized_Hebrew_ASR.pdf
```

```bash
cd paper/Toward_Personalized_Hebrew_ASR_overleaf && rm -f ../Toward_Personalized_Hebrew_ASR_overleaf.zip && zip -q -r -X ../Toward_Personalized_Hebrew_ASR_overleaf.zip . -x '.DS_Store'
```

Figures go in as RGB, not RGBA, so Overleaf compiles in time.

## Decisions about content

- **Every WER is against the protocol.** The paper says so, and treats the protocol as an edited
  record, not a verbatim transcript. The caveat sits in Limitations, without numbers.
- **No two-model agreement as evidence.** Nothing that reads "both models produced it, so it was
  spoken": no forgiven-shared or "protocol-aware" WER, no semi-verbatim targets as a result, no
  "73 % of insertions are shared" as evidence (`docs/STATUS.md` § Withdrawn).
- **No human transcription.** The paper does not mention a human-corrected check or a human test
  set, as a finding or as future work.
- **Personalization is always own minus the budget-matched control**, on the same segments, with
  its own paired bootstrap. A gain without the control is mostly domain.
- **Per-speaker claims only where all three seeds agree.**
- **Style:** no semicolons in new prose or captions (join with "and" or split the sentence),
  multiple citations written out with "and" (`\citet{a}, \citet{b} and \citet{c}`), no bullet
  lists. Leave a co-author's wording alone unless asked.
- **Wording settled in the 2026-10-07 pass:** what adaptation buys is "shared across speakers"
  (not "the committee setting"), training takes "a few minutes on a standard GPU", the code link
  is the last sentence of the abstract, not a footnote.

## Words: the repo and the paper

The docs and the code use different words from the paper. In the paper, use the right-hand column.

| repo, code | paper |
|---|---|
| arm A, arm B | arm A (general), arm B (Hebrew fine-tune) |
| high-quality test, HQ, `hq`, ≥ 0.95 test | clean test |
| ≥ 0.7 test, `test07.*` | full test |
| chunk, clip | segment |
| session | session (not "meeting", anywhere in the paper) |
| `quality` | alignment quality score |
| own adapter, control (`ctrl*`) | own adapter, control |
| `delta_rel` | relative gain over arm B |
| `personalization_rel` | personalization |
| Stage 1 (error map, pipeline steps 1–5) | first experiment, speaker-level analysis |
| Stage 2 (adaptation, steps 6–7) | second experiment, personalization |
| profiles S1, S2, S3, S4, C | helped least, hard under both, typical, low WER, served well by the fine-tune ("control" is kept for the control adapter only) |
| run 3 | the main experiment (5, 20, 80 minutes) |
| run 4 | beyond 80 minutes |

## Where the numbers come from

| paper section | source |
|---|---|
| Data (corpus span, 13,177 sessions, 15,466 h, 7,886 h at ≥ 0.7) | the `ivrit-ai/knesset-committees` dataset, not computed in this repo |
| Preprocessing (1,204,617 segments, 3,841 h, 330 speakers, 10,905 sessions) | `coverage.parquet` in `knesset-asr/knesset-committees-inference`, measured 2026-10-06 (`docs/committees_handoff.md`) |
| Speaker-level performance (65,990 segments, 230 h, 267 speakers, language forced to Hebrew) | `docs/inference.md`, `docs/design.md` § 3 |
| Speaker-level error map (267 speakers, 58,180 segments, WER 0.387 / 0.292, SD, CV, ρ, groups) | `docs/error_map.md`, `notebooks/committees_error_map_v2.ipynb`, `src/evaluation/outputs/committees_*.csv` |
| Reliability (0.78, 0.75), recording conditions by year | `notebooks/committees_error_map_v2.ipynb` |
| Appendix A, quality filters (7,810 of 65,990 removed) | `docs/error_map.md`, `docs/training_plan_v3.md` § 1 |
| Appendix A, quality predicts both arms' errors (Spearman −0.62 for A, −0.63 for B) | recomputed 2026-10-07 with `error_map.load_chunks` and `count_errors` over the 65,990 segments |
| Appendix B, the loop guard (compression ratio 3.0, 6-token sequences) | `src/evaluation/evaluate.py` (`LOOP_CR`, `LOOP_NGRAM`) |
| Adaptation results, 5 / 20 / 80 minutes, seeds, bands | `docs/training_run3.md`, `notebooks/training_run3_figures.ipynb`, `src/training/outputs/results_v3.csv` |
| Tuning (64 runs, the rule, the chosen recipe) | `src/training/outputs/tuning_v3.csv`, `recipe_v3.json`, `docs/training_plan_v3.md` § 2 |
| Beyond 80 minutes, capacity | `docs/training_run4.md`, `notebooks/training_run4_figures.ipynb`, `src/training/outputs/curve_v4.csv`, `capacity_v4.csv`, `results_v4.csv` |
| Limitations: 75–82 % of test speech from committees seen in training | recomputed 2026-10-07: 74.7 % (clean test) and 81.8 % (full test) of the test duration, from `panel_plan_v2.parquet` and `panel_test07.parquet` joined to `committee_name` in `knesset-asr/knesset-committees-speakers` |
| Limitations: about $200 in total | not recorded in the repo. The documented costs add up to about $90 (inference ~$24, run 2 ~13 GPU-h, run 3 ~$19, run 4 ~$16) |
| Related Work, Weninger et al. 2019 (gains grow to 20 h, no other-speaker control) | `docs/archive/personalization_research.md` § 2, citation checked against the ISCA archive |

## Figures

The paper's figure files are copies of the notebooks' outputs, renamed and converted to RGB.

| paper figure | notebook | saved there as |
|---|---|---|
| `error_map_hist`, `error_map_gain`, `error_map_reliability` | `committees_error_map_v2.ipynb` | `per_speaker_hist`, `gain`, `reliability` |
| `group_*` | `committees_error_map_v2.ipynb` | `group_*` |
| `conditions_by_year` | `committees_error_map_v2.ipynb` | `conditions` |
| `stage2_*` except the data curves | `training_run3_figures.ipynb` | the same name without `stage2_` |
| `stage2_data_curve_24h`, `stage2_data_curve_6h` | `training_run4_figures.ipynb` | the same name |

The notebooks write to `docs/figures/<notebook>/`.

## Open in the paper

- The Related Work paragraph on Weninger et al. (typical speakers, no other-speaker control) is
  flagged: keep or drop.
- The $200 cost in Limitations has no source (see above).
- Page limit: after the 2026-10-07 pass the body (through the Conclusion) fits in 5 pages.
