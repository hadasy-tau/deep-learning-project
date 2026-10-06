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
- **Style:** no semicolons in new prose (join with "and" or split the sentence), multiple citations
  joined with "and". Leave a co-author's wording alone unless asked.

## Words: the repo and the paper

The docs and the code use different words from the paper. In the paper, use the right-hand column.

| repo, code | paper |
|---|---|
| arm A, arm B | arm A (general), arm B (Hebrew fine-tune) |
| high-quality test, HQ, `hq`, ≥ 0.95 test | clean test |
| ≥ 0.7 test, `test07.*` | full test |
| chunk, clip | segment |
| session | meeting (personalization), session (error map, data) |
| `quality` | alignment quality score |
| own adapter, control (`ctrl*`) | own adapter, control |
| `delta_rel` | relative gain over arm B |
| `personalization_rel` | personalization |
| Stage 1 (error map, pipeline steps 1–5) | first experiment, speaker-level analysis |
| Stage 2 (adaptation, steps 6–7) | second experiment, personalization |
| profiles S1, S2, S3, S4, C | helped least, hard under both, typical, low WER, high-gain control |
| run 3 | the main experiment (5, 20, 80 minutes) |
| run 4 | beyond 80 minutes |

## Where the numbers come from

| paper section | source |
|---|---|
| Data (corpus span, 13,177 sessions, 15,466 h, 7,886 h at ≥ 0.7) | the `ivrit-ai/knesset-committees` dataset, not computed in this repo |
| Speaker-level error map (267 speakers, 58,180 segments, WER 0.387 / 0.292, SD, CV, ρ, groups) | `docs/error_map.md`, `notebooks/committees_error_map_v2.ipynb`, `src/evaluation/outputs/committees_*.csv` |
| Reliability (0.78, 0.75), recording conditions by year | `notebooks/committees_error_map_v2.ipynb` |
| Appendix A, quality filters (7,810 of 65,990 removed) | `docs/error_map.md`, `docs/training_plan_v3.md` § 1 |
| Adaptation results, 5 / 20 / 80 minutes, seeds, bands | `docs/training_run3.md`, `notebooks/training_run3_figures.ipynb`, `src/training/outputs/results_v3.csv` |
| Tuning (64 runs, the rule, the chosen recipe) | `src/training/outputs/tuning_v3.csv`, `recipe_v3.json`, `docs/training_plan_v3.md` § 2 |
| Beyond 80 minutes, the decision rule, capacity | `docs/training_run4.md`, `notebooks/training_run4_figures.ipynb`, `src/training/outputs/curve_v4.csv`, `capacity_v4.csv`, `results_v4.csv` |
| Limitations: 75–82 % of test speech from committees seen in training, about $200 in total | not recorded in the repo |

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

- The abstract and the introduction's contributions are still TODO. The abstract's TODO note
  still says "speaker-level diagnosis on VoxKnesset", but the corpus is the committees.
- Experimental Setup § Overview says the 12 speakers cover "four profiles ... plus two high-gain
  speakers as controls", and § Speaker selection says "five profiles". Pick one.
- Limitations has two paragraphs on the same thing ("The reference is the protocol" and
  "Reference transcripts"). Keep one.
- The "Decision rule" paragraph in the appendix is marked as possibly droppable.
- The two sources marked "not recorded in the repo" above.
