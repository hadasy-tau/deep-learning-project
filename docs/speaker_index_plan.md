# Stage 3 — Speaker identity for the Knesset committees corpus (revised)

Goal, unchanged: give every speaker occurrence in `ivrit-ai/knesset-committees` a stable
`person_id` in the KnessetCorpus / VoxKnesset id space, and through it verified
demographics, so the committees audio can be used the way VoxKnesset is used in
`stage1/` and `stage2/`. Accuracy over coverage: a wrong id is far worse than no id.

This revision replaces the fuzzy-matching design. Everything below is grounded in
measurements taken on the live data (scripts in the session scratchpad:
`jointest.py`, `cutoff.py`, `aligntest3.py`, `spksweep.py`).

## What changed since the previous plan, and why

| Previous plan assumed | Measured | Consequence |
|---|---|---|
| KnessetCorpus protocols stop at K24 | 2,758 K25 committee shards exist, last protocol **2024-03-26** | Protocol join covers ~59% of hours, not 42% |
| Join on `(knesset, committee, session_number, date)` via the 13–24 CSV | The `.doc` inside each session dir *is* the corpus `protocol_name`; ODATA returns the same name (12/12) | Exact string join, no CSV, works for K25 too |
| Corpus `speaker_id` is the gold standard | K25 shards assign real MK ids to guests in **12.3%** of MK-labelled segments (unreviewed gestalt matches to retired MKs); K20: 0% | Gold must itself be validated; never a single source of truth |
| Roster = corpus `all_knesset_members_jsons.jsonl` (140 K25 MKs) | Official ODATA: 151 K25 MKs; **7 absent** from the corpus roster (entered 2025–26); 43 stints start after 2023-03 | Roster and service dates come from ODATA; corpus supplies demographics only |
| Name matching is the primary path (rapidfuzz cascade) | Raw names come from the Knesset editor's speaker picker (`ET_speaker_<id>` bookmarks) → highly standardised; 75% of post-2024 MK names already occur verbatim pre-2024 | Exact matching against roster variants + alias table; **no fuzzy ratio anywhere** |
| Drop segments with >1 speaker (8–40% of time) | Words concatenate char-exactly to segment text (0/15,612) → cut at the speaker boundary | Recovers 83% of multi-speaker time as clean single-speaker pieces |
| Three unrelated id spaces | `person_id` == ODATA `PersonID`; ivrit `session_id` == ODATA `CommitteeSessionID` | One key across VoxKnesset, corpus, ODATA, audio |

Kept from the previous plan: the id space (KnessetCorpus/VoxKnesset), the garbage-name
rejection rules, the alias table (promoted from fallback to primary), `match_method`
exposed per row, quality score stored not filtered, no audio cut at index time, the
`stage2/pipeline.py` normalisation and house style, and the target of cross-person
error < 0.5%.

## Evidence in numbers

Sample: 300 random sessions for the join; 39 joined sessions (15,612 refined segments)
for label agreement; 180 sessions for `speakers.txt`; 150 for directory completeness.

- **Sessions.** 13,176 directories; 17% contain only the `.doc` (no audio/transcripts) →
  ~10,900 usable, matching the previous plan's 11,144.
- **Join by `.doc` basename.** K20 12/12, K21 3/3, K22 2/2, K23 51/51, K24 56/57, K25 55/175.
  K25 misses have zero overlap with hits in `ptv` order: everything after the corpus
  cutoff, nothing before. Hours: ≈6,200 (K20–24) + ≈2,900 (K25 ≤ Mar 2024) ≈ 9,100 h of
  ~15,500 → **59% with a protocol, 41% without**.
- **Text identity.** Corpus sentences vs ivrit `raw.protocol.txt`: 99.6% similar; 82–100%
  of gold sentences found verbatim per session (median ~98%). Both derive from the same
  `.doc`.
- **Label agreement** (MK-labelled, single-speaker segments, n = 5,941):
  agree ∧ active 5,679 (**95.6%**) · conflict ∧ inactive 168 (2.8%, the K25 guest→retired-MK
  errors plus former MKs speaking as guests) · agree ∧ inactive 28 · conflict ∧ active 66
  (1.1%: mid-session chair changes where one parser lags, one impure editor id, one
  guest→MK fuzzy match). After both filters, **0 known-wrong ids survive** in the sample.
- **Editor ids (`ET_*`).** 69 verified 4-digit ids → 1 impure (5768: גינזבורג/ברק). Several
  MKs carry two ids across terms (גפני 4762/5561, מיקי זוהר 5279/5832). Secondary signal only.
- **Roster.** ODATA `KNS_PersonToPosition` K20–25, PositionID 54: 1,102 rows (12 paged
  requests). Corpus demographics coverage: date_of_birth 98.6%, religion 98.2%,
  place_of_birth 98.2%, religious_orientation 56.6%, year_of_aliya 45.5%.
- **Quality score.** `metadata.json.per_segment_quality_scores` = median word probability,
  keyed to the 1,433 *aligned* segments, not the 815 refined ones. Recompute per emitted
  segment from `words[].probability`.
- **Multi-speaker time.** 7.20 h in the sample; 5.94 h (83%) recoverable as single-speaker
  sub-segments ≥ 1 s.

## Design

Two labelling paths feed one segment builder; validation is independent of both text
sources.

```
ODATA (sessions, protocol names, MK service dates)      corpus roster (demographics)
                 │                                                 │
                 ▼                                                 ▼
 Step 0  manifest + roster + name variants ────────────────────────┤
                 │                                                 │
   has corpus shard? ──yes──► Step 1  Path A: text alignment       │
                 │              gold ∧ raw-name ∧ active_on         │
                 └────no───► Step 2  Path B: exact roster/alias     │
                               + editor-id consistency             │
                                          │                        │
                                          ▼                        │
                              Step 3  turn-aware segments ◄────────┘
                                          │
                                          ▼
                              Step 4  validation (holdout sim + audio embeddings)
```

### Step 0 — Manifest, roster, name variants (no HF data downloads)

**Manifest.** List `ivrit-ai/knesset-committees` directories; keep sessions with all of
`audio.m4a, transcript.refined.json, transcript.refined.map.json, speakers.txt,
speakers.segments.txt, metadata.json`. Record `protocol_name` = basename of the `.doc/.docx`
in the directory; cross-check against ODATA `KNS_DocumentCommitteeSession`
(`GroupTypeID eq 23`) and flag mismatches. `session_date`, `knesset_num`,
`committee_name`, `session_number` come from `metadata.json`.

**Roster.** `KNS_Person` + `KNS_PersonToPosition` (`KnessetNum ge 20 and PositionID eq 54`,
page size 100). `active_on(date)` = PersonIDs with a stint covering the date. Join
demographics from `all_knesset_members_jsons.jsonl` on `person_id`; leave nulls for the
MKs the corpus lacks (7 as of Sep 2026) rather than imputing.

**Name variants**, generated deterministically per person: `first last`, `last first`,
nickname form when the roster has `first (nick) last`, hyphen↔space, geresh/quote
stripping, NFKC — through `stage2/pipeline.py:normalize_he()`. Hard-fail if two MKs
active on the same date share a variant (homonyms, e.g. two ישראל כץ in different terms
are fine; same term is not).

### Step 1 — Path A: protocol-aligned labels (sessions with a shard, ~59% of hours)

1. Fetch `protocols_sentences/committee_protocols/data/<k>/<protocol_name>.jsonl.bz2`
   (~100 KB each; only the ~7,700 joined ones, ≈0.8 GB total).
2. Place each gold sentence verbatim in `refined.text` with a monotonic cursor. Unplaced
   sentences are skipped, not fuzzed.
3. Compute word char offsets by accumulating `len(word)` inside each segment (exact).
   For each word: gold speaker = overlapping gold sentence; ivrit speaker = range in
   `speakers.segments.txt`.
4. A word is labelled `person_id = P` only if **all** hold:
   - gold `speaker_id` is numeric and equals P;
   - the ivrit raw name (title and party stripped) matches a name variant of P — strict:
     token sets equal, or one is a ≥2-token subset of the other;
   - P ∈ `active_on(session_date)`.
   Otherwise the word is `non_mk` (gold UUID / `is_valid_speaker` false) or `unresolved`
   (any disagreement). Both are kept in the index with the reason, never assigned.

This step also **produces** the alias tables used by Path B:
`alias_names[(cleaned_raw_name) → person_id]` and `alias_et[(et_id) → person_id]`, each
with occurrence count and purity.

### Step 2 — Path B: sessions without a shard (K25 after 2024-03-26, ~41% of hours)

Raw name → reject garbage (digits, glued `:` dialogue, and — **after** the parenthetical
is stripped — length > 60 or > 8 tokens) → strip role prefix and party suffix. Ministers
appear as `שר הרווחה והביטחון החברתי יעקב מרגי`; therefore match the **trailing** 2–4
tokens as well as the whole string. Then, in order, accept the first that fires:

1. exact match to a name variant of an MK in `active_on(date)`;
2. `alias_names` entry with purity 1.0 and count ≥ 3, and that person active on the date.

If the speaker carries a known `et_id` whose table entry disagrees with the name-based
result → `unresolved`. No rapidfuzz, no thresholds, no margins. Everything else →
`unresolved`. Record `match_method` ∈ {A, B1, B2}.

Two rules here were set by the pilot rather than by design:

- **Measure length after the parenthetical, not before.** `מיכאל מרדכי ביטון (יו"ר הוועדה
  המיוחדת לפיקוח על תהליכי הסרת חסמים)` is 67 characters but names a three-token MK.
  Checking the raw string first discarded him — ~25 minutes in a 50-session pilot.
- **`alias_et` is not an assignment source.** It was rung 3. The holdout scored it at
  **0.47 precision on 26 seconds** of speech, and it produced *every* genuine cross-person
  error in the run (`יורם בן דוד` → יואב בן צור, `דובר` → טלי פלוסקוב) because
  stenographers reuse picker entries. Removing it took the cross-person rate to zero at a
  cost of 26 seconds. It survives only as a veto, where being wrong costs recall, not
  correctness.

### Step 3 — Turn-aware segments

From `refined.json` words: cut wherever the label changes (speaker boundary from
`speakers.segments.txt` or gold), then chunk to ≤ 30 s without crossing a cut; drop
pieces < 1 s. `quality` = median word probability of the piece (ivrit's own definition).
Emit `segments.parquet`, one row per piece:

```
filename, speaker_id, session, start, end, duration_s, reference_text, quality,
label, label_path, match_method, reason, raw_name, local_speaker_id, et_id,
gold_speaker_id, seg, knesset, committee_name, session_date, speaker_name,
gender, age, date_of_birth, place_of_birth, year_of_aliya, religion, nationality,
religious_orientation, is_chairman
```

Names follow VoxKnesset and `stage2/pipeline.py` rather than this plan's first draft:
`session_id → session`, `duration → duration_s`, `text → reference_text`,
`person_id → speaker_id`, plus `filename` = `{speaker_id}_{session}_{start_ms}_{end_ms}.wav`
so Stage 2's regex parses the index unchanged. `label`, `reason`, `speaker_name` and
`gold_speaker_id` carry the decision and its evidence. `is_chairman` is derived from the
title prefix in `raw_name`, which works on both paths — the gold field exists only for
Path A. Types are pinned by an explicit `SCHEMA`: `year_of_aliya` is `'1950'` for some MKs
and empty for others, so per-batch inference produced parts that would not concatenate.

`age = (session_date − date_of_birth) / 365.25` as in VoxKnesset. No audio is cut;
`materialize()` in `stage2/pipeline.py` already knows how to slice from
`session/start/end`.

### Step 4 — Validation, independent of both text sources

1. **Holdout simulation (text).** Run Path B on Path-A sessions as if they had no shard.
   Score against Path-A labels: precision, recall, and cross-person rate per knesset and
   per `match_method`. This is the old plan's Step 3, now against a *validated* gold.
2. **Audio check (the real gate).** Extract speaker embeddings (ECAPA-TDNN, as VoxKnesset
   benchmarks) for a stratified sample of labelled segments; per `person_id` centroid;
   flag segments whose cosine to their own centroid is an outlier and whose nearest
   centroid is another person → review list and a measured contamination rate. For MKs
   also in VoxKnesset, compare committee centroid to the plenum centroid of the same
   `person_id`. This catches the one failure both text sources share: both copying the
   same wrong header.
3. **Ear check.** ~10 segments per path via `stage0/explore.ipynb`.
4. **Consistency.** `person_id` and derived `age` agree with VoxKnesset columns for
   overlapping people and dates.

Acceptance: audio-estimated cross-person < 0.5% on labelled segments; Path-B precision
≥ 99% on the holdout simulation; the K25 guest→retired-MK cases all land in
`unresolved`/`non_mk` (regression test with the five named cases).

**Holdout result (200 sessions, 55.4 h assigned).** Precision **1.000**, cross-person
**0.000**, recall 0.9999, zero cross-person cases in any Knesset or method. The regression
passes: all ten corpus guest→retired-MK errors stay unassigned.

Scoring this needed a correction. A word where Path B says MK and the corpus says guest is
not automatically an error — the corpus paper reports ~19.7k false negatives, and 44 of 47
such cases here were real MKs it had failed to match (`גדעון סער`, `אחמד טיבי`,
`היו"ר משה גפני`). They are split by whether the protocol's own speaker line names the
person B chose, and reported as `corpus_missed`: **0.75 h that Path B recovered and the
corpus had lost.** The three that were genuinely wrong were all `alias_et`, now removed.

Recall of 1.000 is structural, not luck: Path A requires gold ∧ name ∧ seat, Path B
requires name ∧ seat, so A's assignments are a strict subset of B's. The only open
question was whether B's *extra* assignments are right.

**Step 4.2 has not run.** It needs a GPU. The text gate passing does not substitute for
it — it is the only check that catches both sources copying the same wrong header.

## Files

New `stage3/`, house style (plain functions, driven from a notebook, self-checks):

| File | Holds |
|---|---|
| `stage3/roster.py` | ODATA pull (paged, ordered, cached), `active_on(date)`, demographics join, name variants |
| `stage3/manifest.py` | session inventory, the `.doc` → shard join, `label_path` per session |
| `stage3/names.py` | cleaning rules, strict `agree()`, `match_exact()`, alias tables — imports `normalize_he` |
| `stage3/protocol.py` | session and shard fetch, monotonic placement, verified word offsets |
| `stage3/label.py` | Path A and Path B word labelling |
| `stage3/segments.py` | turn-aware cutting, quality, `SCHEMA`, `segments.parquet` |
| `stage3/validate.py` | holdout simulation, regression, report |
| `stage3/run_pilot.py` | 200 Path-A + 50 Path-B sessions, learns the alias tables |
| `stage3/run_full.py` | full build: threaded, batched, resumable |
| `stage3/publish.py` | dataset card and upload — dry-run by default |
| `stage3/outputs/` | `manifest.csv`, `mk_metadata.csv`, `mk_stints.csv`, `mk_name_variants.csv`, `alias_names.csv`, `alias_et.csv`, `segments.parquet`, `match_report.txt`, `holdout_report.txt` |

Data volume: per-session small files only (`refined.json` ≈ 3 MB × 11.1k ≈ 33 GB; streamed
and discarded — only the pilot's sessions are cached, for `validate.py` to re-read);
shards ≈ 0.8 GB; ODATA ≈ 30 requests. No GPU except the embedding check.

The full build needs three things the pilot did not: a **thread pool** (the work is
network-bound), **batches of 400** written to `outputs/parts/` so memory stays flat
instead of holding ~5M rows, and **resume** — a batch whose part and stats files both
exist is skipped, so an interruption costs at most one batch. Fetching is the binding
constraint: `hf_hub_download` issues a HEAD before every GET, so four small files per
session cost eight hub-API calls, and twelve workers drew HTTP 429s within two batches.
The small files now go over `resolve/main` (five calls per session) with six workers and
jittered backoff.

Run order: roster → manifest → Path A on a 200-session pilot → build alias tables →
Path B on 50 post-2024 sessions → holdout simulation → full build (which relearns the
alias tables from all ~7,000 Path-A sessions) → audio check.

## Self-checks

- `roster.py`: PersonIDs unique; `active_on` returns 120–150 for a date in each of
  K20–K25; 526 (גפני) active on 2018-11-06; 30894 (סמיר בן סעיד) active on 2025-07-01 and
  not on 2025-06-01.
- `names.py`: the garbage cases reject; `אורית פרקש הכהן`/`אורית פרקש-הכהן` and the three
  `אלעזר שטרן (…)` spellings resolve to one id; `אברהם (אבי) ניסנקורן` matches
  `אבי ניסנקורן`; `מיכל חסון` does **not** match יואל חסון.
- `protocol.py`: `''.join(words) == segment.text` **and** the segment's text is where its
  `start_char` says it is; words in a segment that fails carry `offset_ok=False` and are
  never given an identity; a session below 90% verified is skipped. The earlier check —
  asserting `end_char` arithmetic — was wrong: map spans disagree with text lengths by
  1–5 characters in ~1.5% of sessions, yet one such session placed **5/5** segments
  correctly and another 36/39. Word offsets are anchored at each segment's own
  `start_char`, so what matters is whether the text is really there.
- `label.py`: session 2073683 yields 403 `mk` runs, 0 `name_disagree`, and the same six
  MKs under both paths; the ten K25 guest→retired-MK cases are `unresolved`; no editor id
  ever assigns.
- `segments.py`: the `SCHEMA` survives an all-guest batch, where every demographic is null.
- `segments.py`: pieces are within their parent segment, monotonic, ≥ 1 s, ≤ 30 s, and
  the sum of piece durations ≤ parent duration.

## Risks

- **Chair changes mid-session.** One parser lags the other by a turn; the agreement rule
  turns those words `unresolved`. Accepted loss; the `(היו"ר …, hh:mm)` marker could
  later be used to re-split.
- **Sentences that don't place verbatim** (≈2–18% per session, from stenographic
  normalisation): dropped, not fuzzed. Could be recovered later with anchored fuzzy
  placement between two verbatim neighbours.
- **New MKs without demographics** (7 as of Sep 2026): rows carry `person_id` and nulls;
  a Wikipedia pass is out of scope.
- **Editor ids are impure** (5768, and 4% in the raw sweep): never used alone.
- **ODATA paging and availability**: page size is 100; cache every response to disk.
  `$skip` alone overlaps — the first manifest run returned 13,188 sessions for 13,176
  directories. Every query is ordered by its key and deduplicated on it.
- **HuggingFace rate limits**: the build is one long burst of small reads. Use
  `resolve/main` rather than `hf_hub_download` (which HEADs before every GET), keep the
  pool small, and jitter the backoff. Resume makes a throttled run cheap to restart.
- **Corpus updates.** If HaifaCLGroup extends beyond March 2024, Path A coverage grows
  automatically; re-run the manifest join.

## Decisions (6 Sep 2026)

- **Publish the index only** — `segments.parquet` + side tables, no audio. Audio is
  materialized locally from `ivrit-ai/knesset-committees` by `session/start/end`.
- **Former MKs speaking as guests** get `label = former_mk` with a full `speaker_id` and
  demographics; `label == 'mk'` stays "MK on the recording date".
- **Dataset** `Dolevabudi/knesset-committees-speakers`, private until Step 4 passes.
  README credits ivrit.ai and the Knesset Corpus; licence CC-BY-SA 4.0 (inherited).
- **Runs on the local Mac** in `.venv` (pandas, pyarrow, huggingface_hub, requests,
  rapidfuzz). No GPU except the optional Step 4 embedding check.
- `speaker_id = 0` for `non_mk`/`unresolved` rows so `filename` stays parseable by
  `stage2/pipeline.py`'s regex.

## Note

Read `HF_TOKEN` from the environment; never embed it. The token pasted into an earlier
chat should be rotated at https://huggingface.co/settings/tokens.
