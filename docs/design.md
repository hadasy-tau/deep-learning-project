# Project design

The whole flow in one page, at macro level. Every stage has its own document; this one says
what each stage does and why it exists. Numbers are measured, not planned; the source doc
for each stage is linked at the bottom.

**The question.** Does per-speaker adaptation help a Hebrew ASR model, and for whom? Two
arms answer it: **A** `openai/whisper-large-v3` (general multilingual) and
**B** `ivrit-ai/whisper-large-v3` (Hebrew fine-tune). A is the positive control — without
it, a null result on B would be unreadable.

## The pipeline

**The data.** `ivrit-ai/knesset-committees`: about 15,500 hours of Knesset *committee*
audio, one recording per session, with the official written protocol force-aligned to it and
speaker names given as plain strings. Committees rather than plenums because arm B was
fine-tuned on plenum audio — measuring it on plenums would be measuring train-on-test,
whereas these committee recordings postdate the model and are held out from it. About 10,900
sessions carry usable audio.

**1 · Speaker name → unique id.** The protocol says who holds the floor, but only by name.
That name is resolved to a stable identifier: the Knesset's own `PersonID`, read from the
parliament's public ODATA service, which also supplies each member's name variants and
service dates. The match must be exact — against those variants plus an alias table — and
must fall inside the person's term of service; nothing fuzzy is allowed anywhere, and a name
that cannot be resolved is dropped rather than guessed, because a wrong id is far worse than
no id. Two things make this worth doing once, properly. The id is the same one
`HaifaCLGroup/KnessetCorpus` uses, so the official roster, the demographics and the
protocols all join directly; and it is what every later stage selects audio by. The output
is an index, not audio: about 5.16 million `(session, start, end)` spans, each carrying a
verified id — 3,345 hours of identified speech.

**2 · Index × audio → chunks.** The index and the recordings are combined into the corpus
everything downstream reads: ~1.2 million chunks of at most 30 seconds, each holding exactly
one speaker, with the protocol text alongside as the reference. Chunks tile a speaker's turn
back to back and never cross a speaker change. Roughly 3,840 hours, 330 speakers, 410
parquet shards. Nothing is filtered at build time — that is deliberately the consumer's
decision, made on the per-chunk alignment `quality` score.

**3 · Inference, both arms.** Both models transcribe a selection of those chunks, and the
result is one row per (chunk, arm) holding the hypothesis beside the protocol reference. The
selection here is one hour per speaker, drawn round-robin across sessions so it spans a
speaker's tenure: 65,990 chunks, 230 hours, 267 speakers. One hour is enough because the
per-speaker confidence interval is already down to about ±0.04 WER at 45 minutes, and this
stage only needs to *rank* speakers — transcribing all 3,840 hours would cost roughly $104
for arm A and $150 for arm B and change no decision. Neither arm needs a local GPU: arm A
runs through HF Inference Providers, arm B on RunPod using ivrit.ai's own worker image.

**4 · Per-speaker performance.** Each hypothesis is normalised with the project's Hebrew
normalisation, then compared to the reference by word- and character-level edit distance.
Error *counts* are stored rather than rates, so a speaker's WER stays Σerrors / Σwords no
matter how the rows are later grouped — averaging per-chunk rates would be badly biased at
30 seconds a chunk. The output is a WER and CER per speaker, under each arm.

**5 · Choose who to adapt.** For each speaker, `gain = WER_A − WER_B` is how much the
*generic Hebrew* fine-tune already bought that voice. The speakers with the **lowest** gain
are the ones a general Hebrew model left behind, so that is where per-speaker adaptation has
room to act; high-gain speakers are already well served. The comparison is a gain and not an
absolute WER because absolute WER on this corpus is dominated by register — the reference is
a cleaned stenographic protocol while both models faithfully transcribe the repetitions and
false starts on the tape. Both arms pay that cost equally, so the difference between them is
the signal.

**6 · Fine-tune the chosen speakers.** For each selected speaker, more of their audio goes
through *the same inference code as stage 3* — only the selection filter changes, from "one
hour of every speaker" to "everything left of this one speaker". Their chunks are then split
by session and by date: the latest sessions become the personal test set, the next ones dev,
and the rest are training audio. Splitting on whole sessions means no session ever appears
on both sides, and the test set sits temporally after training. Training produces one adapter
per *cell* — a combination of speaker, arm, where in the network the adapter is attached,
which adaptation method, how many minutes of audio, and the usual hyperparameters — so the
experiment can ask not just whether adaptation works but where in the model it acts and how
much audio it needs.

**7 · Evaluate the adapters.** The base model and the tuned model are scored down the
identical path, on that speaker's held-out sessions, with a paired bootstrap over chunks to
say whether the difference is real. One practical seam shapes this stage: a freshly trained
adapter is not served by the inference providers used in stage 3, so it has to be loaded and
run locally. Comparisons therefore stay inside one engine — A versus B over the providers,
base versus tuned locally — and what makes the two comparisons commensurable is that the
*scoring* is shared: the same Hebrew normalisation, error counts rather than rates,
aggregated the same way.

**The caveat that outlives every stage.** Speaker labels are verified against the protocol,
not against the voice. The protocol records who held the floor, not who was audible, and
committee cross-talk is heavy. If one speaker's WER looks anomalous, suspect the labels
before the model.

## Where to read more

| doc | covers |
|---|---|
| [`committees_handoff.md`](committees_handoff.md) | **read first.** The corpus, how to read it, what is known to be wrong with it |
| [`speaker_index_plan.md`](speaker_index_plan.md) | stage 1: identity resolution, and the measurements that killed the fuzzy-matching design |
| [`chunk_corpus_build.html`](chunk_corpus_build.html) | stage 2: the build record |
| [`inference.md`](inference.md) | stage 3: provider contracts, the three bottlenecks, measured cost |
| [`design.html`](design.html) | the same design as a page, with diagrams — published at https://claude.ai/code/artifact/9460e029-e28f-439c-9ecb-ebf0bde37d2d |
