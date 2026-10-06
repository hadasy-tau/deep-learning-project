"""Semi-verbatim training targets.

**Rests on a withdrawn assumption.**  A word goes back when *both* base models produced it,
on the reading that two models agreeing against the protocol means the protocol dropped a
spoken word (docs/STATUS.md § Withdrawn), so these targets may put back unspoken words as
well as spoken ones.  Kept so the second and third runs'
`verbatim` cells stay reproducible; do not read their results as evidence about what the
speaker said.  The forgiven-shared count that used to live here is removed.

docs/archive/training_next.md § A1.  The reference is a cleaned protocol: words the speaker said
and the stenographer dropped count as insertions against both models, and a model trained
toward the protocol learns to drop them too (docs/training_run2.md: 34 % of the base
model's errors are insertions arm A also produced).  This builds a target closer to what
was said: the protocol text with the words *both* base models produced at the same place
put back.  Only insertions are added; nothing is removed, so a protocol word the models
missed stays -- deletions are where recognition is hard, which is the thing under study.

    build_targets(P, results_dir)   -> DataFrame chunk_id, part, text, text_verbatim, n_ref, n_inserted

The insertion logic is error_analysis.forgiven_counts' (an inserted word counts as shared
if the other arm's hypothesis holds it, multiset), applied to arm B's alignment against the
reference and written back into the raw text at the aligned position.  Tokens are aligned on
their normalize_he key so punctuation and geresh differences do not block a match, and the
raw hypothesis token (with its punctuation) is what gets inserted.
"""
import collections, json, os, sys
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(ROOT, 'src', 'evaluation'))
from common import normalize_he

ARMS = {'A': 'openai/whisper-large-v3', 'B': 'ivrit-ai/whisper-large-v3'}


def _keyed(text):
    """Raw whitespace tokens with their normalised keys; tokens whose key is empty
    (punctuation only) are kept in the raw list but left out of the alignment."""
    raw = text.split()
    keys = [normalize_he(t) for t in raw]
    idx = [i for i, k in enumerate(keys) if k]
    return raw, [keys[i] for i in idx], idx


def verbatim_target(ref, hyp_b, hyp_a):
    """The reference with arm B's shared insertions put back.  Returns (text, n_inserted).

    Consecutive non-equal alignment ops are merged into one mismatch block, because
    which of a block's extra words the aligner labels 'substitution' and which
    'insertion' is arbitrary.  In a block with k reference words and m > k hypothesis
    words, up to m - k of the hypothesis words that arm A also heard go back into the
    text, after the block; the substitutions are then the words A did not hear."""
    raw_r, keys_r, idx_r = _keyed(ref)
    raw_b, keys_b, _ = _keyed(hyp_b)
    pool = collections.Counter(normalize_he(hyp_a).split())
    blocks, cur = [], None
    for op, i1, i2, j1, j2 in Levenshtein.opcodes(keys_r, keys_b):
        if op == 'equal':
            if cur: blocks.append(cur); cur = None
        else:
            cur = [i1, i2, j1, j2] if cur is None else [cur[0], i2, cur[2], j2]
    if cur: blocks.append(cur)
    inserts = collections.defaultdict(list)              # raw ref position -> tokens to put before it
    for i1, i2, j1, j2 in blocks:
        excess = (j2 - j1) - (i2 - i1)
        if excess <= 0: continue
        pos = idx_r[i2] if i2 < len(idx_r) else len(raw_r)
        for j in range(j1, j2):
            if excess == 0: break
            words = keys_b[j].split()
            if words and all(pool[w] > 0 for w in words):
                for w in words: pool[w] -= 1
                inserts[pos].append(raw_b[j]); excess -= 1
    out, n = [], 0
    for i, tok in enumerate(raw_r):
        if i in inserts: out += inserts[i]; n += len(inserts[i])
        out.append(tok)
    if len(raw_r) in inserts: out += inserts[len(raw_r)]; n += len(inserts[len(raw_r)])
    return ' '.join(out), n


def _load_hyps(results_dir, arm, speaker, part):
    tag = ARMS[arm].replace('/', '__')
    p = os.path.join(results_dir, f'hyps_{arm}_{speaker}_{part}_{tag}.json')
    d = json.load(open(p, encoding='utf-8'))
    return dict(zip(d['chunk_ids'], d['hyps']))


def build_targets(P, results_dir, parts=('train', 'dev')):
    rows = []
    for sid in sorted(int(s) for s in P.speaker_id.unique()):
        for part in parts:
            g = P[(P.speaker_id == sid) & (P.part == part)]
            ha, hb = _load_hyps(results_dir, 'A', sid, part), _load_hyps(results_dir, 'B', sid, part)
            for r in g.itertuples():
                tv, n = verbatim_target(r.text, hb[r.chunk_id], ha[r.chunk_id])
                rows.append(dict(chunk_id=r.chunk_id, speaker_id=sid, part=part, text=r.text, text_verbatim=tv,
                                 n_ref=len(r.text.split()), n_inserted=n))
    T = pd.DataFrame(rows)
    return T


def report(T):
    by = T.groupby('speaker_id').apply(lambda d: pd.Series(dict(chunks=len(d), changed=(d.n_inserted > 0).mean(),
                                                                ins_per_100=100 * d.n_inserted.sum() / d.n_ref.sum())), include_groups=False)
    print(by.round(3).to_string())
    print(f'all: {len(T):,} chunks, {(T.n_inserted > 0).mean():.1%} changed, {100 * T.n_inserted.sum() / T.n_ref.sum():.1f} words inserted per 100 reference words')
    c = collections.Counter()
    for r in T[T.n_inserted > 0].itertuples():
        a, b = collections.Counter(normalize_he(r.text).split()), collections.Counter(normalize_he(r.text_verbatim).split())
        c.update(b - a)
    print('top inserted words:', ', '.join(f'{w} ({n})' for w, n in c.most_common(15)))
    return by


# ---- self-checks ----------------------------------------------------------
if __name__ == '__main__':
    # a filler both models heard is put back where B put it; one only B heard is not
    ref, hb, ha = 'אני חושב שזה נכון.', 'אני, אה, חושב שזה כאילו נכון', 'אני אה חושב שזה נכון'
    t, n = verbatim_target(ref, hb, ha)
    assert t == 'אני אה, חושב שזה נכון.' and n == 1, (t, n)   # B's raw token, comma and all; the ref's own tokens untouched
    # nothing shared -> unchanged; identical -> unchanged
    assert verbatim_target(ref, hb, 'משהו אחר לגמרי') == (ref, 0)
    assert verbatim_target(ref, ref, ref) == (ref, 0)
    # a repetition at the start, and an insertion at the end
    t, n = verbatim_target('כן זה נכון', 'כן כן זה נכון בסדר', 'כן כן זה נכון בסדר')
    assert t == 'כן כן זה נכון בסדר' and n == 2, (t, n)
    # a plain insertion in the middle; and a substitution plus extra words, where only the
    # word A also heard goes back (which alignment the aligner picks is its business)
    t, n = verbatim_target('א ב ג', 'א ב y ג', 'y'); assert t == 'א ב y ג' and n == 1, (t, n)
    t, n = verbatim_target('א ב ג', 'א x y z ג', 'z'); assert n == 1 and t.replace(' z', '') == 'א ב ג' and t.index('z') > t.index('ב'), (t, n)
    # the multiset rule: A heard the word once, B inserted it twice -> one goes back
    t, n = verbatim_target('א ב', 'א נו נו ב', 'נו'); assert n == 1, (t, n)
    # deletions never touch the reference
    t, n = verbatim_target('א ב ג ד', 'א ד', 'א ד'); assert t == 'א ב ג ד' and n == 0
    print('targets.py: OK')
