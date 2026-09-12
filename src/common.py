"""Shared by every stage: Hebrew normalisation, WAV reading, session-disjoint splits.

Carved verbatim out of stage2/pipeline.py when the repo moved from stage-shaped
folders to src/ organised by function.  The rest of that file loaded and chunked
VoxKnesset and went with it; these four functions did not, because five modules
across four folders import them:

    normalize_he   evaluation/evaluate.py, preprocessing/speaker_index/{names,roster}.py,
                   inference/{verify,validate_final}.py
    read_wav, SR   training/train.py, evaluation/evaluate.py
    make_splits    the committees corpus ships unsplit; this is what splits it
    budget_order   training/train.py's nested budget rungs

Reach it from any leaf folder with the repo's src/ on the path:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import normalize_he
"""
import re, unicodedata, wave
import numpy as np, pandas as pd

SR = 16000

def read_wav(path, start=0.0, end=None):
    """The audio is 16 kHz 16-bit mono WAV, so stdlib `wave` is enough --
    no soundfile/librosa/torchaudio dependency. Returns float32 in [-1, 1]."""
    with wave.open(path, 'rb') as w:
        assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (SR, 2, 1), \
            f'unexpected wav format in {path}'
        w.setpos(min(int(start * SR), w.getnframes()))
        n = w.getnframes() - w.tell() if end is None else int((end - start) * SR)
        raw = w.readframes(max(n, 0))
    return np.frombuffer(raw, '<i2').astype(np.float32) / 32768.0

# ---- from Stage 1, verbatim.  Do not re-derive. ---------------------------
# The rule that matters: geresh and gershayim map to a SPACE, not to nothing.
# Deleting them instead costs about 5.7% relative WER on arm B, because it welds
# the two halves of an abbreviation into one token the model never emits.
_niqqud = re.compile(r'[֑-ׇ]')
_quotes = re.compile(r'[׳״"\'‘’“”]')
_dashes = re.compile(r'[־\-‐-―_/\\]')
_punct  = re.compile(r'[^\w\s֐-׿]', flags=re.UNICODE)

def normalize_he(s):
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = _niqqud.sub('', s)
    s = _quotes.sub(' ', s)      # -> space, not nothing; see the note above
    s = _dashes.sub(' ', s)
    s = _punct.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# ---- split ----------------------------------------------------------------
def make_splits(chunks, test_min, dev_min=15):
    """Session-disjoint per speaker: latest sessions -> test, then dev, rest train.

    test_min: minutes of personal-test, scalar or {speaker_id: minutes}.
    Size it per speaker: a flat 45 min leaves low-WER speakers unable to
    resolve anything short of a halving of WER, because resolution tracks WER
    -- a speaker with few errors left has few to remove.
    """
    def one(spk, g):
        want = test_min[spk] if isinstance(test_min, dict) else test_min
        dur  = g.groupby('session').duration_s.sum().sort_index(ascending=False)
        cum  = dur.cumsum() / 60
        test = set(dur.index[cum <= want]) or {dur.index[0]}
        rest = dur.drop(index=list(test))
        dev  = set(rest.index[rest.cumsum() / 60 <= dev_min]) or set(rest.index[:1])
        return g.session.map(lambda s: 'test' if s in test else 'dev' if s in dev else 'train')

    out = chunks.copy()
    if out.empty:
        out['part'] = pd.Series(dtype=object)
        return out
    # Concatenated explicitly rather than through groupby.apply: apply infers
    # its own return shape, and on pandas 3 a SINGLE group comes back as a
    # 1xN DataFrame rather than a Series -- so a one-speaker call (which is
    # exactly what the smoke-test notebook does) failed where the four-speaker
    # panel worked. Nothing here should depend on how many speakers there are.
    parts = [one(spk, g) for spk, g in out.groupby('speaker_id')]
    out['part'] = pd.concat(parts).reindex(out.index)
    return out


def budget_order(train_chunks):
    """Train chunks ordered latest session first, so D4's nested budgets
    (1,2,5,10,20,40,80 min) are prefixes of one ordering and nested by
    construction.  Latest-first keeps train temporally closest to dev/test."""
    return train_chunks.sort_values(['session', 'start'], ascending=[False, True])

# ---- self-checks ----------------------------------------------------------
if __name__ == '__main__':
    # normalize_he, case by case.  These values were captured from the function
    # as it stood in stage2/pipeline.py before the carve-out, so this block is
    # also the regression test for the move itself.
    assert normalize_he('שָׁלוֹם עֲלֵיכֶם') == 'שלום עליכם'          # niqqud stripped
    assert normalize_he('צה"ל')             == 'צה ל'                # gershayim -> SPACE
    assert normalize_he('אונ׳')             == 'אונ'                 # geresh -> space, then trimmed
    assert normalize_he('בן-גוריון')        == 'בן גוריון'           # maqaf and hyphen -> space
    assert normalize_he('ד"ר יעקב, ז׳בוטינסקי — "כן"') == 'ד ר יעקב ז בוטינסקי כן'
    assert normalize_he('  א   ב  ')        == 'א ב'                 # runs collapsed, ends trimmed
    assert normalize_he('Hello, world! 123') == 'Hello world 123'    # latin survives
    assert normalize_he(None) == '' and normalize_he(3) == ''        # non-str -> ''
    print('normalize_he: OK')

    # make_splits: session-disjoint per speaker, latest sessions first.
    ch = pd.DataFrame(dict(speaker_id=[1] * 6 + [2] * 4,
                           session=[10, 10, 9, 8, 7, 6, 20, 19, 18, 17],
                           start=[0] * 10,
                           duration_s=[600] * 6 + [900] * 4))
    sp = make_splits(ch, test_min=20, dev_min=10)
    for spk, g in sp.groupby('speaker_id'):
        by = {p: set(x.session) for p, x in g.groupby('part')}
        for a, b in (('train', 'test'), ('train', 'dev'), ('dev', 'test')):
            assert not (by.get(a, set()) & by.get(b, set())), f'{a}/{b} overlap, speaker {spk}'
        assert by.get('test'), f'speaker {spk} got no test'
    # The newest sessions go to test: speaker 1's 10 is 20 min on its own.
    assert set(sp[(sp.speaker_id == 1) & (sp.part == 'test')].session) == {10}

    # One speaker must behave exactly as four.  On pandas 3 a single group comes
    # back from groupby.apply as a 1xN frame rather than a Series, which is why
    # make_splits concatenates explicitly -- this is that regression.
    one = make_splits(ch[ch.speaker_id == 1], test_min=20, dev_min=10)
    assert one.part.tolist() == sp[sp.speaker_id == 1].part.tolist()

    assert make_splits(ch.iloc[:0], test_min=20).empty                # empty in, empty out
    print('make_splits: session-disjoint, one-speaker case OK')

    # budget_order: latest session first, ascending within a session, so the
    # nested budget rungs are prefixes of one ordering.
    assert budget_order(ch).session.tolist() == [20, 19, 18, 17, 10, 10, 9, 8, 7, 6]
    print('budget_order: OK')
