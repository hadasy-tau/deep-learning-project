"""Stage 3 names: turn a raw protocol speaker string into something comparable,
and compare it -- exactly -- with the roster.

Facts that shape this file (see docs/speaker_index_plan.md):
  * Raw names come from the Knesset editor's speaker picker (ET_speaker bookmarks),
    so MK names are consistently spelled; the noise is role prefixes (היו"ר),
    party suffixes ((הליכוד)), ministry titles (שר הרווחה ... יעקב מרגי) and
    upstream parse failures (dialogue glued after a colon, 42k-char blocks).
  * The corpus's own matcher used a similarity ratio and produced guest->retired-MK
    errors (מיכל חסון -> יואל חסון).  Nothing here computes a ratio.  A name either
    equals an accepted spelling of exactly one candidate, or it is unresolved.
"""
import os, re, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common import normalize_he

# Leading role words.  Applied once, after party-suffix removal, before normalize_he.
TITLE_RE = re.compile(
    r'^(?:היו"ר|היו״ר|היו\'ר|יו"ר|יו״ר|מ"מ היו"ר|מ״מ היו״ר|ממלאת? מקום היו"ר|היושבת?[- ]ראש|'
    r'ח"כ|חה"כ|ח״כ|חה״כ|חבר הכנסת|חברת הכנסת|ראש הממשלה|סגנית? השרה?|השרה?|שרת?|'
    r'מזכירת? הכנסת|מנהלת? הו?ועדה|היועצת? המשפטית?|יועצת? משפטית?)\s+')
PAREN_RE = re.compile(r'\s*\([^)]*\)')
MAX_CHARS, MAX_TOKENS = 60, 8

def is_garbage(raw):
    """Upstream parse failures, never names: dialogue after a colon, digits, or
    -- once the parenthetical is gone -- too long or too many tokens.

    Length is measured AFTER the parenthetical is removed.  Measuring it before
    rejected real MKs whose line carries a long role title, e.g.
    'מיכאל מרדכי ביטון (יו"ר הוועדה המיוחדת לפיקוח על תהליכי הסרת חסמים)' is 67
    chars but names a 3-token MK; that cost 20 minutes in a 50-session pilot."""
    if not isinstance(raw, str) or not raw.strip():
        return True
    if ':' in raw or re.search(r'\d', raw):
        return True
    bare = PAREN_RE.sub('', raw).strip()
    return not bare or len(bare) > MAX_CHARS or len(bare.split()) > MAX_TOKENS

def clean(raw):
    """Normalised comparable form, or None for garbage.  Party suffix and one
    leading title are removed; everything else is left for the matcher."""
    if is_garbage(raw):
        return None
    s = PAREN_RE.sub(' ', raw).strip()
    s = TITLE_RE.sub('', s)
    s = normalize_he(s)
    return s or None

def agree(raw_clean, variants):
    """Strict: the raw tokens equal one accepted spelling, or one side is a
    >=2-token subset of the other (extra middle name, missing middle name).
    A single shared token (a surname alone) never agrees."""
    r = set(raw_clean.split()) if raw_clean else set()
    for v in variants:
        g = set(v.split())
        if r and g and (r == g or (len(g) >= 2 and g <= r) or (len(r) >= 2 and r <= g)):
            return True
    return False

def match_exact(raw_clean, candidates):
    """raw_clean against {person_id: set(variants)} of the MKs active that day.
    Whole string first, then the trailing 2-4 tokens (ministry titles precede the
    name).  Exactly one person or nothing."""
    if not raw_clean:
        return None, None
    toks = raw_clean.split()
    forms = [(' '.join(toks), 'exact')] + [(' '.join(toks[-n:]), 'suffix') for n in (2, 3, 4) if len(toks) > n]
    for form, how in forms:
        hits = {pid for pid, vs in candidates.items() if form in vs}
        if len(hits) == 1:
            return hits.pop(), how
        if len(hits) > 1:
            return None, 'ambiguous'
    return None, None

def build_alias(pairs, min_count=3):
    """pairs: iterable of (key, person_id) observed under Path A with all three
    checks passed.  Returns one row per key with its count and purity; Path B
    accepts only purity == 1.0 and count >= min_count."""
    df = pd.DataFrame(pairs, columns=['key', 'person_id'])
    if df.empty:
        return pd.DataFrame(columns=['key', 'person_id', 'count', 'purity', 'usable'])
    g = df.groupby(['key', 'person_id']).size().rename('count').reset_index()
    tot = g.groupby('key')['count'].transform('sum')
    g['purity'] = g['count'] / tot
    g = g.sort_values(['key', 'count'], ascending=[True, False]).drop_duplicates('key')
    g['usable'] = (g.purity == 1.0) & (g['count'] >= min_count)
    return g.reset_index(drop=True)

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    n = normalize_he
    assert clean('היו"ר שלי יחימוביץ') == n("שלי יחימוביץ'") == 'שלי יחימוביץ'
    assert clean('מכלוף מיקי זוהר (הליכוד)') == 'מכלוף מיקי זוהר'
    assert clean('אלעזר שטרן (יש עתיד-תל"ם)') == clean('אלעזר שטרן (כחול לבן)') == 'אלעזר שטרן'
    assert clean('אורית פרקש-הכהן') == clean('אורית פרקש הכהן')
    assert clean('שר הרווחה והביטחון החברתי יעקב מרגי') == 'הרווחה והביטחון החברתי יעקב מרגי'
    for g in ['דורון נגרין:    אני מהלשכה המשפטית', 'ז מיקי לוי 12', 'א' * 61,
              'כלומר נתח השוק של אחת מהחברות הגדולות עמד על היו"ר משה גפני']:
        assert is_garbage(g), g
    # a real MK behind a long role title survives (length is measured after the parens)
    long_role = 'מיכאל מרדכי ביטון (יו"ר הוועדה המיוחדת לפיקוח על תהליכי הסרת חסמים)'
    assert not is_garbage(long_role) and clean(long_role) == 'מיכאל מרדכי ביטון'
    assert agree(clean(long_role), {n('מיכאל ביטון'), n('ביטון מיכאל')}), 'middle name must not block'
    assert clean('אבל יש להם בעיה שאומרים') is not None       # a fragment survives cleaning ...
    # ... but never matches: the roster has no such spelling.
    yh = {n("שלי יחימוביץ'"), n("יחימוביץ' שלי")}
    assert agree(clean('היו"ר שלי יחימוביץ'), yh)
    assert not agree(clean('מיכל חסון'), {n('יואל חסון'), n('חסון יואל')}), 'guest must not agree with MK'
    assert not agree('חסון', {n('יואל חסון')}), 'a surname alone never agrees'
    assert agree(clean('אבי ניסנקורן (כחול לבן)'), {n('אבי ניסנקורן'), n('אברהם ניסנקורן')})
    cands = {4405: yh, 526: {n('משה גפני'), n('גפני משה')}, 30758: {n('וליד טאהא'), n('טאהא וליד')}}
    assert match_exact(clean('היו"ר שלי יחימוביץ'), cands) == (4405, 'exact')
    assert match_exact(clean('שר האוצר משה גפני'), cands) == (526, 'suffix')
    assert match_exact(clean('מיכל חסון'), cands) == (None, None)
    two = {1: {'ישראל כץ'}, 2: {'ישראל כץ'}}
    assert match_exact('ישראל כץ', two) == (None, 'ambiguous')
    al = build_alias([('a', 1)] * 3 + [('b', 1), ('b', 2), ('c', 3)])
    assert al.set_index('key').usable.to_dict() == {'a': True, 'b': False, 'c': False}, al
    print('names.py self-checks OK')
