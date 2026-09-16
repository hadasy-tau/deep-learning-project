"""What the inference also tells, beyond Stage 1's error map.

error_map.py answers Stage 1's questions.  These functions answer the ones the
committees corpus raised on top of them, each a plain function over the scored
chunk table `kept` that error_map.run() returns (or the per-speaker table
`spk`).  Driven from notebooks/committees_error_map.ipynb § 11; results in
docs/error_map.md § Beyond Stage 1.

  by_condition        corpus WER by year, committee, chunk length, speaking rate
  chunk_level         the distribution of per-chunk WER; where both arms fail
  error_content       what the errors ARE: top substitutions, insertions,
                      deletions; the share that are one-character (orthographic)
                      differences; the share involving digits; WER with numeric
                      tokens removed (the size of Stage 1's "digit problem")
  insertion_agreement how many of B's inserted words A also heard -- real speech
                      the protocol left out, versus hallucination
  hallucination       looping decodes (a 3-gram repeated 3+ times), runaway, empty
  language_forcing    Arm A with the language forced vs auto-detected, by length
  split_half          reliability of the per-speaker ranking (odd/even sessions,
                      Spearman-Brown corrected)
  cross_corpus        the same speakers on the plenums (Stage 1's table)
"""
import collections, os, re, sys
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from common import normalize_he

PLENUM = os.path.join(HERE, 'outputs', 'speaker_performance.csv')     # Stage 1 on VoxKnesset

def _wer(d, tag):
    return d[f'werr_{tag}'].sum() / d.n_words.sum()

def _hyps(kept):
    """Normalised hypotheses, computed once and cached on the frame."""
    if 'hA_n' not in kept:
        kept['hA_n'] = kept.hyp_A.map(normalize_he); kept['hB_n'] = kept.hyp_B.map(normalize_he)
    return kept

# ---- conditions -------------------------------------------------------------------
def by_condition(kept, col, bins=None, labels=None, top=None):
    """Corpus WER per level of `col` (a column, or bins over a numeric one)."""
    key = pd.cut(kept[col], bins, labels=labels, include_lowest=True) if bins is not None else kept[col]
    t = kept.groupby(key, observed=True).apply(lambda d: pd.Series({'chunks': len(d), 'speakers': d.speaker_id.nunique(),
                                                                    'WER_A': _wer(d, 'A'), 'WER_B': _wer(d, 'B')}), include_groups=False)
    if top: t = t.sort_values('chunks', ascending=False).head(top).sort_values('WER_B')
    return t

def chunk_level(kept):
    a, b = kept.werr_A / kept.n_words, kept.werr_B / kept.n_words
    return dict(quantiles_A=a.quantile([.25, .5, .75, .9]).round(3).to_dict(), quantiles_B=b.quantile([.25, .5, .75, .9]).round(3).to_dict(),
                perfect_A=float((kept.werr_A == 0).mean()), perfect_B=float((kept.werr_B == 0).mean()),
                fail_A=float((a >= 1).mean()), fail_B=float((b >= 1).mean()), fail_both=float(((a >= 1) & (b >= 1)).mean()),
                spearman_errors_A_B=float(stats.spearmanr(kept.werr_A, kept.werr_B).statistic))

# ---- what the errors are -------------------------------------------------------------
def error_content(kept, n_top=15):
    """Per arm: top substitution pairs, insertions and deletions; the share of
    substitutions one character apart; the share of errors involving a digit;
    WER with numeric tokens removed from both sides."""
    kept = _hyps(kept); out = {}
    for tag, col in (('A', 'hA_n'), ('B', 'hB_n')):
        subs, ins, dele = collections.Counter(), collections.Counter(), collections.Counter()
        near = nsub = digit = 0
        for r, h in zip(kept.ref_n, kept[col]):
            rw, hw = r.split(), h.split()
            for op, i1, i2, j1, j2 in Levenshtein.opcodes(rw, hw):
                if op == 'replace':
                    for a, b in zip(rw[i1:i2], hw[j1:j2]):
                        subs[(a, b)] += 1; nsub += 1
                        near += Levenshtein.distance(a, b) == 1
                        digit += bool(re.search(r'\d', a + b))
                    for a in rw[i1 + (j2 - j1):i2]: dele[a] += 1
                    for b in hw[j1 + (i2 - i1):j2]: ins[b] += 1
                elif op == 'delete':
                    for a in rw[i1:i2]: dele[a] += 1
                elif op == 'insert':
                    for b in hw[j1:j2]: ins[b] += 1
        tot = int(kept[f'werr_{tag}'].sum())
        strip = lambda s: [w for w in s.split() if not re.search(r'\d', w)]
        e = sum(Levenshtein.distance(strip(r), strip(h)) for r, h in zip(kept.ref_n, kept[col])); n = sum(len(strip(r)) for r in kept.ref_n)
        out[tag] = dict(top_substitutions=subs.most_common(n_top), top_insertions=ins.most_common(n_top), top_deletions=dele.most_common(n_top),
                        near_miss_share_of_substitutions=near / max(nsub, 1), near_miss_share_of_errors=near / max(tot, 1),
                        digit_share_of_errors=digit / max(tot, 1), wer=_wer(kept, tag), wer_without_numbers=e / n)
    return out

def content_table(content):
    """The counters as one long table (arm, kind, reference, hypothesis, count)."""
    rows = []
    for tag, c in content.items():
        rows += [dict(arm=tag, kind='substitution', reference=a, hypothesis=b, count=n) for (a, b), n in c['top_substitutions']]
        rows += [dict(arm=tag, kind='insertion', reference='', hypothesis=w, count=n) for w, n in c['top_insertions']]
        rows += [dict(arm=tag, kind='deletion', reference=w, hypothesis='', count=n) for w, n in c['top_deletions']]
    return pd.DataFrame(rows)

def insertion_agreement(kept):
    """Share of B's inserted words that also occur in A's hypothesis of the same
    chunk.  Two independent models hearing the same absent word is speech the
    protocol did not write down; a word only B produces may be hallucination."""
    kept = _hyps(kept); agree = total = 0
    for r, ha, hb in zip(kept.ref_n, kept.hA_n, kept.hB_n):
        rw, aw, bw = r.split(), ha.split(), hb.split()
        inserted = [bw[j] for op, i1, i2, j1, j2 in Levenshtein.opcodes(rw, bw) if op == 'insert' for j in range(j1, j2)]
        if not inserted: continue
        pool = collections.Counter(aw)
        for w in inserted:
            total += 1
            if pool[w] > 0: agree += 1; pool[w] -= 1
    return dict(share_of_B_insertions_also_in_A=agree / max(total, 1), inserted_words_B=total)

def _loops(h, n=3, k=3):
    w = h.split()
    if len(w) < n * k: return False
    return collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)).most_common(1)[0][1] >= k

def hallucination(kept):
    kept = _hyps(kept)
    return pd.DataFrame({tag: dict(looping=float(kept[col].map(_loops).mean()), runaway=float(kept[f'runaway_{tag}'].mean()), empty=float((kept[col] == '').mean()))
                         for tag, col in (('A', 'hA_n'), ('B', 'hB_n'))}).T

def language_forcing(kept, bins=(0, 3, 6, 10, 20, 31)):
    """Arm A auto-detect vs forced Hebrew on the same chunks, by chunk length."""
    k = kept.copy(); k['hAa'] = k.hyp_A_auto.fillna('').map(normalize_he)
    k['werr_Aa'] = [Levenshtein.distance(r.split(), h.split()) for r, h in zip(k.ref_n, k.hAa)]
    t = k.groupby(pd.cut(k.duration_s, list(bins)), observed=True).apply(
        lambda d: pd.Series({'chunks': len(d), 'WER_A_auto': d.werr_Aa.sum() / d.n_words.sum(), 'WER_A_forced': _wer(d, 'A')}), include_groups=False)
    return t, k.werr_Aa.sum() / k.n_words.sum(), _wer(k, 'A')

# ---- reliability, and the plenums -------------------------------------------------------
def split_half(kept, min_words=200):
    """Odd/even sessions per speaker; Spearman between halves, Spearman-Brown corrected
    to the reliability of the full-length measure."""
    rows = []
    for sid, d in kept.groupby('speaker_id'):
        sess = sorted(d.session.unique()); a, b = d[d.session.isin(sess[::2])], d[d.session.isin(sess[1::2])]
        if a.n_words.sum() < min_words or b.n_words.sum() < min_words: continue
        rows.append(dict(speaker_id=sid, wer_A_1=_wer(a, 'A'), wer_A_2=_wer(b, 'A'), wer_B_1=_wer(a, 'B'), wer_B_2=_wer(b, 'B')))
    H = pd.DataFrame(rows).set_index('speaker_id')
    H['gain_abs_1'], H['gain_abs_2'] = H.wer_A_1 - H.wer_B_1, H.wer_A_2 - H.wer_B_2
    H['gain_rel_1'], H['gain_rel_2'] = H.gain_abs_1 / H.wer_A_1, H.gain_abs_2 / H.wer_A_2
    out = {}
    for m in ('wer_A', 'wer_B', 'gain_abs', 'gain_rel'):
        r = stats.spearmanr(H[f'{m}_1'], H[f'{m}_2']).statistic; out[m] = dict(split_half_rho=float(r), reliability=float(2 * r / (1 + r)))
    return pd.DataFrame(out).T.assign(speakers=len(H)), H

def cross_corpus(spk, plenum_path=PLENUM, min_seg=20):
    """Join the committees error map to Stage 1's VoxKnesset table on speaker_id."""
    vox = pd.read_csv(plenum_path, index_col=0)[['n_seg', 'hours', 'wer_A', 'wer_B', 'gain_rel']].rename(columns=lambda c: 'plenum_' + c)
    J = spk[spk.n_seg >= min_seg].join(vox, how='inner'); J = J[J.plenum_n_seg >= min_seg]
    rows = {c: dict(spearman=float(stats.spearmanr(J[c], J['plenum_' + c]).statistic), p=float(stats.spearmanr(J[c], J['plenum_' + c]).pvalue),
                    median_committees=float(J[c].median()), median_plenum=float(J['plenum_' + c].median())) for c in ('wer_A', 'wer_B', 'gain_rel')}
    return pd.DataFrame(rows).T.assign(speakers=len(J)), J

# ---- run ----------------------------------------------------------------------------------
def with_committee(kept):
    """kept + committee_name and year, from the inference index."""
    import error_map as E
    idx = pd.read_parquet(E.INDEX, columns=['chunk_id', 'committee_name']) if os.path.exists(E.INDEX) else None
    k = kept.merge(idx, on='chunk_id', how='left') if idx is not None else kept.assign(committee_name='?')
    k['year'] = k.session_date.str[:4]
    return k

def run(out=None, verbose=True):
    """Every table above, written next to error_map's outputs."""
    import json, error_map as E
    out = out or E.OUT; log = print if verbose else (lambda *a, **k: None)
    R = E.run(verbose=False); kept, spk = R['kept'], R['spk']; k = with_committee(kept)
    cond = pd.concat({'year': by_condition(k, 'year'), 'committee (top 12)': by_condition(k, 'committee_name', top=12),
                      'chunk length (s)': by_condition(k, 'duration_s', [0, 3, 6, 10, 20, 31]),
                      'speaking rate (wpm)': by_condition(k, 'wpm', [0, 50, 100, 150, 200, 250, 300, 5000])}, names=['condition', 'level'])
    cond.to_csv(os.path.join(out, 'committees_conditions.csv'), encoding='utf-8-sig', float_format='%.4g')
    content = error_content(kept); content_table(content).to_csv(os.path.join(out, 'committees_error_content.csv'), index=False, encoding='utf-8-sig')
    rel, _ = split_half(kept); cc, _ = cross_corpus(spk); lang, wa_auto, wa = language_forcing(kept)
    checks = dict(chunk_level=chunk_level(kept), insertion_agreement=insertion_agreement(kept), hallucination=hallucination(kept).to_dict(),
                  error_content={t: {k2: v for k2, v in c.items() if not k2.startswith('top_')} for t, c in content.items()},
                  language_forcing=dict(wer_A_auto=wa_auto, wer_A_forced=wa, by_length=lang.round(4).rename(index=str).to_dict()),
                  split_half_reliability=rel.to_dict(), cross_corpus=cc.to_dict())
    json.dump(checks, open(os.path.join(out, 'committees_checks.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=str)
    log(f'wrote committees_conditions.csv, committees_error_content.csv, committees_checks.json to {out}')
    return dict(conditions=cond, content=content, checks=checks, reliability=rel, cross=cc)

# ---- self-checks ------------------------------------------------------------------------
if __name__ == '__main__' and '--run' in sys.argv:
    run()
elif __name__ == '__main__':
    k = pd.DataFrame(dict(chunk_id=list('abcd'), speaker_id=[1, 1, 2, 2], session=[1, 2, 3, 4], session_date=['2020'] * 4, duration_s=[5.0] * 4,
                          ref=['א ב ג', 'א ב ג', 'א ב ג', 'א ב ג'], hyp_A=['א ב ג', 'א ב', 'א ב ג ד', 'x ב ג'], hyp_B=['א ב ג ד', 'א ב ג ד', 'א ב ג', 'א ב ג'],
                          hyp_A_auto=['', '', '', ''], n_words=[3] * 4, werr_A=[0, 1, 1, 1], werr_B=[1, 1, 0, 0], runaway_A=[False] * 4, runaway_B=[False] * 4))
    k['ref_n'] = k.ref.map(normalize_he)
    c = error_content(k)
    assert c['A']['top_substitutions'] == [(('א', 'x'), 1)] and c['A']['top_insertions'] == [('ד', 1)] and c['A']['top_deletions'] == [('ג', 1)], c['A']
    assert c['B']['top_insertions'] == [('ד', 2)] and c['B']['digit_share_of_errors'] == 0
    print('error_content: opcodes -> counters OK')
    ia = insertion_agreement(k)                      # B inserts ד twice; A also has ד in chunk c only... chunk a: A has no ד -> 0/1; chunk b: no -> 0/1
    assert ia['inserted_words_B'] == 2 and ia['share_of_B_insertions_also_in_A'] == 0.0, ia
    k2 = k.copy(); k2.loc[0, 'hyp_A'] = 'א ב ג ד'; k2.pop('hA_n'); k2.pop('hB_n')
    assert insertion_agreement(k2)['share_of_B_insertions_also_in_A'] == 0.5
    print('insertion_agreement: OK')
    assert _loops('א ב ג א ב ג א ב ג') and not _loops('א ב ג ד ה ו ז ח ט')
    print('hallucination: looping detector OK')
    assert abs(chunk_level(k)['fail_both'] - 0.0) < 1e-12 and chunk_level(k)['perfect_A'] == 0.25
    print('chunk_level: OK')
    print('error_analysis.py self-checks OK')
