"""The per-speaker error map, on the committees corpus.

This is Stage 1's analysis (notebooks/speaker_error_map.ipynb, run on VoxKnesset)
re-done on the inference the project actually stands on: the 1 h/speaker subset
of Hadasy/knesset-committees-chunks transcribed by both arms
(Dolevabudi/knesset-committees-inference).  Same questions, same scoring:

  1. how well does each model do per speaker (WER, CER; counts, never rates)
  2. each speaker's adaptation gain, WER_A - WER_B, and who gains least
  3. does the amount of speech predict the level (no) or the spread (yes)
  4. which subgroup rules are viable and which actually separate speakers

What changed from VoxKnesset, and why (docs/error_map.md has the numbers):

  * The QC filter.  Stage 1 dropped segments whose reference could not account
    for their audio (a flat 300 s allowance).  That rule cannot fire on <=30 s
    chunks.  Here the same failure -- reference text that does not match the
    audio -- is what the corpus's per-chunk alignment `quality` measures: chunks
    under 0.7 score a WER near 1.0 under BOTH arms.  So the one filter is
    quality >= 0.7, the consumer threshold docs/committees_handoff.md recommends.
  * Viability pools hours from the whole corpus (what a subgroup arm could train
    on), while WER and gain come from the subset (what was transcribed).
  * Per-speaker WER and gain carry a bootstrap CI (chunks resampled), because
    "who gains least" is a ranking and the ranking's noise must be visible.
  * Arm A here is Whisper told the language is Hebrew; its auto-detect run is
    kept as hypothesis_A_auto and reported once, as the detection-failure rate.

Plain functions, driven from notebooks/committees_error_map.ipynb; `run()` writes
every table under src/evaluation/outputs/.  Self-checks at the bottom.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from common import normalize_he
from evaluate import score

INFERENCE = os.path.join(ROOT, 'src', 'inference', 'outputs', 'inference.parquet')
INDEX     = os.path.join(ROOT, 'src', 'inference', 'cache', 'index.parquet')
SPEAKERS  = os.path.join(ROOT, 'src', 'preprocessing', 'speaker_index', 'outputs', 'segments.parquet')
OUT       = os.path.join(HERE, 'outputs')
INFERENCE_REPO, SPEAKERS_REPO = 'Dolevabudi/knesset-committees-inference', 'Dolevabudi/knesset-committees-speakers'

MIN_QUALITY  = 0.7      # the one QC filter; see the module docstring and docs/error_map.md
MIN_SEG      = 20       # below this a speaker's WER is mostly sampling noise (Stage 1 §9; here too)
MIN_SPEAKERS = 3        # a subgroup needs at least this many members
MIN_HOURS    = 5.0      # ... and this much corpus audio after holding out its largest member

# ---- load -------------------------------------------------------------------
def _local_or_hub(path, repo, filename):
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type='dataset')

def load_chunks():
    """One row per subset chunk with both hypotheses and the reference."""
    w = pd.read_parquet(_local_or_hub(INFERENCE, INFERENCE_REPO, 'inference.parquet'))
    w = w[w.in_stage1_subset & w.has_A & w.has_B].copy()
    w = w.rename(columns={'reference': 'ref', 'hypothesis_A': 'hyp_A', 'hypothesis_B': 'hyp_B',
                          'hypothesis_A_auto': 'hyp_A_auto'})
    return w[['chunk_id', 'speaker_id', 'session', 'session_date', 'knesset', 'duration_s', 'quality',
              'ref', 'hyp_A', 'hyp_B', 'hyp_A_auto']].reset_index(drop=True)

# ---- errors -----------------------------------------------------------------
def count_errors(seg):
    """Per-chunk error counts for both arms, from evaluate.score (S/D/I, runaway).
    Adds n_words, n_chars, wpm.  Counts, so any later grouping is Σerr/Σwords."""
    seg = seg.copy()
    seg['ref_n'] = seg.ref.map(normalize_he)
    for tag in ('A', 'B'):
        s = score(seg.ref, seg[f'hyp_{tag}'])
        for c in ('werr', 'cerr', 'S', 'D', 'I', 'runaway'):
            seg[f'{c}_{tag}'] = s[c].values
        if tag == 'A':
            seg['n_words'], seg['n_chars'] = s.n_words.values, s.n_chars.values
    seg['wpm'] = seg.n_words / (seg.duration_s / 60)
    return seg

def corpus_scores(df):
    return pd.Series({'chunks': len(df), 'hours': df.duration_s.sum() / 3600, 'ref_words': int(df.n_words.sum()),
                      'WER_A': df.werr_A.sum() / df.n_words.sum(), 'WER_B': df.werr_B.sum() / df.n_words.sum(),
                      'CER_A': df.cerr_A.sum() / df.n_chars.sum(), 'CER_B': df.cerr_B.sum() / df.n_chars.sum()})

def qc(seg, min_quality=MIN_QUALITY):
    """The filter and what it does: (kept, dropped, effect table)."""
    keep = seg.quality >= min_quality
    effect = pd.DataFrame({'before filter': corpus_scores(seg), 'after filter': corpus_scores(seg[keep]),
                           'the dropped chunks': corpus_scores(seg[~keep])}).T
    effect[['chunks', 'ref_words']] = effect[['chunks', 'ref_words']].astype(int)
    return seg[keep].reset_index(drop=True), seg[~keep].reset_index(drop=True), effect

def by_bucket(seg, col, edges, labels=None):
    """Corpus WER per bucket of `col` -- the diagnostic behind the filter."""
    b = pd.cut(seg[col], edges, labels=labels, include_lowest=True)
    return seg.groupby(b, observed=True).apply(
        lambda d: pd.Series({'chunks': len(d), 'hours': d.duration_s.sum() / 3600,
                             'WER_A': d.werr_A.sum() / d.n_words.sum(), 'WER_B': d.werr_B.sum() / d.n_words.sum()}),
        include_groups=False)

# ---- per speaker --------------------------------------------------------------
def per_speaker(seg, n_boot=1000, seed=0):
    """The error map: one row per speaker, Stage 1's columns plus S/D/I shares and
    95% bootstrap CIs (chunks resampled within the speaker) on WER_A, WER_B, gain."""
    g = seg.groupby('speaker_id')
    spk = g.agg(n_seg=('chunk_id', 'size'), hours=('duration_s', lambda s: s.sum() / 3600),
                n_words=('n_words', 'sum'), n_chars=('n_chars', 'sum'),
                werr_A=('werr_A', 'sum'), werr_B=('werr_B', 'sum'), cerr_A=('cerr_A', 'sum'), cerr_B=('cerr_B', 'sum'),
                S_A=('S_A', 'sum'), D_A=('D_A', 'sum'), I_A=('I_A', 'sum'),
                S_B=('S_B', 'sum'), D_B=('D_B', 'sum'), I_B=('I_B', 'sum'),
                runaway_A=('runaway_A', 'sum'), runaway_B=('runaway_B', 'sum'),
                sessions=('session', 'nunique'), first_date=('session_date', 'min'), last_date=('session_date', 'max'))
    for tag in ('A', 'B'):
        spk[f'wer_{tag}'] = spk[f'werr_{tag}'] / spk.n_words
        spk[f'cer_{tag}'] = spk[f'cerr_{tag}'] / spk.n_chars
        for k in ('S', 'D', 'I'):                    # share of the arm's word errors
            spk[f'{k}_share_{tag}'] = spk[f'{k}_{tag}'] / spk[f'werr_{tag}'].replace(0, np.nan)
    spk['gain_abs'] = spk.wer_A - spk.wer_B
    spk['gain_rel'] = spk.gain_abs / spk.wer_A
    # bootstrap: resample chunks within each speaker; Σerr/Σwords per resample
    rng = np.random.default_rng(seed); ci = {}
    for sid, d in g:
        eA, eB, w = d.werr_A.values, d.werr_B.values, d.n_words.values
        idx = rng.integers(0, len(w), size=(n_boot, len(w)))
        W = w[idx].sum(1); a = eA[idx].sum(1) / W; b = eB[idx].sum(1) / W
        ci[sid] = dict(wer_A_lo=np.percentile(a, 2.5), wer_A_hi=np.percentile(a, 97.5),
                       wer_B_lo=np.percentile(b, 2.5), wer_B_hi=np.percentile(b, 97.5),
                       gain_abs_lo=np.percentile(a - b, 2.5), gain_abs_hi=np.percentile(a - b, 97.5))
    spk = spk.join(pd.DataFrame(ci).T)
    spk['reliable'] = spk.n_seg >= MIN_SEG
    cols = ['n_seg', 'sessions', 'hours', 'n_words', 'n_chars', 'wer_A', 'cer_A', 'wer_B', 'cer_B', 'gain_abs', 'gain_rel',
            'wer_A_lo', 'wer_A_hi', 'wer_B_lo', 'wer_B_hi', 'gain_abs_lo', 'gain_abs_hi', 'reliable',
            'S_share_A', 'D_share_A', 'I_share_A', 'S_share_B', 'D_share_B', 'I_share_B',
            'runaway_A', 'runaway_B', 'first_date', 'last_date']
    return spk[cols]

def volume_table(spk):
    """Stage 1 §9: level flat, spread shrinks with more chunks (a measurement effect)."""
    bucket = pd.cut(spk.n_seg, [0, 5, 20, 50, 150, 400, 20000], labels=['1-5', '6-20', '21-50', '51-150', '151-400', '400+'])
    return spk.groupby(bucket, observed=True).agg(speakers=('wer_A', 'size'),
                                                  median_wer_A=('wer_A', 'median'), sd_wer_A=('wer_A', 'std'),
                                                  median_wer_B=('wer_B', 'median'), sd_wer_B=('wer_B', 'std'))

# ---- subgroups ----------------------------------------------------------------
ORIGIN = {'ישראל': 'Israel', 'ארץ ישראל': 'Israel',
          'ברית המועצות': 'FSU', 'האימפריה הרוסית': 'FSU',
          'אנגליה': 'Europe', 'בלגיה': 'Europe', 'גרמניה': 'Europe', 'פולין': 'Europe', 'צרפת': 'Europe', 'רומניה': 'Europe',
          'אוסטריה': 'Europe', 'דנמרק': 'Europe', 'האימפריה האוסטרו-הונגרית': 'Europe',   # absent from Stage 1's dict
          'איראן': 'MENA', "אלג'יריה": 'MENA', 'מצרים': 'MENA', 'מרוקו': 'MENA', 'סוריה': 'MENA', 'עיראק': 'MENA',
          'תוניסיה': 'MENA', 'טורקיה': 'MENA',
          'ארגנטינה': 'Americas', 'ארצות הברית': 'Americas', 'קנדה': 'Americas', 'מקסיקו': 'Americas',
          'אתיופיה': 'Ethiopia'}
NATION   = {'יהודי': 'Jewish', 'ערבי': 'Arab', 'דרוזי': 'Druze', 'בדואי': 'Bedouin'}
RELIGION = {'יהודי': 'Jewish', 'מוסלמי': 'Muslim', 'דרוזי': 'Druze', 'נוצרי': 'Christian'}
ORIENT   = {'חילוני': 'Secular', 'דתי': 'Religious', 'חרדי': 'Haredi', 'סוני': 'Sunni', 'יווני-קתולי': 'Greek-Catholic'}
RULES = {'gender': 'gender', 'nationality': 'nationality', 'religion': 'religion', 'religious orientation': 'orientation',
         'country of origin': 'origin', 'age quartile': 'age_band', 'speaking rate tertile': 'rate_band'}
NOT_A_GROUP = {'Unknown', 'Unrecorded'}

def _clean(s):
    return s.astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan, '<NA>': np.nan})

def speaker_attrs(seg, min_quality=MIN_QUALITY):
    """Per-speaker attributes: demographics from the speaker index, speaking rate
    and age from the subset, and corpus hours (quality >= min_quality) from the
    inference index -- the audio a subgroup arm could actually train on."""
    import pyarrow.parquet as pq
    ids = sorted(seg.speaker_id.unique())
    cols = ['speaker_id', 'speaker_name', 'place_of_birth', 'religion', 'nationality', 'religious_orientation', 'year_of_aliya', 'label']
    dem = pq.read_table(_local_or_hub(SPEAKERS, SPEAKERS_REPO, 'segments.parquet'), columns=cols).to_pandas()
    dem = dem[dem.speaker_id.isin(ids)].drop_duplicates('speaker_id').set_index('speaker_id')
    idx = pd.read_parquet(_local_or_hub(INDEX, INFERENCE_REPO, 'coverage.parquet') if not os.path.exists(INDEX) else INDEX,
                          columns=['speaker_id', 'duration_s', 'quality', 'gender', 'age'] if os.path.exists(INDEX) else ['speaker_id', 'duration_s'])
    idx = idx[idx.speaker_id.isin(ids)]
    hours = idx[idx.quality >= min_quality].groupby('speaker_id').duration_s.sum().div(3600).rename('hours_corpus') if 'quality' in idx else \
            idx.groupby('speaker_id').duration_s.sum().div(3600).rename('hours_corpus')
    a = seg.groupby('speaker_id').agg(wpm_median=('wpm', 'median'))
    if 'gender' in idx:
        a = a.join(idx.groupby('speaker_id').agg(gender_code=('gender', 'first'), age_mean=('age', 'mean')))
    a = a.join(dem).join(hours)
    a['speaker_name'] = a.speaker_name.fillna('')
    a['gender']      = np.where(a.gender_code == 1, 'Male', np.where(a.gender_code == 0, 'Female', 'Unknown'))
    a['origin']      = _clean(a.place_of_birth).map(ORIGIN).fillna('Unknown')
    a['nationality'] = _clean(a.nationality).map(NATION).fillna('Unknown')
    a['religion']    = _clean(a.religion).map(RELIGION).fillna('Unknown')
    a['orientation'] = _clean(a.religious_orientation).map(ORIENT).fillna('Unrecorded')
    a['age_band']    = pd.qcut(a.age_mean, 4, labels=['age Q1 (youngest)', 'age Q2', 'age Q3', 'age Q4 (oldest)'])
    a['rate_band']   = pd.qcut(a.wpm_median, 3, labels=['slow speech', 'medium speech', 'fast speech'])
    unmapped = sorted(set(_clean(a.place_of_birth).dropna()) - set(ORIGIN))
    assert not unmapped, f'birthplaces missing from ORIGIN: {unmapped}'
    return a

def subgroup_table(G, col, min_speakers=MIN_SPEAKERS, min_hours=MIN_HOURS):
    """Stage 1 §10.1.  WER pools subset error counts; hours are corpus hours."""
    t = G.groupby(col, observed=True).agg(
        speakers=('n_seg', 'size'), chunks=('n_seg', 'sum'), hours_subset=('hours', 'sum'),
        hours_corpus=('hours_corpus', 'sum'), largest=('hours_corpus', 'max'), words=('n_words', 'sum'),
        eA=('werr_A_total', 'sum'), eB=('werr_B_total', 'sum'))
    t['wer_A'] = t.eA / t.words; t['wer_B'] = t.eB / t.words
    t['gain_abs'] = t.wer_A - t.wer_B; t['gain_rel'] = t.gain_abs / t.wer_A
    t['hours_minus_largest'] = t.hours_corpus - t.largest
    t['viable'] = (t.speakers >= min_speakers) & (t.hours_minus_largest >= min_hours) & ~t.index.astype(str).isin(NOT_A_GROUP)
    return t[['speakers', 'chunks', 'hours_subset', 'hours_corpus', 'hours_minus_largest',
              'wer_A', 'wer_B', 'gain_abs', 'gain_rel', 'viable']].sort_values('wer_B', ascending=False)

def separation_set(G, col, min_seg=MIN_SEG):
    d = G[(G.n_seg >= min_seg) & (~G[col].astype(str).isin(NOT_A_GROUP))]
    counts = d[col].value_counts()
    return d[d[col].isin(counts[counts >= MIN_SPEAKERS].index)]

def eta_squared(labels, y):
    """Share of variance in y between groups.  numpy, because the permutation
    null calls this thousands of times."""
    y = np.asarray(y, float); codes = pd.factorize(pd.Series(labels).astype(str))[0]
    ok = ~np.isnan(y); y, codes = y[ok], codes[ok]
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot == 0 or len(y) == 0:
        return np.nan
    n = np.bincount(codes); m = np.bincount(codes, weights=y) / n
    return (n * (m - y.mean()) ** 2).sum() / ss_tot

def separation_stats(G, col, ycol, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = separation_set(G, col)[[col, ycol]].dropna()
    labels, y = d[col].astype(str).values, d[ycol].values
    obs = eta_squared(labels, y)
    null = np.array([eta_squared(rng.permutation(labels), y) for _ in range(n_perm)])
    return obs, float(np.percentile(null, 95)), float((null >= obs).mean()), len(d)

def rank_rules(G, rules=RULES, n_perm=2000):
    rows = []
    for name, col in rules.items():
        row = {'rule': name, 'groups': separation_set(G, col)[col].nunique()}
        for ycol in ('gain_rel', 'wer_B'):
            obs, null95, p, n = separation_stats(G, col, ycol, n_perm)
            row['speakers'] = n; row[f'eta2_{ycol}'] = obs; row[f'null95_{ycol}'] = null95; row[f'p_{ycol}'] = p
        rows.append(row)
    r = pd.DataFrame(rows).set_index('rule').sort_values('eta2_gain_rel', ascending=False)
    r['separates_gain'] = r.p_gain_rel < 0.05; r['separates_wer'] = r.p_wer_B < 0.05
    return r

# ---- who to adapt, and what to distrust ------------------------------------------
def candidates(spk):
    """Design step 5: lowest gain first, reliable speakers only, with the CI so the
    ranking's noise is visible.  gain_abs_hi < corpus median gain = clearly below."""
    d = spk[spk.reliable].sort_values('gain_rel')
    return d[['n_seg', 'hours', 'wer_A', 'wer_B', 'gain_abs', 'gain_abs_lo', 'gain_abs_hi', 'gain_rel']]

def flagged_sessions(seg, min_chunks=5, wer=0.9):
    """docs/committees_handoff.md: a time-origin offset between audio and alignment
    would show as a whole session near WER 1.0 under BOTH arms.  Listed, not removed."""
    s = seg.groupby('session').agg(speaker_id=('speaker_id', 'first'), session_date=('session_date', 'first'),
                                   chunks=('chunk_id', 'size'), words=('n_words', 'sum'),
                                   eA=('werr_A', 'sum'), eB=('werr_B', 'sum'), quality=('quality', 'median'))
    s['wer_A'] = s.eA / s.words; s['wer_B'] = s.eB / s.words
    return s[(s.chunks >= min_chunks) & (s.wer_A > wer) & (s.wer_B > wer)].drop(columns=['eA', 'eB']).sort_values('wer_B', ascending=False)

def language_detection(seg):
    """What Arm A's first run got wrong: share of chunks whose auto-detect
    hypothesis is not Hebrew, by duration -- the reason A is now told the language."""
    heb = seg.hyp_A_auto.fillna('').str.contains(r'[א-ת]')
    b = pd.cut(seg.duration_s, [0, 3, 6, 10, 20, 31])
    t = pd.DataFrame({'chunks': seg.groupby(b, observed=True).size(), 'not_hebrew': (~heb).groupby(b, observed=True).mean()})
    return t, float((~heb).mean())

# ---- run --------------------------------------------------------------------------
def run(out=OUT, n_boot=1000, n_perm=2000, verbose=True):
    os.makedirs(out, exist_ok=True); log = print if verbose else (lambda *a, **k: None)
    seg = count_errors(load_chunks())
    log(f'{len(seg):,} chunks, {seg.speaker_id.nunique()} speakers, {seg.duration_s.sum()/3600:.1f} h; scored both arms')
    kept, dropped, effect = qc(seg)
    log(effect.round(4).to_string())
    spk = per_speaker(kept, n_boot=n_boot)
    attrs = speaker_attrs(kept)
    totals = kept.groupby('speaker_id').agg(werr_A_total=('werr_A', 'sum'), werr_B_total=('werr_B', 'sum'))
    G = spk.join(totals).join(attrs)
    subgroups = pd.concat({name: subgroup_table(G, col) for name, col in RULES.items()}, names=['rule', 'group'])
    rank = rank_rules(G, n_perm=n_perm)
    cand = candidates(spk); flagged = flagged_sessions(kept); lang, lang_share = language_detection(seg)
    # write
    spk.join(attrs[['speaker_name', 'gender', 'origin', 'nationality', 'religion', 'orientation', 'age_mean', 'wpm_median', 'hours_corpus', 'label']]) \
       .to_csv(os.path.join(out, 'committees_speaker_performance.csv'), encoding='utf-8-sig', float_format='%.6g')
    effect.to_csv(os.path.join(out, 'committees_qc_effect.csv'), encoding='utf-8-sig', float_format='%.6g')
    subgroups.to_csv(os.path.join(out, 'committees_subgroups.csv'), encoding='utf-8-sig', float_format='%.6g')
    rank.to_csv(os.path.join(out, 'committees_subgroup_rank.csv'), encoding='utf-8-sig', float_format='%.6g')
    cand.to_csv(os.path.join(out, 'committees_adaptation_candidates.csv'), encoding='utf-8-sig', float_format='%.6g')
    flagged.to_csv(os.path.join(out, 'committees_flagged_sessions.csv'), encoding='utf-8-sig', float_format='%.6g')
    summary = dict(chunks_scored=int(len(seg)), chunks_kept=int(len(kept)), speakers=int(spk.shape[0]),
                   reliable_speakers=int(spk.reliable.sum()), min_quality=MIN_QUALITY,
                   corpus=effect.loc['after filter'].to_dict(), corpus_unfiltered=effect.loc['before filter'].to_dict(),
                   speakers_helped=int((spk.gain_abs > 0).sum()), speakers_hurt=int((spk.gain_abs < 0).sum()),
                   median_gain_rel=float(spk.gain_rel.median()), median_gain_rel_reliable=float(spk[spk.reliable].gain_rel.median()),
                   flagged_sessions=int(len(flagged)), arm_A_autodetect_not_hebrew=lang_share)
    json.dump(summary, open(os.path.join(out, 'committees_summary.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
    log(f'wrote 6 tables + summary to {out}')
    return dict(seg=seg, kept=kept, dropped=dropped, effect=effect, spk=spk, attrs=attrs, G=G,
                subgroups=subgroups, rank=rank, candidates=cand, flagged=flagged, language=lang, summary=summary)

# ---- self-checks -----------------------------------------------------------------------
if __name__ == '__main__' and '--run' not in sys.argv:
    # pooling: a group's WER is Σerr/Σwords, not the mean of member WERs
    G = pd.DataFrame(dict(n_seg=[30, 30, 30], hours=[1, 1, 1], hours_corpus=[10, 2, 4], n_words=[100, 400, 100],
                          werr_A_total=[50, 40, 20], werr_B_total=[10, 40, 10], g=['x', 'x', 'y']))
    t = subgroup_table(G, 'g', min_speakers=1, min_hours=1)
    assert abs(t.loc['x', 'wer_A'] - 90 / 500) < 1e-12 and abs(t.loc['x', 'wer_B'] - 50 / 500) < 1e-12, t
    assert t.loc['x', 'hours_minus_largest'] == 2 and t.loc['y', 'hours_minus_largest'] == 0
    print('subgroup_table: pooled, corpus-hours viability OK')
    # eta squared: perfect separation -> 1, identical groups -> 0, matches the pandas formula
    assert abs(eta_squared(['a', 'a', 'b', 'b'], [1, 1, 5, 5]) - 1) < 1e-12
    assert abs(eta_squared(['a', 'b', 'a', 'b'], [1, 1, 5, 5])) < 1e-12
    rng = np.random.default_rng(1); lab = rng.choice(list('abc'), 60); y = rng.normal(size=60)
    d = pd.DataFrame({'g': lab, 'y': y}); grand = d.y.mean()
    ref = d.groupby('g').y.apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum() / ((d.y - grand) ** 2).sum()
    assert abs(eta_squared(lab, y) - ref) < 1e-12
    print('eta_squared: OK')
    # per-speaker bootstrap contains the point estimate; gain = wer_A - wer_B
    seg = pd.DataFrame(dict(chunk_id=range(60), speaker_id=[1] * 30 + [2] * 30, session=[1] * 60, session_date=['2020-01-01'] * 60,
                            duration_s=10.0, n_words=20, n_chars=100, werr_A=rng.integers(0, 8, 60), werr_B=rng.integers(0, 4, 60),
                            cerr_A=5, cerr_B=2, S_A=1, D_A=1, I_A=1, S_B=1, D_B=0, I_B=0, runaway_A=False, runaway_B=False))
    s = per_speaker(seg, n_boot=300)
    assert ((s.wer_A_lo <= s.wer_A) & (s.wer_A <= s.wer_A_hi)).all() and ((s.gain_abs_lo <= s.gain_abs) & (s.gain_abs <= s.gain_abs_hi)).all()
    assert np.allclose(s.gain_abs, s.wer_A - s.wer_B) and s.reliable.all()
    print('per_speaker: counts, gain, bootstrap CI OK')
    # the scoring is evaluate.score, i.e. Stage 1's normalisation and word Levenshtein
    e = count_errors(pd.DataFrame(dict(chunk_id=['c'], speaker_id=[1], session=[1], session_date=['2020-01-01'], knesset=[25],
                                       duration_s=[6.0], quality=[0.9], ref=['צה"ל שלום עולם'], hyp_A=['צה ל שלום עולם.'], hyp_B=['שלום'], hyp_A_auto=[''])))
    assert e.werr_A[0] == 0 and e.werr_B[0] == 3 and e.n_words[0] == 4 and abs(e.wpm[0] - 40) < 1e-9, e[['werr_A', 'werr_B', 'n_words', 'wpm']]
    print('count_errors: normalisation + Levenshtein OK')
    print('error_map.py self-checks OK')
elif __name__ == '__main__':
    run()
