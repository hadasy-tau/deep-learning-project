"""Stage 3 segments: cut labelled word runs into audio pieces, attach the
person's demographics, and write the index.  No audio is touched.

Conventions (stage3/plan.md, VoxKnesset README):
  filename   {speaker_id}_{session_id}_{start_ms}_{end_ms}.wav  (0 = no id), so
             stage2/pipeline.py's regex parses speaker and session unchanged
  gender     0 = female, 1 = male
  age        (session_date - date_of_birth) / 365.25
  quality    median word probability of the piece -- ivrit.ai's own definition,
             recomputed because the shipped scores are keyed to aligned segments
Pieces are <= MAX_S and >= MIN_S, never cross a speaker or label change, and
stay inside their refined segment.
"""
import datetime as dt, os, re, sys, time
import pandas as pd

from roster import OUT, CACHE
from names import build_alias
from protocol import load_session, load_shard, place_gold, word_table, quality
from label import Roster, label_words, runs

MAX_S, MIN_S = 30.0, 1.0
MIN_VERIFIED = 0.90        # skip a session whose map places fewer segments than this

# Explicit output schema.  Without it pyarrow infers per batch and the parts
# disagree: year_of_aliya is '1950' for some MKs and empty for others, so a batch
# of only-empty values infers double and fails to concatenate with a batch that
# has strings.  Declaring the types also keeps every part file identical.
SCHEMA = {'filename': 'string', 'speaker_id': 'int64', 'session': 'int64', 'start': 'float64',
          'end': 'float64', 'duration_s': 'float64', 'reference_text': 'string', 'quality': 'float64',
          'label': 'string', 'label_path': 'string', 'match_method': 'string', 'reason': 'string',
          'raw_name': 'string', 'local_speaker_id': 'Int64', 'et_id': 'Int64', 'gold_speaker_id': 'string',
          'seg': 'int64', 'knesset': 'int64', 'committee_name': 'string', 'session_date': 'string',
          'speaker_name': 'string', 'gender': 'Int64', 'age': 'float64', 'date_of_birth': 'string',
          'place_of_birth': 'string', 'year_of_aliya': 'string', 'religion': 'string',
          'nationality': 'string', 'religious_orientation': 'string'}

CHAIR_RE = re.compile(r'^\s*(?:היו"ר|היו״ר|יו"ר|יו״ר|מ"מ היו"ר|מ״מ היו״ר|ממלאת? מקום היו"ר|היושבת?[- ]ראש)\s')

def add_is_chairman(df):
    """The plan lists is_chairman; it is a pure function of raw_name, so it is
    derived here rather than carried through the row builder.  Taking it from the
    protocol's own title prefix works on both paths -- the gold shard's
    is_chairman field exists only for Path A sessions."""
    df['is_chairman'] = df.raw_name.fillna('').str.match(CHAIR_RE).astype('boolean')
    return df

def coerce(df):
    """One dtype per column, so every part file shares a schema."""
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})
    for c, t in SCHEMA.items():
        if c not in df:
            df[c] = None
        df[c] = df[c].astype('string').astype(t) if t == 'string' else pd.to_numeric(df[c], errors='coerce').astype(t)
    return df[list(SCHEMA)]
DEMO = ['gender', 'date_of_birth', 'place_of_birth', 'year_of_aliya', 'religion', 'nationality', 'religious_orientation']

def cut(run):
    """Split one run into pieces of at most MAX_S seconds at word boundaries."""
    pieces, cur = [], []
    for w in run['words']:
        if cur and w['t1'] - cur[0]['t0'] > MAX_S:
            pieces.append(cur); cur = []
        cur.append(w)
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p[-1]['t1'] - p[0]['t0'] >= MIN_S]

def demographics(pid, roster, date):
    """From the K20-25 roster, else the corpus roster (former MKs)."""
    if pid in roster.mk.index:
        r = roster.mk.loc[pid]; full = r.full_name
    elif pid in roster.corpus.index:
        r = roster.corpus.loc[pid].rename({'corpus_gender': 'gender'}); full = r.corpus_full_name
    else:
        return dict(speaker_name=None, age=None, **{c: None for c in DEMO})
    d = {c: (None if pd.isna(r.get(c)) else r.get(c)) for c in DEMO}
    d['gender'] = None if d['gender'] is None else int(d['gender'])
    dob = d['date_of_birth']
    d['age'] = round((dt.date.fromisoformat(date) - dt.date.fromisoformat(dob)).days / 365.25, 2) if dob else None
    d['speaker_name'] = full
    return d

def rows_for(session, rs, roster, label_path):
    m = session['meta']; sid = session['session_id']; date = m['session_date'][:10]
    out = []
    for r in rs:
        for p in cut(r):
            t0, t1 = p[0]['t0'], p[-1]['t1']; pid = r['person_id']
            out.append(dict(filename=f'{pid}_{sid}_{int(round(t0*1000))}_{int(round(t1*1000))}.wav',
                            speaker_id=pid, session=sid, start=t0, end=t1, duration_s=round(t1 - t0, 3),
                            reference_text=''.join(w['word'] for w in p).strip(), quality=round(quality(p), 4),
                            label=r['label'], label_path=label_path, match_method=r['method'], reason=r['reason'],
                            raw_name=r['raw_name'], local_speaker_id=r['local_speaker_id'], et_id=r['et_id'],
                            gold_speaker_id=r['gold_id'], seg=r['seg'],
                            knesset=int(m['knesset_num']), committee_name=m['committee_name'], session_date=date,
                            **demographics(pid, roster, date) if pid else dict(speaker_name=None, age=None, **{c: None for c in DEMO})))
    return out

def process(session_id, manifest_row, roster, aliases=None, keep=False):
    """Label one session end to end.  Returns (rows, alias_pairs, stats)."""
    s = load_session(session_id, keep=keep)
    words, verified = word_table(s)
    if verified < MIN_VERIFIED:
        raise ValueError(f'map places only {verified:.0%} of segments at their own text')
    if manifest_row.label_path == 'A':
        placed, frac = place_gold(load_shard(int(manifest_row.knesset), manifest_row.protocol_name), s['text'])
        label_words(words, s, roster, placed=placed)
    else:
        frac = None
        label_words(words, s, roster, placed=None, aliases=aliases)
    rs = runs(words)
    rows = rows_for(s, rs, roster, manifest_row.label_path)
    pairs = dict(names=[], et=[])
    if manifest_row.label_path == 'A':
        from names import clean
        for r in rs:
            if r['label'] in ('mk', 'former_mk'):
                rc = clean(r['raw_name'])
                if rc: pairs['names'].append((rc, r['person_id']))
                if r['et_id'] is not None: pairs['et'].append((r['et_id'], r['person_id']))
    stats = dict(session_id=session_id, path=manifest_row.label_path, placed=frac, verified=verified,
                 words=len(words), rows=len(rows),
                 mk_s=sum(x['duration_s'] for x in rows if x['label'] == 'mk'),
                 total_s=sum(x['duration_s'] for x in rows))
    return rows, pairs, stats

def build(session_ids, roster, aliases=None, out_name='segments.parquet', keep=False, log_every=20):
    man = pd.read_csv(os.path.join(OUT, 'manifest.csv')).set_index('session_id')
    rows, pairs, stats, t = [], dict(names=[], et=[]), [], time.time()
    for i, sid in enumerate(session_ids, 1):
        try:
            r, p, st = process(sid, man.loc[sid], roster, aliases, keep)
        except Exception as e:                       # one bad session must not sink the run
            stats.append(dict(session_id=sid, path=man.loc[sid].label_path, error=repr(e)[:200])); continue
        rows += r; pairs['names'] += p['names']; pairs['et'] += p['et']; stats.append(st)
        if i % log_every == 0:
            print(f'  {i}/{len(session_ids)} sessions, {len(rows):,} rows, {time.time()-t:.0f}s', flush=True)
    df = coerce(pd.DataFrame(rows))
    df.to_parquet(os.path.join(OUT, out_name), index=False)
    return df, pairs, pd.DataFrame(stats)

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    R = Roster()
    man = pd.read_csv(os.path.join(OUT, 'manifest.csv')).set_index('session_id')
    rows, pairs, st = process(2073683, man.loc[2073683], R, keep=True)
    df = pd.DataFrame(rows)
    assert (df.duration_s >= MIN_S).all() and (df.duration_s <= MAX_S + 1e-6).all()
    assert df.filename.str.match(r'^\d+_\d+_\d+_\d+\.wav$').all()
    seg = df.groupby('seg')
    assert (seg.start.min() >= 0).all() and (df.sort_values(['seg', 'start']).groupby('seg').start.diff().dropna() > 0).all()
    y = df[df.speaker_id == 4405].iloc[0]
    assert y.gender == 0 and abs(y.age - 58.61) < 0.02 and y.place_of_birth == 'ישראל', y.to_dict()
    assert df[df.label == 'mk'].speaker_id.nunique() == 6
    # the schema must survive an all-null demographic batch (the year_of_aliya trap)
    guests = coerce(pd.DataFrame([r for r in rows if r['speaker_id'] == 0]))
    assert list(guests.columns) == list(coerce(df.copy()).columns) and str(guests.year_of_aliya.dtype) == 'string'
    c = add_is_chairman(df.copy())
    assert c[c.raw_name == 'היו"ר שלי יחימוביץ'].is_chairman.all()
    assert not c[c.raw_name == 'מאיר בן שבת'].is_chairman.any()
    al = build_alias(pairs['names'])
    assert al.set_index('key').at['שלי יחימוביץ', 'person_id'] == 4405 and al.set_index('key').at['שלי יחימוביץ', 'usable']
    print(f'segments.py self-checks OK: {len(df)} rows; '
          + ', '.join(f'{k} {v/60:.1f} min' for k, v in df.groupby('label').duration_s.sum().items()))
