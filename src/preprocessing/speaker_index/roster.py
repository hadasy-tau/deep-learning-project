"""Stage 3 roster: who was a Knesset member, when, and what we know about them.

Facts that shape this file (measured 6 Sep 2026, see docs/speaker_index_plan.md):
  * KnessetCorpus person_id == the Knesset's official PersonID (526 גפני, 528 ליצמן
    in both).  VoxKnesset's speaker_id is the same number.  So the id space is the
    Knesset's, and the Knesset ODATA API is the authority on *who was an MK when*.
  * The corpus roster (all_knesset_members_jsons.jsonl, 1,137 people) was last
    updated Nov 2022: 7 MKs who entered K25 in 2025-26 are missing from it, and 43
    stints started after Mar 2023.  It is used for demographics only.
  * ODATA pages at 100 rows; every response is cached under cache/odata so
    re-runs are free and offline.

Outputs (outputs/):
  mk_metadata.csv       one row per MK of K20-K25: names, gender (0=f, 1=m as in
                        VoxKnesset), demographics from the corpus where present
  mk_stints.csv         one row per (person, knesset, faction) stint with dates
  mk_name_variants.csv  every normalised spelling we will accept for each person

No fuzzy matching lives here: variants are generated from the roster, never guessed.
"""
import datetime as dt, hashlib, json, os, re, sys
import pandas as pd, requests
from huggingface_hub import hf_hub_download, get_token

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common import normalize_he                        # Stage 1 normalisation, verbatim

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, 'outputs')
CACHE  = os.path.join(HERE, 'cache')
ODATA  = 'https://knesset.gov.il/Odata/ParliamentInfo.svc/'
CORPUS = 'HaifaCLGroup/KnessetCorpus'
KNESSETS    = list(range(20, 26))
POSITION_MK = 54                                       # KNS_Position: חבר כנסת

# ---- ODATA ------------------------------------------------------------------
def odata(entity, filt, key_field, page=100):
    """All rows of `entity` matching `filt`, paged and cached.  Pages are ordered
    by `key_field`: without $orderby the server's $skip paging overlaps, and the
    first run returned 12 duplicate sessions.  Rows are deduplicated on the key."""
    key = hashlib.md5(f'{entity}|{filt}|{key_field}'.encode()).hexdigest()[:12]
    cp  = os.path.join(CACHE, 'odata', f'{entity}_{key}.json')
    if os.path.exists(cp):
        return json.load(open(cp, encoding='utf-8'))
    rows, skip = {}, 0
    while True:
        r = requests.get(ODATA + entity, timeout=120,
                         params={'$filter': filt, '$orderby': key_field, '$top': page, '$skip': skip, '$format': 'json'})
        r.raise_for_status()
        v = r.json()['value']
        for x in v:
            rows[x[key_field]] = x
        if len(v) < page:
            break
        skip += page
    rows = list(rows.values())
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    json.dump(rows, open(cp, 'w', encoding='utf-8'), ensure_ascii=False)
    return rows

def _date(s):
    return s[:10] if s else None

def load_stints():
    """MK stints for K20-K25 from KNS_PersonToPosition (1,102 rows, 12 requests)."""
    rows = odata('KNS_PersonToPosition', f'KnessetNum ge {KNESSETS[0]} and PositionID eq {POSITION_MK}', 'PersonToPositionID')
    st = pd.DataFrame([dict(person_id=int(r['PersonID']), knesset=int(r['KnessetNum']),
                            faction=r.get('FactionName'), start=_date(r.get('StartDate')),
                            end=_date(r.get('FinishDate'))) for r in rows])
    return st.sort_values(['person_id', 'start']).reset_index(drop=True)

def load_persons(ids):
    """KNS_Person for the given ids, 25 per request."""
    ids = sorted(set(int(i) for i in ids)); out = []
    for i in range(0, len(ids), 25):
        filt = ' or '.join(f'PersonID eq {p}' for p in ids[i:i + 25])
        out += odata('KNS_Person', filt, 'PersonID')
    return pd.DataFrame([dict(person_id=int(p['PersonID']), first_name=(p['FirstName'] or '').strip(),
                              last_name=(p['LastName'] or '').strip(),
                              gender={'זכר': 1, 'נקבה': 0}.get(p.get('GenderDesc')),
                              is_current=bool(p.get('IsCurrent'))) for p in out])

# ---- corpus demographics ---------------------------------------------------
def load_corpus_roster():
    p = hf_hub_download(CORPUS, 'all_knesset_members_jsons.jsonl', repo_type='dataset',
                        token=get_token(), cache_dir=os.path.join(CACHE, 'hf'))
    rows = [json.loads(l) for l in open(p, encoding='utf-8')]
    return pd.DataFrame([dict(person_id=int(r['person_id']), corpus_full_name=r['full_name'],
                              corpus_gender={'male': 1, 'female': 0}.get(r.get('gender')),
                              date_of_birth=_date(r.get('date_of_birth')),
                              place_of_birth=r.get('place_of_birth') or None,
                              year_of_aliya=r.get('year_of_aliya') or None,
                              religion=r.get('religion') or None,
                              nationality=r.get('nationality') or None,
                              religious_orientation=r.get('religious_orientation') or None,
                              residence=r.get('residence') or None,
                              date_of_death=_date(r.get('date_of_death'))) for r in rows])

# ---- names -------------------------------------------------------------------
_paren = re.compile(r'^(.*?)\s*\((.+?)\)\s*(.*)$')

def name_variants(first, last, corpus_full=None):
    """Every spelling we accept for one person, normalised.  Deterministic:
    first/last, last/first, the nickname form when a name carries '(nick)',
    with dashes, quotes and niqqud unified by normalize_he."""
    raw = {f'{first} {last}', f'{last} {first}'}
    if not isinstance(corpus_full, str):            # NaN for MKs the corpus lacks
        corpus_full = None
    for full in filter(None, [corpus_full, f'{first} {last}']):
        m = _paren.match(full)
        if m:
            before, nick, after = (x.strip() for x in m.groups())
            base = f'{before} {after}'.strip()
            raw |= {base, ' '.join(reversed(base.split()))}
            toks = base.split()
            if toks:                       # nick replaces the token right before the parens
                i = len(before.split()) - 1 if before else 0
                sub = toks[:]; sub[i] = nick
                raw |= {' '.join(sub), ' '.join(reversed(sub))}
        else:
            raw |= {full, ' '.join(reversed(full.split()))}
    return {normalize_he(v) for v in raw if normalize_he(v)}

# ---- assembly ---------------------------------------------------------------
def build():
    stints  = load_stints()
    persons = load_persons(stints.person_id.unique())
    corpus  = load_corpus_roster()
    mk = persons.merge(corpus, on='person_id', how='left')
    mk['in_corpus'] = mk.corpus_full_name.notna()
    mk['full_name'] = mk.corpus_full_name.where(mk.in_corpus, mk.first_name + ' ' + mk.last_name)
    # gender: ODATA is authoritative; report disagreement rather than silently pick
    dis = mk[mk.corpus_gender.notna() & (mk.corpus_gender != mk.gender)]
    assert dis.empty, f'gender disagrees ODATA vs corpus:\n{dis[["person_id","full_name"]]}'
    mk = mk.drop(columns=['corpus_gender'])
    variants = pd.DataFrame([dict(person_id=r.person_id, variant=v)
                             for r in mk.itertuples()
                             for v in name_variants(r.first_name, r.last_name, r.corpus_full_name)])
    return mk, stints, variants

def active_on(date, stints):
    """PersonIDs holding an MK seat on `date` ('YYYY-MM-DD')."""
    s = stints[(stints.start <= date) & (stints.end.isna() | (stints.end >= date))]
    return set(s.person_id)

def homonyms(stints, variants):
    """Pairs of different people who share a spelling AND overlap in office."""
    by = variants.groupby('variant').person_id.apply(set)
    bad = []
    for v, ids in by[by.map(len) > 1].items():
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sa, sb = stints[stints.person_id == a], stints[stints.person_id == b]
                for x in sa.itertuples():
                    for y in sb.itertuples():
                        if x.start <= (y.end or '9999') and y.start <= (x.end or '9999'):
                            bad.append((v, a, b, max(x.start, y.start)))
    return bad

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    mk, stints, variants = build()
    os.makedirs(OUT, exist_ok=True)
    assert mk.person_id.is_unique
    print(f'roster: {len(mk)} MKs in K{KNESSETS[0]}-K{KNESSETS[-1]}, {len(stints)} stints, '
          f'{len(variants)} name variants; {(~mk.in_corpus).sum()} without corpus demographics')
    print('  missing from corpus:', ', '.join(mk[~mk.in_corpus].full_name))

    probes = {20: '2018-11-06', 21: '2019-06-01', 22: '2019-12-01',
              23: '2020-11-01', 24: '2022-03-01', 25: '2024-01-01'}
    for k, d in probes.items():
        n = len(active_on(d, stints))
        assert 115 <= n <= 150, (k, d, n)
        print(f'  active_on({d}) K{k}: {n}')
    assert 526   in active_on('2018-11-06', stints), 'גפני should be an MK on 2018-11-06'
    assert 30894 in active_on('2025-07-01', stints) and 30894 not in active_on('2025-06-01', stints), \
        'סמיר בן סעיד entered 2025-06-20'
    assert 4405  in active_on('2018-11-06', stints) and 4405 not in active_on('2020-11-01', stints), \
        "שלי יחימוביץ' left in 2019"

    v = set(variants[variants.person_id == 30678].variant)
    assert normalize_he('אבי ניסנקורן') in v and normalize_he('אברהם ניסנקורן') in v, v
    # hyphen/space and order variants collapse onto one person, found by name not by id
    who = set(variants[variants.variant == normalize_he('אורית פרקש-הכהן')].person_id)
    assert len(who) == 1, who
    assert normalize_he('פרקש הכהן אורית') in set(variants[variants.person_id.isin(who)].variant)

    bad = homonyms(stints, variants)
    assert not bad, f'homonyms in office at the same time: {bad}'
    print('  homonyms in office simultaneously: none')

    cov = mk[mk.in_corpus][['date_of_birth', 'religion', 'place_of_birth', 'religious_orientation', 'year_of_aliya']].notna().mean()
    print('  corpus demographic coverage:', ', '.join(f'{c} {x:.0%}' for c, x in cov.items()))

    mk.to_csv(os.path.join(OUT, 'mk_metadata.csv'), index=False)
    stints.to_csv(os.path.join(OUT, 'mk_stints.csv'), index=False)
    variants.to_csv(os.path.join(OUT, 'mk_name_variants.csv'), index=False)
    print(f'wrote {OUT}/mk_metadata.csv, mk_stints.csv, mk_name_variants.csv')
