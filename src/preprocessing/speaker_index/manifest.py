"""Stage 3 manifest: which committee sessions exist, what each one has, and
which labelling path it takes.

Facts that shape this file (measured 6 Sep 2026, see docs/speaker_index_plan.md):
  * ivrit-ai/knesset-committees has 13,176 session directories named by the
    Knesset's own CommitteeSessionID.  17% hold only the protocol .doc (no
    audio, no transcripts) and are skipped.
  * The .doc/.docx in each directory is the KnessetCorpus protocol_name; the
    corpus shard is protocols_sentences/committee_protocols/data/<k>/<doc>.jsonl.bz2.
    Exact string match -- the join needs no CSV and no dates.
  * The corpus stops at 2024-03-26: every K25 session after that has no shard and
    goes to Path B.  Nothing before that date is missing.
  * ODATA KNS_CommitteeSession supplies knesset, date, number and committee for
    every session id; it is the same id, so the join is exact.

Output: outputs/manifest.csv, one row per session directory.
"""
import json, os, re
import pandas as pd
from huggingface_hub import HfApi, get_token

from roster import odata, CACHE, OUT                     # cached ODATA, same dirs

COMMITTEES = 'ivrit-ai/knesset-committees'
CORPUS     = 'HaifaCLGroup/KnessetCorpus'
SHARD_DIR  = 'protocols_sentences/committee_protocols/data'
KNESSETS   = list(range(20, 26))
REQUIRED   = {'audio.m4a': 'has_audio', 'transcript.refined.json': 'has_refined',
              'transcript.refined.map.json': 'has_map', 'speakers.txt': 'has_speakers',
              'speakers.segments.txt': 'has_segments', 'metadata.json': 'has_metadata'}

def _cached(name, fn):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    v = fn()
    os.makedirs(CACHE, exist_ok=True)
    json.dump(v, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return v

def session_files():
    """{session_id: [basenames]} for every directory, from one recursive listing."""
    def fetch():
        api = HfApi(token=get_token()); files = {}
        for e in api.list_repo_tree(COMMITTEES, repo_type='dataset', recursive=True):
            parts = e.path.split('/')
            if len(parts) == 2:
                files.setdefault(parts[0], []).append(parts[1])
            elif len(parts) == 1 and e.path.isdigit():
                files.setdefault(e.path, [])
        return files
    return _cached('committee_files.json', fetch)

def shard_names():
    """{protocol_name: knesset} for every corpus committee shard of K20-K25."""
    def fetch():
        api = HfApi(token=get_token()); out = {}
        for k in KNESSETS:
            for e in api.list_repo_tree(CORPUS, repo_type='dataset', path_in_repo=f'{SHARD_DIR}/{k}'):
                n = e.path.split('/')[-1]
                if n.endswith('.jsonl.bz2'):
                    out[n[:-len('.jsonl.bz2')]] = k
        return out
    return _cached('corpus_shards.json', fetch)

def odata_sessions():
    rows = odata('KNS_CommitteeSession', f'KnessetNum ge {KNESSETS[0]}', 'CommitteeSessionID')
    com  = {int(c['CommitteeID']): c['Name'] for c in odata('KNS_Committee', f'KnessetNum ge {KNESSETS[0]}', 'CommitteeID')}
    df = pd.DataFrame([dict(session_id=int(r['CommitteeSessionID']), knesset=int(r['KnessetNum']),
                              session_number=r.get('Number'), session_date=(r.get('StartDate') or '')[:10] or None,
                              committee_id=int(r['CommitteeID']) if r.get('CommitteeID') else None,
                              committee_name=com.get(int(r['CommitteeID'])) if r.get('CommitteeID') else None)
                         for r in rows])
    assert df.session_id.is_unique, 'ODATA returned a session twice'
    return df

_doc = re.compile(r'^(\d+)_ptv_\d+\.docx?$', re.I)

def build():
    files, shards = session_files(), shard_names()
    rows = []
    for sid, fs in files.items():
        docs = [f for f in fs if _doc.match(f)] or [f for f in fs if f.lower().endswith(('.doc', '.docx'))]
        doc  = docs[0] if docs else None
        have = {col: (f in fs) for f, col in REQUIRED.items()}
        rows.append(dict(session_id=int(sid), protocol_name=doc,
                         knesset_from_doc=int(_doc.match(doc).group(1)) if doc and _doc.match(doc) else None,
                         **have, complete=all(have.values()),
                         has_shard=bool(doc) and doc in shards))
    m = pd.DataFrame(rows).merge(odata_sessions(), on='session_id', how='left')
    m['in_odata']   = m.knesset.notna()
    m['knesset']    = m.knesset.fillna(m.knesset_from_doc).astype('Int64')
    m['label_path'] = pd.Series('', index=m.index).mask(m.complete & m.has_shard, 'A').mask(m.complete & ~m.has_shard, 'B')
    return m.sort_values('session_id').reset_index(drop=True)

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    m = build()
    os.makedirs(OUT, exist_ok=True)
    assert m.session_id.is_unique and len(m) == len(session_files()), 'one row per session directory'
    print(f'manifest: {len(m)} session directories, {m.complete.sum()} complete ({m.complete.mean():.0%}), '
          f'{m.has_shard.sum()} with a corpus shard, {m.in_odata.sum()} found in ODATA')
    r = m[m.session_id == 2073683].iloc[0]
    assert r.protocol_name == '20_ptv_519810.doc' and r.has_shard and r.session_date == '2018-11-06' \
        and r.label_path == 'A', r.to_dict()
    assert m.protocol_name.notna().all(), 'every directory should hold a .doc'
    both = m.knesset_from_doc.notna() & m.in_odata
    assert (m.loc[both, 'knesset_from_doc'] == m.loc[both, 'knesset']).all(), \
        'knesset in .doc name disagrees with ODATA'
    print('\n' + m.groupby('knesset').agg(sessions=('session_id', 'size'), complete=('complete', 'sum'),
                                        with_shard=('has_shard', 'sum'),
                                        path_A=('label_path', lambda s: (s == 'A').sum()),
                                        path_B=('label_path', lambda s: (s == 'B').sum())).to_string())
    k25 = m[(m.knesset == 25) & m.has_shard].session_date.max()
    k25_after = ((m.knesset == 25) & m.has_shard & (m.session_date > '2024-04-30')).sum()
    print(f'\nlast K25 session with a shard: {k25}; K25 sessions after Apr 2024 that still have one: {k25_after}')
    assert k25 is not None and k25 < '2024-05-01', k25
    m.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(f'wrote {OUT}/manifest.csv')
