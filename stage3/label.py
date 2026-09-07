"""Stage 3 label: decide, word by word, who is speaking -- and record why.

Two paths (stage3/plan.md):
  Path A  the session has a corpus shard.  A word gets a person_id only if the
          gold sentence covering it names that numeric id, the ivrit.ai raw name
          agrees with an accepted spelling of that person, and the person held a
          seat on the session date.  Not seated but same person -> former_mk.
  Path B  no shard (K25 after 2024-03-26).  The cleaned raw name must equal an
          accepted spelling of exactly one MK seated that day (B1), or an alias
          learned under Path A with purity 1.0 (B2 by name, B3 by editor id).

Every other word is non_mk (gold says guest) or unresolved (any disagreement),
kept with its reason.  There is no similarity score anywhere in this file.
"""
import json, os
import pandas as pd

from roster import OUT, active_on, load_corpus_roster, name_variants
from names import clean, agree, match_exact
from protocol import gold_at

SYNTH_ID = 10_000_000                     # ivrit.ai ids >= this are per-session; below: editor ids

class Roster:
    """K20-K25 MKs with stints and spellings, plus the full corpus roster for
    former MKs who appear as guests."""
    def __init__(self):
        self.mk = pd.read_csv(os.path.join(OUT, 'mk_metadata.csv')).set_index('person_id')
        self.stints = pd.read_csv(os.path.join(OUT, 'mk_stints.csv'))
        v = pd.read_csv(os.path.join(OUT, 'mk_name_variants.csv'))
        self.variants = v.groupby('person_id').variant.apply(set).to_dict()
        self.corpus = load_corpus_roster().set_index('person_id')
        self._active = {}

    def active(self, date):
        if date not in self._active:
            self._active[date] = active_on(date, self.stints)
        return self._active[date]

    def candidates(self, date):
        return {pid: self.variants[pid] for pid in self.active(date) if pid in self.variants}

    def spellings(self, pid):
        """Accepted spellings for any numeric id we know: K20-25 roster first,
        else the corpus roster (former MKs)."""
        if pid in self.variants:
            return self.variants[pid]
        if pid in self.corpus.index:
            full = self.corpus.at[pid, 'corpus_full_name']
            self.variants[pid] = name_variants(full.split()[0], ' '.join(full.split()[1:]), full)
            return self.variants[pid]
        return None

def label_words(words, session, roster, placed=None, aliases=None):
    """Attach label / person_id / reason to every word record, in place.
    placed:  output of protocol.place_gold  (Path A) or None (Path B)
    aliases: dict(names=DataFrame, et=DataFrame) from names.build_alias, Path B only"""
    date = session['meta']['session_date'][:10]
    active = roster.active(date)
    cands = roster.candidates(date)
    cache = {}
    an = aliases['names'].set_index('key') if aliases is not None and len(aliases['names']) else None
    ae = aliases['et'].set_index('key') if aliases is not None and len(aliases['et']) else None

    for w in words:
        lid = w['local_speaker_id']
        raw = session['speakers'].get(lid, '')
        et  = lid if (lid is not None and lid < SYNTH_ID) else None
        w.update(raw_name=raw, et_id=et, gold_id=None, person_id=0, label='unresolved', reason='', method='')
        if not w['offset_ok']:              # segment sits at an unverified position
            w['reason'] = 'bad_offset'; continue
        if raw not in cache:
            cache[raw] = clean(raw)
        rc = cache[raw]

        if placed is not None:                                     # ---- Path A
            g = gold_at(placed, w['a'], w['b'])
            if g is None:
                w['reason'] = 'no_gold'; continue
            gid, gname, valid = g[2], g[3], g[4]
            w['gold_id'] = gid
            if not str(gid).isdigit() or not valid:
                w['label'], w['reason'] = 'non_mk', 'gold_guest'; continue
            pid = int(gid)
            if rc is None:
                w['reason'] = 'raw_garbage'; continue
            sp = roster.spellings(pid)
            if sp is None:
                w['reason'] = 'unknown_id'; continue
            if not agree(rc, sp):
                w['reason'] = 'name_disagree'; continue
            w['person_id'], w['method'] = pid, 'A'
            w['label'] = 'mk' if pid in active else 'former_mk'
        else:                                                      # ---- Path B
            if rc is None:
                w['reason'] = 'raw_garbage'; continue
            pid, how = match_exact(rc, cands)
            method = None
            if pid is not None:
                method = 'B1'
            elif how == 'ambiguous':
                w['reason'] = 'ambiguous'; continue
            elif an is not None and rc in an.index and an.at[rc, 'usable'] and int(an.at[rc, 'person_id']) in active:
                pid, method = int(an.at[rc, 'person_id']), 'B2'
            # The editor id is NOT an assignment source.  A holdout over 200
            # sessions scored it at 0.47 precision on 26 seconds of speech -- it
            # produced every genuine cross-person error in the run and nothing
            # else ('יורם בן דוד' -> יואב בן צור, 'דובר' -> טלי פלוסקוב) because
            # stenographers reuse picker entries.  It is kept below as a veto,
            # where being occasionally wrong only costs recall.
            if pid is None:
                w['reason'] = 'no_match'; continue
            # editor-id consistency: a known, pure editor id that names someone else vetoes
            if ae is not None and et in ae.index and ae.at[et, 'usable'] and int(ae.at[et, 'person_id']) != pid:
                w['reason'] = 'et_conflict'; continue
            w['person_id'], w['method'], w['label'] = pid, method, 'mk'
    return words

def runs(words):
    """Maximal runs of consecutive words, within one refined segment, sharing
    (label, person_id, raw speaker).  Each run is a single-speaker span."""
    out = []
    key = lambda w: (w['seg'], w['label'], w['person_id'], w['local_speaker_id'], w['reason'])
    for w in words:
        if out and key(out[-1]['words'][0]) == key(w):
            out[-1]['words'].append(w)
        else:
            out.append(dict(words=[w]))
    for r in out:
        f = r['words'][0]
        r.update(seg=f['seg'], label=f['label'], person_id=f['person_id'], method=f['method'], reason=f['reason'],
                 raw_name=f['raw_name'], local_speaker_id=f['local_speaker_id'], et_id=f['et_id'], gold_id=f['gold_id'],
                 t0=f['t0'], t1=r['words'][-1]['t1'])
    return out

# ---- self-checks -------------------------------------------------------------
if __name__ == '__main__':
    from protocol import load_session, load_shard, place_gold, word_table
    R = Roster()
    s = load_session(2073683, keep=True)
    words, _ = word_table(s)
    placed, _ = place_gold(load_shard(20, '20_ptv_519810.doc'), s['text'])
    label_words(words, s, R, placed=placed)
    rs = runs(words)
    by = pd.Series([r['label'] for r in rs]).value_counts()
    why = pd.Series([r['reason'] for r in rs if r['label'] == 'unresolved']).value_counts()
    assert by.get('mk', 0) > 300 and why.get('name_disagree', 0) == 0, (by, why)
    assert {r['person_id'] for r in rs if r['label'] == 'mk'} == {4405, 30065, 30548, 30058, 30095, 2178}
    # Path B on the same session, no aliases: the six MKs by exact name, nobody else
    w2, _ = word_table(s); label_words(w2, s, R, placed=None)
    got = {r['person_id'] for r in runs(w2) if r['label'] == 'mk'}
    assert got == {4405, 30065, 30548, 30058, 30095, 2178}, got
    assert all(r['label'] != 'mk' for r in runs(w2) if r['raw_name'] == 'מאיר בן שבת')
    print('label.py self-checks OK:', by.to_dict(), '| unresolved reasons:', why.to_dict())
