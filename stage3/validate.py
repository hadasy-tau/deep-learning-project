"""Stage 3 validate: how often would Path B put the wrong person on a word?

Holdout simulation (stage3/plan.md, Step 4.1): on sessions that HAVE a corpus
shard, label every word twice -- Path A (gold + raw name + active_on) and Path B
(raw name only, with the alias tables) -- and score B against A by duration.
The number that matters is cross-person: B assigned an id and A says it is
someone else, or a guest.  Recall is reported but is not the gate.

Regression: the K25 guest->retired-MK errors found in the corpus must never
receive an id under Path A.  The audio-embedding check (Step 4.2) needs
materialised audio and is not in this file.
"""
import os
import pandas as pd

from roster import OUT
from names import clean, agree
from protocol import load_session, load_shard, place_gold, word_table
from label import Roster, label_words

# raw names the corpus mis-assigned to retired MKs (session -> raw names); none may be 'mk'
REGRESSION = {2208066: ['מיכל חסון'], 2206662: ['ראובן פרי', 'ראובן לורנצי'], 2202358: ['בנימין למקין'],
              2213229: ['ערן אוחנה'], 2210760: ['גדעון זעירא', 'אילן ברוש', 'אבי אורן', 'יונית אפרתי', 'מיכל רימון']}

def holdout(session_ids, roster, aliases, manifest):
    rows = []
    for sid in session_ids:
        mr = manifest.loc[sid]
        s = load_session(sid, keep=True)          # cached by the pilot; no re-download
        placed, _ = place_gold(load_shard(int(mr.knesset), mr.protocol_name), s['text'])
        wa = label_words(word_table(s)[0], s, roster, placed=placed)
        wb = label_words(word_table(s)[0], s, roster, placed=None, aliases=aliases)
        for a, b in zip(wa, wb):
            d = a['t1'] - a['t0']
            if b['label'] == 'mk':
                if a['label'] in ('mk', 'former_mk'):
                    kind = 'correct' if a['person_id'] == b['person_id'] else 'cross_person'
                elif a['label'] == 'non_mk':
                    # The corpus gave this word a UUID.  That is not automatically
                    # a Path B error: the corpus paper reports ~19.7k false
                    # negatives, and in a 200-session holdout 44 of 47 such cases
                    # were real MKs the corpus failed to match (גדעון סער, אחמד
                    # טיבי, היו"ר משה גפני ...).  Split them by whether the
                    # protocol's own speaker line names the person B chose.
                    sp = roster.spellings(b['person_id'])
                    kind = 'corpus_missed' if sp and agree(clean(b['raw_name']), sp) else 'guest_as_mk'
                else:
                    kind = 'unjudged'
            else:
                kind = 'missed' if a['label'] == 'mk' else 'both_unassigned'
            rows.append(dict(session_id=sid, knesset=int(mr.knesset), method=b['method'], kind=kind, d=d,
                             raw_name=a['raw_name'], a_id=a['person_id'], b_id=b['person_id']))
    return pd.DataFrame(rows)

def summarise(h):
    def block(g):
        t = g.groupby('kind').d.sum()
        right = t.get('correct', 0) + t.get('corpus_missed', 0)
        wrong = t.get('cross_person', 0) + t.get('guest_as_mk', 0)
        assigned = right + wrong
        return pd.Series(dict(hours_assigned=assigned / 3600,
                              precision=right / assigned if assigned else float('nan'),
                              cross_person_rate=wrong / assigned if assigned else float('nan'),
                              corpus_missed_h=t.get('corpus_missed', 0) / 3600,
                              recall=t.get('correct', 0) / (t.get('correct', 0) + t.get('missed', 0)) if (t.get('correct', 0) + t.get('missed', 0)) else float('nan')))
    out = {'ALL': block(h)}
    out.update({f'K{k}': block(g) for k, g in h.groupby('knesset')})
    out.update({m: block(g) for m, g in h[h.method != ''].groupby('method')})
    return pd.DataFrame(out).T.round(4)

def regression(roster, manifest):
    bad = []
    for sid, names in REGRESSION.items():
        if sid not in manifest.index or manifest.loc[sid].label_path != 'A':
            continue
        mr = manifest.loc[sid]; s = load_session(sid)
        placed, _ = place_gold(load_shard(int(mr.knesset), mr.protocol_name), s['text'])
        for w in label_words(word_table(s)[0], s, roster, placed=placed):
            if w['raw_name'] in names and w['label'] in ('mk', 'former_mk'):
                bad.append((sid, w['raw_name'], w['person_id']))
    return sorted(set(bad))

if __name__ == '__main__':
    R = Roster()
    man = pd.read_csv(os.path.join(OUT, 'manifest.csv')).set_index('session_id')
    aliases = dict(names=pd.read_csv(os.path.join(OUT, 'alias_names.csv')), et=pd.read_csv(os.path.join(OUT, 'alias_et.csv')))
    bad = regression(R, man)
    assert not bad, f'corpus guest->MK errors received an id: {bad}'
    print('regression: the known corpus errors all stay unassigned')
    ids = pd.read_parquet(os.path.join(OUT, 'segments_pilot_A.parquet')).session.unique()
    h = holdout(list(ids), R, aliases, man)
    h.to_parquet(os.path.join(OUT, 'holdout_words.parquet'), index=False)
    s = summarise(h)
    print(s.to_string())
    x = h[h.kind.isin(['cross_person', 'guest_as_mk'])].groupby(['kind', 'raw_name', 'a_id', 'b_id']).d.sum().sort_values(ascending=False)
    print('\nevery cross-person case (seconds):'); print(x.round(1).to_string() if len(x) else '  none')
    open(os.path.join(OUT, 'holdout_report.txt'), 'w', encoding='utf-8').write(s.to_string() + '\n\n' + x.round(1).to_string())
