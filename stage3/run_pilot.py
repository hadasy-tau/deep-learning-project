"""Stage 3 pilot: Path A on a stratified slice, learn the alias tables, then Path
B on post-cutoff K25 sessions.  Writes segments_pilot.parquet, alias_names.csv,
alias_et.csv and a match report.  Run order from stage3/plan.md.
"""
import os, sys, time
import pandas as pd

from roster import OUT
from names import build_alias
from label import Roster
from segments import build

N_A = {20: 40, 21: 10, 22: 10, 23: 50, 24: 45, 25: 45}      # Path-A sessions per knesset
N_B = 50                                                      # Path-B sessions (K25 after the cutoff)

def pick(man, path, per_k, seed=0):
    m = man[man.label_path == path]
    return [sid for k, n in per_k.items()
            for sid in m[m.knesset == k].sample(min(n, (m.knesset == k).sum()), random_state=seed).index]

def report(df, st, aliases):
    lines = []
    t = df.groupby(['label_path', 'label']).duration_s.sum().div(3600).round(2).unstack(fill_value=0)
    lines += ['hours by path and label', t.to_string(), '']
    r = df[df.label == 'unresolved'].groupby(['label_path', 'reason']).duration_s.sum().div(3600).round(2)
    lines += ['unresolved hours by reason', r.to_string(), '']
    m = df[df.label == 'mk'].groupby(['label_path', 'match_method']).duration_s.sum().div(3600).round(2)
    lines += ['mk hours by match method', m.to_string(), '']
    lines += [f"mk speakers: {df[df.label == 'mk'].speaker_id.nunique()}  former_mk speakers: {df[df.label == 'former_mk'].speaker_id.nunique()}", '']
    lines += ['sessions', st.groupby('path').agg(n=('session_id', 'size'), errors=('error', lambda s: s.notna().sum()) if 'error' in st else ('session_id', 'size'),
                                                 placed_median=('placed', 'median')).to_string(), '']
    lines += [f"alias_names: {len(aliases['names'])} keys, {aliases['names'].usable.sum()} usable; "
              f"alias_et: {len(aliases['et'])} keys, {aliases['et'].usable.sum()} usable, "
              f"{(aliases['et'].purity < 1).sum()} impure", '']
    return '\n'.join(lines)

if __name__ == '__main__':
    R = Roster()
    man = pd.read_csv(os.path.join(OUT, 'manifest.csv')).set_index('session_id')
    a_ids = pick(man, 'A', N_A)
    print(f'Path A pilot: {len(a_ids)} sessions', flush=True); t = time.time()
    # keep=True caches refined.json (~3 MB each) so validate.py re-reads these
    # sessions from disk instead of re-downloading 600 MB.
    dfa, pairs, sta = build(a_ids, R, out_name='segments_pilot_A.parquet', keep=True)
    aliases = dict(names=build_alias(pairs['names']), et=build_alias(pairs['et']))
    aliases['names'].to_csv(os.path.join(OUT, 'alias_names.csv'), index=False)
    aliases['et'].to_csv(os.path.join(OUT, 'alias_et.csv'), index=False)
    print(f'  done in {time.time()-t:.0f}s; aliases learned', flush=True)

    b_ids = pick(man, 'B', {25: N_B})
    print(f'Path B pilot: {len(b_ids)} sessions', flush=True); t = time.time()
    dfb, _, stb = build(b_ids, R, aliases=aliases, out_name='segments_pilot_B.parquet')
    print(f'  done in {time.time()-t:.0f}s', flush=True)

    df = pd.concat([dfa, dfb], ignore_index=True); st = pd.concat([sta, stb], ignore_index=True)
    df.to_parquet(os.path.join(OUT, 'segments_pilot.parquet'), index=False)
    st.to_csv(os.path.join(OUT, 'pilot_sessions.csv'), index=False)
    if 'error' in st and st.error.notna().any():
        print('\nsessions that failed:')
        print(st[st.error.notna()][['session_id', 'path', 'error']].to_string(index=False))
    rep = report(df, st, aliases)
    open(os.path.join(OUT, 'match_report_pilot.txt'), 'w', encoding='utf-8').write(rep)
    print(rep)
    # the unresolved names Path B could not place, most time first -- what a human should read
    u = dfb[dfb.label == 'unresolved'].groupby(['reason', 'raw_name']).duration_s.sum().sort_values(ascending=False)
    print('\ntop unresolved raw names under Path B (minutes):')
    print(u.div(60).round(1).head(40).to_string())
