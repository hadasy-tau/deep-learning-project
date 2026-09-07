"""Stage 3 full build: label every complete committee session.

Path A over all sessions that have a corpus shard, learning the alias tables from
all of them (the pilot learned from 200); then Path B over the rest.

Three things make an 11k-session run practical:
  * threads -- the work is network-bound (one ~3 MB refined.json per session),
    so a pool turns ~4 s/session into well under one
  * batches -- rows are written to outputs/parts/*.parquet as each batch closes,
    so peak memory stays flat instead of holding ~5M rows
  * resume -- a batch whose part file exists is skipped, so an interrupted run
    costs at most one batch

Run:  python stage3/run_full.py            (both paths, then assemble)
"""
import os, sys, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from roster import OUT
from names import build_alias
from label import Roster
from segments import process, coerce

# 12 workers × 9 hub calls per session drew HTTP 429s within two batches; the small
# files now go over resolve/main (5 calls per session) and the pool is smaller.
WORKERS, BATCH = 6, 400
PARTS = os.path.join(OUT, 'parts')

def batches(xs, n):
    for i in range(0, len(xs), n):
        yield i // n, xs[i:i + n]

def run_path(path, session_ids, roster, man, aliases=None):
    """Label `session_ids` in parallel batches.  Returns (alias pairs, stats)."""
    os.makedirs(PARTS, exist_ok=True)
    pairs, stats, t0 = dict(names=[], et=[]), [], time.time()
    done_rows = fresh = 0
    for bi, chunk in batches(session_ids, BATCH):
        pp = os.path.join(PARTS, f'{path}_{bi:04d}.parquet')
        sp = os.path.join(PARTS, f'{path}_{bi:04d}_stats.csv')
        ap = os.path.join(PARTS, f'{path}_{bi:04d}_alias.csv')
        if os.path.exists(pp) and os.path.exists(sp):
            st = pd.read_csv(sp); stats.append(st)
            if os.path.exists(ap):
                a = pd.read_csv(ap)
                pairs['names'] += list(a[a.kind == 'name'][['key', 'person_id']].itertuples(index=False, name=None))
                pairs['et'] += list(a[a.kind == 'et'][['key', 'person_id']].itertuples(index=False, name=None))
            done_rows += int(st.rows.fillna(0).sum())
            print(f'  [{path} {bi:04d}] resumed', flush=True)
            continue
        rows, bstats, bpairs = [], [], []
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(process, sid, man.loc[sid], roster, aliases, False): sid for sid in chunk}
            for f in as_completed(futs):
                sid = futs[f]
                try:
                    r, p, st = f.result()
                except Exception as e:
                    bstats.append(dict(session_id=sid, path=path, error=repr(e)[:200])); continue
                rows += r; bstats.append(st)
                bpairs += [dict(kind='name', key=k, person_id=v) for k, v in p['names']]
                bpairs += [dict(kind='et', key=k, person_id=v) for k, v in p['et']]
        coerce(pd.DataFrame(rows)).to_parquet(pp, index=False)
        pd.DataFrame(bstats).to_csv(sp, index=False)
        if bpairs:
            pd.DataFrame(bpairs).to_csv(ap, index=False)
            pairs['names'] += [(d['key'], d['person_id']) for d in bpairs if d['kind'] == 'name']
            pairs['et'] += [(d['key'], d['person_id']) for d in bpairs if d['kind'] == 'et']
        stats.append(pd.DataFrame(bstats)); done_rows += len(rows); fresh += len(chunk)
        n = min((bi + 1) * BATCH, len(session_ids))
        el = time.time() - t0
        # rate over sessions actually fetched -- resumed batches cost no time, and
        # dividing by n made the first ETA after a resume wildly optimistic
        print(f'  [{path} {bi:04d}] {n}/{len(session_ids)} sessions, {done_rows:,} rows, '
              f'{el/60:.1f} min, eta {el/fresh*(len(session_ids)-n)/60:.0f} min', flush=True)
    return pairs, pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()

def assemble():
    parts = sorted(f for f in os.listdir(PARTS) if f.endswith('.parquet'))
    df = pd.concat([pd.read_parquet(os.path.join(PARTS, f)) for f in parts], ignore_index=True)
    df.to_parquet(os.path.join(OUT, 'segments.parquet'), index=False)
    return df

if __name__ == '__main__':
    R = Roster()
    man = pd.read_csv(os.path.join(OUT, 'manifest.csv')).set_index('session_id')
    a_ids = sorted(man[man.label_path == 'A'].index)
    b_ids = sorted(man[man.label_path == 'B'].index)
    print(f'Path A: {len(a_ids):,} sessions   Path B: {len(b_ids):,} sessions', flush=True)

    pairs, sta = run_path('A', a_ids, R, man)
    aliases = dict(names=build_alias(pairs['names']), et=build_alias(pairs['et']))
    aliases['names'].to_csv(os.path.join(OUT, 'alias_names.csv'), index=False)
    aliases['et'].to_csv(os.path.join(OUT, 'alias_et.csv'), index=False)
    print(f"aliases: {len(aliases['names'])} names ({aliases['names'].usable.sum()} usable), "
          f"{len(aliases['et'])} editor ids", flush=True)

    _, stb = run_path('B', b_ids, R, man, aliases)
    st = pd.concat([sta, stb], ignore_index=True)
    st.to_csv(os.path.join(OUT, 'build_sessions.csv'), index=False)
    df = assemble()

    from run_pilot import report
    rep = report(df, st, aliases)
    open(os.path.join(OUT, 'match_report.txt'), 'w', encoding='utf-8').write(rep)
    print('\n' + rep)
    err = st.error.notna().sum() if 'error' in st else 0
    print(f'sessions attempted {len(st):,}, failed {err}, rows {len(df):,}')
