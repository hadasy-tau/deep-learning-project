"""Run 4's GPU work as one queue (docs/training_plan_v4.md), so no GPU waits on a stage.

Every job is one command -- a base transcription, one own cell, one control fold, the lr
decision -- with the jobs it needs first.  Workers (one or more per GPU) take the ready job
with the highest priority, longest first, so the long 1,440-minute jobs start as soon as
the lr is decided and nothing long is left for the end.  The queue is a JSON file under a
lock: workers on any number of GPUs share it, a failed job is retried once, and run_panel's
own idempotence (scored cells and finished adapters are skipped) makes every rerun safe.

    python src/training/box/jobs_v4.py build                # outputs/queue_v4.json from the plan (keeps statuses)
    python src/training/box/jobs_v4.py dry-run --gpus 4     # the jobs, GPU-hours, and the wall time on N GPUs
    python src/training/box/jobs_v4.py worker --gpu 0       # take jobs until none can become ready
    python src/training/box/jobs_v4.py status               # done / running / failed, elapsed and cost so far
    python src/training/box/jobs_v4.py release              # jobs put on hold by hand -> pending
    python src/training/box/jobs_v4.py --self-check

What runs (seeds: 3 at 80 and at each speaker's top budget, seed 0 elsewhere):
  23558  160 360 720 1440   (5/20/80 are run 3's)      23641  5 20 80 160 360 720 1440
  30752  160 360            (5/20/80 are run 3's)      23635  5 20 80 160 360
(5 minutes for 23641 and 23635 was added after the run, 2026-10-04, at run 3's 5-minute recipe.)
Controls: run 3's folds over the panel (run_panel.fold_plan), one job per fold that
evaluates any of these speakers at a (budget, seed) that has own cells; 23641 and 23635 are
attached to 23558's fold.  If the plan holds fewer than 1,440 train minutes for 23558 or
23641, its top budget falls back to 1,200 (docs/training_plan_v4.md § Data).
"""
import argparse, fcntl, json, os, subprocess, sys, time
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__)); TRAINING = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(TRAINING, '..', '..'))
OUTPUTS = os.path.join(TRAINING, 'outputs'); LOGS = os.path.join(OUTPUTS, 'logs')
QUEUE = os.path.join(OUTPUTS, 'queue_v4.json'); LOCK = QUEUE + '.lock'
PLAN = 'src/training/panel_plan_v4.parquet'; T07 = 'src/training/panel_test07_v4.parquet'
PY = os.environ.get('PY', 'python')
RP = f'{PY} src/training/run_panel.py --plan {PLAN} --test07-plan {T07} --eval-batch 64 --no-summary'
DECIDE = f'{PY} src/training/box/v4_decide.py'

PANEL_NEW = [23641, 23635]; ATTACH = 23558
SEEDS3 = [0, 1, 2]
# per speaker: budget -> seeds.  The top of 23558 and 23641 is 1440, or 1200 if the plan is short (top_budget)
OWN = {23558: {160: [0], 360: SEEDS3, 720: [0], 1440: SEEDS3},
       23641: {5: [0], 20: [0], 80: SEEDS3, 160: [0], 360: SEEDS3, 720: [0], 1440: SEEDS3},
       30752: {160: [0], 360: SEEDS3},
       23635: {5: [0], 20: [0], 80: SEEDS3, 160: [0], 360: SEEDS3}}
LR_CHECK = [3e-4, 1e-3]          # the 360-minute seed-0 cells at both rates are the lr check (v4_decide.py)
# The capacity check (added 2026-10-04): at each long-branch speaker's top budget, seed 0, the own
# adapter and 23558's fold control at r = 32 -- if r = 8 cannot absorb 1,440 minutes, it shows here.
CAPACITY_RANK, CAPACITY_SPEAKERS = 32, [23558, 23641]
PRICE = 1.59                     # A100 SXM 80 GB, secure cloud, $ per GPU-hour (2026-10-03)


def top_budget(P, sid):
    """1440, or 1200 when the plan holds fewer than 1,440 train minutes for the speaker."""
    m = P[(P.speaker_id == sid) & (P.part == 'train')].duration_s.sum() / 60
    return 1440 if m >= 1440 else 1200


def own_spec(P):
    spec = {}
    for s, bs in OWN.items():
        spec[s] = {}
        for b, seeds in bs.items():
            if b == 1440: b = top_budget(P, s)
            spec[s][b] = seeds
    return spec


def est_min(budget, n_evals=1):
    """Minutes on one A100 at batch 8: steps ~ 2 epochs + patience (80), at least 160 (run 3:
    early-stopped near 160 at 80 min), capped by v4_decide.MAX_STEPS; 0.52 s a step, a
    validation every 20 steps, ~1.5 min to score one speaker's two test sets."""
    sys.path.insert(0, HERE); import v4_decide as V4
    chunks = budget * 60 / 15.7; steps = min(V4.MAX_STEPS[budget], max(160, 2 * chunks / 8 + 80))
    if budget in V4.MIN_EPOCHS: steps = max(steps, V4.MIN_EPOCHS[budget] * chunks / 8 + 80)
    return (steps * 0.52 + steps / 20 * 4 + 60) / 60 + 1.5 * n_evals


def build_jobs(P):
    """The job list: id, cmd, deps, est (minutes), prio (higher first)."""
    sys.path.insert(0, TRAINING); import run_panel as RPM
    spec = own_spec(P); jobs = []
    def job(id, cmd, deps=(), est=1.0, prio=0): jobs.append(dict(id=id, cmd=cmd, deps=list(deps), est=round(est, 1), prio=prio))
    # the training loop's sanity check (run 2: 0.58 -> 0.0005 in 60 steps); a loss that does not fall stops the queue
    job('overfit', f'{PY} -c "import sys, numpy as np, pandas as pd; sys.path.insert(0, \'src/training\'); import train; '
                   f'L = train.overfit_check(pd.read_parquet(\'{PLAN}\'), \'src/training/outputs/panel_audio\', speaker=23641, arm=\'B\'); '
                   f'assert np.mean(L[-5:]) < 0.2 * np.mean(L[:5]), \'the loss did not fall\'"', est=2, prio=100)
    for s in spec: job(f'base_{s}', f'{DECIDE} base {s}', deps=['overfit'], est=4, prio=90)
    flags = lambda b: f'$({DECIDE} flags {b})'
    # the lr check: 360-minute seed-0 cells at both rates, with explicit flags (no pick needed)
    for s in spec:
        for lr in LR_CHECK:
            job(f'own_{s}_b360_s0_lr{lr:g}', f'{RP} --cell-speakers {s} --budgets 360 --seeds 0 --lrs {lr:g} --ranks 8 --dropouts 0 '
                f'--augments none --max-steps 1200', deps=[f'base_{s}'], est=est_min(360), prio=80)
    job('decide', f'{DECIDE} pick', deps=[f'own_{s}_b360_s0_lr{lr:g}' for s in spec for lr in LR_CHECK], est=0.5, prio=85)
    after = lambda b: ['decide'] if b >= 360 else []
    for s, bs in spec.items():
        for b, seeds in bs.items():
            for seed in seeds:
                if b == 360 and seed == 0: continue          # the lr check made it
                job(f'own_{s}_b{b}_s{seed}', f'{RP} --cell-speakers {s} --budgets {b} --seeds {seed} {flags(b)}',
                    deps=[f'base_{s}'] + after(b), est=est_min(b))
    # controls: one job per fold that evaluates someone at a (budget, seed) with own cells
    panel = sorted(int(x) for x in P.speaker_id.unique() if int(x) not in PANEL_NEW)
    for b in sorted({b for bs in spec.values() for b in bs}):
        for seed in sorted({x for s in spec for x in spec[s].get(b, [])}):
            evals = sorted(s for s in spec if seed in spec[s].get(b, []))
            for k, members, attached, ev, trainers, label in RPM.fold_plan(panel + PANEL_NEW, 2, seed, PANEL_NEW, ATTACH, evals):
                job(f'ctrl_b{b}_s{seed}_{label}', f'{RP} --control-only --control-folds 2 --control-budgets {b} --budgets {b} --seeds {seed} '
                    f'{flags(b)} --control-evals {" ".join(map(str, ev))} --control-extra {" ".join(map(str, PANEL_NEW))} --control-attach {ATTACH}',
                    deps=[f'base_{s}' for s in ev if s in spec] + after(b), est=est_min(b, len(ev)))
    rflags = lambda b: f'$({DECIDE} flags {b} {CAPACITY_RANK})'
    for s in CAPACITY_SPEAKERS:
        b = max(spec[s])
        job(f'own_{s}_b{b}_s0_r{CAPACITY_RANK}', f'{RP} --cell-speakers {s} --budgets {b} --seeds 0 {rflags(b)}',
            deps=[f'base_{s}', 'decide'], est=est_min(b))
    b = max(spec[CAPACITY_SPEAKERS[0]]); ev = sorted(CAPACITY_SPEAKERS)
    for k, members, attached, e, trainers, label in RPM.fold_plan(panel + PANEL_NEW, 2, 0, PANEL_NEW, ATTACH, ev):
        job(f'ctrl_b{b}_s0_{label}_r{CAPACITY_RANK}', f'{RP} --control-only --control-folds 2 --control-budgets {b} --budgets {b} --seeds 0 '
            f'{rflags(b)} --control-evals {" ".join(map(str, e))} --control-extra {" ".join(map(str, PANEL_NEW))} --control-attach {ATTACH}',
            deps=[f'base_{x}' for x in e] + ['decide'], est=est_min(b, len(e)))
    ids = [j['id'] for j in jobs]; assert len(ids) == len(set(ids)), 'duplicate job ids'
    assert all(d in ids for j in jobs for d in j['deps']), 'a dependency that is not a job'
    return jobs


def simulate(jobs, gpus):
    """Wall minutes on `gpus` workers with the queue's own rule (ready, prio, longest first)."""
    done, t, free = {}, 0.0, [0.0] * gpus; left = {j['id']: j for j in jobs}
    while left:
        g = min(range(gpus), key=lambda i: free[i]); t = free[g]
        ready = [j for j in left.values() if all(d in done and done[d] <= t for d in j['deps'])]
        if not ready:
            nxt = min(v for v in done.values() if v > t) if any(v > t for v in done.values()) else None
            if nxt is None: raise RuntimeError('deadlock in the job graph')
            free[g] = nxt; continue
        j = max(ready, key=lambda j: (j['prio'], j['est'])); del left[j['id']]
        done[j['id']] = free[g] = t + j['est']
    return max(done.values())


@contextmanager
def locked():
    os.makedirs(OUTPUTS, exist_ok=True)
    with open(LOCK, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            Q = json.load(open(QUEUE)) if os.path.exists(QUEUE) else {}
            yield Q
            tmp = QUEUE + '.tmp'; json.dump(Q, open(tmp, 'w'), indent=1); os.replace(tmp, QUEUE)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build(P):
    jobs = build_jobs(P)
    with locked() as Q:
        old = {j['id']: j for j in Q.get('jobs', [])}
        for j in jobs:
            if j['id'] in old: j.update({k: old[j['id']][k] for k in ('status', 'attempts', 'start', 'end', 'gpu') if k in old[j['id']]})
            j.setdefault('status', 'pending'); j.setdefault('attempts', 0)
        Q['jobs'] = jobs; Q.setdefault('created', time.time())
    print(f'{len(jobs)} jobs, {sum(j["est"] for j in jobs) / 60:.1f} GPU-hours estimated -> {QUEUE}')


def release():
    with locked() as Q:
        n = 0
        for j in Q['jobs']:
            if j['status'] == 'hold': j['status'] = 'pending'; n += 1
    print(f'released {n} job(s) from hold')


def _ready(Q):
    st = {j['id']: j['status'] for j in Q['jobs']}
    return [j for j in Q['jobs'] if j['status'] == 'pending' and all(st[d] == 'done' for d in j['deps'])]


def _dead(Q):
    """Pending jobs that can never run: a dependency failed for good, directly or not."""
    st = {j['id']: j['status'] for j in Q['jobs']}; deps = {j['id']: j['deps'] for j in Q['jobs']}
    def blocked(i, seen=()):
        return st[i] == 'failed' or any(blocked(d, seen + (i,)) for d in deps[i] if d not in seen)
    return [j['id'] for j in Q['jobs'] if j['status'] == 'pending' and blocked(j['id'])]


def _finished(Q):
    """True when no job can run any more: nothing running, nothing on hold (a job put on hold by hand
    is waited for, never read as the end -- run 4's first pass closed itself that way), and every
    pending job blocked by a failure."""
    st = [j['status'] for j in Q['jobs']]
    if 'running' in st or 'hold' in st: return False
    pending = st.count('pending')
    return pending == 0 or len(_dead(Q)) == pending


def worker(gpu, poll=20):
    os.makedirs(LOGS, exist_ok=True); name = f'gpu{gpu}.pid{os.getpid()}'
    while True:
        with locked() as Q:
            ready = _ready(Q)
            if ready:
                j = max(ready, key=lambda j: (j['prio'], j['est']))
                j.update(status='running', start=time.time(), gpu=gpu, worker=name); jid, cmd = j['id'], j['cmd']
            else:
                if _finished(Q):
                    print(f'[{name}] nothing left to run ({sum(j["status"] == "pending" for j in Q["jobs"])} pending blocked)', flush=True); return
                jid = None
        if jid is None: time.sleep(poll); continue
        log = os.path.join(LOGS, f'job_{jid}.log'); t0 = time.time()
        print(f'[{name}] {time.strftime("%H:%M:%S")} start {jid}', flush=True)
        with open(log, 'a') as f:
            f.write(f'\n=== {time.strftime("%F %T")} {name}\n$ {cmd}\n'); f.flush()
            rc = subprocess.call(['bash', '-c', f'set -euo pipefail; {cmd}'], stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, cwd=ROOT)
        with locked() as Q:
            j = next(x for x in Q['jobs'] if x['id'] == jid); j['attempts'] = j.get('attempts', 0) + 1; j['end'] = time.time()
            j['status'] = 'done' if rc == 0 else ('pending' if j['attempts'] < 2 else 'failed')
            if rc != 0 and jid == 'overfit': j['status'] = 'failed'
        print(f'[{name}] {time.strftime("%H:%M:%S")} {"done" if rc == 0 else f"FAILED rc={rc}"} {jid} in {(time.time()-t0)/60:.1f} min (log {log})', flush=True)
        if rc != 0 and jid == 'overfit':
            print(f'[{name}] the overfit check failed: the training loop is broken, stopping', flush=True); return


def status(gpus=None):
    Q = json.load(open(QUEUE)); J = Q['jobs']; now = time.time()
    by = {}
    for j in J: by.setdefault(j['status'], []).append(j)
    print(' '.join(f'{k} {len(v)}' for k, v in sorted(by.items())))
    for j in by.get('running', []): print(f'  running {j["id"]} on gpu {j["gpu"]} for {(now - j["start"]) / 60:.0f} min (est {j["est"]})')
    for j in by.get('failed', []): print(f'  FAILED {j["id"]} -- outputs/logs/job_{j["id"]}.log')
    dead = _dead(Q)
    if dead: print(f'  blocked by a failure: {dead}')
    started = [j['start'] for j in J if j.get('start')]
    if started:
        el = (now - min(started)) / 3600; n = gpus or len({j['gpu'] for j in J if 'gpu' in j})
        left = sum(j['est'] for j in J if j['status'] in ('pending', 'running')) / 60
        took = [(j['end'] - j['start']) / 60 / j['est'] for j in J if j['status'] == 'done' and j.get('end') and j['est'] >= 3]
        ratio = sorted(took)[len(took) // 2] if took else 1.0
        print(f'elapsed {el:.2f} h on {n} GPU(s) = ${el * n * PRICE:.1f}; remaining ~{left * ratio:.1f} GPU-h '
              f'(estimates x{ratio:.2f} as measured) = ~{left * ratio / max(n, 1):.1f} h wall, ~${left * ratio * PRICE:.1f} more')


def _self_check():
    import pandas as pd
    sys.path.insert(0, TRAINING); import run_panel as RPM
    panel = [556, 23558, 30685, 30701, 30718, 30752, 30777, 30813, 30831, 30843, 30859, 30868]
    rows = [dict(speaker_id=s, part='train', duration_s=1500 * 60 if s in (23558, 23641) else 60.0) for s in panel + PANEL_NEW]
    P = pd.DataFrame(rows)
    J = build_jobs(P); ids = {j['id'] for j in J}
    assert 'own_23558_b1440_s2' in ids and 'own_23558_b80_s0' not in ids and 'own_30752_b720_s0' not in ids
    assert {'own_23641_b360_s0_lr0.0003', 'own_23641_b360_s0_lr0.001', 'own_23641_b360_s1'} <= ids and 'own_23641_b360_s0' not in ids
    # the 20- and 80-minute controls exist only for the new speakers, on 23558's fold (x), never on run 3's own name
    c80 = [j for j in J if j['id'].startswith('ctrl_b80_')]
    assert len(c80) == 3 and all(j['id'].endswith('x') and '--control-evals 23641 23635' in j['cmd'] for j in c80)
    # at 360 every speaker is evaluated, on run 3's folds
    for seed in SEEDS3:
        c = [j for j in J if j['id'].startswith(f'ctrl_b360_s{seed}_')]
        ev = sorted(int(x) for j in c for x in j['cmd'].split('--control-evals ')[1].split(' --')[0].split())
        assert ev == sorted(OWN), (seed, ev)
        for j in c:
            k = int(j['id'].split('ctrl')[-1][0]); members = RPM.control_folds(panel, 2, seed)[k]
            assert all(int(x) in members + PANEL_NEW for x in j['cmd'].split('--control-evals ')[1].split(' --')[0].split())
    assert all('decide' in j['deps'] for j in J if '_b1440_' in j['id'] or '_b720_' in j['id'] or (('_b360_' in j['id']) and '_lr' not in j['id']))
    assert not any('decide' in j['deps'] for j in J if any(f'_b{b}_' in j['id'] for b in (5, 20, 80, 160)))
    # 5 minutes: the new speakers only, with one copy of 23558's fold control for them (run 3 has 23558's own)
    assert {j['id'] for j in J if '_b5_' in j['id']} == {'own_23641_b5_s0', 'own_23635_b5_s0', 'ctrl_b5_s0_ctrl1of2x'}, \
        sorted(j['id'] for j in J if '_b5_' in j['id'])
    P2 = P.assign(duration_s=[1300 * 60 if s == 23641 else d for s, d in zip(P.speaker_id, P.duration_s)])
    assert 'own_23641_b1200_s1' in {j['id'] for j in build_jobs(P2)}            # the 1,200 fallback
    Qt = dict(jobs=[dict(id='a', status='done', deps=[]), dict(id='b', status='hold', deps=['a'])])
    assert not _finished(Qt) and _finished(dict(jobs=[dict(id='a', status='done', deps=[])]))
    assert _finished(dict(jobs=[dict(id='a', status='failed', deps=[]), dict(id='b', status='pending', deps=['a'])]))
    assert {'own_23558_b1440_s0_r32', 'own_23641_b1440_s0_r32', 'ctrl_b1440_s0_ctrl1of2x_r32'} <= ids
    assert all('flags 1440 32)' in j['cmd'] for j in J if j['id'].endswith('_r32'))
    w1, w4 = simulate(J, 1), simulate(J, 4)
    assert abs(w1 - sum(j['est'] for j in J)) < 1e-6 and w4 < w1 / 3, (w1, w4)
    print(f'jobs_v4 self-check: OK ({len(J)} jobs, {w1 / 60:.1f} GPU-h, {w4 / 60:.1f} h on 4 GPUs)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', nargs='?', choices=['build', 'dry-run', 'worker', 'status', 'release'])
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--gpus', type=int, default=4)
    ap.add_argument('--self-check', action='store_true')
    a = ap.parse_args()
    if a.self_check: _self_check(); sys.exit(0)
    if a.cmd in ('build', 'dry-run'):
        import pandas as pd
        P = pd.read_parquet(os.path.join(ROOT, PLAN))
        if a.cmd == 'build': build(P)
        else:
            J = build_jobs(P)
            for j in sorted(J, key=lambda j: (-j['prio'], -j['est'])): print(f'{j["est"]:6.1f}  {j["id"]:40s} after {",".join(j["deps"]) or "-"}')
            gh = sum(j['est'] for j in J) / 60; w = simulate(J, a.gpus) / 60 + 0.25
            print(f'\n{len(J)} jobs; {gh:.1f} GPU-hours; on {a.gpus} GPU(s) ~{w:.1f} h wall incl. ~15 min boot = ~${w * a.gpus * PRICE:.0f}')
    elif a.cmd == 'worker': worker(a.gpu)
    elif a.cmd == 'status': status(a.gpus)
    elif a.cmd == 'release': release()
    else: ap.error('give a command or --self-check')
