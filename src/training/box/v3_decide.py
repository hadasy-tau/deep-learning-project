"""The decisions of plan v3, made by rule rather than by eye (docs/pod_runbook_v3.md).

pod_v3.sh calls this between stages, so the run never waits on someone reading a table:

    python src/training/box/v3_decide.py grid A 80     # the tuning stage's sweep flags for one budget
    python src/training/box/v3_decide.py pick A        # apply the rule to tuning.csv -> outputs/recipe_v3.json
    python src/training/box/v3_decide.py flags 80      # the chosen recipe as run_panel flags
    python src/training/box/v3_decide.py base [ids]    # base WER on the high-quality test vs the error map (exit 1 = stop)
    python src/training/box/v3_decide.py --self-check

The tuning rule (docs/training_plan_v3.md § 2), per budget (5 and 80 minutes), on the four
tuning speakers: a candidate replaces the incumbent only if it raises the mean rel_drop
(relative drop in validation loss) by at least 0.01 absolute AND beats the incumbent for at
least 3 of the 4 speakers; among candidates that pass, the largest mean gain wins.  The
incumbents are the simpler settings: lr 3e-4 (the second run's rate), rank 8, Whisper
dropout 0, no augmentation.  20 minutes takes 80's recipe; a 5/80 disagreement is recorded.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TRAINING = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(TRAINING, '..', '..'))
OUTPUTS = os.path.join(TRAINING, 'outputs')
RECIPE = os.path.join(OUTPUTS, 'recipe_v3.json')
TUNING = os.path.join(OUTPUTS, 'tuning.csv')
PLAN_V2 = os.path.join(TRAINING, 'panel_plan_v2.parquet')
PERF = os.path.join(ROOT, 'src', 'evaluation', 'outputs', 'committees_speaker_performance.csv')

TUNE_SPEAKERS = [30685, 23558, 30718, 30859]          # one per profile (S1, S2, S3, S4)
TUNE_BUDGETS = [5, 80]
INCUMBENT = dict(lr=3e-4, rank=8, dropout=0.0, augment='none')
LRS, RANKS, DROPOUTS, AUGMENTS = [1e-4, 3e-4, 1e-3], [8, 16], [0.0, 0.1], ['specaug', 'specaug+tempo']
MIN_GAIN, MIN_HELPED = 0.01, 3
BASE_MAX_ABOVE = 0.10  # training_plan_v3.md § 6: base WER more than this ABOVE the error map's wer_B = broken metric


# ---- tuning ------------------------------------------------------------------------------
def _key(r):
    return dict(lr=float(r['lr']), rank=int(r['rank']), dropout=float(r['dropout']), augment=str(r['augment']))

def _same(a, b):
    return np.isclose(a['lr'], b['lr']) and a['rank'] == b['rank'] and np.isclose(a['dropout'], b['dropout']) and a['augment'] == b['augment']

def rel_drops(T, budget, recipe):
    """rel_drop per tuning speaker for one recipe at one budget; all four must be there."""
    m = (T.budget == budget) & np.isclose(T.lr, recipe['lr']) & (T['rank'] == recipe['rank']) \
        & np.isclose(T.dropout.fillna(0.0), recipe['dropout']) & (T.augment.fillna('none') == recipe['augment'])
    s = T[m].assign(speaker=T[m].speaker.astype(int)).groupby('speaker').rel_drop.mean()
    missing = sorted(set(TUNE_SPEAKERS) - set(s.index))
    if missing:
        raise SystemExit(f'tuning.csv has no run for {recipe} at {budget} min for speakers {missing}: run that stage first')
    return s.loc[TUNE_SPEAKERS]

def candidates(stage, base):
    if stage == 'A': return [{**base, 'lr': lr} for lr in LRS]
    if stage == 'B': return [{**base, 'rank': r, 'dropout': d} for r in RANKS for d in DROPOUTS]
    return [{**base, 'augment': g} for g in ['none'] + AUGMENTS]

def incumbent(stage, base):
    return {**base, 'lr': INCUMBENT['lr']} if stage == 'A' else {**base, 'rank': INCUMBENT['rank'], 'dropout': INCUMBENT['dropout']} \
        if stage == 'B' else {**base, 'augment': INCUMBENT['augment']}

def pick_budget(T, stage, budget, base):
    """The stage's winner at one budget, and the comparison table that decided it."""
    inc = incumbent(stage, base); a = rel_drops(T, budget, inc)
    best, best_gain, rows = inc, 0.0, []
    for c in candidates(stage, base):
        if _same(c, inc): continue
        d = rel_drops(T, budget, c) - a
        ok = bool(d.mean() >= MIN_GAIN and int((d > 0).sum()) >= MIN_HELPED)
        rows.append(dict(budget=budget, **c, mean_rel_drop=float(a.mean() + d.mean()), gain_vs_incumbent=float(d.mean()),
                         helped=f'{int((d > 0).sum())}/4', passes=ok))
        if ok and d.mean() > best_gain: best, best_gain = c, float(d.mean())
    rows.insert(0, dict(budget=budget, **inc, mean_rel_drop=float(a.mean()), gain_vs_incumbent=0.0, helped='-', passes='incumbent'))
    return best, rows

def load_recipe():
    if os.path.exists(RECIPE): return json.load(open(RECIPE))
    return dict(stages=[], recipe={str(b): dict(INCUMBENT) for b in (5, 20, 80)}, log=[])

def pick(stage, T=None, save=True):
    R = load_recipe()
    need = {'A': [], 'B': ['A'], 'C': ['A', 'B']}[stage]
    if any(s not in R['stages'] for s in need): raise SystemExit(f'pick {stage} needs stages {need} picked first (have {R["stages"]})')
    T = T if T is not None else _tuning()
    table = []
    for b in TUNE_BUDGETS:
        win, rows = pick_budget(T, stage, b, R['recipe'][str(b)]); table += rows
        R['recipe'][str(b)] = win
    R['recipe']['20'] = dict(R['recipe']['80'])       # the plan's rule: 20 minutes takes the nearer end's value
    if not _same(R['recipe']['5'], R['recipe']['80']):
        R['log'].append(f'stage {stage}: 5 and 80 minutes disagree -- 5: {R["recipe"]["5"]}, 80: {R["recipe"]["80"]}; 20 takes 80 (a finding, report it)')
    R['stages'] = sorted(set(R['stages']) | {stage}); R.setdefault('tables', {})[stage] = table
    print(pd.DataFrame(table).to_string(index=False))
    print(f'\nstage {stage} picks: ' + '; '.join(f'{b} min {R["recipe"][str(b)]}' for b in (5, 20, 80)))
    for line in R['log']: print('NOTE', line)
    if save:
        os.makedirs(OUTPUTS, exist_ok=True); json.dump(R, open(RECIPE, 'w'), indent=1)
    return R

def _tuning():
    sys.path.insert(0, TRAINING); import run_panel as RP
    T = RP.tuning_report()
    if T is None: raise SystemExit('no tuning runs yet')
    return T

def _fmt(x): return f'{x:g}'

def grid(stage, budget):
    """The sweep flags of a tuning stage at one budget, at the recipe picked so far."""
    r = load_recipe()['recipe'][str(budget)]
    if stage == 'A': return f'--lrs {" ".join(_fmt(x) for x in LRS)}'
    if stage == 'B': return f'--lrs {_fmt(r["lr"])} --ranks {" ".join(map(str, RANKS))} --dropouts {" ".join(_fmt(x) for x in DROPOUTS)}'
    return f'--lrs {_fmt(r["lr"])} --ranks {r["rank"]} --dropouts {_fmt(r["dropout"])} --augments {" ".join(AUGMENTS)}'

def flags(budget):
    R = load_recipe()
    if R['stages'] != ['A', 'B', 'C']: raise SystemExit(f'the recipe is not final: stages picked {R["stages"]}')
    r = R['recipe'][str(budget)]
    return f'--lrs {_fmt(r["lr"])} --ranks {r["rank"]} --dropouts {_fmt(r["dropout"])} --augments {r["augment"]}'


# ---- the base-WER sanity check ------------------------------------------------------------
def base(audio_dir=None, batch=64, speakers=None):
    """Base B on each speaker's high-quality test set (cached as base_B_*_hq.json, the same file the
    final run reuses) against the error map's wer_B.  Clean clips should score BELOW it.  `speakers`
    restricts it to a subset: pod_v3.sh gives each GPU its own speakers, then reruns it on all of
    them, which only reads the caches, for the one table."""
    sys.path.insert(0, TRAINING); sys.path.insert(0, os.path.join(ROOT, 'src', 'evaluation'))
    import run_panel as RP, evaluate as EV
    P = RP.load_plan(PLAN_V2); perf = pd.read_csv(PERF, index_col='speaker_id', encoding='utf-8-sig')
    rows = []
    for s in sorted(int(x) for x in P.speaker_id.unique() if not speakers or int(x) in speakers):
        test = P[(P.speaker_id == s) & (P.part == 'test')].reset_index(drop=True)
        hb = RP.base_hyps('B', s, test, audio_dir or RP.AUDIO, batch, data='hq')
        sc = EV.score(test.text, hb); w = EV.wer(sc)
        rows.append(dict(speaker=s, n_test=len(test), wer_base_hq=w, wer_B_error_map=perf.at[s, 'wer_B'], above=w - perf.at[s, 'wer_B'],
                         runaway=int(sc.runaway.sum())))
    T = pd.DataFrame(rows); T['ok'] = T.above <= BASE_MAX_ABOVE
    print(T.round(3).to_string(index=False))
    if T.ok.all(): print('PASS: every speaker within the rule'); return True
    print(f'FAIL: base WER more than {BASE_MAX_ABOVE} above the error map for {T[~T.ok].speaker.tolist()} -- materialization or scoring is broken; stop')
    return False


# ---- self-checks ---------------------------------------------------------------------------
def _self_check():
    import tempfile
    global RECIPE
    RECIPE = os.path.join(tempfile.mkdtemp(), 'recipe.json')
    rows = []
    def add(budget, drops, **kw):
        r = {**INCUMBENT, **kw}
        for s, d in zip(TUNE_SPEAKERS, drops):
            rows.append(dict(speaker=str(s), budget=budget, lr=r['lr'], rank=r['rank'], dropout=r['dropout'] or None, augment=r['augment'], rel_drop=d))
    # A: at 80, 1e-3 gains 0.02 on 3 of 4 -> wins; 1e-4 loses.  At 5, 1e-3 gains 0.02 but on 2 of 4 -> 3e-4 stays.
    add(80, [.30, .30, .30, .30], lr=3e-4); add(80, [.33, .33, .33, .29], lr=1e-3); add(80, [.25, .25, .25, .25], lr=1e-4)
    add(5, [.10, .10, .10, .10], lr=3e-4); add(5, [.15, .15, .08, .08], lr=1e-3); add(5, [.05, .05, .05, .05], lr=1e-4)
    R = pick('A', pd.DataFrame(rows))
    assert np.isclose(R['recipe']['80']['lr'], 1e-3) and np.isclose(R['recipe']['5']['lr'], 3e-4) and np.isclose(R['recipe']['20']['lr'], 1e-3), R['recipe']
    assert any('disagree' in l for l in R['log'])
    assert grid('B', 80) == '--lrs 0.001 --ranks 8 16 --dropouts 0 0.1', grid('B', 80)
    # B at each budget's lr: a gain of 0.009 is below the line -> rank 8 / dropout 0 stays
    add(80, [.339, .339, .339, .299], lr=1e-3, rank=16); add(80, [.33, .33, .33, .29], lr=1e-3, rank=8, dropout=0.1); add(80, [.2] * 4, lr=1e-3, rank=16, dropout=0.1)
    add(5, [.1] * 4, lr=3e-4, rank=16); add(5, [.1] * 4, lr=3e-4, dropout=0.1); add(5, [.12, .12, .12, .12], lr=3e-4, rank=16, dropout=0.1)
    R = pick('B', pd.DataFrame(rows))
    assert R['recipe']['80']['rank'] == 8 and R['recipe']['80']['dropout'] == 0.0, R['recipe']['80']
    assert R['recipe']['5']['rank'] == 16 and np.isclose(R['recipe']['5']['dropout'], 0.1), R['recipe']['5']
    try: flags(80); raise AssertionError('flags before stage C must refuse')
    except SystemExit: pass
    for b, base_ in ((80, dict(lr=1e-3)), (5, dict(lr=3e-4, rank=16, dropout=0.1))):
        add(b, [.0] * 4, augment='specaug', **base_); add(b, [.0] * 4, augment='specaug+tempo', **base_)
    R = pick('C', pd.DataFrame(rows))
    assert flags(80) == '--lrs 0.001 --ranks 8 --dropouts 0 --augments none', flags(80)
    assert flags(5) == '--lrs 0.0003 --ranks 16 --dropouts 0.1 --augments none', flags(5)
    assert flags(20) == flags(80)
    try: rel_drops(pd.DataFrame(rows), 80, {**INCUMBENT, 'lr': 3e-3}); raise AssertionError('a missing recipe must refuse')
    except SystemExit: pass
    print('v3_decide self-check: OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', nargs='?', choices=['grid', 'pick', 'flags', 'base'])
    ap.add_argument('args', nargs='*'); ap.add_argument('--self-check', action='store_true'); ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    if a.self_check: _self_check(); sys.exit(0)
    if a.cmd == 'grid': print(grid(a.args[0], int(a.args[1])))
    elif a.cmd == 'pick': pick(a.args[0])
    elif a.cmd == 'flags': print(flags(int(a.args[0])))
    elif a.cmd == 'base': sys.exit(0 if base(batch=a.batch, speakers=[int(x) for x in a.args]) else 1)
    else: ap.error('give a command or --self-check')
