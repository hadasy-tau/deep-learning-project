"""The decisions of run 4, made by rule (docs/training_plan_v4.md).

    python src/training/box/v4_decide.py pick           # the lr check at 360 min -> outputs/recipe_v4.json
    python src/training/box/v4_decide.py flags 360      # one budget's recipe as run_panel flags (refuses before pick)
    python src/training/box/v4_decide.py base [ids]     # base B on the v4 test sets vs the error map (exit 1 = stop)
    python src/training/box/v4_decide.py --self-check

The recipe is run 3's (outputs/recipe_v3.json at 20 and 80 minutes: lr 3e-4, rank 8, no
dropout, no augmentation).  Only the learning rate is checked again, at 360 minutes, where
nothing was tuned: 3e-4 against 1e-3, on validation, over run 4's four speakers, with run
3's rule (v3_decide.pick_budget: a candidate replaces the incumbent only if it raises the
mean rel_drop by >= 0.01 AND helps >= 3 of the 4).  720 and 1,440 take 360's rate; 160 takes
360's if it equals 80's, otherwise 80's -- 160 is nearer 80 on a log scale -- so 160 is
always 3e-4 and needs no pick.  The step cap grows with the data: 400 to 80 minutes (run
3's, unchanged), 1,200 at 160 and 360, 3,000 at 720 and 1,440.

The check is run as real cells: each speaker's 360-minute seed-0 cell at both rates (the
queue's first jobs after the base transcriptions).  The rule reads only their validation
curves (train_meta.json: base_eval_loss and the best eval loss), so scoring them on the test
set as well decides nothing, and the winning rate's seed-0 cells need not be trained again.
The losing rate's cells get no control, so they never enter a personalization number.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TRAINING = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(TRAINING, '..', '..'))
sys.path.insert(0, HERE)
import v3_decide as V3

OUTPUTS = os.path.join(TRAINING, 'outputs')
RECIPE = os.path.join(OUTPUTS, 'recipe_v4.json')
TUNING = os.path.join(OUTPUTS, 'tuning_v4.csv')
PLAN_V4 = os.path.join(TRAINING, 'panel_plan_v4.parquet')
TEST07_V4 = os.path.join(TRAINING, 'panel_test07_v4.parquet')

SPEAKERS = [23558, 23641, 30752, 23635]               # run 4: two panel speakers, two new
TUNE_BUDGET = 360
LRS = [3e-4, 1e-3]
INCUMBENT = dict(V3.INCUMBENT)                          # lr 3e-4, rank 8, dropout 0, augment none
MAX_STEPS = {20: 400, 80: 400, 160: 1200, 360: 1200, 720: 3000, 1440: 3000, 1200: 3000}


def pick(T=None, save=True):
    """The lr at 360 minutes by run 3's rule, over SPEAKERS -> recipe_v4.json."""
    if T is None:
        sys.path.insert(0, TRAINING); import run_panel as RP
        T = RP.tuning_report(out=TUNING, runs=RP.RUNS)
        if T is None: raise SystemExit('no 360-minute cells yet')
        T = T[(T.seed == 0) & (T.max_steps == MAX_STEPS[TUNE_BUDGET])]
    T = T[T.speaker.astype(int).isin(SPEAKERS) & (T.budget == TUNE_BUDGET)]
    V3.TUNE_SPEAKERS, V3.LRS = SPEAKERS, LRS            # run 3's rule, on run 4's speakers and grid
    win, table = V3.pick_budget(T, 'A', TUNE_BUDGET, dict(INCUMBENT))
    lr360 = float(win['lr']); lr80 = float(INCUMBENT['lr'])
    lr = {20: lr80, 80: lr80, 160: lr360 if np.isclose(lr360, lr80) else lr80, 360: lr360, 720: lr360, 1440: lr360, 1200: lr360}
    R = dict(picked=True, lr={str(b): v for b, v in lr.items()}, rank=INCUMBENT['rank'], dropout=INCUMBENT['dropout'],
             augment=INCUMBENT['augment'], max_steps={str(b): v for b, v in MAX_STEPS.items()}, table=table)
    print(pd.DataFrame(table).to_string(index=False))
    print(f'\nlr: ' + '; '.join(f'{b} min {v:g}' for b, v in lr.items()))
    if save:
        os.makedirs(OUTPUTS, exist_ok=True); json.dump(R, open(RECIPE, 'w'), indent=1)
    return R


def flags(budget):
    """run_panel flags for one budget.  20, 80 and 160 never depend on the pick; 360 and above refuse before it."""
    budget = int(budget)
    if budget <= 160: lr = INCUMBENT['lr']
    else:
        if not os.path.exists(RECIPE): raise SystemExit(f'the lr at {TUNE_BUDGET} min is not picked yet: run `v4_decide.py pick` after the tuning jobs')
        lr = json.load(open(RECIPE))['lr'][str(budget)]
    return (f'--lrs {V3._fmt(lr)} --ranks {INCUMBENT["rank"]} --dropouts {V3._fmt(INCUMBENT["dropout"])} '
            f'--augments {INCUMBENT["augment"]} --max-steps {MAX_STEPS[budget]}')


def base(audio_dir=None, batch=64, speakers=None):
    """Base B on each speaker's high-quality and >= 0.7 test sets (cached as base_B_*_hq.json and
    *_test07.json, the files every cell then reuses) against the error map's wer_B, by v3's rule:
    clean clips should score below it, and more than BASE_MAX_ABOVE above means a broken metric."""
    sys.path.insert(0, TRAINING); sys.path.insert(0, os.path.join(ROOT, 'src', 'evaluation'))
    import run_panel as RP, evaluate as EV
    P = RP.load_plan(PLAN_V4); T07 = RP.load_plan(TEST07_V4)
    perf = pd.read_csv(V3.PERF, index_col='speaker_id', encoding='utf-8-sig')
    rows = []
    for s in sorted(speakers or SPEAKERS):
        test = P[(P.speaker_id == s) & (P.part == 'test')].reset_index(drop=True)
        hb = RP.base_hyps('B', s, test, audio_dir or RP.AUDIO, batch, data='hq')
        t07 = T07[T07.speaker_id == s].reset_index(drop=True)
        RP.base_hyps('B', s, t07, audio_dir or RP.AUDIO, batch, data='test07')
        sc = EV.score(test.text, hb); w = EV.wer(sc)
        rows.append(dict(speaker=s, n_test=len(test), n_test07=len(t07), wer_base_hq=w, wer_B_error_map=perf.at[s, 'wer_B'],
                         above=w - perf.at[s, 'wer_B'], runaway=int(sc.runaway.sum())))
    T = pd.DataFrame(rows); T['ok'] = T.above <= V3.BASE_MAX_ABOVE
    print(T.round(3).to_string(index=False))
    if T.ok.all(): print('PASS: every speaker within the rule'); return True
    print(f'FAIL: base WER more than {V3.BASE_MAX_ABOVE} above the error map for {T[~T.ok].speaker.tolist()} -- stop')
    return False


def _self_check():
    import tempfile
    global RECIPE
    RECIPE = os.path.join(tempfile.mkdtemp(), 'recipe.json')
    assert flags(80) == '--lrs 0.0003 --ranks 8 --dropouts 0 --augments none --max-steps 400', flags(80)
    assert flags(160).endswith('--max-steps 1200') and '--lrs 0.0003' in flags(160)
    try: flags(360); raise AssertionError('flags(360) before pick must refuse')
    except SystemExit: pass
    def table(d_inc, d_cand):
        rows = [dict(speaker=str(s), budget=TUNE_BUDGET, lr=lr, rank=8, dropout=None, augment='none', rel_drop=d)
                for lr, ds in ((3e-4, d_inc), (1e-3, d_cand)) for s, d in zip(SPEAKERS, ds)]
        return pd.DataFrame(rows + [dict(speaker='30685', budget=80, lr=1e-3, rank=8, dropout=None, augment='none', rel_drop=.9)])
    R = pick(table([.40] * 4, [.43, .43, .43, .39]))                     # +0.02 mean, 3 of 4: 1e-3 wins at 360
    assert np.isclose(R['lr']['360'], 1e-3) and np.isclose(R['lr']['1440'], 1e-3) and np.isclose(R['lr']['160'], 3e-4)
    assert flags(1440) == '--lrs 0.001 --ranks 8 --dropouts 0 --augments none --max-steps 3000', flags(1440)
    R = pick(table([.40] * 4, [.405, .405, .405, .405]))                 # +0.005: below the line, 3e-4 stays
    assert np.isclose(R['lr']['360'], 3e-4) and flags(360).startswith('--lrs 0.0003')
    try: pick(table([.40] * 4, [.43] * 4).query('speaker != "23635"')); raise AssertionError('a missing speaker must refuse')
    except SystemExit: pass
    print('v4_decide self-check: OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', nargs='?', choices=['pick', 'flags', 'base'])
    ap.add_argument('args', nargs='*'); ap.add_argument('--self-check', action='store_true'); ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    if a.self_check: _self_check(); sys.exit(0)
    if a.cmd == 'pick': pick()
    elif a.cmd == 'flags': print(flags(a.args[0]))
    elif a.cmd == 'base': sys.exit(0 if base(batch=a.batch, speakers=[int(x) for x in a.args]) else 1)
    else: ap.error('give a command or --self-check')
