"""Drive the panel through train.py and evaluate.py: cells in, scored results out.

One cell = (speaker, arm, site, method, budget, rank, lr, seed), D4's axes.  For
each cell this trains the adapter (train.train_cell, idempotent), transcribes
the speaker's personal-test chunks with the base model and with the tuned one
(evaluate.transcribe_short, D7's primary protocol), scores both (evaluate.score:
counts, S/D/I, runaway), runs the paired bootstrap, and writes one JSON per
cell under outputs/results/.  The base model's transcription of a speaker's
test set is cached once per (arm, speaker) -- it is the same for every cell.
`--summary` collects the JSONs into outputs/results.csv.

The acceptance rule from docs/adaptation_plan.md is applied in the summary,
not hidden: a gain that comes mostly from fewer insertions (share of the
improvement that is insertions > 0.5) is flagged `style_not_speaker`.

    python src/training/run_panel.py --dry-run                                  # list the cells, nothing loaded
    python src/training/run_panel.py --arm B --budgets 5 20 80 --seeds 0 1 2    # the first experiment (the plan's)
    python src/training/run_panel.py --speakers 30831 --budgets 1 --seeds 0 --epochs 1 --model openai/whisper-tiny   # smoke
    python src/training/run_panel.py --summary

Needs the materialized audio (materialize.py extract or download) and a GPU
for anything but the smoke test.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(ROOT, 'src', 'evaluation'))

PLAN, AUDIO = os.path.join(HERE, 'panel_plan.parquet'), os.path.join(HERE, 'outputs', 'panel_audio')
RUNS, RESULTS = os.path.join(HERE, 'runs'), os.path.join(HERE, 'outputs', 'results')

def load_plan(plan_path=PLAN, speakers=None):
    P = pd.read_parquet(plan_path)
    if speakers: P = P[P.speaker_id.isin(speakers)]
    return P.reset_index(drop=True)

def cells(P, arms, sites, methods, budgets, ranks, lrs, seeds):
    for sid in sorted(P.speaker_id.unique()):
        for arm in arms:
            for site in sites:
                for method in methods:
                    for budget in budgets:
                        for rank in ranks:
                            for lr in lrs:
                                for seed in seeds:
                                    yield dict(speaker=int(sid), arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed)

def _shares(s):
    tot = max(int(s.werr.sum()), 1)
    return dict(S=int(s.S.sum()) / tot, D=int(s.D.sum()) / tot, I=int(s.I.sum()) / tot, runaway=int(s.runaway.sum()))

def base_hyps(arm, speaker, test, audio_dir, batch, model_override=None):
    """The base model on the speaker's test chunks, cached per (arm, speaker)."""
    import evaluate as EV
    tag = (model_override or EV.ARMS[arm]).replace('/', '__')
    path = os.path.join(RESULTS, f'base_{arm}_{speaker}_{tag}.json')
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        if d['chunk_ids'] == list(test.chunk_id): return d['hyps']
    model, proc, device = EV.load(arm)
    hyps = EV.transcribe_short(model, proc, test, audio_dir, batch=batch, device=device)
    del model
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(dict(chunk_ids=list(test.chunk_id), hyps=hyps), open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    return hyps

def run_cell(c, P, audio_dir, epochs, batch, grad_accum, eval_batch, model_override=None):
    import train as T, evaluate as EV
    if model_override:
        T.ARMS[c['arm']] = model_override; EV.ARMS[c['arm']] = model_override
    name = T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']], c['seed'])
    out_json = os.path.join(RESULTS, name + '.json')
    if os.path.exists(out_json):
        print(f'skip (scored): {name}'); return json.load(open(out_json, encoding='utf-8'))
    t0 = time.time()
    spk = P[P.speaker_id == c['speaker']]
    adapter = T.train_cell(spk, audio_dir, RUNS, c['speaker'], arm=c['arm'], site=c['site'], method=c['method'], budget=c['budget'],
                           rank=c['rank'], lr=c['lr'], seed=c['seed'], epochs=epochs, batch=batch, grad_accum=grad_accum)
    t_train = time.time() - t0
    test = spk[spk.part == 'test'].reset_index(drop=True)
    hb = base_hyps(c['arm'], c['speaker'], test, audio_dir, eval_batch, model_override)
    model, proc, device = EV.load(c['arm'], adapter=adapter)
    ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device); del model
    sb, st = EV.score(test.text, hb), EV.score(test.text, ht)
    pb = EV.paired_bootstrap(sb, st)
    tr = T.take_budget(spk[spk.part == 'train'], c['budget'])
    res = dict(cell=name, **c, adapter=adapter, train_minutes=float(tr.duration_s.sum() / 60), train_chunks=int(len(tr)),
               n_test=int(len(test)), test_minutes=float(test.duration_s.sum() / 60), **pb,
               base=_shares(sb), tuned=_shares(st), cer_base=float(sb.cerr.sum() / sb.n_chars.sum()), cer_tuned=float(st.cerr.sum() / st.n_chars.sum()),
               seconds_train=t_train, seconds_total=time.time() - t0, model=T.ARMS[c['arm']])
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(out_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
    json.dump(dict(chunk_ids=list(test.chunk_id), hyps=ht), open(os.path.join(RESULTS, name + '.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{name}: WER {pb["wer_base"]:.4f} -> {pb["wer_tuned"]:.4f} ({pb["delta_rel"]:+.1%}, p {pb["p_boot"]:.3f}) in {(time.time()-t0)/60:.1f} min')
    return res

def summary(out=os.path.join(HERE, 'outputs', 'results.csv')):
    rows = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(RESULTS, '*.json'))) if not f.endswith('.hyps.json') and not os.path.basename(f).startswith('base_')]
    if not rows: print('no results yet'); return None
    R = pd.json_normalize(rows)
    # the acceptance rule: where did the improvement come from?
    R['improvement_from_insertions'] = ((R['base.I'] * R.wer_base - R['tuned.I'] * R.wer_tuned) / (R.wer_base - R.wer_tuned).replace(0, np.nan)).clip(-5, 5)
    R['style_not_speaker'] = R.improvement_from_insertions > 0.5
    R.to_csv(out, index=False, float_format='%.5g'); print(f'{len(R)} cells -> {out}')
    cols = ['speaker', 'arm', 'site', 'method', 'budget', 'seed', 'n_test', 'wer_base', 'wer_tuned', 'delta_rel', 'ci_lo', 'ci_hi', 'p_boot', 'improvement_from_insertions', 'style_not_speaker']
    print(R[cols].round(4).to_string(index=False)); return R

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--speakers', nargs='+', type=int); ap.add_argument('--arm', nargs='+', default=['B'], dest='arms')
    ap.add_argument('--sites', nargs='+', default=['both']); ap.add_argument('--methods', nargs='+', default=['lora'])
    ap.add_argument('--budgets', nargs='+', type=int, default=[5, 20, 80]); ap.add_argument('--ranks', nargs='+', type=int, default=[8])
    ap.add_argument('--lrs', nargs='+', type=float, default=[None]); ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--eval-batch', type=int, default=8); ap.add_argument('--model', help='override the arm\'s checkpoint (smoke tests only)')
    ap.add_argument('--audio', default=AUDIO); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.summary: summary(); sys.exit(0)
    P = load_plan(speakers=a.speakers)
    C = list(cells(P, a.arms, a.sites, a.methods, a.budgets, a.ranks, a.lrs, a.seeds))
    print(f'{len(C)} cells over {P.speaker_id.nunique()} speakers; test {P[P.part=="test"].duration_s.sum()/60:.0f} min, train {P[P.part=="train"].duration_s.sum()/60:.0f} min available')
    if a.dry_run:
        import train as T
        for c in C: print('  ' + T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']], c['seed']))
        sys.exit(0)
    for c in C: run_cell(c, P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, a.model)
    summary()
