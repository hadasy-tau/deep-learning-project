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

Two additions from docs/training_handoff.md § Before the second run:

  forgiven-shared WER   every cell is also scored under the protocol-aware count
        (error_analysis.forgiven_counts): an inserted word that the OTHER arm
        also produced at that chunk is not charged.  Arm A's transcription of
        each speaker's test set is cached once like arm B's.  `*_f` columns.
  the cross-speaker control (D3)   --control-folds K trains K adapters, each on a
        budget-matched pool drawn from the panel speakers of the other folds,
        and evaluates every speaker on the adapter that never saw them.  K=2 is
        one extra training run over the handoff's single adapter and leaves no
        speaker evaluated on their own audio.  The summary joins it as
        personalization = delta(own adapter) - delta(others' adapter).

    python src/training/run_panel.py --dry-run                                  # list the cells, nothing loaded
    python src/training/run_panel.py --arm B --budgets 5 20 80 --seeds 0 1 2    # the first experiment (the plan's)
    python src/training/run_panel.py --speakers 30831 --budgets 1 --seeds 0 --epochs 1 --model openai/whisper-tiny   # smoke
    python src/training/run_panel.py --summary

Needs the materialized audio (materialize.py extract or download) and a GPU
for anything but the smoke test.
"""
import argparse, glob, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):   # see train.py: 255 BLAS threads made log-mel 1.4 s a chunk
    os.environ.setdefault(_v, '8')
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

from targets import forgiven_score          # the protocol-aware per-chunk count, shared with train.py's checkpoint selection

def _fshares(s):
    tot = max(int(s.werr.sum()), 1)
    return dict(S=int(s.S.sum()) / tot, D=int(s.D.sum()) / tot, I=int(s.I.sum()) / tot, Ish=int(s.Ish.sum()))

def compare(test, hb, ht, ha):
    """Base vs tuned hypotheses on the same chunks: standard WER, then the
    forgiven-shared count against arm A's transcription `ha`.  One dict, the
    forgiven fields suffixed _f."""
    import evaluate as EV
    sb, st = EV.score(test.text, hb), EV.score(test.text, ht)
    pb = EV.paired_bootstrap(sb, st)
    res = dict(**pb, base=_shares(sb), tuned=_shares(st),
               cer_base=float(sb.cerr.sum() / sb.n_chars.sum()), cer_tuned=float(st.cerr.sum() / st.n_chars.sum()))
    if ha is not None:
        fb, ft = forgiven_score(test.text, hb, ha), forgiven_score(test.text, ht, ha)
        pf = EV.paired_bootstrap(fb, ft)
        res.update({k + '_f': v for k, v in pf.items() if k not in ('n_segments', 'n_words')}, base_f=_fshares(fb), tuned_f=_fshares(ft))
    return res

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

OTHER_ARM = {'A': 'B', 'B': 'A'}

def part_hyps(arm, speaker, part, rows, audio_dir, batch, model=None):
    """The base model on a speaker's train or dev chunks, cached per (arm, speaker,
    part) -- the raw material for the semi-verbatim targets (targets.py) and for
    checkpoint selection on dev forgiven WER (train.py).  `model` lets a caller
    keep one loaded model across speakers."""
    import evaluate as EV
    tag = EV.ARMS[arm].replace('/', '__')
    path = os.path.join(RESULTS, f'hyps_{arm}_{speaker}_{part}_{tag}.json')
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        if d['chunk_ids'] == list(rows.chunk_id): return d['hyps']
    own = model is None
    if own: model = EV.load(arm)
    m, proc, device = model
    hyps = EV.transcribe_short(m, proc, rows, audio_dir, batch=batch, device=device)
    if own: del m
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(dict(chunk_ids=list(rows.chunk_id), hyps=hyps), open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    return hyps

def transcribe_parts(P, audio_dir, batch=64, arms=('A', 'B'), parts=('train', 'dev')):
    """Both arms over every speaker's train and dev chunks, one model load per arm."""
    import evaluate as EV
    for arm in arms:
        model = EV.load(arm)
        for sid in sorted(int(s) for s in P.speaker_id.unique()):
            for part in parts:
                rows = P[(P.speaker_id == sid) & (P.part == part)].reset_index(drop=True)
                t = time.time(); part_hyps(arm, sid, part, rows, audio_dir, batch, model)
                print(f'arm {arm} speaker {sid} {part}: {len(rows)} chunks in {time.time()-t:.0f}s', flush=True)
        del model

def _train_meta(adapter):
    p = os.path.join(adapter, 'train_meta.json')
    return json.load(open(p)) if os.path.exists(p) else {}

RECIPE_TAG = {('protocol', 'loss'): '', ('protocol', 'forgiven'): 'selF', ('verbatim', 'loss'): 'verbatim', ('verbatim', 'forgiven'): 'verbatim-selF'}

def apply_targets(P, targets):
    """Swap the train/dev reference text for the semi-verbatim one (targets.py);
    the test text is never touched -- it is what every cell is scored against."""
    if targets == 'protocol': return P
    T = pd.read_parquet(os.path.join(HERE, 'outputs', 'targets_verbatim.parquet')).set_index('chunk_id').text_verbatim
    P = P.copy(); m = P.part.isin(['train', 'dev'])
    P.loc[m, 'text'] = P.loc[m, 'chunk_id'].map(T).fillna(P.loc[m, 'text'])
    return P

def select_kw(P_protocol, dev_rows, select, audio_dir, eval_batch, arm):
    """What train_cell needs for checkpoint selection on dev forgiven WER: the protocol
    dev text (not the training target) and arm A's dev transcription, cached."""
    if select != 'forgiven': return {}
    refs = P_protocol.set_index('chunk_id').loc[dev_rows.chunk_id, 'text'].tolist()
    other = []
    for sid, g in dev_rows.groupby('speaker_id', sort=False):
        d = P_protocol[(P_protocol.speaker_id == sid) & (P_protocol.part == 'dev')].reset_index(drop=True)
        h = dict(zip(d.chunk_id, part_hyps(OTHER_ARM[arm], int(sid), 'dev', d, audio_dir, eval_batch)))
        other += [h[c] for c in g.chunk_id]
    return dict(select='forgiven', select_refs=refs, select_other=other)

def run_cell(c, P, audio_dir, epochs, batch, grad_accum, eval_batch, model_override=None, forgiven=True, targets='protocol', select='loss'):
    import train as T, evaluate as EV
    if model_override:                         # smoke tests: both arms, so the other-arm transcription is tiny too
        for a in T.ARMS: T.ARMS[a] = model_override; EV.ARMS[a] = model_override
    c = {**c, 'lr': c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']]}   # the rate actually used, so the summary can join on it
    tag = RECIPE_TAG[(targets, select)]
    name = T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'], c['seed'], tag)
    out_json = os.path.join(RESULTS, name + '.json')
    if os.path.exists(out_json):
        print(f'skip (scored): {name}'); return json.load(open(out_json, encoding='utf-8'))
    t0 = time.time(); phase = {}
    Pt = apply_targets(P, targets)
    spk = Pt[Pt.speaker_id == c['speaker']]
    skw = select_kw(P, spk[spk.part == 'dev'], select, audio_dir, eval_batch, c['arm'])
    adapter = T.train_cell(spk, audio_dir, RUNS, c['speaker'], arm=c['arm'], site=c['site'], method=c['method'], budget=c['budget'],
                           rank=c['rank'], lr=c['lr'], seed=c['seed'], epochs=epochs, batch=batch, grad_accum=grad_accum, tag=tag, **skw)
    phase['train'] = time.time() - t0
    spk = P[P.speaker_id == c['speaker']]      # scoring: the protocol test text
    test = spk[spk.part == 'test'].reset_index(drop=True)
    t = time.time(); hb = base_hyps(c['arm'], c['speaker'], test, audio_dir, eval_batch, model_override); phase['base'] = time.time() - t
    # arm A's (the other arm's) transcription of the same chunks, for the forgiven-shared count; cached per speaker
    t = time.time(); ha = base_hyps(OTHER_ARM[c['arm']], c['speaker'], test, audio_dir, eval_batch) if forgiven else None; phase['other_arm'] = time.time() - t
    t = time.time(); model, proc, device = EV.load(c['arm'], adapter=adapter); phase['load_tuned'] = time.time() - t
    t = time.time(); ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device); del model; phase['transcribe_tuned'] = time.time() - t
    t = time.time(); cmp = compare(test, hb, ht, ha); phase['score'] = time.time() - t
    tr = T.take_budget(spk[spk.part == 'train'], c['budget']); meta = _train_meta(adapter)
    res = dict(cell=name, **c, targets=targets, select=select, adapter=adapter, train_minutes=float(tr.duration_s.sum() / 60), train_chunks=int(len(tr)),
               train_steps=meta.get('global_step'), train_epochs=meta.get('epochs', epochs), best_eval_loss=meta.get('best_eval_loss'), best_epoch=meta.get('best_epoch'),
               n_test=int(len(test)), test_minutes=float(test.duration_s.sum() / 60), **cmp,
               seconds_train=phase['train'], seconds_total=time.time() - t0, seconds_phase=phase, model=T.ARMS[c['arm']])
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(out_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
    json.dump(dict(chunk_ids=list(test.chunk_id), hyps=ht), open(os.path.join(RESULTS, name + '.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    f = f'; forgiven {cmp["wer_base_f"]:.4f} -> {cmp["wer_tuned_f"]:.4f} ({cmp["delta_rel_f"]:+.1%}, p {cmp["p_boot_f"]:.3f})' if ha is not None else ''
    print(f'{name}: WER {cmp["wer_base"]:.4f} -> {cmp["wer_tuned"]:.4f} ({cmp["delta_rel"]:+.1%}, p {cmp["p_boot"]:.3f}){f} '
          f'in {(time.time()-t0)/60:.1f} min ' + ' '.join(f'{k} {v:.0f}s' for k, v in phase.items()), flush=True)
    return res

# ---- the cross-speaker control (D3) --------------------------------------------------
def pool_budget(P, speakers, budget_min, part='train', seed=0):
    """A budget-matched pool over several speakers: round-robin over each
    speaker's own budget ordering (latest session first, the same prefix the
    personal cells use), one chunk from each in turn, until the next chunk would
    cross the budget.  Every speaker contributes about budget/len(speakers)
    minutes, so no one voice dominates the 'others' adapter."""
    import train as T
    queues = [T.budget_order(P[(P.speaker_id == s) & (P.part == part)]).reset_index(drop=True) for s in speakers]
    rows, total, i = [], 0.0, 0
    while any(i < len(q) for q in queues):
        for q in queues:
            if i < len(q):
                r = q.iloc[i]
                if total + r.duration_s / 60 > budget_min: return pd.DataFrame(rows).reset_index(drop=True)
                rows.append(r); total += r.duration_s / 60
        i += 1
    return pd.DataFrame(rows).reset_index(drop=True)

def control_folds(speakers, k, seed=0):
    """Speakers -> k folds (a fixed shuffle, so the assignment is reproducible)."""
    rng = np.random.default_rng(seed); order = list(rng.permutation(sorted(speakers)))
    return [sorted(int(s) for s in order[i::k]) for i in range(k)]

def run_control(P, audio_dir, epochs, batch, grad_accum, eval_batch, folds=2, budget=80, arm='B', site='both', method='lora',
                rank=8, lr=None, seed=0, dev_min=15, forgiven=True, targets='protocol', select='loss'):
    """k adapters, each trained on a pool from the speakers NOT in its fold and
    evaluated on the fold's speakers.  With k=1 (the handoff's single adapter)
    the pool spans all the speakers and each is evaluated on an adapter that saw
    ~budget/11 minutes of them -- stated in the JSON as `saw_own_minutes`."""
    import train as T, evaluate as EV
    spk_ids = sorted(int(s) for s in P.speaker_id.unique()); fold_of = control_folds(spk_ids, folds, seed)
    lr = lr if lr is not None else T.DEFAULT_LR[method]; out = []; tag = RECIPE_TAG[(targets, select)]; Pt = apply_targets(P, targets)
    for k, evals in enumerate(fold_of):
        trainers = [s for s in spk_ids if s not in evals] if folds > 1 else spk_ids
        label = f'ctrl{k}of{folds}'
        name = T.cell_name(label, arm, site, method, budget, rank, lr, seed, tag)
        todo = [s for s in evals if not os.path.exists(os.path.join(RESULTS, f'{name}__eval{s}.json'))]
        if not todo:
            print(f'skip (scored): {name} on {evals}'); out += [json.load(open(os.path.join(RESULTS, f'{name}__eval{s}.json'), encoding='utf-8')) for s in evals]; continue
        t0 = time.time()
        tr = pool_budget(Pt, trainers, budget, 'train', seed); dev = pool_budget(Pt, trainers, dev_min, 'dev', seed)
        print(f'{name}: pool of {len(trainers)} speakers, {len(tr)} chunks ({tr.duration_s.sum()/60:.1f} min), dev {len(dev)}; evaluates {evals}', flush=True)
        skw = select_kw(P, dev, select, audio_dir, eval_batch, arm)
        adapter = T.train_cell(Pt, audio_dir, RUNS, label, arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed,
                               epochs=epochs, batch=batch, grad_accum=grad_accum, train_rows=tr, dev_rows=dev, tag=tag, **skw)
        t_train = time.time() - t0; meta = _train_meta(adapter)
        per_spk_min = tr.groupby('speaker_id').duration_s.sum().div(60).to_dict()
        model, proc, device = EV.load(arm, adapter=adapter)
        for s in todo:
            t1 = time.time(); test = P[(P.speaker_id == s) & (P.part == 'test')].reset_index(drop=True)
            hb = base_hyps(arm, s, test, audio_dir, eval_batch); ha = base_hyps(OTHER_ARM[arm], s, test, audio_dir, eval_batch) if forgiven else None
            ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device)
            cmp = compare(test, hb, ht, ha)
            res = dict(cell=f'{name}__eval{s}', speaker=s, arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed,
                       targets=targets, select=select, control=True, fold=k, folds=folds, trained_on=trainers, saw_own_minutes=float(per_spk_min.get(s, 0.0)),
                       adapter=adapter, train_minutes=float(tr.duration_s.sum() / 60), train_chunks=int(len(tr)), train_steps=meta.get('global_step'),
                       train_epochs=meta.get('epochs', epochs), best_eval_loss=meta.get('best_eval_loss'),
                       n_test=int(len(test)), test_minutes=float(test.duration_s.sum() / 60), **cmp,
                       seconds_train=t_train, seconds_eval=time.time() - t1, model=T.ARMS[arm])
            json.dump(res, open(os.path.join(RESULTS, f'{name}__eval{s}.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
            json.dump(dict(chunk_ids=list(test.chunk_id), hyps=ht), open(os.path.join(RESULTS, f'{name}__eval{s}.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
            f = f'; forgiven {cmp["wer_base_f"]:.4f} -> {cmp["wer_tuned_f"]:.4f} ({cmp["delta_rel_f"]:+.1%})' if ha is not None else ''
            print(f'{name} on {s}: WER {cmp["wer_base"]:.4f} -> {cmp["wer_tuned"]:.4f} ({cmp["delta_rel"]:+.1%}, p {cmp["p_boot"]:.3f}){f} in {time.time()-t1:.0f}s', flush=True)
            out.append(res)
        del model
    return out

def summary(out=os.path.join(HERE, 'outputs', 'results.csv')):
    rows = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(RESULTS, '*.json'))) if not f.endswith('.hyps.json') and not os.path.basename(f).startswith('base_')]
    if not rows: print('no results yet'); return None
    R = pd.json_normalize(rows)
    if 'control' not in R: R['control'] = False
    R['control'] = R.control.fillna(False).astype(bool)
    for col, default in (('targets', 'protocol'), ('select', 'loss')):
        if col not in R: R[col] = default
        R[col] = R[col].fillna(default)
    # the acceptance rule: where did the improvement come from?
    R['improvement_from_insertions'] = ((R['base.I'] * R.wer_base - R['tuned.I'] * R.wer_tuned) / (R.wer_base - R.wer_tuned).replace(0, np.nan)).clip(-5, 5)
    R['style_not_speaker'] = R.improvement_from_insertions > 0.5
    # D3: personalization = delta(own adapter) - delta(others' adapter), per speaker, on the same test chunks.
    # The control is trained at one budget (80 min); every personal budget is compared against it, so only
    # the 80-minute row is budget-matched -- the smaller budgets read as 'less audio of the speaker versus
    # 80 minutes of everyone else'.
    ctrl = R[R.control]
    if len(ctrl):
        keys = ['speaker', 'arm', 'site', 'method', 'rank', 'lr', 'seed', 'targets', 'select']
        keep = keys + ['delta_abs', 'delta_rel', 'wer_tuned', 'ci_lo', 'ci_hi', 'p_boot'] + [c for c in ('delta_abs_f', 'delta_rel_f', 'wer_tuned_f') if c in ctrl]
        c = ctrl[keep].rename(columns={k: k.replace('delta', 'control_delta').replace('wer_tuned', 'wer_control').replace('ci_', 'control_ci_').replace('p_boot', 'control_p_boot') for k in keep[len(keys):]})
        R = R.merge(c, on=keys, how='left')
        R['personalization_abs'] = R.delta_abs - R.control_delta_abs
        R['personalization_rel'] = R.personalization_abs / R.wer_base
        if 'delta_abs_f' in R: R['personalization_abs_f'] = R.delta_abs_f - R.control_delta_abs_f
    R.to_csv(out, index=False, float_format='%.5g'); print(f'{len(R)} rows ({int(R.control.sum())} control evaluations) -> {out}')
    cols = [c for c in ['speaker', 'site', 'lr', 'targets', 'select', 'budget', 'seed', 'control', 'n_test', 'train_steps', 'wer_base', 'wer_tuned', 'delta_rel', 'ci_lo', 'ci_hi', 'p_boot', 'wer_base_f', 'wer_tuned_f', 'delta_rel_f', 'p_boot_f',
                        'improvement_from_insertions', 'style_not_speaker', 'control_delta_rel', 'personalization_rel'] if c in R]
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
    ap.add_argument('--no-forgiven', action='store_true', help='skip the arm-A transcription and the forgiven-shared count')
    ap.add_argument('--control-folds', type=int, default=0, help='D3: train the cross-speaker control in K folds after the cells (0 = none)')
    ap.add_argument('--control-budget', type=int, default=80); ap.add_argument('--control-only', action='store_true')
    ap.add_argument('--transcribe-parts', action='store_true', help='both arms over every speaker\'s train and dev chunks (cached), then exit')
    ap.add_argument('--build-targets', action='store_true', help='write outputs/targets_verbatim.parquet from the cached train/dev transcriptions, then exit')
    ap.add_argument('--targets', choices=['protocol', 'verbatim'], default='protocol', help='training target text (docs/training_next.md A1)')
    ap.add_argument('--select', choices=['loss', 'forgiven'], default='loss', help='checkpoint selection: dev loss, or dev forgiven-shared WER (A2)')
    a = ap.parse_args()
    if a.summary: summary(); sys.exit(0)
    P = load_plan(speakers=a.speakers)
    if a.transcribe_parts: transcribe_parts(P, a.audio, a.eval_batch); sys.exit(0)
    if a.build_targets:
        import targets as TG
        T = TG.build_targets(P, RESULTS); TG.report(T); T.to_parquet(os.path.join(HERE, 'outputs', 'targets_verbatim.parquet'), index=False); sys.exit(0)
    C = [] if a.control_only else list(cells(P, a.arms, a.sites, a.methods, a.budgets, a.ranks, a.lrs, a.seeds))
    print(f'{len(C)} cells over {P.speaker_id.nunique()} speakers; test {P[P.part=="test"].duration_s.sum()/60:.0f} min, train {P[P.part=="train"].duration_s.sum()/60:.0f} min available'
          + (f'; then the cross-speaker control in {a.control_folds} fold(s) at {a.control_budget} min' if a.control_folds else ''))
    if a.dry_run:
        import train as T
        for c in C: print('  ' + T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']], c['seed'], RECIPE_TAG[(a.targets, a.select)]))
        if a.control_folds:
            for k, f in enumerate(control_folds(sorted(int(s) for s in P.speaker_id.unique()), a.control_folds, a.seeds[0])): print(f'  ctrl{k}of{a.control_folds}: evaluates {f}')
        sys.exit(0)
    for c in C: run_cell(c, P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, a.model, forgiven=not a.no_forgiven, targets=a.targets, select=a.select)
    if a.control_folds:                        # one control per recipe, so every sweep point has a budget-matched 'others' adapter
        for site in a.sites:
            for lr in a.lrs:
                run_control(P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, folds=a.control_folds, budget=a.control_budget, arm=a.arms[0], site=site,
                            method=a.methods[0], rank=a.ranks[0], lr=lr, seed=a.seeds[0], forgiven=not a.no_forgiven, targets=a.targets, select=a.select)
    summary()
