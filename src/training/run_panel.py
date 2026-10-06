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

The forgiven-shared count (an inserted word the other arm also produced is not
charged; `*_f` columns, --forgiven, --select forgiven) is removed: it assumed that two
models agreeing against the protocol means the protocol is wrong, an assumption too strong
to rest a result on (docs/STATUS.md § Withdrawn).  The second
and third runs' result files still carry `*_f` columns; the summary drops them.

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

Step mode and tuning (docs/training_plan_v3.md).  --max-steps switches train.py to a
fixed number of optimiser steps for every budget, validation every 20 steps and
early stopping after --patience validations without improvement (4 by default);
--dropouts and --augments are swept like --lrs and --ranks; --plan
panel_plan_v2.parquet trains and tests on the high-quality data.  --tune trains and
validates only -- the test set is never transcribed -- and --tuning-report
collects the validation results into outputs/tuning.csv.

    python src/training/run_panel.py --tune --plan src/training/panel_plan_v2.parquet --max-steps 400 \
        --speakers 30685 23558 30718 30859 --budgets 5 80 --lrs 1e-4 3e-4 1e-3 --seeds 0
    python src/training/run_panel.py --tuning-report

The second test set (docs/training_plan_v3.md § 4).  --test07-plan panel_test07.parquet
also scores every cell and every control evaluation on plan v2's test sessions at
quality >= 0.7 (materialize.py plan-test07): `test07.*` columns beside the high-quality
ones, split by quality band (`test07.bands.q070` ... `q095`).  The high-quality test is a
subset of it, so its hypotheses are reused and only the rest is transcribed.  A cell
scored before the flag was given gets its test07 scores on the next run, from its
saved adapter.

Run 4 (docs/training_plan_v4.md).  --cell-speakers trains own cells for those speakers
only, while the plan keeps every speaker for the control's pool.  The control keeps run
3's definition -- control_folds over the panel's 12 speakers, at the seed -- and
--control-extra names speakers outside the panel (23641, 23635): they never enter the
folds or a pool, they are attached to the fold of --control-attach (23558), and that
fold's adapter, with their test and dev sessions also kept out of its pool, is labelled
`ctrl<k>of<K>x`.  --control-evals limits which speakers a control is evaluated on (and
so which folds are trained at all).  --curve writes outputs/curve_v4.csv: per speaker,
personalization at the top budget minus at 80 minutes, with a paired bootstrap over test
chunks shared by every seed, and the plan's decision rule.

    python src/training/run_panel.py --plan src/training/panel_plan_v4.parquet --max-steps 1200 \
        --cell-speakers 23641 --budgets 360 --seeds 0 --lrs 3e-4 --dropouts 0 --test07-plan src/training/panel_test07_v4.parquet
    python src/training/run_panel.py --plan src/training/panel_plan_v4.parquet --max-steps 1200 --control-only \
        --control-folds 2 --control-budgets 360 --seeds 0 --lrs 3e-4 --dropouts 0 \
        --control-evals 23558 23641 23635 --control-extra 23641 23635 --control-attach 23558 \
        --test07-plan src/training/panel_test07_v4.parquet
    python src/training/run_panel.py --curve
    python src/training/run_panel.py --self-check

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
TUNE_RUNS, TUNING = os.path.join(HERE, 'runs_tune'), os.path.join(HERE, 'outputs', 'tuning.csv')
PLAN_TEST07 = os.path.join(HERE, 'panel_test07.parquet')

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

def compare(test, hb, ht):
    """Base vs tuned hypotheses on the same chunks: WER and CER, the S/D/I shares,
    the paired bootstrap."""
    import evaluate as EV
    sb, st = EV.score(test.text, hb), EV.score(test.text, ht)
    pb = EV.paired_bootstrap(sb, st)
    return dict(**pb, base=_shares(sb), tuned=_shares(st),
                cer_base=float(sb.cerr.sum() / sb.n_chars.sum()), cer_tuned=float(st.cerr.sum() / st.n_chars.sum()))

QBANDS = ((0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01))

def compare_bands(test, hb, ht):
    """compare() within each alignment-quality band of the >= 0.7 test set: where the
    adapter helps, on the clean clips or on the hard ones the high-quality filter drops."""
    import evaluate as EV
    sb, st = EV.score(test.text, hb), EV.score(test.text, ht)
    out = {}
    for lo, hi in QBANDS:
        m = ((test.quality >= lo) & (test.quality < hi)).values
        if not m.any(): continue
        pb = EV.paired_bootstrap(sb[m], st[m])
        out[f'q{round(lo * 100):03d}'] = dict(minutes=float(test.duration_s[m].sum() / 60), n_segments=int(m.sum()),
                                             **{k: pb[k] for k in ('wer_base', 'wer_tuned', 'delta_abs', 'delta_rel', 'ci_lo', 'ci_hi', 'p_boot')})
    return out

def score_test07(T07, speaker, arm, test, ht, model, proc, device, audio_dir, batch, name):
    """One adapter on the speaker's >= 0.7 test set.  `test`/`ht` are its high-quality
    test chunks and hypotheses: they are a subset, so only the rest is transcribed.
    Writes <name>.test07.hyps.json; returns the dict stored under the result's `test07`."""
    import evaluate as EV
    t07 = T07[T07.speaker_id == speaker].reset_index(drop=True)
    known = dict(zip(test.chunk_id, ht))
    rest = t07[~t07.chunk_id.isin(known)].reset_index(drop=True)
    known.update(zip(rest.chunk_id, EV.transcribe_short(model, proc, rest, audio_dir, batch=batch, device=device)))
    ht7 = [known[c] for c in t07.chunk_id]
    hb7 = base_hyps(arm, speaker, t07, audio_dir, batch, data='test07')
    json.dump(dict(chunk_ids=list(t07.chunk_id), hyps=ht7), open(os.path.join(RESULTS, name + '.test07.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    res = dict(n_test=int(len(t07)), test_minutes=float(t07.duration_s.sum() / 60), hq_share=float(t07.duration_s[t07.hq].sum() / t07.duration_s.sum()),
               **compare(t07, hb7, ht7), bands=compare_bands(t07, hb7, ht7))
    print(f'{name} on the >= 0.7 test: WER {res["wer_base"]:.4f} -> {res["wer_tuned"]:.4f} ({res["delta_rel"]:+.1%}, p {res["p_boot"]:.3f}); '
          + ' '.join(f'{k} {v["delta_rel"]:+.1%}' for k, v in res['bands'].items()), flush=True)
    return res

def add_test07(out_json, T07, arm, adapter, audio_dir, batch):
    """A result scored before --test07-plan was given: add its test07 scores from the
    saved adapter and its saved high-quality hypotheses."""
    import evaluate as EV
    res = json.load(open(out_json, encoding='utf-8'))
    if 'test07' in res or not os.path.exists(os.path.join(adapter, 'adapter_config.json')): return res
    name = os.path.basename(out_json)[:-len('.json')]
    h = json.load(open(os.path.join(RESULTS, name + '.hyps.json'), encoding='utf-8'))
    test = pd.DataFrame(dict(chunk_id=h['chunk_ids']))
    model, proc, device = EV.load(arm, adapter=adapter)
    res['test07'] = score_test07(T07, int(res['speaker']), arm, test, h['hyps'], model, proc, device, audio_dir, batch, name); del model
    json.dump(res, open(out_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
    return res

def base_hyps(arm, speaker, test, audio_dir, batch, model_override=None, data=''):
    """The base model on the speaker's test chunks, cached per (arm, speaker, plan):
    `data` names a plan other than v1 ('hq'), so its different test set gets its own
    cache instead of overwriting the second run's."""
    import evaluate as EV
    tag = (model_override or EV.ARMS[arm]).replace('/', '__')
    path = os.path.join(RESULTS, f'base_{arm}_{speaker}_{tag}' + (f'_{data}' if data else '') + '.json')
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        # a cache decoded differently (the second run's, before the loop guard) is stale:
        # base and tuned must go through the same decoding
        if d['chunk_ids'] == list(test.chunk_id) and d.get('decode') == EV.DECODE: return d['hyps']
    model, proc, device = EV.load(arm)
    hyps = EV.transcribe_short(model, proc, test, audio_dir, batch=batch, device=device)
    del model
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(dict(chunk_ids=list(test.chunk_id), hyps=hyps, decode=EV.DECODE), open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    return hyps

OTHER_ARM = {'A': 'B', 'B': 'A'}

def part_hyps(arm, speaker, part, rows, audio_dir, batch, model=None):
    """The base model on a speaker's train or dev chunks, cached per (arm, speaker,
    part) -- the raw material for the semi-verbatim targets (targets.py).  `model`
    lets a caller keep one loaded model across speakers."""
    import evaluate as EV
    tag = EV.ARMS[arm].replace('/', '__')
    path = os.path.join(RESULTS, f'hyps_{arm}_{speaker}_{part}_{tag}.json')
    if os.path.exists(path):
        d = json.load(open(path, encoding='utf-8'))
        if d['chunk_ids'] == list(rows.chunk_id) and d.get('decode') == EV.DECODE: return d['hyps']
    own = model is None
    if own: model = EV.load(arm)
    m, proc, device = model
    hyps = EV.transcribe_short(m, proc, rows, audio_dir, batch=batch, device=device)
    if own: del m
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(dict(chunk_ids=list(rows.chunk_id), hyps=hyps, decode=EV.DECODE), open(path, 'w', encoding='utf-8'), ensure_ascii=False)
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

# ('…', 'forgiven') named the second run's cells selected on dev forgiven WER ('selF'); that selection is removed
RECIPE_TAG = {('protocol', 'loss'): '', ('protocol', 'forgiven'): 'selF', ('verbatim', 'loss'): 'verbatim', ('verbatim', 'forgiven'): 'verbatim-selF'}

def full_tag(targets, select, step):
    """The cell-name suffix: the target/selection recipe, then the step-mode
    settings (train.recipe_tag); empty for the second run's reference recipe."""
    import train as T
    return '_'.join(t for t in (RECIPE_TAG[(targets, select)], T.recipe_tag(**step)) if t)

def step_kw(step):
    """train_cell's keywords from the step-mode settings (data is only a name tag)."""
    return {k: v for k, v in step.items() if k != 'data'}

def apply_targets(P, targets):
    """Swap the train/dev reference text for the semi-verbatim one (targets.py);
    the test text is never touched -- it is what every cell is scored against."""
    if targets == 'protocol': return P
    T = pd.read_parquet(os.path.join(HERE, 'outputs', 'targets_verbatim.parquet')).set_index('chunk_id').text_verbatim
    P = P.copy(); m = P.part.isin(['train', 'dev'])
    P.loc[m, 'text'] = P.loc[m, 'chunk_id'].map(T).fillna(P.loc[m, 'text'])
    return P

def run_cell(c, P, audio_dir, epochs, batch, grad_accum, eval_batch, model_override=None, targets='protocol', select='loss', step=None, T07=None):
    import train as T, evaluate as EV
    if model_override:                         # smoke tests
        for a in T.ARMS: T.ARMS[a] = model_override; EV.ARMS[a] = model_override
    c = {**c, 'lr': c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']]}   # the rate actually used, so the summary can join on it
    step = step or {}
    tag = full_tag(targets, select, step)
    name = T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'], c['seed'], tag)
    out_json = os.path.join(RESULTS, name + '.json')
    if os.path.exists(out_json):
        print(f'skip (scored): {name}')
        if T07 is not None: return add_test07(out_json, T07, c['arm'], os.path.join(RUNS, name), audio_dir, eval_batch)
        return json.load(open(out_json, encoding='utf-8'))
    t0 = time.time(); phase = {}
    Pt = apply_targets(P, targets)
    spk = Pt[Pt.speaker_id == c['speaker']]
    adapter = T.train_cell(spk, audio_dir, RUNS, c['speaker'], arm=c['arm'], site=c['site'], method=c['method'], budget=c['budget'],
                           rank=c['rank'], lr=c['lr'], seed=c['seed'], epochs=epochs, batch=batch, grad_accum=grad_accum, tag=tag, **step_kw(step))
    phase['train'] = time.time() - t0
    spk = P[P.speaker_id == c['speaker']]      # scoring: the protocol test text
    test = spk[spk.part == 'test'].reset_index(drop=True)
    data = step.get('data', '')
    t = time.time(); hb = base_hyps(c['arm'], c['speaker'], test, audio_dir, eval_batch, model_override, data); phase['base'] = time.time() - t
    t = time.time(); model, proc, device = EV.load(c['arm'], adapter=adapter); phase['load_tuned'] = time.time() - t
    t = time.time(); ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device); phase['transcribe_tuned'] = time.time() - t
    t = time.time(); cmp = compare(test, hb, ht); phase['score'] = time.time() - t
    if T07 is not None:
        t = time.time(); cmp['test07'] = score_test07(T07, c['speaker'], c['arm'], test, ht, model, proc, device, audio_dir, eval_batch, name); phase['test07'] = time.time() - t
    del model
    tr = T.take_budget(spk[spk.part == 'train'], c['budget']); meta = _train_meta(adapter)
    res = dict(cell=name, **c, targets=targets, select=select, adapter=adapter, train_minutes=float(tr.duration_s.sum() / 60), train_chunks=int(len(tr)),
               train_steps=meta.get('global_step'), train_epochs=meta.get('epochs', epochs), best_eval_loss=meta.get('best_eval_loss'), best_epoch=meta.get('best_epoch'),
               n_test=int(len(test)), test_minutes=float(test.duration_s.sum() / 60), **cmp,
               seconds_train=phase['train'], seconds_total=time.time() - t0, seconds_phase=phase, model=T.ARMS[c['arm']], decode=EV.DECODE)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(out_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
    json.dump(dict(chunk_ids=list(test.chunk_id), hyps=ht), open(os.path.join(RESULTS, name + '.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{name}: WER {cmp["wer_base"]:.4f} -> {cmp["wer_tuned"]:.4f} ({cmp["delta_rel"]:+.1%}, p {cmp["p_boot"]:.3f}) '
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

def fold_plan(spk_ids, folds, seed, extra=(), attach=None, evals_only=None):
    """The control's folds: [(k, members, attached, evals, trainers, label)].  The folds
    are control_folds over spk_ids minus `extra` -- run 3's assignment when spk_ids are
    the panel.  `extra` speakers join the fold holding `attach`; that fold is labelled
    with an `x` because its pool also leaves out their sessions.  evals_only drops every
    evaluation not in it, and a fold left with none."""
    extra = [int(s) for s in extra]; universe = sorted(int(s) for s in spk_ids if int(s) not in extra)
    assert not extra or attach in universe, f'--control-attach {attach} must be a panel speaker'
    out = []
    for k, members in enumerate(control_folds(universe, folds, seed)):
        attached = extra if (extra and attach in members) else []
        evals = [s for s in members + attached if evals_only is None or s in evals_only]
        if not evals: continue
        trainers = [s for s in universe if s not in members] if folds > 1 else universe
        out.append((k, members, attached, evals, trainers, f'ctrl{k}of{folds}' + ('x' if attached else '')))
    return out

def run_control(P, audio_dir, epochs, batch, grad_accum, eval_batch, folds=2, budget=80, arm='B', site='both', method='lora',
                rank=8, lr=None, seed=0, dev_min=15, targets='protocol', select='loss', step=None, T07=None,
                extra=(), attach=None, evals_only=None):
    """k adapters, each trained on a pool from the speakers NOT in its fold and
    evaluated on the fold's speakers.  With k=1 (the handoff's single adapter)
    the pool spans all the speakers and each is evaluated on an adapter that saw
    ~budget/11 minutes of them -- stated in the JSON as `saw_own_minutes`.
    extra / attach / evals_only: run 4's speakers outside the panel (fold_plan)."""
    import train as T, evaluate as EV
    step = step or {}
    lr = lr if lr is not None else T.DEFAULT_LR[method]; out = []; tag = full_tag(targets, select, step); Pt = apply_targets(P, targets)
    for k, members, attached, evals, trainers, label in fold_plan(P.speaker_id.unique(), folds, seed, extra, attach, evals_only):
        name = T.cell_name(label, arm, site, method, budget, rank, lr, seed, tag)
        todo = [s for s in evals if not os.path.exists(os.path.join(RESULTS, f'{name}__eval{s}.json'))]
        if T07 is not None:                    # scored before --test07-plan: add it from the saved adapter
            for s in evals:
                if s not in todo: add_test07(os.path.join(RESULTS, f'{name}__eval{s}.json'), T07, arm, os.path.join(RUNS, name), audio_dir, eval_batch)
        if not todo:
            print(f'skip (scored): {name} on {evals}'); out += [json.load(open(os.path.join(RESULTS, f'{name}__eval{s}.json'), encoding='utf-8')) for s in evals]; continue
        t0 = time.time()
        # Committee meetings have several panel members in them: a trainer's session can be the very meeting an
        # evaluated speaker is tested on (same topic, names, room).  Keep every session of the evaluated speakers'
        # test and dev out of the control's pool, or the control is scored partly on audio it trained on.
        held = set(P[P.speaker_id.isin(members + attached) & P.part.isin(['test', 'dev'])].session_id)
        if T07 is not None: held |= set(T07[T07.speaker_id.isin(members + attached)].session_id)
        Pc = Pt[~Pt.session_id.isin(held)]
        tr = pool_budget(Pc, trainers, budget, 'train', seed); dev = pool_budget(Pc, trainers, dev_min, 'dev', seed)
        assert not set(tr.session_id) & held and not set(dev.session_id) & held
        print(f'{name}: pool of {len(trainers)} speakers, {len(tr)} chunks ({tr.duration_s.sum()/60:.1f} min), dev {len(dev)}; evaluates {evals}', flush=True)
        adapter = T.train_cell(Pt, audio_dir, RUNS, label, arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed,
                               epochs=epochs, batch=batch, grad_accum=grad_accum, train_rows=tr, dev_rows=dev, tag=tag, **step_kw(step))
        t_train = time.time() - t0; meta = _train_meta(adapter)
        per_spk_min = tr.groupby('speaker_id').duration_s.sum().div(60).to_dict()
        model, proc, device = EV.load(arm, adapter=adapter)
        for s in todo:
            t1 = time.time(); test = P[(P.speaker_id == s) & (P.part == 'test')].reset_index(drop=True)
            data = step.get('data', '')
            hb = base_hyps(arm, s, test, audio_dir, eval_batch, data=data)
            ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device)
            cmp = compare(test, hb, ht)
            if T07 is not None: cmp['test07'] = score_test07(T07, s, arm, test, ht, model, proc, device, audio_dir, eval_batch, f'{name}__eval{s}')
            res = dict(cell=f'{name}__eval{s}', speaker=s, arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed,
                       targets=targets, select=select, control=True, fold=k, folds=folds, trained_on=trainers, saw_own_minutes=float(per_spk_min.get(s, 0.0)),
                       adapter=adapter, train_minutes=float(tr.duration_s.sum() / 60), train_chunks=int(len(tr)), train_steps=meta.get('global_step'),
                       train_epochs=meta.get('epochs', epochs), best_eval_loss=meta.get('best_eval_loss'),
                       n_test=int(len(test)), test_minutes=float(test.duration_s.sum() / 60), **cmp,
                       seconds_train=t_train, seconds_eval=time.time() - t1, model=T.ARMS[arm], decode=EV.DECODE)
            json.dump(res, open(os.path.join(RESULTS, f'{name}__eval{s}.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
            json.dump(dict(chunk_ids=list(test.chunk_id), hyps=ht), open(os.path.join(RESULTS, f'{name}__eval{s}.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
            print(f'{name} on {s}: WER {cmp["wer_base"]:.4f} -> {cmp["wer_tuned"]:.4f} ({cmp["delta_rel"]:+.1%}, p {cmp["p_boot"]:.3f}) in {time.time()-t1:.0f}s', flush=True)
            out.append(res)
        del model
    return out

# ---- tuning: validation only ---------------------------------------------------------
def run_tune(c, P, audio_dir, batch, grad_accum, step, model_override=None):
    """Train one cell with the recipe's early stopping and keep only its validation
    curve (train_meta.json under runs_tune/).  The test set is not read."""
    import train as T
    if model_override:
        for a in T.ARMS: T.ARMS[a] = model_override
    c = {**c, 'lr': c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']]}
    spk = P[P.speaker_id == c['speaker']]
    return T.train_cell(spk, audio_dir, TUNE_RUNS, c['speaker'], arm=c['arm'], site=c['site'], method=c['method'], budget=c['budget'],
                        rank=c['rank'], lr=c['lr'], seed=c['seed'], batch=batch, grad_accum=grad_accum, tag=full_tag('protocol', 'loss', step),
                        keep_adapter=False, **step_kw(step))

def tuning_report(out=TUNING, runs=TUNE_RUNS):
    """One row per tuning run: the settings, the untuned and the best validation
    loss, and rel_drop = the relative drop between them -- the tuning criterion
    (docs/training_plan_v3.md, step 3).  Prints the mean over speakers per recipe
    and budget, with how many speakers each recipe helped."""
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, '*', 'train_meta.json'))):
        m = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
        if not m.get('max_steps') or m.get('base_eval_loss') is None or not m.get('evals'): continue
        head = name.split('_')                        # s<speaker>_arm<A>_<site>_<method>_b<budget>_r<rank>_lr<lr>_seed<seed>_<tag>
        if not head[0][1:].isdigit(): continue        # a control's adapter (runs/ holds them beside the own cells)
        best = min(m['evals'], key=lambda e: e['eval_loss'])
        rows.append(dict(run=name, speaker=head[0][1:], budget=int(head[4][1:]), lr=float(head[6][2:]), rank=int(head[5][1:]),
                         dropout=m.get('dropout') or 0.0, augment=m.get('augment', 'none'), patience=m.get('patience'),
                         base_loss=m['base_eval_loss'], best_loss=best['eval_loss'], rel_drop=1 - best['eval_loss'] / m['base_eval_loss'],
                         best_step=best.get('step'), stop_step=m.get('global_step'), train_minutes=m.get('train_minutes'),
                         seed=int(head[7][4:]), max_steps=m.get('max_steps')))
    if not rows: print('no tuning runs yet'); return None
    T = pd.DataFrame(rows); T.to_csv(out, index=False, float_format='%.5g')
    S = (T.groupby(['budget', 'lr', 'rank', 'dropout', 'augment'])
          .agg(speakers=('speaker', 'nunique'), mean_rel_drop=('rel_drop', 'mean'), mean_stop_step=('stop_step', 'mean')).reset_index())
    print(f'{len(T)} runs -> {out}'); print(S.round(4).to_string(index=False))
    return T

# ---- the results tables ----------------------------------------------------------------
def _recipe(cell):
    """The name suffix after the seed, identical for a recipe's own cells and its controls."""
    tail = cell.split('__eval')[0].split('_seed', 1)[1]
    return tail.split('_', 1)[1] if '_' in tail else ''

import re
FORGIVEN_COL = re.compile(r'_f$|(^|\.)(base|tuned)_f\.')   # the withdrawn forgiven-shared count, in older result files

def _table(rows, control_budget=80):
    R = pd.json_normalize(rows)
    R = R.drop(columns=[c for c in R if FORGIVEN_COL.search(c)])
    if 'control' not in R: R['control'] = False
    R['control'] = R.control.fillna(False).astype(bool)
    for col, default in (('targets', 'protocol'), ('select', 'loss')):
        if col not in R: R[col] = default
        R[col] = R[col].fillna(default)
    # the acceptance rule: where did the improvement come from?
    R['improvement_from_insertions'] = ((R['base.I'] * R.wer_base - R['tuned.I'] * R.wer_tuned) / (R.wer_base - R.wer_tuned).replace(0, np.nan)).clip(-5, 5)
    R['style_not_speaker'] = R.improvement_from_insertions > 0.5
    # D3: personalization = delta(own adapter) - delta(others' adapter), per speaker, on the same test chunks,
    # within one recipe (the cell-name suffix).  A control trained at the cell's own budget is used when it
    # exists; otherwise the recipe's control at `control_budget` (80 min) -- the second run's rows, whose
    # smaller budgets read as 'less audio of the speaker versus 80 minutes of everyone else'.
    ctrl = R[R.control]
    if len(ctrl):
        R['_recipe'] = R.cell.map(_recipe)
        keys = ['speaker', 'arm', 'site', 'method', 'rank', 'lr', 'seed', 'targets', 'select', '_recipe']
        vals = ['delta_abs', 'delta_rel', 'wer_tuned', 'ci_lo', 'ci_hi', 'p_boot'] \
            + [c for c in ctrl if c.startswith('test07.') and ('delta_abs' in c or c == 'test07.delta_rel')]
        ren = {k: k.replace('delta', 'control_delta').replace('wer_tuned', 'wer_control').replace('ci_', 'control_ci_').replace('p_boot', 'control_p_boot') for k in vals}
        vals.append('cell'); ren['cell'] = 'control_cell'     # which control evaluation was joined: personalization_ci reads its hypotheses
        C = R[R.control][keys + ['budget'] + vals].rename(columns=ren)
        same = R[keys + ['budget']].merge(C, on=keys + ['budget'], how='left')
        fall = R[keys].merge(C[C.budget == control_budget].drop(columns='budget'), on=keys, how='left')
        for c in ren.values():
            R[c] = same[c].where(same[list(ren.values())].notna().any(axis=1), fall[c]).values
        R = R.drop(columns='_recipe')
        R['personalization_abs'] = R.delta_abs - R.control_delta_abs
        R['personalization_rel'] = R.personalization_abs / R.wer_base
        # the same on the >= 0.7 test set, overall and per quality band
        for c in [c for c in ren if c.startswith('test07.') and 'delta_abs' in c]:
            p = R[c] - R[ren[c]]; R[c.replace('delta_abs', 'personalization_abs')] = p
            R[c.replace('delta_abs', 'personalization_rel')] = p / R[c.replace('delta_abs', 'wer_base')]
    return R

def _hyps(name):
    p = os.path.join(RESULTS, name + '.hyps.json')
    if not os.path.exists(p): return None
    d = json.load(open(p, encoding='utf-8'))
    return d['chunk_ids'], d['hyps']

def personalization_ci(R, plans=None):
    """A paired bootstrap on the headline number.  personalization_abs = delta(own) -
    delta(control) = WER(control) - WER(own), because both deltas are taken against the
    same base on the same test chunks -- so it is exactly paired_bootstrap(control, own),
    resampling chunks, and needs only the two saved hypothesis files.  Adds
    personalization_ci_lo / _ci_hi / _p (absolute WER, like ci_lo / ci_hi), and test07.*
    for the >= 0.7 test set.  The point estimate must reproduce the joined
    personalization_abs: an assert, so a wrong control join cannot pass silently."""
    import evaluate as EV
    plans = plans or sorted(glob.glob(os.path.join(HERE, 'panel_*.parquet')))
    text = pd.concat([pd.read_parquet(p, columns=['chunk_id', 'text']) for p in plans]).drop_duplicates('chunk_id').set_index('chunk_id').text
    rows = R.index[~R.control & R.control_cell.notna()]
    sets = [('', '')] + ([('.test07', 'test07.')] if 'test07.personalization_abs' in R else [])
    for _, pre in sets:
        for c in ('personalization_ci_lo', 'personalization_ci_hi', 'personalization_p'):
            if pre + c not in R: R[pre + c] = np.nan
    for i in rows:
        r = R.loc[i]
        for suffix, pre in sets:
            own, ctrl = _hyps(r.cell + suffix), _hyps(r.control_cell + suffix)
            if own is None or ctrl is None: continue
            assert own[0] == ctrl[0], f'{r.cell}{suffix}: own and control were scored on different chunks'
            refs = text.loc[own[0]].tolist()
            pb = EV.paired_bootstrap(EV.score(refs, ctrl[1]), EV.score(refs, own[1]))
            assert abs(pb['delta_abs'] - r[pre + 'personalization_abs']) < 1e-9, (r.cell, suffix, pb['delta_abs'], r[pre + 'personalization_abs'])
            R.loc[i, [pre + 'personalization_ci_lo', pre + 'personalization_ci_hi', pre + 'personalization_p']] = pb['ci_lo'], pb['ci_hi'], pb['p_boot']
    return R

def summary(out=os.path.join(HERE, 'outputs', 'results.csv')):
    rows = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(RESULTS, '*.json'))) if not f.endswith('.hyps.json') and not os.path.basename(f).startswith(('base_', 'hyps_'))]
    if not rows: print('no results yet'); return None
    R = _table(rows)
    if 'control_cell' in R: R = personalization_ci(R)
    R.to_csv(out, index=False, float_format='%.5g'); print(f'{len(R)} rows ({int(R.control.sum())} control evaluations) -> {out}')
    cols = [c for c in ['speaker', 'site', 'lr', 'targets', 'select', 'budget', 'seed', 'control', 'n_test', 'train_steps', 'wer_base', 'wer_tuned', 'delta_rel', 'ci_lo', 'ci_hi', 'p_boot',
                        'improvement_from_insertions', 'style_not_speaker', 'control_delta_rel', 'personalization_rel', 'personalization_ci_lo', 'personalization_ci_hi', 'personalization_p',
                        'test07.n_test', 'test07.hq_share', 'test07.wer_base', 'test07.delta_rel', 'test07.p_boot', 'test07.personalization_rel', 'test07.personalization_p'] if c in R]
    print(R[cols].round(4).to_string(index=False))
    return R

# ---- run 4: the data curve (docs/training_plan_v4.md) ----------------------------------
CURVE = os.path.join(HERE, 'outputs', 'curve_v4.csv')
CURVE_REF, CURVE_MIN_GAIN, CURVE_MIN_SPEAKERS = 80, 0.02, 2   # the plan's decision rule, fixed before the run

def _errors(name, suffix, chunk_ids, text):
    """Per-chunk word errors of a saved hypothesis file, in `chunk_ids` order."""
    import evaluate as EV
    h = _hyps(name + suffix)
    if h is None: return None
    hyp = dict(zip(h[0], h[1]))
    if set(chunk_ids) - set(hyp): return None
    sc = EV.score([text[c] for c in chunk_ids], [hyp[c] for c in chunk_ids])
    return sc.werr.values.astype(float), sc.n_words.values.astype(float)

def curve_delta(pairs_top, pairs_ref, base_err, n_words, n_boot=2000, seed=0):
    """Delta = personalization_rel at the top budget minus at the reference budget, each the
    mean over its seeds.  pairs_*: [(own errors, control errors)] per seed, per chunk;
    personalization_rel = (control - own) / base, all summed over chunks.  The bootstrap
    resamples chunks once per draw for every seed and both budgets, so the interval holds
    the seeds' shared test set fixed.  Returns (delta, lo, hi, p, pers_top, pers_ref)."""
    def stat(ix):
        b = base_err[ix].sum(); f = lambda pairs: np.mean([(c[ix].sum() - o[ix].sum()) / b for o, c in pairs])
        return f(pairs_top) - f(pairs_ref), f(pairs_top), f(pairs_ref)
    obs = stat(np.arange(len(base_err)))
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(base_err), size=(n_boot, len(base_err)))
    d = np.array([stat(i)[0] for i in ix])
    lo, hi = np.percentile(d, [2.5, 97.5]); p = min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)
    return obs[0], lo, hi, p, obs[1], obs[2]

def curve_verdict(C, test='test07', min_gain=CURVE_MIN_GAIN, min_speakers=CURVE_MIN_SPEAKERS):
    """The plan's rule on the >= 0.7 test: 'data limits' if Delta >= +2 points with its
    interval above zero for at least 2 speakers; 'data does not limit' if every speaker's
    interval holds zero and its upper end is below +2; else 'undecided'."""
    c = C[C.test == test]
    up = ((c.delta >= min_gain) & (c.ci_lo > 0)).sum()
    if up >= min_speakers: return 'data limits'
    if ((c.ci_lo <= 0) & (c.ci_hi >= 0) & (c.ci_hi < min_gain)).all(): return 'data does not limit'
    return 'undecided'

def capacity(R, out=os.path.join(HERE, 'outputs', 'capacity_v4.csv')):
    """The capacity check (docs/STATUS.md, run 4): personalization of the r = 32 cells
    beside the r = 8 cell of the same speaker, budget and seed.  If r = 32 is clearly higher, the
    adapter's size, not the data, was the limit."""
    big = R[R['rank'] > 8]
    if not len(big): return None
    cols = ['personalization_rel', 'personalization_ci_lo', 'personalization_ci_hi', 'test07.personalization_rel',
            'test07.personalization_ci_lo', 'test07.personalization_ci_hi', 'delta_rel', 'test07.delta_rel']
    rows = []
    for r in big.itertuples(index=False):
        base = R[(R.speaker == r.speaker) & (R.budget == r.budget) & (R.seed == r.seed) & (R['rank'] == 8)]
        if not len(base): continue
        b, rr = base.iloc[0], big[(big.cell == r.cell)].iloc[0]
        rows.append(dict(speaker=r.speaker, budget=r.budget, seed=r.seed, rank=getattr(r, 'rank'),
                         **{f'r{getattr(r, "rank")}.{c}': rr[c] for c in cols if c in rr}, **{f'r8.{c}': b[c] for c in cols if c in b}))
    C = pd.DataFrame(rows)
    if len(C):
        C.to_csv(out, index=False, float_format='%.5g')
        print('capacity check (r = 32 vs r = 8, same speaker, budget, seed):')
        print(C[[c for c in C if c in ('speaker', 'budget', 'seed') or c.endswith('personalization_rel')]].round(4).to_string(index=False))
    return C

def data_curve(results=os.path.join(HERE, 'outputs', 'results.csv'), out=CURVE, ref=CURVE_REF):
    """Per speaker and test set: personalization at every budget (mean over seeds, own and
    control gains apart), and Delta = top budget - `ref` with its shared-chunk bootstrap."""
    R = pd.read_csv(results); R = R[~R.control & R.control_cell.notna() & (R.targets == 'protocol') & R.cell.str.contains('_hq')]
    capacity(R)
    R = R[R['rank'] == 8]                      # the curve is run 3's adapter; the r = 32 check is reported apart
    plans = sorted(glob.glob(os.path.join(HERE, 'panel_*.parquet')))
    text = pd.concat([pd.read_parquet(p, columns=['chunk_id', 'text']) for p in plans]).drop_duplicates('chunk_id').set_index('chunk_id').text
    import evaluate as EV
    rows = []
    for s, g in R.groupby('speaker'):
        budgets = sorted(g.budget.unique()); top = max(budgets)
        if ref not in budgets or top == ref: continue
        for suffix, pre, label in (('', '', 'hq'), ('.test07', 'test07.', 'test07')):
            col = pre + 'personalization_rel'
            if col not in g: continue
            curve = g.groupby('budget').agg(pers=(col, 'mean'), own=(pre + 'delta_rel', 'mean'), seeds=('seed', 'nunique'))
            def pairs(b):
                out, ids = [], None
                for r in g[g.budget == b].itertuples():
                    h = _hyps(r.cell + suffix)
                    if h is None: return None, None
                    ids = h[0] if ids is None else ids
                    e_own = _errors(r.cell, suffix, ids, text); e_ctl = _errors(r.control_cell, suffix, ids, text)
                    if e_own is None or e_ctl is None: return None, None
                    out.append((e_own[0], e_ctl[0]))
                return out, ids
            pt, ids = pairs(top); pr, ids_r = pairs(ref)
            if not pt or not pr or ids != ids_r: continue
            base = json.load(open(glob.glob(os.path.join(RESULTS, f'base_B_{s}_*' + ('_test07' if label == 'test07' else '_hq') + '.json'))[0], encoding='utf-8'))
            bh = dict(zip(base['chunk_ids'], base['hyps'])); sb = EV.score([text[c] for c in ids], [bh[c] for c in ids])
            d, lo, hi, p, p_top, p_ref = curve_delta(pt, pr, sb.werr.values.astype(float), sb.n_words.values.astype(float))
            rows.append(dict(speaker=s, test=label, ref_budget=ref, top_budget=top, seeds_ref=len(pr), seeds_top=len(pt),
                             pers_ref=p_ref, pers_top=p_top, delta=d, ci_lo=lo, ci_hi=hi, p=p,
                             **{f'pers_b{b}': curve.at[b, 'pers'] for b in budgets}, **{f'own_b{b}': curve.at[b, 'own'] for b in budgets}))
    C = pd.DataFrame(rows)
    if not len(C): print('no speaker has both the reference and a larger budget yet'); return None
    C.to_csv(out, index=False, float_format='%.5g')
    print(C[['speaker', 'test', 'top_budget', 'pers_ref', 'pers_top', 'delta', 'ci_lo', 'ci_hi', 'p']].round(4).to_string(index=False))
    print(f'decision (>= 0.7 test, docs/training_plan_v4.md): {curve_verdict(C)}')
    return C

def _self_check():
    panel = [556, 23558, 30685, 30701, 30718, 30752, 30777, 30813, 30831, 30843, 30859, 30868]
    # the folds over the panel are run 3's (docs/training_run3.md): an extra speaker never moves them
    for seed in (0, 1, 2):
        plain = control_folds(panel, 2, seed)
        fp = fold_plan(panel + [23641, 23635], 2, seed, extra=[23641, 23635], attach=23558)
        assert [m for _, m, *_ in fp] == plain, seed
        x = [f for f in fp if f[2]]; assert len(x) == 1 and 23558 in x[0][1] and x[0][5].endswith('x') and set(x[0][3]) >= {23558, 23641, 23635}
        assert all(23641 not in f[4] and 23635 not in f[4] for f in fp)                   # never in a pool
    assert control_folds(panel, 2, 0)[1] == [23558, 30685, 30701, 30718, 30859, 30868]   # run 3's seed-0 fold of 23558
    only_new = fold_plan(panel + [23641, 23635], 2, 0, extra=[23641, 23635], attach=23558, evals_only=[23641, 23635])
    assert len(only_new) == 1 and only_new[0][3] == [23641, 23635]
    # the curve: own beats control by 10 % of base errors at the top, by 0 at the reference
    base = np.full(50, 10.0); own_t, ctl_t = np.full(50, 8.0), np.full(50, 9.0); own_r = ctl_r = np.full(50, 9.0)
    d, lo, hi, p, top, refv = curve_delta([(own_t, ctl_t)] * 3, [(own_r, ctl_r)] * 3, base, np.ones(50))
    assert abs(d - 0.1) < 1e-12 and abs(top - 0.1) < 1e-12 and refv == 0 and lo == hi == d
    C = pd.DataFrame(dict(test='test07', delta=[0.05, 0.03, 0.0], ci_lo=[0.01, 0.005, -0.01], ci_hi=[0.08, 0.06, 0.01]))
    assert curve_verdict(C) == 'data limits'
    assert curve_verdict(C.assign(delta=0.0, ci_lo=-0.01, ci_hi=0.01)) == 'data does not limit'
    assert curve_verdict(C.assign(delta=[0.05, 0.0, 0.0], ci_lo=[0.01, -0.01, -0.01], ci_hi=[0.08, 0.03, 0.01])) == 'undecided'
    print('run_panel.py: OK')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--speakers', nargs='+', type=int); ap.add_argument('--arm', nargs='+', default=['B'], dest='arms')
    ap.add_argument('--sites', nargs='+', default=['both']); ap.add_argument('--methods', nargs='+', default=['lora'])
    ap.add_argument('--budgets', nargs='+', type=int, default=[5, 20, 80]); ap.add_argument('--ranks', nargs='+', type=int, default=[8])
    ap.add_argument('--lrs', nargs='+', type=float, default=[None]); ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--eval-batch', type=int, default=8); ap.add_argument('--model', help='override the arm\'s checkpoint (smoke tests only)')
    ap.add_argument('--audio', default=AUDIO); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--control-folds', type=int, default=0, help='D3: train the cross-speaker control in K folds after the cells (0 = none)')
    ap.add_argument('--control-budget', type=int, default=80); ap.add_argument('--control-only', action='store_true')
    ap.add_argument('--transcribe-parts', action='store_true', help='both arms over every speaker\'s train and dev chunks (cached), then exit')
    ap.add_argument('--build-targets', action='store_true', help='write outputs/targets_verbatim.parquet from the cached train/dev transcriptions, then exit')
    ap.add_argument('--targets', choices=['protocol', 'verbatim'], default='protocol', help='training target text (docs/archive/training_next.md A1)')
    ap.add_argument('--plan', default=PLAN, help='the data plan; panel_plan_v2.parquet = the high-quality plan (docs/training_plan_v3.md)')
    ap.add_argument('--max-steps', type=int, help='step mode: this many optimiser steps for every budget (docs/training_plan_v3.md)')
    ap.add_argument('--eval-steps', type=int, default=20, help='step mode: validate every N steps')
    ap.add_argument('--patience', type=int, default=4, help='step mode: stop after N validations without improvement (fixed, not tuned)')
    ap.add_argument('--min-epochs', type=float, help='step mode: early stopping may not stop before this many passes over the train set (run 4: 720 and 1,440 min)')
    ap.add_argument('--dropouts', nargs='+', type=float, default=[None], help="step mode: Whisper's `dropout` (0 in the checkpoint)")
    ap.add_argument('--augments', nargs='+', default=['none'], choices=['none', 'specaug', 'specaug+tempo', 'specaug+tempo+noise'])
    ap.add_argument('--control-budgets', nargs='+', type=int, help='train the control at each of these budgets (default: --control-budget)')
    ap.add_argument('--tune', action='store_true', help='train + validate only, no test (runs_tune/); then --tuning-report')
    ap.add_argument('--tuning-report', action='store_true', help='write outputs/tuning.csv from the tuning runs and print the comparison')
    ap.add_argument('--test07-plan', help='also score on this second test set (panel_test07.parquet: plan v2\'s test sessions at quality >= 0.7)')
    ap.add_argument('--cell-speakers', nargs='+', type=int, help='run 4: own cells only for these speakers; the plan keeps everyone for the control\'s pool')
    ap.add_argument('--control-evals', nargs='+', type=int, help='run 4: evaluate the control only on these speakers (folds without any are not trained)')
    ap.add_argument('--control-extra', nargs='+', type=int, default=[], help='run 4: speakers outside the panel -- not in the folds, attached to --control-attach\'s fold')
    ap.add_argument('--control-attach', type=int, help='run 4: the panel speaker whose fold the --control-extra speakers join')
    ap.add_argument('--no-report', action='store_true', help='--tune: do not rewrite tuning.csv (parallel workers)')
    ap.add_argument('--no-summary', action='store_true', help='do not rewrite results.csv at the end (parallel workers; the queue runs it once)')
    ap.add_argument('--curve', action='store_true', help='run 4: outputs/curve_v4.csv from outputs/results.csv and the saved hypotheses')
    ap.add_argument('--self-check', action='store_true', help='the fold and curve logic on synthetic data, no GPU')
    a = ap.parse_args()
    if a.self_check: _self_check(); sys.exit(0)
    if a.curve: data_curve(); sys.exit(0)
    if a.summary: summary(); sys.exit(0)
    if a.tuning_report: tuning_report(); sys.exit(0)
    if (a.dropouts != [None] or a.augments != ['none'] or a.tune) and not a.max_steps:
        ap.error('--dropouts, --augments and --tune are step-mode settings: give --max-steps')
    P = load_plan(a.plan, speakers=a.speakers)
    T07 = load_plan(a.test07_plan, speakers=a.speakers) if (a.test07_plan and not a.tune) else None
    if T07 is not None:
        assert set(P[P.part == 'test'].chunk_id) <= set(T07.chunk_id), f'{a.test07_plan} does not contain {a.plan}\'s test set: rebuild it (materialize.py plan-test07)'
    data_tag = '' if os.path.abspath(a.plan) == os.path.abspath(PLAN) else 'hq'
    steps = [dict(max_steps=a.max_steps, eval_steps=a.eval_steps, patience=a.patience, dropout=d, augment=g, data=data_tag,
                  **({'min_epochs': a.min_epochs} if a.min_epochs else {}))
             for d in a.dropouts for g in a.augments] if a.max_steps else [{}]
    if a.transcribe_parts: transcribe_parts(P, a.audio, a.eval_batch); sys.exit(0)
    if a.build_targets:
        import targets as TG
        T = TG.build_targets(P, RESULTS); TG.report(T); T.to_parquet(os.path.join(HERE, 'outputs', 'targets_verbatim.parquet'), index=False); sys.exit(0)
    C = [] if a.control_only else list(cells(P[P.speaker_id.isin(a.cell_speakers)] if a.cell_speakers else P, a.arms, a.sites, a.methods, a.budgets, a.ranks, a.lrs, a.seeds))
    print(f'{len(C)} cells over {P.speaker_id.nunique()} speakers; test {P[P.part=="test"].duration_s.sum()/60:.0f} min, train {P[P.part=="train"].duration_s.sum()/60:.0f} min available'
          + (f'; then the cross-speaker control in {a.control_folds} fold(s) at {"/".join(map(str, a.control_budgets or [a.control_budget]))} min' if a.control_folds else ''))
    if a.dry_run:
        import train as T
        for st in steps:
            for c in C: print('  ' + ('[tune] ' if a.tune else '') + T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']], c['seed'],
                                                full_tag(a.targets, 'loss', st)))
        if a.control_folds:
            for k, members, attached, evals, trainers, label in fold_plan(P.speaker_id.unique(), a.control_folds, a.seeds[0], a.control_extra, a.control_attach, a.control_evals):
                print(f'  {label}: trains on {trainers}, evaluates {evals}')
        sys.exit(0)
    if a.tune:                                 # validation only: nothing here reads the test set
        for st in steps:
            for c in C: run_tune(c, P, a.audio, a.batch, a.grad_accum, st, a.model)
        if not a.no_report: tuning_report()
        sys.exit(0)
    for st in steps:
        for c in C: run_cell(c, P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, a.model, targets=a.targets, step=st, T07=T07)
    if a.control_folds:                        # one control per recipe, so every sweep point has a budget-matched 'others' adapter
        for st in steps:
            for cb in (a.control_budgets or [a.control_budget]):
                for site in a.sites:
                    for lr in a.lrs:
                        run_control(P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, folds=a.control_folds, budget=cb, arm=a.arms[0], site=site,
                                    method=a.methods[0], rank=a.ranks[0], lr=lr, seed=a.seeds[0], targets=a.targets, step=st, T07=T07,
                                    extra=a.control_extra, attach=a.control_attach, evals_only=a.control_evals)
    if not a.no_summary: summary()
