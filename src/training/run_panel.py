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

  forgiven-shared WER   with --forgiven, every cell is also scored under the
        protocol-aware count (error_analysis.forgiven_counts): an inserted word
        that the OTHER arm also produced at that chunk is not charged.  Arm A's
        transcription of each speaker's test set is cached once like arm B's.
        `*_f` columns.  Off by default since plan v3, which trains and tests on
        high-quality clips only (docs/training_plan_v3.md); the second run's rows
        carry it.
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
ones, always with the forgiven-shared count (arm A on it, cached once per speaker) and
split by quality band (`test07.bands.q070` ... `q095`).  The high-quality test is a
subset of it, so its hypotheses are reused and only the rest is transcribed.  A cell
scored before the flag was given gets its test07 scores on the next run, from its
saved adapter.

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

QBANDS = ((0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01))

def compare_bands(test, hb, ht, ha):
    """compare() within each alignment-quality band of the >= 0.7 test set: where the
    adapter helps, on the clean clips or on the hard ones the high-quality filter drops."""
    import evaluate as EV
    sb, st = EV.score(test.text, hb), EV.score(test.text, ht)
    fb, ft = forgiven_score(test.text, hb, ha), forgiven_score(test.text, ht, ha)
    out = {}
    for lo, hi in QBANDS:
        m = ((test.quality >= lo) & (test.quality < hi)).values
        if not m.any(): continue
        pb = EV.paired_bootstrap(sb[m], st[m]); pf = EV.paired_bootstrap(fb[m], ft[m])
        out[f'q{round(lo * 100):03d}'] = dict(minutes=float(test.duration_s[m].sum() / 60), n_segments=int(m.sum()),
                                             **{k: pb[k] for k in ('wer_base', 'wer_tuned', 'delta_abs', 'delta_rel', 'ci_lo', 'ci_hi', 'p_boot')},
                                             **{k + '_f': pf[k] for k in ('wer_base', 'wer_tuned', 'delta_abs', 'delta_rel', 'p_boot')})
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
    ha7 = base_hyps(OTHER_ARM[arm], speaker, t07, audio_dir, batch, data='test07')
    json.dump(dict(chunk_ids=list(t07.chunk_id), hyps=ht7), open(os.path.join(RESULTS, name + '.test07.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    res = dict(n_test=int(len(t07)), test_minutes=float(t07.duration_s.sum() / 60), hq_share=float(t07.duration_s[t07.hq].sum() / t07.duration_s.sum()),
               **compare(t07, hb7, ht7, ha7), bands=compare_bands(t07, hb7, ht7, ha7))
    print(f'{name} on the >= 0.7 test: WER {res["wer_base"]:.4f} -> {res["wer_tuned"]:.4f} ({res["delta_rel"]:+.1%}, p {res["p_boot"]:.3f}); '
          f'forgiven {res["delta_rel_f"]:+.1%}; ' + ' '.join(f'{k} {v["delta_rel"]:+.1%}' for k, v in res['bands'].items()), flush=True)
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
    part) -- the raw material for the semi-verbatim targets (targets.py) and for
    checkpoint selection on dev forgiven WER (train.py).  `model` lets a caller
    keep one loaded model across speakers."""
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

def run_cell(c, P, audio_dir, epochs, batch, grad_accum, eval_batch, model_override=None, forgiven=False, targets='protocol', select='loss', step=None, T07=None):
    import train as T, evaluate as EV
    if model_override:                         # smoke tests: both arms, so the other-arm transcription is tiny too
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
    skw = select_kw(P, spk[spk.part == 'dev'], select, audio_dir, eval_batch, c['arm'])
    adapter = T.train_cell(spk, audio_dir, RUNS, c['speaker'], arm=c['arm'], site=c['site'], method=c['method'], budget=c['budget'],
                           rank=c['rank'], lr=c['lr'], seed=c['seed'], epochs=epochs, batch=batch, grad_accum=grad_accum, tag=tag, **skw, **step_kw(step))
    phase['train'] = time.time() - t0
    spk = P[P.speaker_id == c['speaker']]      # scoring: the protocol test text
    test = spk[spk.part == 'test'].reset_index(drop=True)
    data = step.get('data', '')
    t = time.time(); hb = base_hyps(c['arm'], c['speaker'], test, audio_dir, eval_batch, model_override, data); phase['base'] = time.time() - t
    # arm A's (the other arm's) transcription of the same chunks, for the forgiven-shared count; cached per speaker
    t = time.time(); ha = base_hyps(OTHER_ARM[c['arm']], c['speaker'], test, audio_dir, eval_batch, data=data) if forgiven else None; phase['other_arm'] = time.time() - t
    t = time.time(); model, proc, device = EV.load(c['arm'], adapter=adapter); phase['load_tuned'] = time.time() - t
    t = time.time(); ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device); phase['transcribe_tuned'] = time.time() - t
    t = time.time(); cmp = compare(test, hb, ht, ha); phase['score'] = time.time() - t
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
                rank=8, lr=None, seed=0, dev_min=15, forgiven=False, targets='protocol', select='loss', step=None, T07=None):
    """k adapters, each trained on a pool from the speakers NOT in its fold and
    evaluated on the fold's speakers.  With k=1 (the handoff's single adapter)
    the pool spans all the speakers and each is evaluated on an adapter that saw
    ~budget/11 minutes of them -- stated in the JSON as `saw_own_minutes`."""
    import train as T, evaluate as EV
    spk_ids = sorted(int(s) for s in P.speaker_id.unique()); fold_of = control_folds(spk_ids, folds, seed)
    step = step or {}
    lr = lr if lr is not None else T.DEFAULT_LR[method]; out = []; tag = full_tag(targets, select, step); Pt = apply_targets(P, targets)
    for k, evals in enumerate(fold_of):
        trainers = [s for s in spk_ids if s not in evals] if folds > 1 else spk_ids
        label = f'ctrl{k}of{folds}'
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
        held = set(P[P.speaker_id.isin(evals) & P.part.isin(['test', 'dev'])].session_id)
        if T07 is not None: held |= set(T07[T07.speaker_id.isin(evals)].session_id)
        Pc = Pt[~Pt.session_id.isin(held)]
        tr = pool_budget(Pc, trainers, budget, 'train', seed); dev = pool_budget(Pc, trainers, dev_min, 'dev', seed)
        assert not set(tr.session_id) & held and not set(dev.session_id) & held
        print(f'{name}: pool of {len(trainers)} speakers, {len(tr)} chunks ({tr.duration_s.sum()/60:.1f} min), dev {len(dev)}; evaluates {evals}', flush=True)
        skw = select_kw(P, dev, select, audio_dir, eval_batch, arm)
        adapter = T.train_cell(Pt, audio_dir, RUNS, label, arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed,
                               epochs=epochs, batch=batch, grad_accum=grad_accum, train_rows=tr, dev_rows=dev, tag=tag, **skw, **step_kw(step))
        t_train = time.time() - t0; meta = _train_meta(adapter)
        per_spk_min = tr.groupby('speaker_id').duration_s.sum().div(60).to_dict()
        model, proc, device = EV.load(arm, adapter=adapter)
        for s in todo:
            t1 = time.time(); test = P[(P.speaker_id == s) & (P.part == 'test')].reset_index(drop=True)
            data = step.get('data', '')
            hb = base_hyps(arm, s, test, audio_dir, eval_batch, data=data); ha = base_hyps(OTHER_ARM[arm], s, test, audio_dir, eval_batch, data=data) if forgiven else None
            ht = EV.transcribe_short(model, proc, test, audio_dir, batch=eval_batch, device=device)
            cmp = compare(test, hb, ht, ha)
            if T07 is not None: cmp['test07'] = score_test07(T07, s, arm, test, ht, model, proc, device, audio_dir, eval_batch, f'{name}__eval{s}')
            res = dict(cell=f'{name}__eval{s}', speaker=s, arm=arm, site=site, method=method, budget=budget, rank=rank, lr=lr, seed=seed,
                       targets=targets, select=select, control=True, fold=k, folds=folds, trained_on=trainers, saw_own_minutes=float(per_spk_min.get(s, 0.0)),
                       adapter=adapter, train_minutes=float(tr.duration_s.sum() / 60), train_chunks=int(len(tr)), train_steps=meta.get('global_step'),
                       train_epochs=meta.get('epochs', epochs), best_eval_loss=meta.get('best_eval_loss'),
                       n_test=int(len(test)), test_minutes=float(test.duration_s.sum() / 60), **cmp,
                       seconds_train=t_train, seconds_eval=time.time() - t1, model=T.ARMS[arm], decode=EV.DECODE)
            json.dump(res, open(os.path.join(RESULTS, f'{name}__eval{s}.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False, default=float)
            json.dump(dict(chunk_ids=list(test.chunk_id), hyps=ht), open(os.path.join(RESULTS, f'{name}__eval{s}.hyps.json'), 'w', encoding='utf-8'), ensure_ascii=False)
            f = f'; forgiven {cmp["wer_base_f"]:.4f} -> {cmp["wer_tuned_f"]:.4f} ({cmp["delta_rel_f"]:+.1%})' if ha is not None else ''
            print(f'{name} on {s}: WER {cmp["wer_base"]:.4f} -> {cmp["wer_tuned"]:.4f} ({cmp["delta_rel"]:+.1%}, p {cmp["p_boot"]:.3f}){f} in {time.time()-t1:.0f}s', flush=True)
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

def tuning_report(out=TUNING):
    """One row per tuning run: the settings, the untuned and the best validation
    loss, and rel_drop = the relative drop between them -- the tuning criterion
    (docs/training_plan_v3.md, step 3).  Prints the mean over speakers per recipe
    and budget, with how many speakers each recipe helped."""
    rows = []
    for d in sorted(glob.glob(os.path.join(TUNE_RUNS, '*', 'train_meta.json'))):
        m = json.load(open(d)); name = os.path.basename(os.path.dirname(d))
        if not m.get('max_steps') or m.get('base_eval_loss') is None or not m.get('evals'): continue
        head = name.split('_')                        # s<speaker>_arm<A>_<site>_<method>_b<budget>_r<rank>_lr<lr>_seed<seed>_<tag>
        best = min(m['evals'], key=lambda e: e['eval_loss'])
        rows.append(dict(run=name, speaker=head[0][1:], budget=int(head[4][1:]), lr=float(head[6][2:]), rank=int(head[5][1:]),
                         dropout=m.get('dropout') or 0.0, augment=m.get('augment', 'none'), patience=m.get('patience'),
                         base_loss=m['base_eval_loss'], best_loss=best['eval_loss'], rel_drop=1 - best['eval_loss'] / m['base_eval_loss'],
                         best_step=best.get('step'), stop_step=m.get('global_step'), train_minutes=m.get('train_minutes')))
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

def _table(rows, control_budget=80):
    R = pd.json_normalize(rows)
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
        vals = ['delta_abs', 'delta_rel', 'wer_tuned', 'ci_lo', 'ci_hi', 'p_boot'] + [c for c in ('delta_abs_f', 'delta_rel_f', 'wer_tuned_f') if c in ctrl] \
            + [c for c in ctrl if c.startswith('test07.') and ('delta_abs' in c or c in ('test07.delta_rel', 'test07.delta_rel_f'))]
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
        if 'delta_abs_f' in R: R['personalization_abs_f'] = R.delta_abs_f - R.control_delta_abs_f
        # the same on the >= 0.7 test set, overall and per quality band, standard and forgiven
        for c in [c for c in ren if c.startswith('test07.') and 'delta_abs' in c]:
            p = R[c] - R[ren[c]]; R[c.replace('delta_abs', 'personalization_abs')] = p
            R[c.replace('delta_abs', 'personalization_rel')] = p / R[c.replace('delta_abs', 'wer_base')]
    return R

def _hyps(name):
    p = os.path.join(RESULTS, name + '.hyps.json')
    if not os.path.exists(p): return None
    d = json.load(open(p, encoding='utf-8'))
    return d['chunk_ids'], d['hyps']

def _other_arm_hyps(arm, speaker, chunk_ids):
    """The other arm's cached base transcription of exactly these chunks (any plan's cache), or None."""
    for p in sorted(glob.glob(os.path.join(RESULTS, f'base_{OTHER_ARM[arm]}_{speaker}_*.json'))):
        d = json.load(open(p, encoding='utf-8'))
        if d['chunk_ids'] == chunk_ids: return d['hyps']
    return None

def personalization_ci(R, plans=None):
    """A paired bootstrap on the headline number.  personalization_abs = delta(own) -
    delta(control) = WER(control) - WER(own), because both deltas are taken against the
    same base on the same test chunks -- so it is exactly paired_bootstrap(control, own),
    resampling chunks, and needs only the two saved hypothesis files.  Adds
    personalization_ci_lo / _ci_hi / _p (absolute WER, like ci_lo / ci_hi), the same with
    _f under the forgiven-shared count when the other arm's transcription is cached, and
    test07.* for the >= 0.7 test set.  The point estimate must reproduce the joined
    personalization_abs: an assert, so a wrong control join cannot pass silently."""
    import evaluate as EV
    plans = plans or sorted(glob.glob(os.path.join(HERE, 'panel_*.parquet')))
    text = pd.concat([pd.read_parquet(p, columns=['chunk_id', 'text']) for p in plans]).drop_duplicates('chunk_id').set_index('chunk_id').text
    rows = R.index[~R.control & R.control_cell.notna()]
    sets = [('', '')] + ([('.test07', 'test07.')] if 'test07.personalization_abs' in R else [])
    for _, pre in sets:
        for c in ('personalization_ci_lo', 'personalization_ci_hi', 'personalization_p', 'personalization_abs_f', 'personalization_ci_lo_f', 'personalization_ci_hi_f', 'personalization_p_f'):
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
            ha = _other_arm_hyps(r.arm, int(r.speaker), own[0])
            if ha is not None:
                pf = EV.paired_bootstrap(forgiven_score(refs, ctrl[1], ha), forgiven_score(refs, own[1], ha))
                R.loc[i, [pre + 'personalization_abs_f', pre + 'personalization_ci_lo_f', pre + 'personalization_ci_hi_f', pre + 'personalization_p_f']] = \
                    pf['delta_abs'], pf['ci_lo'], pf['ci_hi'], pf['p_boot']
    return R

def summary(out=os.path.join(HERE, 'outputs', 'results.csv')):
    rows = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(os.path.join(RESULTS, '*.json'))) if not f.endswith('.hyps.json') and not os.path.basename(f).startswith(('base_', 'hyps_'))]
    if not rows: print('no results yet'); return None
    R = _table(rows)
    if 'control_cell' in R: R = personalization_ci(R)
    R.to_csv(out, index=False, float_format='%.5g'); print(f'{len(R)} rows ({int(R.control.sum())} control evaluations) -> {out}')
    cols = [c for c in ['speaker', 'site', 'lr', 'targets', 'select', 'budget', 'seed', 'control', 'n_test', 'train_steps', 'wer_base', 'wer_tuned', 'delta_rel', 'ci_lo', 'ci_hi', 'p_boot', 'wer_base_f', 'wer_tuned_f', 'delta_rel_f', 'p_boot_f',
                        'improvement_from_insertions', 'style_not_speaker', 'control_delta_rel', 'personalization_rel', 'personalization_ci_lo', 'personalization_ci_hi', 'personalization_p',
                        'test07.n_test', 'test07.hq_share', 'test07.wer_base', 'test07.delta_rel', 'test07.p_boot', 'test07.delta_rel_f', 'test07.personalization_rel', 'test07.personalization_p'] if c in R]
    print(R[cols].round(4).to_string(index=False))
    return R

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--speakers', nargs='+', type=int); ap.add_argument('--arm', nargs='+', default=['B'], dest='arms')
    ap.add_argument('--sites', nargs='+', default=['both']); ap.add_argument('--methods', nargs='+', default=['lora'])
    ap.add_argument('--budgets', nargs='+', type=int, default=[5, 20, 80]); ap.add_argument('--ranks', nargs='+', type=int, default=[8])
    ap.add_argument('--lrs', nargs='+', type=float, default=[None]); ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--eval-batch', type=int, default=8); ap.add_argument('--model', help='override the arm\'s checkpoint (smoke tests only)')
    ap.add_argument('--audio', default=AUDIO); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--forgiven', action='store_true', help='also transcribe with arm A and score the forgiven-shared count (off since plan v3)')
    ap.add_argument('--control-folds', type=int, default=0, help='D3: train the cross-speaker control in K folds after the cells (0 = none)')
    ap.add_argument('--control-budget', type=int, default=80); ap.add_argument('--control-only', action='store_true')
    ap.add_argument('--transcribe-parts', action='store_true', help='both arms over every speaker\'s train and dev chunks (cached), then exit')
    ap.add_argument('--build-targets', action='store_true', help='write outputs/targets_verbatim.parquet from the cached train/dev transcriptions, then exit')
    ap.add_argument('--targets', choices=['protocol', 'verbatim'], default='protocol', help='training target text (docs/training_next.md A1)')
    ap.add_argument('--select', choices=['loss', 'forgiven'], default='loss', help='checkpoint selection: dev loss, or dev forgiven-shared WER (A2)')
    ap.add_argument('--plan', default=PLAN, help='the data plan; panel_plan_v2.parquet = the high-quality plan (docs/training_plan_v3.md)')
    ap.add_argument('--max-steps', type=int, help='step mode: this many optimiser steps for every budget (docs/training_plan_v3.md)')
    ap.add_argument('--eval-steps', type=int, default=20, help='step mode: validate every N steps')
    ap.add_argument('--patience', type=int, default=4, help='step mode: stop after N validations without improvement (fixed, not tuned)')
    ap.add_argument('--dropouts', nargs='+', type=float, default=[None], help="step mode: Whisper's `dropout` (0 in the checkpoint)")
    ap.add_argument('--augments', nargs='+', default=['none'], choices=['none', 'specaug', 'specaug+tempo', 'specaug+tempo+noise'])
    ap.add_argument('--control-budgets', nargs='+', type=int, help='train the control at each of these budgets (default: --control-budget)')
    ap.add_argument('--tune', action='store_true', help='train + validate only, no test (runs_tune/); then --tuning-report')
    ap.add_argument('--tuning-report', action='store_true', help='write outputs/tuning.csv from the tuning runs and print the comparison')
    ap.add_argument('--test07-plan', help='also score on this second test set (panel_test07.parquet: plan v2\'s test sessions at quality >= 0.7)')
    a = ap.parse_args()
    if a.summary: summary(); sys.exit(0)
    if a.tuning_report: tuning_report(); sys.exit(0)
    if (a.dropouts != [None] or a.augments != ['none'] or a.tune) and not a.max_steps:
        ap.error('--dropouts, --augments and --tune are step-mode settings: give --max-steps')
    P = load_plan(a.plan, speakers=a.speakers)
    T07 = load_plan(a.test07_plan, speakers=a.speakers) if (a.test07_plan and not a.tune) else None
    if T07 is not None:
        assert set(P[P.part == 'test'].chunk_id) <= set(T07.chunk_id), f'{a.test07_plan} does not contain {a.plan}\'s test set: rebuild it (materialize.py plan-test07)'
    data_tag = '' if os.path.abspath(a.plan) == os.path.abspath(PLAN) else 'hq'
    steps = [dict(max_steps=a.max_steps, eval_steps=a.eval_steps, patience=a.patience, dropout=d, augment=g, data=data_tag)
             for d in a.dropouts for g in a.augments] if a.max_steps else [{}]
    if a.transcribe_parts: transcribe_parts(P, a.audio, a.eval_batch); sys.exit(0)
    if a.build_targets:
        import targets as TG
        T = TG.build_targets(P, RESULTS); TG.report(T); T.to_parquet(os.path.join(HERE, 'outputs', 'targets_verbatim.parquet'), index=False); sys.exit(0)
    C = [] if a.control_only else list(cells(P, a.arms, a.sites, a.methods, a.budgets, a.ranks, a.lrs, a.seeds))
    print(f'{len(C)} cells over {P.speaker_id.nunique()} speakers; test {P[P.part=="test"].duration_s.sum()/60:.0f} min, train {P[P.part=="train"].duration_s.sum()/60:.0f} min available'
          + (f'; then the cross-speaker control in {a.control_folds} fold(s) at {"/".join(map(str, a.control_budgets or [a.control_budget]))} min' if a.control_folds else ''))
    if a.dry_run:
        import train as T
        for st in steps:
            for c in C: print('  ' + ('[tune] ' if a.tune else '') + T.cell_name(c['speaker'], c['arm'], c['site'], c['method'], c['budget'], c['rank'], c['lr'] if c['lr'] is not None else T.DEFAULT_LR[c['method']], c['seed'],
                                                full_tag(a.targets, a.select, st)))
        if a.control_folds:
            for k, f in enumerate(control_folds(sorted(int(s) for s in P.speaker_id.unique()), a.control_folds, a.seeds[0])): print(f'  ctrl{k}of{a.control_folds}: evaluates {f}')
        sys.exit(0)
    if a.tune:                                 # validation only: nothing here reads the test set
        for st in steps:
            for c in C: run_tune(c, P, a.audio, a.batch, a.grad_accum, st, a.model)
        tuning_report(); sys.exit(0)
    for st in steps:
        for c in C: run_cell(c, P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, a.model, forgiven=a.forgiven, targets=a.targets, select=a.select, step=st, T07=T07)
    if a.control_folds:                        # one control per recipe, so every sweep point has a budget-matched 'others' adapter
        for st in steps:
            for cb in (a.control_budgets or [a.control_budget]):
                for site in a.sites:
                    for lr in a.lrs:
                        run_control(P, a.audio, a.epochs, a.batch, a.grad_accum, a.eval_batch, folds=a.control_folds, budget=cb, arm=a.arms[0], site=site,
                                    method=a.methods[0], rank=a.ranks[0], lr=lr, seed=a.seeds[0], forgiven=a.forgiven, targets=a.targets, select=a.select, step=st, T07=T07)
    summary()
