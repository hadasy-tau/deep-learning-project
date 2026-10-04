"""Stage 2 training: one cell -> one adapter.

A cell is (speaker, arm, site, method, budget, rank, lr, seed) -- D4's axes.
Idempotent: a cell whose out_dir already holds an adapter is skipped.

Written against transformers 5.x, which differs from the plan's gotchas in one
place: `forced_decoder_ids` no longer exists (GenerationConfig dropped it), so
language/task are set on the generation config instead. The rest of the plan's
gotchas -- -100 label masking, use_cache=False under gradient checkpointing,
enable_input_require_grads() under PEFT -- still apply and are encoded below.
"""
import contextlib, os
# Cap the BLAS/OpenMP pool BEFORE numpy loads.  Whisper's log-mel extraction is a
# numpy STFT + mel matmul, and on a 255-core box the default pool (one thread per
# core) made it 1.4 s per chunk against 8 ms with 4 threads (measured 2026-09-20 on
# an A100 pod).  Through __getitem__ and the per-epoch dev eval that alone was
# ~25 of the first run's ~30 minutes per cell.  A value already set in the
# environment wins.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import numpy as np, torch
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
from torch.utils.data import Dataset
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import read_wav, budget_order

ARMS = {'A': 'openai/whisper-large-v3', 'B': 'ivrit-ai/whisper-large-v3'}

# D4's site axis. PEFT accepts a regex string for target_modules, which is the
# only way to isolate decoder self-attn from decoder cross-attn -- keeping them
# separate is what makes "acoustic or lexical?" answerable.
SITES = {
    'encoder':      r'.*model\.encoder.*\.(q_proj|v_proj)',
    'decoder_self': r'.*model\.decoder.*\.self_attn\.(q_proj|v_proj)',
    'decoder_cross': r'.*model\.decoder.*\.encoder_attn\.(q_proj|v_proj)',
    'both':         r'.*\.(q_proj|v_proj)',
    'all_linear':   r'.*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)',
    # the decoder's MLPs: where a placement study on atypical speakers found the
    # speaker information lands (self-attention "unstable and rarely beneficial")
    'decoder_mlp':  r'.*model\.decoder.*\.(fc1|fc2)',
}

# D4's reference lr of 1e-3 is an ADAPTER learning rate. Applied to full
# fine-tuning it is ~100x too high, and the failure is silent: the run completes
# and reports a wrecked model rather than raising. ivrit-ai trained this very
# checkpoint at 1e-5, so that is the full-FT default here. Passing lr explicitly
# overrides these -- the optimization axis sweeps {3e-4, 1e-3, 3e-3} on adapters.
DEFAULT_LR = {'lora': 1e-3, 'dora': 1e-3, 'ia3': 1e-3, 'full': 1e-5}

# docs/training_plan_v3.md.  Step mode (max_steps set): every budget gets the same
# number of optimiser steps, validation every EVAL_STEPS, a constant rate after
# WARMUP_STEPS (a decaying schedule would tie the rate to max_steps, and a run
# that stops early would never reach its low-rate phase).
EVAL_STEPS, WARMUP_STEPS = 20, 20
# Augmentation settings, fixed (not tuned): SpecAugment through Whisper's config,
# tempo and noise on the waveform (ChunkDataset).
SPECAUG = dict(apply_spec_augment=True, mask_time_prob=0.05, mask_time_length=10,
               mask_feature_prob=0.05, mask_feature_length=10)
TEMPO, NOISE_SNR_DB = (0.9, 1.1), (10.0, 25.0)
AUGMENTS = {'none': (), 'specaug': ('specaug',), 'specaug+tempo': ('specaug', 'tempo'),
            'specaug+tempo+noise': ('specaug', 'tempo', 'noise')}


def recipe_tag(max_steps=None, eval_steps=EVAL_STEPS, patience=None, dropout=None, augment='none', data='', min_epochs=None):
    """The step-mode settings as a cell-name suffix, so no two recipes share a
    directory or a results file.  Empty in epoch mode (the old cells' names)."""
    if not max_steps:
        return ''
    t = [data] if data else []
    t.append(f'ms{max_steps}')
    if eval_steps != EVAL_STEPS: t.append(f'ev{eval_steps}')
    if patience: t.append(f'pat{patience}')
    if dropout: t.append(f'do{dropout:g}')
    if augment and augment != 'none': t.append('aug-' + augment.replace('+', '-'))
    if min_epochs: t.append(f'minep{min_epochs:g}')
    return '_'.join(t)


def _min_steps_early_stopping(patience, min_steps):
    """EarlyStoppingCallback that may not stop before `min_steps` (docs/training_plan_v4.md):
    at 720 and 1,440 minutes the fixed patience would end training before one pass over the
    data, so the largest budgets would not really have been seen.  It keeps counting
    validations without improvement all along, so once min_steps is reached it stops at the
    first validation where patience is used up; the best checkpoint is restored as before."""
    from transformers import EarlyStoppingCallback
    class MinStepsEarlyStopping(EarlyStoppingCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            super().on_evaluate(args, state, control, metrics, **kwargs)
            if state.global_step < min_steps: control.should_training_stop = False
    return MinStepsEarlyStopping(early_stopping_patience=patience)


class ChunkDataset(Dataset):
    """Chunks -> (log-mel features, label ids). Audio is sliced out of the
    whole-recording wavs that `materialize` wrote.

    augment: waveform augmentations for the TRAINING set only (docs/training_plan_v3.md
    § Augmentation), drawn afresh every time a chunk is read.  'tempo' changes the
    speaking rate at the same pitch (0.9-1.1, half the chunks) -- unlike resampling
    'speed' perturbation it keeps the voice; 'noise' adds coloured noise at
    10-25 dB SNR (half the chunks).  SpecAugment is not here: it is Whisper's own,
    switched on in the model config (train_cell's augment)."""

    def __init__(self, chunks, audio_dir, processor, dtype=torch.float32, augment=()):
        self.rows = chunks.reset_index(drop=True)
        self.audio_dir, self.proc, self.dtype = audio_dir, processor, dtype
        self.augment, self._rng = tuple(augment), None

    def __len__(self):
        return len(self.rows)

    def _augmented(self, wav):
        import audiomentations as AU
        if self._rng is None:        # one generator per worker, seeded from the Trainer's seed (torch gives each worker its own)
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        r = self._rng
        if 'tempo' in self.augment and r.random() < 0.5:
            # a slower rate lengthens the chunk; never past Whisper's 30 s window
            lo = max(TEMPO[0], len(wav) / (29.5 * 16000))
            rate = r.uniform(lo, TEMPO[1]) if lo < TEMPO[1] else 1.0
            if rate != 1.0:
                wav = AU.TimeStretch(min_rate=rate, max_rate=rate, leave_length_unchanged=False, p=1.0)(wav, sample_rate=16000)
        if 'noise' in self.augment and r.random() < 0.5:
            snr = r.uniform(*NOISE_SNR_DB)
            wav = AU.AddColorNoise(min_snr_db=snr, max_snr_db=snr, p=1.0)(wav, sample_rate=16000)
        return np.asarray(wav, dtype=np.float32)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        wav = read_wav(os.path.join(self.audio_dir, r.filename), r.start, r.end)
        if self.augment:
            wav = self._augmented(np.asarray(wav, dtype=np.float32))
        # in the base weights' dtype: the Trainer's generate() for checkpoint
        # selection runs outside autocast, and a bf16 conv1d rejects fp32 input
        feats = self.proc.feature_extractor(wav, sampling_rate=16000,
                                            return_tensors='pt').input_features[0].to(self.dtype)
        ids = self.proc.tokenizer(r.text).input_ids
        return {'input_features': feats, 'labels': ids}


def collate(batch, decoder_start_token_id):
    feats = torch.stack([b['input_features'] for b in batch])
    n = max(len(b['labels']) for b in batch)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)   # -100 = ignored by the loss
    for i, b in enumerate(batch):
        labels[i, :len(b['labels'])] = torch.tensor(b['labels'])
    # The tokenizer prepends <|startoftranscript|>, and the model's
    # shift_tokens_right prepends decoder_start_token_id again -- same id. Drop
    # our copy or the decoder trains on a doubled start token.
    if (labels[:, 0] == decoder_start_token_id).all():
        labels = labels[:, 1:]
    return {'input_features': feats, 'labels': labels}


def take_budget(train_chunks, budget_min):
    """Nested by construction: every budget is a prefix of one ordering."""
    g = budget_order(train_chunks)
    return g[g.duration_s.cumsum() / 60 <= budget_min]


def cell_name(speaker, arm, site, method, budget, rank, lr, seed, tag=''):
    return (f's{speaker}_arm{arm}_{site}_{method}_b{budget}_r{rank}'
            f'_lr{lr:g}_seed{seed}' + (f'_{tag}' if tag else ''))


def train_cell(chunks, audio_dir, out_root, speaker, arm='B', site='both',
               method='lora', budget=30, rank=8, lr=None, seed=0,
               epochs=8, batch=8, grad_accum=1, timestamps=False,
               train_rows=None, dev_rows=None, grad_ckpt=None, num_workers=4,
               tag='', select='loss',
               max_steps=None, eval_steps=EVAL_STEPS, patience=None, dropout=None,
               augment='none', keep_adapter=True, min_epochs=None):
    """Train one cell. Returns the output dir; skips it if already finished.

    lr defaults per method (see DEFAULT_LR) because one value cannot serve both
    adapters and full fine-tuning. Resolved before cell_name, so the directory
    records the rate actually used.

    train_rows / dev_rows: pass the chunks explicitly instead of filtering
    `chunks` by speaker -- the cross-speaker control (D3) trains on a pool
    drawn from several speakers, under a string `speaker` label.
    grad_ckpt: None = only where it is needed (full fine-tuning, or a card
    under 40 GB). Checkpointing recomputes every activation in the backward
    pass: 823 -> 518 ms per LoRA step on an A100 without it, at 21 GB peak.
    tag: appended to the cell name (e.g. 'verbatim' when the targets are not the
    protocol), so variants of one recipe never collide.
    select: 'loss', the checkpoint with the lowest dev loss.  Selection on dev
    forgiven-shared WER ('forgiven', the second run's runs A and B) is removed with
    that count (docs/personalization_research.md § 1.5).

    Step mode (docs/training_plan_v3.md), off unless max_steps is given:
    max_steps: the same number of optimiser steps for every budget (epochs is then
    ignored); validation loss every eval_steps, constant lr after WARMUP_STEPS,
    the best checkpoint restored at the end.
    patience: stop after this many validations without improvement (None = run
    to max_steps).  Fixed at 4 in the plan, tuning runs included -- not tuned.
    dropout: Whisper's own `dropout` (0 in this checkpoint); LoRA's stays 0.05.
    augment: a key of AUGMENTS -- SpecAugment via the model config, tempo and
    noise on the training waveforms only.
    keep_adapter=False: delete the adapter weights once train_meta.json is written
    (tuning runs keep only their validation curve).
    """
    if method not in DEFAULT_LR:
        raise ValueError(f'unknown method {method}; expected one of '
                         f'{sorted(DEFAULT_LR)}')
    if site not in SITES:
        raise ValueError(f'unknown site {site}; expected one of {sorted(SITES)}')
    if lr is None:
        lr = DEFAULT_LR[method]

    out = os.path.join(out_root, cell_name(speaker, arm, site, method,
                                           budget, rank, lr, seed, tag))
    if os.path.exists(os.path.join(out, 'DONE')):
        print(f'skip (done): {out}')
        return out

    torch.manual_seed(seed)
    proc = WhisperProcessor.from_pretrained(ARMS[arm], language='he', task='transcribe')
    # Hold the timestamp decision constant across every cell: ivrit-ai trained
    # with timestamps, so dropping them can degrade timestamp behaviour.
    proc.tokenizer.set_prefix_tokens(language='he', task='transcribe',
                                     predict_timestamps=timestamps)
    # bf16 wherever the card has it (L4, A10G, A100 -- the plan's target VMs),
    # fp16 only where it does not (T4, the PoC box). They are not
    # interchangeable at this lr: fp16's narrow range is exactly where LoRA at
    # 1e-3 trips the gradient scaler, and bf16 needs no scaler at all.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    # Adapters: the frozen base lives in bf16 (PEFT keeps the adapter weights
    # in fp32), half the memory of fp32 and no per-op cast under autocast.
    # Full fine-tuning keeps fp32 master weights for the optimizer.
    base_dtype = torch.bfloat16 if (use_bf16 and method != 'full') else torch.float32
    aug = AUGMENTS[augment]
    # Whisper's layers read config.dropout when they are built, so it is set at load time
    cfg_kw = dict(dropout=dropout) if dropout is not None else {}
    if 'specaug' in aug:
        cfg_kw.update(SPECAUG)       # applied by WhisperModel._mask_input_features, in training mode only
    model = WhisperForConditionalGeneration.from_pretrained(ARMS[arm], dtype=base_dtype, **cfg_kw)
    dsti = model.config.decoder_start_token_id   # capture before the PEFT wrap

    # transformers 5: no forced_decoder_ids. Language/task live here, and an
    # empty suppress list keeps the loss off tokens we never forced.
    model.generation_config.language = 'he'
    model.generation_config.task = 'transcribe'
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False               # required with gradient checkpointing

    if train_rows is None:
        spk = chunks[chunks.speaker_id == speaker]
        tr = take_budget(spk[spk.part == 'train'], budget)
        dev = spk[spk.part == 'dev']
    else:
        tr, dev = train_rows, dev_rows
    print(f'{cell_name(speaker, arm, site, method, budget, rank, lr, seed, tag)}: '
          f'{len(tr)} train chunks ({tr.duration_s.sum()/60:.1f} min), {len(dev)} dev, select on {select}')

    if grad_ckpt is None:
        grad_ckpt = method == 'full' or not torch.cuda.is_available() or \
            torch.cuda.get_device_properties(0).total_memory < 40 * 2**30
    model.config.use_cache = not grad_ckpt         # generation on the dev set needs the cache; checkpointing forbids it
    if method == 'full':
        model.gradient_checkpointing_enable()
    else:
        from peft import LoraConfig, IA3Config, get_peft_model
        if method in ('lora', 'dora'):
            cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                             target_modules=SITES[site], use_dora=(method == 'dora'),
                             bias='none')
        elif method == 'ia3':
            cfg = IA3Config(target_modules=SITES[site], feedforward_modules=[])
        else:
            raise ValueError(f'unknown method {method}')
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        if grad_ckpt:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()   # else checkpointing yields no grads
    print(f'precision: {"bf16" if use_bf16 else "fp16" if use_fp16 else "fp32 (cpu)"}'
          f', base weights {base_dtype}, grad checkpointing {grad_ckpt}'
          f', lr {lr:g}, method {method}')

    if select != 'loss':
        raise ValueError(f"select={select!r}: only 'loss' remains (forgiven-WER selection was removed)")
    compute_metrics = None
    if max_steps:                    # step mode
        sched = dict(max_steps=max_steps, warmup_steps=WARMUP_STEPS, lr_scheduler_type='constant_with_warmup',
                     eval_strategy='steps', save_strategy='steps', eval_steps=eval_steps, save_steps=eval_steps)
    else:                            # epoch mode, as the second run ran
        # transformers 5 dropped warmup_ratio; warmup_steps takes a float in [0, 1) as the ratio
        sched = dict(num_train_epochs=epochs, warmup_steps=0.1, eval_strategy='epoch', save_strategy='epoch')
    args = Seq2SeqTrainingArguments(
        output_dir=out, per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum, learning_rate=lr,
        weight_decay=0.01, logging_steps=5, **sched,       # the training-loss curve: one point every 5 steps
        load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, save_total_limit=3,
        predict_with_generate=False, generation_max_length=448,
        bf16=use_bf16, fp16=use_fp16, report_to=[], seed=seed,
        remove_unused_columns=False, label_names=['labels'],
        per_device_eval_batch_size=2 * batch,
        # workers overlap the wav read + log-mel with the GPU step
        dataloader_num_workers=num_workers, dataloader_persistent_workers=num_workers > 0,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=ChunkDataset(tr, audio_dir, proc, base_dtype, augment=[a for a in aug if a != 'specaug']),
        eval_dataset=ChunkDataset(dev, audio_dir, proc, base_dtype),
        data_collator=lambda b: collate(b, dsti), compute_metrics=compute_metrics,
    )
    base_eval_loss = None
    if max_steps:
        from transformers import EarlyStoppingCallback
        # the untuned model's validation loss: the tuning criterion is the relative drop from it
        # (before the callback is added, so this evaluation cannot count toward patience)
        base_eval_loss = float(trainer.evaluate()['eval_loss'])
        if patience:
            if min_epochs:      # run 4's largest budgets: at least min_epochs passes before early stopping may stop
                min_steps = int(np.ceil(min_epochs * len(tr) / (batch * grad_accum)))
                trainer.add_callback(_min_steps_early_stopping(patience, min_steps))
            else:
                trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(out)
    proc.save_pretrained(out)
    # What the run did, for the results table: steps and the dev-loss curve.
    # Point 4 of the handoff's "Before the second run": passes are fixed at
    # `epochs`, so steps scale with the budget -- record them rather than hide it.
    st = trainer.state
    evals = [{k: h[k] for k in h if k.startswith('eval_') or k in ('epoch', 'step')} for h in st.log_history if 'eval_loss' in h]
    if max_steps:
        evals = [e for e in evals if e.get('step', 0) > 0]      # the step-0 evaluate() is base_eval_loss, not a checkpoint
    # Everything the analysis notebook needs about the run, beside the scores in
    # results.csv: the settings, the training-loss curve, the validation curve, cost.
    train_log = [{k: h[k] for k in ('step', 'epoch', 'loss', 'grad_norm', 'learning_rate') if k in h} for h in st.log_history if 'loss' in h]
    summary = next((h for h in st.log_history if 'train_runtime' in h), {})
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    settings = dict(speaker=str(speaker), arm=arm, model=ARMS[arm], site=site, method=method, budget=budget, rank=rank,
                    lora_alpha=2 * rank if method in ('lora', 'dora') else None, lora_dropout=0.05 if method in ('lora', 'dora') else None,
                    lr=lr, seed=seed, batch=batch, grad_accum=grad_accum, weight_decay=0.01,
                    schedule='constant_with_warmup' if max_steps else 'linear', warmup_steps=WARMUP_STEPS if max_steps else 0.1)
    meta_extra = dict(settings=settings, train_log=train_log, trainable_params=int(n_trainable),
                      train_runtime_s=summary.get('train_runtime'), train_loss_mean=summary.get('train_loss'),
                      peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None),
                      stopped_early=bool(max_steps and st.global_step < max_steps),
                      train_chunk_ids=list(tr.chunk_id), dev_chunk_ids=list(dev.chunk_id))
    meta = dict(global_step=st.global_step, epochs=(float(st.epoch) if max_steps else epochs), train_chunks=len(tr),
                train_minutes=float(tr.duration_s.sum() / 60), dev_chunks=len(dev),
                best_eval_loss=min((e['eval_loss'] for e in evals), default=None), evals=evals,
                select=select, best_epoch=(min(evals, key=lambda e: e['eval_loss'])['epoch'] if evals else None),
                base_dtype=str(base_dtype), grad_ckpt=bool(grad_ckpt), tag=tag,
                max_steps=max_steps, eval_steps=eval_steps if max_steps else None, patience=patience, dropout=dropout, min_epochs=min_epochs,
                augment=augment, base_eval_loss=base_eval_loss,
                best_step=(min(evals, key=lambda e: e['eval_loss']).get('step') if (evals and max_steps) else None),
                **meta_extra)
    import json, shutil, glob
    json.dump(meta, open(os.path.join(out, 'train_meta.json'), 'w'), indent=1)
    # The per-epoch checkpoints carry optimizer state (~3x the adapter) and are
    # only there for load_best_model_at_end, which has already run.  The
    # persistent volume on the GPU box has a 10 GB quota; drop them.
    for ck in glob.glob(os.path.join(out, 'checkpoint-*')):
        shutil.rmtree(ck, ignore_errors=True)
    if not keep_adapter:             # tuning: the curve in train_meta.json is the result
        for f in glob.glob(os.path.join(out, 'adapter_model.*')):
            os.remove(f)
    open(os.path.join(out, 'DONE'), 'w').close()
    return out


def overfit_check(chunks, audio_dir, speaker, arm='B', n=20, steps=60, batch=4):
    """Training sanity (docs/adaptation_plan.md § Verification): a correct setup drives a 20-example
    subset to near-zero loss. If this does not fall, nothing downstream is
    worth running.

    This is a hand-rolled loop, not a Trainer, so device placement and mixed
    precision are ours to do -- there is nothing here to do them for us.
    """
    spk = chunks[(chunks.speaker_id == speaker) & (chunks.part == 'train')].head(n)
    proc = WhisperProcessor.from_pretrained(ARMS[arm], language='he', task='transcribe')
    # Same prefix tokens train_cell uses. Without this the sanity check vouches
    # for a slightly different setup from the one it is meant to vouch for.
    proc.tokenizer.set_prefix_tokens(language='he', task='transcribe',
                                     predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(ARMS[arm])
    dsti = model.config.decoder_start_token_id
    model.generation_config.language, model.generation_config.task = 'he', 'transcribe'
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16,
                                             target_modules=SITES['both'], bias='none'))

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    # fp32 master weights with an autocast forward: 1.55 B in fp32 leaves a
    # 16 GB T4 no room for batch-4 activations. bf16 needs no loss scaler,
    # fp16 does -- and a T4 has no bf16, which is why both paths exist.
    amp = (torch.bfloat16 if dev == 'cuda' and torch.cuda.is_bf16_supported()
           else torch.float16 if dev == 'cuda' else None)
    scaler = torch.amp.GradScaler(dev, enabled=(amp is torch.float16))
    print(f'device {dev}' + (f', autocast {amp}' if amp is not None else '')
          + f', batch {batch}')

    ds = ChunkDataset(spk, audio_dir, proc)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     collate_fn=lambda b: collate(b, dsti))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    losses, it = [], iter(dl)
    for s in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        b = {k: v.to(dev) for k, v in b.items()}
        with (torch.autocast(dev, dtype=amp) if amp is not None
              else contextlib.nullcontext()):
            loss = model(**b).loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()
        losses.append(loss.item())
        if s % 10 == 0:
            print(f'  step {s:3d}  loss {loss.item():.4f}')
    print(f'first {np.mean(losses[:5]):.4f} -> last {np.mean(losses[-5:]):.4f}')
    return losses


def _self_check():
    from types import SimpleNamespace as NS
    assert recipe_tag(400, patience=4, data='hq') == 'hq_ms400_pat4'                       # run 3's cells keep their names
    assert recipe_tag(3000, patience=4, data='hq', min_epochs=1) == 'hq_ms3000_pat4_minep1'
    # the min-steps rule on a validation curve that stops improving at once: without it, stop at
    # the 4th validation (step 80); with min_steps 200, keep going and stop at the first
    # validation from step 200 on, since patience has long been used up
    def run(cb, losses):
        args = NS(metric_for_best_model='eval_loss', greater_is_better=False)
        state = NS(best_metric=None, global_step=0)
        for k, l in enumerate(losses, 1):
            state.global_step = 20 * k; control = NS(should_training_stop=False)
            cb.on_evaluate(args, state, control, {'eval_loss': l})
            if state.best_metric is None or l < state.best_metric: state.best_metric = l
            if control.should_training_stop: return state.global_step
        return None
    from transformers import EarlyStoppingCallback
    flat = [0.5] + [0.6] * 20
    assert run(EarlyStoppingCallback(early_stopping_patience=4), flat) == 100
    assert run(_min_steps_early_stopping(4, 200), flat) == 200
    late = [0.5] * 3 + [0.4] + [0.6] * 20                                                   # an improvement at step 80 resets the count
    assert run(_min_steps_early_stopping(4, 60), late) == 160
    print('train.py: OK')


if __name__ == '__main__':
    _self_check()
