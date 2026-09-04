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
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from pipeline import read_wav, budget_order

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
}


class ChunkDataset(Dataset):
    """Chunks -> (log-mel features, label ids). Audio is sliced out of the
    whole-recording wavs that `materialize` wrote."""

    def __init__(self, chunks, audio_dir, processor):
        self.rows = chunks.reset_index(drop=True)
        self.audio_dir, self.proc = audio_dir, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        wav = read_wav(os.path.join(self.audio_dir, r.filename), r.start, r.end)
        feats = self.proc.feature_extractor(wav, sampling_rate=16000,
                                            return_tensors='pt').input_features[0]
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


def cell_name(speaker, arm, site, method, budget, rank, lr, seed):
    return (f's{speaker}_arm{arm}_{site}_{method}_b{budget}_r{rank}'
            f'_lr{lr:g}_seed{seed}')


def train_cell(chunks, audio_dir, out_root, speaker, arm='B', site='both',
               method='lora', budget=30, rank=8, lr=1e-3, seed=0,
               epochs=8, batch=8, grad_accum=1, timestamps=False):
    """Train one cell. Returns the output dir; skips it if already finished."""
    out = os.path.join(out_root, cell_name(speaker, arm, site, method,
                                           budget, rank, lr, seed))
    if os.path.exists(os.path.join(out, 'DONE')):
        print(f'skip (done): {out}')
        return out

    torch.manual_seed(seed)
    proc = WhisperProcessor.from_pretrained(ARMS[arm], language='he', task='transcribe')
    # Hold the timestamp decision constant across every cell: ivrit-ai trained
    # with timestamps, so dropping them can degrade timestamp behaviour.
    proc.tokenizer.set_prefix_tokens(language='he', task='transcribe',
                                     predict_timestamps=timestamps)
    model = WhisperForConditionalGeneration.from_pretrained(ARMS[arm])
    dsti = model.config.decoder_start_token_id   # capture before the PEFT wrap

    # transformers 5: no forced_decoder_ids. Language/task live here, and an
    # empty suppress list keeps the loss off tokens we never forced.
    model.generation_config.language = 'he'
    model.generation_config.task = 'transcribe'
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False               # required with gradient checkpointing

    spk = chunks[chunks.speaker_id == speaker]
    tr = take_budget(spk[spk.part == 'train'], budget)
    dev = spk[spk.part == 'dev']
    print(f'{cell_name(speaker, arm, site, method, budget, rank, lr, seed)}: '
          f'{len(tr)} train chunks ({tr.duration_s.sum()/60:.1f} min), {len(dev)} dev')

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
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()       # else checkpointing yields no grads

    args = Seq2SeqTrainingArguments(
        output_dir=out, per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum, learning_rate=lr,
        num_train_epochs=epochs, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy='epoch', save_strategy='epoch', logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, save_total_limit=3,
        fp16=torch.cuda.is_available(), report_to=[], seed=seed,
        remove_unused_columns=False, label_names=['labels'],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=ChunkDataset(tr, audio_dir, proc),
        eval_dataset=ChunkDataset(dev, audio_dir, proc),
        data_collator=lambda b: collate(b, dsti),
    )
    trainer.train()
    trainer.save_model(out)
    proc.save_pretrained(out)
    open(os.path.join(out, 'DONE'), 'w').close()
    return out


def overfit_check(chunks, audio_dir, speaker, arm='B', n=20, steps=60, batch=4):
    """Plan section 08 training sanity: a correct setup drives a 20-example
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
