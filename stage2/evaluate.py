"""Stage 2 scoring: transcribe, count errors, compare.

Covers the plan's stages 05 (baseline), 07 (eval) and 08 (stats). Baseline and
tuned models go through the identical path -- that is the point, and it is what
makes the like-for-like gate in section 08 meaningful.

Error COUNTS are stored, never rates, so speaker WER stays
total_errors / total_words however the rows are later grouped.
"""
import os
import numpy as np, pandas as pd, torch
from rapidfuzz.distance import Levenshtein
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pipeline import read_wav, normalize_he

ARMS = {'A': 'openai/whisper-large-v3', 'B': 'ivrit-ai/whisper-large-v3'}


def load(arm='B', adapter=None, device=None):
    """Base model, optionally with a trained adapter merged on top."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    proc = WhisperProcessor.from_pretrained(ARMS[arm], language='he', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(
        ARMS[arm], dtype=torch.float16 if device == 'cuda' else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.generation_config.language = 'he'
    model.generation_config.task = 'transcribe'
    return model.to(device).eval(), proc, device


@torch.no_grad()
def transcribe_short(model, proc, chunks, audio_dir, batch=8, device='cpu'):
    """Short-form: one <=30 s chunk per example. D7's primary protocol."""
    out = []
    for i in range(0, len(chunks), batch):
        rows = chunks.iloc[i:i + batch]
        wavs = [read_wav(os.path.join(audio_dir, r.filename), r.start, r.end)
                for r in rows.itertuples()]
        feats = proc.feature_extractor(wavs, sampling_rate=16000,
                                       return_tensors='pt').input_features.to(device, model.dtype)
        ids = model.generate(feats, language='he', task='transcribe')
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return out


@torch.no_grad()
def transcribe_long(model, proc, filenames, audio_dir, device='cpu'):
    """Long-form: the whole recording in one pass, so Whisper's own sequential
    algorithm handles the >30 s audio. D7's secondary protocol -- and the only
    one comparable with Stage 1, which scored whole recordings."""
    out = []
    for fn in filenames:
        wav = read_wav(os.path.join(audio_dir, fn))
        inputs = proc(wav, sampling_rate=16000, return_tensors='pt', truncation=False,
                      padding='longest', return_attention_mask=True)
        inputs = {k: (v.to(device, model.dtype) if v.is_floating_point() else v.to(device))
                  for k, v in inputs.items()}
        ids = model.generate(**inputs, language='he', task='transcribe',
                             return_timestamps=True)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    return out


def score(refs, hyps):
    """Per-segment error counts + S/D/I breakdown. Counts, not rates.

    The S/D/I split is not decoration: if gains come mostly from fewer
    insertions the model learned protocol style, not the speaker (section 08).
    """
    rows = []
    for ref, hyp in zip(refs, hyps):
        r, h = normalize_he(ref).split(), normalize_he(hyp).split()
        S = D = I = 0
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(r, h):
            if tag == 'replace':
                # an unequal replace block is min() substitutions plus the
                # leftover as deletions/insertions -- max() would total
                # correctly but dump the excess into S and blur the breakdown
                n = min(i2 - i1, j2 - j1)
                S += n
                D += (i2 - i1) - n
                I += (j2 - j1) - n
            elif tag == 'delete':
                D += i2 - i1
            elif tag == 'insert':
                I += j2 - j1
        rows.append(dict(n_words=len(r), n_chars=len(' '.join(r)),
                         werr=S + D + I, cerr=Levenshtein.distance(' '.join(r), ' '.join(h)),
                         S=S, D=D, I=I,
                         # Stage 1's runaway flag: a looping decoder, which
                         # fine-tuned Whisper is known to regress into (D5).
                         runaway=len(h) > 3 * max(len(r), 1)))
    return pd.DataFrame(rows)


def wer(df):
    return df.werr.sum() / df.n_words.sum()


def paired_bootstrap(base, tuned, n_boot=2000, seed=0):
    """Base vs tuned on the SAME segments. Resamples segments, not words.

    stage0_gate predicted what this can resolve; this measures it for real.
    """
    assert len(base) == len(tuned), 'paired comparison needs the same segments'
    rng = np.random.default_rng(seed)
    eb, et, w = base.werr.values, tuned.werr.values, base.n_words.values
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))
    W = w[idx].sum(1)
    delta = (eb[idx].sum(1) - et[idx].sum(1)) / W          # >0 means tuned is better
    obs = (eb.sum() - et.sum()) / w.sum()
    lo, hi = np.percentile(delta, [2.5, 97.5])
    from scipy.stats import wilcoxon
    if np.array_equal(eb, et):                 # no differences: nothing to test
        p_w = 1.0
    else:
        p_w = wilcoxon(eb, et).pvalue
        if not np.isfinite(p_w):
            p_w = 1.0
    # two-sided bootstrap p, capped -- doubling the smaller tail can exceed 1
    p_boot = min(2 * min((delta <= 0).mean(), (delta >= 0).mean()), 1.0)
    return dict(wer_base=eb.sum() / w.sum(), wer_tuned=et.sum() / w.sum(),
                delta_abs=obs, delta_rel=obs / (eb.sum() / w.sum()),
                ci_lo=lo, ci_hi=hi, p_boot=p_boot,
                p_wilcoxon=p_w, n_segments=len(w), n_words=int(w.sum()))


def stage1_wer(index, filenames):
    """Stage-1 arm-B long-form WER over EXACTLY these recordings.

    The ct2 dump's `model_transcription` IS Stage 1's arm-B output, so the
    Stage-1 number for any subset of recordings is recomputable from the index
    and no Stage-1 artifact is needed. Restricting to the personal-test
    recordings is what makes the gate below like-for-like: a speaker's
    published wer_B covers ALL their recordings, most of which the gate never
    re-scores, so comparing against it would fail on set difference alone.
    """
    want = set(filenames)
    rows = index[index.filename.isin(want)]
    missing = want - set(rows.filename)
    assert not missing, f'{len(missing)} recording(s) absent from the index'
    return wer(score(rows.reference_text.tolist(), rows.model_transcription.tolist()))


def like_for_like(long_counts, index, filenames, tol=0.02):
    """Section 08 hard gate. Stage 1 measured LONG-form WER over whole
    recordings, so personal-test must be re-scored long-form with the base
    model and reproduce Stage 1 ON THE SAME RECORDINGS. Comparing Stage-1
    long-form against Stage-2 chunk-level WER would fail for reasons that have
    nothing to do with a bug -- so this must pass before any tuned model is
    believed.

    `long_counts` is score() over transcribe_long(..., filenames); the target
    is recomputed from the index over that same file set. What is left between
    the two is runtime only -- Stage 1 decoded through CTranslate2, this decodes
    through transformers -- and that is what `tol` absorbs.
    """
    target = stage1_wer(index, filenames)
    got = wer(long_counts)
    ok = abs(got - target) <= tol
    print(f'like-for-like over {len(set(filenames))} recording(s): '
          f'stage1 {target:.4f} vs long-form base {got:.4f} '
          f'-> {"OK" if ok else "FAIL"}')
    if not ok:
        raise AssertionError('chunking or normalization changed the metric; '
                             'every later comparison would be invalid')
    return got


# ---- self-check (no model, no audio) --------------------------------------
if __name__ == '__main__':
    ref = ['a b c d', 'a b c', 'a b c', 'a b c', 'a b c']
    hyp = ['a b c d',        # perfect
           'a x c',          # 1 substitution
           'a c',            # 1 deletion
           'a b z c',        # 1 insertion
           'a b c ' * 9]     # runaway decoder
    s = score(ref, hyp)
    assert list(s.werr) == [0, 1, 1, 1, 24], list(s.werr)
    assert list(s.S) == [0, 1, 0, 0, 0] and list(s.D) == [0, 0, 1, 0, 0]
    assert list(s.I) == [0, 0, 0, 1, 24], list(s.I)
    assert list(s.runaway) == [False] * 4 + [True]
    # every error must be exactly one of S, D or I
    assert (s.werr == s.S + s.D + s.I).all()
    # unequal replace: 2 words -> 3 is 1 sub + 1 ins, not 2 subs
    u = score(['a b c'], ['a x y c'])
    assert (u.S[0], u.I[0], u.D[0]) == (1, 1, 0), u[['S', 'D', 'I']].to_dict('records')
    print('score(): OK')

    base = score(['a b c d e'] * 200, ['a x c d e'] * 200)     # 1 error each
    tuned = score(['a b c d e'] * 200, ['a b c d e'] * 200)    # 0 errors
    r = paired_bootstrap(base, tuned)
    assert r['delta_abs'] > 0 and r['p_boot'] < 0.05, r
    assert abs(r['delta_rel'] - 1.0) < 1e-9, r
    same = paired_bootstrap(base, base)
    assert abs(same['delta_abs']) < 1e-12 and same['p_boot'] > 0.9, same
    print(f"paired_bootstrap(): OK  (delta_rel {r['delta_rel']:.3f}, "
          f"p {r['p_boot']:.4f}; null p {same['p_boot']:.3f})")

    # stage1_wer() must read its target off the SAME recordings the gate
    # re-scores. Taking it from a speaker's whole corpus instead is the bug
    # this exists to prevent, so the two must be visibly different here.
    idx = pd.DataFrame({'filename': ['a.wav', 'b.wav', 'c.wav'],
                        'reference_text': ['a b c d e'] * 3,
                        'model_transcription': ['a x c d e', 'a x c d e',
                                                'q q q q q']})
    assert abs(stage1_wer(idx, ['a.wav', 'b.wav']) - 2 / 10) < 1e-12
    assert abs(stage1_wer(idx, ['a.wav', 'b.wav', 'c.wav']) - 7 / 15) < 1e-12

    lf = score(['a b c d e'] * 2, ['a x c d e'] * 2)      # a.wav + b.wav, rescored
    assert abs(like_for_like(lf, idx, ['a.wav', 'b.wav']) - 0.2) < 1e-12
    try:
        like_for_like(lf, idx, ['a.wav', 'b.wav', 'c.wav'])   # wrong file set
    except AssertionError:
        pass
    else:
        raise SystemExit('like_for_like accepted a mismatched file set')
    print('stage1_wer() / like_for_like(): OK')
