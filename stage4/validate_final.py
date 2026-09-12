"""Stage 4 final validation: the merged inference dataset is complete and
correct for the Stage-1 subset, and the pipeline produced no silent errors.

    python stage4/validate_final.py           # exits non-zero on any failure

Checks, each independent:
  coverage   every subset chunk has a successful A row and a successful B row
  no errors  no subset chunk is left with an error and no success, for either arm
  identity   each row's speaker/session/duration match the index (nothing mislabelled)
  reference  the reference text equals the chunk's `text` in the source dataset
  hypothesis non-empty, Hebrew, and not the reference verbatim on more than a
             trivial share (a leaked reference would look like a perfect model)
  audio      the JSONL's duration_s matches the FLAC decoded from the source shard,
             on a random sample -- the audio sent was the audio indexed
  scoring    Stage 1's WER on the merged table is finite and B beats A on the
             corpus, as on the 10-chunk verification
"""
import io, json, os, sys, random
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stage2'))
import data as D
from pipeline import normalize_he
from rapidfuzz.distance import Levenshtein
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'outputs')
F = []
def chk(name, cond, detail=''):
    print(f'  [{"OK " if cond else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
    if not cond: F.append(name)

w = pd.read_parquet(os.path.join(OUT, 'inference.parquet'))
long = pd.read_parquet(os.path.join(OUT, 'inference_long.parquet'))
sub = pd.read_parquet(os.path.join(HERE, 'subset_stage1.parquet'))
idx = D.index().set_index('chunk_id')
S = set(sub.chunk_id); ws = w[w.chunk_id.isin(S)].set_index('chunk_id')

print('=== coverage ===')
chk('every subset chunk present in the merged table', len(ws) == len(S), f'{len(ws):,}/{len(S):,}')
chk('every subset chunk has hypothesis_A', ws.has_A.all(), f'missing {int((~ws.has_A).sum())}')
chk('every subset chunk has hypothesis_B', ws.has_B.all(), f'missing {int((~ws.has_B).sum())}')
ls = long[long.chunk_id.isin(S)]
pend = ls[ls.error.notna()]
chk('no subset chunk left with an unrecovered error', pend.empty, f'{len(pend)} rows: ' + str(pend.error.str[:60].value_counts().to_dict())[:200])
chk('one row per (chunk, arm)', not ls.duplicated(['chunk_id','arm']).any())

print('=== identity ===')
j = ws.join(idx[['speaker_id','session','duration_s','text']], rsuffix='_idx')
chk('speaker_id matches the index', (j.speaker_id == j.speaker_id_idx).all())
chk('session matches the index', (j.session == j.session_idx).all())
chk('duration_s matches the index', ((j.duration_s - j.duration_s_idx).abs() < 1e-3).all())
chk('reference == source text', (j.reference == j.text).all(), f'{int((j.reference != j.text).sum())} differ')
chk('speakers: all 267 of the subset', ws.speaker_id.nunique() == sub.speaker_id.nunique(), f'{ws.speaker_id.nunique()}')
chk('models: A is openai/whisper-large-v3, B is ivrit-ai/whisper-large-v3-turbo-ct2',
    set(ws.model_A.dropna()) == {'openai/whisper-large-v3'} and set(ws.model_B.dropna()) == {'ivrit-ai/whisper-large-v3-turbo-ct2'})

print('=== hypotheses ===')
for arm in 'AB':
    h = ws[f'hypothesis_{arm}'].fillna('')
    chk(f'{arm}: non-empty', (h.str.strip() != '').all(), f'{int((h.str.strip()=="").sum())} empty')
    chk(f'{arm}: Hebrew', h.str.contains(r'[א-ת]').mean() > 0.99, f'{h.str.contains(r"[א-ת]").mean():.3%}')
    same = (h.map(normalize_he) == ws.reference.map(normalize_he)).mean()
    chk(f'{arm}: not the reference leaked back (verbatim-equal share small)', same < 0.05, f'{same:.2%} identical after normalisation')

print('=== audio (sample of 20: the bytes sent were the bytes indexed) ===')
import soundfile as sf
samp = ws.sample(20, random_state=0)
rows = idx.loc[samp.index].reset_index()
ok = 0
for r, b in D.audio_iter(rows.assign(shard=rows.shard)):
    d = sf.info(io.BytesIO(b)); ok += abs(d.frames/d.samplerate - r.duration_s) < 0.05
chk('sampled FLAC durations match', ok == 20, f'{ok}/20')

print('=== scoring (Stage 1 method) ===')
def wer(hyp):
    ref = ws.reference.map(normalize_he); h = hyp.fillna('').map(normalize_he)
    e = sum(Levenshtein.distance(r.split(), x.split()) for r, x in zip(ref, h)); n = ref.str.split().str.len().sum()
    return e / n
wa, wb = wer(ws.hypothesis_A), wer(ws.hypothesis_B)
chk('WER finite and sane', 0 < wa < 1.5 and 0 < wb < 1.5, f'A {wa:.4f}  B {wb:.4f}')
chk('B (Hebrew fine-tune) beats A on the corpus', wb < wa, f'gain {(wa-wb)/wa:.1%}')
sp = ws.groupby('speaker_id').size(); chk('every speaker has >= 20 chunks scored', sp.min() >= 20, f'min {sp.min()}')

print()
print(f'SUBSET: {len(ws):,} chunks, {ws.duration_s.sum()/3600:.1f} h, {ws.speaker_id.nunique()} speakers | WER A {wa:.4f}  B {wb:.4f}')
print('RESULT:', 'ALL CHECKS PASSED' if not F else f'FAILED: {F}')
sys.exit(1 if F else 0)
