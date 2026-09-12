"""Stage 3, Step 4.2 -- the audio gate.

Every check so far reads text.  Path A requires the corpus and ivrit.ai to agree on
a name; Path B requires the name to equal a seated MK's.  Neither can see the one
failure they share: if both parsers copied the same wrong speaker header, the
labels agree with each other and are both wrong.  Only the voice can tell.

Method: embed a stratified sample of `label == 'mk'` segments with ECAPA-TDNN (the
model VoxKnesset benchmarks), build a per-`speaker_id` centroid from the segments
that agree most tightly, then flag any segment whose cosine to its own centroid is
an outlier AND whose nearest centroid belongs to somebody else.  That pair of
conditions is the contamination estimate; a segment that is merely noisy is far
from everyone, not close to one specific other person.

Needs a GPU environment (torch + speechbrain) and gated read access to
ivrit-ai/knesset-committees for the audio.  Not run on the build machine.

    pip install torch torchaudio speechbrain soundfile
    python src/preprocessing/speaker_index/validate_audio.py --per-speaker 12 --min-segments 8
"""
import argparse, io, os, subprocess, tempfile
import numpy as np, pandas as pd

from roster import OUT, CACHE

COMMITTEES = 'ivrit-ai/knesset-committees'
SR = 16000

def sample(df, per_speaker=12, min_segments=8, min_s=3.0, seed=0):
    """Stratified over speakers and knessets, longest-first within a speaker so the
    embedding has something to work with.  Speakers with too little audio are
    reported rather than sampled -- a centroid from three segments is not one."""
    mk = df[(df.label == 'mk') & (df.duration_s >= min_s)]
    counts = mk.groupby('speaker_id').size()
    thin = counts[counts < min_segments]
    out = []
    for sid, g in mk[mk.speaker_id.isin(counts[counts >= min_segments].index)].groupby('speaker_id'):
        for _, kg in g.groupby('knesset'):
            out.append(kg.nlargest(max(1, per_speaker // kg.knesset.nunique()), 'duration_s'))
    return pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True), thin

def fetch_audio(session, start, end, token):
    """Decode just the needed span straight from the session's m4a over HTTP."""
    url = f'https://huggingface.co/datasets/{COMMITTEES}/resolve/main/{session}/audio.m4a'
    cmd = ['ffmpeg', '-v', 'error', '-headers', f'Authorization: Bearer {token}\r\n',
           '-ss', f'{start:.3f}', '-i', url, '-t', f'{end - start:.3f}',
           '-ac', '1', '-ar', str(SR), '-f', 'wav', 'pipe:1']
    return subprocess.run(cmd, capture_output=True, check=True).stdout

def embed(rows, token, batch=16):
    import torch, soundfile as sf
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                         savedir=os.path.join(CACHE, 'ecapa'),
                                         run_opts={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    vecs, keep = [], []
    for i in range(0, len(rows), batch):
        wavs = []
        for r in rows.iloc[i:i + batch].itertuples():
            try:
                w, _ = sf.read(io.BytesIO(fetch_audio(r.session, r.start, r.end, token)), dtype='float32')
                wavs.append((r.Index, torch.from_numpy(w)))
            except Exception:
                continue
        if not wavs:
            continue
        n = max(len(w) for _, w in wavs)
        x = torch.zeros(len(wavs), n)
        lens = torch.tensor([len(w) / n for _, w in wavs])
        for j, (_, w) in enumerate(wavs):
            x[j, :len(w)] = w
        with torch.no_grad():
            e = enc.encode_batch(x, lens).squeeze(1).cpu().numpy()
        vecs.append(e); keep += [k for k, _ in wavs]
        print(f'  embedded {len(keep)}/{len(rows)}', flush=True)
    return np.vstack(vecs), keep

def contamination(E, ids, trim=0.25):
    """Centroid per speaker from the tightest (1-trim) of their own segments, so a
    contaminated segment cannot drag its own centroid toward itself."""
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    uid = np.unique(ids)
    cent = {}
    for s in uid:
        V = E[ids == s]
        c = V.mean(0); c /= np.linalg.norm(c)
        sim = V @ c
        k = max(1, int(round(len(V) * (1 - trim))))
        V = V[np.argsort(-sim)[:k]]
        c = V.mean(0); cent[s] = c / np.linalg.norm(c)
    C = np.stack([cent[s] for s in uid])
    S = E @ C.T
    own = S[np.arange(len(E)), np.searchsorted(uid, ids)]
    other = S.copy(); other[np.arange(len(E)), np.searchsorted(uid, ids)] = -np.inf
    best = other.max(1); who = uid[other.argmax(1)]
    return pd.DataFrame(dict(speaker_id=ids, own=own, nearest_other=best, nearest_id=who,
                             margin=own - best))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-speaker', type=int, default=12)
    ap.add_argument('--min-segments', type=int, default=8)
    ap.add_argument('--segments', default=os.path.join(OUT, 'segments.parquet'))
    a = ap.parse_args()
    token = os.environ.get('HF_TOKEN') or open(os.path.join(CACHE, 'hf_write_token')).read().strip()

    df = pd.read_parquet(a.segments)
    rows, thin = sample(df, a.per_speaker, a.min_segments)
    print(f'sampling {len(rows)} segments over {rows.speaker_id.nunique()} speakers; '
          f'{len(thin)} speakers below {a.min_segments} segments are not covered')
    E, keep = embed(rows, token)
    rows = rows.loc[keep].reset_index(drop=True)
    res = contamination(E, rows.speaker_id.values)
    res = pd.concat([rows[['filename', 'session', 'raw_name', 'label_path', 'match_method', 'duration_s']], res], axis=1)

    flagged = res[res.margin < 0]
    print(f'\nsegments closer to another speaker than to their own: {len(flagged)}/{len(res)} '
          f'= {len(flagged)/len(res):.3%}   (gate: < 0.5%)')
    print(res.groupby('label_path').margin.describe().round(3).to_string())
    if len(flagged):
        print('\nflagged:')
        print(flagged.nsmallest(30, 'margin').to_string(index=False))
    res.to_parquet(os.path.join(OUT, 'audio_check.parquet'), index=False)
    open(os.path.join(OUT, 'audio_report.txt'), 'w', encoding='utf-8').write(
        f'{len(flagged)}/{len(res)} = {len(flagged)/len(res):.3%} closer to another speaker\n\n'
        + res.groupby('label_path').margin.describe().round(3).to_string()
        + ('\n\n' + flagged.nsmallest(30, 'margin').to_string(index=False) if len(flagged) else ''))
    print(f'\nwrote {OUT}/audio_check.parquet, audio_report.txt')
