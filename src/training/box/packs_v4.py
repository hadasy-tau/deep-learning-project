"""Run 4's audio as 14 packs instead of 44,033 files, so the GPU pod boots in minutes.

Measured 2026-10-04: a snapshot of knesset-committees-panel-hq's WAVs runs at ~5 files/s
whatever the bandwidth (per-file overhead), i.e. ~2.3 h for run 4's clips while the GPUs
wait.  So the clips also go up as one tar per speaker of FLAC-encoded clips (lossless: the
16-bit PCM comes back bit for bit), and the pod decodes them back to the WAVs common.read_wav
reads.  The WAVs stay in the dataset as the browsable copy.

    python src/training/box/packs_v4.py pack                 # outputs/panel_packs/<speaker>.tar from both v4 plans
    python src/training/box/packs_v4.py upload               # the packs to panel-hq under packs/, sizes checked
    python src/training/box/packs_v4.py fetch                # GPU pod: download the packs, decode to panel_audio/
    python src/training/box/packs_v4.py --self-check
"""
import argparse, io, os, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TRAINING = os.path.dirname(HERE)
AUDIO = os.path.join(TRAINING, 'outputs', 'panel_audio')
PACKS = os.path.join(TRAINING, 'outputs', 'panel_packs')
PLANS = [os.path.join(TRAINING, 'panel_plan_v4.parquet'), os.path.join(TRAINING, 'panel_test07_v4.parquet')]
REPO = 'knesset-asr/knesset-committees-panel-hq'
SR = 16000


def wav_to_flac(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype='int16'); assert sr == SR, path
    b = io.BytesIO(); sf.write(b, x, SR, format='FLAC', subtype='PCM_16'); return b.getvalue()


def flac_to_wav(data, path):
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(data), dtype='int16'); assert sr == SR, path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path + '.part', x, SR, format='WAV', subtype='PCM_16'); os.replace(path + '.part', path)


def _pack_one(args):
    sid, files = args
    out = os.path.join(PACKS, f'{sid}.tar')
    with tarfile.open(out + '.part', 'w') as tf:
        for f in files:
            data = wav_to_flac(os.path.join(AUDIO, f))
            ti = tarfile.TarInfo(f[:-4] + '.flac'); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    os.replace(out + '.part', out)
    return sid, len(files), os.path.getsize(out)


def pack(workers=None):
    P = pd.concat([pd.read_parquet(p) for p in PLANS]).drop_duplicates('chunk_id')
    os.makedirs(PACKS, exist_ok=True)
    jobs = [(int(s), sorted(set(g.filename))) for s, g in P.groupby('speaker_id')]
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for sid, n, size in ex.map(_pack_one, sorted(jobs, key=lambda j: -len(j[1]))):
            print(f'{sid}: {n:,} clips, {size / 2**30:.2f} GB', flush=True)


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_large_folder(repo_id=REPO, repo_type='dataset', folder_path=os.path.dirname(PACKS), allow_patterns=['panel_packs/*.tar'])
    remote = {f.path: f.size for f in api.list_repo_tree(REPO, path_in_repo='panel_packs', repo_type='dataset') if hasattr(f, 'size')}
    bad = [f for f in os.listdir(PACKS) if f.endswith('.tar') and remote.get(f'panel_packs/{f}') != os.path.getsize(os.path.join(PACKS, f))]
    print(f'{len(remote)} packs in {REPO}' + (f'; missing or different: {bad}' if bad else ', sizes match'))
    sys.exit(1 if bad else 0)


def _unpack_one(path):
    n = 0
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile(): continue
            flac_to_wav(tf.extractfile(m).read(), os.path.join(AUDIO, m.name[:-5] + '.wav')); n += 1
    os.remove(path)
    return os.path.basename(path), n


def fetch(workers=None):
    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download(REPO, repo_type='dataset', local_dir=os.path.dirname(PACKS), allow_patterns=['panel_packs/*.tar'], max_workers=16)
    print(f'packs downloaded in {(time.time() - t) / 60:.1f} min', flush=True)
    tars = sorted((os.path.join(PACKS, f) for f in os.listdir(PACKS) if f.endswith('.tar')), key=os.path.getsize, reverse=True)
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for name, n in ex.map(_unpack_one, tars): print(f'{name}: {n:,} wavs', flush=True)
    print(f'audio ready in {(time.time() - t) / 60:.1f} min', flush=True)


def _self_check():
    import tempfile, soundfile as sf
    d = tempfile.mkdtemp(); rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 2) * 3000).astype(np.int16)
    src, dst = os.path.join(d, 'a.wav'), os.path.join(d, 'b', 'a.wav')
    sf.write(src, x, SR, format='WAV', subtype='PCM_16')
    data = wav_to_flac(src); flac_to_wav(data, dst)
    sys.path.insert(0, TRAINING); sys.path.insert(0, os.path.dirname(TRAINING))
    from common import read_wav
    assert np.array_equal(read_wav(src), read_wav(dst)), 'FLAC round trip changed the audio'
    assert len(data) < os.path.getsize(src)
    print('packs_v4 self-check: OK (bit-identical round trip)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', nargs='?', choices=['pack', 'upload', 'fetch'])
    ap.add_argument('--workers', type=int); ap.add_argument('--self-check', action='store_true')
    a = ap.parse_args()
    if a.self_check: _self_check(); sys.exit(0)
    if a.cmd == 'pack': pack(a.workers)
    elif a.cmd == 'upload': upload()
    elif a.cmd == 'fetch': fetch(a.workers)
    else: ap.error('give a command or --self-check')
