"""Copy the run's small artefacts to a private HuggingFace dataset, every N minutes.

docs/training_handoff.md § Operational: a preempted pod loses everything, so the
results (outputs/results/*.json, results.csv) and the adapters (runs/<cell>/
adapter_*.safetensors, a few MB each at r=8) go to a private dataset on a
loop.  Trainer checkpoints (checkpoint-*/, optimizer state, ~3x the adapter)
are skipped: the finished adapter is what matters and the driver retrains a
cell whose DONE marker is missing.  The token comes from `hf auth login`
(huggingface_hub reads it); nothing here prints it.

    nohup python src/training/backup.py --repo knesset-asr/knesset-committees-adapters --every 30 &
    python src/training/backup.py --repo ... --once
"""
import argparse, glob, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = {'results': os.path.join(HERE, 'outputs', 'results'), 'runs': os.path.join(HERE, 'runs'),
           'runs_tune': os.path.join(HERE, 'runs_tune')}    # plan v3's tuning runs: train_meta.json only (adapters deleted)
FILES = {'results.csv': os.path.join(HERE, 'outputs', 'results.csv'), 'tuning.csv': os.path.join(HERE, 'outputs', 'tuning.csv')}
IGNORE = ['**/checkpoint-*/**', 'checkpoint-*/**', '**/checkpoint-*', '*.pt', '*.pth', 'optimizer*', 'scheduler*', 'rng_state*']

def log(msg):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + msg, flush=True)

def backup_once(repo, extra_logs=()):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    for name, folder in FOLDERS.items():
        if not os.path.isdir(folder):
            log(f'{name}: nothing yet'); continue
        api.upload_folder(folder_path=folder, path_in_repo=name, repo_id=repo, repo_type='dataset',
                          ignore_patterns=IGNORE, commit_message=f'backup {name} {time.strftime("%Y-%m-%d %H:%M")}')
        log(f'{name}: uploaded')
    # extra_logs may be glob patterns, expanded on every pass: a run's per-GPU worker logs appear as it goes
    logs = sorted({p for pat in extra_logs for p in (glob.glob(pat) or [pat])})
    for name, path in list(FILES.items()) + [(os.path.basename(p), p) for p in logs]:
        if os.path.exists(path):
            api.upload_file(path_or_fileobj=path, path_in_repo=f'logs/{name}' if path in logs else name,
                            repo_id=repo, repo_type='dataset', commit_message=f'backup {name}')
            log(f'{name}: uploaded')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True); ap.add_argument('--every', type=float, default=30, help='minutes')
    ap.add_argument('--once', action='store_true'); ap.add_argument('--logs', nargs='*', default=[])
    a = ap.parse_args()
    while True:
        try:
            backup_once(a.repo, a.logs)
        except Exception as e:                       # a transient HF error must not kill the loop
            log(f'backup failed: {type(e).__name__}: {str(e)[:200]}')
        if a.once: break
        time.sleep(a.every * 60)
