# Pod runbook: plan v3 on a RunPod GPU

For whoever runs the training, a person or a fresh Claude session on the pod. It is the
*how*. The *why* is `docs/training_plan_v3.md` § 1–5; read those first if you have not.
`docs/training_run2.md` is what the last run found, and why every rule below exists.

Everything runs through `src/training/box/pod_v3.sh <stage>`. Each stage is idempotent: a
scored cell, a finished adapter, a tuning run and a cached base transcription are skipped on a
rerun. After a crash or a preemption, run the same stage again. Every stage logs to
`src/training/outputs/logs/<stage>.log`, and each GPU's share of it to `<stage>.gpu<k>.log`.

## How many GPUs: four A100s, in one pod

The run is about 230 independent jobs of ~3 minutes each (train an adapter, or transcribe a
test set). That totals ~10.5 GPU-hours. RunPod bills GPU-hours, so the same work costs about
the same on 1 GPU for 10.5 hours or on 4 GPUs for 3 hours. `pod_v3.sh` uses every GPU the pod has
(`GPUS=N` to override) and splits the work so no two processes write the same file:

- **tuning:** one tuning speaker per GPU;
- **own adapters:** the speakers in round-robin shares;
- **controls:** one (seed, budget) per GPU, and only after every own adapter is done. A
  control is always trained on *all* the panel's speakers, never on one worker's share.

| pod | $/h (2026-10-02) | wall time | cost | |
|---|---|---|---|---|
| 1× A100 SXM 80 GB | 1.59 | ~10.5 h | ~$17 | the measured baseline |
| 2× A100 SXM 80 GB | 3.18 | ~5.5 h | ~$18 | |
| **4× A100 SXM 80 GB** | **6.36** | **~3 h** | **~$19** | **recommended**: tuning runs exactly 4 speakers at once, and the 12 speakers split 3 per GPU |
| H100 SXM | 3.49 | not measured | more | batch-8 jobs, with CPU-side audio loading, do not use its extra compute |
| A40 / L4 / RTX A5000 | 0.27–0.49 | not measured | similar or more | ~2.5–3× slower per job; cheaper per hour, not per job |

Only the A100 is measured (`docs/training_run2.md`): 0.52 s a training step at batch 8,
21 GB peak, 24 s for a 224-chunk test set at `--eval-batch 64`. A 48 GB card would fit, but
nothing here has timed one.

## What this run does, in one table

| stage | what | reads the test set? | time (A100 80 GB) |
|---|---|---|---|
| `setup` | deps, GPU / torch / HF-login checks, model weights, self-checks | no | ~10 min |
| `data` | panel audio from HF; any clip missing is extracted from the corpus; both plans verified | no | minutes (≤ 1 h if HF is incomplete) |
| `backup` | mirrors results, adapters, tuning curves, logs to `knesset-asr/knesset-committees-v3-results` every 30 min | — | background |
| `sanity` | overfit check (loss must fall) + one timed tuning run | no | ~5 min |
| `tune` | stages A (lr), B (rank × dropout), C (augmentation) on 4 speakers × 5 / 80 min; each picked by rule → `outputs/recipe_v3.json` | **no** — validation only | ~3.2 h |
| `base` | base B on each speaker's high-quality test vs the error map: the hard-stop check | base model only | ~0.3 h |
| `final` | 12 speakers × 5 / 20 / 80 min, seed 0; the 2-fold control at each budget; both test sets (the run also scored the forgiven count, since withdrawn) | yes, once | ~4 h |
| `seeds` | seeds 1 and 2 at 80 min, own adapters and their controls | yes | ~2.5 h |
| `summary` | `outputs/results.csv`, a last backup | — | seconds |

About 10.5 GPU-hours, ~$17–19 at A100 SXM's $1.59/h. The times above are GPU time; divide
`tune`, `base`, `final` and `seeds` by the number of GPUs for wall time (controls by 3 and 2,
since those stages have 3 and 2 control jobs).

## Before the pod (on the laptop)

- [ ] `main` has PR [hadasy-tau/deep-learning-project#31](https://github.com/hadasy-tau/deep-learning-project/pull/31) (the personalization interval and the loop guard). It does, since 2026-10-02.
- [ ] **The panel is 12 speakers** (30601 deferred, `docs/training_plan_v3.md` § 1). `panel_plan_v2.parquet` (6,887 clips) and `panel_test07.parquet` (6,158) on `main` say which.
- [ ] `knesset-asr/knesset-committees-panel-hq` holds every clip of both plans (10,972 WAVs; checked file by file on 2026-10-02), so `data` takes minutes. Should a clip ever be missing, `data` extracts it from the corpus; extra WAVs (another speaker's) are downloaded and ignored.
- [ ] A HuggingFace token that can **read** the `knesset-asr` datasets and **write** to `knesset-asr` (the backup creates `knesset-committees-v3-results` itself, private). A fine-grained token scoped to a personal account gets a 404 on the org.
- [ ] RunPod credit for ~$30.

## The pod

- **Template:** a RunPod **PyTorch** template, so torch and torchaudio come matched to the CUDA build. `setup` stops if torch cannot see the GPU.
- **GPU:** **4× A100 SXM 80 GB** in one pod (§ How many GPUs). The speeds above, `--eval-batch 64` and bf16 without gradient checkpointing were measured on one.
- **Container disk: 100 GB.** Model weights and the HF cache go there (`box/env.sh`).
- **Network volume: optional.** If you attach one, note that in the second run `/workspace` refused writes past ~10 GB however much it reported free (`docs/training_run2.md` § The box). This run keeps audio (~5 GB), adapters and results in the repo; put the repo on the container disk if the volume quota bites.
- The RunPod API key in the laptop's `cache/` is serverless-scoped: create the pod in the web console.

## Known pod quirks (met in run 3, 2026-10-02)

- **The system Python refuses `pip install`** (PEP 668, "externally managed"). Make a venv that
  keeps the template's CUDA-matched torch, and activate it in every shell:
  `python -m venv --system-site-packages /root/venv && source /root/venv/bin/activate`.
  `pod_v3.sh setup` calls plain `pip`, so it needs the venv active.
- **GitHub's default addresses may be unreachable** from the pod while HuggingFace and PyPI work
  (US-MD-1: every `140.82.112–114.x` address timed out, `140.82.121.4` worked). Test with
  `curl -sI https://github.com`; if it fails, clone with
  `git -c http.curloptResolve=github.com:443:140.82.121.4 clone …` and set the same with
  `git config http.curloptResolve …` inside the clone so `git pull` works.
- **`hf` exists only after `huggingface_hub` is installed** (inside the venv), and on the PyTorch
  template the login lands under `HF_HOME=/workspace/.cache/huggingface`. Non-interactive SSH
  shells do not inherit that variable: export it before running a stage over SSH.
- **Stopping the pod from inside:** PID 1's environment holds a pod-scoped `RUNPOD_API_KEY`;
  `export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=') && runpodctl stop pod "$RUNPOD_POD_ID"`.
  A stopped pod still bills a little for its container disk until it is terminated in the console.

## The session, in order

```bash
python -m venv --system-site-packages /root/venv && source /root/venv/bin/activate   # § Known pod quirks
git clone https://github.com/hadasy-tau/deep-learning-project.git && cd deep-learning-project   # if GitHub is unreachable: § Known pod quirks
pip install -q -r src/training/requirements.txt    # installs `hf`
hf auth login                                   # typed interactively; never paste a token into a chat, a file or a command line
bash src/training/box/pod_v3.sh setup
bash src/training/box/pod_v3.sh data            # must end with "data: both plans verified"
bash src/training/box/pod_v3.sh backup
bash src/training/box/pod_v3.sh sanity          # read the time it prints, then tell Dolev the price before going on
```

`sanity` prints how long one 80-minute tuning run took. The whole plan is about 230 such
jobs' worth of GPU time, split over the GPUs. At about 3 minutes a job, the estimate above holds. **At over 6
minutes, stop and report before spending more:** in the second run a 10× slowdown was BLAS
thread oversubscription, fixed in code but worth suspecting first (`docs/training_run2.md`
§ Why the first run took 30 minutes a cell).

Then the rest, detached so it survives a dropped SSH session:

```bash
setsid nohup bash src/training/box/pod_v3.sh all > src/training/outputs/logs/all.log 2>&1 < /dev/null &
tail -f src/training/outputs/logs/all.log
```

`all` runs `sanity` (skipped work is instant), `tune`, `base`, `final`, `seeds` and `summary`.
It stops at the first stage that fails. To run a stage by hand instead, call it by name. The
order matters: `tune` before `final` and `seeds`, which refuse until the recipe is final.

## Stop rules

| where | condition | what to do |
|---|---|---|
| `setup` | torch sees no GPU; not logged in; no read access to `panel-hq` | fix it; nothing else is worth running |
| `data` | `verify` fails on either plan | stop and report. Do not edit the plans on the pod |
| `sanity` | the overfit check's loss does not fall steeply (the second run: 0.58 → 0.0005 in 60 steps) | stop: the training loop is broken |
| `sanity` | one tuning run takes more than ~6 min | stop and report the price before going on |
| `tune` | `v3_decide.py pick` says a run is missing | rerun `tune`: finished runs are skipped |
| `base` | FAIL: a speaker's base WER on the high-quality test is more than 0.10 **above** their error-map `wer_B` (clean clips should score *below* it) | **stop**. Materialization or scoring changed the metric, and every number after it would be invalid |
| any | a cell's loss is NaN, or one cell takes 3× the others | look before continuing |

## How the decisions are made (`src/training/box/v3_decide.py`)

- **Tuning, per stage and per budget (5 and 80 min), on 30685, 23558, 30718 and 30859.** A
  candidate replaces the incumbent only if it raises the mean `rel_drop` (relative drop in
  validation loss) by ≥ 0.01 absolute **and** beats the incumbent for ≥ 3 of the 4 speakers.
  Among candidates that pass, the largest gain wins. Incumbents are the simpler settings: lr
  3e-4, rank 8, Whisper dropout 0, no augmentation. 20 minutes takes 80's recipe. If 5 and 80
  disagree, it is written to `recipe_v3.json`'s `log`; that is a finding, report it. The test
  set is never read here.
- **The final run uses each budget's own recipe** with its own budget-matched control: one
  `run_panel.py` call per budget.
- **Seeds 1 and 2:** one call per seed, because `run_control` trains its control at the first
  seed it is given. Each seed's control folds are a different shuffle; no speaker is ever
  evaluated on an adapter that heard them.
- **Decoding:** greedy with the loop guard (`evaluate.py`), identical for base, own adapter and
  control. Each time it fires, the log says `loop guard: k of n chunks decoded again`.

## What to report

The backup already holds everything (`results/`, `runs/`, `runs_tune/`, `results.csv`,
`tuning.csv`, `logs/`). The report to Dolev and Hadas is short:

1. **The recipe:** `outputs/recipe_v3.json`, its comparison tables, and any 5/80 disagreement.
2. **The base check:** the table `v3_decide.py base` printed.
3. **The result, per budget:**
   - **personalization:** median `personalization_rel` over speakers, and how many speakers have `personalization_p < 0.05`, on the high-quality test (`personalization_*`) and on the ≥ 0.7 test (`test07.personalization_*`);
   - **the quality bands:** where the gain was earned (`test07.bands.*`).
4. **For whom:** the speakers whose personal effect is significant, and whether it holds across seeds 0, 1 and 2 at 80 min. That is the question the seeds were added to answer.
5. **The style flag:** how many cells have `style_not_speaker`. On high-quality data it should be rare; if not, say so.
6. **Loop guard counts** from the logs (`grep -c "loop guard" src/training/outputs/logs/*.log`), split by base versus adapters if they differ much.
7. **Anything that broke,** and what was rerun.

Do not pool these rows with the second run's: different data plan, different test sets
(`hq` in the cell name marks this run's).

## After the run

Stop the pod in the RunPod console, or from inside (§ Known pod quirks). A stopped pod still
bills for its disk, and a network volume for storage, until terminated. Everything that matters
is in the backup dataset; check it file by file against the pod before stopping.
