# Amazon ML Challenge 2026: Business Entity Resolution

For every Source 1 business, find the Source 2 / Source 3 records that are the same business. The score is F0.5 per S1, averaged over all S1s.

Pipeline: text normalisation (including 9 Indic scripts), multi-key blocking, about 50 pair features, a two-stage LightGBM model, and match-set selection tuned for F0.5.

| Run | Validation F0.5 (test country mix) | Leaderboard |
|---|---|---|
| v1_gbdt | 0.9458 | **0.942** (160/490) |
| v3_stage2 | *running* | |

## Documentation

| Read | For |
|---|---|
| [doc/problem_statement.md](doc/problem_statement.md) | The official challenge statement: task, metric, rules, submission format |
| [doc/analysis.md](doc/analysis.md) | Data analysis, measured baselines, strategy |
| [doc/architecture.md](doc/architecture.md) | How the pipeline works, stage by stage, and what's next |
| [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) | Running the pipeline, the config system, which stage to re-run after a change |
| [dataset/README.md](dataset/README.md) | Where to put the data (not in the repo) |

## Setup

Requires **Python 3.11 or newer**. Run everything from the repo root.

```bash
git clone <repo-url> && cd <repo>
scripts/setup_env.sh                 # creates .venv and installs requirements.txt
source .venv/bin/activate
```

With conda instead: `conda create -n ber python=3.12 -y && conda activate ber && pip install -r requirements.txt`.

Then download the competition files into `dataset/`, as described in [dataset/README.md](dataset/README.md).

## Running

```bash
C=code/business_entity_resolution

# 1. Smoke test: a small regional slice of the data, runs in minutes on a laptop
python $C/src/run.py --config $C/configs/smoke.yaml --stage make_smoke
python $C/src/run.py --config $C/configs/smoke.yaml

# 2. Full run: needs about 32 vCPU and 128 GB RAM (we use SageMaker ml.g5.8xlarge)
python $C/src/run.py --config $C/configs/v3.yaml
```

- **Outputs:** `output/matching_results.tsv` is the file to upload; `output/candidate_pairs.tsv` goes in the final zip. A copy of both is kept in each run's folder, `work/runs/<run>/submission/`.
- **Tracking:** every prediction run adds a row to `work/experiments.csv`. Add the leaderboard score there after uploading.
- **Checking a submission:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Experiments

Each experiment is a small YAML file in `code/business_entity_resolution/configs/` that inherits from another one (`base: v2.yaml`) and overrides a few keys:
- `base.yaml`: v1.
- `v2.yaml`: better blocking keys and faster training.
- `v3.yaml`: number-difference features and a second-stage model.

For one-off changes, use `--set key=value` together with a new `run_name`.

## Running on a remote machine from your laptop

`scripts/remote.sh` syncs code to a SageMaker space and runs commands there, over the AWS Toolkit tunnel. Copy `scripts/remote.env.example` to `scripts/remote.env` (git-ignored) and set your space's host. Then:

```bash
scripts/remote.sh push                 # sync code, configs, scripts, docs
scripts/remote.sh push-data            # upload dataset/ once
scripts/remote.sh run 'python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v3.yaml'
scripts/remote.sh pull                 # fetch metrics, logs and submissions
```

## Diagnostics

- `scripts/diag_blocking.py <run_dir>`: why true matches are missed by blocking (frequency caps, top-N cut, or no shared key).
- `scripts/error_analysis.py <run_name>`: where the out-of-fold loss comes from, with score ceilings and examples of wrong and missed matches.

## Repository layout

```
code/business_entity_resolution/   pipeline package: src/, configs/, README, pinned requirements
scripts/                           setup, remote execution, diagnostics
utils/validate_submission.py       official submission validator
doc/                               problem statement, analysis, architecture
dataset/                           competition data goes here (git-ignored)
work/, output/, submissions/       generated files (git-ignored)
```
