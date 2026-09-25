# Business Entity Resolution: pipeline

The pipeline is: blocking, then pair features, then a LightGBM pair classifier, then best-S1 assignment and F0.5-optimal match-set selection.
The analysis behind these choices is in `doc/analysis.md`.

## Layout

```
configs/
  base.yaml        full-data settings (AWS box)
  smoke.yaml       tiny regional dataset for bug-hunting (inherits base.yaml)
src/
  run.py           CLI entry point
  ber/
    config.py      YAML loading, `base:` inheritance, --set overrides
    io.py          TSV reading, submission writing
    normalize.py   transliteration (9 Indic scripts), name/address cleaning, numbers, phonetic skeleton
    blocking.py    multi-key inverted-index blocking -> candidate pairs
    features.py    pair features (rapidfuzz) + blocking-context features
    model.py       LightGBM with S1-grouped cross-fitting (out-of-fold predictions for every train pair)
    postprocess.py best-S1 assignment, caps, threshold / expected-F0.5 set selection, tuning
    metrics.py     exact macro F0.5 and blocking recall
    pipeline.py    stage orchestration, caching, validator, experiment log
    smoke.py       builds the smoke dataset by streaming the full files
```

## Setup

Run all commands from the project root, the folder that contains `dataset/` and `utils/`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Running

```bash
C=code/business_entity_resolution

# 1. Smoke test (a few minutes, any machine)
python $C/src/run.py --config $C/configs/smoke.yaml --stage make_smoke
python $C/src/run.py --config $C/configs/smoke.yaml

# 2. Full run (AWS). Stages: prepare block features train predict
python $C/src/run.py --config $C/configs/base.yaml

# 3. Re-run only some stages, with overrides and a new run name
python $C/src/run.py --config $C/configs/base.yaml --stage train predict \
    --set run_name=v1_leaves127 model.params.num_leaves=127
```

## Running on the AWS box (ml.g5.4xlarge: 16 vCPU, 64 GB RAM, A10G 24 GB, 75 GB disk)

Here is how the pipeline uses the machine:

| Resource | Use |
|---|---|
| CPU | `features.n_jobs: 14` worker processes for normalisation and pair features. LightGBM uses 16 threads |
| RAM | `data.train_s1_frac: 0.4` keeps train features at roughly 10 GB. Test features are written as ~10M-pair parquet parts and scored part by part |
| Disk | Estimates below. Delete old `work/runs/*/*_features` folders to free space |
| GPU | Not used by v1. Planned for: multilingual embedding blocking/features, a fine-tuned cross-encoder re-ranker, and the Qwen2.5-7B (Apache 2.0) France diagnostic, which fits in bf16 on the A10G |

Disk use (estimates): dataset about 2.3 GB, `work/cache` about 2 GB, and each run's features about 10–20 GB.

Watch the first full run with `htop` and `free -g`. If memory is tight, lower `blocking.s1_chunk`, `features.part_rows` or `data.train_s1_frac`.

First full run:

```bash
nohup python $C/src/run.py --config $C/configs/base.yaml > work/full.log 2>&1 &
tail -f work/full.log
```

## Outputs

| Path | What it holds |
|---|---|
| `output/matching_results.tsv` | The file to upload |
| `output/candidate_pairs.tsv` | The exact pairs the model scored |
| `work/cache/<data>/` | Normalised sources. They are built once and reused by every run; delete this folder after changing `normalize.py` |
| `work/runs/<run_name>/` | Per-run files: `config.yaml`, `log.txt`, `metrics.json`, pairs, features, models, OOF predictions, `feature_importance.csv`, `selection.json` |
| `work/experiments.csv` | One row per prediction run. Fill in `lb_score` and `notes` after each upload |

## Which stages to re-run after a change

Each stage reads the previous stage's files, so after a change re-run from the first affected stage onward:

| Changed | Re-run from |
|---|---|
| `normalize.py` | delete `work/cache/`, then `prepare` |
| blocking keys or caps | `block` |
| features | `features` |
| model parameters | `train` |
| selection rules | `train` (re-tunes), then `predict` |

## Metrics to watch

| Metric | Meaning |
|---|---|
| `block_recall` | Share of true links that reach the model. This is the ceiling for recall |
| `macro_f05` / `f05_<country>` | Out-of-fold score with full S1-vs-S1 competition |
| `f05_test_mix` | Out-of-fold score re-weighted to the test country mix. Compare this with the leaderboard |

If `data.train_s1_frac` is below 1, the out-of-fold score is slightly pessimistic. Some S2/S3 records then have their true S1 missing from the sample, so they act as extra decoys.
