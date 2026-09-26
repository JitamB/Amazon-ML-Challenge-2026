# Business Entity Resolution: pipeline

**Stages:**
1. Prepare: normalise text.
2. Block: an inverted-index lookup, then a learned ranker that keeps a small candidate set per S1.
3. Features: pair features.
4. Train: a two-stage LightGBM model.
5. Predict: best-S1 assignment, then F0.5-optimal match-set selection.

How each stage works, and the history of every version: [doc/architecture.md](../../doc/architecture.md). The data analysis behind the choices: [doc/analysis.md](../../doc/analysis.md).

## Layout

```
configs/
  base.yaml        v1 settings; every tunable number lives here
  smoke.yaml       tiny regional dataset for bug-hunting (inherits base.yaml)
  v2.yaml … v9.yaml  one experiment each; each inherits the previous one (see the main README)
src/
  run.py           CLI entry point
  ber/
    config.py      YAML loading, `base:` inheritance, --set overrides
    io.py          TSV reading, submission writing
    normalize.py   transliteration (9 Indic scripts), name/address cleaning, numbers, phonetic skeleton,
                   learned transliteration and address-word maps
    blocking.py    multi-key inverted-index blocking -> loose candidate pairs
    candidates.py  learned candidate ranker -> the pruned set reported in candidate_pairs.tsv
    features.py    pair features (rapidfuzz) + blocking-context features
    model.py       LightGBM with S1-grouped cross-fitting (out-of-fold predictions for every train pair)
    stage2.py      competition, head-to-head and support features built from stage-1 out-of-fold scores
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

# 2. Full run. Stages: prepare block features train predict
python $C/src/run.py --config $C/configs/v9.yaml

# 3. Re-run only some stages, with overrides and a new run name
python $C/src/run.py --config $C/configs/v9.yaml --stage train predict \
    --set run_name=v9_leaves127 model.params.num_leaves=127
```

## Key settings (v9)

| Setting | Value | Meaning |
|---|---|---|
| `data.train_s1_frac` | 0.81 | Share of train S1s used. It matches the test's records-per-S1 density |
| `data.learned_translit`, `data.learned_addr_map` | true | Learn word maps from train links and apply them to S2/S3 |
| `data.cache_tag` | `_translit_addr_guard_fr_acr` | Separates prepared-data caches. Change it whenever `normalize.py` changes |
| `blocking.key_types` | 13 key types | See `doc/architecture.md`, section 3.2 |
| `blocking.max_df_s1` / `max_df_s23` / `top_n` | 30 / 400 / 120 | Key frequency caps; loose candidates per S1 |
| `candidates.keep_k` / `min_cscore` / `min_k` | 12 / 0.001 / 1 | At most 12 per S1; drop ranker scores below 0.001; always keep 1 |
| `model.n_folds` | 3 | Folds by S1 hash, for both model stages |
| `model.stage2` | true | Train the stacking model |
| `postprocess.mode` | tune | Choose threshold or expected-F0.5 selection on out-of-fold predictions |

## Running on the SageMaker space (ml.g5.8xlarge: 32 vCPU, 124 GB RAM, A10G 24 GB, 70 GB disk)

| Resource | Use |
|---|---|
| CPU | `features.n_jobs: -1` uses every core for normalisation, blocking keys, ranker and pair features. LightGBM uses all threads |
| RAM | Peak about 70 GB (blocking). Features are written as parquet parts of up to 10M pairs and scored part by part |
| Disk | Each prepared-data cache is about 3 GB; each run's pairs and features are about 5 GB. Delete old `work/runs/*/*_features` folders and unused `work/cache/*` to free space |
| GPU | Not used by the pipeline. It is free for a neural model running alongside |

Time per full run (v6): prepare 6 min, block 86 min, features 9 min, train 36 min (2 folds), predict 14 min.

```bash
nohup python $C/src/run.py --config $C/configs/v9.yaml > work/v9.log 2>&1 &
tail -f work/v9.log
```

## Outputs

| Path | What it holds |
|---|---|
| `output/matching_results.tsv` | The file to upload |
| `output/candidate_pairs.tsv` | The exact pairs the model scored (the pruned candidate set) |
| `work/cache/<data><cache_tag>/` | Normalised sources, plus `translit.json` / `addr_map.json`. They are built once per cache tag and reused |
| `work/runs/<run_name>/` | Per-run files: `config.yaml`, `log.txt`, `metrics.json`, pairs, `ranker/`, features, `models/`, `models_s2/`, `oof.parquet`, `test_preds.parquet`, feature importances, `selection.json`, `submission/` |
| `work/experiments.csv` | One row per prediction run. Fill in `lb_score` and `notes` after each upload |

## Which stages to re-run after a change

Each stage reads the previous stage's files, so after a change re-run from the first affected stage onward:

| Changed | Re-run from |
|---|---|
| `normalize.py` or learned maps | a new `data.cache_tag`, then `prepare` |
| blocking keys, caps or the candidate ranker (`candidates.*`) | `block` |
| features | `features` |
| model parameters, stage-2 features | `train` |
| selection rules | `train` (re-tunes), then `predict` |

## Metrics to watch

| Metric | Meaning |
|---|---|
| `loose_block_recall` / `block_recall` | Share of true links in the loose / pruned candidate set. This is the ceiling for recall |
| `cands_per_s1` | Average pruned candidate set size. Smaller ranks higher in the final evaluation |
| `stage1_f05_test_mix` | Stage-1 model alone |
| `macro_f05` / `f05_<country>` | Final out-of-fold score with full S1-vs-S1 competition |
| `f05_test_mix` | Out-of-fold score re-weighted to the test country mix. Compare this with the leaderboard. So far it has run 0.004–0.011 above it; see `doc/architecture.md`, section 5 |
