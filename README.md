# Amazon ML Challenge 2026: Business Entity Resolution

For every Source 1 business, find the Source 2 / Source 3 records that are the same business. The score is F0.5 per S1, averaged over all S1s. The submission also includes the candidate set that was scored, and smaller candidate sets rank higher.

**Pipeline:**
1. Text normalisation (9 Indic scripts, learned transliteration and address maps, French address cleanup).
2. Multi-key inverted-index blocking.
3. A learned candidate ranker that keeps about 6 candidates per S1.
4. About 55 pair features.
5. A two-stage LightGBM model.
6. Match-set selection tuned for F0.5.

## Results

| Run | Main change | Blocking recall | Cands / S1 | Validation (test mix) | Leaderboard |
|---|---|---|---|---|---|
| v1_gbdt | Baseline | 94.6% | 74 | 0.9458 | 0.942 (160/490) |
| v3_stage2 | Number-difference features + stage-2 model | 95.7% | 76 | 0.9617 | 0.9578 (251) |
| v4_translit | Learned transliteration | 96.05% | 76 | 0.9656 | 0.9606 (313) |
| v5b_fullfold | Candidate ranker (top 12), address map, all train S1s | 95.47% | 11.9 | 0.9680 | 0.959 |
| v6_frfix_keys | France-safe address map, name × address keys | 97.07% | 12.0 | 0.9762 | 0.963 (702) |
| v8_density | French cleanup, flexible candidate sets, 81% of train S1s | 97.15% | **5.75** | 0.9758 | **0.9645** |
| v9_acr_3fold | Dotted acronyms, 3 model folds | 97.15% | 5.74 | 0.9763 | |
| blend v9 + v8 + v6 | Probability average | | | 0.9774 | 0.9653 |
| v11_caps + v11b_s2robust | Second blocking pass; average of full and density-robust stage 2 | 97.98% | 6.10 | 0.9790 | 0.9733 |
| + France shift | EM label-shift correction on French predictions | | | 0.9790 | 0.9737 |
| v12_caps2 + v12b_s2robust + France shift | Looser second pass | 98.30% | 6.40 | 0.9799 | 0.9744 |
| **v13 + v13b + v12 + v12b + France shift (final)** | More keys, tie-aware top-N; four-model average | 98.61% | 7.0 (test 7.4) | **0.9808** | **0.97526** |
| final + stage 3 | Record-cluster features (twins, shared numbers) | | | 0.9819 | 0.97347 (not used) |

*Validation = out-of-fold F0.5 over India and the US, re-weighted to the test mix. France (15% of test) has no training data, so it can't be validated. A probe submission put France at about 0.94 on the leaderboard and India/US at about 0.969. Details: [doc/architecture.md](doc/architecture.md), sections 2 and 5.*

## Documentation

| Read | For |
|---|---|
| [doc/problem_statement.md](doc/problem_statement.md) | The official challenge statement: task, metric, rules, submission format |
| [doc/analysis.md](doc/analysis.md) | Data analysis, measured baselines, the original strategy |
| [doc/architecture.md](doc/architecture.md) | **Master document:** how the pipeline works, every version's changes and results, open problems |
| [doc/investigations.md](doc/investigations.md) | Every analysis we ran: error analysis, blocking diagnostics, the France bug and probe, density, stage-2 inflation, loss decomposition, orphan records, the unseen-country simulation, what didn't work |
| [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) | Running the pipeline, the config system, which stage to re-run after a change |
| [dataset/README.md](dataset/README.md) | Where to put the data (not in the repo) |

## Setup

Requires **Python 3.11 or newer**. Run everything from the repo root.

```bash
git clone https://github.com/JitamB/Amazon-ML-Challenge-2026.git && cd Amazon-ML-Challenge-2026
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

# 2. The final submission end to end (full run, robust second model, blend, France correction, validation):
#    needs about 32 vCPU and 128 GB RAM (we use SageMaker ml.g5.8xlarge)
bash $C/reproduce.sh

# 3. Or step by step: one full run, then blend finished runs (first run's candidate set)
python $C/src/run.py --config $C/configs/v12.yaml
python $C/src/blend.py v12_caps2 v12b_s2robust
```

- **Outputs:** `output/matching_results.tsv` is the file to upload; `output/candidate_pairs.tsv` goes in the final zip. A copy of both is kept in each run's folder, `work/runs/<run>/submission/`.
- **Tracking:** every prediction run adds a row to `work/experiments.csv`. Add the leaderboard score there after uploading.
- **Checking a submission:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Experiments

Each experiment is a small YAML file in `code/business_entity_resolution/configs/` that inherits from another one and overrides a few keys:

| Config | Inherits | Adds |
|---|---|---|
| `base.yaml` | | v1 settings; every tunable number lives here |
| `v2.yaml` | base | Number keys over the first 3 numbers; 2 folds, lr 0.1 |
| `v3.yaml` | v2 | Stage-2 stacking model |
| `v4.yaml` | v3 | Learned transliteration map |
| `v5.yaml`, `v5b.yaml` | v4 | Candidate ranker, learned address map, all train S1s; full-size folds |
| `v6.yaml` | v5b | France-safe address map, name/phonetic × address-word keys, 120 loose candidates |
| `v7.yaml` | v6 | French region/department cleanup, flexible candidate sets |
| `v8.yaml` | v7 | 81% of train S1s, matching the test's records-per-S1 density |
| `v9.yaml` | v8 | Dotted legal acronyms collapsed; 3 model folds |
| `v10.yaml` | v9 | Stage 2 without the S1-level group features (`model.stage2_exclude`) |
| `v11.yaml`, `v11b.yaml` | v9 | Second blocking pass (`blocking.extra_pass`, caps 100 / 1500); b = without S1-level stage-2 features |
| `v12.yaml`, `v12b.yaml` | v11 | Second pass caps 200 / 3000, top 100 |
| `v13.yaml`, `v13b.yaml` | v12 | Compact name × number / address word and one-digit-deletion keys; ties kept at the top-N cut |

For one-off changes, use `--set key=value` together with a new `run_name`. A change to `normalize.py` needs a new `data.cache_tag`, so the prepared data is rebuilt instead of reused.

## Adding a model score from another approach (e.g. a neural model)

An external model's score can go into the stage-2 model as an extra feature. The loader isn't written yet; it will be added when the first score file exists. Provide one parquet or TSV per split with the columns `s1_id, rid, score`:
- **Pairs:** the rows of `work/runs/<run>/{train,test}_pairs.parquet`, which are the candidates the pipeline scores.
- **Train scores must be out-of-fold, using our S1 folds:** fold = `pandas.util.hash_array(s1_rid.astype(object)) % 3` (as in `pipeline.py`). A pair's score must come from a model that never saw that S1's fold.
- **Test scores:** any model trained on all of train.

## Running on a remote machine from your laptop

`scripts/remote.sh` syncs code to a SageMaker space and runs commands there, over the AWS Toolkit tunnel. Copy `scripts/remote.env.example` to `scripts/remote.env` (git-ignored) and set your space's host. Then:

```bash
scripts/remote.sh push                 # sync code, configs, scripts, docs
scripts/remote.sh push-data            # upload dataset/ once
scripts/remote.sh run 'nohup python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v12.yaml > work/v12.log 2>&1 &'
scripts/remote.sh pull                 # fetch metrics, logs and submissions
```

## Diagnostics (`scripts/`)

| Script | Answers |
|---|---|
| `error_analysis.py <run>` | Score ceilings, loss split (blocking / model miss / wrong match), examples |
| `diag_blocking.py <run_dir> [n] [extra_key_types]` | Why true links miss blocking; what-if for new key types |
| `flex_k.py <run>` | Candidate-set size vs blocking recall for fixed and score-based cuts |
| `compare_runs.py <a> <b>` | Two runs' test predictions compared by country |
| `blend.py <runs…>` | Averages runs' probabilities, re-tunes selection, writes a submission |
| `label_shift.py <run> [strength] [preds_dir] [--only=France] [--strata]` | Re-selects test matches after an EM label-shift correction |
| `funnel.py <run> <blend_dir>` | True links kept at each step, and what blocking misses |
| `cap_whatif.py <run_dir> [n] [extra types]` | Share of blocking misses that looser caps or new key types would recover |
| `loss_decomp.py <run> <blend_dir>` | Loss after blocking in F0.5 points, by cause |
| `loss_examples.py`, `orphan_signals.py` | Examples of each loss class; signals separating records of no S1 from true matches |
| `select_exp.py <run> <blend_dir>` | Selection rules compared with 2-way cross-validation |
| `stage3.py <run> <blend_dir> [robust]` | Stage 3 on a blend (development copy of `src/stage3.py`) |
| `compare_sel.py <run> <dir_a> <dir_b>` | Matches added / removed by b vs a, on train and on test (transfer check) |
| `loco_sim.py <run> <train_country> <unseen_country>` | Leave-one-country-out simulation of the France correction |
| `stage2_check.py`, `stage2_features_shift.py`, `decoy_rates.py` | Stage-2 inflation of decoy-type pairs on test |
| `density_check.py`, `density_sim.py`, `transfer_check.py` | Train vs test differences: records per S1, candidate strength, map coverage |
| `france_map_check.py`, `france_look.py` | What the address map does to French records; uncertain French predictions |

## Repository layout

```
code/business_entity_resolution/   pipeline package: src/, configs/, README, pinned requirements
scripts/                           setup, remote execution, blending, diagnostics
utils/validate_submission.py       official submission validator
doc/                               problem statement, analysis, architecture (master doc)
dataset/                           competition data goes here (git-ignored)
work/, output/, submissions/       generated files (git-ignored)
```
