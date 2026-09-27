#!/bin/bash
# Regenerates output/matching_results.tsv and output/candidate_pairs.tsv from the data in dataset/.
# Run from the project root (the folder containing dataset/). Needs ~32 vCPU, ~128 GB RAM, ~40 GB disk;
# about 6 hours on an ml.g5.8xlarge (CPU only).
set -euo pipefail
C=code/business_entity_resolution
# 1. Full pipeline, final configuration: prepare -> block (+ learned candidate ranker) -> features -> 2-stage LightGBM -> predict
python $C/src/run.py --config $C/configs/v12.yaml
# 2. Second model on the same candidates and features, without the density-sensitive stage-2 features
R=work/runs
mkdir -p $R/v12b_s2robust/submission
for f in train_pairs.parquet test_pairs.parquet; do ln -f $R/v12_caps2/$f $R/v12b_s2robust/$f; done
cp -al $R/v12_caps2/train_features $R/v12_caps2/test_features $R/v12b_s2robust/
cp $R/v12_caps2/submission/candidate_pairs.tsv $R/v12b_s2robust/submission/
python $C/src/run.py --config $C/configs/v12b.yaml --stage train predict
# 3. Average the two models' probabilities, re-tune match-set selection
python $C/src/blend.py v12_caps2 v12b_s2robust
# 4. Correct the French predictions (no French training data) for label shift, write output/
python $C/src/france_shift.py v12_caps2 $R/blend_v12_caps2+v12b_s2robust
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
