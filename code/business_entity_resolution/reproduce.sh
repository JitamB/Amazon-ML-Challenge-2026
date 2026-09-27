#!/bin/bash
# Regenerates output/matching_results.tsv and output/candidate_pairs.tsv from the data in dataset/.
# Run from the project root (the folder containing dataset/). Needs ~32 vCPU, ~128 GB RAM, ~60 GB disk;
# about 12 hours on an ml.g5.8xlarge (CPU only), most of it blocking.
set -euo pipefail
C=code/business_entity_resolution
R=work/runs
STAGE3=0   # stage 3 (record-cluster features) is off: +0.0011 validation but 0.9735 vs 0.9753 on the leaderboard

# A second model on a run's candidates and features, without the density-sensitive stage-2 features
robust() {  # robust <run> <robust_run> <config>
  mkdir -p $R/$2/submission
  for f in train_pairs.parquet test_pairs.parquet; do ln -f $R/$1/$f $R/$2/$f; done
  cp -al $R/$1/train_features $R/$1/test_features $R/$2/
  cp $R/$1/submission/candidate_pairs.tsv $R/$2/submission/
  python $C/src/run.py --config $C/configs/$3 --stage train predict
}

# 1. Two full pipelines: prepare -> block (+ learned candidate ranker) -> features -> 2-stage LightGBM -> predict.
#    v13 = v12 + more blocking keys and tie-aware top-N; its candidate set is the submitted one.
python $C/src/run.py --config $C/configs/v12.yaml
robust v12_caps2 v12b_s2robust v12b.yaml
python $C/src/run.py --config $C/configs/v13.yaml
robust v13_keys_ties v13b_s2robust v13b.yaml
# 2. Average the four models' probabilities on v13's candidates, re-tune match-set selection
python $C/src/blend.py v13_keys_ties v13b_s2robust v12_caps2 v12b_s2robust
B=$R/blend_v13_keys_ties+v13b_s2robust+v12_caps2+v12b_s2robust
# 3. (optional, off) Stage 3: re-score with record-cluster features (exact twins, shared number sets)
if [ "$STAGE3" = 1 ]; then
  python $C/src/stage3.py v13_keys_ties $B
  B=${B}_s3
fi
# 4. Correct the French predictions (no French training data) for label shift, write output/
python $C/src/france_shift.py v13_keys_ties $B
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
