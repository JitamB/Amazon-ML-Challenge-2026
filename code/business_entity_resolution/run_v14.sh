#!/bin/bash
# v14: every post-deadline improvement (see configs/v14.yaml). Not part of the submitted output
# (that is reproduce.sh). Needs a GPU for the cross-encoder and the embedding pass, plus
# requirements-v14.txt. Roughly: prepare 0.1 h, blocking 5 h, cross-encoder 3 h,
# features + training (3 seeds) 2 h, v14b 1.5 h.
set -euo pipefail
C=code/business_entity_resolution
R=work/runs
python $C/src/run.py --config $C/configs/v14.yaml --stage prepare block
python $C/src/cross_encoder.py --config $C/configs/v14.yaml
python $C/src/run.py --config $C/configs/v14.yaml --stage features train predict
mkdir -p $R/v14b_s2robust/submission
for f in train_pairs.parquet test_pairs.parquet; do ln -f $R/v14_all/$f $R/v14b_s2robust/$f; done
cp -al $R/v14_all/train_features $R/v14_all/test_features $R/v14b_s2robust/
cp $R/v14_all/submission/candidate_pairs.tsv $R/v14b_s2robust/submission/
python $C/src/run.py --config $C/configs/v14b.yaml --stage train predict
# full / robust weighted toward the robust model, as the leaderboard preferred
python $C/src/blend.py v14_all v14b_s2robust --weights=0.36,0.64
python $C/src/france_shift.py v14_all $R/blend_v14_all+v14b_s2robust
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
