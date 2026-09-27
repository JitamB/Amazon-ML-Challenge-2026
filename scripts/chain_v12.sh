#!/bin/bash
# Run v12, then v12b on v12's features, then blend v12 + v12b.
cd ~/ml2026 && source ~/venv-ber/bin/activate
python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v12.yaml --stage block features train predict > work/v12_caps2.log 2>&1
grep -q PASS work/v12_caps2.log || { echo "v12 did not finish cleanly"; exit 1; }
mkdir -p work/runs/v12b_s2robust
for f in train_pairs.parquet test_pairs.parquet; do ln -f work/runs/v12_caps2/$f work/runs/v12b_s2robust/$f; done
cp -al work/runs/v12_caps2/train_features work/runs/v12_caps2/test_features work/runs/v12b_s2robust/
mkdir -p work/runs/v12b_s2robust/submission && cp work/runs/v12_caps2/submission/candidate_pairs.tsv work/runs/v12b_s2robust/submission/
python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v12b.yaml --stage train predict > work/v12b_s2robust.log 2>&1
python -u scripts/blend.py v12_caps2 v12b_s2robust > work/blend_v12.log 2>&1
echo CHAIN_DONE >> work/blend_v12.log
