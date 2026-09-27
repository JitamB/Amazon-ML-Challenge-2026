#!/bin/bash
# After v11 finishes: train v11b on v11's features, then blend v11 + v11b.
cd ~/ml2026 && source ~/venv-ber/bin/activate
while pgrep -f "[c]onfigs/v11.yaml" >/dev/null; do sleep 60; done
grep -q PASS work/v11_caps.log || { echo "v11 did not finish cleanly"; exit 1; }
mkdir -p work/runs/v11b_s2robust
for f in train_pairs.parquet test_pairs.parquet; do ln -f work/runs/v11_caps/$f work/runs/v11b_s2robust/$f; done
cp -al work/runs/v11_caps/train_features work/runs/v11_caps/test_features work/runs/v11b_s2robust/
mkdir -p work/runs/v11b_s2robust/submission && cp work/runs/v11_caps/submission/candidate_pairs.tsv work/runs/v11b_s2robust/submission/
python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v11b.yaml --stage train predict > work/v11b_s2robust.log 2>&1
python -u scripts/blend.py v11_caps v11b_s2robust > work/blend_v11.log 2>&1
echo CHAIN_DONE >> work/blend_v11.log
