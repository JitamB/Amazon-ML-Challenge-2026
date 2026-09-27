#!/bin/bash
# Resume v13 at test blocking (train pairs exist), then v13b on v13's features, then blend v13 + v13b.
cd ~/ml2026 && source ~/venv-ber/bin/activate
BER_BLOCK_SPLITS=test python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v13.yaml --stage block > work/v13_keys_ties.log 2>&1 && python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v13.yaml --stage features train predict >> work/v13_keys_ties.log 2>&1
grep -q PASS work/v13_keys_ties.log || { echo "v13 did not finish cleanly"; exit 1; }
mkdir -p work/runs/v13b_s2robust
for f in train_pairs.parquet test_pairs.parquet; do ln -f work/runs/v13_keys_ties/$f work/runs/v13b_s2robust/$f; done
cp -al work/runs/v13_keys_ties/train_features work/runs/v13_keys_ties/test_features work/runs/v13b_s2robust/
mkdir -p work/runs/v13b_s2robust/submission && cp work/runs/v13_keys_ties/submission/candidate_pairs.tsv work/runs/v13b_s2robust/submission/
python code/business_entity_resolution/src/run.py --config code/business_entity_resolution/configs/v13b.yaml --stage train predict > work/v13b_s2robust.log 2>&1
python -u scripts/blend.py v13_keys_ties v13b_s2robust > work/blend_v13.log 2>&1
echo CHAIN_DONE >> work/blend_v13.log
