#!/bin/bash
# After the v13 chain: blend v13 + v13b with v12 + v12b, keep the better blend (validation),
# run stage 3 on it, and apply the France correction to both the blend and its stage 3.
cd ~/ml2026 && source ~/venv-ber/bin/activate
until grep -q CHAIN_DONE work/blend_v13.log 2>/dev/null; do sleep 60; done
python -u scripts/blend.py v13_keys_ties v13b_s2robust v12_caps2 v12b_s2robust > work/blend_v13v12.log 2>&1
R=work/runs
A="$R/blend_v13_keys_ties+v13b_s2robust"
B="$R/blend_v13_keys_ties+v13b_s2robust+v12_caps2+v12b_s2robust"
BEST=$(python -c "
import json
a, b = (json.load(open(f'$A/metrics.json'))['f05_test_mix'], json.load(open(f'$B/metrics.json'))['f05_test_mix'])
print('$B' if b > a else '$A')")
echo "best blend: $BEST" > work/after_v13.log
python -u code/business_entity_resolution/src/france_shift.py v13_keys_ties "$BEST" >> work/after_v13.log 2>&1
python -u code/business_entity_resolution/src/stage3.py v13_keys_ties "$BEST" > work/stage3_v13.log 2>&1
python -u code/business_entity_resolution/src/france_shift.py v13_keys_ties "${BEST}_s3" >> work/after_v13.log 2>&1
echo AFTER_DONE >> work/after_v13.log
