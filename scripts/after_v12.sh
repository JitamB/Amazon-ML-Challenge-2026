#!/bin/bash
# Start the v13 chain once the v12 chain has finished (the two don't fit in memory together).
cd ~/ml2026
while pgrep -f "[c]hain_v12.sh" >/dev/null; do sleep 60; done
rm -rf work/runs/v13_keys_ties
nohup scripts/chain_v13.sh > work/chain_v13.out 2>&1
