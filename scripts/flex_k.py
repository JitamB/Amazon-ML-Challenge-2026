"""Candidate-set size vs blocking recall for flexible cuts on the ranker score (within the kept top-k).
    python scripts/flex_k.py v5b_fullfold
"""
import sys

import numpy as np
import pandas as pd

run = sys.argv[1]
cache = "work/cache/dataset_translit_addr" if len(sys.argv) < 3 else sys.argv[2]
p = pd.read_parquet(f"work/runs/{run}/train_pairs.parquet", columns=["s1_idx", "r_idx", "cscore"])
gt = pd.read_parquet(f"{cache}/train_gt.parquet")
n_s1 = pd.read_parquet(f"{cache}/train_s1.parquet", columns=["country"]).shape[0]
n_r = int(max(p["r_idx"].max(), gt["r_idx"].max())) + 1
p["y"] = np.isin(p["s1_idx"].values * n_r + p["r_idx"].values, gt["s1_idx"].values * n_r + gt["r_idx"].values)
p = p.sort_values(["s1_idx", "cscore"], ascending=[True, False])
p["rank"] = p.groupby("s1_idx").cumcount() + 1
tot = len(gt)
print(f"links {tot:,}; S1 {n_s1:,}")
for k in [4, 6, 8, 10, 12]:
    m = p["rank"] <= k
    print(f"fixed top-{k:>2}: {m.sum() / n_s1:5.2f}/S1  recall {p.loc[m, 'y'].sum() / tot:.4%}")
for t in [0.001, 0.003, 0.01, 0.02, 0.05]:
    for mn in [1, 3]:
        m = (p["cscore"] >= t) | (p["rank"] <= mn)
        print(f"cscore>={t:<5} (min {mn}): {m.sum() / n_s1:5.2f}/S1  recall {p.loc[m, 'y'].sum() / tot:.4%}")
