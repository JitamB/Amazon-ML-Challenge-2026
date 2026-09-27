"""How many decoy-type records does each S1 face, train vs test? (label-free on test)

For every S2/S3 record in a run's candidate pairs, take the candidate S1 with the most
similar name (token-set; ties by ranker score). If that similarity is >= 85, classify the
record by how its house number relates to that S1's: equal / substring (digits dropped or
added, typical of true matches) / shift 1-20 (typical of generated decoys) / other / missing.
Rates are per S1, by country. On train the share of true matches per class is also shown.

    python scripts/decoy_rates.py <run> <split>
"""
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, split = sys.argv[1], sys.argv[2]
pl = Pipeline(load_config(f"work/runs/{run}/config.yaml"))
s1, s23 = pl.load_prepared(split)
p = pd.read_parquet(pl.run_file(f"{split}_pairs.parquet"), columns=["s1_idx", "r_idx", "cscore"])
p["nsim"] = process.cpdist(s1["name"].values[p.s1_idx.values], s23["name"].values[p.r_idx.values],
                           scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
p = p.sort_values(["r_idx", "nsim", "cscore"], ascending=[True, False, False]).drop_duplicates("r_idx")
p = p[p.nsim >= 85]
a, b = s1["hn"].values[p.s1_idx.values], s23["hn"].values[p.r_idx.values]


def rel(x, y):
    if not x or not y:
        return "missing"
    if x == y:
        return "equal"
    if x in y or y in x:
        return "substring"
    return "shift1-20" if len(x) < 15 and len(y) < 15 and abs(int(x) - int(y)) <= 20 else "other"


p["rel"] = [rel(x, y) for x, y in zip(b, a)]
p["country"] = s1["country"].values[p.s1_idx.values]
n_s1 = pd.Series(s1["country"].values[pl.s1_subset(split, s1)]).value_counts()
t = p.groupby(["country", "rel"]).size().unstack(fill_value=0).div(n_s1, axis=0)
print(f"{run} {split}: S1 {n_s1.sum():,}, records {len(s23):,}; strong-name records per S1 by house-number relation")
print(t.round(3).to_string())
if split == "train":
    gt = pd.read_parquet(pl.cache("train_gt.parquet"))
    n_r = len(s23)
    p["true"] = np.isin(p.s1_idx.values * n_r + p.r_idx.values, gt.s1_idx.values * n_r + gt.r_idx.values)
    print("share that are true matches:")
    print(p.groupby(["country", "rel"])["true"].mean().unstack().round(3).to_string())
