"""Are test pairs of a given kind scored differently from train pairs of the same kind?
Pairs are grouped by house-number relation (strong-name pairs only); for each group the
final-p distribution and a few raw traits are compared, train (OOF) vs test.
    python scripts/pair_shift.py <run>
"""
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run = sys.argv[1]
pl = Pipeline(load_config(f"work/runs/{run}/config.yaml"))
bins = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0001]


def rel(x, y):
    if not x or not y:
        return "missing"
    if x == y:
        return "equal"
    if x in y or y in x:
        return "substring"
    return "shift1-20" if len(x) < 15 and len(y) < 15 and abs(int(x) - int(y)) <= 20 else "other"


rows = []
for split, pf in (("train", "oof.parquet"), ("test", "test_preds.parquet")):
    s1, s23 = pl.load_prepared(split)
    d = pd.read_parquet(pl.run_file(pf))[["s1_idx", "r_idx", "p"]]
    a, b = d.s1_idx.values, d.r_idx.values
    d["nsim"] = process.cpdist(s1["name"].values[a], s23["name"].values[b], scorer=fuzz.token_set_ratio,
                               workers=-1, dtype=np.float32)
    d = d[d.nsim >= 85]
    a, b = d.s1_idx.values, d.r_idx.values
    d["rel"] = [rel(x, y) for x, y in zip(s23["hn"].values[b], s1["hn"].values[a])]
    d["name_eq"] = s1["name"].values[a] == s23["name"].values[b]
    d["addr_empty"] = s23["addr_empty"].values[b]
    d["asim"] = process.cpdist(s1["addr"].values[a], s23["addr"].values[b], scorer=fuzz.token_set_ratio,
                               workers=-1, dtype=np.float32)
    d["country"] = s1["country"].values[a]
    d["split"] = split
    if split == "train":
        gt = pd.read_parquet(pl.cache("train_gt.parquet"))
        n_r = len(s23)
        d["true"] = np.isin(a * n_r + b, gt.s1_idx.values * n_r + gt.r_idx.values)
    rows.append(d)
d = pd.concat(rows, ignore_index=True)
for rl in ["shift1-20", "other", "missing", "equal", "substring"]:
    x = d[d.rel == rl]
    h = x.groupby(["split", "country", pd.cut(x.p, bins, right=False)], observed=False).size().unstack()
    h = h.div(h.sum(axis=1), axis=0).round(3)
    tr = x.groupby(["split", "country"]).agg(n=("p", "size"), name_eq=("name_eq", "mean"), addr_empty=("addr_empty", "mean"),
                                             asim=("asim", "mean"), true_share=("true", "mean")).round(3)
    print(f"\n=== {rl}: p distribution (share of pairs) and traits\n{pd.concat([h, tr], axis=1).to_string()}")
