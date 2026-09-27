"""How two prediction dirs differ in their selected matches, on train OOF and on test.

    python scripts/compare_sel.py <run> <dir_a> <dir_b>

Per country: pairs added / removed by b relative to a, per 1000 S1 (train and test), and
on train the share of added / removed pairs that are true. If b adds far more on test than
on train, its change is not transferring (as with the density-inflated stage-2 features).
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, da, db = sys.argv[1:4]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
gt = pd.read_parquet(pl.cache("train_gt.parquet"))


def chosen(d, name, split):
    s1 = pd.read_parquet(pl.cache(f"{split}_s1.parquet"), columns=["rid", "country"])
    x = pd.read_parquet(f"{d}/{name}.parquet")
    prm = json.load(open(f"{d}/metrics.json"))["selection"]
    if split == "train":
        x = x[np.isin(x.s1_idx.values, pl.s1_subset("train", s1))]
    sel = postprocess.select(postprocess.assign(x, cfg), prm)
    return set(sel.s1_idx.values.astype(np.int64) * 20_000_000 + sel.r_idx.values), s1


tk = set(gt.s1_idx.values.astype(np.int64) * 20_000_000 + gt.r_idx.values)
for split, name in (("train", "oof"), ("test", "test_preds")):
    a, s1 = chosen(da, name, split)
    b, _ = chosen(db, name, split)
    n_s1 = (s1.country.value_counts() if split == "test"
            else s1.iloc[pl.s1_subset("train", s1)].country.value_counts())
    for lab, diff in (("added", b - a), ("removed", a - b)):
        k = np.fromiter(diff, np.int64)
        cty = s1.country.values[k // 20_000_000]
        per = {c: round(1000 * (cty == c).sum() / n_s1[c], 2) for c in n_s1.index}
        msg = f"{split:5s} {lab:7s} per 1000 S1: {per}"
        if split == "train":
            msg += f"  true share {np.mean([x in tk for x in diff]):.3f}"
        print(msg, flush=True)
