"""Upper bound on the blocking recall the short-token phonetic fix could add.

    python scripts/phonetic_recall.py <run>

For the run's missed train links: does the S1 and its record share a phonetic-word (pt) or
phonetic-compact (pc) key that is under the pass-1 caps (<= 30 S1s, <= 400 S2/S3 records
per country), with the old codes and with the new ones?
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
sys.path.insert(0, "scripts")
from ber.config import load_config  # noqa: E402
from ber.normalize import LEGAL  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402
from phonetic_check import new, old  # noqa: E402

run = sys.argv[1]
pl = Pipeline(load_config(f"work/runs/{run}/config.yaml"))
s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["rid", "country", "name"])
s23 = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["country", "name"])
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
view = pl.s1_subset("train", s1)
gt = gt[np.isin(gt.s1_idx.values, view)]
pairs = pd.read_parquet(f"work/runs/{run}/train_pairs.parquet", columns=["s1_idx", "r_idx"])
n_r = len(s23)
miss = gt[~np.isin(gt.s1_idx.values.astype(np.int64) * n_r + gt.r_idx.values,
                   pairs.s1_idx.values.astype(np.int64) * n_r + pairs.r_idx.values)]
print(f"missed links {len(miss):,} of {len(gt):,}")
for label, fn in (("old", old), ("new", new)):
    legal = {fn(t) for t in LEGAL | {"private", "limited", "praivet"}}

    def keys(country, name):
        c = [fn(t) for t in name.split() if t]
        c = [x for x in c if x and x not in legal]
        k = {"pt|" + x for x in c if len(x) >= 3}
        pc = "".join(c)
        if len(pc) >= 4:
            k.add("pc|" + pc)
        return {country + "|" + x for x in k}

    df1 = pd.Series([k for i in view for k in keys(s1.country.values[i], s1.name.values[i])]).value_counts()
    df2 = pd.Series([k for c, n in zip(s23.country.values, s23.name.values) for k in keys(c, n)]).value_counts()
    ok = set(df1.index[df1.values <= 30]) & set(df2.index[df2.values <= 400])
    hit = [bool(keys(s1.country.values[a], s1.name.values[a]) & keys(s23.country.values[b], s23.name.values[b]) & ok)
           for a, b in zip(miss.s1_idx.values, miss.r_idx.values)]
    print(f"{label}: missed links sharing an allowed phonetic key {np.mean(hit):.4f} "
          f"({np.sum(hit):,}); allowed phonetic keys {len(ok):,}", flush=True)
