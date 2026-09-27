"""Train (out-of-fold) vs test score distributions per country: ranker score and final p."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run = sys.argv[1]
pl = Pipeline(load_config(f"work/runs/{run}/config.yaml"))
bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001]
for split, pf in (("train", "oof.parquet"), ("test", "test_preds.parquet")):
    s1, _ = pl.load_prepared(split)
    d = pd.read_parquet(pl.run_file(pf))
    pairs = pd.read_parquet(pl.run_file(f"{split}_pairs.parquet"), columns=["s1_idx", "r_idx", "cscore"])
    d = d.merge(pairs, on=["s1_idx", "r_idx"], how="left")
    d["country"] = s1["country"].values[d.s1_idx.values]
    n = pd.Series(s1["country"].values[pl.s1_subset(split, s1)]).value_counts()
    for col in ("cscore", "p"):
        h = d.groupby(["country", pd.cut(d[col], bins, right=False)], observed=False).size().unstack().div(n, axis=0)
        print(f"{split} {col}: pairs per S1 by score bin\n{h.round(3).to_string()}")
