"""Predicted matches per S1 by country, on train (OOF, the run's view) and test, for several runs."""
import json
import sys

import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

for run in sys.argv[1:]:
    cfg = load_config(f"work/runs/{run}/config.yaml")
    pl = Pipeline(cfg)
    sel = json.load(open(pl.run_file("selection.json")))
    for split, f in (("train", "oof.parquet"), ("test", "test_preds.parquet")):
        s1, _ = pl.load_prepared(split)
        ch = postprocess.select(postprocess.assign(pd.read_parquet(pl.run_file(f)), cfg), sel)
        cc = pd.Series(s1["country"].values[ch.s1_idx.values]).value_counts()
        n = pd.Series(s1["country"].values[pl.s1_subset(split, s1)]).value_counts()
        print(run, split, (cc / n).round(4).to_dict())
