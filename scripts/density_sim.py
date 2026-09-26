"""Score a run's OOF predictions as if train had the test's density of unmatched records.

The test has 5.75 S2/S3 records per S1, train 4.68 (same links per S1), so ~40% of test
records are unmatched vs ~26% in train. Dropping S1s (hash sample, `frac`) turns their
records into unmatched ones; frac = 4.68 / 5.75 = 0.81 mimics the test.
    python scripts/density_sim.py v5b_fullfold 0.81
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402

run, frac = sys.argv[1], float(sys.argv[2])
cfg = load_config(f"work/runs/{run}/config.yaml")
cache = f"work/cache/dataset{cfg['data'].get('cache_tag', '')}"
s1 = pd.read_parquet(f"{cache}/train_s1.parquet", columns=["rid", "country"])
gt = pd.read_parquet(f"{cache}/train_gt.parquet")
oof = pd.read_parquet(f"work/runs/{run}/oof.parquet")
sel = json.load(open(f"work/runs/{run}/selection.json"))
mix = cfg["eval"].get("test_country_mix")

h = pd.util.hash_array(s1["rid"].values.astype(object)) % 10_000
base_frac = cfg["data"]["train_s1_frac"]
for f in sorted({base_frac, frac}, reverse=True):
    keep = np.flatnonzero(h < f * 10_000)
    pos = np.full(len(s1), -1)
    pos[keep] = np.arange(len(keep))
    remap = lambda x: x[pos[x["s1_idx"].values] >= 0].assign(s1_idx=lambda y: pos[y["s1_idx"].values])
    g, d = remap(gt), remap(oof)
    country = s1["country"].values[keep]
    a = postprocess.assign(d, cfg)
    own = metrics.report(postprocess.select(a, sel), g, country, mix)
    tuned = postprocess.tune(a, g, country, cfg, log=lambda m: None)
    re = metrics.report(postprocess.select(a, tuned), g, country, mix)
    n_rec = d["r_idx"].nunique()
    print(f"frac {f}: S1 {len(keep):,}; own selection {sel} -> {own['f05_test_mix']:.4f} "
          f"(India {own['f05_India']:.4f}, US {own['f05_US']:.4f}); retuned {tuned} -> {re['f05_test_mix']:.4f}; "
          f"pred/S1 {len(postprocess.select(a, sel)) / len(keep):.3f}")
