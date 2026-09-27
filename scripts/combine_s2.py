"""Combine a full-stage-2 run (A) with a density-robust run (B) on the same pairs.
Variants are scored on OOF (the run's view) and their test match counts compared: a variant
that keeps A's validation gains but predicts like B on test avoids the test-only inflation.
    python scripts/combine_s2.py A B [--write variant]
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

A, B = sys.argv[1], sys.argv[2]
write = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
cfg = load_config(f"work/runs/{A}/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"]["test_country_mix"]


def decoy_type(s1, s23, d):
    nsim = process.cpdist(s1["name"].values[d.s1_idx.values], s23["name"].values[d.r_idx.values],
                          scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    a, b = s1["hn"].values[d.s1_idx.values], s23["hn"].values[d.r_idx.values]
    return np.array([bool(x) and bool(y) and x != y and x not in y and y not in x and len(x) < 15 and len(y) < 15
                     and abs(int(x) - int(y)) <= 20 for x, y in zip(a, b)]) & (nsim >= 85)


def load(split, f):
    a = pd.read_parquet(f"work/runs/{A}/{f}")[["s1_idx", "r_idx", "source", "p"]]
    b = pd.read_parquet(f"work/runs/{B}/{f}")[["s1_idx", "r_idx", "p"]].rename(columns={"p": "pb"})
    d = a.merge(b, on=["s1_idx", "r_idx"], how="left")
    d["pb"] = d["pb"].fillna(d["p"])
    s1, s23 = pl.load_prepared(split)
    d["dt"] = decoy_type(s1, s23, d)
    return d, s1, s23


VARIANTS = {
    "A": lambda d: d.p,
    "B": lambda d: d.pb,
    "mean": lambda d: (d.p + d.pb) / 2,
    "min": lambda d: np.minimum(d.p, d.pb),
    "A, B on decoy-type": lambda d: np.where(d.dt, d.pb, d.p),
    "A, min on decoy-type": lambda d: np.where(d.dt, np.minimum(d.p, d.pb), d.p),
}
tr, s1, _ = load("train", "oof.parquet")
te, s1t, s23t = load("test", "test_preds.parquet")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
view = pl.s1_subset("train", s1)
n_tr = pd.Series(s1["country"].values[view]).value_counts()
n_te = s1t["country"].value_counts()
for name, fn in VARIANTS.items():
    dtr = tr.assign(p=fn(tr).astype(np.float32))[["s1_idx", "r_idx", "source", "p"]]
    country, g, (asg,) = pl.train_eval_view(s1, gt, postprocess.assign(dtr, cfg))
    params = postprocess.tune(asg, g, country, cfg, log=lambda m: None)
    r = metrics.report(postprocess.select(asg, params), g, country, mix)
    ch_tr = postprocess.select(postprocess.assign(dtr, cfg), params)
    dte = te.assign(p=fn(te).astype(np.float32))[["s1_idx", "r_idx", "source", "p"]]
    ch = postprocess.select(postprocess.assign(dte, cfg), params)
    ctr = pd.Series(s1["country"].values[ch_tr.s1_idx.values]).value_counts() / n_tr
    cte = pd.Series(s1t["country"].values[ch.s1_idx.values]).value_counts() / n_te
    print(f"{name:>22}: val {r['f05_test_mix']:.4f} (India {r['f05_India']:.4f}, US {r['f05_US']:.4f}) | matches/S1 "
          f"train US {ctr['US']:.4f} India {ctr['India']:.4f} | test US {cte['US']:.4f} India {cte['India']:.4f} France {cte['France']:.4f}")
    if name == write:
        out = f"work/runs/combo_{A}+{B}"
        os.makedirs(out, exist_ok=True)
        ids = pd.DataFrame({"s1_id": s1t["rid"].values[ch.s1_idx.values], "rid": s23t["rid"].values[ch.r_idx.values]})
        write_id_lists(f"{out}/matching_results.tsv", s1t["rid"].tolist(), ids, "matched_entity_ids")
        shutil.copy(f"work/runs/{A}/submission/candidate_pairs.tsv", out)
        json.dump({"variant": name, "selection": params, **r}, open(f"{out}/metrics.json", "w"), indent=1)
        print("wrote", out)
