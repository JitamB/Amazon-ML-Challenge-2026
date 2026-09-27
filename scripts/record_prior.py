"""Record-level prior for records without any address number.

Decoys are made by shifting the house number, so a record with no number at all is a true
match of *some* S1 97.2% of the time (empty address: 97.7%), against 73% for other records.
The pair model spreads such a record's probability over several namesake S1s and often leaves
all of them below the selection threshold. Here each no-number record's probabilities are
renormalised across its candidate S1s: p' = max(p, q * p / sum(p)) when sum(p) >= tau.
q and tau are tuned on out-of-fold predictions; --apply writes a test submission.

    python scripts/record_prior.py <config_run> <preds_dir> [--apply]
"""
import itertools
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, pdir = sys.argv[1], sys.argv[2]
apply = "--apply" in sys.argv
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"]["test_country_mix"]


def adjust(d, s23, q, tau):
    nonum = (s23["hn"].values == "")[d.r_idx.values]
    s = d.groupby("r_idx")["p"].transform("sum").values
    p = d["p"].values.astype(np.float64)
    new = np.where(nonum & (s >= tau), np.maximum(p, q * p / np.maximum(s, 1e-9)), p)
    return d.assign(p=new.astype(np.float32))


s1, s23 = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{pdir}/oof.parquet")[["s1_idx", "r_idx", "source", "p"]]
sel = json.load(open(f"{pdir}/metrics.json"))["selection"] if "selection" in json.load(open(f"{pdir}/metrics.json")) \
    else json.load(open(f"work/runs/{run}/selection.json"))


def score(d, params=None):
    country, g, (a,) = pl.train_eval_view(s1, gt, postprocess.assign(d, cfg))
    if params is None:
        params = postprocess.tune(a, g, country, cfg, log=lambda m: None)
    return metrics.report(postprocess.select(a, params), g, country, mix), params


base, _ = score(oof, sel)
print(f"base: {base['f05_test_mix']:.4f} (India {base['f05_India']:.4f}, US {base['f05_US']:.4f})", flush=True)
best = (base["f05_test_mix"], None, None, sel)
for q, tau in itertools.product([0.9, 0.97, 1.0], [0.05, 0.15, 0.3, 0.5]):
    r, params = score(adjust(oof, s23, q, tau), sel)
    print(f"q={q} tau={tau}: {r['f05_test_mix']:.4f} (India {r['f05_India']:.4f}, US {r['f05_US']:.4f})", flush=True)
    if r["f05_test_mix"] > best[0]:
        best = (r["f05_test_mix"], q, tau, params)
print("best", best[:3], flush=True)
if apply and best[1] is not None:
    _, q, tau, params = best
    r, params = score(adjust(oof, s23, q, tau))  # re-tune selection with the adjustment
    print(f"retuned: {r['f05_test_mix']:.4f} {params}", flush=True)
    s1t, s23t = pl.load_prepared("test")
    test = adjust(pd.read_parquet(f"{pdir}/test_preds.parquet")[["s1_idx", "r_idx", "source", "p"]], s23t, q, tau)
    ch = postprocess.select(postprocess.assign(test, cfg), params)
    out = f"{pdir.rstrip('/')}_prior"
    os.makedirs(out, exist_ok=True)
    ids = pd.DataFrame({"s1_id": s1t["rid"].values[ch.s1_idx.values], "rid": s23t["rid"].values[ch.r_idx.values]})
    write_id_lists(f"{out}/matching_results.tsv", s1t["rid"].tolist(), ids, "matched_entity_ids")
    shutil.copy(f"work/runs/{run}/submission/candidate_pairs.tsv", out)
    cc = s1t["country"].values[ch.s1_idx.values]
    print("test matches per S1:", {c: round((cc == c).sum() / (s1t["country"] == c).sum(), 4) for c in s1t["country"].unique()})
    json.dump({"q": q, "tau": tau, "selection": params, **r}, open(f"{out}/metrics.json", "w"), indent=1)
    print("wrote", out)
