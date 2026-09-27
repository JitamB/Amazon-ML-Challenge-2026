"""Full vs density-robust weight in the final four-model average, and label shift per country.

    python scripts/weights_exp.py <w_full> [<w_full> ...]

Leaderboard evidence (v11): full 0.9692, robust 0.9721, 50/50 0.9733. A parabola through
these three points peaks at w_full ~= 0.36. For each weight: validation of the weighted
average (v13 candidates; a pair missing from v12 falls back to the same-kind v13 model),
EM test priors per country, and a submission with the France correction written to
work/runs/final_w<w>/ (plus a variant also correcting India/US where EM finds a lower prior).
"""
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

FULL, ROBUST = ["v13_keys_ties", "v12_caps2"], ["v13b_s2robust", "v12b_s2robust"]
cfg = load_config("work/runs/v13_keys_ties/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"].get("test_country_mix")
s1, _ = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
s1t = pd.read_parquet(pl.cache("test_s1.parquet"), columns=["rid", "country"])
s23t = pd.read_parquet(pl.cache("test_s23.parquet"), columns=["rid"])


def load(name):
    base = pd.read_parquet(f"work/runs/v13_keys_ties/{name}.parquet")
    cols = {}
    for group in (FULL, ROBUST):
        first = pd.read_parquet(f"work/runs/{group[0]}/{name}.parquet", columns=["s1_idx", "r_idx", "p"])
        p0 = base[["s1_idx", "r_idx"]].merge(first, on=["s1_idx", "r_idx"], how="left")["p"].values
        ps = [p0]
        for r in group[1:]:
            o = pd.read_parquet(f"work/runs/{r}/{name}.parquet", columns=["s1_idx", "r_idx", "p"])
            m = base[["s1_idx", "r_idx"]].merge(o, on=["s1_idx", "r_idx"], how="left")["p"].values
            ps.append(np.where(np.isfinite(m), m, p0))
        cols["full" if group is FULL else "robust"] = np.mean(ps, axis=0)
    return base, cols


oof, oc = load("oof")
test, tc = load("test_preds")
n_r = int(max(oof.r_idx.max(), gt.r_idx.max())) + 1
y = np.isin(oof.s1_idx.values * n_r + oof.r_idx.values, gt.s1_idx.values * n_r + gt.r_idx.values)
oof_cty = s1.country.values[oof.s1_idx.values]
test_cty = s1t.country.values[test.s1_idx.values]


def shift(p, pi0, pi):
    a, b = pi / pi0, (1 - pi) / (1 - pi0)
    return a * p / (a * p + b * (1 - p))


def em(p, pi0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi = pi0
    for _ in range(200):
        new = shift(p, pi0, pi).mean()
        if abs(new - pi) < 1e-8:
            break
        pi = new
    return pi


def write(d, params, out):
    chosen = postprocess.select(postprocess.assign(d, cfg), params)
    os.makedirs(out, exist_ok=True)
    ids = pd.DataFrame({"s1_id": s1t.rid.values[chosen.s1_idx.values], "rid": s23t.rid.values[chosen.r_idx.values]})
    write_id_lists(f"{out}/matching_results.tsv", s1t.rid.tolist(), ids, "matched_entity_ids")
    shutil.copy("work/runs/v13_keys_ties/submission/candidate_pairs.tsv", out)
    cc = s1t.country.values[chosen.s1_idx.values]
    return {c: round((cc == c).sum() / (s1t.country == c).sum(), 4) for c in ["India", "US", "France"]}


for w in map(float, sys.argv[1:]):
    po = w * oc["full"] + (1 - w) * oc["robust"]
    d = oof.assign(p=po.astype(np.float32))
    country, g, (a,) = pl.train_eval_view(s1, gt, postprocess.assign(d, cfg))
    params = postprocess.tune(a, g, country, cfg, log=lambda *_: None)
    res = metrics.report(postprocess.select(a, params), g, country, mix)
    pt = w * tc["full"] + (1 - w) * tc["robust"]
    p_fr, p_all = pt.copy(), pt.copy()
    pri = {}
    for c in ["India", "US", "France"]:
        m = test_cty == c
        pi0 = y[oof_cty == c].mean() if c != "France" else y.mean()
        pi = em(pt[m], pi0)
        pri[c] = (round(pi0, 4), round(pi, 4))
        if pi < pi0:
            p_all[m] = shift(np.clip(pt[m], 1e-6, 1 - 1e-6), pi0, pi)
            if c == "France":
                p_fr[m] = p_all[m]
    mf = write(test.assign(p=p_fr.astype(np.float32)), params, f"work/runs/final_w{w}")
    ma = write(test.assign(p=p_all.astype(np.float32)), params, f"work/runs/final_w{w}_emall")
    print(f"w_full {w}: validation {res['f05_test_mix']:.5f} (India {res['f05_India']:.5f}, US {res['f05_US']:.5f}) {params}")
    print(f"   priors train -> EM test {pri}")
    print(f"   matches/S1 France-only shift {mf} | all-country shift {ma}", flush=True)
