"""Correct the predictions of countries absent from train (France) for label shift, then write output/.

    python code/business_entity_resolution/src/france_shift.py <config_run> <preds_dir> [--strength=1.0] [--strata]

The model is calibrated on India + US, where 53% of candidate pairs are true. French S1s get
more candidates (8.2 vs 6.3-6.9 per S1), so a smaller share of them is true and the model is
over-confident there. Following Saerens et al. (2002), the French prior is estimated by EM
on the French test predictions alone (0.42), every French p is re-weighted by the prior
ratio, and the usual best-S1 / caps / selection is re-run. India and US are left unchanged.

A leave-one-country-out simulation (train on US, predict India) found this correction
helps an unseen country whose prior is lower than train's (+0.0007 F0.5), and that the
reverse case (prior higher) should not be corrected, hence countries with labels, or an
estimated prior above train's, are skipped. On the leaderboard it added +0.0004.
Reads <preds_dir>/{oof,test_preds}.parquet and metrics.json (selection); writes
<preds_dir>_france/ and copies both submission files to output/.

Options, from the same simulation (US -> India):
- --strength=s moves the prior s times as far in log-odds (2.0: +0.0013 on the unseen country, against +0.0007 at 1.0);
- --strata estimates the shift separately per record source x empty address (+0.0011).
Both extrapolate from a smaller shift than France's, so the default stays at the leaderboard-tested 1.0, global.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
run, rd = args[0], args[1].rstrip("/")
s_arg = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--strength=")]
strength = float(s_arg[0]) if s_arg else 1.0
strata = "--strata" in sys.argv
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
s1_tr = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["country"])
s1_te = pd.read_parquet(pl.cache("test_s1.parquet"), columns=["rid", "country"])
s23_te = pd.read_parquet(pl.cache("test_s23.parquet"), columns=["rid", "addr_empty"])
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{rd}/oof.parquet")
test = pd.read_parquet(f"{rd}/test_preds.parquet")
params = json.load(open(f"{rd}/metrics.json"))["selection"]

n_r = int(max(oof["r_idx"].max(), gt["r_idx"].max())) + 1
y = np.isin(oof["s1_idx"].values * n_r + oof["r_idx"].values, gt["s1_idx"].values * n_r + gt["r_idx"].values)
train_countries = set(s1_tr["country"].values[oof["s1_idx"].values])


def shift(p, pi0, pi):
    a, b = pi / pi0, (1 - pi) / (1 - pi0)
    return a * p / (a * p + b * (1 - p))


def logit(x):
    return np.log(x / (1 - x))


test_cty = s1_te["country"].values[test["s1_idx"].values]
if strata:
    empty_tr = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["addr_empty"])["addr_empty"].values
    st_tr = oof["source"].values * 2 + empty_tr[oof["r_idx"].values]
    st_te = test["source"].values * 2 + s23_te["addr_empty"].values[test["r_idx"].values]
else:
    st_tr, st_te = np.zeros(len(oof), int), np.zeros(len(test), int)
p_new = test["p"].values.astype(np.float64).copy()
for c in sorted(set(test_cty) - train_countries):
    for k in np.unique(st_te[test_cty == c]):
        m = (test_cty == c) & (st_te == k)
        pi0 = y[st_tr == k].mean()
        p = np.clip(p_new[m], 1e-6, 1 - 1e-6)
        pi = pi0
        for _ in range(100):
            new = shift(p, pi0, pi).mean()
            if abs(new - pi) < 1e-7:
                break
            pi = new
        print(f"{c}{f' stratum {k}' if strata else ''}: train prior {pi0:.4f}, EM test prior {pi:.4f}")
        if pi < pi0:
            pi_s = 1 / (1 + np.exp(-(logit(pi0) + strength * (logit(pi) - logit(pi0)))))
            p_new[m] = shift(p, pi0, pi_s)

chosen = postprocess.select(postprocess.assign(postprocess.with_empty(test.assign(p=p_new.astype(np.float32)), s23_te), cfg), params)
cc = s1_te["country"].values[chosen["s1_idx"].values]
print("matches per S1:", {c: round((cc == c).sum() / (s1_te["country"] == c).sum(), 4) for c in s1_te["country"].unique()})
out = f"{rd}_france"
os.makedirs(out, exist_ok=True)
ids = pd.DataFrame({"s1_id": s1_te["rid"].values[chosen["s1_idx"].values],
                    "rid": s23_te["rid"].values[chosen["r_idx"].values]})
write_id_lists(f"{out}/matching_results.tsv", s1_te["rid"].tolist(), ids, "matched_entity_ids")
shutil.copy(f"{rd}/candidate_pairs.tsv", out)
od = cfg["paths"]["output_dir"]
os.makedirs(od, exist_ok=True)
for name in ("matching_results.tsv", "candidate_pairs.tsv"):
    shutil.copy(f"{out}/{name}", od)
print("wrote", out, "and copied to", od)
