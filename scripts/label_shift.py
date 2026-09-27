"""Re-select a run's test matches after correcting for label shift (no retraining).

The test has more mid-confidence candidates per S1 than train (India: 0.47 vs 0.38 with
ranker score 0.5-0.9), i.e. a lower share of true matches among candidates. A model
calibrated on train is then over-confident on test. Following Saerens et al. (2002), the
test prior is estimated per country by EM on the test predictions alone, and every p is
re-weighted by the prior ratio before the usual best-S1 / caps / expected-F0.5 selection.

    python scripts/label_shift.py v8_density [strength] [preds_dir] [--only=France] [--strata]

--strata estimates the shift separately per record source x empty address (a leave-one-
country-out simulation, train on US and predict India, preferred it to one global prior).

strength 1.0 applies the full estimated shift, 0.5 half of it (in log-odds). Writes
work/runs/<run>/label_shift_<strength>/matching_results.tsv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
only = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")]
strata = "--strata" in sys.argv
run, strength = args[0], float(args[1]) if len(args) > 1 else 1.0
rd = args[2] if len(args) > 2 else f"work/runs/{run}"
cfg = load_config(f"work/runs/{run}/config.yaml")
cache = f"work/cache/dataset{cfg['data'].get('cache_tag', '')}"
s1_tr = pd.read_parquet(f"{cache}/train_s1.parquet", columns=["country"])
s1_te = pd.read_parquet(f"{cache}/test_s1.parquet", columns=["rid", "country"])
s23_te = pd.read_parquet(f"{cache}/test_s23.parquet", columns=["rid", "addr_empty"])
empty_tr = pd.read_parquet(f"{cache}/train_s23.parquet", columns=["addr_empty"])["addr_empty"].values
gt = pd.read_parquet(f"{cache}/train_gt.parquet")
oof = pd.read_parquet(f"{rd}/oof.parquet")
test = pd.read_parquet(f"{rd}/test_preds.parquet")
params = (json.load(open(f"{rd}/metrics.json"))["selection"] if os.path.exists(f"{rd}/metrics.json")
          and "selection" in json.load(open(f"{rd}/metrics.json")) else json.load(open(f"work/runs/{run}/selection.json")))

n_r = int(max(oof["r_idx"].max(), gt["r_idx"].max())) + 1
y = np.isin(oof["s1_idx"].values * n_r + oof["r_idx"].values, gt["s1_idx"].values * n_r + gt["r_idx"].values)
oof_cty = s1_tr["country"].values[oof["s1_idx"].values]
prior_all = y.mean()
print(f"train prior (true share of candidate pairs) {prior_all:.4f}; mean OOF p {oof['p'].mean():.4f}")


def shift(p, pi0, pi):
    a, b = pi / pi0, (1 - pi) / (1 - pi0)
    return a * p / (a * p + b * (1 - p))


test_cty = s1_te["country"].values[test["s1_idx"].values]
stra_tr = oof["source"].values * 2 + empty_tr[oof["r_idx"].values] if strata else np.zeros(len(oof), int)
stra_te = (test["source"].values * 2 + s23_te["addr_empty"].values[test["r_idx"].values] if strata
           else np.zeros(len(test), int))
lo = lambda x: np.log(x / (1 - x))
p_new = test["p"].values.astype(np.float64).copy()
for c in np.unique(test_cty):
    if only and c not in only:
        continue
    for k in np.unique(stra_te):
        m = (test_cty == c) & (stra_te == k)
        mt = (oof_cty == c) if (oof_cty == c).any() else np.ones(len(oof), bool)
        pi0 = y[mt & (stra_tr == k)].mean()
        p = np.clip(test["p"].values[m].astype(np.float64), 1e-6, 1 - 1e-6)
        pi = pi0
        for _ in range(100):
            new = shift(p, pi0, pi).mean()
            if abs(new - pi) < 1e-7:
                break
            pi = new
        # partial strength: move the log-odds of the prior only part of the way
        pi_s = 1 / (1 + np.exp(-(lo(pi0) + strength * (lo(pi) - lo(pi0)))))
        p_new[m] = shift(p, pi0, pi_s)
        print(f"{c} stratum {k}: train prior {pi0:.4f}, test EM prior {pi:.4f}, used {pi_s:.4f}; "
              f"mean p {p.mean():.4f} -> {p_new[m].mean():.4f}")

d = test.assign(p=p_new.astype(np.float32))
chosen = postprocess.select(postprocess.assign(d, cfg), params)
old = postprocess.select(postprocess.assign(test, cfg), params)
for name, x in (("original", old), ("shifted", chosen)):
    cc = s1_te["country"].values[x["s1_idx"].values]
    print(name, "matches per S1:", {c: round((cc == c).sum() / (s1_te['country'] == c).sum(), 4)
                                    for c in s1_te["country"].unique()},
          "empty share", round(1 - x["s1_idx"].nunique() / len(s1_te), 4))
out = f"{rd.rstrip('/')}_shift{strength}{'_' + '_'.join(only) if only else ''}{'_strata' if strata else ''}"
os.makedirs(out, exist_ok=True)
ids = pd.DataFrame({"s1_id": s1_te["rid"].values[chosen["s1_idx"].values],
                    "rid": s23_te["rid"].values[chosen["r_idx"].values]})
write_id_lists(f"{out}/matching_results.tsv", s1_te["rid"].tolist(), ids, "matched_entity_ids")
print("wrote", out)
