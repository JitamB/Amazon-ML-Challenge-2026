"""Leave-one-country-out simulation of the France situation: train on US only, predict India.

    python scripts/loco_sim.py <run> [train_country test_country]

India plays the unseen country. Its F0.5 is measured with no correction and with each
candidate correction for an unlabelled country:
  - EM label-shift (Saerens) at several strengths, global and per stratum;
  - a count prior (true links per S1 taken from the training country);
  - raising the threshold. (Self-training with pseudo-labels was tried: it lost F0.5.)
Stage-1 features only (the stage-2 inputs would need their own out-of-fold pass).
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.metrics import f05_from_counts  # noqa: E402
from ber.model import feature_columns  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run = sys.argv[1]
A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("US", "India")
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["country"])
empty = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["addr_empty"])["addr_empty"].values
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
f = pd.concat([pd.read_parquet(p) for p in pl.feature_parts("train")], ignore_index=True)
cols = feature_columns(f)
X = f[cols].to_numpy(np.float32)
f = f[["s1_idx", "r_idx", "source", "label"]]
cty = s1.country.values[f.s1_idx.values]
y = f.label.values.astype(bool)
us, ind = np.flatnonzero(cty == A), np.flatnonzero(cty == B)
print(f"train on {A}, unseen country {B}")
rng = np.random.default_rng(0)
hold = (f.s1_idx.values % 5) == 0  # US S1s held out for early stopping and threshold tuning
us_tr = us[~hold[us]]
tr = np.sort(rng.choice(us_tr, min(len(us_tr), 4_000_000), replace=False))
params = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 16}
us_va = us[hold[us]]
es = np.sort(rng.choice(us_va, min(len(us_va), 1_000_000), replace=False))


def fit(rows, labels):
    b = lgb.train(params, lgb.Dataset(X[rows], labels), 1500,
                  valid_sets=[lgb.Dataset(X[es], y[es])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return b


b = fit(tr, y[tr])
p_ind = b.predict(X[ind], num_iteration=b.best_iteration)
p_us = b.predict(X[us_va], num_iteration=b.best_iteration)
print(f"US-only model: best_iter {b.best_iteration}", flush=True)

G = np.bincount(gt.s1_idx, minlength=len(s1))


def f05(rows, p, params_sel):
    d = f.iloc[rows][["s1_idx", "r_idx", "source"]].assign(p=p.astype(np.float32))
    sel = postprocess.select(postprocess.assign(d, cfg), params_sel)
    s1s = np.unique(f.s1_idx.values[rows])
    n = len(s1)
    lab = np.isin(sel.s1_idx.values.astype(np.int64) * 20_000_000 + sel.r_idx.values,
                  gt.s1_idx.values.astype(np.int64) * 20_000_000 + gt.r_idx.values)
    P = np.bincount(sel.s1_idx, minlength=n)
    TP = np.bincount(sel.s1_idx[lab], minlength=n)
    # S1s of this country with links but no candidates also count (F = 0); singletons without candidates = 1
    return f05_from_counts(P, TP, G)[s1s].mean()


# selection tuned on the training country (as for France): best global threshold on US holdout
best = max(((t, f05(us_va, p_us, {"mode": "threshold", "threshold": t, "single_threshold": t}))
            for t in np.round(np.arange(0.4, 0.86, 0.05), 2)), key=lambda x: x[1])
sel = {"mode": "threshold", "threshold": best[0], "single_threshold": best[0]}
print(f"US holdout: threshold {best[0]} -> {best[1]:.5f}")
pi0 = y[tr].mean()
print(f"train prior {pi0:.4f}; India true prior {y[ind].mean():.4f}; mean p on India {p_ind.mean():.4f}")


def shift(p, pi0, pi):
    a, c = pi / pi0, (1 - pi) / (1 - pi0)
    return a * p / (a * p + c * (1 - p))


def em(p, pi0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi = pi0
    for _ in range(200):
        new = shift(p, pi0, pi).mean()
        if abs(new - pi) < 1e-8:
            break
        pi = new
    return pi


lo = lambda x: np.log(x / (1 - x))
res = {"no correction": f05(ind, p_ind, sel),
       "oracle: true India prior": f05(ind, shift(p_ind, pi0, y[ind].mean()), sel)}
pi = em(p_ind, pi0)
print(f"EM prior of the unseen country {pi:.4f}")
for st in (1.0, 1.5, 2.0, 3.0):
    pi_s = 1 / (1 + np.exp(-(lo(pi0) + st * (lo(pi) - lo(pi0)))))
    res[f"EM global, strength {st}"] = f05(ind, shift(p_ind, pi0, pi_s), sel)
# per stratum: source x empty address
strat = f.source.values[ind] * 2 + empty[f.r_idx.values[ind]]
strat_tr = f.source.values[tr] * 2 + empty[f.r_idx.values[tr]]
for st in (1.0, 1.5, 2.0):
    p2 = p_ind.copy()
    for k in np.unique(strat):
        m, mt = strat == k, strat_tr == k
        pk0 = y[tr][mt].mean()
        pk = em(p_ind[m], pk0)
        pk_s = 1 / (1 + np.exp(-(lo(pk0) + st * (lo(pk) - lo(pk0)))))
        p2[m] = shift(p_ind[m], pk0, pk_s)
    res[f"EM per stratum, strength {st}"] = f05(ind, p2, sel)
# count prior: true links per S1 as in the training country
links_per_s1 = y[tr].sum() / len(np.unique(f.s1_idx.values[tr]))
pic = links_per_s1 * len(np.unique(f.s1_idx.values[ind])) / len(ind)
res[f"count prior ({pic:.4f})"] = f05(ind, shift(p_ind, pi0, pic), sel)
for dt in (0.05, 0.1, 0.15):
    t = min(sel["threshold"] + dt, 0.95)
    res[f"threshold +{dt}"] = f05(ind, p_ind, {"mode": "threshold", "threshold": t, "single_threshold": t})
print(f"\n{B} F0.5 (unseen country, model trained on {A}):")
for k, v in res.items():
    print(f"  {k:42s} {v:.5f}  ({v - res['no correction']:+.5f})")
