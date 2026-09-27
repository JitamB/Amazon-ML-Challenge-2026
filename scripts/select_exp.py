"""Selection-rule experiments on a blend's OOF, with honest 2-way cross-validation over S1s.

    python scripts/select_exp.py <run> <blend_dir>

Baseline: the blend's tuned global threshold. Variants are tuned on one half of the S1s and
scored on the other (then swapped): binned (monotone) calibrated expected-F0.5, and group-wise
thresholds (record source x empty address x rank of the pair within its S1).
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.metrics import f05_from_counts, summarize  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, bdir = sys.argv[1], sys.argv[2]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"].get("test_country_mix")
s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["rid", "country"])
empty = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["addr_empty"])["addr_empty"].values
gt_full = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{bdir}/oof.parquet")
base = json.load(open(f"{bdir}/metrics.json"))["selection"]
country, gt, (oofv,) = pl.train_eval_view(s1, gt_full, oof)
n = len(country)
d = postprocess.assign(oofv, cfg).reset_index(drop=True)
n_r = int(oof.r_idx.max()) + 1
y = np.isin(d.s1_idx.values.astype(np.int64) * n_r + d.r_idx.values,
            gt.s1_idx.values.astype(np.int64) * n_r + gt.r_idx.values)
G = np.bincount(gt.s1_idx, minlength=n)
half = (pd.util.hash_array(np.arange(n).astype(np.int64)) % 2).astype(bool)
row_half = half[d.s1_idx.values]
rank = d.groupby("s1_idx").cumcount().values  # assign() sorts by s1, p desc
d["grp"] = (d.source.values - 2) * 6 + empty[d.r_idx.values] * 3 + np.minimum(rank, 2)


def score(mask, s1_mask):
    """test-mix macro F0.5 over the S1s in s1_mask when pairs `mask` are kept."""
    s = d.s1_idx.values
    P = np.bincount(s[mask], minlength=n)
    TP = np.bincount(s[mask & y], minlength=n)
    f = f05_from_counts(P, TP, G)
    r = summarize(f[s1_mask], country[s1_mask], mix)
    return r.get("f05_test_mix", r["macro_f05"])


def thr_mask(p, t, st):
    top = pd.Series(p).groupby(d.s1_idx.values).transform("max").values
    return (p >= t) & (top >= st)


res = {}
for h in (False, True):
    ev = half == h
    res.setdefault("baseline global threshold", []).append(
        score(thr_mask(d.p.values, base.get("threshold", .65), base.get("single_threshold", .65)), ev))
    # isotonic calibration fitted on the other half, then expected-F0.5
    tr = row_half != h
    edges = np.unique(np.quantile(d.p.values[tr], np.linspace(0, 1, 301)))
    b_tr = np.clip(np.searchsorted(edges, d.p.values[tr]) - 1, 0, len(edges) - 2)
    rate = np.bincount(b_tr, y[tr], minlength=len(edges) - 1) / np.maximum(np.bincount(b_tr, minlength=len(edges) - 1), 1)
    rate = np.maximum.accumulate(rate)  # monotone
    pc = rate[np.clip(np.searchsorted(edges, d.p.values) - 1, 0, len(edges) - 2)]
    sel = postprocess.select_expected_f(d.assign(p=pc))
    m = np.zeros(len(d), bool)
    m[sel.index.values] = True
    res.setdefault("calibrated + expected-F0.5", []).append(score(m, ev))
    sel = postprocess.select_expected_f(d)
    m = np.zeros(len(d), bool)
    m[sel.index.values] = True
    res.setdefault("raw expected-F0.5", []).append(score(m, ev))
    # group thresholds by coordinate descent on the other half
    th = np.full(12, base.get("threshold", .65))
    grid = np.round(np.arange(0.3, 0.91, 0.05), 2)
    for _ in range(2):
        for g_ in range(12):
            best = None
            for t in grid:
                th2 = th.copy()
                th2[g_] = t
                sc = score(d.p.values >= th2[d.grp.values], ~ev)
                if best is None or sc > best[0]:
                    best = (sc, t)
            th[g_] = best[1]
    res.setdefault("group thresholds", []).append(score(d.p.values >= th[d.grp.values], ev))
    print(f"half {int(h)} group thresholds (src, empty, rank):", th.tolist(), flush=True)
for k, v in res.items():
    print(f"{k:32s} {np.mean(v):.5f}   halves {[round(x, 5) for x in v]}")
