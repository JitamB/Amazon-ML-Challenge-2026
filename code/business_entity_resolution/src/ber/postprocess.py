"""Turn pair probabilities into final match sets.

1. best-S1: every S2/S3 record keeps only its highest-scoring S1 (in train, no
   record ever matches two S1s).
2. caps: at most cap_s2 S2 and cap_s3 S3 records per S1 (train maxima).
3. set selection per S1, either
   - threshold: keep p >= threshold; output nothing if max p < single_threshold
   - expected_f: choose the top-k (k may be 0) maximising expected F0.5.
"""
import itertools

import numpy as np
import pandas as pd

from .metrics import f05_from_counts, report, summarize


def assign(pairs, cfg):
    """pairs: s1_idx, r_idx, source, p. Applies p_min, best-S1 and caps."""
    pp = cfg["postprocess"]
    d = pairs[pairs["p"] >= pp["p_min"]]
    if pp["best_s1"]:
        d = d.loc[d.groupby("r_idx")["p"].idxmax()]
    d = d.sort_values(["s1_idx", "p"], ascending=[True, False])
    rank = d.groupby(["s1_idx", "source"]).cumcount()
    cap = np.where(d["source"].values == 2, pp["cap_s2"], pp["cap_s3"])
    return d[rank.values < cap].reset_index(drop=True)


def select_threshold(d, threshold, single_threshold):
    keep = d["p"] >= threshold
    top = d.groupby("s1_idx")["p"].transform("max")
    return d[keep & (top >= single_threshold)]


def select_expected_f(d):
    """Per S1, pick k maximising E[F0.5] ~= 1.25*sum(top-k p) / (k + 0.25*E[G]); k=0 scores P(no match)."""
    d = d.sort_values(["s1_idx", "p"], ascending=[True, False])
    g = d.groupby("s1_idx")["p"]
    k = g.cumcount().values + 1
    eg = g.transform("sum").values
    val = 1.25 * g.cumsum().values / (k + 0.25 * eg)
    log_q = pd.Series(np.log1p(-np.clip(d["p"].values, 0, 1 - 1e-6)))
    p_empty = np.exp(log_q.groupby(d["s1_idx"].values).transform("sum").values)
    best_val = pd.Series(val).groupby(d["s1_idx"].values).transform("max").values
    best_k = pd.Series(np.where(val == best_val, k, 10**9)).groupby(d["s1_idx"].values).transform("min").values
    keep = (k <= best_k) & (best_val > p_empty)
    return d[keep]


def select(d, params):
    if params["mode"] == "expected_f":
        return select_expected_f(d)
    return select_threshold(d, params["threshold"], params["single_threshold"])


def tune(d, gt, s1_country, cfg, log=print):
    """Grid-search selection parameters on out-of-fold predictions.

    Threshold rules are scored from counts (correctness of each pair is computed once),
    so the whole grid takes seconds instead of re-running the metric per setting.
    """
    mix = cfg["eval"].get("test_country_mix")
    n = len(s1_country)
    s1 = d["s1_idx"].values
    y = d.merge(gt.assign(_y=1), on=["s1_idx", "r_idx"], how="left")["_y"].fillna(0).values.astype(bool)
    G = np.bincount(gt["s1_idx"].values, minlength=n)
    p = d["p"].values
    top = d.groupby("s1_idx")["p"].transform("max").values

    def score(r):
        return r.get("f05_test_mix", r["macro_f05"])

    results = [({"mode": "expected_f"}, score(report(select_expected_f(d), gt, s1_country, mix)))]
    ts = np.round(np.arange(0.30, 0.951, 0.05), 2)
    for t, st in itertools.product(ts, ts):
        if st < t:
            continue
        m = (p >= t) & (top >= st)
        f = f05_from_counts(np.bincount(s1[m], minlength=n), np.bincount(s1[m & y], minlength=n), G)
        results.append(({"mode": "threshold", "threshold": float(t), "single_threshold": float(st)},
                         score(summarize(f, s1_country, mix))))
    best, best_score = max(results, key=lambda x: x[1])
    log(f"  expected_f -> {results[0][1]:.4f}; best selection {best} -> {best_score:.4f}")
    return best
