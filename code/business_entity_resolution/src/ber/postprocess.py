"""Turn pair probabilities into final match sets.

1. best-S1: every S2/S3 record keeps only its highest-scoring S1 (in train, no
   record ever matches two S1s).
2. caps: at most cap_s2 S2 and cap_s3 S3 records per S1 (train maxima).
3. set selection per S1, either
   - threshold: keep p >= threshold; output nothing if max p < single_threshold
   - expected_f: choose the top-k (k may be 0) maximising expected F0.5.
   - group_threshold: a threshold per group of pairs (record source x rank of the pair
     within its S1 x empty record address, when an `r_empty` column is present); tuned by
     coordinate descent when postprocess.group_thresholds is true. 2-way cross-validation
     over S1s put it at +0.0002 over one global threshold.
"""
import itertools

import numpy as np
import pandas as pd

from .metrics import f05_from_counts, report, summarize


def with_empty(d, s23):
    """Add the record's empty-address flag, used by group_threshold selection."""
    return d.assign(r_empty=s23["addr_empty"].values[d["r_idx"].values])


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


def pair_groups(d, use_empty):
    """Group id per pair of an assign() output (sorted by S1, then p descending)."""
    rank = d.groupby("s1_idx").cumcount().values
    g = (d["source"].values.astype(int) - 2) * 3 + np.minimum(rank, 2)
    if use_empty:
        g = g * 2 + d["r_empty"].values.astype(int)
    return g


def select(d, params):
    if params["mode"] == "expected_f":
        return select_expected_f(d)
    if params["mode"] == "group_threshold":
        th = np.asarray(params["thresholds"])
        return d[d["p"].values >= th[pair_groups(d, params["use_empty"])]]
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
    if cfg["postprocess"].get("group_thresholds"):
        use_empty = "r_empty" in d.columns
        grp = pair_groups(d, use_empty)
        th = np.full(grp.max() + 1, best.get("threshold", 0.65))

        def g_score(th):
            m = p >= th[grp]
            f = f05_from_counts(np.bincount(s1[m], minlength=n), np.bincount(s1[m & y], minlength=n), G)
            return score(summarize(f, s1_country, mix))

        cur = g_score(th)
        for _ in range(2):
            for k in range(len(th)):
                for t in np.round(np.arange(0.3, 0.91, 0.05), 2):
                    trial = th.copy()
                    trial[k] = t
                    sc = g_score(trial)
                    if sc > cur:
                        cur, th = sc, trial
        log(f"  group thresholds {th.tolist()} -> {cur:.4f}")
        if cur > best_score:
            best = {"mode": "group_threshold", "thresholds": th.tolist(), "use_empty": use_empty}
    return best
