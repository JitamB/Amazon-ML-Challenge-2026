"""Second-stage (stacking) features: how a pair's first-stage score compares with the
competing pairs of the same record and of the same S1.

A record's true S1 is usually its best-scoring one, and a decoy usually has a better
home elsewhere; singletons have no strong candidate at all. The first-stage model sees
each pair in isolation, so these relations are given to a second model. In training the
inputs are out-of-fold probabilities, so no label information leaks in.
"""
import numpy as np
import pandas as pd

GROUP_SOURCES = ["p", "n_tset", "a_tset", "num_jac"]


def _max_other(groups, values):
    """For each row: the max of `values` over the *other* rows of its group (-1 if none)."""
    order = np.lexsort((-values, groups))
    g_sorted = groups[order]
    first = np.r_[True, g_sorted[1:] != g_sorted[:-1]]
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, len(order)])
    top = values[order[start]]
    second = np.where(size > 1, values[order[np.minimum(start + 1, len(order) - 1)]], -1.0)
    grp_id = np.cumsum(first) - 1
    is_top_row = np.zeros(len(values), dtype=bool)
    is_top_row[order[start]] = True
    top_r, second_r = np.empty(len(values)), np.empty(len(values))
    top_r[order] = top[grp_id]
    second_r[order] = second[grp_id]
    return np.where(is_top_row, second_r, top_r).astype(np.float32)


def group_features(d):
    """d: frame with s1_idx, r_idx and the GROUP_SOURCES columns. Returns new columns (same row order)."""
    out = {}
    s1, r = d["s1_idx"].values, d["r_idx"].values
    for col in GROUP_SOURCES:
        v = d[col].values.astype(np.float64)
        out[f"{col}_maxother_r"] = _max_other(r, v)
        out[f"{col}_maxother_s1"] = _max_other(s1, v)
    p = pd.Series(d["p"].values)
    out["p_rank_r"] = p.groupby(r).rank(ascending=False, method="min").values.astype(np.float32)
    out["p_rank_s1"] = p.groupby(s1).rank(ascending=False, method="min").values.astype(np.float32)
    high = (p > 0.5).astype(np.float32)
    out["s1_n_high"] = high.groupby(s1).transform("sum").values.astype(np.float32)
    out["r_n_high"] = high.groupby(r).transform("sum").values.astype(np.float32)
    out["s1_sum_p"] = p.groupby(s1).transform("sum").values.astype(np.float32)
    out["p1"] = d["p"].values.astype(np.float32)
    return pd.DataFrame(out, index=d.index)
