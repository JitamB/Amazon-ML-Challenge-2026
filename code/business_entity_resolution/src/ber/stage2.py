"""Second-stage (stacking) features, computed from first-stage out-of-fold scores.

1. Competition: how a pair compares with the other pairs of the same record and of the
   same S1 (a record's true S1 is usually its best one; decoys usually have a better home).
2. Head-to-head deltas: the pair's value minus the best competing S1's value for the same
   record, for the signals that separate sibling decoys (numbers, address, name).
3. Mutual support: the records of one entity resemble each other, so each candidate is
   compared with the S1's two strongest *other* candidates (helps alias names and empty
   addresses that sit next to confident matches).

In training all inputs are out-of-fold, so no label information leaks in.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

GROUP_SOURCES = ["p", "n_tset", "a_tset", "num_jac", "num_a_in_b", "aa_jac", "nc_jw", "hn_eq",
                 "hn_a_in_bnums", "cscore"]
DELTA_SOURCES = ["p", "a_tset", "n_tset", "num_jac", "num_a_in_b", "hn_eq"]


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


def _top3(groups, values, items):
    """Per row: the items and values ranked 1-3 within its group (-1 where missing)."""
    order = np.lexsort((-values, groups))
    g_sorted = groups[order]
    first = np.r_[True, g_sorted[1:] != g_sorted[:-1]]
    start = np.flatnonzero(first)
    end = np.r_[start[1:], len(order)]
    grp_id = np.empty(len(values), dtype=np.int64)
    grp_id[order] = np.cumsum(first) - 1
    out_items, out_vals = [], []
    for j in range(3):
        pos = start + j
        ok = pos < end
        it = np.where(ok, items[order[np.minimum(pos, len(order) - 1)]], -1)
        va = np.where(ok, values[order[np.minimum(pos, len(order) - 1)]], -1.0)
        out_items.append(it[grp_id]); out_vals.append(va[grp_id])
    return out_items, out_vals


def _support(d, s23):
    """Similarity of each candidate record to its S1's two strongest other candidates."""
    r = d["r_idx"].values
    (t1, t2, t3), (v1, v2, v3) = _top3(d["s1_idx"].values, d["p"].values.astype(np.float64), r)
    is1, is2 = r == t1, r == t2
    partners = [(np.where(is1, t2, t1), np.where(is1, v2, v1)),
                (np.where(is1 | is2, t3, t2), np.where(is1 | is2, v3, v2))]
    name, addr, hn = s23["name"].values, s23["addr"].values, s23["hn"].values
    out = {}
    for j, (pr, pv) in enumerate(partners, start=1):
        ok = pr >= 0
        a_i, b_i = r[ok], pr[ok]
        for col, arr, scorer in (("name", name, fuzz.token_set_ratio), ("addr", addr, fuzz.token_set_ratio)):
            v = np.full(len(r), -1.0, dtype=np.float32)
            v[ok] = process.cpdist(arr[a_i], arr[b_i], scorer=scorer, workers=-1, dtype=np.float32)
            out[f"sup{j}_{col}"] = v
        h = np.full(len(r), -1.0, dtype=np.float32)
        ha, hb = hn[a_i], hn[b_i]
        h[ok] = np.where((ha == "") | (hb == ""), -1, (ha == hb)).astype(np.float32)
        out[f"sup{j}_hn_eq"] = h
        out[f"sup{j}_p"] = pv.astype(np.float32)
    return out


def group_features(d, s23=None):
    """d: frame with s1_idx, r_idx, p and (some of) GROUP_SOURCES. Returns new columns, same row order."""
    out = {}
    s1, r = d["s1_idx"].values, d["r_idx"].values
    for col in [c for c in GROUP_SOURCES if c in d.columns]:
        v = d[col].values.astype(np.float64)
        out[f"{col}_maxother_r"] = _max_other(r, v)
        out[f"{col}_maxother_s1"] = _max_other(s1, v)
        if col in DELTA_SOURCES:
            out[f"{col}_delta_r"] = (v - out[f"{col}_maxother_r"]).astype(np.float32)
    p = pd.Series(d["p"].values)
    out["p_rank_r"] = p.groupby(r).rank(ascending=False, method="min").values.astype(np.float32)
    out["p_rank_s1"] = p.groupby(s1).rank(ascending=False, method="min").values.astype(np.float32)
    high = (p > 0.5).astype(np.float32)
    out["s1_n_high"] = high.groupby(s1).transform("sum").values.astype(np.float32)
    out["r_n_high"] = high.groupby(r).transform("sum").values.astype(np.float32)
    out["s1_sum_p"] = p.groupby(s1).transform("sum").values.astype(np.float32)
    out["p1"] = d["p"].values.astype(np.float32)
    if s23 is not None:
        out.update(_support(d, s23))
    return pd.DataFrame(out, index=d.index)
