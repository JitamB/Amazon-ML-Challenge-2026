"""Exact replica of the competition metric: F0.5 per S1 entity, macro-averaged."""
import numpy as np
import pandas as pd


def per_entity_f05(pred, gt, n_s1):
    """pred, gt: frames with integer columns s1_idx, r_idx. Returns F0.5 for every S1 index."""
    P = np.bincount(pred["s1_idx"].values, minlength=n_s1)
    G = np.bincount(gt["s1_idx"].values, minlength=n_s1)
    tp_pairs = pred.merge(gt, on=["s1_idx", "r_idx"])
    TP = np.bincount(tp_pairs["s1_idx"].values, minlength=n_s1)
    # F0.5 = 1.25*TP / (P + 0.25*G); an empty prediction for a singleton scores 1.
    return f05_from_counts(P, TP, G)


def f05_from_counts(P, TP, G):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P + G == 0, 1.0, 1.25 * TP / (P + 0.25 * G))


def summarize(f, s1_country, mix=None):
    """Macro F0.5 overall and per country from per-entity scores; `mix` re-weights countries."""
    df = pd.DataFrame({"f": f, "country": s1_country})
    out = {"macro_f05": float(f.mean())}
    by = df.groupby("country")["f"].mean()
    out.update({f"f05_{c}": float(v) for c, v in by.items()})
    if mix:
        present = {c: w for c, w in mix.items() if c in by.index}
        if present:
            tot = sum(present.values())
            out["f05_test_mix"] = float(sum(by[c] * w for c, w in present.items()) / tot)
    return out


def report(pred, gt, s1_country, mix=None):
    return summarize(per_entity_f05(pred, gt, len(s1_country)), s1_country, mix)


def blocking_recall(pairs, gt, s1_country):
    hit = gt.merge(pairs[["s1_idx", "r_idx"]], on=["s1_idx", "r_idx"])
    out = {"block_recall": len(hit) / max(len(gt), 1),
           "cands_per_s1": len(pairs) / max(len(s1_country), 1)}
    c = pd.Series(s1_country)
    for country in c.unique():
        g = gt[c.values[gt["s1_idx"].values] == country]
        h = hit[c.values[hit["s1_idx"].values] == country]
        out[f"block_recall_{country}"] = len(h) / max(len(g), 1)
    return out
