"""Pair features for (S1, S2/S3) candidates.

String similarities use rapidfuzz's C++ `cpdist` (element-wise over two
arrays); set-based number/token features are plain Python. Chunks run in
parallel processes; each task carries only the column slices it needs (sharing
the full frames with forked workers made memory grow with every chunk, as
reference counting copies the shared pages).

To add a feature: compute it in `_chunk_features` and it is picked up
automatically (the model uses every column except those in NON_FEATURES).
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

NON_FEATURES = {"s1_idx", "r_idx", "label", "fold"}

_IDF = None  # set in compute_features; read by workers (small, never mutated)
A_COLS = ["name", "name_compact", "phon", "name_raw", "addr", "addr_alpha", "nums", "hn"]
B_COLS = A_COLS + ["addr_empty", "name_indic", "name_web", "source"]


def _sim(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32)


def _set_feats(a_list, b_list):
    """Overlap features between two space-joined token strings."""
    jac, a_in_b, b_in_a, extra, missing = [], [], [], [], []
    for a, b in zip(a_list, b_list):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            jac.append(-1.0); a_in_b.append(-1.0); b_in_a.append(-1.0)
            extra.append(len(sb)); missing.append(len(sa))
            continue
        inter = len(sa & sb)
        jac.append(inter / len(sa | sb)); a_in_b.append(inter / len(sa)); b_in_a.append(inter / len(sb))
        extra.append(len(sb - sa)); missing.append(len(sa - sb))
    f = lambda x: np.array(x, dtype=np.float32)
    return f(jac), f(a_in_b), f(b_in_a), f(extra), f(missing)


def _idf_overlap(a_list, b_list, idf):
    out_a, out_b = [], []
    for a, b in zip(a_list, b_list):
        sa, sb = set(a.split()), set(b.split())
        wa = sum(idf.get(t, 12.0) for t in sa)
        wb = sum(idf.get(t, 12.0) for t in sb)
        wi = sum(idf.get(t, 12.0) for t in sa & sb)
        out_a.append(wi / wa if wa else -1.0)
        out_b.append(wi / wb if wb else -1.0)
    return np.array(out_a, dtype=np.float32), np.array(out_b, dtype=np.float32)


def _number_diff_feats(ha, hb, na, nb):
    logdiff, substr, lev, a_in_b, b_in_a, min_logdiff = [], [], [], [], [], []
    for x, y, xs, ys in zip(ha, hb, na, nb):
        if not x or not y:
            logdiff.append(-1.0); substr.append(-1.0); lev.append(-1.0)
            a_in_b.append(-1.0); b_in_a.append(-1.0); min_logdiff.append(-1.0)
            continue
        logdiff.append(math.log1p(abs(int(x) - int(y))) if len(x) < 18 and len(y) < 18 else 20.0)
        substr.append(float(x != y and (x in y or y in x)))
        lev.append(float(Levenshtein.distance(x, y)))
        sx, sy = xs.split()[:4], ys.split()[:4]
        a_in_b.append(float(x in sy)); b_in_a.append(float(y in sx))
        diffs = [abs(int(u) - int(v)) for u in sx for v in sy if len(u) < 18 and len(v) < 18]
        min_logdiff.append(math.log1p(min(diffs)) if diffs else -1.0)
    f = lambda v: np.array(v, dtype=np.float32)
    return f(logdiff), f(substr), f(lev), f(a_in_b), f(b_in_a), f(min_logdiff)


def _chunk_features(args):
    a, b = args  # dicts of column arrays for the chunk's S1 side and S2/S3 side
    an, bn = a["name"], b["name"]
    aa, ba = a["addr"], b["addr"]
    f = {}
    # --- names
    f["n_ratio"] = _sim(fuzz.ratio, an, bn)
    f["n_tsort"] = _sim(fuzz.token_sort_ratio, an, bn)
    f["n_tset"] = _sim(fuzz.token_set_ratio, an, bn)
    f["n_partial"] = _sim(fuzz.partial_ratio, an, bn)
    f["nc_jw"] = _sim(JaroWinkler.normalized_similarity, a["name_compact"], b["name_compact"])
    f["nc_partial"] = _sim(fuzz.partial_ratio, a["name_compact"], b["name_compact"])
    f["ph_ratio"] = _sim(fuzz.ratio, a["phon"], b["phon"])
    f["ph_tset"] = _sim(fuzz.token_set_ratio, a["phon"], b["phon"])
    f["raw_ratio"] = _sim(fuzz.ratio, a["name_raw"], b["name_raw"])
    (f["n_jac"], f["n_a_in_b"], f["n_b_in_a"], f["n_extra"], f["n_missing"]) = _set_feats(an, bn)
    (f["ph_jac"], _, _, _, _) = _set_feats(a["phon"], b["phon"])
    f["n_idf_a"], f["n_idf_b"] = _idf_overlap(an, bn, _IDF)
    # --- addresses
    f["a_ratio"] = _sim(fuzz.ratio, aa, ba)
    f["a_tset"] = _sim(fuzz.token_set_ratio, aa, ba)
    f["a_tsort"] = _sim(fuzz.token_sort_ratio, aa, ba)
    f["a_partial"] = _sim(fuzz.partial_ratio, aa, ba)
    (f["aa_jac"], f["aa_a_in_b"], f["aa_b_in_a"], _, _) = _set_feats(a["addr_alpha"], b["addr_alpha"])
    # --- numbers (sibling-decoy killers)
    (f["num_jac"], f["num_a_in_b"], f["num_b_in_a"], f["num_extra"], f["num_missing"]) = \
        _set_feats(a["nums"], b["nums"])
    ha, hb = a["hn"], b["hn"]
    f["hn_eq"] = np.where((ha == "") | (hb == ""), -1, (ha == hb).astype(np.int8)).astype(np.float32)
    f["nums_eq"] = np.where((a["nums"] == "") | (b["nums"] == ""), -1,
                            (a["nums"] == b["nums"])).astype(np.float32)
    # Decoys shift the house number by a little (1-20); true matches drop/insert digits or
    # add unrelated numbers. So *how* the numbers differ matters, not just whether they do.
    (f["hn_logdiff"], f["hn_substr"], f["hn_lev"], f["hn_a_in_bnums"], f["hn_b_in_anums"],
     f["num_min_logdiff"]) = _number_diff_feats(ha, hb, a["nums"], b["nums"])
    # --- record flags
    f["b_addr_empty"] = b["addr_empty"].astype(np.float32)
    f["b_name_indic"] = b["name_indic"].astype(np.float32)
    f["b_name_web"] = b["name_web"].astype(np.float32)
    f["b_source3"] = (b["source"] == 3).astype(np.float32)
    f["a_ntok"] = np.array([len(x.split()) for x in an], dtype=np.float32)
    f["b_ntok"] = np.array([len(x.split()) for x in bn], dtype=np.float32)
    f["a_nnum"] = np.array([len(x.split()) for x in a["nums"]], dtype=np.float32)
    f["b_nnum"] = np.array([len(x.split()) for x in b["nums"]], dtype=np.float32)
    return pd.DataFrame(f)


def build_idf(*frames):
    toks = pd.concat([f["name"] for f in frames]).str.split().explode().dropna()
    counts = toks.value_counts()
    n = sum(len(f) for f in frames)
    return {t: math.log(n / c) for t, c in counts.items()}


def context_features(pairs):
    """Features describing a pair relative to the other candidates of its S1 and of its record."""
    g1 = pairs.groupby("s1_idx")["bscore"]
    gr = pairs.groupby("r_idx")["bscore"]
    return pd.DataFrame({
        "b_rank_s1": g1.rank(ascending=False, method="first").values.astype(np.float32),
        "b_n_s1": g1.transform("size").values.astype(np.float32),
        "b_rel_s1": (pairs["bscore"] / g1.transform("max")).values.astype(np.float32),
        "b_rank_r": gr.rank(ascending=False, method="first").values.astype(np.float32),
        "b_n_r": gr.transform("size").values.astype(np.float32),
        "b_rel_r": (pairs["bscore"] / gr.transform("max")).values.astype(np.float32),
    }, index=pairs.index)


def compute_features(pairs, s1, s23, idf, cfg):
    """Pairwise features for `pairs` (which should already carry its context columns)."""
    global _IDF
    _IDF = idf
    fc = cfg["features"]
    size = fc["chunk_size"]
    a_cols = {c: s1[c].values for c in A_COLS}
    b_cols = {c: s23[c].values for c in B_COLS}

    def tasks():
        for i in range(0, len(pairs), size):
            ai = pairs["s1_idx"].values[i:i + size]
            bi = pairs["r_idx"].values[i:i + size]
            yield {c: v[ai] for c, v in a_cols.items()}, {c: v[bi] for c, v in b_cols.items()}

    if fc["n_jobs"] > 1:
        with ProcessPoolExecutor(fc["n_jobs"]) as ex:
            parts = list(ex.map(_chunk_features, tasks()))
    else:
        parts = [_chunk_features(t) for t in tasks()]
    return pd.concat([pairs.reset_index(drop=True), pd.concat(parts, ignore_index=True)], axis=1)
