"""Learned candidate pruning: the last step of blocking.

The inverted-index lookup (blocking.py) returns up to `blocking.top_n` loose candidates
per S1. A small LightGBM ranker on a few cheap features (name / compact-name / phonetic /
address similarity, number overlap, blocking-score context) then keeps the best
`candidates.keep_k` per S1. That pruned set is what the matching model scores, so it is
what `candidate_pairs.tsv` reports.

Measured on v3: ranking the 80 loose candidates by a good score keeps 95.75% of true links
in the top 10, versus 91.0% when ranking by the raw blocking score.

Train pairs get out-of-fold ranker scores (folds by S1), so pruning on train behaves like
pruning on test. The ranker score is also passed on as a feature (`cscore`).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .features import _set_feats, _sim, context_features

A_COLS = ["name", "name_compact", "phon", "addr", "nums", "hn"]


def _cheap_chunk(args):
    a, b = args
    f = {
        "c_n_tset": _sim(fuzz.token_set_ratio, a["name"], b["name"]),
        "c_nc_jw": _sim(JaroWinkler.normalized_similarity, a["name_compact"], b["name_compact"]),
        "c_ph_tset": _sim(fuzz.token_set_ratio, a["phon"], b["phon"]),
        "c_a_tset": _sim(fuzz.token_set_ratio, a["addr"], b["addr"]),
    }
    f["c_num_jac"], f["c_num_a_in_b"], _, _, _ = _set_feats(a["nums"], b["nums"])
    ha, hb = a["hn"], b["hn"]
    f["c_hn_eq"] = np.where((ha == "") | (hb == ""), -1, (ha == hb).astype(np.int8)).astype(np.float32)
    return pd.DataFrame(f)


def cheap_features(pairs, s1, s23, cfg):
    size = cfg["features"]["chunk_size"]
    a_cols = {c: s1[c].values for c in A_COLS}
    b_cols = {c: s23[c].values for c in A_COLS}

    def tasks():
        for i in range(0, len(pairs), size):
            ai, bi = pairs["s1_idx"].values[i:i + size], pairs["r_idx"].values[i:i + size]
            yield {c: v[ai] for c, v in a_cols.items()}, {c: v[bi] for c, v in b_cols.items()}

    if cfg["features"]["n_jobs"] > 1:
        with ProcessPoolExecutor(cfg["features"]["n_jobs"]) as ex:
            parts = list(ex.map(_cheap_chunk, tasks()))
    else:
        parts = [_cheap_chunk(t) for t in tasks()]
    ctx = context_features(pairs).reset_index(drop=True)
    return pd.concat([pd.DataFrame({"bscore": pairs["bscore"].values}), ctx,
                      pd.concat(parts, ignore_index=True)], axis=1)


def prune(pairs, s1, s23, split, cfg, model_dir, gt=None, log=print):
    """Score loose candidates with the ranker and keep the top keep_k per S1 (gt: train links)."""
    cc = cfg["candidates"]
    pairs = pairs.sort_values(["s1_idx", "r_idx"]).reset_index(drop=True)
    if gt is not None:
        n_r = len(s23)
        labels = np.isin(pairs["s1_idx"].values * n_r + pairs["r_idx"].values,
                         gt["s1_idx"].values * n_r + gt["r_idx"].values).astype(np.int8)
    X = cheap_features(pairs, s1, s23, cfg)
    os.makedirs(model_dir, exist_ok=True)
    score = np.zeros(len(pairs), dtype=np.float32)
    if split == "train":
        folds = pd.util.hash_array(s1["rid"].values.astype(object))[pairs["s1_idx"].values] % cc["n_folds"]
        rng = np.random.default_rng(0)
        for k in range(cc["n_folds"]):
            tr = np.flatnonzero(folds != k)
            if len(tr) > cc["max_train_rows"]:
                tr = np.sort(rng.choice(tr, cc["max_train_rows"], replace=False))
            va = np.flatnonzero(folds == k)
            va_es = rng.choice(va, min(len(va), 2_000_000), replace=False)
            booster = lgb.train(cc["params"], lgb.Dataset(X.iloc[tr], labels[tr]),
                                num_boost_round=cc["num_boost_round"],
                                valid_sets=[lgb.Dataset(X.iloc[va_es], labels[va_es])],
                                callbacks=[lgb.early_stopping(30, verbose=False)])
            score[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
            booster.save_model(os.path.join(model_dir, f"ranker{k}.txt"), num_iteration=booster.best_iteration)
            log(f"  candidate ranker fold {k}: {len(tr):,} train rows, best_iter {booster.best_iteration}")
    else:
        for k in range(cc["n_folds"]):
            booster = lgb.Booster(model_file=os.path.join(model_dir, f"ranker{k}.txt"))
            score += booster.predict(X[booster.feature_name()]).astype(np.float32) / cc["n_folds"]
    pairs["cscore"] = score
    pairs = pairs.sort_values(["s1_idx", "cscore"], ascending=[True, False])
    # Flexible size: up to keep_k, dropping pairs below min_cscore (but always at least min_k).
    rank = pairs.groupby("s1_idx").cumcount().values
    keep = (rank < cc["keep_k"]) & ((pairs["cscore"].values >= cc.get("min_cscore", 0.0))
                                    | (rank < cc.get("min_k", cc["keep_k"])))
    return pairs[keep].reset_index(drop=True)
