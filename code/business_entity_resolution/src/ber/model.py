"""LightGBM pair classifier with S1-grouped cross-fitting.

Folds are assigned by S1 (all candidates of an S1 share a fold), so every
train pair gets an out-of-fold probability. The OOF predictions tune
post-processing under full S1-vs-S1 competition; the fold models are averaged
for test.

`model.seeds` (e.g. [0, 1, 2]) trains one model per seed in every fold (different bagging and
feature subsampling) and averages them, out-of-fold and on test. Without it, one model per fold.
"""
import glob
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import NON_FEATURES


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def train_cv(feats, cfg, model_dir, log=print):
    mc = cfg["model"]
    cols = feature_columns(feats)
    os.makedirs(model_dir, exist_ok=True)
    oof = np.zeros(len(feats), dtype=np.float32)
    importance = pd.Series(0.0, index=cols)
    rng = np.random.default_rng(0)
    seeds = mc.get("seeds") or [None]
    for k in range(mc["n_folds"]):
        is_val = feats["fold"].values == k
        tr_s1 = np.unique(feats["s1_idx"].values[~is_val])
        if len(tr_s1) > mc["max_train_s1_per_fold"]:
            tr_s1 = rng.choice(tr_s1, mc["max_train_s1_per_fold"], replace=False)
        tr = ~is_val & np.isin(feats["s1_idx"].values, tr_s1)
        dtr = lgb.Dataset(feats.loc[tr, cols], feats.loc[tr, "label"])
        dva = lgb.Dataset(feats.loc[is_val, cols], feats.loc[is_val, "label"], reference=dtr)
        for j, seed in enumerate(seeds):
            # training-side seeds only: the binned Dataset is shared across seeds
            params = mc["params"] if seed is None else {**mc["params"], "bagging_seed": seed, "feature_fraction_seed": seed}
            booster = lgb.train(params, dtr, num_boost_round=mc["num_boost_round"], valid_sets=[dva],
                                callbacks=[lgb.early_stopping(mc["early_stopping_rounds"], verbose=False)])
            oof[is_val] += booster.predict(feats.loc[is_val, cols], num_iteration=booster.best_iteration) / len(seeds)
            name = f"fold{k}.txt" if seed is None else f"fold{k}_s{j}.txt"
            booster.save_model(os.path.join(model_dir, name), num_iteration=booster.best_iteration)
            importance += pd.Series(booster.feature_importance("gain"), index=cols)
            log(f"  fold {k}{'' if seed is None else f' seed {seed}'}: train rows {tr.sum():,} "
                f"valid rows {is_val.sum():,} best_iter {booster.best_iteration}")
    return oof, importance.sort_values(ascending=False)


def predict(feats, model_dir, n_folds):
    """Average of every saved model: one per fold, or one per fold and seed."""
    files = [f for k in range(n_folds) for f in sorted(glob.glob(os.path.join(model_dir, f"fold{k}*.txt")))]
    preds = np.zeros(len(feats), dtype=np.float32)
    for f in files:
        booster = lgb.Booster(model_file=f)
        preds += booster.predict(feats[booster.feature_name()]).astype(np.float32) / len(files)
    return preds
