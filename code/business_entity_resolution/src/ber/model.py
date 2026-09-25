"""LightGBM pair classifier with S1-grouped cross-fitting.

Folds are assigned by S1 (all candidates of an S1 share a fold), so every
train pair gets an out-of-fold probability. The OOF predictions tune
post-processing under full S1-vs-S1 competition; the fold models are averaged
for test.
"""
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
    for k in range(mc["n_folds"]):
        is_val = feats["fold"].values == k
        tr_s1 = np.unique(feats["s1_idx"].values[~is_val])
        if len(tr_s1) > mc["max_train_s1_per_fold"]:
            tr_s1 = rng.choice(tr_s1, mc["max_train_s1_per_fold"], replace=False)
        tr = ~is_val & np.isin(feats["s1_idx"].values, tr_s1)
        dtr = lgb.Dataset(feats.loc[tr, cols], feats.loc[tr, "label"])
        dva = lgb.Dataset(feats.loc[is_val, cols], feats.loc[is_val, "label"], reference=dtr)
        booster = lgb.train(mc["params"], dtr, num_boost_round=mc["num_boost_round"], valid_sets=[dva],
                            callbacks=[lgb.early_stopping(mc["early_stopping_rounds"], verbose=False)])
        oof[is_val] = booster.predict(feats.loc[is_val, cols], num_iteration=booster.best_iteration)
        booster.save_model(os.path.join(model_dir, f"fold{k}.txt"), num_iteration=booster.best_iteration)
        importance += pd.Series(booster.feature_importance("gain"), index=cols)
        log(f"  fold {k}: train rows {tr.sum():,} valid rows {is_val.sum():,} best_iter {booster.best_iteration}")
    return oof, importance.sort_values(ascending=False)


def predict(feats, model_dir, n_folds):
    preds = np.zeros(len(feats), dtype=np.float32)
    for k in range(n_folds):
        booster = lgb.Booster(model_file=os.path.join(model_dir, f"fold{k}.txt"))
        preds += booster.predict(feats[booster.feature_name()]).astype(np.float32) / n_folds
    return preds
