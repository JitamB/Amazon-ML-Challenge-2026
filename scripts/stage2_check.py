"""Does stage 2 misbehave on test because its stage-1 input is a fold *average* on test but a
single out-of-fold model's score on train? Re-scores test with stage-1 p from one fold model
(like train) and compares decoy-type pairs and match counts with the production path.
    python scripts/stage2_check.py <run>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import model, postprocess, stage2  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402
import lightgbm as lgb  # noqa: E402

run = sys.argv[1]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
s1, s23 = pl.load_prepared("test")
feats = pd.concat([pd.read_parquet(p) for p in pl.feature_parts("test")], ignore_index=True)
k = cfg["model"]["n_folds"]
b1 = [lgb.Booster(model_file=pl.run_file(f"models/fold{i}.txt")) for i in range(k)]
p_each = np.stack([b.predict(feats[b.feature_name()]) for b in b1])
a, b = s1["hn"].values[feats.s1_idx.values], s23["hn"].values[feats.r_idx.values]
shift = np.array([bool(x) and bool(y) and x != y and x not in y and y not in x and len(x) < 15 and len(y) < 15
                  and abs(int(x) - int(y)) <= 20 for x, y in zip(a, b)]) & (feats["n_tset"].values >= 85)
us = s1["country"].values[feats.s1_idx.values] == "US"
keep = ["s1_idx", "r_idx", "source"] + [c for c in stage2.GROUP_SOURCES if c in feats.columns and c != "p"]
sel = pd.read_json(pl.run_file("selection.json"), typ="series").to_dict()
print("stage-1 p spread across fold models (US decoy-type): mean std", p_each[:, us & shift].std(0).mean().round(4))
variants = {"average of folds (production)": p_each.mean(0)}
rng = np.random.default_rng(0)
fold_of = pd.util.hash_array(s1["rid"].values.astype(object))[feats.s1_idx.values] % k
variants["one fold model per S1 (like OOF)"] = p_each[fold_of, np.arange(len(feats))]
for name, p1 in variants.items():
    d = feats[keep].assign(p=p1.astype(np.float32))
    g = stage2.group_features(d, s23)
    X = pd.concat([feats, g], axis=1)
    p2 = model.predict(X, pl.run_file("models_s2"), k)
    chosen = postprocess.select(postprocess.assign(feats[["s1_idx", "r_idx", "source"]].assign(p=p2), cfg), sel)
    cc = s1["country"].values[chosen.s1_idx.values]
    print(f"\n{name}:\n  US decoy-type pairs: stage-1 p>=0.5 {(p1[us & shift] >= 0.5).mean():.4f}, final p>=0.5 {(p2[us & shift] >= 0.5).mean():.4f}")
    print("  matches per S1:", {c: round((cc == c).sum() / (s1['country'] == c).sum(), 4) for c in ["US", "India", "France"]})
