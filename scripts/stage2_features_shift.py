"""Which stage-2 (group) features differ for US decoy-type pairs, train decoys vs test?
    python scripts/stage2_features_shift.py <run>
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import stage2  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run = sys.argv[1]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
k = cfg["model"]["n_folds"]
b1 = [lgb.Booster(model_file=pl.run_file(f"models/fold{i}.txt")) for i in range(k)]
res = {}
for split in ("train", "test"):
    s1, s23 = pl.load_prepared(split)
    feats = pd.concat([pd.read_parquet(p) for p in pl.feature_parts(split)], ignore_index=True)
    if split == "train":
        p1 = np.zeros(len(feats))
        for i in range(k):
            m = feats["fold"].values == i
            p1[m] = b1[i].predict(feats.loc[m, b1[i].feature_name()])
    else:
        p1 = np.mean([b.predict(feats[b.feature_name()]) for b in b1], axis=0)
    keep = ["s1_idx", "r_idx", "source"] + [c for c in stage2.GROUP_SOURCES if c in feats.columns and c != "p"]
    g = stage2.group_features(feats[keep].assign(p=p1.astype(np.float32)), s23)
    a, b = s1["hn"].values[feats.s1_idx.values], s23["hn"].values[feats.r_idx.values]
    shift = np.array([bool(x) and bool(y) and x != y and x not in y and y not in x and len(x) < 15 and len(y) < 15
                      and abs(int(x) - int(y)) <= 20 for x, y in zip(a, b)]) & (feats["n_tset"].values >= 85)
    m = shift & (s1["country"].values[feats.s1_idx.values] == "US")
    if split == "train":
        m &= feats["label"].values == 0
    res[split] = g[m]
    print(split, int(m.sum()))
imp = pd.read_csv(pl.run_file("feature_importance_s2.csv"), index_col=0)["gain"]
rows = []
for c in res["train"].columns:
    tr, te = res["train"][c].values, res["test"][c].values
    rows.append((c, imp.get(c, 0) / imp.sum(), np.mean(tr), np.mean(te), np.quantile(tr, .9), np.quantile(te, .9)))
t = pd.DataFrame(rows, columns=["feature", "importance", "train_mean", "test_mean", "train_p90", "test_p90"])
t["shift"] = (t.test_mean - t.train_mean).abs() / (res["train"].std().values + 1e-6)
print(t.sort_values("shift", ascending=False).round(4).head(20).to_string(index=False))
