"""For US strong-name pairs whose house number is shifted 1-20 (decoy-type): compare stage-1
features and stage-1 p, train vs test. Shows whether the higher test scores come from the
pair's own evidence or from density-dependent context features.
    python scripts/feature_shift.py <run>
"""
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import model  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run = sys.argv[1]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
cols = ["b_n_s1", "b_n_r", "b_rank_s1", "b_rank_r", "b_rel_s1", "b_rel_r", "bscore", "cscore", "n_tset", "a_tset",
        "hn_logdiff", "num_jac", "p1"]
out = {}
for split in ("train", "test"):
    s1, s23 = pl.load_prepared(split)
    parts = []
    for path in pl.feature_parts(split):
        f = pd.read_parquet(path)
        us = s1["country"].values[f.s1_idx.values] == "US"
        f = f[us]
        a, b = s1["hn"].values[f.s1_idx.values], s23["hn"].values[f.r_idx.values]
        sh = np.array([bool(x) and bool(y) and x != y and x not in y and y not in x and len(x) < 15 and len(y) < 15
                       and abs(int(x) - int(y)) <= 20 for x, y in zip(a, b)])
        f = f[sh & (f["n_tset"].values >= 85)]
        f = f.assign(p1=model.predict(f, pl.run_file("models"), cfg["model"]["n_folds"]))
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    fin = pd.read_parquet(pl.run_file("oof.parquet" if split == "train" else "test_preds.parquet"),
                          columns=["s1_idx", "r_idx", "p"])
    f = f.merge(fin.rename(columns={"p": "p_final"}), on=["s1_idx", "r_idx"], how="left")
    print(f"{split}: all decoy-type pairs: stage-1 p>=0.5 {(f.p1 >= 0.5).mean():.4f}, final p>=0.5 {(f.p_final >= 0.5).mean():.4f}")
    if split == "train":
        for lab in (0, 1):
            g = f[f["label"] == lab]
            print(f"   train label {lab}: n {len(g):,}, stage-1 p>=0.5 {(g.p1 >= 0.5).mean():.4f}, final p>=0.5 {(g.p_final >= 0.5).mean():.4f}")
        f = f[f["label"] == 0]  # train: decoys only (test decoys can't be separated, ~94% of test pairs here are decoys)
    out[split] = f
    print(split, len(f))
q = [0.25, 0.5, 0.75, 0.9]
for c in cols:
    print(f"{c:>11}  train {np.round(np.quantile(out['train'][c], q), 3)}   test {np.round(np.quantile(out['test'][c], q), 3)}")
print("stage-1 p >= 0.5 share: train decoys", round((out["train"]["p1"] >= 0.5).mean(), 4),
      " test (all)", round((out["test"]["p1"] >= 0.5).mean(), 4))
