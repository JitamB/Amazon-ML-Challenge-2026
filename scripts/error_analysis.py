"""Where does the score go? Decomposes out-of-fold loss for a finished run.
    python scripts/error_analysis.py <run_name>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.metrics import per_entity_f05, summarize  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run = sys.argv[1]
cfg = load_config(f"work/runs/{run}/config.yaml")
pipe = Pipeline(cfg)
s1, s23 = pipe.load_prepared("train")
gt_all = pd.read_parquet(pipe.cache("train_gt.parquet"))
oof = pd.read_parquet(pipe.run_file("oof.parquet"))
params = json.load(open(pipe.run_file("selection.json")))
country, gt, (d, assigned) = pipe.train_eval_view(s1, gt_all, oof, postprocess.assign(oof, cfg))
sub = pipe.s1_subset("train", s1)  # position in view -> row in s1
n = len(country)
mix = cfg["eval"]["test_country_mix"]
pred = postprocess.select(assigned, params)

def score(p):
    return summarize(per_entity_f05(p, gt, n), country, mix)

key = lambda f: f["s1_idx"].values.astype(np.int64) * len(s23) + f["r_idx"].values
blocked_true = gt[np.isin(key(gt), key(d))]
tp = pred[np.isin(key(pred), key(gt))]
print("=== score ceilings (test-mix weighted) ===")
rows = {
    "current": score(pred),
    "no false positives (pred ∩ truth)": score(tp),
    "+ all blocked true pairs (perfect model)": score(blocked_true),
    "perfect blocking + perfect model": score(gt),
}
for k, v in rows.items():
    print(f"  {k:45s} {v['f05_test_mix']:.4f}   US {v.get('f05_US', 0):.4f}  India {v.get('f05_India', 0):.4f}")

# per-entity loss attribution
f = per_entity_f05(pred, gt, n)
G = np.bincount(gt["s1_idx"], minlength=n)
P = np.bincount(pred["s1_idx"], minlength=n)
TP = np.bincount(tp["s1_idx"], minlength=n)
B = np.bincount(blocked_true["s1_idx"], minlength=n)
cat = np.select(
    [(G == 0) & (P > 0), (G == 0), (P - TP > 0) & (TP < G), P - TP > 0, B < G, TP < G],
    ["singleton_false_match", "singleton_ok", "fp_and_fn", "fp_only", "fn_blocking_miss", "fn_model_miss"],
    "perfect")
loss = pd.DataFrame({"cat": cat, "loss": 1 - f, "country": country})
tot = loss["loss"].sum()
print(f"\n=== loss attribution (share of total loss; total loss = {tot / n:.4f} per S1) ===")
print((loss.groupby("cat")["loss"].sum() / tot).sort_values(ascending=False).round(3).to_string())
print("\nS1 counts per category:\n", loss["cat"].value_counts().to_string())
print("\nsingletons:", int((G == 0).sum()), " predicted empty correctly:", int(((G == 0) & (P == 0)).sum()))

# false positives: what are they?
fp = pred[~np.isin(key(pred), key(gt))].merge(d[["s1_idx", "r_idx"]], on=["s1_idx", "r_idx"])
r_true_owner = pd.Series(gt_all["s1_idx"].values, index=gt_all["r_idx"].values)
fp_r_owned = fp["r_idx"].isin(r_true_owner.index)
print(f"\n=== false positives: {len(fp):,} ===")
print(f"  record truly belongs to another S1: {fp_r_owned.mean():.3f}; unmatched decoy record: {1 - fp_r_owned.mean():.3f}")
fn = blocked_true[~np.isin(key(blocked_true), key(pred))].merge(d, on=["s1_idx", "r_idx"], how="left")
print(f"=== model false negatives (blocked but not predicted): {len(fn):,};  p quantiles "
      f"{fn['p'].quantile([.1, .5, .9]).round(3).to_dict()}")
lost_to_bestS1 = fn["p"] >= params.get("threshold", 0.5)
print(f"  of which p >= threshold (lost to best-S1 or caps): {lost_to_bestS1.mean():.3f}")

def show(frame, title, k=15):
    print(f"\n--- {title} ---")
    ex = frame.sample(min(k, len(frame)), random_state=0)
    for s, r, p in zip(ex["s1_idx"], ex["r_idx"], ex["p"]):
        row = sub[s]
        print(f"  p={p:.2f} S1: {s1.at[row, 'name_raw']!r} | {s1.at[row, 'addr']!r}\n"
              f"         R : {s23.at[r, 'name_raw']!r} | {s23.at[r, 'addr']!r}")

show(fp, "false positives")
show(fn, "false negatives (blocked, model said no)")

traits = s23.iloc[fn["r_idx"]]
print("\nFN record traits:", {c: round(float(traits[c].mean()), 3) for c in ["name_indic", "name_web", "addr_empty"]})
traits = s23.iloc[fp["r_idx"]]
print("FP record traits:", {c: round(float(traits[c].mean()), 3) for c in ["name_indic", "name_web", "addr_empty"]})

# leak check: does file order or numeric id relate S1 rows to their matches?
g = gt_all.sample(200000, random_state=0)
s1_num = s1["rid"].str[3:].astype(np.int64).values
r_num = s23["rid"].str[3:].astype(np.int64).values
print("\n=== leak check on 200k true links (Spearman) ===")
print("  row position S1 vs record:", round(pd.Series(g["s1_idx"].values).corr(pd.Series(g["r_idx"].values), method="spearman"), 4))
print("  numeric id   S1 vs record:", round(pd.Series(s1_num[g["s1_idx"]]).corr(pd.Series(r_num[g["r_idx"]]), method="spearman"), 4))
