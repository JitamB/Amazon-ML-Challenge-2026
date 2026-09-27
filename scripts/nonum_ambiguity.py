"""For records with no address number (97% are true matches of some S1): how ambiguous are they?"""
import numpy as np
import pandas as pd

c = "work/cache/dataset_translit_addr_guard_fr_acr"
s23 = pd.read_parquet(f"{c}/train_s23.parquet", columns=["hn"])
gt = pd.read_parquet(f"{c}/train_gt.parquet")
oof = pd.read_parquet("work/runs/blend_v11_caps+v11b_s2robust/oof.parquet")[["s1_idx", "r_idx", "p"]]
n_r = len(s23)
oof["y"] = np.isin(oof.s1_idx.values * n_r + oof.r_idx.values, gt.s1_idx.values * n_r + gt.r_idx.values)
d = oof[(s23.hn.values == "")[oof.r_idx.values]]
g = d.groupby("r_idx")
rec = pd.DataFrame({"has_true": g.y.any(), "n_cand": g.size(), "n_p10": g.p.apply(lambda x: (x >= 0.1).sum()),
                    "max_p": g.p.max(), "sum_p": g.p.sum()})
top = d.sort_values("p", ascending=False).drop_duplicates("r_idx").set_index("r_idx")
rec["top_is_true"] = top.y.reindex(rec.index)
print("no-number records in candidates:", len(rec), " with the true S1 among candidates:", rec.has_true.mean().round(4))
x = rec[rec.has_true]
print("given the true S1 is a candidate: top-scored S1 is the true one:", x.top_is_true.mean().round(4))
print("max p quantiles:", x.max_p.quantile([.1, .25, .5]).round(3).to_dict())
print("candidates with p>=0.1:", x.n_p10.value_counts(normalize=True).sort_index().head(6).round(3).to_dict())
low = x[x.max_p < 0.5]
print(f"true-candidate records whose best p < 0.5: {len(low):,} ({len(low) / len(x):.3f}); top is true there: {low.top_is_true.mean():.3f}; "
      f"n_p10 dist {low.n_p10.value_counts(normalize=True).sort_index().head(5).round(3).to_dict()}")
