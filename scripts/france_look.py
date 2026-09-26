"""Uncertain French test S1s under a run: S1 text next to its top candidates (no labels exist for France)."""
import sys

import numpy as np
import pandas as pd

run = sys.argv[1]
c = "work/cache/dataset_translit"
s1 = pd.read_parquet(f"{c}/test_s1.parquet", columns=["country", "name_raw", "addr"])
s23 = pd.read_parquet(f"{c}/test_s23.parquet", columns=["name_raw", "addr", "source"])
p = pd.read_parquet(f"work/runs/{run}/test_preds.parquet")
print(p.columns.tolist())
p["country"] = s1["country"].values[p["s1_idx"].values]
top = p.groupby("s1_idx")["p"].max()
cty = s1["country"].values[top.index]
for cc in ["France", "US", "India"]:
    t = top[cty == cc]
    print(cc, "max-p quantiles", t.quantile([.05, .1, .25, .5]).round(3).to_dict(),
          "share of pairs with 0.2<p<0.8:", round(((p.country == cc) & p.p.between(.2, .8)).sum() / (p.country == cc).sum(), 4))
rng = np.random.default_rng(1)
unc = top[(cty == "France") & (top.values > 0.15) & (top.values < 0.6)].index.values
for s in rng.choice(unc, 12, replace=False):
    print(f"\nS1: {s1.at[s, 'name_raw']!r} | {s1.at[s, 'addr']!r}")
    for _, r in p[p.s1_idx == s].nlargest(4, "p").iterrows():
        print(f"   p={r.p:.2f} src{s23.at[r.r_idx, 'source']}: {s23.at[r.r_idx, 'name_raw']!r} | {s23.at[r.r_idx, 'addr']!r}")
