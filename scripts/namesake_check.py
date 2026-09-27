"""Empty-address true links: can the S1 be picked among namesakes (same compact name)?
Checks whether the raw name (legal words and punctuation kept) singles out the true S1."""
import re

import numpy as np
import pandas as pd

c = "work/cache/dataset_translit_addr_guard_fr_acr"
s1 = pd.read_parquet(f"{c}/train_s1.parquet", columns=["country", "name_compact", "name_raw", "addr"])
s23 = pd.read_parquet(f"{c}/train_s23.parquet", columns=["name_compact", "name_raw", "addr_empty", "hn"])
gt = pd.read_parquet(f"{c}/train_gt.parquet")
norm = lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", x)).strip()
s1["rawn"] = [norm(x) for x in s1.name_raw]
s23["rawn"] = [norm(x) for x in s23.name_raw]
print("S1 with an empty address:", (s1.addr == "").mean().round(4))
g = gt.assign(nc1=s1.name_compact.values[gt.s1_idx], nc2=s23.name_compact.values[gt.r_idx],
              raw1=s1.rawn.values[gt.s1_idx], raw2=s23.rawn.values[gt.r_idx],
              r_empty=s23.addr_empty.values[gt.r_idx], cty=s1.country.values[gt.s1_idx])
e = g[g.r_empty & (g.nc1 == g.nc2) & (g.nc1.str.len() >= 4)]
grp = s1.groupby(["country", "name_compact"]).size().rename("namesakes")
e = e.join(grp, on=["cty", "nc1"])
print(f"empty-address true links with identical compact name: {len(e):,}; namesakes median {e.namesakes.median()}, share with >=2: {(e.namesakes >= 2).mean():.3f}")
raw_grp = s1.groupby(["country", "rawn"]).size().rename("raw_namesakes")
e = e.join(raw_grp, on=["cty", "raw2"])
e["raw_namesakes"] = e.raw_namesakes.fillna(0)
e["raw_exact"] = e.raw1 == e.raw2
m = e.namesakes >= 2
print(f"among those with >=2 namesakes: record raw name == true S1 raw name: {e.raw_exact[m].mean():.3f}")
print(f"  ... and that raw name is unique among S1s: {(e.raw_exact & (e.raw_namesakes == 1))[m].mean():.3f}")
print(f"  record's raw name matches NO S1 exactly: {(e.raw_namesakes[m] == 0).mean():.3f}")
print("examples (namesakes >= 2):")
for _, r in e[m].sample(12, random_state=0).iterrows():
    print(f"   S1 {r.raw1!r:45}  R {r.raw2!r:45}  namesakes {int(r.namesakes)}, S1s with R's raw name {int(r.raw_namesakes)}")
