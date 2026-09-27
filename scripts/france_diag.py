"""How French test predictions differ from India / US test (and train OOF), with examples.

    python scripts/france_diag.py <run> <blend_dir>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, bdir = sys.argv[1], sys.argv[2]
pl = Pipeline(load_config(f"work/runs/{run}/config.yaml"))
cols = ["country", "name", "addr", "hn", "addr_empty"]
s1 = pd.read_parquet(pl.cache("test_s1.parquet"), columns=cols)
s23 = pd.read_parquet(pl.cache("test_s23.parquet"), columns=cols + ["source"])
t = pd.read_parquet(f"{bdir}/test_preds.parquet")
t["country"] = s1.country.values[t.s1_idx.values]
bins = [0, .02, .1, .3, .5, .65, .8, .95, 1.0001]


def profile(d, n_s1):
    top = d.groupby("s1_idx").p.max()
    r2 = d[d.p > 0.3].groupby("r_idx").s1_idx.nunique()
    return {"cands/S1": round(len(d) / n_s1, 2),
            "p>0.65 per S1": round((d.p > .65).sum() / n_s1, 3),
            "mid 0.3-0.8 per S1": round(((d.p > .3) & (d.p < .8)).sum() / n_s1, 3),
            "S1 top p<0.65": round(1 - (top >= .65).sum() / n_s1, 4),
            "S1 top p in .3-.8": round(((top > .3) & (top < .8)).sum() / n_s1, 4),
            "records with 2+ S1 at p>0.3": round((r2 >= 2).mean(), 4)}


rows = {}
for c in ["India", "US", "France"]:
    rows[f"test {c}"] = profile(t[t.country == c], (s1.country == c).sum())
print(pd.DataFrame(rows).T.to_string())
print("\np histogram per country (share of pairs):")
print(pd.crosstab(t.country, pd.cut(t.p, bins), normalize="index").round(4).to_string())
print("\nS2/S3 records per S1:", {c: round((s23.country == c).sum() / (s1.country == c).sum(), 3) for c in ["India", "US", "France"]})
print("empty-address share of S2/S3:", s23.groupby("country").addr_empty.mean().round(4).to_dict())

fr = t[(t.country == "France") & (t.p > .3) & (t.p < .8)].sample(24, random_state=2)
print("\nFrench mid-confidence pairs:")
for _, r in fr.iterrows():
    a, b = s1.iloc[int(r.s1_idx)], s23.iloc[int(r.r_idx)]
    print(f"  p={r.p:.3f}  S1 '{a['name']}' | '{a.addr}'\n          R(s{b.source}) '{b['name']}' | '{b.addr}'")
