"""Which simple signals separate orphan false positives from true positives?

    python scripts/orphan_signals.py <run> <blend_dir>

For every selected pair: rates of candidate signals among true positives, orphan FPs and
wrong-S1 FPs. Also: do orphan records come in clusters of their own (other orphans with the
same compact name in the same country)?
"""
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, bdir = sys.argv[1], sys.argv[2]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
cols = ["country", "name", "name_compact", "addr", "hn", "nums", "addr_empty"]
s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=cols + ["rid"])
s23 = pd.read_parquet(pl.cache("train_s23.parquet"), columns=cols + ["source"])
raw = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["name_raw"])["name_raw"].values
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{bdir}/oof.parquet")
params = json.load(open(f"{bdir}/metrics.json"))["selection"]
view = pl.s1_subset("train", s1)
oof = oof[np.isin(oof.s1_idx.values, view)].reset_index(drop=True)
n_r = len(s23)
key = lambda d: d.s1_idx.values.astype(np.int64) * n_r + d.r_idx.values
owner = np.full(n_r, -1)
owner[gt.r_idx.values] = gt.s1_idx.values
pred = postprocess.select(postprocess.assign(oof, cfg), params).reset_index(drop=True)
y = owner[pred.r_idx.values] == pred.s1_idx.values
kind = np.where(y, "TP", np.where(owner[pred.r_idx.values] == -1, "FP orphan", "FP other S1"))

a, b = s1.iloc[pred.s1_idx.values], s23.iloc[pred.r_idx.values]
a_nums = [set(x.split()) for x in a.nums.values]
b_addr = b.addr.values
first = np.array([x.split()[0] if x else "" for x in b_addr])
sig = {}
sig["r addr starts with a bare number not in S1 numbers"] = np.array(
    [f.isdigit() and f not in s for f, s in zip(first, a_nums)])
sig["r has a zero-padded number"] = np.array([bool(re.search(r"(^|\s)0\d", x)) for x in b_addr])
sig["r number set has a number not in S1"] = np.array(
    [bool(set(x.split()) - s) for x, s in zip(b.nums.values, a_nums)])
sig["S1 hn missing from r numbers"] = np.array(
    [h != "" and x != "" and h not in x.split() for h, x in zip(a.hn.values, b.nums.values)])
sig["r empty address"] = b.addr_empty.values.astype(bool)
sig["same compact name"] = a.name_compact.values == b.name_compact.values

# corroboration: other candidates of the same S1 (p>0.3) sharing r's hn when it differs from S1's
c = oof[oof.p > 0.3]
c_hn = s23.hn.values[c.r_idx.values]
cnt = c.assign(hn=c_hn)[c_hn != ""].groupby(["s1_idx", "hn"]).size()
mi = pd.MultiIndex.from_arrays([pred.s1_idx.values, b.hn.values])
same_hn_n = cnt.reindex(mi).fillna(0).values - 1
sig["r hn != S1 hn and >=1 other candidate shares r hn"] = (b.hn.values != a.hn.values) & (b.hn.values != "") & (same_hn_n >= 1)

# exact twin: another candidate of the same S1 with identical name+addr
c_all = oof.assign(t=s23.name.values[oof.r_idx.values] + "|" + s23.addr.values[oof.r_idx.values])
tw = c_all.groupby(["s1_idx", "t"]).size()
sig["exact twin among the S1's candidates"] = tw.reindex(pd.MultiIndex.from_arrays(
    [pred.s1_idx.values, b.name.values + "|" + b_addr])).values > 1
# exact twin in the same source
c_all["src"] = s23.source.values[oof.r_idx.values]
tws = c_all.groupby(["s1_idx", "src", "t"]).size()
sig["exact twin, same source"] = tws.reindex(pd.MultiIndex.from_arrays(
    [pred.s1_idx.values, b.source.values, b.name.values + "|" + b_addr])).values > 1

df = pd.DataFrame(sig).assign(kind=kind)
print(pd.Series(kind).value_counts().to_string())
print("\nsignal rate by kind:")
print(df.groupby("kind").mean().T.round(4).to_string())

# orphan clusters: other orphans with the same compact name in the same country
orph = owner == -1
k = pd.Series(s23.country.values + "|" + s23.name_compact.values)
n_orph_same = k[orph].value_counts()
n_all_same = k.value_counts()
kk = k.values[pred.r_idx.values]
df["orphans_same_name"] = n_orph_same.reindex(kk).fillna(0).values
df["records_same_name"] = n_all_same.reindex(kk).fillna(0).values
print("\nmedian #orphans / #records with the same compact name:")
print(df.groupby("kind")[["orphans_same_name", "records_same_name"]].median().to_string())
print("\nS2/S3 records per S1 by source; orphan share by source:",
      s23.assign(o=orph).groupby("source").o.mean().round(4).to_dict())
