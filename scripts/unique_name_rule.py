"""Empty-address records whose compact name matches exactly ONE S1 (in the run's view):
are they that S1's match, and does the current submission already predict it?"""
import json
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
s1, s23 = pl.load_prepared("train")
view = pl.s1_subset("train", s1)
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
gt = gt[np.isin(gt.s1_idx.values, view)]
owner = np.full(len(s23), -1); owner[gt.r_idx.values] = gt.s1_idx.values
v = s1.iloc[view]
cnt = v.groupby(["country", "name_compact"]).size()
uniq = cnt[cnt == 1].index
first = pd.Series(view, index=pd.MultiIndex.from_arrays([v.country.values, v.name_compact.values]))
first = first[~first.index.duplicated()]
r = s23[s23.addr_empty & (s23.name_compact.str.len() >= 4)].copy()
idx = pd.MultiIndex.from_arrays([r.country.values, r.name_compact.values])
r = r[idx.isin(uniq)]
r["cand"] = first.reindex(pd.MultiIndex.from_arrays([r.country.values, r.name_compact.values])).values
r["owner"] = owner[r.index.values]
print(f"empty-address records with exactly one same-name S1: {len(r):,}; that S1 is the owner: {(r.owner == r.cand).mean():.4f}; "
      f"owned by another S1: {((r.owner >= 0) & (r.owner != r.cand)).mean():.4f}; unmatched: {(r.owner < 0).mean():.4f}")
d = pd.read_parquet(f"{bdir}/oof.parquet")[["s1_idx", "r_idx", "source", "p"]]
sel = json.load(open(f"{bdir}/metrics.json"))["selection"]
ch = postprocess.select(postprocess.assign(d, cfg), sel)
n_r = len(s23)
pred = set((ch.s1_idx.values.astype(np.int64) * n_r + ch.r_idx.values).tolist())
k = r.cand.values.astype(np.int64) * n_r + r.index.values
r["predicted"] = [x in pred for x in k]
in_cand = set((d.s1_idx.values.astype(np.int64) * n_r + d.r_idx.values).tolist())
r["candidate"] = [x in in_cand for x in k]
t = r.owner == r.cand
print(f"  true ones: blocked {r.candidate[t].mean():.4f}, predicted {r.predicted[t].mean():.4f}  ({(t & ~r.predicted).sum():,} missed)")
print(f"  wrong ones (not the owner): predicted anyway {r.predicted[~t].mean():.4f}")
print(f"  rule 'always match' would add {(t & ~r.predicted).sum():,} TP and {(~t & ~r.predicted).sum():,} FP vs now")
