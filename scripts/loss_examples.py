"""Examples and simple stats for the main post-blocking loss classes of a blend.

    python scripts/loss_examples.py <run> <blend_dir>

Classes: orphan false positives, true links below the threshold, true links lost to
another S1 by best-S1. For each: house-number relation to the S1 and name equality.
"""
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
cols = ["rid", "country", "name", "name_compact", "addr", "hn", "nums", "addr_empty"]
s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=cols)
s23 = pd.read_parquet(pl.cache("train_s23.parquet"), columns=cols + ["source"])
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{bdir}/oof.parquet")
params = json.load(open(f"{bdir}/metrics.json"))["selection"]
view = pl.s1_subset("train", s1)
oof = oof[np.isin(oof.s1_idx.values, view)]
gtv = gt[np.isin(gt.s1_idx.values, view)]
n_r = len(s23)
key = lambda d: d.s1_idx.values.astype(np.int64) * n_r + d.r_idx.values
tk = key(gtv)
pred = postprocess.select(postprocess.assign(oof, cfg), params)
pred = pred.assign(y=np.isin(key(pred), tk))
owner = np.full(n_r, -1)
owner[gt.r_idx.values] = gt.s1_idx.values
oof = oof.assign(y=np.isin(key(oof), tk))


def rel(d):
    """House-number and name relation of pairs d (s1_idx, r_idx) to their S1."""
    a, b = s1.iloc[d.s1_idx.values], s23.iloc[d.r_idx.values]
    ha, hb = a.hn.values, b.hn.values
    diff = np.full(len(d), np.nan)
    ok = np.array([x.isdigit() and y.isdigit() and len(x) < 9 and len(y) < 9 for x, y in zip(ha, hb)])
    diff[ok] = np.abs(ha[ok].astype(np.int64) - hb[ok].astype(np.int64))
    hn = np.select([b.addr_empty.values, hb == "", ha == hb, (diff >= 1) & (diff <= 20)],
                   ["r empty addr", "r no number", "same hn", "hn shift 1-20"], "other hn")
    nm = np.where(a.name_compact.values == b.name_compact.values, "same name", "diff name")
    return pd.Series(hn), pd.Series(nm)


def show(title, d, k=8, extra=None):
    print(f"\n===== {title}: {len(d):,}")
    hn, nm = rel(d)
    print("  hn relation:", hn.value_counts(normalize=True).round(3).to_dict())
    print("  name:", nm.value_counts(normalize=True).round(3).to_dict())
    print("  country:", pd.Series(s1.country.values[d.s1_idx.values]).value_counts(normalize=True).round(3).to_dict())
    ex = d.sample(min(k, len(d)), random_state=1)
    for _, r in ex.iterrows():
        a, b = s1.iloc[int(r.s1_idx)], s23.iloc[int(r.r_idx)]
        print(f"  [{a.country}] p={r.p:.3f}  S1 '{a['name']}' | '{a.addr}'")
        print(f"      R(s{b.source}) '{b['name']}' | '{b.addr}'")
        if extra is not None:
            extra(r)


def true_sibs(r):
    t = gt[gt.s1_idx == r.s1_idx]
    for i in t.r_idx.values[:3]:
        b = s23.iloc[i]
        print(f"      true sibling: '{b['name']}' | '{b.addr}'")


def winner(r):
    o = oof[(oof.r_idx == r.r_idx)].sort_values("p", ascending=False).head(2)
    for _, w in o.iterrows():
        a = s1.iloc[int(w.s1_idx)]
        print(f"      cand S1 p={w.p:.3f} {'TRUE' if w.y else 'wrong'}: '{a['name']}' | '{a.addr}'")


fp = pred[~pred.y]
show("ORPHAN false positives", fp[owner[fp.r_idx.values] == -1], extra=true_sibs)
show("true links selected (reference)", pred[pred.y].sample(200_000, random_state=0), k=0)
cand_true = oof[oof.y]
sel = np.isin(key(cand_true), key(pred))
best = oof.loc[oof.groupby("r_idx")["p"].idxmax()].set_index("r_idx")["s1_idx"]
lost = best.reindex(cand_true.r_idx.values).values != cand_true.s1_idx.values
show("true links BELOW threshold (not lost to another S1)", cand_true[~sel & ~lost], extra=winner)
show("true links LOST to another S1", cand_true[~sel & lost], extra=winner)
orph = owner == -1
print(f"\norphan records: {orph.mean():.4f} of S2/S3; of candidate pairs: {orph[oof.r_idx.values].mean():.4f}")
