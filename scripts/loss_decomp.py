"""Where the validation F0.5 is lost after blocking, in F0.5 points (not links).

    python scripts/loss_decomp.py <run> <blend_dir>

Per S1: loss = 1 - F. Split into the part caused by false positives (F without FPs - F) and
the part caused by missed links (1 - F without FPs); missed links are then attributed to
their reason (blocking / lost to another S1 by best-S1 / cap / below threshold) by adding
them back in that order.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.metrics import f05_from_counts  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, bdir = sys.argv[1], sys.argv[2]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["rid", "country"])
gt_full = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{bdir}/oof.parquet")
params = json.load(open(f"{bdir}/metrics.json"))["selection"]
country, gt, (oofv,) = pl.train_eval_view(s1, gt_full, oof)
n = len(country)
n_r = int(max(oof.r_idx.max(), gt_full.r_idx.max())) + 1
key = lambda s, r: s.astype(np.int64) * n_r + r
print(f"view S1 {n:,}, true links {len(gt):,}, selection {params}")

assigned = postprocess.assign(oofv, cfg)
pred = postprocess.select(assigned, params)
tk = key(gt.s1_idx.values, gt.r_idx.values)
pk = key(pred.s1_idx.values, pred.r_idx.values)
pred = pred.assign(y=np.isin(pk, tk))
G = np.bincount(gt.s1_idx, minlength=n)
P = np.bincount(pred.s1_idx, minlength=n)
TP = np.bincount(pred.s1_idx[pred.y], minlength=n)
F = f05_from_counts(P, TP, G)
F_nofp = f05_from_counts(TP, TP, G)
print(f"macro F0.5 {F.mean():.5f}  total loss {1 - F.mean():.5f}")

# --- false positives: whose record is it?
fp = pred[~pred.y]
r_true = pd.Series(gt_full.s1_idx.values, index=gt_full.r_idx.values)
r_true = r_true[~r_true.index.duplicated()]
owner = r_true.reindex(fp.r_idx.values).values
in_view = np.zeros(len(s1), bool)
in_view[pl.s1_subset("train", s1)] = True
fp_kind = np.where(np.isnan(owner), "orphan (no S1)",
                   np.where(in_view[np.nan_to_num(owner, nan=0).astype(int)], "another S1 in view",
                            "S1 outside view (sampling artefact)"))
loss_fp = F_nofp - F
print(f"\nLOSS FROM FALSE POSITIVES: {loss_fp.mean():.5f}  ({len(fp):,} FP pairs)")
# attribute an S1's FP loss to its FP kinds in proportion to counts
fpk = pd.DataFrame({"s1": fp.s1_idx.values, "kind": fp_kind})
cnt = fpk.groupby(["s1", "kind"]).size().reset_index(name="c")
cnt["w"] = cnt.c / cnt.groupby("s1").c.transform("sum") * loss_fp[cnt.s1.values]
att = cnt.groupby("kind").w.sum() / n
print(att.round(5).to_string())
sing = (G == 0)
print(f"  singleton S1s (G=0): {sing.mean():.4f} of S1; predicted non-empty: {(sing & (P > 0)).sum():,}"
      f" -> loss {(sing & (P > 0)).sum() / n:.5f}")
print(f"  non-singleton S1s with >=1 FP: {((~sing) & (P > TP)).sum():,}")

# --- missed links: reason
loss_fn = 1 - F_nofp
print(f"\nLOSS FROM MISSED LINKS: {loss_fn.mean():.5f}")
ok = key(oofv.s1_idx.values, oofv.r_idx.values)
g = gt.assign(k=tk, in_cand=np.isin(tk, ok), sel=np.isin(tk, pk))
ak = key(assigned.s1_idx.values, assigned.r_idx.values)
g["in_assigned"] = np.isin(tk, ak)
best = oofv.loc[oofv.groupby("r_idx")["p"].idxmax()].set_index("r_idx")
g["best_s1"] = best["s1_idx"].reindex(g.r_idx.values).values
pm = pd.Series(oofv.p.values, index=ok)
g["p"] = pm.reindex(g.k.values).values
reason = np.select(
    [~g.in_cand, g.sel, g.best_s1.values != g.s1_idx.values, ~g.in_assigned],
    ["1 blocking", "ok", "2 lost to another S1 (best-S1)", "3 cap"], "4 below threshold")
g["reason"] = reason
print(pd.Series(reason).value_counts().sort_index().to_string())
# sequential add-back in F points
tp_now = TP.copy()
fpc = P - TP
prev = f05_from_counts(P, tp_now, G).mean()
for r in ["4 below threshold", "3 cap", "2 lost to another S1 (best-S1)", "1 blocking"]:
    add = np.bincount(g.s1_idx[g.reason == r], minlength=n)
    tp_now = tp_now + add
    cur = f05_from_counts(tp_now + fpc, tp_now, G).mean()
    print(f"  add back '{r}': +{cur - prev:.5f}")
    prev = cur
print(f"  (then removing all FP: +{f05_from_counts(tp_now, tp_now, G).mean() - prev:.5f} -> 1.0)")

bt = g[g.reason == "4 below threshold"]
print("\nbelow-threshold misses by p:", pd.cut(bt.p, [0, .05, .2, .35, .5, .65]).value_counts().sort_index().to_dict())
st = g[g.reason == "2 lost to another S1 (best-S1)"]
print(f"lost-to-another-S1 misses: {len(st):,}; own p quantiles {st.p.quantile([.25, .5, .75]).round(3).to_dict()}")

# --- per S1 outcome classes
cls = np.select([(G == 0) & (P == 0), (G == 0), (P == 0), (TP == G) & (P == TP), P > TP],
                ["singleton, empty (ok)", "singleton given matches", "has links, predicted EMPTY",
                 "perfect", "has FP"], "partial (FN only)")
d = pd.DataFrame({"cls": cls, "loss": 1 - F, "G": G})
print("\nS1 outcome classes (share of S1, loss in F points):")
print(d.groupby("cls").agg(share=("loss", lambda x: len(x) / n), loss=("loss", lambda x: x.sum() / n)).round(5).to_string())
print("\nloss by true group size G:")
print(d.assign(Gc=np.minimum(G, 8)).groupby("Gc").agg(share=("loss", lambda x: len(x) / n),
                                                     loss=("loss", lambda x: x.sum() / n)).round(5).to_string())

# empty-predicted S1s that have links: were their records stolen?
emp = np.flatnonzero((G > 0) & (P == 0))
ge = g[np.isin(g.s1_idx, emp)]
print(f"\nS1 with links but predicted empty: {len(emp):,}; their links by reason:",
      ge.reason.value_counts().to_dict())

# structure of the truth: S2 / S3 counts per S1
src = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["source"])["source"].values
gs = gt_full.assign(src=src[gt_full.r_idx.values])
c2 = np.bincount(gs.s1_idx[gs.src == 2], minlength=len(s1))
c3 = np.bincount(gs.s1_idx[gs.src == 3], minlength=len(s1))
print("\n(true S2 count, true S3 count) per S1, share:")
print(pd.crosstab(np.minimum(c2, 6), np.minimum(c3, 7), normalize=True).round(4).to_string())
