"""Stage-by-stage funnel on train (out-of-fold, run's S1 view): how many true links survive each
step, and precision where it applies. Then a breakdown of the links lost at blocking.
    python scripts/funnel.py <run> <blend_dir>
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run, bdir = sys.argv[1], sys.argv[2]
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
s1, s23 = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
view = pl.s1_subset("train", s1)
gt = gt[np.isin(gt.s1_idx.values, view)]
n_r = len(s23)
key = lambda d: d.s1_idx.values.astype(np.int64) * n_r + d.r_idx.values
T = len(gt)
print(f"true links in view: {T:,}  (S1 {len(view):,})")

feats = pd.concat([pd.read_parquet(p, columns=None) for p in pl.feature_parts("train")], ignore_index=True)
y = feats["label"].values.astype(bool)
print(f"1. candidates after blocking + ranker trim: {len(feats) / len(view):.2f}/S1, true links kept {y.sum() / T:.4%}")
k = cfg["model"]["n_folds"]
p1 = np.zeros(len(feats))
for i in range(k):
    b = lgb.Booster(model_file=pl.run_file(f"models/fold{i}.txt"))
    m = feats["fold"].values == i
    p1[m] = b.predict(feats.loc[m, b.feature_name()])
pair = feats[["s1_idx", "r_idx", "source"]].copy()
oof_full = pd.read_parquet(pl.run_file("oof.parquet"))[["s1_idx", "r_idx", "p"]].rename(columns={"p": "p2"})
oof_bl = pd.read_parquet(f"{bdir}/oof.parquet")[["s1_idx", "r_idx", "p"]].rename(columns={"p": "pb"})
pair = pair.assign(p1=p1).merge(oof_full, on=["s1_idx", "r_idx"]).merge(oof_bl, on=["s1_idx", "r_idx"])
pair["y"] = np.isin(key(pair), key(gt))


def pr(mask, name):
    tp = (mask & pair.y).sum()
    print(f"   {name}: recall {tp / T:.4%}  precision {tp / max(mask.sum(), 1):.4%}  predicted {mask.sum() / len(view):.3f}/S1")


print("2. pair model, p >= 0.5 (before assignment):")
pr(pair.p1.values >= 0.5, "stage 1          ")
pr(pair.p2.values >= 0.5, "stage 2 (full)   ")
pr(pair.pb.values >= 0.5, "blend full+robust")
sel = __import__("json").load(open(f"{bdir}/metrics.json"))["selection"]
d = pair[["s1_idx", "r_idx", "source"]].assign(p=pair.pb.values)
a = postprocess.assign(d, cfg)
a_y = np.isin(key(a), key(gt))
print(f"3. after best-S1 (p >= {cfg['postprocess']['p_min']}) + caps: recall {a_y.sum() / T:.4%}  (links still available to select)")
ch = postprocess.select(a, sel)
c_y = np.isin(key(ch), key(gt))
print(f"4. after expected-F0.5 set selection: recall {c_y.sum() / T:.4%}  precision {c_y.mean():.4%}  predicted {len(ch) / len(view):.3f}/S1")
country, g, (chv,) = pl.train_eval_view(s1, pd.read_parquet(pl.cache("train_gt.parquet")), ch)
r = metrics.report(chv, g, country, cfg["eval"]["test_country_mix"])
print(f"5. macro F0.5: {r['f05_test_mix']:.4f} test mix (India {r['f05_India']:.4f}, US {r['f05_US']:.4f})")

# ------------------------------------------------ what the blocking misses look like
miss = gt[~np.isin(key(gt), key(pair))].copy()
a_, b_ = miss.s1_idx.values, miss.r_idx.values
miss["nsim"] = process.cpdist(s1.name.values[a_], s23.name.values[b_], scorer=fuzz.token_set_ratio, workers=-1)
miss["asim"] = process.cpdist(s1.addr.values[a_], s23.addr.values[b_], scorer=fuzz.token_set_ratio, workers=-1)
miss["r_noaddr"] = s23.addr_empty.values[b_]
miss["r_nonum"] = s23.hn.values[b_] == ""
miss["indic"] = s23.name_indic.values[b_]
miss["web"] = s23.name_web.values[b_]
miss["country"] = s1.country.values[a_]
nc = pd.Series(s1.name_compact.values[view]).value_counts()
miss["namesakes"] = pd.Series(s1.name_compact.values[a_]).map(nc).fillna(1).values
cat = np.select(
    [miss.nsim < 50,
     (miss.nsim >= 85) & miss.r_noaddr,
     (miss.nsim >= 85) & miss.r_nonum,
     (miss.nsim >= 85) & (miss.asim >= 70),
     (miss.nsim >= 85),
     (miss.asim >= 70)],
    ["alias name (name sim < 50)",
     "good name, EMPTY address",
     "good name, address without number",
     "good name + good address",
     "good name, poor address",
     "partial name, good address"], "partial name, poor address")
miss["cat"] = cat
print(f"\nBlocking misses: {len(miss):,} links ({len(miss) / T:.2%}); by type (share, India share, median #S1 with the same name):")
t = miss.groupby("cat").agg(share=("nsim", "size"), india=("country", lambda x: (x == "India").mean()),
                            namesakes=("namesakes", "median"), indic=("indic", "mean"), web=("web", "mean"))
t["share"] = t["share"] / len(miss)
print(t.sort_values("share", ascending=False).round(3).to_string())
rng = np.random.default_rng(1)
for c in t.sort_values("share", ascending=False).index[:5]:
    x = miss[miss.cat == c]
    print(f"\n-- {c}")
    for _, row in x.sample(min(6, len(x)), random_state=2).iterrows():
        print(f"   S1 {s1.name_raw.values[row.s1_idx]!r} | {s1.addr.values[row.s1_idx]!r}   [{int(row.namesakes)} S1 share this name]")
        print(f"   R  {s23.name_raw.values[row.r_idx]!r} | {s23.addr.values[row.r_idx]!r}")
