"""Stage 3: re-score a blend's pairs with record-cluster features the pair models never saw.

    python code/business_entity_resolution/src/stage3.py <run> <blend_dir> [robust]

Orphan decoys (records of no S1, 26% of S2/S3) are single perturbed copies, while an
entity's true records often repeat each other exactly: 21% of true positives have an
exact twin among their S1's candidates, against 1.4% of orphan false positives. The
features here describe that (twins, shared number sets, global duplicates) plus the
pair's number relation to the S1, on top of the blend's logit. Only pair- and
record-level quantities are used (no S1-level counts, which shift with test density).
LightGBM, 3 folds by S1 hash; writes <blend_dir>_s3/ with oof, test_preds, metrics and
the submission files. "robust" keeps only features that do not count other records (no
twins / shared numbers / global duplicates) and writes <blend_dir>_s3r/.
"""
import json
import os
import shutil
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402
from ber.stage2 import _max_other  # noqa: E402

run, bdir = sys.argv[1], sys.argv[2].rstrip("/")
robust = len(sys.argv) > 3 and sys.argv[3] == "robust"
out = f"{bdir}_s3r" if robust else f"{bdir}_s3"
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"].get("test_country_mix")
FEATS = ["lp", "p_maxother_r", "dup_global", "twin_n", "twin_p", "same_nums_n", "same_nums_p",
         "same_addr_n", "same_addr_p", "first_bare_novel", "s1hn_missing", "r_extra_nums",
         "r_hn_shared_p", "addr_empty", "source"]
if robust:
    FEATS = ["lp", "p_maxother_r", "first_bare_novel", "s1hn_missing", "r_extra_nums", "addr_empty", "source"]


def features(split, d):
    s1 = pd.read_parquet(pl.cache(f"{split}_s1.parquet"), columns=["hn", "nums"])
    s23 = pd.read_parquet(pl.cache(f"{split}_s23.parquet"), columns=["country", "name", "addr", "nums", "hn", "addr_empty"])
    s, r = d.s1_idx.values, d.r_idx.values
    p = d.p.values.astype(np.float64)
    f = pd.DataFrame({"lp": np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)),
                      "source": d.source.values.astype(np.int8),
                      "addr_empty": s23.addr_empty.values[r].astype(np.int8)})
    f["p_maxother_r"] = _max_other(r, p)
    codes = pd.factorize(s23.country + "|" + s23.name + "|" + s23.addr)[0]
    f["dup_global"] = (np.bincount(codes)[codes] - 1)[r]
    for name, col in (("twin", codes), ("same_nums", pd.factorize(s23.nums)[0]),
                      ("same_addr", pd.factorize(s23.addr)[0])):
        c = col[r]
        if name != "twin":
            c = np.where((s23[name.split("_")[1]].values[r] == ""), -1 - np.arange(len(r)), c)
        g = pd.Series(p).groupby([s, c])
        f[f"{name}_n"] = (g.transform("size").values - 1).astype(np.int16)
        f[f"{name}_p"] = (g.transform("sum").values - p).astype(np.float32)
    a_nums = s1.nums.values[s]
    b_nums, b_addr = s23.nums.values[r], s23.addr.values[r]
    a_sets = [set(x.split()) for x in a_nums]
    f["first_bare_novel"] = np.array([bool(t) and t.split()[0].isdigit() and t.split()[0].lstrip("0") not in sa
                                      for t, sa in zip(b_addr, a_sets)], dtype=np.int8)
    f["s1hn_missing"] = np.array([h != "" and x != "" and h not in x.split()
                                  for h, x in zip(s1.hn.values[s], b_nums)], dtype=np.int8)
    f["r_extra_nums"] = np.array([len(set(x.split()) - sa) for x, sa in zip(b_nums, a_sets)], dtype=np.int8)
    # r's first number is not the S1's: how much p-mass of the S1's other candidates carries it
    hn_r = s23.hn.values[r]
    hc = np.where(hn_r == "", -1 - np.arange(len(r)), pd.factorize(hn_r)[0])
    f["r_hn_shared_p"] = np.where(hn_r != s1.hn.values[s],
                                  pd.Series(p).groupby([s, hc]).transform("sum").values - p, -1).astype(np.float32)
    return f


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


s1_tr = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["rid", "country"])
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{bdir}/oof.parquet")
Xtr = features("train", oof)
n_r = int(max(oof.r_idx.max(), gt.r_idx.max())) + 1
y = np.isin(oof.s1_idx.values.astype(np.int64) * n_r + oof.r_idx.values,
            gt.s1_idx.values.astype(np.int64) * n_r + gt.r_idx.values)
fold = pd.util.hash_array(s1_tr.rid.values.astype(object))[oof.s1_idx.values] % 3
params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 16}
p3 = np.zeros(len(oof))
boosters = []
for k in range(3):
    tr, va = fold != k, fold == k
    b = lgb.train(params, lgb.Dataset(Xtr[FEATS][tr], y[tr]), 2000,
                  valid_sets=[lgb.Dataset(Xtr[FEATS][va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p3[va] = b.predict(Xtr[FEATS][va], num_iteration=b.best_iteration)
    boosters.append(b)
    print(f"fold {k}: best_iter {b.best_iteration}", flush=True)
imp = pd.Series(sum(b.feature_importance("gain") for b in boosters), index=FEATS).sort_values(ascending=False)
print("importance:", (imp / imp.sum()).round(3).to_dict())


def evaluate(pp, label):
    d = oof.assign(p=pp.astype(np.float32))
    country, g, (a,) = pl.train_eval_view(s1_tr, gt, postprocess.assign(d, cfg))
    prm = postprocess.tune(a, g, country, cfg, log=lambda *_: None)
    res = metrics.report(postprocess.select(a, prm), g, country, mix)
    print(f"{label:28s}", json.dumps({k: round(v, 5) for k, v in res.items()}), prm, flush=True)
    return res, prm


evaluate(oof.p.values, "blend (baseline)")
r3, prm3 = evaluate(p3, "stage 3")
pa = 1 / (1 + np.exp(-(logit(p3) + logit(oof.p.values)) / 2))
ra, prma = evaluate(pa, "avg logit(blend, stage 3)")
use_avg = ra.get("f05_test_mix", 0) > r3.get("f05_test_mix", 0)
prm = prma if use_avg else prm3
p_final = pa if use_avg else p3

test = pd.read_parquet(f"{bdir}/test_preds.parquet")
Xte = features("test", test)
print("feature means train (p>0.5) vs test (p>0.5):")
print(pd.DataFrame({"train": Xtr[FEATS][oof.p.values > .5].mean(), "test": Xte[FEATS][test.p.values > .5].mean()}).round(3).to_string())
pt = np.mean([b.predict(Xte[FEATS], num_iteration=b.best_iteration) for b in boosters], axis=0)
if use_avg:
    pt = 1 / (1 + np.exp(-(logit(pt) + logit(test.p.values)) / 2))
os.makedirs(out, exist_ok=True)
oof.assign(p=p_final.astype(np.float32)).to_parquet(f"{out}/oof.parquet")
test = test.assign(p=pt.astype(np.float32))
test.to_parquet(f"{out}/test_preds.parquet")
res = r3 if not use_avg else ra
json.dump({"stage3_of": bdir, "averaged": bool(use_avg), "selection": prm, **res}, open(f"{out}/metrics.json", "w"), indent=1)
s1t = pd.read_parquet(pl.cache("test_s1.parquet"), columns=["rid", "country"])
s23t = pd.read_parquet(pl.cache("test_s23.parquet"), columns=["rid"])
chosen = postprocess.select(postprocess.assign(test, cfg), prm)
ids = pd.DataFrame({"s1_id": s1t.rid.values[chosen.s1_idx.values], "rid": s23t.rid.values[chosen.r_idx.values]})
write_id_lists(f"{out}/matching_results.tsv", s1t.rid.tolist(), ids, "matched_entity_ids")
shutil.copy(f"{bdir}/candidate_pairs.tsv", out)
cc = s1t.country.values[chosen.s1_idx.values]
print("test matches per S1:", {c: round((cc == c).sum() / (s1t.country == c).sum(), 4) for c in s1t.country.unique()})
print("wrote", out)
