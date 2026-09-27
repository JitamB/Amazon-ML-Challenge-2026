"""Validation at the test's decoy density, and a correction for it.

The test has ~1.9x (US) / ~1.65x (India) more decoy-type records per S1 than train:
strong name match to an S1 but house number shifted by 1-20 (scripts/decoy_rates.py).
This script
  1. duplicates unmatched decoy-type train records (new record ids, same out-of-fold
     scores) until train has the test's decoy density, and scores the out-of-fold
     predictions on that test-like view;
  2. tries corrections: shrinking the odds of decoy-type pairs by a factor, re-tuned on
     the test-like view.
With --apply it writes a test submission using the best correction.

    python scripts/decoy_sim.py <config_run> [preds_dir] [--apply]
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, "code/business_entity_resolution/src")
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
apply = "--apply" in sys.argv
run = args[0]
pdir = args[1] if len(args) > 1 else f"work/runs/{run}"
cfg = load_config(f"work/runs/{run}/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"].get("test_country_mix")
TARGET = {"US": 1.429, "India": 0.715}  # test decoy-type records per S1 (decoy_rates.py)


def rel_arrays(s1, s23, d):
    nsim = process.cpdist(s1["name"].values[d.s1_idx.values], s23["name"].values[d.r_idx.values],
                          scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    a, b = s1["hn"].values[d.s1_idx.values], s23["hn"].values[d.r_idx.values]
    shift = np.array([bool(x) and bool(y) and x != y and x not in y and y not in x and len(x) < 15 and len(y) < 15
                      and abs(int(x) - int(y)) <= 20 for x, y in zip(a, b)])
    return nsim, shift & (nsim >= 85)


def shrink(d, mask, k):
    p = np.clip(d["p"].values.astype(np.float64), 1e-7, 1 - 1e-7)
    o = p / (1 - p)
    o = np.where(mask, o * k, o)
    return d.assign(p=(o / (1 + o)).astype(np.float32))


# ---------------------------------------------------------------- train: test-like view
s1, s23 = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = pd.read_parquet(f"{pdir}/oof.parquet")[["s1_idx", "r_idx", "source", "p"]]
nsim, dmask = rel_arrays(s1, s23, oof)
oof["nsim"], oof["decoy_type"] = nsim, dmask
n_r = len(s23)
view = pl.s1_subset("train", s1)
gtv = gt[np.isin(gt.s1_idx.values, view)]
matched = np.zeros(n_r, bool); matched[gtv.r_idx.values] = True
# record-level: best-name S1 of each record, decoy if that pair is decoy-type and the record is unmatched
best = oof.sort_values(["r_idx", "nsim"], ascending=[True, False]).drop_duplicates("r_idx")
best = best[best.decoy_type & ~matched[best.r_idx.values]]
best_cty = s1["country"].values[best.s1_idx.values]
rng = np.random.default_rng(0)
dup = []
n_view = pd.Series(s1["country"].values[view]).value_counts()
for c, target in TARGET.items():
    cur_all = oof[(oof.decoy_type) & (s1["country"].values[oof.s1_idx.values] == c)]
    cur = cur_all.sort_values(["r_idx", "nsim"], ascending=[True, False]).drop_duplicates("r_idx").shape[0] / n_view[c]
    cand = best.r_idx.values[best_cty == c]
    prob = min(1.0, (target - cur) / (len(cand) / n_view[c]))
    pick = cand[rng.random(len(cand)) < prob]
    dup.append(pick)
    print(f"{c}: decoy-type records per S1 {cur:.3f} -> target {target}; duplicating {len(pick):,} unmatched decoy records (p={prob:.2f})")
dup = np.concatenate(dup)
extra = oof[np.isin(oof.r_idx.values, dup)].copy()
remap = pd.Series(np.arange(len(dup)) + n_r, index=dup)
extra["r_idx"] = remap.loc[extra.r_idx.values].values
sim = pd.concat([oof, extra], ignore_index=True)
mask_sim = sim["decoy_type"].values

sel = json.load(open(f"{pdir}/metrics.json")).get("selection") if os.path.exists(f"{pdir}/metrics.json") and \
    "selection" in json.load(open(f"{pdir}/metrics.json")) else json.load(open(f"work/runs/{run}/selection.json"))


def evaluate(d, params=None):
    country, g, (a,) = pl.train_eval_view(s1, gt, postprocess.assign(d, cfg))
    if params is None:
        params = postprocess.tune(a, g, country, cfg, log=lambda m: None)
    return metrics.report(postprocess.select(a, params), g, country, mix), params


base, _ = evaluate(oof, sel)
print(f"\noriginal view, current selection:   test-mix {base['f05_test_mix']:.4f} (India {base['f05_India']:.4f}, US {base['f05_US']:.4f})")
r0, _ = evaluate(sim, sel)
print(f"test-like view, current selection:  test-mix {r0['f05_test_mix']:.4f} (India {r0['f05_India']:.4f}, US {r0['f05_US']:.4f})")
best_k, best_r, best_params = 1.0, r0, sel
for k in [0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05]:
    r, params = evaluate(shrink(sim, mask_sim, k))
    print(f"  shrink decoy-type odds x{k:<5} + retune -> test-like {r['f05_test_mix']:.4f} (India {r['f05_India']:.4f}, US {r['f05_US']:.4f}) {params}")
    if r["f05_test_mix"] > best_r["f05_test_mix"]:
        best_k, best_r, best_params = k, r, params
ro, _ = evaluate(shrink(oof, oof["decoy_type"].values, best_k), best_params)
print(f"\nbest: x{best_k} {best_params}: test-like {best_r['f05_test_mix']:.4f} "
      f"(+{best_r['f05_test_mix'] - r0['f05_test_mix']:.4f}); on the original view {ro['f05_test_mix']:.4f}")

if apply:
    s1t, s23t = pl.load_prepared("test")
    test = pd.read_parquet(f"{pdir}/test_preds.parquet")[["s1_idx", "r_idx", "source", "p"]]
    _, tmask = rel_arrays(s1t, s23t, test)
    chosen = postprocess.select(postprocess.assign(shrink(test, tmask, best_k), cfg), best_params)
    out = f"{pdir.rstrip('/')}_decoyfix"
    os.makedirs(out, exist_ok=True)
    ids = pd.DataFrame({"s1_id": s1t["rid"].values[chosen.s1_idx.values], "rid": s23t["rid"].values[chosen.r_idx.values]})
    write_id_lists(f"{out}/matching_results.tsv", s1t["rid"].tolist(), ids, "matched_entity_ids")
    shutil.copy(f"work/runs/{run}/submission/candidate_pairs.tsv", out)
    cc = s1t["country"].values[chosen.s1_idx.values]
    print("test decoy-type pairs:", int(tmask.sum()), "matches per S1:",
          {c: round((cc == c).sum() / (s1t["country"] == c).sum(), 4) for c in s1t["country"].unique()})
    json.dump({"shrink": best_k, "selection": best_params, "test_like": best_r}, open(f"{out}/metrics.json", "w"), indent=1)
    print("wrote", out)
