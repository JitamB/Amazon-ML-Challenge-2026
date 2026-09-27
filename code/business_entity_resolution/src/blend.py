"""Blend the final probabilities of several runs on the first run's candidate set.

    python code/business_entity_resolution/src/blend.py v12_caps2 v12b_s2robust [more runs] [--weights=0.36,0.64]

All runs must use the same prepared-data row order (same data files; cache tags may differ)
and, for tuning, the same train S1 view. The pair set is the first run's (so its
candidate_pairs.tsv stays valid); a pair missing from another run keeps the first run's p.
Selection is re-tuned on the blended out-of-fold predictions, then applied to the blended
test predictions. Writes work/runs/blend_<runs>/ with metrics and both submission files.

--weights gives each run a weight (default: equal). The leaderboard favoured the density-robust
models: v11 full 0.9692, robust 0.9721, 50/50 0.9733; a parabola through these peaks near a
full-model share of 0.36 (e.g. --weights=0.18,0.32,0.18,0.32 for full, robust, full, robust).
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber import metrics, postprocess  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.io import write_id_lists  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

runs = [a for a in sys.argv[1:] if not a.startswith("--")]
w_arg = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--weights=")]
weights = np.array([float(x) for x in w_arg[0].split(",")]) if w_arg else np.ones(len(runs))
assert len(weights) == len(runs), "one weight per run"
cfg = load_config(f"work/runs/{runs[0]}/config.yaml")
pl = Pipeline(cfg)
mix = cfg["eval"].get("test_country_mix")


def blend(name):
    base = pd.read_parquet(f"work/runs/{runs[0]}/{name}.parquet")
    ps = [base["p"].values.astype(np.float64)]
    for r in runs[1:]:
        o = pd.read_parquet(f"work/runs/{r}/{name}.parquet", columns=["s1_idx", "r_idx", "p"])
        m = base[["s1_idx", "r_idx"]].merge(o, on=["s1_idx", "r_idx"], how="left")["p"].values
        print(f"  {name}: {r} covers {np.isfinite(m).mean():.4f} of the pairs")
        ps.append(np.where(np.isfinite(m), m, ps[0]))
    return base.assign(p=np.average(ps, axis=0, weights=weights).astype(np.float32))


s1, s23 = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = blend("oof")
country, g, (assigned,) = pl.train_eval_view(s1, gt, postprocess.assign(postprocess.with_empty(oof, s23), cfg))
params = postprocess.tune(assigned, g, country, cfg)
res = metrics.report(postprocess.select(assigned, params), g, country, mix)
print("blended OOF:", json.dumps({k: round(v, 4) for k, v in res.items()}))
for r in runs:
    print(f"  {r} alone:", json.load(open(f"work/runs/{r}/metrics.json")).get("f05_test_mix"))

s1t, s23t = pl.load_prepared("test")
test = blend("test_preds")
chosen = postprocess.select(postprocess.assign(postprocess.with_empty(test, s23t), cfg), params)
out = f"work/runs/blend_{'+'.join(runs)}"
os.makedirs(out, exist_ok=True)
ids = pd.DataFrame({"s1_id": s1t["rid"].values[chosen["s1_idx"].values],
                    "rid": s23t["rid"].values[chosen["r_idx"].values]})
write_id_lists(f"{out}/matching_results.tsv", s1t["rid"].tolist(), ids, "matched_entity_ids")
shutil.copy(f"work/runs/{runs[0]}/submission/candidate_pairs.tsv", out)
oof.to_parquet(f"{out}/oof.parquet")
test.to_parquet(f"{out}/test_preds.parquet")
json.dump({"runs": runs, "selection": params, **res}, open(f"{out}/metrics.json", "w"), indent=1)
cc = s1t["country"].values[chosen["s1_idx"].values]
print("test matches per S1:", {c: round((cc == c).sum() / (s1t["country"] == c).sum(), 4)
                               for c in s1t["country"].unique()})
print("wrote", out)
# the blend is the final submission: also place it in output/ (paths.output_dir)
od = cfg["paths"]["output_dir"]
os.makedirs(od, exist_ok=True)
for name in ("matching_results.tsv", "candidate_pairs.tsv"):
    shutil.copy(f"{out}/{name}", od)
print("copied to", od)
