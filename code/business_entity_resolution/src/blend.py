"""Blend the final probabilities of several runs on the first run's candidate set.

    python code/business_entity_resolution/src/blend.py v12_caps2 v12b_s2robust [more runs]

All runs must use the same prepared-data row order (same data files; cache tags may differ)
and, for tuning, the same train S1 view. The pair set is the first run's (so its
candidate_pairs.tsv stays valid); a pair missing from another run keeps the first run's p.
Selection is re-tuned on the blended out-of-fold predictions, then applied to the blended
test predictions. Writes work/runs/blend_<runs>/ with metrics and both submission files.
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

runs = sys.argv[1:]
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
    return base.assign(p=np.mean(ps, axis=0).astype(np.float32))


s1, _ = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
oof = blend("oof")
country, g, (assigned,) = pl.train_eval_view(s1, gt, postprocess.assign(oof, cfg))
params = postprocess.tune(assigned, g, country, cfg)
res = metrics.report(postprocess.select(assigned, params), g, country, mix)
print("blended OOF:", json.dumps({k: round(v, 4) for k, v in res.items()}))
for r in runs:
    print(f"  {r} alone:", json.load(open(f"work/runs/{r}/metrics.json")).get("f05_test_mix"))

s1t, s23t = pl.load_prepared("test")
test = blend("test_preds")
chosen = postprocess.select(postprocess.assign(test, cfg), params)
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
