"""Why do true matches miss blocking? Run on the remote from the project root:
    python scripts/diag_blocking.py work/runs/v1_gbdt [n_s1_sample]

For a sample of train S1 it splits every missed true link into:
  - no_shared_key : the pair shares no key at all, even before frequency caps
  - capped        : shares keys, but all of them were dropped by max_df caps
  - cut_by_top_n  : survived the caps but ranked below top_n
and breaks misses down by record traits (Indic name, empty address, website name, country, source).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.blocking import make_keys  # noqa: E402
from ber.config import load_config  # noqa: E402

run_dir = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
cfg = load_config(os.path.join(run_dir, "config.yaml"))
bc = cfg["blocking"]
cache = os.path.join(cfg["paths"]["work_dir"], "cache", "dataset")
s1 = pd.read_parquet(f"{cache}/train_s1.parquet")
s23 = pd.read_parquet(f"{cache}/train_s23.parquet")
gt = pd.read_parquet(f"{cache}/train_gt.parquet")
pairs = pd.read_parquet(os.path.join(run_dir, "train_pairs.parquet"), columns=["s1_idx", "r_idx"])

rng = np.random.default_rng(0)
sample = rng.choice(pairs["s1_idx"].unique(), n_sample, replace=False)
g = gt[gt["s1_idx"].isin(sample)]
n_r = len(s23)
found = set(pairs.loc[pairs["s1_idx"].isin(sample), "s1_idx"].values * n_r
            + pairs.loc[pairs["s1_idx"].isin(sample), "r_idx"].values)
g = g.assign(hit=np.isin(g["s1_idx"].values * n_r + g["r_idx"].values, list(found)))
miss = g[~g["hit"]]
print(f"sample S1 {n_sample:,}: links {len(g):,}, missed {len(miss):,} ({len(miss) / len(g):.2%})")

# keys of the sampled S1 and of the missed records only (cheap), plus global key frequencies
k1 = make_keys(s1.iloc[sample].reset_index(drop=True), bc["key_types"])
k1["idx"] = sample[k1["idx"].values]
k_miss = make_keys(s23.iloc[miss["r_idx"].values].reset_index(drop=True), bc["key_types"])
k_miss["idx"] = miss["r_idx"].values[k_miss["idx"].values]
all_k2 = make_keys(s23, bc["key_types"], n_jobs=cfg["features"]["n_jobs"])
df2 = all_k2["key"].value_counts()
all_k1 = make_keys(s1, bc["key_types"], n_jobs=cfg["features"]["n_jobs"])
df1 = all_k1["key"].value_counts()

shared = miss.merge(k1.rename(columns={"idx": "s1_idx"}), on="s1_idx").merge(
    k_miss.rename(columns={"idx": "r_idx"}), on=["r_idx", "key"])
shared["ok"] = (shared["key"].map(df1).values <= bc["max_df_s1"]) & (shared["key"].map(df2).values <= bc["max_df_s23"])
per = shared.groupby(["s1_idx", "r_idx"])["ok"].agg(["size", "sum"])
miss = miss.merge(per, left_on=["s1_idx", "r_idx"], right_index=True, how="left").fillna(0)
miss["reason"] = np.where(miss["size"] == 0, "no_shared_key", np.where(miss["sum"] == 0, "capped", "cut_by_top_n"))
print("\nmiss reasons:\n", miss["reason"].value_counts(normalize=True).round(3).to_string())

r = s23.iloc[miss["r_idx"].values].reset_index(drop=True)
a = s1.iloc[miss["s1_idx"].values].reset_index(drop=True)
traits = pd.DataFrame({"reason": miss["reason"].values, "country": a["country"].values,
                       "source": r["source"].values, "indic_name": r["name_indic"].values,
                       "web_name": r["name_web"].values, "empty_addr": r["addr_empty"].values})
print("\nmissed-record traits (share of misses):")
for c in ["country", "source", "indic_name", "web_name", "empty_addr"]:
    print(" ", c, traits[c].value_counts(normalize=True).round(3).to_dict())
print("\nby reason x indic_name:\n", pd.crosstab(traits["reason"], traits["indic_name"]).to_string())

# how frequent were the keys that the caps dropped
cap = shared[~shared["ok"]]
print(f"\ncapped key df_s23 quantiles: {cap['key'].map(df2).quantile([.5, .75, .9]).round(0).to_dict()}")
print(f"capped key df_s1  quantiles: {cap['key'].map(df1).quantile([.5, .75, .9]).round(0).to_dict()}")

print("\nexamples of misses:")
ex = miss.sample(min(25, len(miss)), random_state=1)
for s, rr, why in zip(ex["s1_idx"], ex["r_idx"], ex["reason"]):
    print(f"  [{why}] S1: {s1.at[s, 'name_raw']!r} | {s1.at[s, 'addr']!r}\n      R : {s23.at[rr, 'name_raw']!r} | {s23.at[rr, 'addr']!r}")
