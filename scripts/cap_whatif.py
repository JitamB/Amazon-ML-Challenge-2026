"""What-if for the blocking frequency caps: which caps would recover missed true links, and
at what cost in raw pairs? For each missed link (sample of train S1s) the *most specific*
shared key is found (lowest S1 frequency); for cap settings the script reports the share of
misses whose best key would survive, and the raw pair count sum(df1 * df2) the newly allowed
keys would add, by key type.
    python scripts/cap_whatif.py <run_dir> [n_s1_sample]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.blocking import _row_keys  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

run_dir = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
cfg = load_config(os.path.join(run_dir, "config.yaml"))
pl = Pipeline(cfg)
bc = cfg["blocking"]
kt = bc["key_types"]
s1, s23 = pl.load_prepared("train")
gt = pd.read_parquet(pl.cache("train_gt.parquet"))
pairs = pd.read_parquet(os.path.join(run_dir, "train_pairs.parquet"), columns=["s1_idx", "r_idx"])
view = pl.s1_subset("train", s1)
cols = ["country", "name", "name_compact", "phon", "addr_alpha", "hn", "nums"]


def keys_of(df):
    out = []
    for i, row in zip(df.index, df[cols].itertuples(index=False)):
        for k in set(_row_keys(row, kt)):
            out.append((i, row.country + "|" + k))
    return pd.DataFrame(out, columns=["idx", "key"])


# full key frequencies (strings, only for keys that matter: those of sampled S1s)
rng = np.random.default_rng(0)
sample = rng.choice(view, n_sample, replace=False)
g = gt[np.isin(gt.s1_idx.values, sample)]
n_r = len(s23)
found = np.isin(g.s1_idx.values * n_r + g.r_idx.values, pairs.s1_idx.values * n_r + pairs.r_idx.values)
miss = g[~found]
print(f"sample {n_sample:,} S1s: links {len(g):,}, missed {len(miss):,} ({len(miss) / len(g):.2%})")
k1 = keys_of(s1.loc[miss.s1_idx.unique()])
k2 = keys_of(s23.loc[miss.r_idx.unique()])
sh = miss.merge(k1.rename(columns={"idx": "s1_idx"}), on="s1_idx").merge(k2.rename(columns={"idx": "r_idx"}), on=["r_idx", "key"])
need = set(sh.key)
# document frequencies of the needed keys over the whole view
from collections import Counter  # noqa: E402
df1, df2 = Counter(), Counter()
for i, row in zip(view, s1.loc[view, cols].itertuples(index=False)):
    for k in set(_row_keys(row, kt)):
        k = row.country + "|" + k
        if k in need:
            df1[k] += 1
for row in s23[cols].itertuples(index=False):
    for k in set(_row_keys(row, kt)):
        k = row.country + "|" + k
        if k in need:
            df2[k] += 1
sh["df1"] = sh.key.map(df1).astype(int)
sh["df2"] = sh.key.map(df2).astype(int)
sh["type"] = sh.key.str.split("|").str[1]
best = sh.sort_values(["df1", "df2"]).drop_duplicates(["s1_idx", "r_idx"])
print("no shared key at all:", round(1 - len(best) / len(miss), 3))
print("most specific shared key, quantiles of S1 frequency:", best.df1.quantile([.25, .5, .75, .9]).to_dict())
print("... and of S2/S3 frequency:", best.df2.quantile([.25, .5, .75, .9]).to_dict())
print("key type of the most specific shared key:", best.type.value_counts(normalize=True).round(3).head(10).to_dict())
for c1, c2 in [(30, 400), (60, 800), (100, 1500), (200, 3000), (500, 10000)]:
    ok = (best.df1 <= c1) & (best.df2 <= c2)
    print(f"caps S1<={c1:>3}, S2/S3<={c2:>5}: {ok.mean():.3f} of misses would share an allowed key")
