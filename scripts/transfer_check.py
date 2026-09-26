"""Why might India/US score lower on the leaderboard than in validation? Label-free train vs test checks.
1. Strong candidates per S1 (ranker score > 0.5 / 0.9): more on test = more lookalike decoys per S1.
2. Learned address map: share of records it rewrites, train vs test (a much higher train share = in-sample fit).
3. Learned transliteration map: share of Indic-name words it covers, train vs test.
"""
import json
import re

import pandas as pd

run = "work/runs/v8_density"
for sp in ["train", "test"]:
    s1 = pd.read_parquet(f"work/cache/dataset_translit_addr_guard_fr/{sp}_s1.parquet", columns=["country"])
    p = pd.read_parquet(f"{run}/{sp}_pairs.parquet", columns=["s1_idx", "cscore"])
    p["country"] = s1["country"].values[p["s1_idx"].values]
    n_s1 = p.groupby("country")["s1_idx"].nunique()
    out = pd.DataFrame({t: p[p.cscore > t].groupby("country").size() / n_s1 for t in (0.5, 0.9)}).round(3)
    print(f"{sp}: candidates per S1 with ranker score > t\n{out.to_string()}")

for sp in ["train", "test"]:
    a = pd.read_parquet(f"work/cache/dataset_translit/{sp}_s23.parquet", columns=["country", "addr"])
    b = pd.read_parquet(f"work/cache/dataset_translit_addr_guard/{sp}_s23.parquet", columns=["addr"])
    ch = pd.Series(a["addr"].values != b["addr"].values)
    print(f"{sp}: address-map rewrite share", ch.groupby(a["country"].values).mean().round(3).to_dict())

tr = json.load(open("work/cache/dataset_translit/translit.json"))
for sp in ["train", "test"]:
    d = pd.read_parquet(f"work/cache/dataset_translit/{sp}_s23.parquet", columns=["name_indic", "name_raw"])
    d = d[d["name_indic"]]
    # after mapping, a covered word became its English target; count names with >= 1 target word
    targets = set(tr.values())
    hit = [any(t in targets for t in re.sub(r"[^a-z0-9]+", " ", x).split()) for x in d["name_raw"]]
    print(f"{sp}: Indic names {len(d):,}, share containing a mapped word {sum(hit) / len(d):.3f}")
