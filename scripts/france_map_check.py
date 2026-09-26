"""Which learned address-map words fire on French test records (map is learned on India/US only)?"""
import json
import re
from collections import Counter

import pandas as pd

m = json.load(open("work/cache/dataset_translit_addr/addr_map.json"))
old = pd.read_parquet("work/cache/dataset_translit/test_s23.parquet", columns=["country", "addr"])
s1 = pd.read_parquet("work/cache/dataset_translit/test_s1.parquet", columns=["country", "addr"])
fr = old[old["country"] == "France"]["addr"]
cnt = Counter(t for a in fr for t in a.split() if t in m)
n_rows = sum(any(t in m for t in a.split()) for a in fr)
print(f"French S2/S3 records: {len(fr):,}; rewritten by the map: {n_rows:,} ({n_rows / len(fr):.1%})")
s1fr = Counter(t for a in s1[s1["country"] == "France"]["addr"] for t in a.split())
print("top rewritten words (count in FR S2/S3 -> target, count of source word in FR S1, of target in FR S1):")
for w, c in cnt.most_common(40):
    print(f"  {w:>14} -> {m[w]:<14} {c:>8,}  S1 has src {s1fr[w]:>8,}  tgt {s1fr[m[w]]:>8,}")
for cc in ["India", "US"]:
    x = old[old["country"] == cc]["addr"]
    print(cc, f"rewritten share {sum(any(t in m for t in a.split()) for a in x.sample(200000, random_state=0)) / 200000:.1%}")
