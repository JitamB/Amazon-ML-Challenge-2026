"""Compare two runs' test predictions by country: match counts, empty share, agreement.
    python scripts/compare_runs.py v4_translit v5b_fullfold
"""
import sys

import pandas as pd

s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", quoting=3, usecols=["entity_id", "country"],
                 keep_default_na=False, dtype=str)
runs = {}
for r in sys.argv[1:3]:
    m = pd.read_csv(f"work/runs/{r}/submission/matching_results.tsv", sep="\t", keep_default_na=False, dtype=str)
    m.columns = ["s1", "ids"]
    m["set"] = [frozenset(x.split(",")) if x else frozenset() for x in m["ids"]]
    runs[r] = m.set_index("s1")["set"]
a, b = sys.argv[1:3]
d = s1.set_index("entity_id")
d["na"] = runs[a].reindex(d.index).map(lambda s: len(s) if isinstance(s, frozenset) else 0)
d["nb"] = runs[b].reindex(d.index).map(lambda s: len(s) if isinstance(s, frozenset) else 0)
sa, sb = runs[a].reindex(d.index), runs[b].reindex(d.index)
d["jac"] = [len(x & y) / len(x | y) if (x or y) else 1.0 for x, y in zip(sa, sb)]
g = d.groupby("country")
print(pd.DataFrame({f"mean_n_{a}": g["na"].mean(), f"mean_n_{b}": g["nb"].mean(),
                    f"empty_{a}": g["na"].apply(lambda x: (x == 0).mean()),
                    f"empty_{b}": g["nb"].apply(lambda x: (x == 0).mean()),
                    "mean_jaccard": g["jac"].mean(), "identical": g["jac"].apply(lambda x: (x == 1).mean())}).round(4).to_string())
