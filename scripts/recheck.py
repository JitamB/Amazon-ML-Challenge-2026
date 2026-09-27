"""Sanity checks on a final submission, beyond the official validator.

    python scripts/recheck.py <submission_dir> [cache_dir]

1. every matched id exists in test S2/S3, and no id is matched to two S1s;
2. no match crosses countries;
3. every matched pair is in candidate_pairs.tsv (the model only scored those);
4. matches per S1 and empty lists per country.
"""
import sys

import numpy as np
import pandas as pd

sub = sys.argv[1]
cache = sys.argv[2] if len(sys.argv) > 2 else "work/cache/dataset_translit_addr_guard_fr_acr"


def explode(path, col):
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    s1col = d.columns[0]
    d = d.assign(rid=d[col].str.split(",")).explode("rid")
    return d[d.rid.fillna("") != ""][[s1col, "rid"]].rename(columns={s1col: "s1_id"})


m = explode(f"{sub}/matching_results.tsv", "matched_entity_ids")
c = explode(f"{sub}/candidate_pairs.tsv", pd.read_csv(f"{sub}/candidate_pairs.tsv", sep="\t", nrows=1).columns[1])
s1 = pd.read_parquet(f"{cache}/test_s1.parquet", columns=["rid", "country"]).set_index("rid").country
s23 = pd.read_parquet(f"{cache}/test_s23.parquet", columns=["rid", "country"]).set_index("rid").country
print(f"matches {len(m):,}, candidate pairs {len(c):,}")
print("1. matched ids missing from test S2/S3:", (~m.rid.isin(s23.index)).sum(),
      "| ids matched to 2+ S1s:", (m.rid.value_counts() > 1).sum(),
      "| candidate ids missing:", (~c.rid.isin(s23.index)).sum())
print("2. cross-country matches:", (s1.reindex(m.s1_id).values != s23.reindex(m.rid).values).sum())
key = lambda d: d.s1_id + "|" + d.rid
print("3. matches not in candidate_pairs:", (~key(m).isin(set(key(c)))).sum())
per = m.s1_id.map(s1).value_counts() / s1.value_counts()
cand = c.s1_id.map(s1).value_counts() / s1.value_counts()
empty = 1 - m.groupby(m.s1_id.map(s1)).s1_id.nunique() / s1.value_counts()
print("4. per country:\n", pd.DataFrame({"matches/S1": per.round(3), "candidates/S1": cand.round(2),
                                         "empty share": empty.round(4)}).to_string())
