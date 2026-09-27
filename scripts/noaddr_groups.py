"""Do empty-address records with a shared name concentrate on one of the namesake S1s?"""
import numpy as np
import pandas as pd

c = "work/cache/dataset_translit_addr_guard_fr_acr"
s1 = pd.read_parquet(f"{c}/train_s1.parquet", columns=["country", "name_compact", "name_raw"])
s23 = pd.read_parquet(f"{c}/train_s23.parquet", columns=["country", "name_compact", "addr_empty", "source"])
gt = pd.read_parquet(f"{c}/train_gt.parquet")
owner = np.full(len(s23), -1); owner[gt.r_idx.values] = gt.s1_idx.values
r = s23.assign(owner=owner)
e = r[r.addr_empty & (r.name_compact.str.len() >= 4)]
print(f"empty-address records: {len(e):,}; matched {(e.owner >= 0).mean():.3f}")
ns = s1.groupby(["country", "name_compact"]).size().rename("n_s1")
e = e.join(ns, on=["country", "name_compact"])
e["n_s1"] = e.n_s1.fillna(0)
print("by number of S1s sharing the record's exact compact name:")
print(e.groupby(pd.cut(e.n_s1, [-1, 0, 1, 2, 5, 20, 100, 1e9])).agg(records=("owner", "size"),
      matched=("owner", lambda x: (x >= 0).mean())).to_string())
# owner's name == record's name?
e["owner_same_name"] = [(o >= 0) and (s1.name_compact.values[o] == n) for o, n in zip(e.owner.values, e.name_compact.values)]
g = e[e.n_s1 >= 2]
print(f"\nrecords whose name is shared by >=2 S1s: {len(g):,}; owner is one of those namesakes: {g.owner_same_name.mean():.3f}")
# concentration: within a (country, name) group, how are the empty-address records spread over namesakes?
gg = g[g.owner_same_name].groupby(["country", "name_compact"])
conc = gg.owner.agg(lambda x: x.value_counts().iloc[0] / len(x))
size = gg.size()
k = g[g.owner_same_name].groupby(["country", "name_compact"]).n_s1.first()
m = size >= 2
print(f"groups with >=2 such records: {m.sum():,}; share of records held by the single most common owner: mean {conc[m].mean():.3f} "
      f"(uniform would be ~{(1 / np.minimum(k[m], size[m])).mean():.3f}); groups where ONE S1 owns all of them: {(conc[m] == 1).mean():.3f}")
# how many distinct owners per group vs group size
print("records per group vs distinct owners (first rows):")
d = pd.DataFrame({"records": size[m], "owners": gg.owner.nunique()[m], "n_s1": k[m]})
print(d.groupby(pd.cut(d.records, [1, 2, 4, 8, 16, 1e9])).agg(groups=("owners", "size"), mean_owners=("owners", "mean"),
      mean_records=("records", "mean"), mean_namesakes=("n_s1", "mean")).round(2).to_string())
