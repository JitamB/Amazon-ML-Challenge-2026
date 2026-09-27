"""Deeper leak checks on train links: ID arithmetic, row adjacency, file-order relations."""
import numpy as np
import pandas as pd

d = "dataset/train"
rd = lambda f: pd.read_csv(f"{d}/{f}", sep="\t", quoting=3, dtype=str, keep_default_na=False, usecols=["entity_id"])
s1, s2, s3 = rd("train_source1.tsv"), rd("train_source2.tsv"), rd("train_source3.tsv")
gt = pd.read_csv(f"{d}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.matched_entity_ids != ""].assign(rid=lambda x: x.matched_entity_ids.str.split(",")).explode("rid")
pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id)
pos = pd.concat([pd.Series(np.arange(len(s2)), index=s2.entity_id), pd.Series(np.arange(len(s3)), index=s3.entity_id)])
g = pd.DataFrame({"a": gt.source1_entity_id.str.split("-").str[1].astype(np.int64).values,
                  "b": gt.rid.str.split("-").str[1].astype(np.int64).values,
                  "src": gt.rid.str[:2].values,
                  "p1": pos1.loc[gt.source1_entity_id].values, "pr": pos.loc[gt.rid].values,
                  "s1": gt.source1_entity_id.values})
rng = np.random.default_rng(0)
rnd = g.assign(b=rng.permutation(g.b.values), pr=rng.permutation(g.pr.values))
print("links", len(g))
for m in [2, 3, 7, 10, 100, 1000, 9973]:
    print(f"id mod {m:>5} equal: true {np.mean(g.a % m == g.b % m):.4f}  random {np.mean(rnd.a % m == rnd.b % m):.4f}")
ds = lambda x: np.array([sum(map(int, str(v))) for v in x])
print("digit-sum equal: true", np.mean(ds(g.a.values[:200000]) == ds(g.b.values[:200000])).round(4),
      "random", np.mean(ds(rnd.a.values[:200000]) == ds(rnd.b.values[:200000])).round(4))
print("id length equal: true", np.mean(g.a.astype(str).str.len() == g.b.astype(str).str.len()).round(4),
      "random", np.mean(rnd.a.astype(str).str.len() == rnd.b.astype(str).str.len()).round(4))
print("S1 id length distribution:", s1.entity_id.str.len().value_counts().head().to_dict())
# adjacency: records of the same S1 in the same source file - row distance
for src, n in (("S2", len(s2)), ("S3", len(s3))):
    x = g[g.src == src].sort_values(["s1", "pr"])
    diff = x.groupby("s1").pr.diff().dropna()
    print(f"{src}: median row gap between records of the same S1 {diff.median():.0f} (random ~{n / 3:.0f}); share gap<=10: {(diff <= 10).mean():.4f}")
# relative order: S1 position vs record position
r1, rr = g.p1 / len(s1), g.pr / np.where(g.src == "S2", len(s2), len(s3))
print("corr(rel pos S1, rel pos record):", np.corrcoef(r1, rr)[0, 1].round(4), " share |diff|<0.01:", np.mean(np.abs(r1 - rr) < 0.01).round(4), "(random 0.02)")
print("corr(id a, id b):", np.corrcoef(g.a, g.b)[0, 1].round(4))
