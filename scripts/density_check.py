"""Records per S1 by country, train vs test (the test has more unmatched records per S1?)."""
import pandas as pd

c = "work/cache/dataset_translit_addr"
for split in ["train", "test"]:
    s1 = pd.read_parquet(f"{c}/{split}_s1.parquet", columns=["country"])
    s23 = pd.read_parquet(f"{c}/{split}_s23.parquet", columns=["country", "source"])
    t = pd.DataFrame({"s1": s1["country"].value_counts(), "s23": s23["country"].value_counts()})
    t["rec_per_s1"] = (t["s23"] / t["s1"]).round(3)
    print(split, "\n", t.to_string(), "\n total rec/S1", round(len(s23) / len(s1), 3))
gt = pd.read_parquet(f"{c}/train_gt.parquet")
s1 = pd.read_parquet(f"{c}/train_s1.parquet", columns=["country"])
print("train links per S1 by country:\n", (gt.groupby(s1["country"].values[gt["s1_idx"].values]).size()
                                           / s1["country"].value_counts()).round(3).to_string())
