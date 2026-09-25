"""Reading source/ground-truth TSVs and writing submission files."""
import os

import pandas as pd


def read_source(path):
    # quoting=3: fields are never quoted; keep_default_na=False: "" stays "", "null" stays text.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def read_ground_truth(path):
    """Return long-format (s1_id, rid) pairs plus the list of all S1 ids in the file."""
    gt = read_source(path)
    s1_ids = gt["source1_entity_id"].tolist()
    lists = gt["matched_entity_ids"].str.split(",")
    pairs = pd.DataFrame({"s1_id": gt["source1_entity_id"].repeat(lists.str.len()),
                          "rid": [x for l in lists for x in l]})
    pairs = pairs[pairs["rid"] != ""].reset_index(drop=True)
    return pairs, s1_ids


def write_id_lists(path, s1_ids, pairs, col_name):
    """Write one row per S1 id with a comma-joined list of rids (empty if none)."""
    grouped = pairs.groupby("s1_id")["rid"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col_name] = out["source1_entity_id"].map(grouped).fillna("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, sep="\t", index=False, encoding="utf-8")
