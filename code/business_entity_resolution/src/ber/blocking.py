"""Candidate generation: multi-key inverted-index blocking within each country.

Every record emits a set of hashed keys (name tokens, phonetic tokens, compact
name, house number + street token, ...). An S1 record and an S2/S3 record become
a candidate pair when they share at least one key whose frequency is below the
caps. Pairs are scored by summed rarity of their shared keys and the top_n per
S1 are kept: that set is exactly what `candidate_pairs.tsv` reports.

To add a new blocking pass, add a branch to `_row_keys` and list its name in
`blocking.key_types` in the config.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def _row_keys(row, key_types):
    name_toks = row.name.split()
    phon_toks = row.phon.split()
    addr_toks = row.addr_alpha.split()
    keys = []
    if "name_tok" in key_types:
        keys += ["nt|" + t for t in name_toks if len(t) >= 3]
    if "name_compact" in key_types and len(row.name_compact) >= 4:
        keys.append("nc|" + row.name_compact)
    if "name_prefix" in key_types and len(row.name_compact) >= 7:
        keys.append("np|" + row.name_compact[:7])
    if "phon_tok" in key_types:
        keys += ["pt|" + t for t in phon_toks if len(t) >= 3]
    if "phon_compact" in key_types:
        pc = "".join(phon_toks)
        if len(pc) >= 4:
            keys.append("pc|" + pc)
    if "hn_addr_tok" in key_types and row.hn:
        keys += ["ha|" + row.hn + "|" + t for t in addr_toks if len(t) >= 4]
    if "addr_bigram" in key_types:
        keys += ["ab|" + a + "|" + b for a, b in zip(addr_toks, addr_toks[1:])]
    # Compound keys: pairs of individually common tokens are rare, so they survive the
    # frequency caps ("elite"+"cargo", "pnr"+"akr" for transliterated "payonir agro").
    if "name_pair" in key_types:
        keys += ["n2|" + a + "|" + b for a, b in _pairs(name_toks)]
    if "phon_pair" in key_types:
        keys += ["p2|" + a + "|" + b for a, b in _pairs(phon_toks)]
    if "name_hn" in key_types and row.hn:
        keys += ["nh|" + row.hn + "|" + t for t in name_toks if len(t) >= 3]
    # Address components get reordered ("unit 8 2739 cranberry hwy"), so the first number is
    # not always the house number: pair tokens with each of the first few numbers instead.
    nums = row.nums.split()[:3]
    if "num_addr_tok" in key_types:
        keys += ["na|" + n + "|" + t for n in nums for t in addr_toks if len(t) >= 4]
    if "num_name_tok" in key_types:
        keys += ["nn|" + n + "|" + t for n in nums for t in name_toks if len(t) >= 3]
    if "num_phon_tok" in key_types:
        keys += ["nf|" + n + "|" + t for n in nums for t in phon_toks if len(t) >= 2]
    # Name word + address word (street / locality): catches common names whose own keys are
    # capped when the address has no usable number ("mountain trust" + "kniffin").
    words = list(dict.fromkeys(t for t in addr_toks if len(t) >= 4))[:12]
    if "name_addr" in key_types:
        keys += ["nx|" + n + "|" + t for n in name_toks[:3] if len(n) >= 3 for t in words]
    if "phon_addr" in key_types:
        keys += ["px|" + p + "|" + t for p in phon_toks[:3] if len(p) >= 2 for t in words]
    # Full compact name + a number / an address word: specific even when ~40 S1s share the
    # name, since namesakes sit at different addresses.
    nc = row.name_compact
    if "cname_num" in key_types and len(nc) >= 4:
        keys += ["cn|" + nc + "|" + n for n in nums]
    if "cname_addr" in key_types and len(nc) >= 4:
        keys += ["cx|" + nc + "|" + t for t in words]
    # True matches drop or insert a digit (385 -> 85); every number also emits its one-digit
    # deletions, so the two forms share a key.
    if "num_del_addr" in key_types:
        hn = nums[:1]
        dels = {n[:i] + n[i + 1:] for n in hn if len(n) >= 2 for i in range(len(n))} | set(hn)
        keys += ["nd|" + v + "|" + t for v in dels if v for t in words[:4]]
    return keys


def _pairs(toks, max_toks=6):
    toks = sorted(set(t for t in toks[:max_toks] if len(t) >= 2))
    return [(a, b) for i, a in enumerate(toks) for b in toks[i + 1:]]


def _keys_chunk(args):
    sub, start, key_types = args
    keys, owners = [], []
    for i, row in enumerate(sub.itertuples(index=False), start=start):
        ks = set(_row_keys(row, key_types))
        keys.extend(row.country + "|" + k for k in ks)
        owners.extend([i] * len(ks))
    return pd.DataFrame({"key": pd.util.hash_array(np.array(keys, dtype=object)),
                         "idx": np.array(owners, dtype=np.int64)})


def make_keys(df, key_types, n_jobs=1, chunk=200_000):
    """Return a (key uint64, idx int64) frame with one row per distinct key per record."""
    cols = ["country", "name", "name_compact", "phon", "addr_alpha", "hn", "nums"]
    tasks = [(df[cols].iloc[s:s + chunk], s, key_types) for s in range(0, len(df), chunk)]
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            parts = list(ex.map(_keys_chunk, tasks))
    else:
        parts = [_keys_chunk(t) for t in tasks]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"key": [], "idx": []})


def generate_candidates(s1, s23, cfg, log=print):
    """Return pairs (s1_idx, r_idx, bscore) indexing rows of `s1` and `s23`."""
    bc = cfg["blocking"]
    n_jobs = cfg["features"]["n_jobs"]
    k1 = make_keys(s1, bc["key_types"], n_jobs)
    k2 = make_keys(s23, bc["key_types"], n_jobs)
    df1 = k1["key"].value_counts()
    df2 = k2["key"].value_counts()
    keep = df1.index[df1.values <= bc["max_df_s1"]].intersection(df2.index[df2.values <= bc["max_df_s23"]])
    weight = 1.0 / np.log2(2.0 + df2.reindex(keep).astype(np.float32))
    k1 = k1[k1["key"].isin(keep)]
    k2 = k2[k2["key"].isin(keep)]
    k2 = k2.assign(w=k2["key"].map(weight).astype(np.float32).values)
    log(f"  keys kept {len(keep):,} | s1 key rows {len(k1):,} | s23 key rows {len(k2):,}")

    n_r = len(s23)
    out = []
    s1_ids = k1["idx"].unique()
    for start in range(0, len(s1_ids), bc["s1_chunk"]):
        chunk_ids = s1_ids[start:start + bc["s1_chunk"]]
        m = k1[k1["idx"].isin(chunk_ids)].merge(k2, on="key", suffixes=("_s1", "_r"))
        pair = m["idx_s1"].values * n_r + m["idx_r"].values
        agg = pd.Series(m["w"].values).groupby(pair).sum()
        p = pd.DataFrame({"s1_idx": agg.index.values // n_r, "r_idx": agg.index.values % n_r,
                          "bscore": agg.values.astype(np.float32)})
        p = p.sort_values(["s1_idx", "bscore"], ascending=[True, False])
        rank = p.groupby("s1_idx").cumcount().values
        keep = rank < bc["top_n"]
        if bc.get("ties_max"):
            # keep candidates tied with the top_n-th score (e.g. ~190 records sharing only a
            # common name), up to ties_max per S1, so the ranker - not the cut - chooses
            nth = p["bscore"].where(rank == bc["top_n"] - 1).groupby(p["s1_idx"].values).transform("max").values
            keep |= (p["bscore"].values >= np.nan_to_num(nth, nan=np.inf) - 1e-6) & (rank < bc["ties_max"])
        p = p[keep]
        out.append(p)
        log(f"  s1 {start + len(chunk_ids):,}/{len(s1_ids):,}: raw pairs {len(agg):,} -> kept {len(p):,}")
    if not out:
        return pd.DataFrame({"s1_idx": [], "r_idx": [], "bscore": []})
    return pd.concat(out, ignore_index=True)
