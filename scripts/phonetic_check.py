"""Would keeping vowels in short tokens' phonetic codes help? (a proposed fix, measured on train)

    python scripts/phonetic_check.py <run>

Old: every token loses its vowels after the first letter. New: tokens shorter than 7 keep
them. Measured on true links (does the S1 and its record still share a phonetic word?) and
on the phonetic blocking keys (how many are shared by more than 30 S1s and so get capped).
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import load_config  # noqa: E402
from ber.normalize import LEGAL, _PH_MAP  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402


def old(tok):
    first = tok[0]
    body = re.sub(r"[aeiouyh]", "", tok[1:].translate(_PH_MAP))
    head = first.translate(_PH_MAP) if first not in "aeiouyh" else first
    return re.sub(r"(.)\1+", r"\1", head + body)


def new(tok, short_len=7):
    first = tok[0]
    head = first.translate(_PH_MAP) if first not in "aeiouyh" else first
    body = tok[1:].translate(_PH_MAP)
    body = body.replace("h", "") if len(tok) < short_len else re.sub(r"[aeiouyh]", "", body)
    return re.sub(r"(.)\1+", r"\1", head + body)


if __name__ == "__main__":
    pl = Pipeline(load_config(f"work/runs/{sys.argv[1]}/config.yaml"))
    s1 = pd.read_parquet(pl.cache("train_s1.parquet"), columns=["country", "name"])
    s23 = pd.read_parquet(pl.cache("train_s23.parquet"), columns=["name", "name_indic"])
    gt = pd.read_parquet(pl.cache("train_gt.parquet")).sample(300_000, random_state=0)

    for label, fn in (("old", old), ("new", new)):
        legal = {fn(t) for t in LEGAL | {"private", "limited", "praivet"}}
        codes = lambda name: {fn(t) for t in name.split() if t} - legal - {""}
        a = [codes(x) for x in s1.name.values[gt.s1_idx.values]]
        b = [codes(x) for x in s23.name.values[gt.r_idx.values]]
        share = np.array([bool(x & y) for x, y in zip(a, b)])
        jac = np.array([len(x & y) / max(len(x | y), 1) for x, y in zip(a, b)])
        ind = s23.name_indic.values[gt.r_idx.values]
        # blocking view: phonetic-word keys per country, capped when shared by > 30 S1s
        df = pd.Series([c + "|" + k for c, n in zip(s1.country.values, s1.name.values) for k in codes(n)]).value_counts()
        capped = (df > 30).sum() / len(df)
        rows_capped = df[df > 30].sum() / df.sum()
        print(f"{label}: true links sharing a phonetic word {share.mean():.4f} (Indic-name records {share[ind].mean():.4f}); "
              f"mean Jaccard {jac.mean():.4f}; distinct codes {len(df):,}; codes over the cap {capped:.4f}, "
              f"S1 key rows over the cap {rows_capped:.4f}", flush=True)
    for w in ["patel", "patil", "kumar", "kumari", "shri", "shree", "sri", "raj", "raaj", "bharat", "gold", "cold",
              "sai", "sayi", "kamal", "komal", "lata", "latha"]:
        print(f"  {w:8s} old {old(w):6s} new {new(w)}")
