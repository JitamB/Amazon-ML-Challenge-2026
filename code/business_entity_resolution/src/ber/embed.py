"""Embedding blocking pass: nearest neighbours in a multilingual sentence-embedding space.

Key-based blocking needs a shared word, number or phonetic code. What it still misses
(1.4% of links after v13) is mostly records with nothing literal in common: random alias
names at a reworded address, partial names, translated words. An embedding of
"name ; address" puts such records near their S1 anyway. For each S1, the k nearest S2/S3
records of the same country (cosine, FAISS inverted-file index) become extra loose candidates;
the candidate ranker then decides which of them to keep.

Config `blocking.embedding_pass`: {model, k, batch_size, nlist, nprobe, weight}. The pair score
added to `bscore` is weight x cosine. Embeddings are cached per split in the data cache.
Needs torch, transformers and faiss-cpu (MIT).
"""
import os

import numpy as np
import pandas as pd

DEFAULTS = {"model": "intfloat/multilingual-e5-small", "k": 10, "batch_size": 1024, "max_len": 48,
            "nlist": 4096, "nprobe": 16, "weight": 1.0}


def encode(texts, ec, log=print):
    """L2-normalised mean-pooled embeddings (float32), GPU when available."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ec["model"])
    model = AutoModel.from_pretrained(ec["model"]).to(device).eval()
    order = np.argsort([len(t) for t in texts])  # little padding per batch
    out = None
    with torch.no_grad():
        for i in range(0, len(texts), ec["batch_size"]):
            j = order[i:i + ec["batch_size"]]
            enc = tok(["query: " + texts[x] for x in j], truncation=True, max_length=ec["max_len"],
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().cpu().numpy()
            if out is None:
                out = np.empty((len(texts), e.shape[1]), dtype=np.float32)
            out[j] = e
            if (i // ec["batch_size"]) % 1000 == 0:
                log(f"    embedded {i + len(j):,}/{len(texts):,}")
    return out


def cached_embeddings(frame, path, ec, log):
    if os.path.exists(path):
        return np.load(path)
    texts = (frame["name_raw"].fillna("") + " ; " + frame["addr"].fillna("")).tolist()
    e = encode(texts, ec, log)
    np.save(path, e)
    return e


def knn_pairs(s1, s23, e1, e2, ec, log=print):
    """Pairs (s1_idx, r_idx, bscore) linking each S1 to its k nearest same-country records.

    s1_idx indexes `s1` / `e1` (the rows passed in), r_idx indexes `s23` / `e2`.
    """
    import faiss

    out = []
    for c in pd.unique(s1["country"].values):
        qi = np.flatnonzero(s1["country"].values == c)
        di = np.flatnonzero(s23["country"].values == c)
        if len(qi) == 0 or len(di) == 0:
            continue
        d = e2.shape[1]
        nlist = min(ec["nlist"], max(1, len(di) // 40))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        sample = di if len(di) <= 200_000 else np.random.default_rng(0).choice(di, 200_000, replace=False)
        index.train(e2[sample])
        index.add(e2[di])
        index.nprobe = ec["nprobe"]
        sim, nn = index.search(e1[qi], ec["k"])
        ok = nn >= 0
        out.append(pd.DataFrame({"s1_idx": np.repeat(qi, ec["k"])[ok.ravel()],
                                 "r_idx": di[nn[ok]],
                                 "bscore": (ec["weight"] * sim[ok]).astype(np.float32)}))
        log(f"  embedding pass {c}: {len(qi):,} S1 x k={ec['k']} over {len(di):,} records")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame({"s1_idx": [], "r_idx": [], "bscore": []})
