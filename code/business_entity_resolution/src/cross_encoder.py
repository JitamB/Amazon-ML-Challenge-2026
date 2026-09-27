"""Fine-tune a multilingual cross-encoder on the train candidate pairs; score train (out-of-fold) and test.

    python code/business_entity_resolution/src/cross_encoder.py --config code/business_entity_resolution/configs/v14.yaml

Run after the `block` stage (it reads work/runs/<run>/{train,test}_pairs.parquet) and before
`features`. With `ce` listed under `features.external`, the score becomes a pair feature
(`ext_ce`) for both model stages, and its record-level competition (`ext_ce_maxother_r`,
`ext_ce_delta_r`) enters stage 2.

Why: the GBDT sees only string-similarity numbers. A transformer reads both records whole, so
it can learn what the numbers can't express: word order, abbreviations, transliteration and,
above all, French, which has no training data but which a pretrained multilingual model
already understands. Out-of-fold: for each of the pipeline's S1 folds, a model is trained on
the other folds' pairs and scores this fold's pairs; test scores are the average of the fold
models (or of the first `test_folds` of them, to save time).

Writes work/runs/<run>/ce_{train,test}.parquet with columns s1_idx, r_idx, score (a logit).
Settings (config `cross_encoder`): model, max_len, batch_size, lr, epochs, max_train_pairs,
test_folds. The default model, intfloat/multilingual-e5-small (MIT, 118M parameters), scores
about 5-8k pairs/s on an A10G in bf16. The full run (3 folds, 2M training pairs each, ~11M
train + ~13M test pairs) takes about 3 hours.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ber.config import load_config  # noqa: E402
from ber.pipeline import Pipeline  # noqa: E402

DEFAULTS = {"model": "intfloat/multilingual-e5-small", "max_len": 64, "batch_size": 256, "lr": 3e-5,
            "epochs": 1, "max_train_pairs": 2_000_000, "test_folds": 3}


def texts(frame):
    """One string per record: name, then address (both normalised to Latin script)."""
    return (frame["name_raw"].fillna("") + " ; " + frame["addr"].fillna("")).values


class Scorer:
    def __init__(self, cc, device):
        self.cc, self.device = cc, device
        self.tok = AutoTokenizer.from_pretrained(cc["model"])
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def encode(self, a, b):
        enc = self.tok(list(a), list(b), truncation=True, max_length=self.cc["max_len"], padding=True,
                       return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def train(self, a, b, y, log):
        model = AutoModelForSequenceClassification.from_pretrained(self.cc["model"], num_labels=1).to(self.device)
        bs, n = self.cc["batch_size"], len(y)
        steps = self.cc["epochs"] * math.ceil(n / bs)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cc["lr"], weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(0)
        model.train()
        step, t0 = 0, time.time()
        for _ in range(self.cc["epochs"]):
            order = rng.permutation(n)
            for i in range(0, n, bs):
                j = order[i:i + bs]
                with torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda"):
                    logits = model(**self.encode(a[j], b[j])).logits.squeeze(-1)
                loss = loss_fn(logits.float(), torch.tensor(y[j], dtype=torch.float32, device=self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 1000 == 0:
                    log(f"    step {step:,}/{steps:,}  loss {loss.item():.4f}  {step * bs / (time.time() - t0):,.0f} pairs/s")
        model.eval()
        return model

    @torch.no_grad()
    def score(self, model, a, b):
        bs = self.cc["batch_size"] * 4
        # sort by length so each batch pads little
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
        out = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), bs):
            j = order[i:i + bs]
            with torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda"):
                out[j] = model(**self.encode(a[j], b[j])).logits.squeeze(-1).float().cpu().numpy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pl = Pipeline(cfg)
    cc = {**DEFAULTS, **cfg.get("cross_encoder", {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.log(f"cross-encoder {cc['model']} on {device}")
    scorer = Scorer(cc, device)
    n_folds = cfg["model"]["n_folds"]

    s1, s23 = pl.load_prepared("train")
    gt = pd.read_parquet(pl.cache("train_gt.parquet"))
    pairs = pd.read_parquet(pl.run_file("train_pairs.parquet"), columns=["s1_idx", "r_idx"])
    n_r = len(s23)
    y = np.isin(pairs["s1_idx"].values.astype(np.int64) * n_r + pairs["r_idx"].values,
                gt["s1_idx"].values.astype(np.int64) * n_r + gt["r_idx"].values).astype(np.float32)
    # the pipeline's folds (pipeline.features), so the score is out-of-fold for both model stages
    fold = (pd.util.hash_array(s1["rid"].values.astype(object)) % n_folds)[pairs["s1_idx"].values]
    ta, tb = texts(s1)[pairs["s1_idx"].values], texts(s23)[pairs["r_idx"].values]

    s1t, s23t = pl.load_prepared("test")
    tpairs = pd.read_parquet(pl.run_file("test_pairs.parquet"), columns=["s1_idx", "r_idx"])
    xa, xb = texts(s1t)[tpairs["s1_idx"].values], texts(s23t)[tpairs["r_idx"].values]

    oof = np.zeros(len(pairs), dtype=np.float32)
    test = np.zeros(len(tpairs), dtype=np.float32)
    n_test = min(cc["test_folds"], n_folds)
    rng = np.random.default_rng(0)
    for k in range(n_folds):
        tr = np.flatnonzero(fold != k)
        if len(tr) > cc["max_train_pairs"]:
            tr = rng.choice(tr, cc["max_train_pairs"], replace=False)
        pl.log(f"  fold {k}: training on {len(tr):,} pairs (positive rate {y[tr].mean():.3f})")
        model = scorer.train(ta[tr], tb[tr], y[tr], pl.log)
        va = np.flatnonzero(fold == k)
        oof[va] = scorer.score(model, ta[va], tb[va])
        auc_pos, auc_neg = oof[va][y[va] == 1], oof[va][y[va] == 0]
        pl.log(f"  fold {k}: scored {len(va):,} train pairs; mean logit true {auc_pos.mean():.2f} / other {auc_neg.mean():.2f}")
        if k < n_test:
            test += scorer.score(model, xa, xb) / n_test
            pl.log(f"  fold {k}: scored {len(tpairs):,} test pairs")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    pairs.assign(score=oof).to_parquet(pl.run_file("ce_train.parquet"))
    tpairs.assign(score=test).to_parquet(pl.run_file("ce_test.parquet"))
    pl.log("wrote ce_train.parquet, ce_test.parquet")


if __name__ == "__main__":
    main()
