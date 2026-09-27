"""Stage orchestration. Each stage reads the previous stage's files from the run
directory, so any stage can be re-run alone after a change:

  prepare  -> work/cache/<data>/{split}_s1.parquet, {split}_s23.parquet, train_gt.parquet
  block    -> work/runs/<run>/{split}_pairs.parquet            (+ blocking recall on train)
  features -> work/runs/<run>/{split}_features/part-*.parquet
  train    -> models/, oof.parquet, selection.json, metrics.json
  predict  -> output/matching_results.tsv, output/candidate_pairs.tsv (+ validator, experiment log)
"""
import copy
import csv
import datetime as dt
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from . import blocking, candidates, features, metrics, model, postprocess, stage2
from .io import read_ground_truth, read_source, write_id_lists
from .normalize import apply_addr_map, apply_translit, learn_addr_map, learn_translit, prepare as prepare_frame
from .smoke import make_smoke

SPLITS = ("train", "test")


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg["paths"]
        self.data_dir = p["data_dir"]
        self.cache_dir = os.path.join(p["work_dir"], "cache", os.path.basename(self.data_dir.rstrip("/"))
                                      + cfg["data"].get("cache_tag", ""))
        self.run_dir = os.path.join(p["work_dir"], "runs", cfg["run_name"])
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._log_file = open(os.path.join(self.run_dir, "log.txt"), "a")

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self._log_file.write(line + "\n")
        self._log_file.flush()

    def cache(self, name):
        return os.path.join(self.cache_dir, name)

    def run_file(self, name):
        return os.path.join(self.run_dir, name)

    def load_prepared(self, split):
        return pd.read_parquet(self.cache(f"{split}_s1.parquet")), pd.read_parquet(self.cache(f"{split}_s23.parquet"))

    def _prepare_parallel(self, df, source):
        n_jobs = self.cfg["features"]["n_jobs"]
        if n_jobs <= 1 or len(df) < 100_000:
            return prepare_frame(df, source)
        chunks = np.array_split(np.arange(len(df)), n_jobs * 4)
        with ProcessPoolExecutor(n_jobs) as ex:
            parts = list(ex.map(prepare_frame, [df.iloc[c] for c in chunks], [source] * len(chunks)))
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------ stages
    def make_smoke(self):
        make_smoke(self.cfg, self.log)

    def s1_subset(self, split, s1):
        """Positions of the S1 rows used in this run (train may be hash-sampled; S2/S3 always full)."""
        frac = self.cfg["data"]["train_s1_frac"] if split == "train" else 1.0
        if frac >= 1.0:
            return np.arange(len(s1))
        h = pd.util.hash_array(s1["rid"].values.astype(object)) % 10_000
        return np.flatnonzero(h < frac * 10_000)

    def train_eval_view(self, s1, gt, *frames):
        """Restrict metric inputs to the sampled train S1s and renumber them 0..n-1."""
        sub = self.s1_subset("train", s1)
        pos = np.full(len(s1), -1)
        pos[sub] = np.arange(len(sub))
        remap = lambda f: f[pos[f["s1_idx"].values] >= 0].assign(s1_idx=lambda x: pos[x["s1_idx"].values])
        return s1["country"].values[sub], remap(gt), [remap(f) for f in frames]

    def prepare(self):
        for split in SPLITS:
            src = lambda s: os.path.join(self.data_dir, split, f"{split}_{s}.tsv")
            if os.path.exists(self.cache(f"{split}_s23.parquet")):
                self.log(f"prepare {split}: cached, skipping (delete {self.cache_dir} to rebuild)")
                continue
            s1 = read_source(src("source1"))
            s1p = self._prepare_parallel(s1, 1)
            s23p = pd.concat([self._prepare_parallel(read_source(src("source2")), 2),
                              self._prepare_parallel(read_source(src("source3")), 3)], ignore_index=True)
            if split == "train":
                gt, _ = read_ground_truth(src("ground_truth"))
                s1_pos = pd.Series(np.arange(len(s1p)), index=s1p["rid"])
                r_pos = pd.Series(np.arange(len(s23p)), index=s23p["rid"])
                gt_idx = pd.DataFrame({"s1_idx": s1_pos.loc[gt["s1_id"]].values,
                                       "r_idx": r_pos.loc[gt["rid"]].values})
                gt_idx.to_parquet(self.cache("train_gt.parquet"))
                self.log(f"  train ground-truth links: {len(gt_idx):,}")
                if self.cfg["data"].get("learned_translit"):
                    ind = gt_idx[s23p["name_indic"].values[gt_idx["r_idx"].values]]
                    translit = learn_translit(s1p["name_raw"].values[ind["s1_idx"].values],
                                              s23p["name_raw"].values[ind["r_idx"].values])
                    with open(self.cache("translit.json"), "w") as f:
                        json.dump(translit, f, ensure_ascii=False, indent=0)
                    self.log(f"  learned transliteration map: {len(translit):,} words from {len(ind):,} Indic-name links")
                if self.cfg["data"].get("learned_addr_map"):
                    amap = learn_addr_map(s1p["addr"].values[gt_idx["s1_idx"].values],
                                          s23p["addr"].values[gt_idx["r_idx"].values])
                    with open(self.cache("addr_map.json"), "w") as f:
                        json.dump(amap, f, ensure_ascii=False, indent=0)
                    self.log(f"  learned address word map: {len(amap):,} words")
            if self.cfg["data"].get("learned_translit"):
                with open(self.cache("translit.json")) as f:
                    s23p = apply_translit(s23p, json.load(f))
            if self.cfg["data"].get("learned_addr_map"):
                with open(self.cache("addr_map.json")) as f:
                    s23p = apply_addr_map(s23p, json.load(f), s1p)
            s1p.to_parquet(self.cache(f"{split}_s1.parquet"))
            s23p.to_parquet(self.cache(f"{split}_s23.parquet"))
            self.log(f"prepare {split}: S1 {len(s1p):,}  S2+S3 {len(s23p):,}")

    def block(self):
        for split in SPLITS:
            s1, s23 = self.load_prepared(split)
            self.log(f"block {split}")
            sub = self.s1_subset(split, s1)
            pairs = blocking.generate_candidates(s1.iloc[sub], s23, self.cfg, self.log)
            extra = self.cfg["blocking"].get("extra_pass")
            if extra:
                # specific key types (full compact name, word pairs, number + word ...) under looser
                # caps: the most specific key of most missed links is shared by 30-100 S1s
                cfg2 = copy.deepcopy(self.cfg)
                cfg2["blocking"].update(extra)
                p2 = blocking.generate_candidates(s1.iloc[sub], s23, cfg2, self.log)
                pairs = (pd.concat([pairs, p2], ignore_index=True)
                         .groupby(["s1_idx", "r_idx"], as_index=False)["bscore"].sum())
                self.log(f"  extra pass: {len(p2):,} pairs; union {len(pairs):,}")
            pairs["s1_idx"] = sub[pairs["s1_idx"].values]
            pairs["source"] = s23["source"].values[pairs["r_idx"].values].astype(np.int8)
            gt = pd.read_parquet(self.cache("train_gt.parquet")) if split == "train" else None
            if self.cfg.get("candidates", {}).get("enabled"):
                if split == "train":
                    self._log_recall(s1, gt, pairs, prefix="loose_")
                pairs = candidates.prune(pairs, s1, s23, split, self.cfg, self.run_file("ranker"), gt, self.log)
                self.log(f"  pruned to {len(pairs):,} pairs ({len(pairs) / len(sub):.1f} per S1)")
            if split == "train":
                self._log_recall(s1, gt, pairs)
            pairs.to_parquet(self.run_file(f"{split}_pairs.parquet"))

    def _log_recall(self, s1, gt, pairs, prefix=""):
        country, g, (pairs_v,) = self.train_eval_view(s1, gt, pairs)
        rec = {prefix + k: v for k, v in metrics.blocking_recall(pairs_v, g, country).items()}
        self.log(f"  {json.dumps({k: round(v, 4) for k, v in rec.items()})}")
        self._update_metrics(rec)

    def features(self):
        """Writes {split}_features/part-*.parquet; parts split on S1 boundaries to bound memory."""
        part_rows = self.cfg["features"]["part_rows"]
        for split in SPLITS:
            s1, s23 = self.load_prepared(split)
            pairs = pd.read_parquet(self.run_file(f"{split}_pairs.parquet")).sort_values(["s1_idx", "r_idx"])
            pairs = pd.concat([pairs.reset_index(drop=True), features.context_features(pairs).reset_index(drop=True)],
                              axis=1)
            self.log(f"features {split}: {len(pairs):,} pairs")
            idf = features.build_idf(s1, s23)
            if split == "train":
                gt = pd.read_parquet(self.cache("train_gt.parquet"))
                key = pairs["s1_idx"].values * len(s23) + pairs["r_idx"].values
                pairs["label"] = np.isin(key, gt["s1_idx"].values * len(s23) + gt["r_idx"].values).astype(np.int8)
                h = pd.util.hash_array(s1["rid"].values.astype(object)) % self.cfg["model"]["n_folds"]
                pairs["fold"] = h[pairs["s1_idx"].values].astype(np.int8)
            out_dir = self.run_file(f"{split}_features")
            shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(out_dir)
            # part boundaries fall where s1_idx changes, so an S1's candidates never straddle parts
            starts = np.flatnonzero(np.r_[True, np.diff(pairs["s1_idx"].values) != 0])
            bounds = starts[np.searchsorted(starts, np.arange(0, len(pairs), part_rows))].tolist() + [len(pairs)]
            bounds = sorted(set(bounds))
            for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
                feats = features.compute_features(pairs.iloc[a:b], s1, s23, idf, self.cfg)
                feats.to_parquet(os.path.join(out_dir, f"part-{i:04d}.parquet"))
                self.log(f"  part {i}: {b - a:,} pairs")

    def feature_parts(self, split):
        d = self.run_file(f"{split}_features")
        return [os.path.join(d, f) for f in sorted(os.listdir(d))]

    def train(self):
        feats = pd.concat([pd.read_parquet(f) for f in self.feature_parts("train")], ignore_index=True)
        s1, s23 = self.load_prepared("train")
        gt = pd.read_parquet(self.cache("train_gt.parquet"))
        self.log(f"train: {len(feats):,} pairs, positive rate {feats['label'].mean():.4f}")
        oof, imp = model.train_cv(feats, self.cfg, self.run_file("models"), self.log)
        imp.to_csv(self.run_file("feature_importance.csv"), header=["gain"])
        if self.cfg["model"].get("stage2"):
            self._update_metrics({"stage1_" + k: v for k, v in self._oof_report(feats, oof, s1, gt).items()})
            feats = pd.concat([feats, stage2.group_features(feats.assign(p=oof), s23)], axis=1)
            # group features whose values depend on how many decoys an S1 faces (the test has ~2x more)
            drop = [c for c in feats.columns if any(fnmatch.fnmatch(c, pat) for pat in self.cfg["model"].get("stage2_exclude", []))]
            if drop:
                feats = feats.drop(columns=drop)
                self.log(f"  stage 2: excluded {len(drop)} features: {drop}")
            self.log("  stage 2: training on first-stage out-of-fold scores + competition features")
            oof, imp = model.train_cv(feats, self.cfg, self.run_file("models_s2"), self.log)
            imp.to_csv(self.run_file("feature_importance_s2.csv"), header=["gain"])
        d = feats[["s1_idx", "r_idx", "source"]].assign(p=oof)
        d.to_parquet(self.run_file("oof.parquet"))
        country, gt, (assigned, d) = self.train_eval_view(s1, gt, postprocess.assign(d, self.cfg), d)
        pp = self.cfg["postprocess"]
        params = (postprocess.tune(assigned, gt, country, self.cfg, self.log) if pp["mode"] == "tune"
                  else {k: pp[k] for k in ("mode", "threshold", "single_threshold")})
        with open(self.run_file("selection.json"), "w") as f:
            json.dump(params, f)
        mix = self.cfg["eval"].get("test_country_mix")
        res = metrics.report(postprocess.select(assigned, params), gt, country, mix)
        no_assign = metrics.report(d[d["p"] >= 0.5], gt, country, mix)
        res["oof_f05_p05_no_postprocess"] = no_assign["macro_f05"]
        self.log(f"  OOF: {json.dumps({k: round(v, 4) for k, v in res.items()})}")
        self._update_metrics(res)

    def _oof_report(self, feats, oof, s1, gt):
        d = feats[["s1_idx", "r_idx", "source"]].assign(p=oof)
        country, g, (assigned,) = self.train_eval_view(s1, gt, postprocess.assign(d, self.cfg))
        params = postprocess.tune(assigned, g, country, self.cfg, self.log)
        r = metrics.report(postprocess.select(assigned, params), g, country, self.cfg["eval"].get("test_country_mix"))
        self.log(f"  stage 1 OOF: {json.dumps({k: round(v, 4) for k, v in r.items()})}")
        return r

    def predict(self):
        s1, s23 = self.load_prepared("test")
        with open(self.run_file("selection.json")) as f:
            params = json.load(f)
        n_folds = self.cfg["model"]["n_folds"]
        d = []
        for path in self.feature_parts("test"):
            feats = pd.read_parquet(path)
            keep = ["s1_idx", "r_idx", "source"] + [c for c in stage2.GROUP_SOURCES if c in feats.columns]
            d.append(feats[keep].assign(p=model.predict(feats, self.run_file("models"), n_folds)))
        d = pd.concat(d, ignore_index=True)
        if self.cfg["model"].get("stage2"):
            # group features need every part at once (a record's competing S1s span parts)
            g = stage2.group_features(d, s23)
            p2, start = np.empty(len(d), dtype=np.float32), 0
            for path in self.feature_parts("test"):
                feats = pd.read_parquet(path)
                end = start + len(feats)
                feats = pd.concat([feats, g.iloc[start:end].reset_index(drop=True)], axis=1)
                p2[start:end] = model.predict(feats, self.run_file("models_s2"), n_folds)
                start = end
            d["p"] = p2
            self.log("  stage 2 applied to test")
        d = d[["s1_idx", "r_idx", "source", "p"]]
        d.to_parquet(self.run_file("test_preds.parquet"))
        chosen = postprocess.select(postprocess.assign(d, self.cfg), params)
        ids = lambda frame: pd.DataFrame({"s1_id": s1["rid"].values[frame["s1_idx"].values],
                                          "rid": s23["rid"].values[frame["r_idx"].values]})
        out_dir = self.cfg["paths"]["output_dir"]
        write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), s1["rid"].tolist(), ids(d),
                       "candidate_entity_ids")
        write_id_lists(os.path.join(out_dir, "matching_results.tsv"), s1["rid"].tolist(), ids(chosen),
                       "matched_entity_ids")
        stats = {"test_matches": len(chosen),
                 "test_empty_share": 1 - chosen["s1_idx"].nunique() / len(s1),
                 **{f"test_matches_per_s1_{c}": float((s1["country"].values[chosen["s1_idx"].values] == c).sum()
                                                      / max((s1["country"] == c).sum(), 1))
                    for c in s1["country"].unique()}}
        self.log(f"predict: {json.dumps({k: round(v, 4) for k, v in stats.items()})}")
        self._update_metrics(stats)
        self._validate(out_dir)
        # keep a per-run copy so later runs never overwrite a submitted file
        keep = self.run_file("submission")
        os.makedirs(keep, exist_ok=True)
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy(os.path.join(out_dir, name), keep)
        self._log_experiment()

    # ------------------------------------------------------------ bookkeeping
    def _update_metrics(self, new):
        path = self.run_file("metrics.json")
        cur = json.load(open(path)) if os.path.exists(path) else {}
        cur.update(new)
        with open(path, "w") as f:
            json.dump(cur, f, indent=2)

    def _validate(self, out_dir):
        cmd = [sys.executable, self.cfg["paths"]["validator"],
               "--matching", os.path.join(out_dir, "matching_results.tsv"),
               "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"),
               "--test-dir", os.path.join(self.data_dir, "test")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.log("validator:\n" + r.stdout.strip())

    def _log_experiment(self):
        """One row per prediction run; fill `lb_score` by hand after uploading."""
        path = os.path.join(self.cfg["paths"]["work_dir"], "experiments.csv")
        m = json.load(open(self.run_file("metrics.json")))
        row = {"time": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "run": self.cfg["run_name"],
               "block_recall": m.get("block_recall"), "oof_f05": m.get("macro_f05"),
               "oof_f05_test_mix": m.get("f05_test_mix"), "test_matches": m.get("test_matches"),
               "lb_score": "", "notes": ""}
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
