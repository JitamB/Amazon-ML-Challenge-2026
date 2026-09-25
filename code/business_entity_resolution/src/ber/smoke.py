"""Build a tiny region-based copy of the dataset for end-to-end smoke tests.

Records are kept when their address contains one of the configured region
strings, so blocks keep realistic local competition. Train ground truth is
restricted to kept S1s; their true matches are always kept. Files are streamed,
never loaded whole.
"""
import os


def _filter(src, dst, keep_line):
    kept = set()
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        fout.write(fin.readline())
        for line in fin:
            if keep_line(line):
                fout.write(line)
                kept.add(line.split("\t", 1)[0])
    return kept


def make_smoke(cfg, log=print):
    src, dst = cfg["smoke"]["source_data_dir"], cfg["paths"]["data_dir"]
    regions = cfg["smoke"]["regions"]
    def in_region(line):
        parts = line.split("\t")
        return len(parts) > 2 and any(r in parts[2] for r in regions)

    for split in ("train", "test"):
        os.makedirs(os.path.join(dst, split), exist_ok=True)
        p = lambda s: (os.path.join(src, split, f"{split}_{s}.tsv"), os.path.join(dst, split, f"{split}_{s}.tsv"))
        s1 = _filter(*p("source1"), in_region)
        matched = set()
        if split == "train":
            gt_src, gt_dst = p("ground_truth")
            with open(gt_src, encoding="utf-8") as fin, open(gt_dst, "w", encoding="utf-8") as fout:
                fout.write(fin.readline())
                for line in fin:
                    s1_id, ids = line.rstrip("\n").split("\t")
                    if s1_id in s1:
                        fout.write(line)
                        matched.update(x for x in ids.split(",") if x)
        n23 = [len(_filter(*p(s), lambda l: in_region(l) or l.split("\t", 1)[0] in matched))
               for s in ("source2", "source3")]
        log(f"  smoke {split}: S1 {len(s1):,}  S2 {n23[0]:,}  S3 {n23[1]:,}")
