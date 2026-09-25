# dataset/

The competition data is **not in the repo**: it is about 2.3 GB, and it's the organisers' data, not ours to redistribute.
Download it from the challenge page on Unstop and put the files here, exactly as below:

```
dataset/
├── train/
│   ├── train_source1.tsv        # 2,206,821 rows  (S1 reference records)
│   ├── train_source2.tsv        # 5,034,616 rows
│   ├── train_source3.tsv        # 5,285,603 rows
│   └── train_ground_truth.tsv   # 2,206,821 rows  (S1 id -> matched S2/S3 ids)
└── test/
    ├── test_source1.tsv         # 1,732,544 rows
    ├── test_source2.tsv         # 4,887,273 rows
    └── test_source3.tsv         # 5,082,316 rows
```

Row counts exclude the header. Check them with `wc -l dataset/*/*.tsv`: the total should be 26,436,001 lines, including the 7 header lines.

Every file is **tab-separated** with the columns `entity_id, business_name, business_address, country`. The ground-truth file has `source1_entity_id, matched_entity_ids`, and the second column is a comma-separated list.

Git ignores everything in this folder except this README and the empty `train/` and `test/` placeholders.
