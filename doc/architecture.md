# Pipeline Architecture: Business Entity Resolution

*Last updated: 2026-09-25, during the first full run (v1).*
*Why the approach looks like this: `doc/analysis.md`. How to run it: `code/business_entity_resolution/README.md`.*

---

## 1. What the system does

For every Source 1 (S1) business, return the Source 2 / Source 3 records that are the same business.

- **Score:** F0.5 per S1, averaged over all S1s. A wrong match costs about twice as much as a missed one, and an S1 with no true matches (a *singleton*) scores 1.0 only if we return an empty list.
- **Approach:** a classic entity-resolution pipeline. Cheap candidate generation first, then an expensive pair classifier, then decisions per S1 that use the known structure of the data.

```
 TSV sources ──► 1. PREPARE ──► 2. BLOCK ──► 3. FEATURES ──► 4. TRAIN ──► 5. PREDICT ──► matching_results.tsv
 (S1, S2, S3)    clean text      candidate     ~45 numbers     LightGBM      pick match         candidate_pairs.tsv
                 transliterate   pairs         per pair        (3 folds)     sets per S1
                 extract nums    (≤80 / S1)                    + tune rules
```

Each stage reads the previous stage's files from disk. Any stage can be re-run on its own after a change, without repeating the earlier ones.

---

## 2. Code layout

```
code/business_entity_resolution/
├── configs/
│   ├── base.yaml        full-data settings (every tunable number lives here)
│   ├── smoke.yaml       tiny regional dataset for bug-hunting; inherits base.yaml
│   └── v2.yaml          next experiment; inherits base.yaml, overrides a few keys
└── src/
    ├── run.py           command-line entry point: --config, --stage, --set key=value
    └── ber/
        ├── config.py      YAML loading, `base:` inheritance, command-line overrides
        ├── io.py          read TSVs, write submission files
        ├── normalize.py   stage 1: text cleaning, transliteration, numbers, phonetics
        ├── blocking.py    stage 2: candidate generation
        ├── features.py    stage 3: pair features
        ├── model.py       stage 4: LightGBM with S1-grouped folds
        ├── postprocess.py stage 5: best-S1 assignment, caps, match-set selection
        ├── metrics.py     exact competition metric, blocking recall
        ├── pipeline.py    runs the stages, caching, validator, experiment log
        └── smoke.py       builds the small smoke dataset
scripts/
├── remote.sh            sync to and run on the SageMaker space from the Mac
└── diag_blocking.py     explains why true matches miss blocking
```

**Design rules:**
- Every tunable number is in the YAML config.
- Each experiment is a new config file or a `--set` override with a new `run_name`, so runs never overwrite each other.
- To add a blocking key or a feature you change one function; nothing else needs touching.

---

## 3. The stages

### Stage 1: Prepare (`normalize.py`)

Turns raw names and addresses into forms that can be compared across sources, scripts and countries. **Nothing depends on the country label.** France has no training data, so every rule must also work for an unseen country.

| Output column | How it is built | Why |
|---|---|---|
| `name` | Transliterate to Latin → lowercase → strip website parts (`www.`, `.com`, `#`) → fix typo swaps inside words (`h0ldings`→`holdings`, `8ittinger`→`bittinger`) → drop legal words and filler (`pvt`, `ltd`, `llc`, `sarl`, `shri`, `services`, `the` …) | Removes the noise the data generator adds |
| `name_compact` | Name words joined with no spaces | Matches website/hashtag names: `sweetpilates.com` ↔ `Sweet Pilates` |
| `phon` | Sound-alike skeleton of each word: drop vowels and `h`, merge similar consonants (b/p, d/t, g/k, j/z/s, v/w …) | Links transliterations and typos: Tamil/Kannada/Hindi "business" and English "business" all become `psns` |
| `addr` | Transliterate → lowercase → expand abbreviations (US `st`/`rd`/`ave`…, French `r`/`av`/`bd`/`ch`, `zi`/`cc`/`ld`) → drop `null`/`n/a` | Address abbreviations and reordering |
| `addr_alpha` | Address words of 3+ letters | Blocking keys |
| `nums`, `hn` | Distinct numbers in the address (leading zeros removed), and the first number | Catching sibling decoys that differ only in unit number |
| flags | `name_indic`, `name_web`, `addr_empty` | Let the model treat these hard cases differently |

**Transliteration:**
- Covers all 9 Indic scripts found in the data: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam. It uses `indic_transliteration` (MIT license).
- Fixes for Malayalam special letters, zero-width characters and curly apostrophes.
- All other characters go through `unidecode`.

**Output:** `work/cache/dataset/{train,test}_{s1,s23}.parquet` and `train_gt.parquet`. This stage runs once and is reused by every experiment. Delete the cache after changing `normalize.py`.

### Stage 2: Block (`blocking.py`)

We can't compare 1.7M × 10M record pairs. Blocking finds a small candidate set per S1 that still contains nearly all true matches.

**How it works:**
1. Every record emits a set of **keys**. Every key is prefixed with the country, so records are only ever compared within the same country. In training data, no match ever crosses countries.
2. **Keys that are too common are dropped:** more than 30 S1 records, or more than 400 S2/S3 records, per country.
3. An S1 and an S2/S3 record become a candidate pair if they share any surviving key.
4. **Pair score** = sum over the shared keys of `1 / log2(2 + key frequency)`, so rare shared keys count more.
5. **Keep the top 80 candidates per S1.** This set is `candidate_pairs.tsv`.

**Key types** (configurable via `blocking.key_types`):

| Key | Example | Catches |
|---|---|---|
| `name_tok` | `nt|balaji` | Plain name overlap |
| `name_compact`, `name_prefix` | `nc|sweetpilates`, `np|federal` | Websites, hashtags, spacing differences |
| `phon_tok`, `phon_compact` | `pt|psns` | Transliterations, typos |
| `hn_addr_tok` | `ha|2739|cranberry` | Alias names at the same address |
| `addr_bigram` | `ab|cranberry|highway` | Addresses with no number |
| `name_pair`, `phon_pair` *(added after run 1)* | `n2|cargo|elite` | **Common words that are rare in combination.** Single words like "elite" or "balaji" were being dropped as too frequent |
| `name_hn` *(added after run 1)* | `nh|2739|cardiology` | Name + house number |
| `num_addr_tok`, `num_name_tok`, `num_phon_tok` *(v2)* | `na|2739|cranberry` | Same as the number keys above, but using the first 3 numbers. Address parts get reordered, so the first number isn't always the house number |

**Blocking recall** (share of true training links that reach the model):

| Version | Recall | US | India | Candidates / S1 |
|---|---|---|---|---|
| Single-word keys, top 60 | 92.9% | 95.1% | 89.6% | 54 |
| + word-pair and name+number keys, top 80 (**current run**) | **94.6%** | 96.5% | 91.6% | 74 |
| v2 number keys | *pending* | | | |

What still misses, from `scripts/diag_blocking.py`:
- **80%:** the shared keys are too common. Typically a common name with an empty address, or a transliterated name with a partial address.
- **17%:** cut by the top-80 limit.
- **3%:** nothing in common at all, e.g. a random alias name.
- **By record type:** 29% of misses have Indic-script names and 20% have empty addresses.

### Stage 3: Features (`features.py`)

Computes about 45 numbers for each candidate pair.
- **String similarity** uses rapidfuzz's C++ element-wise comparison (`cpdist`), which is fast.
- **Set overlaps** are computed in plain Python.
- **Parallelism:** the work runs on all cores. Each worker gets only the column slices its chunk needs; sharing the full tables with 32 workers made memory grow past 117 GB on the first attempt.
- **Output:** parquet files of about 10M pairs each, to bound memory.

| Group | Features |
|---|---|
| Name similarity | ratio, token-sort, token-set, partial ratio; Jaro-Winkler and partial ratio on the compact name; sound-alike ratio and token-set; raw-name ratio; word Jaccard, containment both ways, extra/missing word counts; rarity-weighted (IDF) word overlap both ways |
| Address similarity | ratio, token-set, token-sort, partial ratio; address-word Jaccard and containment |
| **Numbers (decoy killers)** | Same house number (1/0/unknown), same full number set, number Jaccard, containment both ways, extra/missing number counts |
| Record flags | Candidate has empty address, Indic name, or website name; comes from S3; word and number counts on both sides |
| Blocking context | Blocking score; rank and relative score among this S1's candidates; rank and relative score among the S1s competing for this record; number of candidates on each side |

In the smoke run, the blocking-context and number features carry the most weight.

### Stage 4: Train (`model.py`)

- **Model:** LightGBM binary classifier (MIT license, far under the 8B-parameter limit). `num_leaves` 255, learning rate 0.05 (0.1 in v2), early stopping.
- **Folds are split by S1:** all candidates of one S1 are in the same fold. Every training pair then gets an **out-of-fold** probability from a model that never saw its S1.
- **Why that matters:** the post-processing rules (stage 5) are tuned on these out-of-fold probabilities, with every S1 competing for every record just as on test. That makes the tuned thresholds trustworthy.
- **Training subsample:** each fold trains on at most 400k S1s to bound time. Training uses 70% of training S1s (`data.train_s1_frac`), which bounds memory.
- **Run 1 timing:** each fold trains about 30M rows and grows 1,600–1,900 trees in about 20 minutes on 32 vCPU.

### Stage 5: Predict (`postprocess.py`)

Turns pair probabilities into one match list per S1. The order matters:

1. **Drop** pairs with probability < 0.02.
2. **Best-S1 assignment:** each S2/S3 record is kept only for the S1 that scores it highest. This is exact, not a heuristic: in 7.6M training links no record ever belongs to two S1s.
3. **Caps:** at most 5 S2 and 6 S3 records per S1, the maxima seen in training.
4. **Choose each S1's match set.** Two methods are tried on the out-of-fold predictions and the one with the better score is used:
   - **Threshold:** keep pairs with p ≥ t. If even the best pair is below a separate singleton threshold, return nothing.
   - **Expected F0.5:** sort by p and choose the number k to keep (k can be 0) that maximises `1.25 · Σ top-k p / (k + 0.25 · Σ p)`. The k = 0 option scores the probability that there is no match. This follows from F0.5 = 1.25·TP / (P + 0.25·G), where TP = correct matches, P = predicted matches and G = true matches.
5. **Write the outputs.** Every S1 in test gets a row, with an empty list for singletons. The official `utils/validate_submission.py` then runs automatically.

---

## 4. Measuring progress

`metrics.py` reproduces the competition metric exactly, including singletons. Each run records these in `work/runs/<run>/metrics.json`:

| Metric | Meaning |
|---|---|
| `block_recall` (+ per country) | Share of true links that reach the model. This is the ceiling for recall |
| `macro_f05`, `f05_US`, `f05_India` | Out-of-fold score after post-processing |
| `f05_test_mix` | The same score, re-weighted to the test's country mix (India 46.8%, US 38.3%). **This is the number to compare with the leaderboard** |
| `oof_f05_p05_no_postprocess` | Score with a plain 0.5 threshold, showing how much post-processing adds |
| `test_matches`, `test_empty_share` | Sanity checks on the submission |

`work/experiments.csv` gets one row per prediction run. The leaderboard score is added by hand after each upload. The gap between validation and leaderboard mostly measures how well France, the unseen country, is handled.

---

## 5. Compute setup

```
  Mac (VS Code + Claude Code)                          SageMaker space  (ml.g5.8xlarge: 32 vCPU, 124 GB, A10G, 70 GB disk)
  ├─ edit code                                         ~/ml2026/
  ├─ scripts/remote.sh push     ──── rsync ────►       ├─ code/, utils/, scripts/, doc/
  ├─ scripts/remote.sh run '…'  ──── ssh ─────►        ├─ dataset/            (2.3 GB, uploaded once)
  └─ scripts/remote.sh pull     ◄─── rsync ────        ├─ work/cache/         (prepared data, built once)
                                                       ├─ work/runs/<run>/    (pairs, features, models, metrics, logs)
          (AWS Toolkit SSM tunnel; one shared           └─ output/             (the two submission TSVs)
           connection, so no repeated browser logins)
```

- **Remote runs use `nohup`,** so they keep going even if the Mac disconnects.
- **Python environment:** `~/venv-ber` on the space, from pinned `requirements.txt`.
- **Run 1 memory (70% training sample):**
  - blocking about 55 GB peak;
  - features 40–90 GB;
  - training about 75 GB.

**Run 1 time per stage:**

| Stage | Time |
|---|---|
| Prepare (one-off) | 3.5 min |
| Block | ~20 min |
| Features | ~33 min |
| Train (3 folds) | ~60 min |
| Tune + predict | *pending* |

---

## 6. Known limits and next steps

| # | Change | Targets | Status |
|---|---|---|---|
| 1 | Number keys over the first 3 numbers; sound-alike + number keys (v2) | House number taken from the wrong field; Indic partial addresses | Coded, smoke-tested; runs after v1 |
| 2 | Faster training: 2 folds, learning rate 0.1 (v2) | Iteration speed | In `v2.yaml` |
| 3 | Transliteration dictionary learned from the aligned training pairs | Indic names: 29% of blocking misses and the weakest country score | Planned |
| 4 | Features where one S1's candidates support each other | Alias names and empty addresses near confident matches | Planned |
| 5 | France: check abbreviation coverage, pseudo-label confident test pairs, Qwen2.5-7B spot check | The unseen country (15% of test) | Planned |
| 6 | GPU: multilingual embeddings for blocking, cross-encoder re-ranker | The remaining hard cases | Only if time allows |

**Fixed limits:**
- Pairs that share no key at all, such as a random alias name with an empty address, can't be found.
- An S1 whose only matches are "common name + no address" can't be matched with confidence.
