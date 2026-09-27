# Pipeline Architecture: Business Entity Resolution

*Master document: current design, every version's changes and results, and open problems.*
*Last updated: 2026-09-27 (v13 running). Data analysis and strategy: [analysis.md](analysis.md). How to run: [code/business_entity_resolution/README.md](../code/business_entity_resolution/README.md).*

---

## 1. What the system does

For every Source 1 (S1) business, return the Source 2 / Source 3 records that are the same business.

- **Score:** F0.5 per S1, averaged over all S1s: F0.5 = 1.25·TP / (P + 0.25·G), with TP the correct matches, P the predicted matches and G the true matches. A wrong match costs about twice as much as a missed one. An S1 with no true matches (a *singleton*, 5.6% of train) scores 1.0 only for an empty list.
- **Submission:** `matching_results.tsv` (the matches) and `candidate_pairs.tsv` (the candidate set the model scored). The candidate file is part of the final evaluation: **smaller candidate sets per S1 rank higher**, and blocking must scale.
- **Test mix:** India 46.8%, US 38.3%, **France 15.0% (no French training data at all)**.

```
 TSV sources ─► 1 PREPARE ─► 2 BLOCK ─────────────────► 3 FEATURES ─► 4 TRAIN ──────────────► 5 PREDICT ─► 6 BLEND ─────────► matching_results.tsv
 S1, S2, S3     clean text    inverted-index keys,       ~55 numbers   stage 1 LightGBM        best-S1      full + robust    candidate_pairs.tsv
                translit.     two passes                 per pair      stage 2 LightGBM        caps         stage 2 averaged
                learned maps  → ~160 loose cands / S1                  (competition features)  selection    France shift
                              → ranker keeps ~6.4 / S1
```

Each stage reads the previous stage's files from disk, so any stage can be re-run alone (`--stage block|features|train|predict`).

---

## 2. Version history

### 2.1 Scoreboard

| Version | Main change | Blocking recall | Cands / S1 | Validation (test mix) | India / US | Leaderboard (rank) |
|---|---|---|---|---|---|---|
| v1 `v1_gbdt` | Baseline: multi-key blocking, ~45 features, 1 LightGBM | 94.6% | 74 | 0.9458 | 0.932 / 0.962 | **0.942** (160/490) |
| v2 `v2_numkeys` | Number keys use the first 3 numbers; 2 folds, lr 0.1 | 95.7% | ~76 | not finished | | not submitted |
| v3 `v3_stage2` | Number-difference features + stage-2 stacking model | 95.7% | ~76 | 0.9617 | 0.952 / 0.974 | **0.9578** (251) |
| v4 `v4_translit` | Transliteration dictionary learned from train links | 96.05% | 76 | 0.9656 | 0.959 / 0.974 | **0.9606** (313) |
| v5 `v5_ranker` | Learned candidate ranker (top 12), learned address map, support features, 100% of train S1s | 95.47% | 11.9 | 0.9668 | | not submitted |
| v5b `v5b_fullfold` | v5 with full-size folds | 95.47% | 11.9 | 0.9680 | 0.962 / 0.975 | **0.959** (>650) |
| v6 `v6_frfix_keys` | Address map made safe for France; name × address-word keys; 120 loose candidates | 97.07% | 12.0 | 0.9762 | 0.971 / 0.983 | **0.963** (702) |
| v7 `v7_fr_flex` | French address cleanup; flexible candidate sets | | | | | stopped, merged into v8 |
| v8 `v8_density` | v7 trained on 81% of train S1s (test-like density) | 97.15% | **5.75** (test 6.4) | 0.9758 | 0.970 / 0.983 | **0.9645** |
| v9 `v9_acr_3fold` | Dotted legal acronyms collapsed (`e.u.r.l.` → `eurl`); 3 model folds | 97.15% | 5.74 | 0.9763 | 0.971 / 0.983 | |
| v9 + v8 + v6 | Blend | | | 0.9774 | 0.972 / 0.984 | **0.9653** |
| v10 `v10_s2robust` | v9 without the S1-level stage-2 features | 97.15% | 5.74 | 0.9722 | 0.966 / 0.980 | |
| v11 `v11_caps` | Second blocking pass: specific keys under caps 100 / 1500 | 97.98% | 6.10 | 0.9803 | 0.978 / 0.983 | **0.9692** |
| v11b `v11b_s2robust` | v11 without the S1-level stage-2 features | 97.98% | 6.10 | 0.9763 | 0.973 / 0.980 | **0.9721** |
| v11 + v11b | Average | | | 0.9790 | 0.977 / 0.982 | **0.9733** |
| v11 + v11b + France shift | EM label-shift on French predictions | | | 0.9790 | | **0.9737** |
| v12 `v12_caps2` | Second pass caps 200 / 3000, top 100 | 98.30% | 6.40 | 0.9811 | 0.979 / 0.984 | |
| v12b `v12b_s2robust` | v12 without the S1-level stage-2 features | 98.30% | 6.40 | 0.9773 | 0.975 / 0.980 | |
| v12 + v12b + France shift | Average, France correction | | | 0.9799 | 0.978 / 0.982 | **0.9744** |
| + stage 3 | Record-cluster features on top of the blend (twins, shared numbers) | | | 0.9810 | 0.979 / 0.983 | *pending* |
| v13 `v13_keys_ties` | + compact name × number / address word, one-digit-deletion keys, ties kept at the top-N cut | loose 98.67% | | *running* | | |

*Blocking recall = share of true training links that reach the model (the recall ceiling). Validation = out-of-fold F0.5 re-weighted to the test's India/US mix; France can't be validated.*

### 2.2 What each version changed, and why

**v1: baseline.**
- Prepare: transliteration of 9 Indic scripts, name/address cleaning, phonetic skeleton, number extraction.
- Blocking: single-word keys, plus word-pair and name + house-number keys (added after the first attempt found 92.9% recall, because common single words were dropped by the frequency caps). Top 80 per S1.
- 3-fold LightGBM, lr 0.05, 255 leaves. Trained on 40% of train S1s.
- Post-processing: best-S1 assignment, per-source caps, tuned threshold or expected-F0.5 selection.
- Fixes along the way: a website regex that matched the empty string; workers copying the full tables (117 GB RAM), so each worker now gets only its column slices.

**v2: number keys.**
- Addresses get reordered ("unit 8 2739 cranberry hwy"), so the first number isn't always the house number. Number keys now pair each of the first 3 numbers with address, name and phonetic words.
- 2 folds and lr 0.1 roughly halved training time. 70% of train S1s.

**v3: decoy features and stacking.** Error analysis on v1 found three sources of loss, roughly half, a quarter and a quarter: blocking misses, model misses, wrong matches (mostly sibling decoys: same name, house number shifted by 1–20).
- Number-difference features: log house-number difference, substring (true matches drop or insert digits; decoys shift), Levenshtein, cross-containment.
- **Stage 2:** a second LightGBM on stage-1 out-of-fold scores plus *competition* features: the best competing S1 for the same record, the best competing record for the same S1, rank within each, and head-to-head deltas.

**v4: learned transliteration.**
- Many Indic names are word-by-word transliterations of the English name ("balaji bisines praivet limited"). From training links with equal word counts we learn transliterated word → English word (686 words), then rewrite Indic-origin names in train and test.
- India +0.007 on validation.

**v5: learned candidate ranker (new organiser rule: small candidate sets rank higher).**
- A small LightGBM on 7 cheap features plus blocking context, trained out-of-fold, scores the loose candidates. Only the top 12 per S1 are kept: 72.9 → 11.9 per S1 for −0.01 points of recall.
- Learned address-word map: record word → S1 word (1,188 words, e.g. `wb` → `west bengal`), from links where exactly one word differs on each side.
- Stage-2 *support* features: similarity of each candidate to the S1's two strongest other candidates.
- Switched to 100% of train S1s.
- **The leaderboard went down** (0.9606 → 0.959). The cause was found in v6 (see below).

**v6: France fix and name × address keys.**
- **Bug found:** the address map was learned on India/US and applied everywhere. It rewrote **52% of French records** wrongly: `de → des` (791k times), `saint → street` (115k). A rule now applies in a country only if its target word is more frequent than its source word in that country's S1 addresses. French records rewritten: 51.6% → 2.1%.
- **Blocking gap found:** 71% of blocking misses were common names whose keys were all capped, with no usable number ("mountain trust" + "kniffin street"). New keys pair the first 3 name / phonetic words with up to 12 address words (`nx|`, `px|`). Loose candidates per S1 raised from 80 to 120. Recall 95.47% → 97.07%.
- Validation +0.008; leaderboard +0.004.

**v7: France cleanup and flexible candidate sets** (stopped before training; everything carried into v8).
- French S1 addresses end with the region, records with the department ("bordeaux nouvelle aquitaine" vs "bordeaux gironde"). Both are now stripped, along with `ndeg` (n°).
- **Flexible candidate sets:** instead of a fixed top 12, keep candidates whose ranker score is ≥ 0.001, at least 1 per S1. The ranker is sharply bimodal: true matches have a median score of 0.9975, non-matches 0.00014. The cutoff keeps 99.98% of true matches and drops 74% of non-matches.

**v8: test-like training density.**
- The test has 5.75 S2/S3 records per S1, train 4.68. Only 58.9% of test records have a strong S1 candidate (ranker score > 0.5), against 69.3% in train. The test has many more records with no real match.
- Train now uses 81% of S1s (4.68 / 5.75), so the other 19% of S1s' records become unmatched records, as in the test.
- Result: validation unchanged (0.9758). So the model already rejects such records well, and the hypothesis doesn't explain the leaderboard gap on its own (see section 5).
- Candidate sets: **5.75 per S1 on train, 6.4 on test**, half of v6.
- Leaderboard **0.9645** (+0.0015 over v6).

**v9: acronyms and 3 folds.**
- Dotted legal acronyms ("e.u.r.l.", "s.a.", "l.l.c.") were split into single letters. They are now joined and dropped as legal words; this is common in French names.
- 3 model folds instead of 2: each model trains on 67% of the data, and test uses the average of 3 models.

**Blending** (`scripts/blend.py`): averaging the final probabilities of several runs on the first run's candidate set, then re-tuning selection. v8 + v6 gives validation 0.9772 (+0.0014 over either alone).

**v10: density-robust stage 2.**
- The test has ~1.9× (US) and ~1.65× (India) more decoy-type records per S1 than train. Stage 2 lifts 2.3× more decoy-type test pairs above 0.5 than stage 1 does; on train it doesn't.
- The cause is the S1-level group features (`*_maxother_s1`, `p_rank_s1`, `s1_n_high`, `s1_sum_p`, support features), which depend on how many decoys an S1 faces.
- v10 drops them (`model.stage2_exclude`). Alone it validates lower. The fix that works is to **average** the full and the robust model: see v11 + v11b.

**v11: second blocking pass.**
- For most missed links, the most specific shared key was shared by 30–100 S1s, just above the pass-1 cap of 30.
- A second pass runs only the specific key types (compact name, prefix, word pairs, phonetic compact, number × word, bigrams, name × address) under caps of 100 / 1500 and keeps the top 60. The two passes are joined with summed scores.
- Recall 97.15% → 97.98% (India 95.7% → 97.3%).
- On the leaderboard: v11 alone 0.9692, v11b (robust) 0.9721, **average 0.9733**. The France label-shift correction on top gave 0.9737.

**v12: looser second pass.** Caps 200 / 3000, top 100. Recall 98.30%. Average + France correction: leaderboard **0.9744**.

**Stage 3** (`src/stage3.py`, on top of a blend):
- 26% of S2/S3 records belong to no S1 and cause most false positives. They are one-off perturbed copies, while an entity's true records often repeat each other exactly (21% of true positives have an exact twin among the S1's candidates, 1.4% of orphan false positives).
- A LightGBM on the blend's logit plus twin counts, shared number sets / addresses with their summed p, global duplicate count and number relations: validation +0.0011.
- Its US test additions are 3× the train rate, a possible density effect; the leaderboard decides.

**v13: more keys, tied candidates.**
- Compact name × number and × address word keys, and house numbers with one digit deleted (a true "385" appears as "85").
- Candidates tied with the top-N-th score are kept, up to 240. Before, ~190 records sharing only a common name were cut arbitrarily.
- Loose recall 98.34% → 98.67%.

**France correction** (`src/france_shift.py`): EM label shift (Saerens et al.) on the French predictions only.
- A leave-one-country-out simulation, training on US and predicting India, confirmed that it helps when the unseen country's prior is lower than train's (France: 0.42 vs 0.53).
- In the opposite case it hurts, so it is skipped then.

---

## 3. The stages (current design, v8)

### 3.1 Prepare (`normalize.py`)

Turns raw names and addresses into comparable forms. Rules must work for France, which has no labels.

| Column | How it is built | Why |
|---|---|---|
| `name` | Transliterate to Latin → lowercase → strip web parts (`www.`, `.com`) → fix digit-for-letter typos inside words (`h0ldings`) → drop legal and filler words (`pvt`, `ltd`, `llc`, `sarl`, `sas`, `shri` …) | Removes generator noise |
| `name_compact` | Name words joined without spaces | Website and hashtag names |
| `phon` | Consonant skeleton: vowels and `h` dropped, voiced/unvoiced consonants merged | Transliterations and typos: "praivet" ≈ "private" |
| `addr` | Transliterate → expand abbreviations (US, India, France) → drop filler (`null`, `n/a`, `ndeg`) → drop French region / department names | Abbreviations, reordering, French area names |
| `addr_alpha` | Address words of 3+ letters | Blocking keys |
| `nums`, `hn` | Distinct address numbers (leading zeros removed); the first number | Sibling-decoy detection |
| flags | `name_indic`, `name_web`, `addr_empty` | Hard cases the model treats separately |

**Learned from training links** (train labels only; applied to train and test S2/S3):
- `translit.json`: 686 transliterated words → English, applied to Indic-origin names (v4).
- `addr_map.json`: 1,188 record address words → S1 words, applied per country only where the target word is the more common one in that country's S1 addresses (v5, guarded in v6).

Output: `work/cache/dataset<cache_tag>/`. Changing `normalize.py` needs a new `data.cache_tag`.

### 3.2 Block (`blocking.py` + `candidates.py`)

**Step 1: inverted-index lookup** (`blocking.py`).
1. Every record emits hashed keys, each prefixed with the country: records are only compared within a country, since no training match crosses countries.
2. Keys shared by more than 30 S1s or more than 400 S2/S3 records (per country) are dropped.
3. Pair score = Σ over shared keys of 1 / log2(2 + key frequency). The top 120 per S1 are kept (~115 on average).

| Key | Example | Catches | Since |
|---|---|---|---|
| `name_tok`, `name_compact`, `name_prefix` | `nt|balaji`, `nc|sweetpilates` | Name overlap, website names | v1 |
| `phon_tok`, `phon_compact` | `pt|psns` | Transliterations, typos | v1 |
| `addr_bigram` | `ab|cranberry|highway` | Addresses without numbers | v1 |
| `name_pair`, `phon_pair` | `n2|cargo|elite` | Common words that are rare in combination | v1 |
| `num_addr_tok`, `num_name_tok`, `num_phon_tok` | `na|2739|cranberry` | Any of the first 3 numbers + a word | v2 |
| `name_addr`, `phon_addr` | `nx|mountain|kniffin` | Common names at addresses without usable numbers | v6 |

**Step 2: learned pruning** (`candidates.py`, since v5).
- A LightGBM ranker on cheap features: name token-set, compact-name Jaro-Winkler, phonetic token-set, address token-set, number Jaccard and containment, house-number equality, plus blocking context.
- Trained with 2 folds by S1, so train pairs get out-of-fold scores, and pruning on train behaves like pruning on test. Test uses the average of the 2 fold models.
- Keeps up to `keep_k` = 12 per S1, dropping pairs scoring below `min_cscore` = 0.001 but always keeping at least `min_k` = 1.
- The ranker score is also passed on as a feature (`cscore`).
- **The kept set is what `candidate_pairs.tsv` reports.**

### 3.3 Features (`features.py`)

About 55 numbers per pair, computed in parallel with rapidfuzz `cpdist`. Output is written in parquet parts of up to 10M pairs.

| Group | Features |
|---|---|
| Name | ratio, token-sort, token-set, partial; compact Jaro-Winkler and partial; phonetic ratio and token-set; raw ratio; word Jaccard, containment, extra/missing counts; IDF-weighted overlap both ways |
| Address | ratio, token-set, token-sort, partial; word Jaccard and containment |
| Numbers | house number equal; number set equal; Jaccard, containment, extra/missing counts; log difference, substring, Levenshtein, cross-containment, min log difference |
| Record flags | empty address, Indic name, website name, source 3, word and number counts |
| Context | blocking score, rank and relative score within the S1 and within the record, candidate counts, ranker score |

### 3.4 Train (`model.py`, `stage2.py`)

- **Stage 1:** LightGBM binary (lr 0.1, 255 leaves, early stopping), 2 folds by S1 hash. Every train pair gets an out-of-fold probability `p`.
- **Stage 2:** the same model type on the stage-1 features plus features built from out-of-fold `p`:
  - *Competition:* the maximum of the other pairs' values for the same record and for the same S1, for `p` and for the key similarities; ranks; counts of confident pairs.
  - *Head-to-head:* this pair's value minus the best competing S1's value, for the same record.
  - *Support:* name, address and house-number similarity to the S1's two strongest other candidates.
- Stage 2 adds about +0.005 (v6: 0.9706 → 0.9762).
- Largest importances: `p1`, `p_delta_r`, `cscore`, `r_n_high`, `b_n_r`.

### 3.5 Predict (`postprocess.py`)

1. Drop pairs with p < 0.02.
2. **Best-S1:** each record goes only to its highest-scoring S1. This is exact: no training record belongs to two S1s.
3. **Caps:** at most 5 S2 and 6 S3 records per S1 (the training maxima).
4. **Set selection**, tuned on out-of-fold predictions: either a threshold, or the k per S1 (0 allowed) that maximises expected F0.5. Expected F0.5 has won every run.
5. Write both TSVs. The official validator then runs, and a copy is kept in `work/runs/<run>/submission/`.

---

## 4. Measurement and compute

**Metrics** (`work/runs/<run>/metrics.json`; one row per run in `work/experiments.csv`):
- `block_recall` (with a `loose_` version before pruning) and `cands_per_s1`;
- stage-1 and final out-of-fold F0.5, per country and test-mix weighted;
- test match counts per country.

**Diagnostics** (`scripts/`):

| Script | Answers |
|---|---|
| `error_analysis.py <run>` | Score ceilings; loss split (blocking / model miss / wrong match); examples |
| `diag_blocking.py <run_dir> [n] [extra_keys]` | Why true links miss blocking: capped / cut by top-N / no shared key; what-if for new key types |
| `compare_runs.py <a> <b>` | Test predictions of two runs, by country |
| `flex_k.py <run>` | Candidate-set size vs recall for fixed and score-based cuts |
| `density_check.py`, `density_sim.py` | Records per S1, train vs test; out-of-fold score at other densities |
| `france_map_check.py`, `france_look.py` | What the address map does to French records; uncertain French predictions |

**Compute:**
- SageMaker `ml.g5.8xlarge`: 32 vCPU, 124 GB RAM, A10G (unused so far), 70 GB disk.
- Code synced from the Mac with `scripts/remote.sh`; runs use `nohup`.
- Full run (v6): prepare 6 min, block 86 min, features 9 min, train 36 min, predict 14 min, about **2.5 h** in all.
- Peak RAM about 70 GB.

---

## 5. Open problem: the validation-to-leaderboard gap

| Version | Validation | Leaderboard | Gap | Share of train S1s |
|---|---|---|---|---|
| v1 | 0.9458 | 0.942 | 0.004 | 40% |
| v3 | 0.9617 | 0.9578 | 0.004 | 70% |
| v4 | 0.9656 | 0.9606 | 0.005 | 70% |
| v5b | 0.9680 | 0.959 | 0.009 | 100% |
| v6 | 0.9762 | 0.963 | **0.013** | 100% |

Validation covers only India and the US, so its test-mix number assumes France scores like them. If India/US transfer exactly, the implied France score is (leaderboard − 0.468·India − 0.383·US) / 0.15:

| Version | Implied France |
|---|---|
| v1 | 0.914 |
| v3 | 0.929 |
| v4 | 0.925 |
| v5b | 0.902 |
| v6 | 0.882 |

**What has been ruled out or fixed:**
- The address map breaking French addresses: fixed in v6, yet the implied France score kept falling.
- Missing unmatched records in training: v8 adds them at the test's rate and validation didn't move.

**Remaining suspects:**
1. **France really is weak**, and our India/US-tuned blocking keys and features generalise worse with each version.
2. **The learned maps overfit:** `translit.json` and `addr_map.json` are learned from all training links and applied to those same links, so validation sees their in-sample benefit. Test: learn the maps per fold.
3. **Stage-2 input mismatch:** stage 2 is trained on out-of-fold `p` from single fold models, but test `p` is the average of 2 fold models, which is distributed differently.

**France probe (2026-09-26).** We submitted v8 (0.964459) and v8 with every French S1 emptied (0.831865). Both are scored on the same public subset. With France's weight at about 0.15 and a French singleton share of 5–6%:

| | Leaderboard (derived) | Validation | Gap |
|---|---|---|---|
| **France** | ≈ 0.94 (0.936–0.946) | — | ≈ 0.036 vs India/US |
| **India + US** | ≈ 0.969 | 0.9758 | ≈ 0.007 |

The 0.011 gap of v8 splits roughly in half: about 0.005 from France and about 0.006 from India/US not transferring.

A label-shift check (`scripts/label_shift.py`) doesn't explain the India/US part. EM on the test predictions leaves the India/US prior almost unchanged (0.56 → 0.53), and predicted matches per S1 move by less than 0.01.

---

## 6. Roadmap

| Idea | Expected gain | Status |
|---|---|---|
| France-only probe submission | Diagnosis | Done (France ≈ 0.94 at v8) |
| More French-specific cleanup (legal forms like `e.u.r.l.`) | +0.002 to +0.005 on test | Acronyms done (v9) |
| Second blocking pass, looser caps | +0.004 validation | Done (v11, v12) |
| Density-robust stage 2 (full + robust average) | +0.004 leaderboard | Done (v11 + v11b) |
| France label-shift correction | +0.0004 leaderboard | Done |
| Stage 3 (record-cluster features) | +0.0011 validation | Leaderboard pending |
| v13 keys and tie-aware top-N | +0.3 points of loose recall | Running |
| Per-group selection thresholds | +0.0002 validation | Not adopted (small) |
| Stronger / per-stratum France correction | about +0.0001 leaderboard | Not adopted (see investigations §18) |
| Self-training on French pseudo-labels | Negative in simulation | Rejected |
| LightGBM seed ensemble | +0.0005 to +0.001 | Not done |
| Multilingual cross-encoder as a stage-2 feature (interface `s1_id, rid, score`) | +0.003 to +0.006 on test | Not done |
| Multilingual embedding blocking | +0.002 to +0.004 | Not done |

**Known hard limits:**
- Pairs with no shared key (random alias names, empty addresses).
- S1s whose only matches are "common name, no address": these can't be matched with confidence.
