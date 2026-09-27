# Investigations Log

*Every analysis run during the challenge (25–27 Sep 2026): the question, how it was answered, the numbers, and what we did about it. Negative results are included; they saved as much time as the positive ones.*

*Related: the design and version history are in [architecture.md](architecture.md); the initial data analysis is in [analysis.md](analysis.md). Scripts are in `scripts/` and run from the repo root on the SageMaker space.*

---

## Contents

1. [Data facts that shaped everything](#1-data-facts-that-shaped-everything)
2. [Leak check](#2-leak-check)
3. [Where the score is lost (error analysis)](#3-where-the-score-is-lost-error-analysis)
4. [Why blocking misses true matches](#4-why-blocking-misses-true-matches)
5. [Candidate pruning: fixed vs learned vs flexible](#5-candidate-pruning-fixed-vs-learned-vs-flexible)
6. [The France address-map bug](#6-the-france-address-map-bug)
7. [Estimating France without labels](#7-estimating-france-without-labels)
8. [Train vs test density](#8-train-vs-test-density)
9. [The France probe submission](#9-the-france-probe-submission)
10. [Label-shift check](#10-label-shift-check)
11. [French data observations](#11-french-data-observations)
12. [Blending runs](#12-blending-runs)
13. [Modelling choices we considered and rejected](#13-modelling-choices-we-considered-and-rejected)
14. [Blocking recall push (v11–v13)](#14-blocking-recall-push-v11v13)
15. [Stage-2 features that inflate on test](#15-stage-2-features-that-inflate-on-test)
16. [Where the score is lost after blocking (v12)](#16-where-the-score-is-lost-after-blocking-v12)
17. [Orphan records and stage 3](#17-orphan-records-and-stage-3)
18. [Correcting an unseen country: leave-one-country-out simulation](#18-correcting-an-unseen-country-leave-one-country-out-simulation)
19. [Engineering incidents](#19-engineering-incidents)
20. [Lessons](#20-lessons)

---

## 1. Data facts that shaped everything

Measured by streaming the TSVs (a laptop with 8 GB RAM can't load them whole). Details in [analysis.md](analysis.md).

| Fact | Number | Consequence |
|---|---|---|
| A record matches at most one S1 | 0 exceptions in 7.6M links | **Best-S1 assignment** is exact, not a heuristic |
| Matches never cross countries | 0 | Country is a hard partition for blocking |
| Unmatched S2/S3 records (decoys) | ~26% of train records | Precision matters; F0.5 punishes wrong matches |
| S1s sharing a normalised name with another S1 | 50% | Name alone can't decide; addresses and numbers must |
| Singletons (S1 with no match) | 5.6% | They score 1 only for an empty list |
| Indic-script names | 23.5% of India S2, 13.2% of India S3, 9 scripts | Transliteration layer, then a learned word map |
| Sibling decoys | House number shifted by 1–20; true matches instead drop or insert digits (33% are substrings) | Number-difference features |
| France | 15% of test S1s, 0% of train | Nothing can be validated there; see sections 6, 7 and 9 |
| Test mix | India 46.8%, US 38.3%, France 15.0% | Validation is re-weighted to this mix (`f05_test_mix`) |

---

## 2. Leak check

**Question:** do row order or entity IDs reveal matches?
**Method:** Spearman correlation between an S1's position and ID, and its matched records' position and ID, on 200k true links (`scripts/error_analysis.py`).

| | Spearman |
|---|---|
| Row position, S1 vs record | −0.0015 |
| Numeric ID, S1 vs record | −0.0011 |

**Conclusion:** no leak and no locality. The organisers shuffled properly.

---

## 3. Where the score is lost (error analysis)

**Method:** `scripts/error_analysis.py <run>` classifies every S1 by what went wrong in the out-of-fold predictions, and computes ceilings by fixing one error type at a time.

### v3 (validation 0.9617)

| | Value |
|---|---|
| Ceiling with a perfect model on the blocked candidates | 0.9803 |
| Loss from blocking misses / model misses / wrong matches | 52% / 24% / 24% |
| Model misses lost to best-S1 or set selection (p was above threshold) | 31% |

### v5b (validation 0.9680; total loss 0.030 per S1)

| Ceiling | Test mix | US | India |
|---|---|---|---|
| Current | 0.9680 | 0.9753 | 0.9620 |
| No false positives | 0.9723 | 0.9790 | 0.9667 |
| + perfect model on blocked pairs | 0.9799 | 0.9870 | 0.9740 |
| + perfect blocking | 1.0000 | | |

| Loss category | Share of loss | S1s affected |
|---|---|---|
| True match never blocked | **63.1%** | 250,076 |
| Blocked but model said no | 21.8% | 147,909 |
| Wrong match only | 7.1% | 20,628 |
| Singleton given a match | 4.7% | 3,127 |
| Wrong match and a miss | 3.4% | 6,338 |

- False positives: **83% are unmatched decoys**, 17% belong to another S1.
- Model misses: **47% have an empty address** (name-only evidence).

**Conclusion:** blocking was the biggest lever. The leaderboard's top score (0.987) is above our perfect-model ceiling (0.980), so the leaders must have better recall. This led directly to the v6 blocking keys (section 4).

---

## 4. Why blocking misses true matches

**Method:** `scripts/diag_blocking.py <run_dir> [n] [extra_key_types]`. For 20k sampled train S1s, each missed true link is classified:
- `capped`: the pair shares keys, but all were dropped by the frequency caps;
- `cut_by_top_n`: it survived the caps but ranked below the loose top-N;
- `no_shared_key`: nothing in common.

The optional third argument recomputes the reasons as if extra key types existed, which gives a what-if before a 3-hour run.

| Run | Missed links | Capped | Cut by top-N | No shared key |
|---|---|---|---|---|
| v1 era | 5.22% | 80.0% | 17.1% | 2.9% |
| v5b | 4.23% | 71.4% | 24.8% | 3.7% |
| v5b + name × address keys (what-if) | 4.23% | **44.8%** | 51.6% | 3.6% |

**Typical capped misses:** "mountain trust" at "kniffin street greenwich", "urban nails" at "herndon", "al power private limited" at "mastipur gaya". All are common names with no usable house number. Their name keys are shared by hundreds of S1s, so all were dropped. We paired names with **numbers** but never with **street or locality words**.

**Fix (v6):** `name_addr` and `phon_addr` keys (first 3 name or phonetic words × up to 12 address words), plus 120 loose candidates instead of 80.

**Blocking recall history:**

| Version | Recall | Change |
|---|---|---|
| First attempt | 92.9% | Single-word keys |
| v1 | 94.6% | Word-pair and name + number keys |
| v2 | 95.7% | Number keys over the first 3 numbers |
| v4 | 96.05% | Learned transliteration |
| v5 | 95.48% | Lower: 100% of train S1s makes the per-S1 frequency caps bite harder |
| v6 | **97.08%** | name × address keys, top 120 |
| v8 | 97.18% | 81% of S1s: fewer S1s share each key, so fewer keys are capped |

**Remaining misses:** generic names with empty addresses ("atlantic family", "oncology clean"), and random alias names with nothing in common. Both are close to unfixable.

---

## 5. Candidate pruning: fixed vs learned vs flexible

Context: an organiser update made `candidate_pairs.tsv` part of the final evaluation, with **smaller sets per S1 ranking higher**. We had 76 per S1.

### Ranking by what? (v3 data)

| Keep top 10 by | True links kept |
|---|---|
| Blocking score (sum of key rarities) | 91.0% |
| The full model's score | 95.75% |

Heuristic re-rankings of the blocking score didn't close the gap, so we built a **learned candidate ranker**: LightGBM on 7 cheap features plus blocking context, trained out-of-fold (`candidates.py`).

**Bug on the way:** the first ranker kept only 27% of true links. Labels had been computed before the pairs were sorted and then applied to the sorted rows. Labels are now computed inside `prune()` after sorting.

### Fixed vs flexible size (v5b train candidates, `scripts/flex_k.py`)

| Cut | Per S1 | True links kept |
|---|---|---|
| Fixed top 4 | 3.99 | 83.79% |
| Fixed top 8 | 7.95 | 95.34% |
| Fixed top 12 | 11.89 | 95.47% |
| **Ranker score ≥ 0.001, at least 1** | **5.56** | **95.46%** |
| Ranker score ≥ 0.01, at least 1 | 4.73 | 95.37% |

### Why 0.001 isn't as extreme as it looks

The ranker is a log-loss classifier, so its scores are probabilities. On its kept candidates, 28% are true matches, and the two classes barely overlap:

| Percentile | True matches | Non-matches |
|---|---|---|
| 5% | 0.36 | 0.00001 |
| 25% | 0.975 | 0.00004 |
| Median | **0.9975** | **0.00014** |
| 75% | 0.9996 | 0.0012 |

| Keep if score ≥ | True matches kept | Non-matches kept |
|---|---|---|
| 0.0001 | 100% | 56% |
| 0.001 | 99.98% | 26% |
| 0.1 | 98.83% | 7.4% |
| 0.5 | 93.2% | 1.2% |

Blocking must not lose true matches, so the cut is cautious. The matching model makes the real decision afterwards.

**Result in v8:** 5.75 per S1 on train, **6.4 on test** (v6 had 12), and 97.15% of true links kept.

---

## 6. The France address-map bug

**Symptom:** v5 improved validation, but the leaderboard dropped (v4 0.9606 → v5b 0.959).

**Method:** `scripts/france_map_check.py` counts what the learned address map does to French test records. The map was learned on India/US training links and applied to every S2/S3 record.

| Map rule | Fired on French records | That word in French S1 addresses | Target word in French S1 |
|---|---|---|---|
| `de → des` | **791,194** | 257,496 | 31,485 |
| `saint → street` | **115,083** | 26,280 | 141 |
| `street → saint` | 24,469 | 141 | 26,280 |
| `b → unit` | 11,450 | 1,665 | 0 |
| `aveue → avenue` (typo fix) | 569 | 0 | 33,198 |

**51.6% of French records were rewritten**, mostly wrongly, so French record addresses stopped matching their own S1s.

**Fix (v6):** a rule applies in a country only if its target word is more frequent than its source word in that country's S1 addresses. The idea is to move words towards the local S1 vocabulary. It uses only unlabelled test S1 text.

| Records rewritten | Old map | Guarded map |
|---|---|---|
| France | 51.6% | **2.1%** |
| India | 65.1% | 49.2% |
| US | 71.4% | 58.5% |

**Lesson:** anything learned from India/US labels must be checked for what it does to France.

---

## 7. Estimating France without labels

Validation covers India and the US only. If the leaderboard's India/US parts equal validation, France's score can be backed out:

France ≈ (leaderboard − 0.468 · India − 0.383 · US) / 0.15

| Version | India (val) | US (val) | Leaderboard | Implied France |
|---|---|---|---|---|
| v1 | 0.932 | 0.962 | 0.942 | 0.914 |
| v3 | 0.952 | 0.974 | 0.9578 | 0.929 |
| v4 | 0.959 | 0.974 | 0.9606 | 0.925 |
| v5b | 0.962 | 0.975 | 0.959 | 0.902 |
| v6 | 0.971 | 0.983 | 0.963 | 0.882 |

This found the v5 drop (section 6). But v6's value of 0.882, after the fix, made no sense. The assumption that India/US transfer exactly was wrong, and section 9 measured it directly.

---

## 8. Train vs test density

**Question:** does the test differ structurally from train?

**Records per S1** (`scripts/density_check.py`):

| | S1 | S2 + S3 | Records per S1 |
|---|---|---|---|
| Train India | 883,188 | 4,133,346 | 4.68 |
| Train US | 1,323,633 | 6,186,873 | 4.67 |
| Test India | 809,986 | 4,717,565 | **5.82** |
| Test US | 663,106 | 3,817,031 | **5.76** |
| Test France | 259,452 | 1,434,993 | 5.53 |

Train has 3.46 true links per S1 in both countries. If the test is similar, about 40% of test records are unmatched against 26% in train. The ratio is the same in every country, which looks as if roughly 19% of the test's S1s were withheld.

**Supporting evidence** (`scripts/transfer_check.py` and an inline check on v6/v8 pairs):

| | Train | Test |
|---|---|---|
| Records with a candidate S1 at ranker score > 0.5 | 69.3% | **58.9%** |
| … > 0.9 | 63.2% | 52.3% |
| Candidates per S1 scoring 0.5–0.9, India | 0.38 | **0.47** |
| Candidates per S1 scoring 0.5–0.9, US | 0.26 | 0.30 |

**First simulation (flawed):** `scripts/density_sim.py` dropped 19% of S1s from v5b's out-of-fold predictions and re-scored them. The score didn't change: 0.9680 → 0.9680. This was invalid, because the stage-2 features (best competing S1 and so on) had been computed with the dropped S1s still present.

**Proper test (v8):** retrain everything on 81% of train S1s, so the other 19%'s records become real unmatched records in blocking, features and training. Validation came out **0.9758 vs 0.9762** for v6, and the leaderboard went +0.0015. **Conclusion:** the model already rejects such records well, so density explains little of the gap. v8 is kept because it's more test-like and its candidate sets are smaller.

---

## 9. The France probe submission

**Idea:** submit v8 twice, once normally and once with every French S1's list emptied. Both are scored on the same public subset, so the difference isolates France exactly, whatever the subset is.

| Submission | Leaderboard |
|---|---|
| v8 | 0.964459 |
| v8 with France emptied | 0.831865 |

**Derivation**, with France's weight w ≈ 0.15 and a French true-singleton share s ≈ 0.05–0.06. Emptied French S1s score 1 only when they really are singletons:
- France = s + (0.964459 − 0.831865) / w ≈ **0.94** (range 0.936–0.946)
- India + US = (0.831865 − w · s) / 0.85 ≈ **0.969**, against 0.9758 on validation

| Gap in v8 (0.011) | Contribution |
|---|---|
| France below India/US (0.94 vs 0.969, weight 0.15) | ≈ 0.005 |
| India/US leaderboard below validation (0.969 vs 0.976, weight 0.85) | ≈ 0.006 |

**Conclusions:**
1. France is weaker (≈ 0.94) but not broken.
2. India/US lose about 0.007 from validation to the leaderboard. That gap already existed at v1 (0.004–0.005). Improvements still carry over at about 60–70%: v5b → v8 was +0.008 on validation and +0.005 on the leaderboard.

The probe is a reusable technique: spend one submission to learn the true score of any slice.

---

## 10. Label-shift check

**Question:** the test has more mid-confidence candidates (section 8). Is the model over-confident there?

**Method:** `scripts/label_shift.py` applies the EM correction of Saerens et al. (2002). It estimates the test's share of true matches among candidates, per country, from the test predictions alone, re-weights every probability by the prior ratio, and re-selects.

| Country | Train prior | Test prior (EM) | Matches per S1, before → after |
|---|---|---|---|
| India | 0.561 | 0.535 | 3.274 → 3.267 |
| US | 0.601 | 0.583 | 3.387 → 3.382 |
| France | 0.585 (overall) | 0.435 | 3.363 → 3.309 |

**Conclusion:** India/US barely move, so calibration isn't the cause of their gap. Not worth a submission. The France prior is much lower, which fits France having more ambiguous candidates.

---

## 11. French data observations

From `scripts/france_look.py` (uncertain French test predictions) and word counts of French test addresses:

| Observation | Example | Handling |
|---|---|---|
| S1 addresses end with the region; records with the department | "bordeaux nouvelle aquitaine" vs "bordeaux gironde"; "lille hauts de france" vs "lille nord" | v7: both stripped (the French test has only 3 regions and 4 departments) |
| `n°` appears as "ndeg" | "ndeg 59 rue turenne" | v7: stop word |
| "St Nazaire" expanded to "street nazaire" by our abbreviation table | 24,469 records | The guarded map turns `street → saint` back for France (v6) |
| Dotted legal forms split into letters | "e.u.r.l.", "s.a." | v9: joined and dropped as legal words |
| OCR in addresses | "chem1" for "chemin" | Not handled (a general OCR fix would break "1st", "2nd") |
| Sibling decoys look like the US/India ones | S1 "44 bis rue du port", records "46 bis …" | Handled by the number features |
| French predictions are less certain | Pairs with p in 0.2–0.8: France 3.05%, US 1.82%, India 1.92% | Open problem |

---

## 12. Blending runs

**Method:** `scripts/blend.py <runs…>` averages the final probabilities on the first run's candidate set, falling back to that run's own p where another run lacks the pair. It then re-tunes selection on the blended out-of-fold predictions.

| | Validation |
|---|---|
| v8 alone | 0.9758 |
| v6 alone | 0.9762 |
| **v8 + v6** | **0.9772** |

v6 covers 92% of v8's train pairs and 98% of its test pairs. The runs differ in training view, candidate sets and cleaning, so their errors partly cancel out.

---

## 13. Modelling choices we considered and rejected

| Idea | Verdict | Why |
|---|---|---|
| XGBoost / CatBoost instead of LightGBM, or a GBDT ensemble | Rejected | Same feature set, same information; expected ≤ +0.001 for hours of work |
| Bigger LightGBM | Rejected | Early stopping already picks the size; validation was flat |
| Another re-ranker | Rejected | The candidate ranker and stage 2 already re-rank |
| Multilingual embedding blocking (FAISS) | Deferred | Embedding 22M records plus a full rerun didn't fit the time left |
| Fine-tuned multilingual cross-encoder as a stage-2 feature | Handed to a teammate | Best candidate for France (pretrained French understanding); needs 8–10 GPU hours |
| 7B LLM as matcher | Diagnostic only | Too slow for millions of pairs; licence caution for some models |
| Label-shift correction | Tested, negligible | Section 10 |

**Estimates made at the time** (validation / leaderboard): flexible candidate cut ≈ 0 / 0 (it only shrinks the file); French cleanup 0 / +0.002–0.005; seed ensemble +0.0005–0.001; cross-encoder +0.002–0.004 / +0.003–0.006.

---

## 14. Blocking recall push (v11–v13)

**Question:** how close to 100% can blocking recall get, and where do the remaining misses come from?

**Method:** `scripts/funnel.py` follows every train link through the pipeline, and `scripts/cap_whatif.py` measures what looser caps or new key types would recover.

**Funnel, v11 (train, out-of-fold, 6,188,505 links):**

| Step | True links kept | Note |
|---|---|---|
| Loose blocking | 98.01% | |
| After candidate ranker | 97.98% | 6.1 candidates per S1 |
| Pair model p ≥ 0.5 | 96.14% | Precision 99.00% |
| After selection | 95.21% | Precision 99.49%, 3.31 matches per S1 |
| Macro F0.5 | 0.979 | |

**The 2% of links blocking misses:**

| Type | Share | Recoverable? |
|---|---|---|
| Good name + **empty address** | 44.9% | Mostly not: a median of 42 S1s share the name |
| Good name + good address | 17.5% | Yes (92% India): only the caps dropped them |
| Alias name (name similarity < 50) | 13.8% | Only through the address |
| Partial name + good / poor address | 19.0% | Partly |
| Good name + poor address or no number | 4.7% | Partly |

Only 3.3% of misses share no key at all. For the rest, the most specific shared key has a median frequency of 45 S1s (pass-1 cap: 30).

| Version | Change | Loose recall | After ranker |
|---|---|---|---|
| v11 | Second pass, caps 100 / 1500, top 60 | 98.01% | 97.98% |
| v12 | Second pass, caps 200 / 3000, top 100 | 98.34% | 98.30% |
| v13 | + compact name × number / address word, one-digit-deletion keys, keep ties at the top-N cut | 98.67% | 98.61% (7.0 per S1; test 7.4) |

**Checked and rejected:** "same name + both addresses empty → same S1". In train such records are spread over the namesake S1s (a median of 42 S1s share the name), so the rule would be wrong most of the time.

---

## 15. Stage-2 features that inflate on test

**Observation:** v11 alone scored 0.9692 on the leaderboard, below v9's blend despite +0.0016 validation.

**Method:** `scripts/stage2_check.py` and `scripts/stage2_features_shift.py` compare how often stage 2 lifts decoy-type pairs (house number shifted 1–20, strong name) above p = 0.5, on train and on test.

| | Train | Test |
|---|---|---|
| Decoy-type records per S1, US | 1× | ~1.9× |
| Decoy-type records per S1, India | 1× | ~1.65× |
| Decoy-type pairs lifted above 0.5 by stage 2 (vs stage 1) | no change | 2.3× more (6.9% vs 3.0%) |

The S1-level group features (`*_maxother_s1`, `p_rank_s1`, `s1_n_high`, `s1_sum_p`, support features) describe how an S1's candidates compete. They depend on how many decoys an S1 faces, which differs between train and test.

**Fix:** average the full stage-2 model with one trained without those features (`model.stage2_exclude`).

| | Leaderboard |
|---|---|
| v11 (full stage 2) | 0.9692 |
| v11b (without S1-level features) | 0.9721 |
| **Average** | **0.9733** |
| Average + France correction | 0.9737 |
| v12 average + France correction | 0.9744 |
| v13 + v13b + v12 + v12b average + France correction (final) | **0.97526** |
| final + stage 3 | 0.97347 |

---

## 16. Where the score is lost after blocking (v12)

**Method:** `scripts/loss_decomp.py <run> <blend_dir>` measures the loss in F0.5 points, not links. Per S1, loss = 1 − F. It is split into the part false positives cause and the part missed links cause; missed links are then added back one reason at a time.

v12 average: macro F0.5 0.9806, loss 0.0194.

| Cause | F0.5 lost | Links |
|---|---|---|
| True link below the threshold (0.65) | 0.0062 | 102,490 |
| Blocking miss | 0.0054 | 105,305 |
| Record given to another S1 by best-S1 | 0.0025 | 50,432 (88% have an empty address) |
| Per-S1 caps | 0.0002 | 3,115 |
| False positive: record of no S1 | 0.0033 | 23,727 pairs |
| False positive: record of an S1 outside the 81% train view | 0.0015 | (a sampling artefact) |
| False positive: record of another S1 | 0.0005 | |

- **Below-threshold misses:** 30% have an empty address and 29% a house number shifted by 1–20 (true matches that look exactly like decoys).
- **By S1:** 80% are perfect, 12% miss some links, 2% have a false positive, 0.4% have links but get an empty list, and 0.1% are singletons given a match.
- **Selection rules** (`scripts/select_exp.py`, 2-way cross-validation over S1s):

  | Rule | Validation |
  |---|---|
  | Global threshold 0.65 (current) | 0.97993 |
  | Expected F0.5 on raw p | 0.97991 |
  | Expected F0.5 on calibrated p | 0.98002 |
  | Thresholds per source × empty address × rank | 0.98016 |

  Selection is close to optimal for these probabilities: the loss is in the probabilities themselves.

---

## 17. Orphan records and stage 3

**Question:** 26% of S2/S3 records belong to no S1, and they cause most false positives. What separates them from true matches?

**Method:** `scripts/loss_examples.py` prints examples; `scripts/orphan_signals.py` measures candidate signals on selected pairs.

**Examples:**
- `pentagon warehousing | 29 firdous nest 1st floor ...`: a true record with a bare number added in front.
- `emelyne brown mustang | 327 ackiss avenue` for an S1 at 106 ackiss avenue.
- `marketing sunshine construction | 0026 c o ashok ...`

| Signal (on selected pairs) | True positives | Orphan false positives |
|---|---|---|
| Record address starts with a bare number the S1 doesn't have | 9.0% | 43.1% |
| S1 house number missing from the record's numbers | 7.6% | 41.4% |
| **Exact twin (same name + address) among the S1's candidates** | **21.3%** | **1.4%** |
| Exact twin in the same source | 17.7% | 1.0% |

An entity's true records often repeat each other exactly, while an orphan is a one-off perturbed copy.

**Stage 3** (`src/stage3.py`): LightGBM on the blend's logit plus, per pair:
- twin counts;
- shared number sets and addresses among the S1's other candidates, with their summed p;
- global duplicate count;
- the number relations above.

It uses 3 folds by S1.

| | Validation (test mix) |
|---|---|
| v12 average | 0.97993 |
| + stage 3 | **0.98104** |
| + stage 3 without the features that count other records | 0.98007 |

**Transfer check** (`scripts/compare_sel.py`, pairs changed per 1,000 S1s):

| | Train | Test |
|---|---|---|
| Added, US | 16.2 | 50.7 |
| Added, India | 20.3 | 33.2 |
| Removed, US | 19.5 | 20.8 |
| Removed, India | 24.7 | 32.7 |

In train, 85% of the added pairs are true. On test, the US additions are 3× the train rate: the same warning sign as section 15.

**Leaderboard:** on the final four-model average, stage 3 scored **0.97347 against 0.97526** without it, despite +0.0011 validation. The warning sign was right: counting twins and shared numbers among an S1's candidates depends on how many decoys the S1 faces. **Not used.**

---

## 18. Correcting an unseen country: leave-one-country-out simulation

**Question:** can the France correction (EM label shift, section 10) be made more accurate, and how strong should it be?

**Method:** `scripts/loco_sim.py <run> <train_country> <unseen_country>` makes one labelled country play France:
1. Train a stage-1 model on the other country only.
2. Predict the unseen country and tune the threshold on the training country.
3. Score each correction against the unseen country's labels.

| Correction | US → India (unseen prior lower) | India → US (unseen prior higher) |
|---|---|---|
| None | 0.94047 | 0.96383 |
| EM prior (estimate / truth) | 0.5167 / 0.5069 (train 0.5498) | 0.5538 / 0.5496 (train 0.5074) |
| EM global, strength 1 (current) | +0.0007 | −0.0002 |
| EM global, strength 2 | +0.0013 | −0.0008 |
| EM per stratum (source × empty), strength 1 | +0.0011 | −0.0004 |
| Threshold +0.1 | +0.0019 | −0.0007 |
| Self-training on pseudo-labels | −0.0016 (alone), +0.0005 (averaged) | |

**Findings:**
- EM estimates the direction and size of the prior shift well in both directions.
- Correcting helps only when the unseen country's prior is lower: F0.5 rewards removing doubtful matches, not adding them. The packaged `france_shift.py` therefore applies the correction only in that case. France qualifies: its estimated prior is 0.42 against 0.53 in train.
- Stronger or per-stratum versions gain another 0.0004–0.0012 on the unseen country. That is about +0.0001 on the leaderboard (France is 15% of S1s) and relies on extrapolating to a shift three times larger than the simulated one. Not adopted.
- The unseen-country penalty itself is large: India drops from about 0.978 to 0.940 when no Indian labels are used. That is the real France problem, and no correction without labels closed it.

---

## 19. Engineering incidents

| Incident | Cause | Fix |
|---|---|---|
| Laptop froze on EDA | Loading 200–500 MB TSVs with pandas on 8 GB RAM | Stream line by line or sample |
| First full run at 117 GB RAM | Forked workers shared big DataFrames; reference counting copied every touched page | Each task gets only the column slices it needs |
| Website regex split names into single characters | The pattern could match the empty string | Pattern made non-empty |
| Duplicate `bscore` column | Context features added it twice | Removed |
| Candidate ranker kept 27% of true links | Labels misaligned after sorting | Labels computed after the sort |
| Threshold tuning took 10 min | The metric merge was re-run for every grid point | Vectorised counts: seconds |
| Browser kept opening "Session token refreshed" tabs | Each monitoring poll opened a new SSM session | SSH ControlMaster (one shared connection, `ControlPersist 4h`) and slower polls |
| `pkill -f configs/v2.yaml` killed its own shell | The pattern matched the pkill command itself | Bracket trick: `pkill -f "[c]onfigs/v2.yaml"` |
| Monitor script broke in zsh | `status` is a read-only zsh variable | Renamed |
| Orphaned worker held 18 GB | Child of a killed run | Verified it was orphaned, then killed it |
| macOS rsync rejected `--info=progress2` | Old rsync on macOS | `--stats` |
| Remote disk filling up | ~5 GB per run of features, ~3 GB per data cache | Delete superseded runs' feature folders and caches |
| v13 and v12 together reached 102 of 124 GB | Two blocking stages at once | v13 queued behind v12 (`scripts/after_v12.sh`) |
| v13 killed in test blocking (`BrokenProcessPool`) | Analysis jobs (stage 3, simulations) running beside blocking | Resumed at test blocking with `BER_BLOCK_SPLITS=test`; nothing else runs beside blocking |

---

## 20. Lessons

1. **Measure the ceiling before optimising.** The v5b error analysis showed that even a perfect model couldn't reach the leaderboard leaders, which pointed straight at blocking.
2. **A what-if diagnostic beats a 3-hour run.** `diag_blocking.py` with extra key types predicted the v6 gain before any training.
3. **Anything learned from labelled countries must be checked on the unlabelled one.** The address map cost France about 0.02.
4. **Validation that can't see a slice will mislead you about that slice.** A probe submission is the cheapest way to measure it.
5. **Simulations must rebuild every derived feature.** Dropping rows after the features were computed gave a falsely reassuring answer (section 8).
6. **Confirm structural facts** (one S1 per record, no cross-country matches) **and exploit them exactly.** Best-S1 assignment is free precision.
7. **Watch how validation gains transfer.** From v5b onwards about 60–70% reached the leaderboard, so size expectations accordingly.
8. **Features that summarise an S1's competition depend on the decoy density.** When train and test densities differ, compare what a feature changes on train and on test before trusting it (sections 15 and 17).
9. **Count the loss in metric points, not links.** 2.8% of links lost after blocking is 0.014 of F0.5, spread over causes of very different sizes (section 16).
