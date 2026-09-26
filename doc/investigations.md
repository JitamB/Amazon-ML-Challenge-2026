# Investigations Log

*Every analysis run during the challenge (25–26 Sep 2026): the question, how it was answered, the numbers, and what we did about it. Negative results are included; they saved as much time as the positive ones.*

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
14. [Engineering incidents](#14-engineering-incidents)
15. [Lessons](#15-lessons)

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

## 14. Engineering incidents

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

---

## 15. Lessons

1. **Measure the ceiling before optimising.** The v5b error analysis showed that even a perfect model couldn't reach the leaderboard leaders, which pointed straight at blocking.
2. **A what-if diagnostic beats a 3-hour run.** `diag_blocking.py` with extra key types predicted the v6 gain before any training.
3. **Anything learned from labelled countries must be checked on the unlabelled one.** The address map cost France about 0.02.
4. **Validation that can't see a slice will mislead you about that slice.** A probe submission is the cheapest way to measure it.
5. **Simulations must rebuild every derived feature.** Dropping rows after the features were computed gave a falsely reassuring answer (section 8).
6. **Confirm structural facts** (one S1 per record, no cross-country matches) **and exploit them exactly.** Best-S1 assignment is free precision.
7. **Watch how validation gains transfer.** From v5b onwards about 60–70% reached the leaderboard, so size expectations accordingly.
