# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Jerry
**Team Members:** Jitam Barman, Aditya Rewanshete
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The approach is a classic blocking-plus-classifier pipeline:
- multilingual text normalisation (9 Indic scripts, learned word maps);
- a two-pass inverted-index blocking stage, followed by a learned candidate ranker that keeps about **7.4 candidates per S1** on test (7.0 on train);
- a **two-stage LightGBM** pair model, four variants averaged;
- F0.5-optimal match-set selection that uses the one-S1-per-record structure of the data;
- a label-shift correction for France, the one country with no training data.

Three findings made the largest difference:
1. A second blocking pass with looser frequency caps for specific keys. Most missed matches were common business names whose only shared keys had been discarded as too frequent.
2. A France-safe guard on the learned address map.
3. Averaging a full stage-2 model with one that omits S1-level competition features. These features over-fire on the test set, which has about twice as many decoys per S1 as train.

Public leaderboard: **0.97526**.

---

## 2. Methodology

### 2.1 Problem Analysis

All EDA was done by streaming the TSVs or on samples. Key facts:

| Finding | Evidence | Used for |
|---|---|---|
| A record matches at most one S1; no match crosses countries | 0 exceptions in 7.6M train links | Best-S1 assignment; country-partitioned blocking |
| 50% of S1s share a normalised name with another S1 | Train S1 | Names alone can't decide; numbers and addresses must |
| Decoys copy an S1 with a house number shifted by 1–20 and often an extra name word; true matches drop or insert digits instead | 93.6% of US strong-name records with a 1–20 shift are non-matches | Number-difference features |
| Records with **no address number** are true matches 97% of the time | 97.7% for empty addresses, 73% overall | Explains the empty-address errors (namesakes) |
| Indic-script names: 23.5% of India S2, 13.2% of India S3, 9 scripts | Script detection | Transliteration, then a learned word map |
| The test has 5.75 records per S1 vs 4.68 in train, and **~1.9× (US) / 1.65× (India) more decoy-type records per S1** | Label-free comparison of train and test | Density-robust stage 2 (section 4) |
| France: 15% of test, absent from train | | Country-agnostic rules; checks on French data |

Noise seen:
- **Names:** legal suffixes moved, dropped or abbreviated; typos and OCR digits (`h0ldings`); website/hashtag forms (`sweetpilates.com`); transliterations (`praivet limited`); random aliases.
- **Addresses:** reordered components, abbreviations (US, Indian and French), missing parts or empty addresses, state/region vs department names, digits dropped from numbers.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate ranker + two-stage GBDT classifier + structured set selection (hybrid).
**Core Innovation:**
1. Diagnostic-driven blocking: a what-if tool measured which frequency caps discard true links, which led to a second, specific-key blocking pass.
2. A density-robust ensemble for the train/test decoy shift.
3. Label-free guards that keep rules learned on India/US from breaking France.

---

## 3. Candidate Generation (Blocking)

1. **Keys.** Every record emits hashed keys, each prefixed with its country:

   | Group | Examples |
   |---|---|
   | Name words, compact name, 7-character prefix | `nt|balaji`, `nc|sweetpilates` |
   | Phonetic consonant skeletons | `pt|psns` |
   | Name word pairs, phonetic pairs | `n2|cargo|elite` |
   | Address word bigrams | `ab|cranberry|highway` |
   | Any of the first 3 address numbers × address / name / phonetic words | `na|2739|cranberry` |
   | Name / phonetic words × address words | `nx|mountain|kniffin` |

2. **Pass 1:** all key types. Keys shared by more than 30 S1s or 400 S2/S3 records are dropped. Pair score = Σ 1 / log2(2 + key frequency) over shared keys; the top 120 per S1 are kept.
3. **Pass 2:** only the specific key types (full compact name, prefix, word pairs, number + word, address bigrams, name + address word), under looser caps of 200 / 3000 and the top 100 per S1. The most specific key shared by a missed true link had a median frequency of 42–45 S1s, just above the pass-1 cap. Pass 2 lifted recall from 97.15% to 98.30%, and India from 95.7% to 97.8%. The two passes are joined, with scores summed.
4. **More specific keys and ties (v13):**
   - the full compact name × each number and × each address word;
   - the house number with any one digit deleted × address words (true matches drop digits: 385 → 85);
   - candidates tied with the top-N-th score are kept (up to 240), so that ~190 records sharing only a common name are not cut arbitrarily.

   Loose recall 98.34% → 98.67%.
5. **Learned candidate ranker.** A LightGBM on cheap similarity features (name/phonetic/address token-set, compact-name Jaro-Winkler, number overlap, blocking-score context), trained out-of-fold on train. It keeps up to 12 candidates per S1, dropping any with ranker score below 0.001 (always at least 1). The ranker is sharply bimodal (median 0.9975 for true pairs, 0.00014 for others), so this cuts ~210 loose candidates per S1 to 7.0 (test 7.4) at a cost of 0.06 points of recall.

- **Blocking keys used:** as above; hashed, country-partitioned, two passes with frequency caps.
- **Candidate pairs generated:** about **12.8M test pairs, 7.4 per S1** (train 7.0), from ~374M loose pairs (~215 per S1).
- **How we ensured true matches were not lost:** measured blocking recall on train after every change: 92.9% (first version) → 94.6% → 95.7% → 96.05% → 97.15% → 97.98% → 98.30% → **98.61%** (US 98.8%, India 98.3%). `diag_blocking.py` and `cap_whatif.py` classify every missed link (all shared keys capped / cut by top-N / no shared key) and give a what-if for proposed keys and caps before each run.

---

## 4. Matching Model

**Features used** (about 55 per pair, computed with rapidfuzz):
- **Name features:** ratio, token-sort, token-set, partial ratio; Jaro-Winkler and partial ratio on the compact name; phonetic ratio and token-set; raw-name ratio; word Jaccard, containment, extra/missing words; IDF-weighted overlap.
- **Address features:** ratio, token-set, token-sort, partial ratio; alphabetic word Jaccard and containment.
- **Number features:** house-number equality; number-set equality, Jaccard, containment; log difference, substring (digits dropped), Levenshtein, cross-containment, minimum log difference.
- **Other:**
  - record flags (empty address, Indic name, website name, source);
  - blocking context (score, rank and relative score within the S1 and the record, candidate counts);
  - the candidate ranker's score.

**Text processing before features:**
- Transliteration of 9 Indic scripts (indic_transliteration, MIT), then unidecode.
- A transliteration dictionary learned from train links (686 words, e.g. `praivet → private`).
- An address word map learned from train links (1,188 words, e.g. `wb → west bengal`). It is applied per country only where the target word is more common in that country's S1 addresses; unguarded, it had rewritten 52% of French records (`de → des`, `saint → street`).
- French region/department names removed; dotted legal acronyms collapsed (`e.u.r.l.` → `eurl`).

**Model type:** LightGBM (MIT), two stages, 3 folds split by S1, so every training pair gets an out-of-fold score.
- **Stage 2** adds features built from stage-1 out-of-fold scores:
  - competition for the same record (best competing S1, head-to-head deltas);
  - competition within the S1;
  - support from the S1's strongest other candidates.
- **Final model = average of two stage-2 models:** one with all features, and one without the S1-level group features. On the test those features raise decoy-type pairs above p = 0.5 2.3× more often than stage 1 alone (6.9% vs 3.0%), while on train they don't. For v11 the average scored 0.9733 on the leaderboard, against 0.9692 (full) and 0.9721 (robust) on their own.
- **Final ensemble:** this pair is built for two blocking versions (v12 and v13) and all four models are averaged on v13's candidate set (+0.0005 validation over v13's pair alone, +0.0009 over v12's).

**Threshold selection method:**
1. **Best-S1 assignment:** each record keeps only its highest-scoring S1.
2. **Per-source caps:** at most 5 S2 and 6 S3 records per S1.
3. **Per-S1 set selection:** a global threshold (0.65) or expected-F0.5 sets (`1.25·Σ top-k p / (k + 0.25·Σ p)`, k = 0 allowed), whichever scores better on out-of-fold predictions re-weighted to the test's country mix. For the final blend it is the threshold; per-group thresholds and calibrated expected-F0.5 gained at most +0.0002 under cross-validation.
4. **France correction:** France is 15% of test and absent from train. Its candidate sets are larger (8.2 vs 6.3–6.9 per S1), so a smaller share of pairs is true, and the model is over-confident there. The French prior is estimated by EM on the French test predictions alone (Saerens et al., 2002): 0.42 against 0.53 in train. The French probabilities are re-weighted by the prior ratio before selection (+0.0004 on the leaderboard). A leave-one-country-out simulation, training on US only and predicting India, confirmed that this helps when the unseen country's prior is lower, and hurts when it is higher, so it is applied only in the first case.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** out-of-fold validation **0.9808** (test country mix; India 0.9792, US 0.9829). Public leaderboard **0.97526**.
- **Common false positives (wrong merges):**
  1. Sibling decoys with a small house-number shift that also carry name noise (`nova global (india) group pvt ltd`, plot 366 vs 365).
  2. Empty-address records given to the wrong one of several same-name S1s (39% of false positives belong to another S1).
  3. Singletons given a lookalike record.
- **Common false negatives (missed matches):**
  1. Blocking misses (37% of the remaining loss): common names with no usable number.
  2. Model misses (36%): half of them are **empty-address records**, true matches in 97% of cases, but they must be told apart from 2–4 same-name S1s; the top-scored S1 is correct only 41% of the time in those uncertain cases.
  3. Random alias names (`zetaevo`) at the same address.

Loss after blocking (v12 blend, validation 0.9806; total loss 0.0194 in F0.5 points):

| Cause | F0.5 lost |
|---|---|
| True matches scored below the threshold (30% empty address, 29% decoy-like house-number shifts) | 0.0062 |
| Blocking misses | 0.0054 |
| False positives: records that belong to no S1 (26% of S2/S3) | 0.0033 |
| Records given to another same-name S1 (88% empty address) | 0.0025 |
| False positives from the 81% train S1 sample (absent on test) | 0.0015 |
| Other false positives, per-S1 caps | 0.0007 |

**Tried and not used: stage 3.**
- The idea: 26% of S2/S3 records belong to no S1, and they cause most false positives. They are one-off perturbed copies, while an entity's true records often repeat each other exactly. 21% of true positives have an identical twin among their S1's candidates, against 1.4% of orphan false positives.
- The model: a third LightGBM on the blend's logit plus twin and shared-number-set counts.
- The result: validation +0.0011 (0.9819), but the leaderboard dropped (0.97347 vs 0.97526). On test it added 3× more US matches than on train. These counts depend on how many decoys an S1 faces, the same density shift that affected the S1-level stage-2 features.

Validation vs leaderboard: the gap was 0.004–0.013 over versions. A probe submission, with the French lists emptied, split it into France (≈ 0.94 at the time) and India/US transfer. The decoy-density analysis explained and halved the rest.

---

## 6. Conclusion

A carefully diagnosed classic pipeline reaches 0.981 validation F0.5 (0.9753 on the leaderboard) with a candidate set of about 7 per S1.
- **Biggest gains:** measuring where true links were lost (blocking caps), and checking every learned rule against the unlabelled country and the test's structure.
- **Largest remaining gap:** choosing among same-name S1s for records with no address; a stronger learned pair model (such as a fine-tuned multilingual cross-encoder) is the natural next step.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Path | Role |
|---|---|
| `src/run.py` | Entry point: `--config`, `--stage prepare block features train predict` |
| `src/blend.py` | Averages runs' probabilities (on the first run's candidates), re-tunes selection |
| `src/stage3.py` | Optional third model (off in the final; see section 5) |
| `src/france_shift.py` | EM label-shift correction for countries absent from train (France); writes `output/` |
| `src/ber/normalize.py` | Text normalisation, transliteration, learned word maps |
| `src/ber/blocking.py` | Inverted-index blocking (both passes) |
| `src/ber/candidates.py` | Learned candidate ranker (the pruned set = `candidate_pairs.tsv`) |
| `src/ber/features.py`, `src/ber/stage2.py` | Pair features; stage-2 competition features |
| `src/ber/model.py`, `src/ber/postprocess.py`, `src/ber/metrics.py` | LightGBM with S1-grouped folds; best-S1, caps, expected-F0.5 selection; exact metric |
| `configs/*.yaml` | One file per experiment, each inheriting from an earlier one; final: `v12`, `v12b`, `v13`, `v13b` |
| `reproduce.sh` | Full reproduction, data → `output/` |
| `requirements.txt` | Pinned dependencies (Python ≥ 3.11) |

**Reproduce:** put the data in `dataset/{train,test}/`, then from the project root run `bash code/business_entity_resolution/reproduce.sh`. It needs about 32 vCPU and 128 GB RAM, CPU only, about 12 h (two full pipelines).

**Licences:** LightGBM (MIT), rapidfuzz (MIT), indic_transliteration (MIT), pandas / numpy / scipy / pyarrow (BSD / Apache 2.0), PyYAML (MIT), Unidecode (GPL-2.0, text library). No pretrained or neural model is used.

### B. Additional Results

| Version | Change | Blocking recall | Validation | Leaderboard |
|---|---|---|---|---|
| v1 | Baseline | 94.6% | 0.9458 | 0.942 |
| v3 | Number-difference features + stage 2 | 95.7% | 0.9617 | 0.9578 |
| v4 | Learned transliteration | 96.05% | 0.9656 | 0.9606 |
| v6 | France-safe address map, name × address keys | 97.07% | 0.9762 | 0.963 |
| v8 | French cleanup, flexible candidate sets | 97.15% | 0.9758 | 0.9645 |
| v9 + v8 + v6 | Blend | 97.15% | 0.9774 | 0.9653 |
| v11 + v11b | Second blocking pass; density-robust average | 97.98% | 0.9790 | 0.9733 |
| + France correction | EM label shift on French predictions | 97.98% | 0.9790 | 0.9737 |
| v12 + v12b + France correction | Looser second pass (200 / 3000) | 98.30% | 0.9799 | 0.9744 |
| **v13 + v13b + v12 + v12b + France correction (final)** | More specific keys, ties at the cut; four-model average | 98.61% | 0.9808 | **0.97526** |
| final + stage 3 | Record-cluster features (twins, shared numbers) on top | 98.61% | 0.9819 | 0.97347 |

The full investigation log, with every analysis and its numbers, is in the repository (`doc/investigations.md`, `doc/architecture.md`).
