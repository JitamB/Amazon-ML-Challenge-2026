# Amazon ML Challenge 2026: Problem Analysis & Strategy

*Analysis date: 2026-09-25. All EDA was done by streaming the TSVs line by line or on samples of about 1.5k–4k S1 entities, with peak RAM under 1 GB. No file was loaded whole.*

---

## 1. The problem

This is entity resolution, not ordinary classification. For each S1 record we must return the set of S2/S3 records that are the same real business.

- **Metric:** F0.5 computed per S1 entity, then averaged over all S1 entities (macro).
  - A wrong match costs about twice as much as a missed match.
  - A singleton scores 1.0 if we return an empty list and 0.0 if we return anything.
- **Rules:**
  - The final model must be MIT or Apache-2.0 licensed, with at most 8B parameters.
  - No external lookups: no APIs, no geocoding, no outside data.
- **Deliverables:**
  - `matching_results.tsv`, which is scored.
  - `candidate_pairs.tsv`, the exact candidate set the model runs inference on. Matches must be a subset of it.
  - Code package and a methodology document.

### Dataset size

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2.21M (US 1.32M, India 0.88M) | 5.03M | 5.29M |
| Test | 1.73M (India 810k, US 663k, **France 259k**) | 4.89M | 5.08M |

The raw data is about 2.3 GB, and the ground truth contains 7.64M links.

---

## 2. Findings that shape the approach

1. **Each S2/S3 record matches at most one S1.** All 7.64M links point to distinct IDs. So each S2/S3 candidate can go only to its highest-scoring S1, which removes most wrong matches between neighbouring S1s. This is the most useful structural fact in the data.
2. **About 27% of S2/S3 records match no S1, and some are planted decoys.** Some decoys are near-copies that differ only in unit or house numbers:
   - `Torrent & Sons Pvt Ltd, #13/9 2nd Cross…` vs `Torrent & Sons Group Pvt Limited, #13/16 2ND CROSS…`
   - `Shiv Nursing Home, F-3/94 …` vs `F-3/98 …`
   - `Ganpati Lifestyle, Plot No 479/231` vs `479/236`

   Fuzzy string similarity can't separate these. We need features that compare the exact numbers in the address.
3. **Names are not identities.** 50% of S1 records share a normalised name with another S1. Name plus address is unique, and addresses are shared by 5.3% of S1. Name-only matching will produce false matches.
4. **S1 is clean; S2 and S3 carry the noise.**
   - S1 has no Indic script, no empty addresses and almost no PIN/ZIP codes.
   - S2 addresses are uppercase.
   - S3 uses Indic or full state names (`ಕರ್ನಾಟಕ`, `Keralam`) and `null` tokens.
5. **Indic-script names** make up 23.5% of India S2 names and 13.2% of India S3 names (test: 23.6% and 13.3%), and about 16% of India links in the sample.
   - **Nine scripts appear**, all counted by Unicode block on the full train and test files:

     | Script | Train S2 names |
     |---|---|
     | Devanagari | 269k |
     | Telugu | 39k |
     | Kannada | 37k |
     | Tamil | 34k |
     | Gujarati | 31k |
     | Bengali | 31k |
     | Malayalam | 19k |
     | **Oriya** | 7.5k |
     | Gurmukhi | 6.7k |

   - Indic script also appears in addresses, mostly state names; there are about 277k Devanagari addresses in train S2.
   - About 26k S2 names contain **zero-width joiner/non-joiner characters**. Strip them before transliterating.
   - Test addresses also contain `’` (right single quotation mark). Normalise it to `'`.
   - unidecode transliteration handles these scripts poorly.
6. **Alias names** appear in about 3% of Latin-script links, with almost no name overlap. Examples: `Arccalozeph`, `QUOFAYE`, `sweetpilates.com`, `#federalcascade`, `PEDIATRICDENTI5TRYCOASTAL.COM`. These can only be matched on address, or by splitting the website or hashtag back into words.
7. **Empty addresses** occur in 2–4% of S2/S3 records. These can only be matched on name, which is risky because of finding 3.
8. **Size of each match set:** 1–11 records, mean 3.5. There are at most 5 from S2 and at most 6 from S3, and 5.6% of S1 are singletons.
9. **France has no training data** and is 15% of test S1, so it is a large share of the test score.
   - French address abbreviations include `R.`, `AV`, `BD`, `CH`, `ALL`, `N°` and `bis`.
   - French legal words such as `SARL`, `SAS`, `EURL` and `SCI` are used as noise.
10. **No match ever crosses countries.** Of 342,252 links for 98,956 sampled S1s, 0 cross countries. Country is therefore a hard partition for blocking, features and the best-S1 rule, and the problem splits into three independent subproblems.
11. **The country mix shifts between train and test.** Train is US 60% / India 40%. Test S1 is India 46.8%, US 38.3%, France 15.0%. India is the harder country (baseline 0.65 vs US 0.76), so a validation score on the train mix overstates the leaderboard. Weight validation scores by the test mix, and treat France as an unknown.
12. **Other noise:** leading junk (`***`, `>>`, `--`), honorifics (`Shri`, `Dr`, `M/s`), filler words (`Center`, `Services`), words out of order, typos and OCR-style swaps (`0`↔`O`, `5`↔`S`, `l`↔`I`), and "née" / "D.B.A." / "a/k/a" alias patterns.

### Per-entity score formula

With G = true matches, P = predicted matches and TP = correct predictions, F0.5 = 1.25·TP / (P + 0.25·G). This follows from 1.25·TP / (1.25·TP + 0.25·FN + FP).
- **Examples:** G=2, P=3, TP=2 gives 0.714. G=3, P=2, TP=2 gives 0.909.
- **Singletons:** 1.0 for an empty prediction, 0.0 otherwise.
- **Use:** plug this formula into the expected-F0.5 set selection (Section 5, #6).

### Match-count distribution (train)

| # matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| S1 count | 123k | 119k | 375k | 531k | 484k | 322k | 165k | 64k | 23k |

---

## 3. Measured baseline

This is a rule-based baseline on 3,945 sampled S1 entities.

- **Candidates:** S2/S3 records that share the house number or the exact normalised name with the S1.
- **Rule:** name similarity ≥ 85 and address similarity ≥ 80, using token-set similarity.

| Variant | Macro F0.5 | Precision | Recall | Singletons correct |
|---|---|---|---|---|
| Similarity thresholds only | 0.679 | 0.58 | 0.70 | 43% |
| + address-number check | **0.710** | 0.61 | 0.67 | 69% |

- **Blocking recall is only 87%,** with about 18k candidates per S1. Both recall and efficiency are poor.
- **By country:** US scores 0.76 and India 0.65.
- **Precision is pessimistic.** Other S1s competing for the same records were not in the sample, so the best-S1 rule from finding 1 could not be applied.

### How well true matches and non-matches separate

Among pairs that share a house number:
- **True matches** fall mostly in the region where both name and address similarity are at least 80.
- **Non-matches** in that same region are rare but real. They are the sibling decoys, with about 60 non-matches versus about 2,300 matches in the top cell.
- **Matches with low name similarity** (below 50) usually still have high address similarity. These are the alias names and Indic-script names.

---

## 4. Expected scores

| Stage | Expected macro F0.5 |
|---|---|
| Rules (measured) | ~0.71 |
| Better blocking + LightGBM on similarity/number features + best-S1 rule | 0.88–0.91 |
| + transliteration, alias/website handling, agreement between candidates, threshold tuned for F0.5 | 0.92–0.94 |
| + small fine-tuned multilingual model, ensemble, France-specific work | 0.94–0.96 |
| Realistic ceiling | ~0.97 |

Rows after the first are estimates from the error patterns, not measurements.

What limits the ceiling:
- An alias name combined with an empty address leaves no signal at all.
- Some decoys have had their distinguishing numbers removed by the noise.
- Singletons that have a decoy nearby are easy to get wrong.
- France may perform worse than validation suggests, because we cannot measure it. This is the biggest leaderboard risk.

---

## 5. Approaches, ranked by expected impact

| # | Approach | Why it fits this data | Impact | Cost | Overfit/leak risk | How to validate |
|---|---|---|---|---|---|---|
| 1 | **Several blocking passes combined**, all within a country: house number + street token; rare name tokens / TF-IDF char n-grams; city + name prefix; transliterated-name keys; embedding nearest-neighbour search. Target ≥98% recall with ≤100 candidates per S1 | Blocking recall caps the final score, and it is 87% today | +8–12 pts | CPU-heavy | Low | Recall and candidates per S1 on held-out S1s, broken down by country and script |
| 2 | **Give each S2/S3 record only to its best S1**, then cap S2 ≤5 and S3 ≤6 per S1. A greedy argmax per S2/S3 record is exact here, because only the S2/S3 side is limited to one link; Hungarian assignment is not needed | Exact rule found in the data (finding 1) | +3–6 pts of precision | Trivial | None; it is a real property of the data | Score with and without it |
| 3 | **LightGBM/CatBoost pair classifier** with number-aware features: exact house-number match, overlap of all numbers and unit tokens, token-order-independent name similarity, Jaro-Winkler, name/website with spaces removed, abbreviation-expanded address similarity, token rarity (IDF), empty-field flags, candidate rank within the S1 block, score margin to the next-best S1 for the same candidate, S2-vs-S3 source flag, and legal suffix match/conflict (US, India and France forms) | Needed to beat the decoys (finding 2) | Core model, → ~0.89 | CPU, ~1–3h on full data | Medium. Negatives must come from blocking, not random pairs | 5-fold split by S1, keeping all S2/S3 in the pool so decoys still compete |
| 4 | **Transliteration dictionary learned from training data.** Train has millions of aligned pairs such as `Balaji Business Pvt Ltd` ↔ `ಬಾಲಾಜಿ ಬಿಸಿನೆಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | Fixes the ~16% of India links in Indic script | +2–3 pts on India | Low | Low | India score split by script |
| 5 | **Agreement between candidates:** use S2↔S3 similarity inside a block (e.g. how many strong matches share this address) | Rescues alias and empty-field records that sit next to confident matches | +1–2 pts | Medium | Medium; compute strictly out-of-fold | Ablation on the alias subset |
| 6 | **Choose each S1's match set to maximise expected F0.5**, including the empty-set option, from calibrated probabilities | Scores are averaged per entity and singletons earn full credit | +1–2 pts | Trivial | Low | Out-of-fold scores |
| 7 | **Fine-tuned multilingual model** (e5-small / xlm-roberta / mdeberta, MIT) for pair embeddings or re-ranking | Handles scripts and French better than hand-built features | +1–2 pts, stacked on top of #3 | GPU | Low | Stack its score as a feature and compare |
| 8 | **France without labels:** generic features only (no country one-hot), a French abbreviation table (`R.`/`AV`/`BD`/`CH`/`ALL`, `Z.I.` = zone industrielle, `C.C.` = centre commercial, `L-D` = lieu-dit, `N°`, `bis`), French legal forms (`SARL`, `SAS`, `SASU`, `EURL`, `SCI`, `SELARL`, `EIRL`, `SNC`, `SA`), then pseudo-label confident France test pairs and retrain | Otherwise 15% of test runs on a model that has never seen French | Protects several points | Low | Medium: pseudo-label errors reinforce themselves. Keep only very confident pairs | Hold out US or India entirely and train on the other to see how well a country transfers; LLM spot check (Section 6) |
| 9 | Ensemble / stacking of GBDT seeds and the transformer model | Standard | +0.3–1 pt | Medium | Low | Out-of-fold |

---

## 6. Using an LLM to check our predictions

- **Training labels beat LLM labels.** We have 7.6M exact labels from the same generator, so on US and India a held-out training split is the real ground truth.
- **The LLM is most useful for France**, where we have no labels. The plan:
  1. Run Qwen2.5-7B-Instruct (Apache 2.0, within the 8B limit) on about 2–5k France pairs sampled where our model is least sure.
  2. Compare its answers with our model and group the disagreements, for example by French abbreviations or legal-form noise.
  3. Fix those groups as features. This is compliant, and the same model could even be a final re-ranker for borderline pairs only.
- **Check the license of any model in the final pipeline.** Llama-3.1-8B uses the Llama Community License, not MIT/Apache, so it is fine as an offline diagnostic but not in the final pipeline. Qwen2.5-7B-Instruct is Apache 2.0.
- **Don't send test data to Claude or any other closed API.** That risks violating the "no external services to resolve entities" rule, and top teams' code is audited. Using a closed LLM on training data only, to study error types, is fine.
- **Don't treat LLM output as ground truth.** It is a noisy second opinion. Its value is in finding systematic weaknesses, not in providing labels to copy.

---

## 7. Compute requirements

### Local machine (M2, 8 GB RAM, a few GB free disk)
- The Mac cannot do full runs.
- **Plan for 100–150M test candidate pairs.** At about 60 float32 features each that is roughly 25–35 GB, and training is about the same size.
- Use the Mac only for prototyping on about 100k S1s plus their blocks, using streaming or chunked processing.

### SageMaker

| Job | Instance | Why | Time |
|---|---|---|---|
| Blocking, features, GBDT | `ml.r6i.8xlarge` (32 vCPU, 256 GB) or `ml.m6i.16xlarge`, with 200 GB EBS | Whole feature set fits in RAM, matching runs in parallel with rapidfuzz, one model per country | Features ~2–4h per split, LightGBM ~1–2h |
| Embeddings + nearest-neighbour search | `ml.g5.2xlarge` (A10G 24 GB) | ~22M records with e5-small ≈ 1.5–2h. Search index built per country (IVF-PQ / HNSW, ~15 GB in float32) | ~2–3h |
| Cross-encoder fine-tune + inference | `ml.g5.2xlarge` | ~10M training pairs, one epoch ≈ 2–3h. Re-rank only the top ~5 per S1 (~9M pairs) ≈ 1–1.5h | ~4h |
| 7B LLM (diagnostic or re-ranker) | `ml.g5.2xlarge` bf16 with vLLM, or `ml.g5.12xlarge` for throughput | Only for 5k–300k borderline pairs; running every pair isn't feasible | 1–3h |

Store data as Parquet in S3; it is about a third the size of the TSVs.

---

## 8. Recommended order

1. **Validation harness.** 5-fold split by S1, keep all S2/S3 in the pool, and use an exact copy of the macro F0.5 scorer. Report scores per country and as a mix weighted to the test proportions.
   - **Local prototype sample:** take whole regions, for example all S1 in a set of states or cities plus every S2/S3 record in those regions.
     - Don't take random S1s plus a hand-picked set of decoys. That removes the real competition between S1s and overstates precision.
     - Region sampling keeps the best-S1 rule and decoy density realistic.
   - Check: the harness reproduces the ~0.71 rule baseline.
2. **Multi-pass blocking.**
   - Check: at least 98% recall with at most 100 candidates per S1, reported per country and script.
3. **GBDT, the best-S1 rule and F0.5-optimal set selection.**
   - Check: at least 0.90 out-of-fold.
   - Submit this version to get a real number for France.
4. **Transliteration dictionary, alias/website handling and agreement features.**
   - Check: the gain from each one, measured separately.
5. **Transformer re-ranker, France pseudo-labels and the LLM diagnostic on France.**
6. **Ensemble, then generate `candidate_pairs.tsv` and `matching_results.tsv`.**
   - Check: both files pass `utils/validate_submission.py`.
