# Ablation results: margin loss + severity-weighted CE (kernel 03)

Raw data: `results/kaggle_run_3_ablation/`. Paired statistical analysis
(McNemar exact test, per-image): `scripts/analyze_ablation.py`.

## Summary on RxHandBD (1115 test images — dataset with sufficient sample size for evaluation)

Compared to CE-only (baseline that confirmed the original phenomenon, confusable_wrong_drug=83/1115=7.44%):

| Configuration | confusable_wrong_drug | fixed / newly (paired) | p (McNemar) | correct | hallucination |
|---|---|---|---|---|---|
| **margin** | 74 (6.64%) | 22 fixed / 13 newly | 0.18 (not yet significant) | 510 (no significant change, p=0.39) | 257 (no significant change, p=0.11) |
| severity | 88 (7.89%) | 13 fixed / 18 newly | 0.47 | 522 | 234 |
| severity+margin | 83 (7.44%) | 15 fixed / 15 newly | 1.00 (unchanged) | 513 | 245 |

**Core safety metric** (images that were CORRECT under CE-only turned into NEW confusable_wrong_drug under the intervention configuration — exactly the type of error the project aims to avoid causing): **0.58% (3/519)** across all 3 configurations — low and unchanged, with no sign that the interventions increase this risk relative to each other.

## Honest conclusions (not yet statistically significant, but with a clear direction)

1. **Margin loss (alone) is the only direction showing a positive signal**: a net reduction of 9 confusable_wrong_drug images (22 fixed / 13 newly introduced), without a significant decrease in correct or a significant increase in hallucination — but **p=0.18, not reaching the statistical significance threshold** at this sample size with the default hyperparameters (β=1.0 not used here, margin=1.0, λ=1.0).
2. **Severity-weighted CE alone does NOT help** — it even shows a slight trend in the opposite direction (83→88, p=0.47). This is an important negative finding: merely increasing the loss weight for the high-risk class ("trying harder") does not automatically teach the model to avoid any specific wrong answer — consistent with the concern raised in the proposal.
3. **Combining both does NOT compound the benefit** — it returns to roughly the baseline (83→83, p=1.00); severity appears to add noise to margin's targeted signal.
4. **A few genuine overcorrection cases were observed** (exactly as the project initially worried): e.g. `Dexter` (correct, CE-only) → `Dextal` (confusable, margin); `Raditil` (correct) → `radifil` (confusable); `Fylox` (correct) → `eylox` (confusable). This rate is low (0.58%) but real.
5. **A notable positive qualitative finding**: in many cases margin loss "fixes" a confusable_wrong_drug error NOT by guessing correctly, but by pushing the prediction toward an OBVIOUSLY WRONG form (hallucination/minor_noise) instead of ANOTHER REAL DRUG NAME — e.g. `Vonocab`→`Vonorab` (confusable, CE-only)→`vonoeab` (obviously wrong, margin); `Bislol`→`Biscor` (confusable)→`Psistor` (obviously wrong). In terms of clinical safety, an "obviously wrong" output is much easier for a pharmacist to catch than "a valid-looking prescription for the wrong drug" — this could be a real benefit of margin loss that is not fully captured by the raw confusable_wrong_drug figure.

## Kaggle-BD (780 images, 78 classes)

Only 3-4 confusable_wrong_drug images across all configurations — too few for meaningful statistical analysis; used as a qualitative reference, not primary evidence.

## Limitations (noted in the first run — see hyperparameter sweep results below, since resolved)

- The hyperparameters (β, margin, λ) above are from the first attempt and have not been swept.
- Should try removing severity from the combined configuration, or sharply reducing β when combined with margin.
- Should additionally measure the metric of "transition from confusable_wrong_drug to an OBVIOUSLY WRONG error".

---

# Update: margin loss hyperparameter sweep (kernel 04) — STATISTICAL SIGNIFICANCE REACHED

Raw data: `results/kaggle_run_4_margin_sweep/`. Run only on RxHandBD (Kaggle-BD
was already confirmed to have an insufficient sample size). Compared to the same CE-only baseline (confusable_wrong_drug=83/1115=7.44%,
correct=519/1115=46.5%, hallucination=241/1115=21.6%).

| Configuration | confusable_wrong_drug | p (McNemar) | correct | p | hallucination | p |
|---|---|---|---|---|---|---|
| margin λ=2.0 | 67 (6.01%) | **0.009** ✅ | 491 (44.0%) | **0.014** ⚠️ significant decrease | 271 (24.3%) | **0.0035** ⚠️ significant increase |
| margin λ=3.0 | 59 (5.29%) | **0.0001** ✅✅ | 495 (44.4%) | **0.021** ⚠️ significant decrease | 273 (24.5%) | **0.0022** ⚠️ significant increase |
| margin λ=2.0, 5 epochs | 73 (6.55%) | 0.14 (not significant) | 527 (47.3%) | 0.49 | 245 (22.0%) | 0.75 |
| **margin λ=2.0 + severity β=0.3** | **67 (6.01%)** | **0.011** ✅ | **523 (46.9%)** | **0.76 (no significant change)** ✅ | **250 (22.4%)** | **0.41 (no significant change)** ✅ |

## Main conclusion: the winning configuration is **margin (λ=2.0) + light severity (β=0.3)**

- **λ=2.0 and λ=3.0 alone**: the reduction in confusable_wrong_drug REACHES clear STATISTICAL SIGNIFICANCE (p=0.009 and p=0.0001) — but at a real cost: correct decreases significantly, hallucination increases significantly. A genuine trade-off, not free.
- **Training longer (5 epochs) loses the signal**: no configuration reaches statistical significance — it appears that training too long lets CE dominate over margin.
- **Combining margin λ=2.0 with severity β=0.3 (sharply reduced from 1.0, not removed entirely) is the best result**: it reduces confusable_wrong_drug with statistical significance (p=0.011, comparable to λ=2 alone) **BUT does not significantly decrease correct (p=0.76) and does not significantly increase hallucination (p=0.41)** — meaning the core safety goal is achieved WITHOUT trading off overall performance. In addition, of the 26 images that stopped being confusable, as many as 13 (50%) were fully corrected — a much better rate than margin alone (only 5-7/25-31 ≈ 20-23%).
- The secondary safety metric (correct→newly confusable) remains low across all sweep configurations (0.39-0.77%).

**→ Configuration proposed for the final report: margin loss λ=2.0 combined with severity-weighted CE at β=0.3 (sharply reduced, not removed entirely), 3 epochs.** This is solid statistical evidence for the project's main argument: the most dangerous type of LASA error can be significantly reduced without trading off overall accuracy, provided the right combination and hyperparameters are chosen.

---

# Update: cross-checking the winning configuration on Kaggle-BD (kernel 05)

Raw data: `results/kaggle_run_5_winning_kaggle_bd/`. Compared to the CE-only baseline on Kaggle-BD (confusable_wrong_drug=3/780=0.38%, correct=684/780=87.7%, hallucination=44/780=5.6%).

| Metric | CE-only | Winning configuration | Paired | p (McNemar) |
|---|---|---|---|---|
| confusable_wrong_drug | 3 (0.38%) | 4 (0.51%) | 1 fixed / 2 newly | 1.00 (completely uninformative — n too small) |
| correct | 684 (87.7%) | 671 (86.0%) | 24 lost / 11 gained | **0.041 (significant decrease)** ⚠️ |
| hallucination | 44 (5.6%) | 51 (6.5%) | 11 fixed / 18 newly | 0.26 (not significant) |

**Honest conclusion: the result does NOT replicate on Kaggle-BD.** As predicted, the confusable_wrong_drug sample size (only 3-4 images) is completely insufficient to say anything about the target metric (p=1.00). More notably: on this dataset, the winning configuration actually **decreases correct rate with statistical significance** (87.7%→86.0%, p=0.041) with no measurable offsetting benefit. The most plausible interpretation: the benefit of margin+light severity may depend on **vocabulary size/diversity** — on RxHandBD (~1430 words, many genuine confusion pairs) the intervention has room to take effect; on Kaggle-BD (a closed vocabulary of only 78 classes, with only 3-4 images of the exact target error type) the intervention mostly just adds noise to a problem that is already near saturation, without enough "room" to produce a clear benefit.

---

# MOST IMPORTANT update: reconfirmation with 5 full seeds (kernel 06) — REVISING the "no trade-off" conclusion

Raw data: `results/kaggle_run_6_confirmatory/`. Analysis: `scripts/analyze_confirmatory.py`.
Context: a review round found that the "winning configuration" (p=0.011) does not survive
Holm-Bonferroni correction (p_holm=0.057, see `scripts/multiple_comparisons.py`)
because it was selected after looking at 7 configurations on the same test set, and each
prior configuration was run only **once, without full seeding** (only `random.seed(42)` for
Python, with torch/numpy not seeded). Kernel 06 reruns with **5 independent seeds** (torch+numpy+random
all set), training PAIRS (CE-only, winning configuration) with the SAME seed for each seed to
isolate the confounding variable of initialization/data ordering, comparing paired results
separately for each seed.

## Per-seed results (CE-only vs winning configuration)

| Seed | confusable_wrong_drug | p | correct | p | hallucination | p |
|---|---|---|---|---|---|---|
| 1 | 79→65 (decrease) | 0.049 ✅ | 522→495 (decrease) | 0.011 ⚠️ | 242→268 (increase) | 0.018 ⚠️ |
| 2 | 74→68 (decrease) | 0.362 | 524→508 (decrease) | 0.101 | 238→275 (increase) | 0.0002 ⚠️ |
| 3 | 90→79 (decrease) | 0.043 ✅ | 513→503 (decrease) | 0.368 | 238→254 (increase) | 0.109 |
| 4 | 78→66 (decrease) | 0.036 ✅ | 526→498 (decrease) | 0.010 ⚠️ | 235→255 (increase) | 0.045 ⚠️ |
| 5 | 78→71 (decrease) | 0.265 | 529→498 (decrease) | 0.005 ⚠️ | 243→264 (increase) | 0.033 ⚠️ |

**Key point: all 3 metrics move in THE SAME DIRECTION across ALL 5/5 SEEDS** —
confusable_wrong_drug always decreases, correct always decreases, hallucination always
increases — with no exceptions. This is very strong consistent evidence, not chance.

## Pooling 5 seeds (pooled McNemar, ~5x increase in statistical power)

| Metric | Fixed / Newly introduced (pooled) | Pooled p | Number of individual seeds reaching p<0.05 |
|---|---|---|---|
| confusable_wrong_drug | 103 / 53 | **p = 0.000077** | 3/5 |
| correct | 312 lost / 200 gained | **p = 0.000001** | 3/5 |
| hallucination | 178 fixed / 298 newly introduced | **p < 0.000001** | 4/5 |

Converting to average rates across 5 seeds: confusable_wrong_drug **7.16%→6.26%**
(a relative decrease of ~12.6%), correct **46.9%→44.9%** (a decrease of ~2.0 percentage
points), hallucination **21.5%→23.6%** (an increase of ~2.2 percentage points).

## The previous conclusion MUST BE REVISED

The old conclusion ("significant decrease in confusable_wrong_drug WITHOUT significant
trade-off in correct/hallucination", based on a single run with p=0.76/p=0.41) is **WRONG,
or at least not representative** — that single run was a **fortunate outlier by chance**,
exactly matching the review round's concern about "only 1 run, without seeding". With
5 independent seeds, correct rate DECREASES and hallucination INCREASES **consistently and
with very strong statistical significance** (pooled p < 0.00001 for both) — this is
**a genuine trade-off, not a free lunch**.

**Good news**: the core benefit itself (reducing confusable_wrong_drug — the most
dangerous error type) is confirmed **MUCH MORE ROBUSTLY** than before (pooled p=0.000077,
based on 5 independent replicates, instead of a single run with p=0.011 that was easy to
suspect as luck).

## Correct statement for the paper (fully replacing the old "no trade-off" statement)

> Margin loss (λ=2.0) combined with light severity-weighting (β=0.3) produces a
> statistically significant, consistent reduction across multiple independent runs in the
> rate of "confused with a different real drug" errors (confusable_wrong_drug, ~12.6%
> relative) — but this is A GENUINE TRADE-OFF: it comes with a ~2 percentage point
> decrease in correct rate and a ~2 percentage point increase in hallucination, both with
> strong statistical significance across 5 independent seeds. This is a point on the
> safety-performance trade-off curve, not a free improvement.

This actually makes the paper **more honest and more credible**, true to the project's
spirit: not "we solved the problem at no cost," but rather "we precisely quantify the
trade-off between safety and performance, and provide a tool for users to choose the
operating point that matches their acceptable level of risk" (exactly matching the earlier
review round's suggestion to "present the full trade-off curve").

**Implication for the final report**: it must be stated clearly that this is a real limitation of the method — the effect is demonstrated with statistical significance on RxHandBD (larger, more realistic vocabulary), but is NOT established (and shows an opposite signal on correct rate) on the small closed-vocabulary dataset. The winning configuration should not be presented as a general solution across all vocabulary sizes.

---

# Extended analysis on kernel 06 data (effect size + CI, no additional Kaggle runs needed)

Script: `scripts/analyze_confirmatory_extended.py`. Supplements the section above (which
only had p-values) with effect size (risk difference) + 95% confidence intervals, and digs
deeper into the breakdown of "where do fixed confusable errors end up," pooled across all 5
seeds (a much larger n than the single kernel 03 run).

## Effect size + 95% CI (pooled across 5 seeds, n=5575 image pairs)

| Metric | Risk difference (winning configuration − CE-only) | 95% CI |
|---|---|---|
| confusable_wrong_drug | **−0.90 percentage points** | [−1.34, −0.46] |
| correct | **−2.01 percentage points** | [−2.80, −1.22] |
| hallucination_far_off | **+2.15 percentage points** | [+1.39, +2.92] |

All 3 confidence intervals **exclude 0** — consistent with the previously reported
p-values, but now with a concrete effect size + uncertainty, matching the reporting
standard that Q2 journal reviewers typically require (not just p-values).

## Where do "fixed" confusable_wrong_drug errors go? (pooled across 5 seeds, n=399 confusable errors at baseline)

| | Count | % of "no longer confusable" |
|---|---|---|
| Total confusable errors under CE-only (pooled across 5 seeds) | 399 | — |
| No longer confusable under the winning configuration | 103 (25.8% of 399) | 100% |
| — of which fully corrected (correct) | 36 | 35.0% |
| — of which became OBVIOUSLY WRONG (minor_ocr_noise 37 + hallucination 30) | 67 | **65.0%** |

**Clinical significance**: when the intervention "fixes" a dangerous error (confusion
with another real drug name), the majority (65%, n=103, much more reliable than the small
qualitative observations in kernel 03) is not because the model guesses correctly, but
because it pushes the prediction toward an OBVIOUSLY WRONG error — much easier for a
pharmacist/doctor to catch than "a valid-looking prescription for the wrong drug." This is
a real safety benefit that the raw confusable_wrong_drug number does not fully capture, and
it has now been robustly quantified on a sufficiently large n.

## "Overcorrection" rate (correct under CE-only → NEWLY confusable under the winning configuration), pooled across 5 seeds

**16/2614 = 0.61%** (Wilson 95% CI: [0.38%, 0.99%]).

Compared to the overall decrease in correct rate (~2 percentage points, see above), the
share that "overcorrected" into precisely the DANGEROUS ERROR TYPE the intervention is
trying to avoid is only a very small fraction (0.61% of all images that were originally
correct) — most of the remaining "lost correct" instead converts to minor_ocr_noise/
hallucination (less dangerous), not new confusable errors. This point should be included
in the Discussion section to clarify the nature of the trade-off: the price paid (losing
~2pp of correct) is mostly NOT in the most dangerous form.

**Conclusion**: these 2 additional analyses do not change the core conclusion (a
genuine trade-off, confirmed via multi-seed) but make the paper much richer in
quantitative information — providing effect size + CI for the Results section, and a new
Discussion argument ("the price paid is mostly not the most dangerous type, and when a
dangerous error is fixed, it is mostly pushed toward a more easily detectable error rather
than a correct guess") that makes the trade-off narrative more balanced and persuasive,
rather than just stating raw increase/decrease numbers.