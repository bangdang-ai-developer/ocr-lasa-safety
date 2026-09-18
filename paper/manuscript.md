<!-- EDITORIAL NOTE (assembly pass): sections drafted in parallel from one shared,
verified facts sheet (numbers/citations only from project docs 00-04 and
scripts/analyze_confirmatory_extended.py); no numeric or framing inconsistency was
found across drafts. Table numbers in the Results section were renumbered to run
continuously after Methods' Table 1-3. Outstanding [TODO: confirm] items before
submission: (1) optimizer name for LoRA fine-tuning (facts sheet did not record it -
check kernel scripts, likely AdamW per torch.optim.AdamW in the code); (2) manual
verification of Millan-Hernandez et al. (2019)'s exact benchmark figures before
quoting any number from it; (3) a general LASA/medication-error epidemiology
citation if reviewers expect one beyond the algorithmic-identification literature;
(4) citation for the prior motivating project, once that manuscript is public; (5)
final code repository URL.
UPDATE (2026-09-18): all 7 Holm-Bonferroni-adjusted p-values are now filled in
(Table 6 in Methods, Table 7 in Results), re-run directly from
scripts/multiple_comparisons.py: severity-only p_holm=0.946; margin-only (lam1)
p_holm=0.573; severity+margin combined p_holm=1.000; margin lam2 p_holm=0.054;
margin lam3 p_holm=0.0008 (only one that survives at alpha=0.05); margin lam2
5-epoch p_holm=0.573; margin lam2+severity beta0.3 (selected) p_holm=0.057. -->

# A Quantified Safety–Accuracy Trade-off: Margin-Ranking and Severity-Weighted LoRA Fine-Tuning for Look-Alike/Sound-Alike Drug Name Errors in Handwritten Prescription OCR

<!-- Candidate titles considered (selected #1 above):
1. A Quantified Safety–Accuracy Trade-off: Margin-Ranking and Severity-Weighted LoRA Fine-Tuning for Look-Alike/Sound-Alike Drug Name Errors in Handwritten Prescription OCR
2. Fewer Dangerous Errors, Fewer Correct Reads: A Confirmatory, Multi-Seed Study of the Safety–Accuracy Trade-off in LoRA-Fine-Tuned Handwritten Prescription OCR
3. Trading Accuracy for Safety in Handwritten Prescription OCR: A Margin- and Severity-Weighted Loss for Look-Alike/Sound-Alike Drug Names
4. The Cost of Catching Confusable Drug Names: An Exploratory–Confirmatory Study of LoRA Fine-Tuning Trade-offs in Handwritten Prescription OCR
5. A Margin-Ranking and Severity-Weighted Loss for Look-Alike/Sound-Alike Drug Names: Quantifying the Safety–Accuracy Trade-off in LoRA-Fine-Tuned Handwritten Prescription OCR
-->

## Abstract

Low-Rank Adaptation (LoRA) fine-tuning of transformer-based optical character recognition (OCR) models improves overall handwriting-recognition accuracy, but a related, independent study found that fine-tuning a sequence-to-sequence OCR model on handwritten medical text can simultaneously increase a specific, clinically dangerous error type: misreading one real drug name as a different real drug name, a look-alike/sound-alike (LASA) confusion that a pharmacist can mistake for a valid prescription. We independently reproduced this trade-off on two new public datasets (Kaggle-BD, 78 drug classes; RxHandBD, a larger and more LASA-rich vocabulary) with zero-shot and LoRA-fine-tuned TrOCR, and we derived confusable drug-name candidates directly from each dataset's own vocabulary with a lexicon-free edit-distance/phonetic similarity score (BI-SIM), avoiding any curated clinical LASA list. We used these candidates to train LoRA with a combined margin-ranking and severity-weighted cross-entropy loss, evaluated in a two-phase design: an exploratory, single-run hyperparameter sweep that selected one configuration, followed by a confirmatory replication of that configuration across five independently seeded training runs, each paired against a matched cross-entropy-only baseline. On RxHandBD, the confirmatory replication (pooled paired McNemar tests, n = 5,575) shows that the selected loss reduces the LASA (confusable_wrong_drug) error rate by 0.90 percentage points (95% CI [-1.34, -0.46]; ~12.6% relative), but at a statistically robust cost: overall correct-prediction rate falls by 2.01 points (95% CI [-2.80, -1.22]) and hallucination-type errors rise by 2.15 points (95% CI [1.39, 2.92]), with all three metrics moving in the same direction in all 5 of 5 seeds. A destination-of-errors analysis shows most resolved LASA errors become clearly wrong rather than correct, and few correct predictions turn newly dangerous. This intervention did not replicate on the smaller, closed-vocabulary Kaggle-BD dataset. We therefore present this loss combination as a quantified, disclosed operating point on a safety–accuracy trade-off curve for large, LASA-rich prescription vocabularies, not as a cost-free improvement.

## Keywords

optical character recognition; handwritten prescription recognition; LoRA (Low-Rank Adaptation) fine-tuning; look-alike/sound-alike (LASA) drug names; medication safety; margin-ranking loss; severity-weighted loss; safety-accuracy trade-off


---

## 1. Introduction

Handwritten medical prescriptions remain common in many health systems, and automatic transcription of them — optical character recognition (OCR) applied to clinical handwriting — is a natural target for record digitization and for downstream safety checks before a prescription reaches a pharmacy. Not every OCR error carries the same clinical weight, however. An output that is obviously garbled is easy for a pharmacist to flag and set aside. An error in which the model outputs a different, but equally real, drug name is a different kind of failure: it produces a fluent, valid-looking prescription for the wrong medication, and is correspondingly harder to catch on routine review. This class of error — "look-alike, sound-alike" (LASA), or confusable-drug-name, confusion — is treated in the pharmacy-informatics and computational-linguistics literature as a distinct problem worth its own detection methodology, independent of raw transcription accuracy (Kondrak & Dorr, 2004, 2006). [TODO: confirm — a dedicated clinical-epidemiology citation quantifying the general burden of LASA-related medication error was not part of this project's verified bibliography; add one before submission if reviewers expect a citation for the underlying clinical-safety claim beyond the algorithmic-identification literature cited here.]

A prior, separate project by the author examined this problem for OCR directly: fine-tuning TrOCR with Low-Rank Adaptation (LoRA) on handwritten medical text reduced wildly-wrong, non-word ("hallucination") errors overall, but specifically increased the rate at which the model misread one real drug name as a *different* real drug name — the confusable-drug-name category described above [TODO: confirm citation once that manuscript is public]. That finding rests on a different dataset, label set, and codebase than the present study, and we cite it here strictly as motivating context for the research question below, not as evidence this paper itself produced.

The present study asks two questions that finding raises but does not answer. First, does the same pattern — overall transcription accuracy improving while the single most dangerous error type gets worse — reproduce independently, on new public data, with a new, unrelated implementation? Second, if the loss function used during fine-tuning is partly responsible for this pattern, can the loss itself be modified to discourage the dangerous error category directly, rather than relying on a post-hoc filter to catch it afterward?

We address the first question with a zero-shot-versus-fine-tuned reproduction on two public datasets. On Kaggle-BD (a 78-class closed vocabulary, n = 780 test images), zero-shot TrOCR reaches 22.95% correct (179/780) with a confusable-wrong-drug rate of 0.38% (3/780); a single-run LoRA cross-entropy (CE)-only fine-tune raises correct predictions to 87.95% (686/780) while the confusable-wrong-drug rate stays at 0.64% (5/780) — a shift from 3 to 5 affected images, too small a sample on its own to read as a meaningful change. On RxHandBD, a larger and more LASA-rich vocabulary (approximately 1,430 normalized label instances, n = 1,115 test images), the same comparison tells a different story. Zero-shot TrOCR reaches only 19.55% correct (218/1,115); LoRA CE-only fine-tuning raises correct predictions to 46.64% (520/1,115) and cuts far-off hallucination from 60.90% to 22.15% — but the confusable-wrong-drug rate rises from 1.79% (20/1,115) to 7.26% (81/1,115), roughly a four-fold relative increase. A second, independent CE-only fine-tuning run gave a consistent result (confusable-wrong-drug 7.44%, 83/1,115; correct 46.5%, 519/1,115), which we treat as ordinary run-to-run variation rather than a discrepancy — and it is exactly this kind of variation that later motivated the fully-seeded confirmatory design described below. Taken together, these results reproduce, on new public data and a codebase unrelated to the prior project's, the same central paradox: fine-tuning improves overall accuracy sharply while making the single most dangerous error type worse — and it does so specifically in the setting with a larger, more confusable-rich drug vocabulary, consistent with the effect depending on vocabulary size and LASA density rather than being a fixed property of LoRA fine-tuning in general.

Having reproduced the phenomenon, we test whether it can be addressed inside the fine-tuning loss itself rather than through post-hoc filtering. We add two terms on top of standard cross-entropy: a severity-weighted CE term, adapted from a risk-calibrated loss originally proposed for single-label medical-image classification, and a margin-ranking term that trains the model to score the true label ahead of an algorithmically-derived confusable neighbor by a fixed margin. Critically, we do not treat a favorable result from a single exploratory hyperparameter-sweep run as sufficient evidence that this intervention works. An earlier internal draft of this manuscript did exactly that — reporting, from the one exploratory run that selected it, that the chosen configuration reduced the confusable-wrong-drug rate "without a significant cost" to accuracy or hallucination. A subsequent five-seed, fully-seeded confirmatory replication overturned that framing: the apparent free improvement was a favorable statistical outlier, not a property of the configuration. We therefore report a two-phase design — an exploratory, hypothesis-generating phase using single, unseeded runs, followed by a confirmatory phase using five independent, fully-seeded, paired replications — as both a methodological finding in its own right and the structure required to support this paper's central claim.

This paper makes five contributions:

1. An independent reproduction of the fine-tuning-increases-LASA-error phenomenon previously observed in a separate, prior project [TODO: confirm citation], carried out on two new public datasets and a codebase unrelated to that project's, showing the effect is negligible on a small closed vocabulary (Kaggle-BD, 3 → 5 affected images) but rises roughly four-fold in relative terms on a larger, more LASA-rich vocabulary (RxHandBD, 1.79% → 7.26%).
2. A lexicon-free method for deriving candidate confusable drug-name pairs directly from a dataset's own vocabulary, combining edit-distance-based orthographic similarity with phonetic similarity (Double Metaphone plus Jaro–Winkler), usable without a curated, RxNorm/ISMP-scale drug-name list.
3. A combined margin-ranking-plus-severity-weighted loss for sequence-to-sequence OCR fine-tuning: a severity-weighted cross-entropy term that adapts an idea from single-label medical-image classification (Mohammadi-Seif & Baeza-Yates, 2026) to sequence generation, combined with a margin-ranking term scored against the algorithmically-derived confusable candidates from Contribution 2.
4. Confirmatory evidence from five independent, fully-seeded training replications, in a paired design (the same seed is reset immediately before each configuration's own training run), evaluated with exact McNemar tests and effect sizes reported with 95% confidence intervals, showing that this loss combination defines a statistically robust, quantified operating point on a safety–accuracy trade-off curve rather than a cost-free improvement: a pooled 12.6% relative reduction in the confusable-wrong-drug rate (7.16% → 6.26%, pooled exact McNemar p = 0.000077) is accompanied by a real, disclosed cost of approximately 2.0 percentage points of overall correct-rate (46.9% → 44.9%, p = 0.000001) and a rise of approximately 2.2 percentage points in far-off hallucination (21.5% → 23.6%, p < 0.000001), with all three 95% confidence intervals for the corresponding risk differences excluding zero.
5. A destination-of-fixed-errors and overcorrection analysis showing that most of this cost is comparatively safe in a specific, measured sense: of the confusable-wrong-drug errors that the winning configuration resolves, the majority (65.0%, 67 of 103, pooled across five seeds) become an obviously-wrong output rather than a correct one — a category we argue is easier for a pharmacist to catch on review than a fluent wrong drug name — and only 0.61% (16 of 2,614) of previously-correct predictions become newly confusable-wrong-drug errors under the winning configuration.

The remainder of the paper is organized as follows. Section 2 reviews related work on confusable-drug-name identification, severity- and risk-calibrated loss functions for medical AI, and contrastive and margin-based learning for text recognition and biomedical entity normalization. The Materials and Methods section describes the two public datasets and their provenance, the confusable-pair-derivation method, and the severity-weighted and margin-ranking loss design in full. A dedicated section then lays out the two-phase exploratory/confirmatory experimental design in detail, followed by the Phase 1 exploratory results (the ablation and hyperparameter sweep, and the negative cross-dataset replication check on Kaggle-BD) and the Phase 2 confirmatory results (the five-seed replication, effect sizes, and error-destination analysis) in turn. A Discussion section interprets these results as a quantified safety–accuracy trade-off and draws a methodological lesson about single-run, unseeded hyperparameter sweeps and test-set reuse in this research area; a Limitations section states the study's evidentiary boundaries explicitly, including the shared test split across selection and confirmation, the single-dataset scope of the confirmed effect, and the non-clinical provenance of the confusable-pair list; and a Conclusion closes the paper.

---

## 2. Related Work

### 2.1 Algorithmic identification of confusable ("look-alike, sound-alike") drug names

Identifying confusable drug names is an established sub-problem in pharmacy informatics and computational linguistics, and has historically been pursued independently of any OCR or handwriting-recognition system. Kondrak and Dorr (2004) introduced an approach and evaluation methodology for identifying confusable drug names by combining several orthographic and phonetic similarity measures into a single score, and Kondrak and Dorr (2006) extended this work with a fuller treatment of the same identification task. Millán-Hernández, García-Hernández, Ledeneva, and Hernández-Castañeda (2019) propose a related soft-bigram-similarity scoring method for the same task [TODO: verify the exact benchmark figures reported in this source manually before quoting a specific number in the final manuscript; SpringerLink access constraints have so far prevented full confirmation of the abstract's reported comparison].

Our own confusable-pair-derivation method follows the same general strategy as this line of work — combine an edit-distance-based orthographic score with a phonetic-similarity score into a single bi-similarity measure — but differs in where the candidate vocabulary comes from: rather than drawing on an external, curated list such as RxNorm or an ISMP-published LASA list, we derive candidate neighbors directly from each evaluation dataset's own observed drug-name vocabulary. This choice keeps the method reproducible from public data alone, at the cost of the face-validity limitations we report directly (see the Limitations section). More importantly for the present section, none of the identification methods above trains or modifies a recognition model: each produces a static candidate list or ranking, intended for downstream use such as tall-man-lettering recommendations or pharmacy-system alerts, not as a training signal inside a sequence-generation model. That is the specific gap our margin-ranking loss occupies: we reuse the same style of similarity scoring as Kondrak and Dorr's and Millán-Hernández et al.'s work, but as a per-example hard-negative source inside the fine-tuning objective itself, rather than as a static, externally-consulted list.

### 2.2 Severity- and risk-calibrated loss functions for medical AI

A separate line of work modifies the training loss of a medical AI system so that clinically severe errors are penalized more heavily than benign ones, rather than treating every misclassification as equally costly. Mohammadi-Seif and Baeza-Yates (2026) propose Risk-Calibrated Learning, which embeds a clinical severity/risk matrix directly into the training loss for single-label medical image classification, evaluated on brain tumor MRI grading, ISIC 2018 skin-lesion classification, BreaKHis histopathology, and SICAPv2 prostate grading, and report a 20.0–92.4% reduction in Critical Error Rate relative to Focal Loss. This is, at the time of writing, a preprint accepted for publication at IJCNN 2026 rather than an already-published final version, and we cite it as such throughout. Alansary, Mohamed, and Hamdi (2026) — also a preprint, accepted at ICTIS 2026 — apply a related, token-level severity-weighting idea to a markedly different setting: Arabic medical dialogue text generation with a large language model. That is a text-generation task with no paired input image, whereas the present work is a vision-to-text OCR task; we cite Alansary et al. here specifically to distinguish the two settings, not to claim direct relevance to OCR.

Neither severity-weighting method above is designed for, or evaluated on, sequence-to-sequence visual recognition, where the "label" is a full variable-length character sequence conditioned on an image and the loss must be computed per token across that sequence rather than once per example. Our own severity-weighted cross-entropy term is explicitly an *adaptation* of Mohammadi-Seif and Baeza-Yates' reweighting idea from single-label classification to sequence generation — not a reproduction of their method — and we name this adaptation gap explicitly for a specific reason: our own exploratory experiments (Phase 1) found that severity-weighting transplanted this way does not reliably help on its own, and improved results only when combined, at a low weight, with the margin-ranking term described in Section 2.3 below.

### 2.3 Contrastive, margin, and hard-negative learning for text recognition and biomedical entity normalization

Margin-based and contrastive objectives that push a model's representation of the correct answer away from a hard-negative candidate have precedent in both biomedical entity normalization and general scene-text recognition, though not, to our knowledge, applied together to OCR-based medication-name confusion. Sung, Jeon, Lee, and Kang (2020) train biomedical entity representations with synonym marginalization, using contrastive and hard-negative learning to normalize text mentions to a knowledge-base entity; this is closely related in spirit to our margin-ranking term, but is entirely text-only, with no OCR or vision component. Yuan et al. (2022) likewise perform text-only, cross-lingual medical term normalization (CODER) via knowledge-infused embeddings, again without a vision-to-text step. In scene-text recognition specifically, Aberdam et al. (2021) and Zhang, Lin, Xu, Chen, and Zhang (2023) both apply contrastive-learning objectives — sequence-to-sequence contrastive learning and relational contrastive learning, respectively — to improve general text-recognition representations, and Lee, Lee, and Hwang (2021) apply a related contrastive-with-adversarial-perturbation objective to conditional text generation for machine translation and summarization. In each of these three cases, the negative examples are drawn in-batch or generated by adversarial perturbation; none targets a specific, pre-identified set of confusable candidates the way a margin-ranking loss over algorithmically-derived LASA pairs does.

One further paper requires an explicit note here to preempt a reviewer reasonably expecting it to be related: Tupakula (2025) uses a margin-style contrastive objective and features the word "drug" prominently in its title ("Thin Bridges for Drug Text Alignment: Lightweight Contrastive Learning for Target Specific Drug Retrieval"), but is not about look-alike/sound-alike drug names. It aligns molecular embeddings (ECFP4 fingerprints) with biomedical text (ChEMBL) for target-based drug retrieval — an unrelated retrieval task operating at the molecule level rather than the surface-text level. We note it here only to distinguish it explicitly from the present work, not as related art for the confusable-drug-name problem.

Taken together, none of the work reviewed above combines all three elements of our approach: a severity-weighted loss adapted specifically from single-label classification to sequence generation (Section 2.2), a margin-ranking term over confusable candidates derived without a curated RxNorm/ISMP lexicon (Section 2.1 and 2.3), and a direct application to OCR-induced medication-name confusion rather than to text-only entity normalization or general scene-text recognition (Section 2.3). Single-run, unseeded hyperparameter comparisons are common practice across this combination of literatures, a pattern this paper's own exploratory-phase multiple-comparisons analysis illustrates directly. Unusually for that reason, we support our central claim instead with a five-seed, fully-seeded confirmatory replication reporting exact paired significance tests and effect sizes with 95% confidence intervals, rather than a single exploratory result.


---

## 3. Methods

### 3.1 Datasets

Two pre-existing, publicly released handwritten drug-name image datasets were used. No new data were
collected from any clinical site, and no institutional review was required, consistent with the
independent, public-data-only scope of this project.

**Table 1. Datasets used.**

| Dataset | Source | Vocabulary | Test set (n) | Stated license | Independent verification |
|---|---|---|---|---|---|
| Kaggle-BD | Kaggle, "Doctor's Handwritten Prescription BD dataset" (`mamun1113`) | 78-class closed vocabulary | 780 images | Open Database License (ODbL) | Accessed directly from its original Kaggle listing; single-source, no cross-referencing required |
| RxHandBD | Kaggle mirror `abrahametry/rxhandbd-handwritten-word-image-dataset` of the RxHandBD dataset (Md. Masudul Islam, BUBT) | ~1,430 normalized label instances (larger, more LASA-rich, open vocabulary) | 1,115 images | Disputed between the two original hosts (see below); stricter CC BY 4.0 terms followed | MD5-checksum-verified direct comparison against both original releases; 13/13 sampled label-to-image-ID correspondences matched |

#### 3.1.1 Kaggle-BD

Kaggle-BD is a closed, 78-class vocabulary of handwritten drug names released under the Open Database
License. It is used as-is from its original Kaggle page, with direct citation to the dataset listing.
Because of its small, closed vocabulary it is used in this study primarily as a **generalization/scope
check** (§3.5) rather than the primary confirmatory dataset.

#### 3.1.2 RxHandBD: provenance verification and disclosure

RxHandBD is accessed for this project through a third-party Kaggle community mirror, not directly from
either of its author's own hosting platforms. Because the paper's confirmatory claims rest on this
dataset, its provenance was independently verified before use, as follows:

- The two official releases of the underlying dataset were downloaded directly from their original
  sources — Zenodo (record 18478741) and Mendeley Data (DOI 10.17632/dsb5r6vskg.3) — and their file
  checksums (MD5) were compared to confirm the two official releases are themselves identical,
  uncorrupted copies of one another.
- The Kaggle mirror's image-label pairing was then spot-checked against both official releases: for 13
  randomly sampled items, the label-to-image-ID correspondence matched exactly (13/13), indicating that
  the mirror reuses genuine, uncorrupted underlying data rather than a relabeled or corrupted derivative.
- A discrepancy was found in image format: the Kaggle mirror's images are 128×64, RGB, whereas both
  official releases are 128×128, grayscale. This indicates the mirror's author independently
  resized/reprocessed the same underlying images and labels. This reprocessing step is not documented by
  the mirror's author, and is disclosed here as a data-provenance limitation (see also §6, Limitations,
  in the full manuscript).
- The two official sources disagree on license for what is otherwise the same underlying release by the
  same author: the Zenodo record's metadata states MIT, while the Mendeley Data record states CC BY 4.0.
  This inconsistency was not resolved at the source. It is resolved pragmatically in this work by
  following the stricter of the two — CC BY 4.0, with full attribution to the original author — for all
  use and any future redistribution of derived materials.
- The Kaggle download used for RxHandBD experiments bundles more than one dataset. A grep across every
  experiment kernel used in this study confirmed that all code paths read exclusively from the
  `RxHandBDMain` folder of that download; the unrelated, co-bundled second dataset was never loaded or
  used by any experiment reported here.

Neither dataset's raw images are redistributed directly in this project's code repository; data
availability for both datasets is addressed separately (Availability section).

---

### 3.2 Confusable-Pair Derivation

No curated, RxNorm/ISMP-scale look-alike, sound-alike (LASA) drug-name list was used. This was a
deliberate design choice, so that the whole pipeline remains lexicon-free and reproducible from public
data and public vocabulary alone (`src/confusable_pairs.py`). Instead, confusable-neighbor candidates
were derived algorithmically from each dataset's own vocabulary, following the general algorithmic
tradition of confusable drug-name identification established by Kondrak & Dorr (2004, 2006), using a
composite string-similarity score referred to here as BI-SIM.

#### 3.2.1 BI-SIM score

For two vocabulary words *a* and *b*:

```
bi_sim_score(a, b) = 0.5 × normalized_edit_similarity(a, b) + 0.5 × phonetic_similarity(a, b)
```

- `normalized_edit_similarity(a, b)` is computed from Levenshtein edit distance.
- `phonetic_similarity(a, b)` is computed from the Double Metaphone phonetic code of each word combined
  with Jaro–Winkler string similarity between the two codes.
- Implementation uses the `jellyfish` and `rapidfuzz` Python libraries for the edit-distance and
  phonetic/Jaro–Winkler components, respectively.

A related algorithmic approach, soft bigram similarity (Millan-Hernandez et al., 2019), targets the same
problem of scoring confusable drug-name pairs; [TODO: verify exact benchmark figures manually before
quoting any specific numbers from Millan-Hernandez et al., 2019].

To ensure the candidate list targets genuinely *different* drug names rather than spelling variants of
the same underlying word, near-duplicate/typo pairs — defined as pairs with edit distance ≤ 1 between the
same underlying word — are filtered out before neighbor-map construction.

#### 3.2.2 Neighbor-map construction

For each vocabulary word *w*, a neighbor list *N(w)* is built by keeping the top
`NEIGHBOR_TOP_K = 5` other vocabulary words *other* satisfying
`bi_sim_score(w, other) ≥ NEIGHBOR_MIN_SCORE = 0.5`, ranked by descending score. This neighbor map is
computed once per dataset vocabulary and is reused both for loss weighting/margin-candidate sampling
during training (§3.4) and for error classification during evaluation (§3.2.3).

#### 3.2.3 Evaluation-time confusable threshold and error taxonomy

A separate, higher threshold, `CONFUSABLE_THRESHOLD = 0.65`, is used only at evaluation time to decide
whether an incorrect prediction counts as a dangerous, look-alike/sound-alike confusion. A wrong
prediction is classified via `classify_error()` into one of the following mutually exclusive categories,
used identically for evaluation reporting and as the basis for the loss weighting scheme in §3.4:

**Table 2. Error taxonomy (`classify_error()`).**

| Category | Definition |
|---|---|
| `correct` | Predicted text exactly matches the true label |
| `minor_ocr_noise` | Not a real vocabulary word, but edit-similarity to the true label ≥ 0.7 |
| `confusable_wrong_drug` | Predicted text is itself a real vocabulary word **and** `bi_sim_score(true, predicted) ≥ CONFUSABLE_THRESHOLD (0.65)` |
| `other_wrong_real_word` | Predicted text is a real vocabulary word, but `bi_sim_score(true, predicted) < 0.65` |
| `hallucination_far_off` | Empty prediction, or far from both the true label and any real vocabulary word |

Note that the neighbor-construction threshold (`NEIGHBOR_MIN_SCORE = 0.5`) and the evaluation-time
confusability threshold (`CONFUSABLE_THRESHOLD = 0.65`) are deliberately different and serve different
purposes: the former is a permissive threshold for building training-time candidate lists, while the
latter is a stricter threshold for labeling a specific wrong prediction as clinically dangerous.

#### 3.2.4 Spot-check validation and its limitation

Candidate-pair face validity was spot-checked manually (AI-assisted) by the author against the top-100
scoring candidate pairs per dataset. This spot-check was **not** verified by a pharmacist or other
clinical domain expert, which is disclosed here as an explicit limitation of the method (see also §6).

For Kaggle-BD's top-100 candidate pairs, 38% were judged plausible LASA pairs, 61% were judged not
plausible, and 1% were same-brand-family variants. Restricting to pairs scoring ≥ 0.65, 68.75% were judged
plausible, versus 23.5% for pairs below 0.65 — a face-validity result consistent with `CONFUSABLE_THRESHOLD
= 0.65` doing real discriminative work, rather than being an arbitrary cutoff.

For RxHandBD's top-100 candidate pairs (all of which already scored ≥ 0.65 by construction of that
particular export), 42% were judged clearly plausible, 40% were judged noise/implausible, and 18% were
same-brand-family variants (e.g., "Calboral" vs. "Calboral D"). The relatively high rate of same-brand
variants is itself a useful secondary qualitative observation, but it also illustrates that the
algorithmically derived candidate list is an imperfect proxy for true clinical LASA pairs, a point revisited
in the Limitations section.

---

### 3.3 Model and Fine-Tuning Setup

#### 3.3.1 Backbone selection: TrOCR over Tesseract

The OCR backbone used throughout this study is the pretrained sequence-to-sequence transformer
`microsoft/trocr-base-handwritten`. A classical OCR engine, Tesseract, was also benchmarked zero-shot as a
reference point at the outset of the project. Tesseract performed far worse than TrOCR on handwritten drug
names — for example, on RxHandBD, zero-shot correct rate was 4.30% for Tesseract versus 19.55% for TrOCR.
On this basis, Tesseract was not carried forward, and TrOCR was selected as the sole backbone for all
fine-tuning experiments reported in this paper.

#### 3.3.2 LoRA fine-tuning configuration

Low-Rank Adaptation (LoRA) was applied to the TrOCR backbone using the `peft` library (version 0.13.2),
with the following configuration used as the reference/default configuration throughout Phases 1–2 (§3.5)
unless a swept hyperparameter is stated otherwise:

**Table 3. LoRA and training configuration.**

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Bias | none |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| PEFT library version | peft 0.13.2 |
| Optimizer | [TODO: confirm] |

The two reference model conditions compared throughout this paper are (i) the zero-shot pretrained
TrOCR backbone with no fine-tuning, and (ii) a LoRA fine-tune of that backbone using only the standard
cross-entropy (CE) objective ("CE-only"), which serves as the baseline against which the loss-design
interventions of §3.4 are evaluated.

---

### 3.4 Loss Design

Two additional loss terms are layered on top of standard token-level cross-entropy (CE) during LoRA
fine-tuning. Both terms are computed using the confusable-neighbor structure derived in §3.2.

#### 3.4.1 Severity-weighted cross-entropy

This term adapts the loss-weighting idea of Risk-Calibrated Learning (Mohammadi-Seif & Baeza-Yates, 2026),
which was originally proposed for **single-label image classification** (e.g., brain tumor MRI grading),
to **sequence generation**. This is explicitly an adaptation, not a reproduction: Mohammadi-Seif &
Baeza-Yates (2026) embeds a clinical severity/risk matrix into a classification loss over a fixed label
set, whereas this project applies an analogous per-example weighting to an autoregressive, open-vocabulary
sequence-generation loss, for which no direct equivalent of a fixed risk matrix exists.

For a training example *i* with true label text *y_i*, define a per-example weight:

```
w_i = 1 + β · s₁(y_i)
```

where `s₁(y_i)` is the BI-SIM score of *y_i*'s single **nearest** neighbor in that dataset's neighbor map
(top-1 only — not an average or maximum over all *k* = `NEIGHBOR_TOP_K` neighbors). The severity-weighted
CE loss over a batch is then:

```
weighted_CE = mean_i( w_i · NLL(y_i) )
```

where `NLL(y_i)` is the per-example, token-averaged negative log-likelihood of the true label sequence
under the current model.

#### 3.4.2 Margin-ranking loss over confusable neighbor candidates

The second term is a hard-negative margin-ranking objective, conceptually related to prior
hard-negative-style training against confusable candidates in biomedical entity normalization (Sung et
al., 2020; Yuan et al., 2022), adapted here to score against an OCR model's own token-sequence
likelihoods rather than embedding-space contrasts.

For each training example with true label *y_i*, one negative candidate *n_i* is sampled **uniformly at
random** from *y_i*'s precomputed neighbor list *N(y_i)* (§3.2.2). Examples whose true label has no
qualifying neighbor (i.e., `N(y_i)` is empty) are excluded from this term. The negative candidate's
sequence negative log-likelihood under the current model, `NLL(n_i)`, is computed by scoring the *same*
input image against the negative candidate's token sequence. The margin-ranking loss is:

```
margin_loss = mean_i( relu( margin − (NLL(n_i) − NLL(y_i)) ) )
```

computed over the subset of examples with at least one qualifying neighbor, with `margin` fixed at 1.0
nat. The intuition is to push the model to assign the true label a log-likelihood at least `margin` nats
better than a plausible confusable alternative — directly targeting the specific LASA-style confusion that
the CE term alone provides no explicit signal against.

#### 3.4.3 Combined objective

Severity weighting and margin-ranking are structurally independent loss terms; various on/off
combinations and hyperparameter settings were explored (§3.5). The combined objective used for the
confirmatory ("winning") configuration carried into Phase 2 is:

```
total_loss = weighted_CE + λ_margin · margin_loss,   with β = 0.3, λ_margin = 2.0
```

---

### 3.5 Experimental Design

The evaluation of the combined loss (§3.4) is deliberately structured as two explicit, sequential
phases — an **exploratory** phase and a **confirmatory** phase — rather than as a single hyperparameter
search reported as final evidence. This separation is a core methodological feature of the study, not
an incidental detail, and the two phases are not to be blurred together when interpreting results.

#### 3.5.1 Phase 1 — exploratory ablation and hyperparameter sweep (kernels 03, 04)

Phase 1's purpose is hypothesis generation and the **selection of a single candidate configuration** to
carry forward to confirmatory testing. It is explicitly **not** intended, and not valid, as standalone
confirmatory evidence, for two compounding reasons: (i) each configuration in Phase 1 was trained and
evaluated only once, and (ii) training was not fully reproducibly seeded — only Python's `random.seed`
was set, while `torch` and `numpy` RNGs were left unseeded and `cudnn` was not set to deterministic mode.
All Phase 1 comparisons use RxHandBD's single, fixed, 1,115-image test split.

Notably, even before any loss-design intervention, the fixed CE-only baseline used throughout Phase 1
(83/1,115 = 7.44% `confusable_wrong_drug`; 519/1,115 = 46.5% `correct`; 241/1,115 = 21.6%
`hallucination_far_off`) was itself one of two independent single CE-only fine-tuning runs on RxHandBD. A
separate, earlier single CE-only run gave 81/1,115 = 7.26% `confusable_wrong_drug`, 520/1,115 = 46.64%
`correct`, and 247/1,115 = 22.15% `hallucination_far_off` — consistent with the Phase 1 baseline within
ordinary run-to-run noise. This run-to-run variability, present even in the absence of any loss-design
change, is itself part of the motivation for the fully seeded, multi-run, paired confirmatory design of
Phase 2 (§3.5.2).

**Batch (a) — kernel 03** (`λ_margin = 1.0` fixed), compared against the kernel-03 CE-only baseline
(`confusable_wrong_drug` = 83/1,115 = 7.44%):

**Table 4. Phase 1, batch (a) — kernel 03 (single run per row, uncorrected p-values).**

| Configuration | confusable_wrong_drug | p (uncorrected) |
|---|---|---|
| CE-only baseline | 83/1,115 (7.44%) | — |
| Severity-only (β not tuned) | 88/1,115 (7.89%) | 0.47 |
| Margin-only | 74/1,115 (6.64%) | 0.18 |
| Severity + margin combined (kernel-03 defaults) | 83/1,115 (7.44%) | 1.00 |

Severity-only weighting alone did not help, and directionally slightly hurt — an informative *negative*
finding, since up-weighting the loss for "risky" classes does not by itself teach the model to avoid one
specific wrong answer. Margin-only was the only directionally positive signal in this batch. Severity and
margin combined at their kernel-03 defaults returned to baseline, suggesting severity weighting can dilute
margin's more targeted signal when combined naively at that weighting level.

**Batch (b) — kernel 04 sweep**, all compared against the same fixed CE-only baseline
(`confusable_wrong_drug` = 83/1,115 = 7.44%; `correct` = 519/1,115 = 46.5%; `hallucination_far_off` =
241/1,115 = 21.6%):

**Table 5. Phase 1, batch (b) — kernel 04 sweep (single run per row, uncorrected p-values).**

| Configuration | confusable_wrong_drug (p) | correct (p) | hallucination_far_off (p) |
|---|---|---|---|
| λ_margin = 2.0 | 67/1,115, 6.01% (0.009) | 44.0% (0.014) | 24.3% (0.0035) |
| λ_margin = 3.0 | 59/1,115, 5.29% (0.0001) | 44.4% (0.021) | 24.5% (0.0022) |
| λ_margin = 2.0, 5 epochs (vs. 3) | 73/1,115, 6.55% (0.14, n.s.) | — | — |
| λ_margin = 2.0 + low severity β = 0.3 | 67/1,115, 6.01% (0.011) | 523/1,115, 46.9% (0.76, n.s.) | 250/1,115, 22.4% (0.41, n.s.) |

`λ_margin = 3.0` gave the single largest raw reduction in `confusable_wrong_drug` found in Phase 1, but
at a significant single-run cost to both `correct` and `hallucination_far_off`. The margin effect faded
with longer training (5 vs. 3 epochs), consistent with CE eventually dominating the margin term. The
configuration **λ_margin = 2.0 + low severity β = 0.3** (β reduced roughly 3× from the kernel-03 default,
not dropped entirely), trained for 3 epochs, was **selected** at the end of Phase 1 as the sole candidate
carried forward to Phase 2, because in this single run it appeared to reduce the dangerous error category
without a statistically significant single-run cost to `correct` or `hallucination_far_off`. As shown in
§3.5.2 below, this apparent cost-free benefit was a favorable statistical outlier rather than a robust
property of the configuration.

**Multiple-comparisons sensitivity check.** A Holm–Bonferroni correction (`scripts/multiple_comparisons.py`)
was applied post hoc across the family of these seven Phase-1 single-run comparisons against the same
CE-only baseline on the same RxHandBD test split, as a sensitivity/motivation check — **not** as
confirmatory evidence:

**Table 6. Holm–Bonferroni-adjusted results for selected Phase-1 comparisons.**

| Configuration | p (uncorrected) | p (Holm–Bonferroni) | Survives at α = 0.05? |
|---|---|---|---|
| λ_margin = 3.0 | 0.0001 | 0.0008 | **Yes** |
| λ_margin = 2.0 (alone) | 0.009 | 0.054 | No |
| λ_margin = 2.0 + severity β = 0.3 (**selected**) | 0.011 | 0.057 | No |
| λ_margin = 2.0, 5 epochs | 0.143 | 0.573 | No |
| λ_margin = 1.0 (margin-only, kernel 03) | 0.176 | 0.573 | No |
| Severity-only, β not tuned (kernel 03) | 0.473 | 0.946 | No |
| Severity + margin combined, kernel-03 defaults | 1.000 | 1.000 | No |

Critically, the configuration ultimately selected for Phase 2 does **not** survive this correction. This
result is the explicit motivation for Phase 2: it demonstrates that none of the Phase 1 single-run
findings — including the one eventually selected — can be trusted as final evidence on their own, given
the researcher-degrees-of-freedom inherent in comparing many configurations on one fixed test set with
unseeded single runs.

**Cross-dataset check (kernel 05).** The Phase-1-selected configuration was additionally applied, in a
single run, to Kaggle-BD. It did not replicate: `confusable_wrong_drug` moved from 3/780 (0.38%) to 4/780
(0.51%) (p = 1.00, uninformative given only 3–4 affected images), while the `correct` rate significantly
*decreased* (87.7% → 86.0%, p = 0.041) with no measurable compensating benefit, and `hallucination_far_off`
was unchanged (p = 0.26). The most likely interpretation is that the intervention needs a large,
LASA-rich vocabulary to have "room" to act; on a small, closed, 78-class vocabulary already close to
saturated performance, it mostly adds training noise. Accordingly, this study's confirmed claims (§3.5.2)
are scoped to RxHandBD-like settings — larger, more LASA-dense, open vocabularies — and are not claimed to
generalize to arbitrary handwritten drug-name OCR settings.

#### 3.5.2 Phase 2 — confirmatory replication (kernel 06)

Phase 2 constitutes this paper's primary evidence. The single Phase-1-selected configuration
(λ_margin = 2.0 + severity β = 0.3, 3 epochs) was retrained from scratch **five times**, each with a
different, **fully seeded** random seed: Python's `random`, NumPy, and both `torch` and `torch.cuda` RNGs
were all seeded, with `cudnn.deterministic = True`, `cudnn.benchmark = False`, and `DataLoader
num_workers = 0` for determinism.

Each of the five seeded runs of the winning configuration is **paired** against a CE-only baseline
retrained with the *same* seed, reset immediately before that baseline's own training begins. This pairing
design is what isolates the loss-function effect from other sources of run-to-run variation: because the
seed governing LoRA weight initialization and data-shuffle order is identical within each pair, any
systematic difference between the two arms of a pair is attributable to the change in loss function, not
to initialization or shuffling noise. All five seed-pairs are evaluated on the same fixed RxHandBD
1,115-image test split used throughout this study.

**Statistical testing.** Per-seed and pooled significance are assessed with paired, **exact** McNemar
tests, computed as `scipy.stats.binomtest` applied to the discordant pairs between the two arms of each
comparison — never the approximate/chi-square McNemar approximation. Effect sizes for the pooled analysis
are reported with 95% confidence intervals using a Wald interval for the difference of two **paired**
proportions (`scripts/analyze_confirmatory_extended.py`):

```
d = p₀₁ − p₁₀
Var(d) = [p₁₀ + p₀₁ − (p₀₁ − p₁₀)²] / n
```

evaluated on the pooled n = 5,575 paired comparisons (5 seeds × 1,115 test images). Full per-seed and
pooled results, effect sizes, and the associated destination-of-error and overcorrection analyses are
reported in the Results section.

#### 3.5.3 Two-phase design as a reusable template

Beyond its role in this specific study, the exploratory/confirmatory structure described above — an
unseeded, single-run hyperparameter sweep used only for hypothesis generation and configuration selection,
combined with a Holm–Bonferroni sensitivity check on that sweep, followed by a small number of fully
seeded, paired-by-seed confirmatory replications with exact paired significance testing — is proposed as a
reusable template for similarly small-scale OCR safety studies, where full-scale multi-seed replication of
every candidate configuration is computationally impractical but a single-run sweep alone is not a
credible basis for a final claim.


---

## 4. Results

### 4.1 Phase 0: Reproduction of the Motivating Phenomenon

Before any loss-function intervention, both datasets were evaluated with zero-shot pretrained TrOCR (`microsoft/trocr-base-handwritten`) and with a single-run, CE-only LoRA fine-tune (r=16, α=32, dropout=0.05, target modules `[q_proj, v_proj]`, bias=`none`, peft 0.13.2, 3 epochs, batch size 8, lr=1e-4). A classical OCR engine (Tesseract) was also benchmarked zero-shot as a reference point on RxHandBD and performed far worse than TrOCR on handwritten drug names (correct rate: Tesseract 4.30% vs. TrOCR 19.55%); Tesseract was therefore not carried forward, and TrOCR was used as the sole fine-tuning backbone for the remainder of the study.

**Table 4.** Zero-shot vs. single-run CE-only LoRA fine-tuning, by dataset and error category.

| Dataset | Configuration | Correct | Hallucination (far-off) | Confusable (wrong drug) |
|---|---|---|---|---|
| Kaggle-BD (n=780, 78-class) | Zero-shot TrOCR | 22.95% (179/780) | 60.77% (474/780) | 0.38% (3/780) |
| Kaggle-BD (n=780, 78-class) | LoRA CE-only fine-tune | 87.95% (686/780) | 5.77% (45/780) | 0.64% (5/780) |
| RxHandBD (n=1115, ~1430 label instances) | Zero-shot TrOCR | 19.55% (218/1115) | 60.90% (679/1115) | 1.79% (20/1115) |
| RxHandBD (n=1115, ~1430 label instances) | LoRA CE-only fine-tune (Run 1) | 46.64% (520/1115) | 22.15% (247/1115) | 7.26% (81/1115) |

A second, independently-trained CE-only run on RxHandBD gave confusable_wrong_drug = 83/1115 (7.44%), correct = 519/1115 (46.5%), hallucination_far_off = 241/1115 (21.6%) — consistent with Run 1 within expected run-to-run noise. This second run is reused as the fixed reference baseline for the exploratory ablation/sweep phase (Section 4.2), and the run-to-run variability observed here is precisely what later motivated the fully-seeded, multi-run confirmatory design of Phase 2 (Section 4.4).

**Interpretation.** On both datasets, LoRA fine-tuning produces a large overall improvement: hallucination_far_off drops sharply and correct rate rises substantially. On the small, closed-vocabulary Kaggle-BD dataset (78 classes), the dangerous confusable_wrong_drug category remains negligible after fine-tuning (3→5 images out of 780; not a meaningful signal on its own). On the larger, more LASA-rich RxHandBD vocabulary, however, the same fine-tuning procedure that drives the large overall gains also increases confusable_wrong_drug by roughly 4× in relative terms (1.79% → ~7.3%). This is the motivating paradox for the present study — overall accuracy improves sharply while the single most clinically dangerous error type gets worse — and it is most pronounced on RxHandBD, independently reproducing the prior project's finding on new public data, a new codebase, and new labels.

[Figure: grouped bar chart of correct / hallucination_far_off / confusable_wrong_drug rates, zero-shot vs. fine-tuned, faceted by dataset]

---

### 4.2 Phase 1: Exploratory Ablation and Hyperparameter Sweep

**This subsection is exploratory and hypothesis-generating only.** Each configuration below was trained and evaluated exactly once, without full RNG seeding (only Python's `random.seed` was set; PyTorch and NumPy generators were not seeded, and cuDNN was not set to deterministic mode), on RxHandBD's single fixed 1115-image test split. None of the p-values in this subsection are corrected for the multiple configurations compared, and none should be read as confirmed statistical significance; Phase 1's sole purpose is to select one candidate configuration for confirmatory replication in Phase 2.

All comparisons in this subsection are against the same CE-only baseline (Run 2 from Section 4.1): confusable_wrong_drug = 83/1115 (7.44%), correct = 519/1115 (46.5%), hallucination_far_off = 241/1115 (21.6%).

**Kernel 03 — fixed λ_margin = 1.0, varying loss components** (confusable_wrong_drug only; p-values uncorrected, single run):

**Table 5.** Kernel 03 ablation batch.

| Configuration | Confusable (wrong drug) | p (uncorrected) |
|---|---|---|
| CE-only baseline | 83/1115 (7.44%) | — |
| Severity-only (β not tuned) | 88/1115 (7.89%) | 0.47 |
| Margin-only | 74/1115 (6.64%) | 0.18 |
| Severity + margin combined (kernel-03 defaults) | 83/1115 (7.44%) | 1.00 |

Severity-only is directionally *worse* than baseline; margin-only is the single directionally-positive signal in this batch; severity+margin combined returns to baseline, suggesting severity weighting dilutes margin's targeted signal when combined naively at these settings.

**Kernel 04 — sweep over margin λ, training length, and low-β severity:**

**Table 6.** Kernel 04 sweep, all vs. the CE-only baseline above (uncorrected, single run).

| Configuration | Confusable (wrong drug), p | Correct, p | Hallucination (far-off), p |
|---|---|---|---|
| CE-only baseline | 83/1115 (7.44%) | 519/1115 (46.5%) | 241/1115 (21.6%) |
| Margin λ=2.0 (3 epochs) | 67/1115 (6.01%), p=0.009 | 44.0%, p=0.014 | 24.3%, p=0.0035 |
| Margin λ=3.0 (3 epochs) | 59/1115 (5.29%), p=0.0001 | 44.4%, p=0.021 | 24.5%, p=0.0022 |
| Margin λ=2.0, 5 epochs | 73/1115 (6.55%), p=0.14 (NS) | [TODO: confirm] | [TODO: confirm] |
| **Margin λ=2.0 + severity β=0.3 (selected)** | 67/1115 (6.01%), p=0.011 | 523/1115 (46.9%), p=0.76 (NS) | 250/1115 (22.4%), p=0.41 (NS) |

Margin λ=3.0 gives the single largest raw reduction in confusable_wrong_drug found in Phase 1, but at significant single-run costs to correct rate and hallucination rate. Extending margin λ=2.0 training from 3 to 5 epochs erodes the confusable-rate effect (p=0.14), suggesting the CE term eventually dominates the margin term with longer training. The configuration **margin λ=2.0 + severity β=0.3** (β reduced ~3× from the kernel-03 default, not dropped) was selected at the end of Phase 1 as the sole candidate carried forward to Phase 2, because in this single run it appeared to reduce the dangerous error category without a statistically significant single-run cost to correct rate or hallucination rate. As shown in Section 4.4, this apparent "free lunch" was a favorable statistical outlier rather than a real property of the configuration.

**Holm-Bonferroni sensitivity check.** A Holm-Bonferroni correction (`scripts/multiple_comparisons.py`) was applied post hoc across the family of the 7 Phase-1 single-run comparisons above (all vs. the same CE-only baseline, same RxHandBD split), as a sensitivity/motivation check — not as confirmatory evidence.

**Table 7.** Holm-Bonferroni outcome for the 7 Phase-1 configurations (confusable_wrong_drug metric).

| Configuration | Uncorrected p | p (Holm-Bonferroni) | Survives at α=0.05? |
|---|---|---|---|
| Kernel 03: severity-only | 0.473 | 0.946 | No |
| Kernel 03: margin-only | 0.176 | 0.573 | No |
| Kernel 03: severity+margin combined | 1.000 | 1.000 | No |
| Kernel 04: margin λ=2.0 | 0.009 | 0.054 | No |
| Kernel 04: margin λ=3.0 | 0.0001 | 0.0008 | **Yes** |
| Kernel 04: margin λ=2.0, 5 epochs | 0.143 | 0.573 | No |
| Kernel 04: margin λ=2.0 + severity β=0.3 (selected) | 0.011 | 0.057 | No |

**Stated conclusion.** Only "margin λ=3.0" survives Holm-Bonferroni correction; critically, the configuration actually selected for Phase 2 ("margin λ=2.0 + severity β=0.3") does *not* survive (p_holm=0.057), nor does "margin λ=2.0" alone (p_holm=0.054). This result is the explicit motivation for Phase 2: it demonstrates that none of the Phase-1 single-run findings — including the one eventually selected — can be trusted as final evidence on their own, given the researcher-degrees-of-freedom inherent in comparing many configurations on one fixed test set with unseeded single runs.

---

### 4.3 Cross-Dataset Check (Kaggle-BD Non-Replication)

The Phase-1-selected configuration (margin λ=2.0 + severity β=0.3) was applied, in a single run, to Kaggle-BD (kernel 05).

**Table 8.** Cross-dataset check on Kaggle-BD (CE-only baseline vs. selected configuration).

| Metric | CE-only baseline | Selected configuration | p-value | Note |
|---|---|---|---|---|
| Confusable (wrong drug) | 3/780 (0.38%) | 4/780 (0.51%) | 1.00 | Uninformative given only 3–4 affected images |
| Correct | 87.7% | 86.0% | 0.041 | Significant decrease |
| Hallucination (far-off) | — | — | 0.26 | Unchanged |

**Interpretation.** The Phase-1-selected configuration did not replicate on Kaggle-BD: the confusable_wrong_drug change is statistically uninformative given the tiny number of affected images, while correct rate significantly *decreased* with no measurable compensating benefit, and hallucination rate was unchanged. This is consistent with the intervention needing a large, LASA-rich vocabulary to have "room" to help; on a small, closed 78-class vocabulary already close to saturated performance, it mostly adds training noise. Accordingly, the paper's confirmed claims (Section 4.4) are scoped to RxHandBD-like settings — larger, more LASA-dense, open vocabularies — and are not claimed to generalize to arbitrary handwritten drug-name OCR settings.

---

### 4.4 Phase 2: Confirmatory Replication (Primary Evidence)

The single Phase-1-selected configuration (margin λ=2.0 + severity β=0.3, 3 epochs — "the winning configuration") was retrained from scratch 5 times with 5 different, fully seeded random seeds (Python `random`, NumPy, and PyTorch/CUDA RNGs all seeded; `cudnn.deterministic=True`, `cudnn.benchmark=False`; `DataLoader(num_workers=0)`), each paired against a CE-only baseline retrained with the same seed reset immediately before that baseline's own training — isolating the loss-function effect from LoRA-initialization and data-shuffle-order noise. All 5 seed-pairs were evaluated on the same fixed RxHandBD 1115-image test split. Per-seed and pooled paired McNemar exact tests (`scipy.stats.binomtest` on discordant pairs) were used throughout — never an approximate/chi-square McNemar test.

**Table 9.** Per-seed results, CE-only → winning configuration (exact McNemar p-values).

| Seed | Confusable (wrong drug) | Correct | Hallucination (far-off) |
|---|---|---|---|
| 1 | 79 → 65 (p=0.049) | 522 → 495 (p=0.011) | 242 → 268 (p=0.018) |
| 2 | 74 → 68 (p=0.362) | 524 → 508 (p=0.101) | 238 → 275 (p=0.0002) |
| 3 | 90 → 79 (p=0.043) | 513 → 503 (p=0.368) | 238 → 254 (p=0.109) |
| 4 | 78 → 66 (p=0.036) | 526 → 498 (p=0.010) | 235 → 255 (p=0.045) |
| 5 | 78 → 71 (p=0.265) | 529 → 498 (p=0.005) | 243 → 264 (p=0.033) |

All three metrics move in the **same direction in all 5/5 seeds, with no exceptions**: confusable_wrong_drug always decreases, correct always decreases, hallucination_far_off always increases.

**Table 10.** Pooled results across 5 seeds (n=5,575 paired comparisons; one pooled exact McNemar test per metric).

| Metric | Discordant pairs (favor CE-only / favor winning) | Pooled p | Mean rate, CE-only → winning | Individually significant seeds (p<0.05) |
|---|---|---|---|---|
| Confusable (wrong drug) | 103 fixed / 53 newly | 0.000077 | 7.16% → 6.26% (relative reduction ~12.6%) | 3/5 |
| Correct | 312 lost / 200 gained | 0.000001 | 46.9% → 44.9% (absolute drop ~2.0 points) | 3/5 |
| Hallucination (far-off) | 178 lost / 298 gained | <0.000001 | 21.5% → 23.6% (absolute rise ~2.2 points) | 4/5 |

**Table 11.** Effect sizes with 95% CI for paired proportions (Wald CI, pooled n=5,575; `scripts/analyze_confirmatory_extended.py`).

| Metric | Risk difference (percentage points) | 95% CI |
|---|---|---|
| Confusable (wrong drug) | −0.90 | [−1.34, −0.46] |
| Correct | −2.01 | [−2.80, −1.22] |
| Hallucination (far-off) | +2.15 | [+1.39, +2.92] |

All three 95% CIs exclude zero, consistent with the pooled p-values above.

[Figure: forest plot of the three risk differences with 95% CIs]

**Correction of the earlier single-run impression.** The Phase-1 single run for this exact configuration (Table 6, Section 4.2) reported correct = 46.9% (p=0.76, not significant) and hallucination_far_off = 22.4% (p=0.41, not significant) alongside the confusable-rate reduction, creating an impression that the dangerous error type could be reduced "without a significant cost." The 5-seed Phase-2 confirmatory replication corrects this impression: pooled effects on all three metrics are statistically robust (Table 10, Table 11), with correct rate significantly *decreasing* and hallucination rate significantly *increasing* — not neutral, as the single Phase-1 run had suggested. The Phase-1 result was a favorable statistical outlier, not a real, cost-free property of the configuration. The robust, confirmed effect is that this loss combination fixes ~12.6% (relative) of the most dangerous error type at a real, statistically robust cost of ~2.0 points of overall correct rate and ~2.2 points of hallucination rate.

---

### 4.5 Safety-Oriented Secondary Analyses

**Destination of "fixed" confusable errors.** Pooled across all 5 seeds, CE-only produced 399 confusable_wrong_drug errors in total; of these, 103 (25.8%) were no longer classified as confusable_wrong_drug under the winning configuration.

**Table 12.** Destination of the 103 resolved confusable_wrong_drug errors (pooled, 5 seeds).

| Destination category | n (of 103) | Share |
|---|---|---|
| Correct | 36 | 35.0% |
| Minor OCR noise | 37 | — |
| Hallucination (far-off) | 30 | — |
| **Clearly wrong (minor noise + hallucination)** | **67** | **65.0%** |

Only 35.0% of resolved dangerous errors became fully correct; the majority (65.0%) were converted into an obviously-wrong error type instead (minor_ocr_noise or hallucination_far_off) rather than into a correct reading. Clinically, an obviously-wrong output (e.g., a nonsense string) is far easier for a pharmacist or clinician to catch and query than a fluent, plausible but wrong real drug name — so this destination shift represents a genuine safety benefit that the raw confusable_wrong_drug rate alone does not capture.

**Overcorrection ("new confusion") check.** Pooled across the 5 seeds, of the images that were correct under CE-only, only 16/2,614 (0.61%, Wilson 95% CI [0.38%, 0.99%]) became *newly* classified as confusable_wrong_drug under the winning configuration. Only a small minority of the ~2-point overall correct-rate loss documented in Section 4.4 is converted into the specific dangerous error type; most of the lost correct predictions instead become less-dangerous minor-noise or hallucination errors.

---

**Note on error categorization.** All counts above (correct / minor_ocr_noise / confusable_wrong_drug / other_wrong_real_word / hallucination_far_off) are produced by the automated, threshold-based `classify_error()` classifier (BI-SIM similarity to the true label and to the dataset vocabulary; CONFUSABLE_THRESHOLD=0.65), not by human pharmacist adjudication of individual predictions. This is a modeling choice, not ground truth, and is treated as an explicit limitation of the study (see Limitations).


---

## 5. Discussion

### 5.1 A quantified operating point, not a free improvement

The five-seed confirmatory replication (Phase 2) shows that the combined margin-ranking-plus-severity-weighted loss (λ_margin = 2.0, β = 0.3) does not remove the confusable_wrong_drug problem at no cost. It instead defines one statistically robust point on a trade-off curve between the rate of dangerous look-alike, sound-alike (LASA) errors and overall recognition accuracy. Pooled across 5,575 paired predictions (five seeds × 1,115 RxHandBD test images), the confusable_wrong_drug rate fell from a mean of 7.16% under CE-only training to 6.26% under the winning configuration — a 12.6% relative reduction (pooled exact McNemar p = 0.000077; risk difference −0.90 points, 95% CI [−1.34, −0.46]) — while the overall correct rate fell from 46.9% to 44.9% (pooled p = 0.000001; risk difference −2.01 points, 95% CI [−2.80, −1.22]) and the hallucination_far_off rate rose from 21.5% to 23.6% (pooled p < 0.000001; risk difference +2.15 points, 95% CI [+1.39, +2.92]).

| Metric | CE-only (mean) | Winning configuration (mean) | Pooled exact McNemar *p* | Risk difference (pp) | 95% CI |
|---|---|---|---|---|---|
| confusable_wrong_drug | 7.16% | 6.26% | 0.000077 | −0.90 | [−1.34, −0.46] |
| correct | 46.9% | 44.9% | 0.000001 | −2.01 | [−2.80, −1.22] |
| hallucination_far_off | 21.5% | 23.6% | <0.000001 | +2.15 | [+1.39, +2.92] |

All three metrics moved in the same direction in every one of the five seeds, with no exceptions — direction-consistent evidence for a real property of this loss combination rather than a single favorable run. That uniformity matters because the corresponding Phase-1 single run for this exact configuration reported changes in correct rate and hallucination rate that were not statistically significant (p = 0.76, p = 0.41), a result an earlier internal draft of this manuscript read as reducing the dangerous error "without a significant cost." The confirmatory replication corrects that reading: the apparent absence of cost was a favorable statistical outlier in one unseeded run, not a property of the method. We therefore frame the contribution as a disclosed, measured operating point — a roughly one-eighth relative reduction in the single most dangerous error category, purchased at a roughly two-point cost to raw accuracy and a roughly two-point increase in nonsense output — rather than a strictly dominant improvement.

### 5.2 Where resolved dangerous errors go: a fail-safe interpretation

Not all of the correct-rate loss carries equal clinical weight, because a real drug-name substitution and a nonsense string are not equally likely to be caught downstream. Of the 399 confusable_wrong_drug errors CE-only training produced across all five seeds, 103 (25.8%) were no longer classified as confusable_wrong_drug under the winning configuration. Of those 103 resolved errors, only 36 (35.0%) became fully correct; the remaining 67 (65.0%) became minor_ocr_noise (37 cases) or hallucination_far_off (30 cases) — outputs that do not resemble a plausible, valid drug name. A LASA substitution is dangerous precisely because it produces a fluent, plausible prescription that gives a pharmacist reading the output no obvious reason to double-check it; an overtly wrong string, by contrast, is the kind of failure a verification step is built to catch. On that basis, converting a LASA error into an overt error has downstream safety value that the raw correct/incorrect split does not capture, though this argument presumes a human verification step exists downstream and is not a claim that the system is safe to deploy without one.

### 5.3 Overcorrection is rare: the added cost is mostly safe in kind

The accuracy cost is also not concentrated in manufacturing new instances of the error type this study targets. Of images CE-only training got correct, only 16 of 2,614 (0.61%, Wilson 95% CI [0.38%, 0.99%]) became newly confusable_wrong_drug under the winning configuration; the rest of the lost correct predictions moved into minor_ocr_noise or hallucination_far_off. We describe this cost as mostly safe in kind: the roughly two-point drop in correct rate reported in Section 5.1 is real and statistically robust, but it is largely not being converted into the one failure mode the intervention is designed to reduce. This qualifies the size of the trade-off without removing it.

### 5.4 A vocabulary-size hypothesis for the cross-dataset non-replication

The winning configuration did not replicate when applied, in a single run, to Kaggle-BD's 78-class closed vocabulary: confusable_wrong_drug moved from 3/780 (0.38%) to 4/780 (0.51%) (p = 1.00), a change driven by only three or four affected images and uninformative on its own, while correct rate fell significantly (87.7% → 86.0%, p = 0.041) with no offsetting benefit and no measurable change in hallucination (p = 0.26). We hypothesize, without direct evidence beyond this one cross-dataset comparison, that the margin-ranking term needs a vocabulary that is both large and LASA-dense to have room to help: Kaggle-BD's CE-only baseline was already close to a performance ceiling (87.95% correct) with very few confusable_wrong_drug errors to begin with, so a term designed to separate confusable neighbors mostly adds training noise rather than correcting a substantial dangerous-error category. RxHandBD's larger, more diverse label space gave the same intervention a correspondingly larger category to act on (81/1,115 confusable errors under CE-only). Because the two datasets differ simultaneously in vocabulary size, label diversity, and baseline task difficulty, no single factor can be isolated as the cause from this comparison alone, and we scope our confirmed claims to RxHandBD-like settings — large, open, LASA-rich drug-name vocabularies — rather than to handwritten drug-name OCR in general.

### 5.5 A cautionary tale about single-run sweeps and test-set reuse

The gap between Phase 1 and Phase 2 is itself worth reporting on its own terms. Phase 1's seven single-run comparisons against the same CE-only baseline used no random seeding beyond Python's `random.seed`; a Holm–Bonferroni correction applied post hoc across that family left only one configuration significant (margin λ = 3.0, p_holm = 0.0008), while the configuration eventually carried forward to Phase 2 did not survive correction (p_holm = 0.057), nor did margin λ = 2.0 alone (p_holm = 0.054). More consequentially, the Phase-1 run of the selected configuration showed no significant change in correct rate or hallucination rate (p = 0.76, p = 0.41) — the pattern that motivated selecting it as an apparently cost-free improvement. The five-seed Phase-2 replication overturned that reading: the same two costs were significant in 3/5 and 4/5 seeds respectively, and highly significant pooled. This is the inverse of the usual multiple-comparisons warning, which concerns inflated false positives; here, a single-run design nearly certified a configuration as cost-free when a fully seeded replication shows a robust, replicable cost. This risk is compounded by the fact that Phase 1 and Phase 2 evaluate on the same fixed test split (Section 6, Limitation 1); seeding the training run does not by itself address test-set reuse, and we flag that gap as an open item rather than one seeding alone resolves. We read this combination as a concrete argument for treating any single-run hyperparameter sweep on a fixed, reused test set as hypothesis-generating only, and we propose the two-phase design used here — an inexpensive, single-run, multi-configuration exploratory phase with a multiplicity-aware sensitivity check, followed by a small number (here, five) of independently seeded, paired confirmatory replications of exactly one selected configuration — as a practical template for other small-scale OCR and medical-AI safety studies that cannot afford to seed-sweep every candidate configuration but still need a trustworthy final claim.

## 6. Limitations

1. **Test-set reuse.** The same fixed 1,115-image RxHandBD test split was used twice: to select the Phase-1 configuration, and again as the evaluation set for all five Phase-2 confirmatory seeds, with only the training seed varied. No held-out validation set independent of this final test set existed at any stage. The exploratory/confirmatory separation and the Holm–Bonferroni sensitivity check in Phase 1 reduce, but do not eliminate, the resulting optimism; a replication with a test split withheld from every Phase-1 decision would give a cleaner estimate of the effect sizes reported in Section 5.1.

2. **Single-dataset confirmation.** The Phase-2 confirmatory result holds only for RxHandBD; the same configuration did not replicate in a direct cross-dataset check on Kaggle-BD. Whether this reflects RxHandBD's larger, more LASA-dense vocabulary, as hypothesized in Section 5.4, or some other difference between the two datasets, is not established here, and we make no claim that the confirmed trade-off generalizes beyond RxHandBD-like settings.

3. **Confusable-pair list validity.** Confusable-neighbor candidates were derived algorithmically from BI-SIM scores rather than from a curated, clinically validated list such as RxNorm or an ISMP LASA list (cf. Kondrak & Dorr, 2004, 2006), and the candidate lists were spot-checked only by the author with AI assistance, not by a pharmacist or other clinical domain expert. Face-validity spot-checks on the top-100 candidate pairs per dataset found a meaningful minority of implausible or same-brand-family-variant pairs — 61% judged not plausible on Kaggle-BD, 40% noise/implausible on RxHandBD — particularly below the 0.65 evaluation threshold; because this dilutes rather than sharpens the neighbor lists used for loss weighting and margin sampling, the effects reported in Section 5.1 are more likely understated than inflated by this limitation.

4. **Dataset provenance and processing.** RxHandBD was accessed through a third-party Kaggle community mirror rather than directly from its two official hosts. We independently verified the mirror's content as genuine and uncorrupted against both official releases (MD5-checksum comparison and 13/13 matched label-to-image-ID samples), but the mirror's specific image preprocessing — 128×64 RGB versus the official releases' 128×128 grayscale — is undocumented by its author and could not be independently reconstructed. The two official sources also disagree on license for the same underlying release (Zenodo metadata states MIT; Mendeley states CC BY 4.0); we followed the stricter CC BY 4.0 terms with full attribution, but this discrepancy is not resolved at the source and should be checked directly by any reuser.

5. **Citation status of two central sources.** Mohammadi-Seif & Baeza-Yates (2026), the primary source for the severity-weighted loss term, and Alansary, Mohamed & Hamdi (2026), cited to distinguish this paper's OCR setting from prior severity-weighted text generation, are, at the time of writing, arXiv preprints accepted at IJCNN 2026 and ICTIS 2026 respectively rather than published final journal or conference versions; their content could still change before final publication.

6. **Automated, not clinician-adjudicated, error labels.** Every error label used in this study (correct, minor_ocr_noise, confusable_wrong_drug, other_wrong_real_word, hallucination_far_off) is assigned by an automated, threshold-based classifier operating on BI-SIM similarity to the true label and to the dataset vocabulary, not by a pharmacist or clinician reading each prediction. This is a modeling choice, not ground truth: two errors a human reviewer would judge equally dangerous could receive different labels on opposite sides of the 0.65 threshold, so the effect sizes in Section 5.1 describe movement of this specific automated taxonomy rather than an independently adjudicated clinical-risk scale.

7. **The severity-weighted term is an adaptation, not a reproduction.** The severity-weighted cross-entropy term adapts Mohammadi-Seif & Baeza-Yates's (2026) risk-calibrated loss from single-label medical image classification to sequence generation, a setting the original method was not designed or evaluated for. Consistent with this being an untested transfer, Phase 1 found severity weighting alone did not help and directionally hurt (confusable_wrong_drug 88/1,115 vs. an 83/1,115 baseline, p = 0.47); it showed benefit only in combination with margin-ranking loss at a low weight, and even then only in the single, unseeded Phase-1 run that Section 5.1 shows was a favorable outlier.

## 7. Conclusion

We set out to determine whether a loss function designed to discourage look-alike, sound-alike drug-name confusions can reduce that specific, clinically dangerous error type in LoRA-fine-tuned handwritten prescription OCR, and at what cost. Using an independent reproduction of the motivating fine-tuning-increases-LASA-errors phenomenon on two new public datasets, a lexicon-free, algorithmically derived confusable-pair method, and a two-phase exploratory-then-confirmatory design with five fully seeded replications, we find that a combined margin-ranking-plus-severity-weighted loss (λ_margin = 2.0, β = 0.3) reduces the confusable_wrong_drug rate on RxHandBD by 12.6% in relative terms, at a real, statistically robust cost of roughly two accuracy points and roughly two additional points of nonsense output. Most of that cost converts already-dangerous errors into overtly wrong, easily flagged output rather than manufacturing new dangerous confusions. We offer this as a measured operating point and a reproducible method, not a final verdict on whether the trade-off is worthwhile in any specific clinical workflow — that judgment depends on downstream review practices this study did not model. Future work should trace a fuller Pareto curve across additional λ_margin values under the same five-seed confirmatory protocol, replace the algorithmically derived confusable-pair lists with pharmacist-verified LASA pairs, and test the vocabulary-size dependence hypothesized in Section 5.4 across further prescription datasets spanning a wider range of vocabulary sizes.

## Availability of Data and Code

**Code.** All training, evaluation, and analysis code will be released on GitHub in a repository independent of the prior project that motivated this study. [TODO: confirm final repository URL before submission.]

**Data.** Kaggle-BD ("Doctor's Handwritten Prescription BD dataset," mamun1113) is publicly available from its original Kaggle page under the Open Database License and is cited directly rather than redistributed. RxHandBD was accessed via the Kaggle community mirror used for all experiments (abrahametry/rxhandbd-handwritten-word-image-dataset); readers are pointed to the two independently verified original sources — Mendeley Data (DOI 10.17632/dsb5r6vskg.3, CC BY 4.0) and Zenodo (record 18478741) — and are advised that the mirror's image preprocessing (128×64 RGB) differs from, and is not independently documented relative to, both official releases (128×128 grayscale; see §6, Limitation 4). Neither dataset's raw images will be redistributed directly within the code repository; users should obtain images from the original sources cited above.

## References

Aberdam, A., Litman, R., Tsiper, S., Anschel, O., Slossberg, R., Mazor, S., Manmatha, R., & Perona, P. (2021). Sequence-to-sequence contrastive learning for text recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021)*. arXiv:2012.10873.

Alansary, A., Mohamed, M., & Hamdi, A. (2026). Severity-aware weighted loss for Arabic medical text generation. arXiv:2604.06346 [cs.CL]. Accepted at ICTIS 2026.

Kondrak, G., & Dorr, B. (2004). Identification of confusable drug names: A new approach and evaluation methodology. In *Proceedings of COLING 2004* (pp. 952–958). https://doi.org/10.3115/1220355.1220492

Kondrak, G., & Dorr, B. (2006). Automatic identification of confusable drug names. *Artificial Intelligence in Medicine, 36*(1), 29–42. https://doi.org/10.1016/j.artmed.2005.07.005

Lee, S., Lee, D. B., & Hwang, S. J. (2021). Contrastive learning with adversarial perturbations for conditional text generation. In *Proceedings of ICLR 2021*. arXiv:2012.07280.

Millán-Hernández, C. E., García-Hernández, R. A., Ledeneva, Y., & Hernández-Castañeda, A. (2019). Soft bigram similarity to identify confusable drug names. In *Pattern Recognition (MCPR 2019)*, LNCS vol. 11524 (pp. 433–442). Springer. https://doi.org/10.1007/978-3-030-21077-9_40

Mohammadi-Seif, A., & Baeza-Yates, R. (2026). Risk-calibrated learning: Minimizing fatal errors in medical AI. arXiv:2604.12693 [cs.CV]. Accepted at IJCNN 2026.

Sung, M., Jeon, H., Lee, J., & Kang, J. (2020). Biomedical entity representations with synonym marginalization. In *Proceedings of ACL 2020* (pp. 3641–3650). https://doi.org/10.18653/v1/2020.acl-main.335

Tupakula, M. (2025). Thin bridges for drug text alignment: Lightweight contrastive learning for target specific drug retrieval. arXiv:2510.03309.

Yuan, Z., Zhao, Z., Sun, H., Li, J., Wang, F., & Yu, S. (2022). CODER: Knowledge-infused cross-lingual medical term embedding for term normalization. *Journal of Biomedical Informatics, 126*, 103983. https://doi.org/10.1016/j.jbi.2021.103983 (arXiv:2011.02947)

Zhang, J., Lin, T., Xu, Y., Chen, K., & Zhang, R. (2023). Relational contrastive learning for scene text recognition. In *Proceedings of the 31st ACM International Conference on Multimedia (ACM MM 2023)*. arXiv:2308.00508.
