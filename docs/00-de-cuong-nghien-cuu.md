# Research Proposal: Risk-Calibrated Fine-tuning Against Look-Alike-Sound-Alike (LASA) Drug Name Confusion in Handwritten OCR

This is an **independent** research project, not affiliated with CCI Australia, not Vietnamese-language OCR, using public data, requiring no IRB. It is entirely separate from the `ocr-research` project (handwritten prescriptions + LoRA TrOCR + 7-model benchmark) — that project is used only as motivational background, and this project does NOT reuse its code/data/labels.

## 1. Motivation

Original observation (from the previous project, to be **independently re-verified** on this project's own public dataset, without reusing the old figures): after fine-tuning a handwritten OCR model, "nonsensical output" errors (hallucination) dropped sharply, but errors of "misreading as ANOTHER REAL DRUG NAME" (confusable_wrong_drug / look-alike-sound-alike, LASA) increased — a more dangerous error type because it looks like a valid prescription.

## 2. Main Contribution

Rather than building a post-hoc processing layer or a separate flagging classifier, this project **directly modifies the loss function used to fine-tune the OCR model**, combining two additional components so that the training process itself learns to avoid LASA-type errors instead of patching them afterward:

- **(a) Confusion-specific margin loss**: for each image with ground-truth label `w`, take the set of "confusable neighbors" `N(w)` from the list of confusion pairs inferred algorithmically (Section 4). The decoder runs an additional teacher-forcing pass with the false label `w' ∈ N(w)`, computes `logP(w'|x)`, and applies a penalty of `max(0, margin − (logP(w|x) − logP(w'|x)))`. No special paired images are needed — any existing image from class `N(w)` is sufficient.
- **(b) Severity-weighted cross-entropy** (adapted from Risk-Calibrated Learning, Mohammadi-Seif & Baeza-Yates, IJCNN 2026, arXiv:2604.12693 — the original paper addresses single-label image classification; here it is adapted for sequence generation): multiply the per-sample CE loss by a coefficient based on that class's similarity-danger level: `weight = 1 + β·max_similarity(w, other classes)`.
- **Total loss**: `L = Σᵢ wᵢ·CEᵢ + λ·margin_loss`.

Hypothesized mechanism: LoRA fine-tuning "sharpens" the language prior already present in the decoder (the TrOCR decoder is initialized from RoBERTa) toward favoring common/plausible sequences — for LASA pairs, the "other plausible sequence" is precisely another real drug name. Margin loss intervenes directly in this mechanism; severity-weighting allocates more "corrective effort" to higher-risk classes.

## 3. Dataset (run entirely independently, not reusing results from the previous project)

- **RxHandBD** — the main dataset. A single canonical version must be settled on before use (2 versions with conflicting metadata):
  - Zenodo DOI [10.5281/zenodo.18478741](https://zenodo.org/records/18478741) — v1, 128×128px, MIT license.
  - Mendeley DOI [10.17632/dsb5r6vskg.3](https://data.mendeley.com/datasets/dsb5r6vskg/3) — v3, 512×512px, CC BY 4.0.
  - 5.578 images, ~1.559 unique entries (including dosage/instruction tokens, which need to be filtered out to isolate real drug names).
- **Kaggle "Doctor's Handwritten Prescription BD"** — secondary/cross-check dataset, 78 classes, 4.680 images. Uses a separate Kaggle account `bangdang007112` (see `scripts/kaggle_env.sh`).
- Both use Bangladeshi brand-name drugs — **not matched to RxNorm/ISMP** (confirmed in the previous research round: "Sergel" → 0 RxNav results, "Napa" → incorrectly mapped to "Napabucasin"). Therefore, algorithmically inferred confusion pairs are used (Section 4), rather than ISMP.

## 4. Confusion Pair Inference (Algorithmic, Not ISMP)

Combines weighted edit-distance with phonetic similarity (double metaphone / soft bigram similarity), following the method of Kondrak & Dorr (BI-SIM/ALINE) — a general-purpose algorithm that has been validated by matching against real-world confusion lists (USP/ISMP) prior to publication, and has precedent for application to real non-US drug markets (Australia-TGA "LASA v2", Canada, Iran). After the algorithm produces a ranking, **a sample is manually checked (top 30-50 highest-scoring pairs)** to reduce circularity risk (since there is no independent Bangladeshi ground truth).

## 5. Experimental Pipeline

1. Settle on a RxHandBD version, filter the real vocabulary (removing dosage/instruction tokens).
2. Build and manually check the confusion pair list.
3. Reduced OCR benchmark (no need to repeat the previous project's 8-model benchmark): Tesseract (cheap baseline) + TrOCR-base-handwritten zero-shot — to obtain **independently self-measured** baseline figures.
4. Fine-tune LoRA TrOCR with 4 loss configurations for comparison (ablation): CE-only / CE+severity / CE+margin / CE+severity+margin.
5. Evaluate under **genuine free-running decoding** (not just teacher-forcing likelihood).

## 6. Evaluation

Overall CER/WER plus the separate confusable_wrong_drug error rate (before/after each loss configuration), with paired statistical testing (Wilcoxon/bootstrap CI).

## 7. Main Risks

- **Small-sample overfitting**: each class has only a few dozen images — the negative pool must be strictly separated from the test set, `β`/`λ` must be tuned carefully, and early-stopping must be based on the correct confusion-pair metric.
- The positive sample size (confusable_wrong_drug) may be small — check the actual rate from week one, before committing to detailed statistical analysis.
- The RxHandBD license (Zenodo vs Mendeley) needs to be clarified before public release.

## 8. Novelty

No existing study combines: (a) an autoregressive sequence-generation model (encoder-decoder OCR/HTR) + (b) a loss explicitly directed by a specific confusion-pair list + (c) the medical drug-name domain. The closest related works (BioSyn/CODER, SeqCLR/RCLSTR/CLAPS, Risk-Calibrated Learning) each cover only a single isolated angle.

## 9. Infrastructure
- Free Kaggle T4 GPU, using a separate account `bangdang007112` for this project (see `scripts/kaggle_env.sh` before running any `kaggle` command).
- Git author: `bangdang007112 <bangdang007112@gmail.com>` (set locally, not globally).