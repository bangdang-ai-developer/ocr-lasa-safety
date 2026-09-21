# ocr-lasa-safety

Independent research: **Risk-Calibrated Fine-tuning** to reduce "misreading a
drug name as a different, real drug name" errors (look-alike/sound-alike,
LASA) when fine-tuning a handwritten-text OCR model for drug names, without
needing an RxNorm-scale lexicon and without needing to generate synthetic
data.

See the full research proposal: [`docs/00-de-cuong-nghien-cuu.md`](docs/00-de-cuong-nghien-cuu.md).

This project is **fully independent** of the `ocr-research` project (handwritten
prescriptions + LoRA TrOCR 7-model benchmark) — it does not reuse that
project's code/data/labels, and uses it only as research-motivation context.

## Structure

- `docs/` — research proposal, design notes
- `src/` — shared modules (data loading, confusable-pair inference, loss function...)
- `scripts/` — scripts for each step of the pipeline
- `data/raw/` — downloaded raw datasets (not committed, see `.gitignore`)
- `results/` — intermediate output (confusable-pair CSVs, training logs, metrics)
- `notebooks/` — notebooks run on Kaggle T4

## Infrastructure

Free Kaggle T4 GPU, dedicated account `bangdang007112`. Before EVERY `kaggle`
command, run:

```bash
source scripts/kaggle_env.sh
```

## Current status

- [x] Downloaded 2 public datasets: Kaggle "Doctor's Handwritten Prescription BD"
      (78 classes) and RxHandBD (~1,430 entries after normalization, mirrored on
      Kaggle from `abrahametry/rxhandbd-handwritten-word-image-dataset`).
- [x] Algorithmically inferred confusable drug-name pairs (BI-SIM: edit distance
      + phonetic similarity), exported top-100/top-5000 CSVs for manual review.
- [x] Manually checked (spot-check, AI-assisted) the confusable-pair list —
      see [docs/02-spot-check-cap-nham-lan.md](docs/02-spot-check-cap-nham-lan.md).
      **Not yet verified by a Bangladeshi pharmacist** — still something that should
      be done if possible.
- [x] Ran a reduced OCR benchmark (Tesseract + TrOCR zero-shot) independently —
      confirmed the original phenomenon (kernel 01, 02).
- [x] Implemented margin loss + severity-weighted loss into the LoRA fine-tuning
      pipeline, ran a full ablation (CE-only / +severity / +margin / +both) +
      hyperparameter search (kernel 03, 04) — see [docs/01-ket-qua-ablation.md](docs/01-ket-qua-ablation.md).
- [x] Cross-checked the winning configuration on Kaggle-BD (kernel 05) — did NOT
      replicate; the real limitation has been recorded.
- [x] Applied multiple-comparisons correction (Holm-Bonferroni) across all 7
      configurations tried — `scripts/multiple_comparisons.py`. Result: the old
      "winning configuration" is NO LONGER significant after correction
      (p_holm=0.057); only margin λ=3.0 remains significant, but it has the
      heaviest accuracy trade-off.
- [x] Reran confirmatory analysis with full seeding (torch+numpy+random),
      5 independent seeds for the pair (CE-only, winning configuration) — kernel 06.
      **The project's most important result**: the reduction in
      confusable_wrong_drug was confirmed more robustly (pooled p=0.000077,
      5/5 seeds in the same direction) — BUT the earlier "no trade-off" claim
      was WRONG: correct decreased and hallucination increased, with strong
      significance, consistently across all 5 seeds (pooled p<0.00001 for both).
      This is a REAL TRADE-OFF, not a free improvement — see the final update
      in [docs/01-ket-qua-ablation.md](docs/01-ket-qua-ablation.md).
- [x] Traced RxHandBD's provenance/license — see [docs/03-provenance-rxhandbd.md](docs/03-provenance-rxhandbd.md).
      Downloaded it ourselves + manually cross-checked against the original Zenodo
      copy (checksums match, labels match on 13/13 samples) — **confirmed the data
      is genuine**, not mixed up with another dataset (the code only reads from the
      correct `RxHandBDMain` folder). Remaining issue: Zenodo (MIT) and Mendeley
      (CC BY 4.0) have self-contradictory licenses — resolved by citing the
      original authors + following CC BY 4.0 (the stricter terms). No need to
      change datasets or rerun experiments.
- [x] Compiled a verified bibliography — [docs/04-tai-lieu-tham-khao.md](docs/04-tai-lieu-tham-khao.md).
      Fixed 1 incorrect DOI (Kondrak & Dorr 2006) and 1 incorrect content
      description (arXiv:2510.03309 is not about LASA).