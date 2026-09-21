# Verified References

All citations below have been verified as genuine (not AI-fabricated) through direct lookup on arXiv/ACL Anthology/CrossRef/PubMed/SpringerLink — not quoted from memory. Format: simplified APA.

## Main loss mechanism (severity-weighted CE)

- Mohammadi-Seif, A., & Baeza-Yates, R. (2026). *Risk-Calibrated Learning:
  Minimizing Fatal Errors in Medical AI*. arXiv:2604.12693 [cs.CV]. Accepted
  at IJCNN 2026. https://arxiv.org/abs/2604.12693
  — Confirmed as described: embeds a clinical severity-level matrix into the
  loss for medical image classification (Brain Tumor MRI, ISIC 2018, BreaKHis,
  SICAPv2), reducing the Critical Error Rate by 20.0-92.4% compared to Focal
  Loss. **This paper addresses a single-label classification problem** — this
  project adapts it to a sequence-generation (OCR) context, and this must be
  explicitly stated as an "adaptation," not a direct reproduction.

## Closely related papers (must be cited to distinguish, so reviewers do not consider them overlooked)

- Alansary, A., Mohamed, M., & Hamdi, A. (2026). *Severity-Aware Weighted Loss
  for Arabic Medical Text Generation*. arXiv:2604.06346 [cs.CL]. Accepted at
  ICTIS 2026. https://arxiv.org/abs/2604.06346
  — Token-level severity-weighting for Arabic medical dialogue text generation
  (LLM), NOT OCR/vision. ⚠️ Caution: there is another paper with a very close
  ID (arXiv:2604.06365, curriculum learning, different method) — do not
  confuse the two.
- Tupakula, M. (2025). *Thin Bridges for Drug Text Alignment: Lightweight
  Contrastive Learning for Target Specific Drug Retrieval*. arXiv:2510.03309.
  https://arxiv.org/abs/2510.03309
  — ⚠️ **Previous description was WRONG**: this paper is NOT about
  Look-Alike Sound-Alike (LASA) drug names. It is contrastive learning that
  aligns molecular embeddings (ECFP4) with biomedical text (ChEMBL) for
  target-based drug retrieval, using margin loss + hard negatives, but for an
  entirely different purpose. If cited, the description must be corrected —
  or it should be removed from related work, as it is not actually relevant
  to LASA.

## Biomedical entity normalization using contrastive/hard-negative learning (text-only, no OCR)

- Sung, M., Jeon, H., Lee, J., & Kang, J. (2020). *Biomedical Entity
  Representations with Synonym Marginalization*. ACL 2020, pp. 3641–3650.
  DOI: 10.18653/v1/2020.acl-main.335. https://aclanthology.org/2020.acl-main.335/
- Yuan, Z., Zhao, Z., Sun, H., Li, J., Wang, F., & Yu, S. (2022). *CODER:
  Knowledge-infused cross-lingual medical term embedding for term
  normalization*. Journal of Biomedical Informatics, 126, 103983.
  DOI: 10.1016/j.jbi.2021.103983. (arXiv:2011.02947)

## Contrastive learning for sequence generation/text recognition (general, not targeting specific confusable pairs)

- Aberdam, A., Litman, R., Tsiper, S., Anschel, O., Slossberg, R., Mazor, S.,
  Manmatha, R., & Perona, P. (2021). *Sequence-to-Sequence Contrastive
  Learning for Text Recognition*. CVPR 2021. arXiv:2012.10873.
- Zhang, J., Lin, T., Xu, Y., Chen, K., & Zhang, R. (2023). *Relational
  Contrastive Learning for Scene Text Recognition*. ACM MM 2023.
  arXiv:2308.00508.
- Lee, S., Lee, D. B., & Hwang, S. J. (2021). *Contrastive Learning with
  Adversarial Perturbations for Conditional Text Generation*. ICLR 2021.
  arXiv:2012.07280. (Note: this paper is about MT/summarization text
  generation, not OCR.)

## Algorithm for inferring confusable drug name pairs (foundation for src/confusable_pairs.py)

- Kondrak, G., & Dorr, B. (2004). *Identification of Confusable Drug Names: A
  New Approach and Evaluation Methodology*. COLING 2004, pp. 952–958.
  DOI: 10.3115/1220355.1220492.
- Kondrak, G., & Dorr, B. (2006). *Automatic identification of confusable drug
  names*. Artificial Intelligence in Medicine, 36(1), 29–42.
  **Correct DOI: 10.1016/j.artmed.2005.07.005** (the previously guessed DOI
  10.1016/j.artmed.2005.06.001 was WRONG — it pointed to a different paper by
  mistake).
- Millán-Hernández, C. E., García-Hernández, R. A., Ledeneva, Y., &
  Hernández-Castañeda, Á. (2019). *Soft Bigram Similarity to Identify
  Confusable Drug Names*. MCPR 2019, LNCS vol. 11524, pp. 433–442. Springer.
  DOI: 10.1007/978-3-030-21077-9_40.
  (Note: one co-author was previously missing from the internal notes — has
  been added.)

## Additional work needed before submission

- Manually check the full abstract of Millán-Hernández et al. 2019
  (SpringerLink requires login for automated lookup) before citing the
  specific figures ("outperforms 17 other metrics on 396.900 pairs") in the
  manuscript.
- Fix every place that has mistakenly described arXiv:2510.03309 as being
  about "confusable drug names."