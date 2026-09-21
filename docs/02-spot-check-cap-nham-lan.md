# Manual spot-check of the confusable-pair list (required per docs/00 §4/§7)

**Important transparency note**: this is an AI-assisted (Claude) spot-check, based on
general knowledge of common spelling/phonetics/pharmacology — this is **NOT** an
assessment verified by a Bangladeshi pharmacist or cross-checked against the actual
product catalog in Bangladesh. This is a first "face validity" check meant to break
part of the circularity risk (the algorithm scoring itself and then evaluating
itself), not a full substitute for that verification. Recommendation: if possible,
have someone familiar with Bangladeshi brand-name drugs review this before
submission.

## Protocol

For each pair (name_a, name_b) in the top-100 highest-scoring pairs (already
exported earlier), assess the question: *"could someone reading handwriting in a
hurry mistake name_a for name_b (or vice versa)?"* One additional category was
identified during review that needs to be separated out: **"same-brand variant"**
— a pair that differs only by a short dosage-form/combination suffix (e.g.
`Calboral` vs `Calboral D`, `Fenix` vs `Fenix-V`, `Spilac` vs `Spilac F`) — this is
**NOT** two independent products that happen to look alike by coincidence, but
rather an intentional variant of the SAME parent brand. The inference algorithm
cannot distinguish this category from a genuine LASA confusable pair.

3 labels used: `co` (yes, plausible), `khong` (no/questionable — most likely
algorithmic noise or a labeling error in the dataset), `bien_the_cung_nhan_hieu`
(same-brand variant, automatically detected via a short-suffix heuristic of ≤3
characters).

## Results

| Dataset | n | co (plausible) | khong (questionable) | same-brand variant |
|---|---|---|---|---|
| Kaggle-BD (top 100) | 100 | 38 | 61 | 1 |
| RxHandBD (top 100) | 100 | 42 | 40 | 18 |

**Key finding — checking the `CONFUSABLE_THRESHOLD=0.65` threshold currently used
in the pipeline**:

- **Kaggle-BD**: at exactly the production threshold (score ≥ 0.65, n=32): the
  plausibility rate is **22/32 = 68.75%**. Below the threshold (score < 0.65,
  n=68): the plausibility rate drops to **16/68 = 23.5%**. → **The 0.65 threshold
  works reasonably well for Kaggle-BD**, filtering out most of the noise.
- **RxHandBD**: the entire exported top-100 already has score ≥ 0.65 (no sampling
  around the threshold boundary has been done yet), so the quality of the
  threshold cannot yet be assessed from this data. Within the top-100 (all of
  which are ≥ the threshold): only **42% are genuinely plausible as two
  independent brands**, **18% are same-brand variants** (a real risk but of a
  different nature), and **40% appear to be algorithmic noise/dataset labeling
  errors**.

## Implications to state in the paper (real limitations, not concealed)

1. For RxHandBD — the primary dataset used for the statistically significant
   results — about **4/10 of the top-scoring confusable pairs are noise**, not
   genuine LASA pairs. The neighbor list N(w) used for the margin loss (built from
   the same algorithm) therefore likely also contains a similar proportion of
   noise — this should be stated explicitly as a limitation of the method, not as
   a fully verified assumption.
2. **Specific improvement for the next round** (not yet applied to the
   already-run kernels 03-06, since the confirmatory kernels ran before this
   finding became clear): automatically remove "same-brand variant" pairs (short
   suffix ≤3 characters after stripping whitespace/punctuation) from N(w) and from
   the taxonomy labels — the heuristic has already been written; see the
   `is_family_variant` function in the spot-check script.
3. The 42-60% agreement rate (depending on whether same-brand variants are
   counted as "plausible" or not) should be reported transparently in the Methods
   section, with a clear note that this is an AI-assisted review that has not yet
   been verified by a local pharmacist.

CSV files with the `manual_review_ok` column filled in:
`results/confusable_pairs_kaggle_bd_top100.csv`,
`results/confusable_pairs_rxhandbd_top100.csv`.