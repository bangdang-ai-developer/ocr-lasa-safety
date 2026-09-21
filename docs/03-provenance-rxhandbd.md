# Provenance/License Tracing for RxHandBD

**Important update (personally downloaded + directly cross-checked the original files from Zenodo)**:
the situation is **much better** than the initial assessment. The license issue remains,
but there is NO LONGER any doubt about whether the data is mixed up/fake.

## Directly self-verified (not just based on webpage metadata)

Personally downloaded both files directly from Zenodo (`RxHand Original.zip`, `RxHandBD.zip`),
confirmed the MD5 checksums match 100% with what Zenodo publishes, then manually cross-checked
against the data currently used in the project (`data/raw/rxhandbd/RxHandBDMain/`):

1. **Cross-checked text labels against the corresponding image IDs — matched on 13/13 random samples**
   (P0001→"Nexcital", P0100→"Etorix", P1000→"celebrex", P0050→"CORTAN",
   P2000→"Napa extra", P4200→"Ceevit", etc. — identical between the official Zenodo version
   and the Kaggle version currently in use). **→ Confirmed with certainty: this is genuinely the real
   RxHandBD dataset by Md. Masudul Islam (BUBT), not fake data or someone else's data.**
2. **Confirmed the code has NEVER touched the "mistakenly merged" folder**: all
   kernels (01-06) use a glob pattern that requires the exact folder name `RxHandBDMain`
   (`glob.glob(.../ "RxHandBDMain" / "Test.csv")`) — meaning that even though the Kaggle download
   bundles in an additional `BD Handwritten Prescription Dataset` folder (a different dataset,
   different license) as part of the same download, **the code never reads from that folder
   for the experiments labeled "rxhandbd"**. The "kaggle_bd" dataset in the project
   was downloaded SEPARATELY, directly from its own original Kaggle page (mamun1113), which had a
   clear license (Open Database License) from the start. → **There is no data mixing
   between the 2 "datasets" in the experimental results that were run.**
3. **The remaining difference is only in image processing, not content**: the image files in the
   Kaggle version currently in use are 128×64 RGB (normalized, not "raw" arbitrary
   sizes as reported earlier — the earlier measurement of "130-230×55-105px" was WRONG,
   20 samples were re-measured, giving a consistent result of 128×64). Both official Zenodo
   files are 128×128 grayscale (different in both size and color). → Whoever
   prepared the Kaggle version resized/processed the images their own way, NOT using the raw
   Zenodo/Mendeley files as-is — but the content (handwriting images + labels) is indeed
   still the same original dataset, only the image-processing step differs.

## License issue — still needs to be addressed, but much simpler now

Zenodo (MIT) and Mendeley (CC BY 4.0) still contradict each other on the license for the same
data by the same author — this is an inconsistency from the original author themselves,
not a fault of the Kaggle version. Since the content has been confirmed genuine, the **simple
solution**: cite the original directly (recommended: Mendeley DOI
10.17632/dsb5r6vskg.3, the latest v3), give full author credit, and comply with
the CC BY 4.0 terms (the stricter set of terms — complying with this
remains valid whether the real license is MIT or CC-BY).

## Updated recommendations (replacing the earlier recommendations)

1. **No need to change datasets, no need to re-run experiments** — the content has been
   verified to be genuine and correct.
2. In the **Data Availability statement**: state clearly "the images used were obtained via a
   mirror on Kaggle (specific URL); the image-ID↔label correspondence was independently cross-checked
   against the original published by Md. Masudul Islam (BUBT) on Mendeley Data (DOI
   10.17632/dsb5r6vskg.3, CC BY 4.0) and Zenodo (DOI 10.5281/zenodo.18478741,
   MIT) — matching 100% on the checked sample; the specific image processing/resizing step used
   for the Kaggle version is not documented independently of the original." Cite the original author,
   comply with CC BY 4.0.
3. **Still do NOT redistribute the image files directly** in the project's public code release
   (even though the source is now clear, readers should still be pointed to the original source
   rather than re-hosting the files, in keeping with the spirit of CC BY 4.0 — attribution + leaving
   the source intact).
4. No need to downgrade RxHandBD's role in the paper anymore — the dataset has been verified as
   genuine, it only needs to be cited correctly and be transparent about the image-processing step not being independently documented.
5. (Not required) It is still possible to contact the author to clarify which license is
   official, but this is no longer a blocking issue.

## Sources personally verified
- Zenodo: https://zenodo.org/records/18478741 (downloaded, MD5 matches)
- Mendeley: https://data.mendeley.com/datasets/dsb5r6vskg/3
- Kaggle version currently in use: https://www.kaggle.com/datasets/abrahametry/rxhandbd-handwritten-word-image-dataset