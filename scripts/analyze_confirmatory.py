"""Pooled analysis across 5 independent seeds (kernel 06_confirmatory_reruns):
CE-only vs the "winning" configuration (margin lambda=2.0 + severity beta=0.3),
same seeds for both (overlap comes from LoRA init/shuffle order - see the
kernel 06 docstring). This is the confirmatory step after Holm-Bonferroni
correction (scripts/multiple_comparisons.py) showed that the earlier
single-run result did NOT reach significance after correction.

Run:
    PYTHONIOENCODING=utf-8 python scripts/analyze_confirmatory.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

RUN_DIR = Path(__file__).resolve().parent.parent / "results" / "kaggle_run_6_confirmatory"
SEEDS = [1, 2, 3, 4, 5]
CATEGORIES = ["confusable_wrong_drug", "correct", "hallucination_far_off"]


def paired_fixed_newly(base_df, other_df, category):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]["error_category"] == category
    o = other_df.loc[common]["error_category"] == category
    fixed = int((b & ~o).sum())
    newly = int((~b & o).sum())
    return fixed, newly, int(b.sum()), int(o.sum())


def main() -> None:
    per_seed = {cat: [] for cat in CATEGORIES}
    print("=== Per seed (CE-only vs winning, same seed) ===")
    for seed in SEEDS:
        base = pd.read_csv(RUN_DIR / f"predictions_rxhandbd_seed{seed}_ce_only.csv").set_index("image")
        other = pd.read_csv(RUN_DIR / f"predictions_rxhandbd_seed{seed}_winning.csv").set_index("image")
        print(f"\n-- seed {seed} --")
        for cat in CATEGORIES:
            fixed, newly, b_n, o_n = paired_fixed_newly(base, other, cat)
            n = fixed + newly
            p = binomtest(fixed, n, 0.5).pvalue if n else float("nan")
            per_seed[cat].append((fixed, newly))
            direction = "DOWN" if o_n < b_n else ("UP" if o_n > b_n else "no change")
            print(f"  {cat:24s} ce_only={b_n:4d} winning={o_n:4d} ({direction}) fixed={fixed:3d} newly={newly:3d} p={p:.4f}")

    print("\n=== POOLED ACROSS 5 SEEDS (pooled McNemar exact) ===")
    n_sig_uncorrected = {cat: 0 for cat in CATEGORIES}
    for cat in CATEGORIES:
        total_fixed = sum(f for f, n in per_seed[cat])
        total_newly = sum(n for f, n in per_seed[cat])
        total_n = total_fixed + total_newly
        p = binomtest(total_fixed, total_n, 0.5).pvalue if total_n else float("nan")
        n_sig = sum(1 for f, n in per_seed[cat] if n and binomtest(f, f + n, 0.5).pvalue < 0.05)
        direction = "DOWN (fixed>newly)" if total_fixed > total_newly else "UP (newly>fixed)"
        print(
            f"  {cat:24s} fixed={total_fixed:4d} newly={total_newly:4d} total_diff={total_n:4d} "
            f"p_pooled={p:.6f} ({direction}) | {n_sig}/5 individual seeds reach p<0.05"
        )

    print("\nCONCLUSION:")
    print("- confusable_wrong_drug: ROBUST decrease across all 5 seeds (5/5 same direction), pooled p very small.")
    print("- correct & hallucination: BOTH change consistently across all 5 seeds in an UNFAVORABLE direction")
    print("  (correct decreases, hallucination increases) - contradicting the 'no trade-off' conclusion from the single")
    print("  earlier run (p=0.76/0.41) - that run was a favorable outlier, NOT representative.")
    print("  The paper's claim needs to be corrected: this is a REAL trade-off, not a free lunch.")


if __name__ == "__main__":
    main()
