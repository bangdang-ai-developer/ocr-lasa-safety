"""Multiple-comparisons correction (Holm-Bonferroni) across ALL configurations tested
for confusable_wrong_drug on the same fixed RxHandBD test set (1115 images).

Discovered during the adversarial review round (4 independent agents all computed
the same result): the "winning-ticket configuration" (margin lambda=2.0 + severity
beta=0.3, raw p=0.011) was selected AFTER looking at the results of 7 other
configurations on the SAME test set - this is the classic multiple-comparisons /
test-set-leakage problem, and the corrected p-value should be reported instead of
just the raw p-value.

Run:
    PYTHONIOENCODING=utf-8 python scripts/multiple_comparisons.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN3 = PROJECT_ROOT / "results" / "kaggle_run_3_ablation"
RUN4 = PROJECT_ROOT / "results" / "kaggle_run_4_margin_sweep"

# (display label, directory, config file name) - CE-only baseline is always taken from run3
CONFIGS = [
    ("severity (lam1)", RUN3, "severity"),
    ("margin (lam1)", RUN3, "margin"),
    ("severity+margin (lam1)", RUN3, "severity_margin"),
    ("margin (lam2)", RUN4, "margin_lam2"),
    ("margin (lam3)", RUN4, "margin_lam3"),
    ("margin (lam2, 5 epoch)", RUN4, "margin_lam2_ep5"),
    ("margin(lam2)+severity(beta0.3) [WINNING TICKET]", RUN4, "margin_lam2_sevlow"),
]


def mcnemar_p(base_df, other_df, category="confusable_wrong_drug"):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]
    o = other_df.loc[common]
    b_flag = b["error_category"] == category
    o_flag = o["error_category"] == category
    fixed = int((b_flag & ~o_flag).sum())
    newly = int((~b_flag & o_flag).sum())
    n = fixed + newly
    p = binomtest(fixed, n, 0.5).pvalue if n else float("nan")
    return fixed, newly, p, int(b_flag.sum()), int(o_flag.sum())


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Return the adjusted p-values (same order as input), following the Holm algorithm."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):  # rank: 0-indexed
        val = min((m - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def main() -> None:
    base = pd.read_csv(RUN3 / "predictions_rxhandbd_ce_only.csv").set_index("image")

    rows = []
    for label, run_dir, cfg in CONFIGS:
        other = pd.read_csv(run_dir / f"predictions_rxhandbd_{cfg}.csv").set_index("image")
        fixed, newly, p, base_n, other_n = mcnemar_p(base, other)
        rows.append(
            {"label": label, "base_n": base_n, "other_n": other_n, "fixed": fixed, "newly": newly, "p_raw": p}
        )

    p_raw = [r["p_raw"] for r in rows]
    p_holm = holm_bonferroni(p_raw)
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph

    print(f"Number of configurations included in the correction (family size): {len(rows)}")
    print(f"{'Configuration':45s} {'base':>5s} {'other':>5s} {'fixed':>5s} {'newly':>5s} {'p_raw':>8s} {'p_holm':>8s}  conclusion (alpha=0.05)")
    for r in sorted(rows, key=lambda x: x["p_raw"]):
        ket_luan = "STILL SIGNIFICANT" if r["p_holm"] < 0.05 else "no longer significant"
        print(
            f"{r['label']:45s} {r['base_n']:5d} {r['other_n']:5d} {r['fixed']:5d} {r['newly']:5d} "
            f"{r['p_raw']:8.4f} {r['p_holm']:8.4f}  {ket_luan}"
        )

    print()
    print("CONCLUSION: after Holm-Bonferroni correction across the family of 7 configurations tried on")
    print("the same RxHandBD test set, only a configuration that remains 'STILL SIGNIFICANT' above can be")
    print("considered statistically confirmed - the remaining configurations (including the old 'winning")
    print("ticket') are only EXPLORATORY results and need to be re-confirmed with an independent run / different seed.")


if __name__ == "__main__":
    main()
