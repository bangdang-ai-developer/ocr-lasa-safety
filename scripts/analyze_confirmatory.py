"""Phan tich gop (pooled) tren 5 seed doc lap (kernel 06_confirmatory_reruns):
CE-only vs cau hinh "thang cuoc" (margin lambda=2.0 + severity beta=0.3),
cung seed cho ca 2 (co lap nhieu do LoRA init/thu tu shuffle - xem docstring
kernel 06). Day la buoc xac nhan (confirmatory) sau khi hieu chinh Holm-
Bonferroni (scripts/multiple_comparisons.py) cho thay ket qua 1-lan-chay
truoc do KHONG dat y nghia sau hieu chinh.

Chay:
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
    print("=== Tung seed (CE-only vs thang cuoc, cung seed) ===")
    for seed in SEEDS:
        base = pd.read_csv(RUN_DIR / f"predictions_rxhandbd_seed{seed}_ce_only.csv").set_index("image")
        other = pd.read_csv(RUN_DIR / f"predictions_rxhandbd_seed{seed}_winning.csv").set_index("image")
        print(f"\n-- seed {seed} --")
        for cat in CATEGORIES:
            fixed, newly, b_n, o_n = paired_fixed_newly(base, other, cat)
            n = fixed + newly
            p = binomtest(fixed, n, 0.5).pvalue if n else float("nan")
            per_seed[cat].append((fixed, newly))
            direction = "GIAM" if o_n < b_n else ("TANG" if o_n > b_n else "khong doi")
            print(f"  {cat:24s} ce_only={b_n:4d} winning={o_n:4d} ({direction}) fixed={fixed:3d} newly={newly:3d} p={p:.4f}")

    print("\n=== GOP 5 SEED (pooled McNemar exact) ===")
    n_sig_uncorrected = {cat: 0 for cat in CATEGORIES}
    for cat in CATEGORIES:
        total_fixed = sum(f for f, n in per_seed[cat])
        total_newly = sum(n for f, n in per_seed[cat])
        total_n = total_fixed + total_newly
        p = binomtest(total_fixed, total_n, 0.5).pvalue if total_n else float("nan")
        n_sig = sum(1 for f, n in per_seed[cat] if n and binomtest(f, f + n, 0.5).pvalue < 0.05)
        direction = "GIAM (fixed>newly)" if total_fixed > total_newly else "TANG (newly>fixed)"
        print(
            f"  {cat:24s} fixed={total_fixed:4d} newly={total_newly:4d} tong_khac_biet={total_n:4d} "
            f"p_gop={p:.6f} ({direction}) | {n_sig}/5 seed rieng le dat p<0.05"
        )

    print("\nKET LUAN:")
    print("- confusable_wrong_drug: giam ROBUST qua ca 5 seed (5/5 cung huong), p gop rat nho.")
    print("- correct & hallucination: CUNG thay doi nhat quan qua ca 5 seed theo huong BAT LOI")
    print("  (giam correct, tang hallucination) - khac voi ket luan 'khong danh doi' cua 1 lan")
    print("  chay truoc do (p=0.76/0.41) - lan chay do la mot outlier thuan loi, KHONG dai dien.")
    print("  Can sua lai tuyen bo trong bai bao: day la MOT DANH DOI THAT, khong phai mien phi.")


if __name__ == "__main__":
    main()
