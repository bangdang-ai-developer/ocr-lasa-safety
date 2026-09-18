"""Phan tich mo rong cho ket qua xac nhan 5-seed (kernel 06): effect size (risk
difference) + 95% CI, va breakdown "confusable duoc sua thi di ve dau" gop tren
ca 5 seed (giong da lam rieng le cho kernel 03/04 trong analyze_ablation.py).

Ly do can them: scripts/analyze_confirmatory.py hien chi bao cao p-value
(McNemar exact, gop 5 seed). Bai bao Q2 journal thuong doi hoi effect size +
khoang tin cay di kem p-value, khong chi p-value don le.

Phuong phap risk difference + CI: dung cong thuc Wald chuan cho hieu 2 ti le
GHEP CAP (paired proportions), vd Newcombe (1998) / Fagerland et al.:
    p10 = fixed/n, p01 = newly/n, d = p01 - p10
    Var(d) = [p10 + p01 - (p01-p10)^2] / n
Gop 5 seed bang cach CONG don fixed/newly/n truoc khi tinh d va CI - nhat quan
voi cach da dung de tinh p-value gop trong analyze_confirmatory.py (cong don
roi chay 1 binomtest), tuong duong gia dinh fixed-effect qua cac seed (hop ly
vi n moi seed gan bang nhau va seed la doc lap/hoan vi duoc).

Chay:
    PYTHONIOENCODING=utf-8 python scripts/analyze_confirmatory_extended.py
"""

import math
from pathlib import Path

import pandas as pd
from scipy.stats import norm

RUN_DIR = Path(__file__).resolve().parent.parent / "results" / "kaggle_run_6_confirmatory"
SEEDS = [1, 2, 3, 4, 5]
CATEGORIES = ["confusable_wrong_drug", "correct", "hallucination_far_off"]

Z95 = norm.ppf(0.975)


def paired_counts(base_df, other_df, category):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]["error_category"] == category
    o = other_df.loc[common]["error_category"] == category
    fixed = int((b & ~o).sum())
    newly = int((~b & o).sum())
    n = len(common)
    return fixed, newly, n, int(b.sum()), int(o.sum())


def risk_diff_ci(fixed: int, newly: int, n: int):
    p10 = fixed / n
    p01 = newly / n
    d = p01 - p10
    var_d = (p10 + p01 - (p01 - p10) ** 2) / n
    se = math.sqrt(max(var_d, 0.0))
    return d, d - Z95 * se, d + Z95 * se


def wilson_ci(k: int, n: int):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / denom
    half = (Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2))) / denom
    return p, center - half, center + half


def confusable_fix_destination(base_df, other_df):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]
    o = other_df.loc[common]
    was_confusable = b["error_category"] == "confusable_wrong_drug"
    fixed_mask = was_confusable & (o["error_category"] != "confusable_wrong_drug")
    dest_counts = o.loc[fixed_mask, "error_category"].value_counts().to_dict()
    return int(was_confusable.sum()), int(fixed_mask.sum()), dest_counts


def overcorrection(base_df, other_df):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]
    o = other_df.loc[common]
    was_correct = b["error_category"] == "correct"
    now_confusable = o["error_category"] == "confusable_wrong_drug"
    return int((was_correct & now_confusable).sum()), int(was_correct.sum())


def main() -> None:
    pooled = {cat: {"fixed": 0, "newly": 0, "n": 0} for cat in CATEGORIES}
    pooled_destinations: dict[str, int] = {}
    pooled_was_confusable = 0
    pooled_fixed_total = 0
    pooled_overcorrect = 0
    pooled_was_correct = 0

    print("=== Effect size (risk difference, winning - CE_only) + 95% CI moi seed ===")
    for seed in SEEDS:
        base = pd.read_csv(RUN_DIR / f"predictions_rxhandbd_seed{seed}_ce_only.csv").set_index("image")
        other = pd.read_csv(RUN_DIR / f"predictions_rxhandbd_seed{seed}_winning.csv").set_index("image")
        print(f"\n-- seed {seed} --")
        for cat in CATEGORIES:
            fixed, newly, n, b_n, o_n = paired_counts(base, other, cat)
            d, lo, hi = risk_diff_ci(fixed, newly, n)
            pooled[cat]["fixed"] += fixed
            pooled[cat]["newly"] += newly
            pooled[cat]["n"] += n
            print(
                f"  {cat:24s} rate_ce={b_n/n:.4f} rate_win={o_n/n:.4f} "
                f"risk_diff={d:+.4f} 95%CI=[{lo:+.4f}, {hi:+.4f}]"
            )

        was_conf, fixed_total, dests = confusable_fix_destination(base, other)
        pooled_was_confusable += was_conf
        pooled_fixed_total += fixed_total
        for k, v in dests.items():
            pooled_destinations[k] = pooled_destinations.get(k, 0) + v

        overcorrect_n, was_correct_n = overcorrection(base, other)
        pooled_overcorrect += overcorrect_n
        pooled_was_correct += was_correct_n

    print("\n=== GOP 5 SEED: effect size + 95% CI (risk difference, winning - CE_only) ===")
    for cat in CATEGORIES:
        fixed = pooled[cat]["fixed"]
        newly = pooled[cat]["newly"]
        n = pooled[cat]["n"]
        d, lo, hi = risk_diff_ci(fixed, newly, n)
        print(
            f"  {cat:24s} risk_diff={d:+.4f} ({d*100:+.2f} diem %) "
            f"95%CI=[{lo*100:+.2f}, {hi*100:+.2f}] diem % (n={n})"
        )

    print("\n=== GOP 5 SEED: confusable_wrong_drug 'duoc sua' (het confusable) thi di ve dau ===")
    n_to_correct = pooled_destinations.get("correct", 0)
    n_to_clearly_wrong = pooled_fixed_total - n_to_correct
    print(f"  Tong so anh confusable duoi CE-only (gop 5 seed): {pooled_was_confusable}")
    print(
        f"  Tong so 'het confusable' duoi cau hinh thang cuoc: {pooled_fixed_total} "
        f"({100*pooled_fixed_total/pooled_was_confusable:.1f}% cua so confusable ban dau)"
    )
    print(f"  Chi tiet diem den: {pooled_destinations}")
    print(
        f"  -> Sua DUNG hoan toan: {n_to_correct} "
        f"({100*n_to_correct/pooled_fixed_total:.1f}% cua so 'het confusable')"
    )
    print(
        f"  -> Chuyen thanh RO RANG SAI (khong con la mot ten thuoc that khac): {n_to_clearly_wrong} "
        f"({100*n_to_clearly_wrong/pooled_fixed_total:.1f}% cua so 'het confusable')"
    )

    print("\n=== GOP 5 SEED: ty le 'qua tay' (dung o CE-only -> confusable MOI o thang cuoc) ===")
    p, lo, hi = wilson_ci(pooled_overcorrect, pooled_was_correct)
    print(
        f"  {pooled_overcorrect}/{pooled_was_correct} = {p*100:.2f}% "
        f"(Wilson 95% CI: [{lo*100:.2f}%, {hi*100:.2f}%])"
    )


if __name__ == "__main__":
    main()
