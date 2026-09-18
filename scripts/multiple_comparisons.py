"""Hieu chinh da so sanh (Holm-Bonferroni) tren TOAN BO cac cau hinh da thu
nghiem confusable_wrong_drug tren cung 1 tap test RxHandBD co dinh (1115 anh).

Phat hien tu vong danh gia phan bien (4 agent doc lap deu tu tinh ra ket qua
giong nhau): "cau hinh thang cuoc" (margin lambda=2.0 + severity beta=0.3,
p tho=0.011) duoc chon SAU KHI da xem ket qua cua 7 cau hinh khac tren CUNG
1 tap test - day la van de multiple-comparisons/test-set-leakage kinh dien,
can bao cao p da hieu chinh thay vi chi p tho.

Chay:
    PYTHONIOENCODING=utf-8 python scripts/multiple_comparisons.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN3 = PROJECT_ROOT / "results" / "kaggle_run_3_ablation"
RUN4 = PROJECT_ROOT / "results" / "kaggle_run_4_margin_sweep"

# (nhan hien thi, thu muc, ten file config) - CE-only baseline luon lay tu run3
CONFIGS = [
    ("severity (lam1)", RUN3, "severity"),
    ("margin (lam1)", RUN3, "margin"),
    ("severity+margin (lam1)", RUN3, "severity_margin"),
    ("margin (lam2)", RUN4, "margin_lam2"),
    ("margin (lam3)", RUN4, "margin_lam3"),
    ("margin (lam2, 5 epoch)", RUN4, "margin_lam2_ep5"),
    ("margin(lam2)+severity(beta0.3) [THANG CUOC]", RUN4, "margin_lam2_sevlow"),
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
    """Tra ve p da hieu chinh (cung thu tu voi input), theo thuat toan Holm."""
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

    print(f"So cau hinh dua vao hieu chinh (family size): {len(rows)}")
    print(f"{'Cau hinh':45s} {'base':>5s} {'other':>5s} {'fixed':>5s} {'newly':>5s} {'p_raw':>8s} {'p_holm':>8s}  ket luan (alpha=0.05)")
    for r in sorted(rows, key=lambda x: x["p_raw"]):
        ket_luan = "CON Y NGHIA" if r["p_holm"] < 0.05 else "khong con y nghia"
        print(
            f"{r['label']:45s} {r['base_n']:5d} {r['other_n']:5d} {r['fixed']:5d} {r['newly']:5d} "
            f"{r['p_raw']:8.4f} {r['p_holm']:8.4f}  {ket_luan}"
        )

    print()
    print("KET LUAN: sau hieu chinh Holm-Bonferroni tren ho 7 cau hinh da thu tren cung 1 tap")
    print("test RxHandBD, chi cau hinh nao con 'CON Y NGHIA' o tren moi duoc coi la ket qua")
    print("xac nhan duoc bang thong ke - cau hinh con lai (bao gom ca 'thang cuoc' cu) chi la")
    print("ket qua THAM DO (exploratory), can xac nhan lai bang lan chay doc lap / seed khac.")


if __name__ == "__main__":
    main()
