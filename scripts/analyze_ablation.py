"""Phan tich thong ke ghep cap (paired McNemar exact test) cho ket qua ablation
(results/kaggle_run_3_ablation/). Chay:
    PYTHONIOENCODING=utf-8 python scripts/analyze_ablation.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

RUN_DIR = Path(__file__).resolve().parent.parent / "results" / "kaggle_run_3_ablation"


def paired_test(base_df, other_df, category, on="image"):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]
    o = other_df.loc[common]
    b_flag = b["error_category"] == category
    o_flag = o["error_category"] == category
    fixed = (b_flag & ~o_flag).sum()
    newly = (~b_flag & o_flag).sum()
    n = fixed + newly
    p = binomtest(fixed, n, 0.5).pvalue if n else float("nan")
    return {
        "category": category,
        "base_count": int(b_flag.sum()),
        "other_count": int(o_flag.sum()),
        "fixed": int(fixed),
        "newly": int(newly),
        "mcnemar_p": p,
    }


def main(dataset: str) -> None:
    base = pd.read_csv(RUN_DIR / f"predictions_{dataset}_ce_only.csv").set_index("image")
    print(f"\n=== {dataset}: paired vs CE-only baseline ===")
    for cfg in ["margin", "severity", "severity_margin"]:
        other = pd.read_csv(RUN_DIR / f"predictions_{dataset}_{cfg}.csv").set_index("image")
        print(f"\n-- config: {cfg} --")
        for cat in ["confusable_wrong_drug", "correct", "hallucination_far_off"]:
            r = paired_test(base, other, cat)
            print(
                f"  {cat:24s} base={r['base_count']:4d} {cfg}={r['other_count']:4d} "
                f"fixed={r['fixed']:3d} newly={r['newly']:3d} p={r['mcnemar_p']:.4f}"
            )

        # metric an toan cot loi: dung o CE-only -> bi bien thanh confusable duoi cfg
        common = base.index.intersection(other.index)
        b = base.loc[common]
        o = other.loc[common]
        was_correct = b["error_category"] == "correct"
        now_confusable = o["error_category"] == "confusable_wrong_drug"
        overcorrect_n = int((was_correct & now_confusable).sum())
        print(
            f"  [an toan] dung->confusable moi: {overcorrect_n}/{int(was_correct.sum())} "
            f"({100*overcorrect_n/max(was_correct.sum(),1):.2f}%)"
        )


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "rxhandbd"
    main(dataset_arg)
