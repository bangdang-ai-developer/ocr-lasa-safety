"""Phan tich thong ke ghep cap (paired McNemar exact test) cho ket qua ablation.

Chay:
    PYTHONIOENCODING=utf-8 python scripts/analyze_ablation.py [dataset] [run_dir] [config1,config2,...]

Mac dinh: dataset=rxhandbd, run_dir=results/kaggle_run_3_ablation,
configs=margin,severity,severity_margin
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "kaggle_run_3_ablation"
DEFAULT_CONFIGS = ["margin", "severity", "severity_margin"]

ALL_CATEGORIES = ["correct", "minor_ocr_noise", "confusable_wrong_drug", "other_wrong_real_word", "hallucination_far_off"]


def paired_test(base_df, other_df, category):
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


def confusable_fix_destination_breakdown(base_df, other_df):
    """Trong so anh confusable_wrong_drug o baseline duoc 'sua' (khong con
    confusable) duoi cfg, di ve dau: correct (sua dung) hay chi la
    minor_ocr_noise/hallucination_far_off/other_wrong_real_word (khong con
    nham sang MOT TEN THUOC THAT KHAC, nhung cung khong dung - ve an toan
    lam sang van la mot cai thien vi de bi phat hien hon la 'don thuoc hop
    le nhung nham thuoc').
    """
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]
    o = other_df.loc[common]
    was_confusable = b["error_category"] == "confusable_wrong_drug"
    fixed_mask = was_confusable & (o["error_category"] != "confusable_wrong_drug")
    dest_counts = o.loc[fixed_mask, "error_category"].value_counts().to_dict()
    return {
        "n_was_confusable": int(was_confusable.sum()),
        "n_fixed_total": int(fixed_mask.sum()),
        "destinations": dest_counts,
    }


def overcorrection_rate(base_df, other_df):
    common = base_df.index.intersection(other_df.index)
    b = base_df.loc[common]
    o = other_df.loc[common]
    was_correct = b["error_category"] == "correct"
    now_confusable = o["error_category"] == "confusable_wrong_drug"
    overcorrect_n = int((was_correct & now_confusable).sum())
    return overcorrect_n, int(was_correct.sum())


def main(dataset: str, run_dir: Path, configs: list[str]) -> None:
    base = pd.read_csv(run_dir / f"predictions_{dataset}_ce_only.csv").set_index("image")
    print(f"\n=== {dataset} ({run_dir.name}): paired vs CE-only baseline ===")
    for cfg in configs:
        path = run_dir / f"predictions_{dataset}_{cfg}.csv"
        if not path.exists():
            print(f"-- config: {cfg}: KHONG TIM THAY {path}, bo qua --")
            continue
        other = pd.read_csv(path).set_index("image")
        print(f"\n-- config: {cfg} --")
        for cat in ["confusable_wrong_drug", "correct", "hallucination_far_off"]:
            r = paired_test(base, other, cat)
            print(
                f"  {cat:24s} base={r['base_count']:4d} {cfg}={r['other_count']:4d} "
                f"fixed={r['fixed']:3d} newly={r['newly']:3d} p={r['mcnemar_p']:.4f}"
            )

        overcorrect_n, n_correct = overcorrection_rate(base, other)
        print(
            f"  [an toan] dung->confusable moi: {overcorrect_n}/{n_correct} "
            f"({100*overcorrect_n/max(n_correct,1):.2f}%)"
        )

        breakdown = confusable_fix_destination_breakdown(base, other)
        print(
            f"  [pha vo confusable] {breakdown['n_fixed_total']}/{breakdown['n_was_confusable']} "
            f"anh confusable(baseline) khong con confusable duoi {cfg}, di ve: {breakdown['destinations']}"
        )
        n_to_correct = breakdown["destinations"].get("correct", 0)
        n_to_clearly_wrong = breakdown["n_fixed_total"] - n_to_correct
        print(
            f"    trong do: sua DUNG={n_to_correct}, chuyen thanh RO RANG SAI "
            f"(khong con la 1 ten thuoc that khac)={n_to_clearly_wrong}"
        )


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "rxhandbd"
    run_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RUN_DIR
    configs_arg = sys.argv[3].split(",") if len(sys.argv) > 3 else DEFAULT_CONFIGS
    main(dataset_arg, run_dir_arg, configs_arg)
