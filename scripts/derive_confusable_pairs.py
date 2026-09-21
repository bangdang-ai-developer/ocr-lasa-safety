"""Run confusable-pair inference for both datasets, export top-N CSV for manual review.

How to run:
    PYTHONIOENCODING=utf-8 python scripts/derive_confusable_pairs.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confusable_pairs import rank_confusable_pairs  # noqa: E402
from vocab import load_kaggle_bd_vocab, load_rxhandbd_vocab  # noqa: E402

KAGGLE_BD_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle-bd" / "prescription-bd"
RXHANDBD_ROOT = PROJECT_ROOT / "data" / "raw" / "rxhandbd" / "RxHandBDMain"
RESULTS_DIR = PROJECT_ROOT / "results"

TOP_K_EXPORT = 100  # number of pairs to export for manual review


def export_pairs(pairs, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name_a", "name_b", "score", "ortho_sim", "phon_sim", "manual_review_ok"]
        )
        for p in pairs:
            writer.writerow(
                [p.name_a, p.name_b, f"{p.score:.4f}", f"{p.ortho:.4f}", f"{p.phon:.4f}", ""]
            )


def main() -> None:
    print("=== Kaggle-BD (78 classes) ===")
    kb_vocab = load_kaggle_bd_vocab(KAGGLE_BD_ROOT)
    kb_pairs = rank_confusable_pairs(list(kb_vocab.values()))
    print(f"Vocabulary size: {len(kb_vocab)} | Pairs computed: {len(kb_pairs)}")
    print("Top 15 highest-scoring pairs:")
    for p in kb_pairs[:15]:
        print(f"  {p.name_a!r:20s} <-> {p.name_b!r:20s}  score={p.score:.3f} (ortho={p.ortho:.2f}, phon={p.phon:.2f})")
    export_pairs(kb_pairs[:TOP_K_EXPORT], RESULTS_DIR / "confusable_pairs_kaggle_bd_top100.csv")
    export_pairs(kb_pairs, RESULTS_DIR / "confusable_pairs_kaggle_bd_full.csv")

    print()
    print("=== RxHandBD ===")
    rx_vocab = load_rxhandbd_vocab(RXHANDBD_ROOT)
    rx_pairs = rank_confusable_pairs(list(rx_vocab.values()))
    print(f"Vocabulary size (normalized): {len(rx_vocab)} | Pairs computed: {len(rx_pairs)}")
    print("Top 15 highest-scoring pairs:")
    for p in rx_pairs[:15]:
        print(f"  {p.name_a!r:20s} <-> {p.name_b!r:20s}  score={p.score:.3f} (ortho={p.ortho:.2f}, phon={p.phon:.2f})")
    export_pairs(rx_pairs[:TOP_K_EXPORT], RESULTS_DIR / "confusable_pairs_rxhandbd_top100.csv")
    # RxHandBD's full list is very large (~1M pairs) -> only keep the top 5000 as a working data cache
    export_pairs(rx_pairs[:5000], RESULTS_DIR / "confusable_pairs_rxhandbd_top5000.csv")

    print()
    print(f"Exported CSVs to: {RESULTS_DIR}")
    print("REQUIRED NEXT STEP: manually review the 'manual_review_ok' column in the top100 file")
    print("before using it as the official confusable-pair list for margin loss / severity weighting.")


if __name__ == "__main__":
    main()
