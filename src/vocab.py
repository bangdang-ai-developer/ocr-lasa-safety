"""Build a closed drug-name vocabulary from the raw datasets, for confusable-pair inference."""

import csv
import re
from pathlib import Path

_TRAILING_DOSAGE_RE = re.compile(
    r"\s*\d+(\.\d+)?\s*(mg|ml|gm|g|mcg|iu)?\s*$", re.IGNORECASE
)
_DOSAGE_FORM_PREFIX_RE = re.compile(
    r"^(tab\.?|cap\.?|inj\.?|syp\.?|susp\.?)\s*", re.IGNORECASE
)
_PUNCT_WS_RE = re.compile(r"[.\-]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_label(raw: str) -> str:
    """Strip trailing dosage numbers/units from a raw label (e.g. 'Dexter 60' -> 'Dexter').

    This is a deliberately MINIMAL cleaning step: it does not try to strip
    dosage-form suffixes (e.g. 'Plus Sw', 'Max', 'Dx') because of the risk of
    cutting off part of a real drug name. Remaining entries (including some
    that look meaningless) get filtered out later during the manual spot-check
    of the top-ranked confusable-pair list, per the design in
    docs/00-de-cuong-nghien-cuu.md section 4.
    """
    s = raw.strip()
    s = _TRAILING_DOSAGE_RE.sub("", s).strip()
    return s


def canonicalize_for_dedup(norm_label: str) -> str:
    """Deeper normalization used only to merge duplicates caused by
    labeling-format differences (e.g. 'Tab. Nexum' vs 'Tab.  NEXUM' vs 'Nexum'
    are all the same product).

    Not meant for display — only used as a merge key, since the RxHandBD
    dataset has many labeling variants (whitespace, dosage-form prefixes,
    case) for the SAME drug name, which the confusable-pair inference
    algorithm could easily mistake for two different names (false positive)
    if they aren't merged first.
    """
    s = norm_label.lower()
    s = _DOSAGE_FORM_PREFIX_RE.sub("", s)
    s = _PUNCT_WS_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def load_kaggle_bd_vocab(root: Path) -> dict[str, str]:
    """Kaggle 'Doctor's Handwritten Prescription BD' — 78 classes, MEDICINE_NAME column already clean."""
    vocab: dict[str, str] = {}
    for split, fname in [
        ("Training", "training_labels.csv"),
        ("Validation", "validation_labels.csv"),
        ("Testing", "testing_labels.csv"),
    ]:
        path = root / split / fname
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["MEDICINE_NAME"].strip()
                if name:
                    vocab.setdefault(name.lower(), name)
    return vocab


def load_rxhandbd_vocab(root: Path) -> dict[str, str]:
    """RxHandBD (RxHandBDMain) — ~1,559 raw entries; needs dosage stripping
    and merging of labeling-format variants for the SAME drug name (see
    canonicalize_for_dedup) before feeding into confusable-pair inference —
    otherwise the top of the ranked list would be dominated by pairs like
    'Tab. Nexum' vs 'Tab.  NEXUM' (same drug, different label formatting)
    instead of genuine LASA confusable pairs (two different REAL drugs).
    """
    vocab: dict[str, str] = {}
    for fname in ["Train.csv", "Test.csv"]:
        path = root / fname
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = row.get("Labels", "").strip()
                if not raw:
                    continue
                norm = normalize_label(raw)
                if len(norm) < 3:
                    continue
                dedup_key = canonicalize_for_dedup(norm)
                if not dedup_key:
                    continue
                vocab.setdefault(dedup_key, norm)
    return vocab


if __name__ == "__main__":
    kaggle_root = Path(
        r"C:\Users\BangDang\Project\ocr-lasa-safety\data\raw\kaggle-bd\prescription-bd"
    )
    rxhandbd_root = Path(
        r"C:\Users\BangDang\Project\ocr-lasa-safety\data\raw\rxhandbd\RxHandBDMain"
    )

    kb_vocab = load_kaggle_bd_vocab(kaggle_root)
    rx_vocab = load_rxhandbd_vocab(rxhandbd_root)

    print(f"Kaggle-BD: {len(kb_vocab)} unique drug names (expected 78)")
    print(f"RxHandBD (dosage-normalized): {len(rx_vocab)} unique entries")
    print("RxHandBD sample after normalization:", list(rx_vocab.values())[:15])
