"""Xây dựng từ vựng tên thuốc (đóng) từ các dataset thô, phục vụ suy luận cặp nhầm lẫn."""

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
    """Bỏ số/đơn vị liều lượng ở cuối nhãn thô (vd. 'Dexter 60' -> 'Dexter').

    Đây là bước làm sạch TỐI THIỂU, có chủ đích: không cố loại các hậu tố dạng
    bào chế (vd. 'Plus Sw', 'Max', 'Dx') vì rủi ro cắt nhầm một phần tên thuốc
    thật. Các mục còn lại (kể cả một số chuỗi có vẻ vô nghĩa) sẽ được lọc lại
    ở bước kiểm tra thủ công (spot-check) trên danh sách cặp nhầm lẫn xếp hạng
    cao nhất, theo đúng thiết kế trong docs/00-de-cuong-nghien-cuu.md mục 4.
    """
    s = raw.strip()
    s = _TRAILING_DOSAGE_RE.sub("", s).strip()
    return s


def canonicalize_for_dedup(norm_label: str) -> str:
    """Chuẩn hoá SÂU HƠN chỉ để gộp trùng lặp do khác biệt định dạng ghi nhãn
    (vd. 'Tab. Nexum' vs 'Tab.  NEXUM' vs 'Nexum' đều cùng một sản phẩm).

    Không dùng để hiển thị — chỉ dùng làm khoá gộp, vì loại dữ liệu RxHandBD
    có nhiều biến thể ghi nhãn (khoảng trắng, tiền tố dạng bào chế, hoa/thường)
    cho CÙNG một tên thuốc, dễ bị thuật toán suy luận cặp nhầm lẫn hiểu lầm là
    hai tên khác nhau (false positive) nếu không gộp trước.
    """
    s = norm_label.lower()
    s = _DOSAGE_FORM_PREFIX_RE.sub("", s)
    s = _PUNCT_WS_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def load_kaggle_bd_vocab(root: Path) -> dict[str, str]:
    """Kaggle 'Doctor's Handwritten Prescription BD' — 78 lớp, cột MEDICINE_NAME đã sạch."""
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
    """RxHandBD (RxHandBDMain) — ~1.559 mục thô, cần chuẩn hoá bỏ liều lượng và
    gộp các biến thể định dạng ghi nhãn của CÙNG một tên thuốc (xem
    canonicalize_for_dedup) trước khi đưa vào suy luận cặp nhầm lẫn — nếu
    không, top của danh sách xếp hạng sẽ bị chiếm bởi các cặp kiểu
    'Tab. Nexum' vs 'Tab.  NEXUM' (cùng 1 thuốc, khác định dạng ghi) thay vì
    các cặp nhầm lẫn LASA thật sự (hai thuốc THẬT khác nhau).
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

    print(f"Kaggle-BD: {len(kb_vocab)} tên thuốc duy nhất (kỳ vọng 78)")
    print(f"RxHandBD (đã chuẩn hoá bỏ liều lượng): {len(rx_vocab)} mục duy nhất")
    print("Mẫu RxHandBD sau chuẩn hoá:", list(rx_vocab.values())[:15])
