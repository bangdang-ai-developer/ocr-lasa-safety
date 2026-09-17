"""Suy luận cặp tên thuốc dễ nhầm lẫn (LASA) bằng thuật toán tổng quát
(không dùng ISMP/RxNorm — không khớp thị trường Bangladesh).

Phương pháp: kết hợp độ tương đồng chính tả (normalized edit distance) và
độ tương đồng ngữ âm (metaphone code + Jaro-Winkler), theo tinh thần BI-SIM
của Kondrak & Dorr, "Automatic identification of confusable drug names"
(Artificial Intelligence in Medicine, 2006).

Đầu ra là danh sách ĐÃ XẾP HẠNG, không phải danh sách "đúng" — bước tiếp theo
bắt buộc là kiểm tra thủ công (spot-check) một mẫu top-N trước khi dùng chính
thức, vì không có ground-truth độc lập cho thị trường Bangladesh (rủi ro
circularity, xem docs/00-de-cuong-nghien-cuu.md mục 4 và 7).
"""

import itertools
from dataclasses import dataclass

import jellyfish
from rapidfuzz.distance import Levenshtein


def normalized_edit_similarity(a: str, b: str) -> float:
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def phonetic_similarity(a: str, b: str) -> float:
    ma, mb = jellyfish.metaphone(a), jellyfish.metaphone(b)
    if not ma or not mb:
        return 0.0
    return jellyfish.jaro_winkler_similarity(ma, mb)


def bi_sim_score(a: str, b: str, w_ortho: float = 0.5, w_phon: float = 0.5) -> float:
    a_l, b_l = a.lower(), b.lower()
    ortho = normalized_edit_similarity(a_l, b_l)
    phon = phonetic_similarity(a_l, b_l)
    return w_ortho * ortho + w_phon * phon


@dataclass
class ConfusablePair:
    name_a: str
    name_b: str
    score: float
    ortho: float
    phon: float


def looks_like_typo_of_same_word(a: str, b: str) -> bool:
    """True nếu a/b nhiều khả năng là MỘT tên thuốc bị ghi nhãn khác nhau
    (lỗi đánh máy/định dạng), không phải hai tên thuốc THẬT khác nhau.

    Heuristic: khoảng cách edit-distance <= 1 giữa hai chuỗi đủ dài. Đánh đổi
    có chủ đích: có thể bỏ sót vài cặp LASA thật sự chỉ khác 1 ký tự, để đổi
    lấy việc loại phần lớn nhiễu do lỗi ghi nhãn trong RxHandBD (xem
    scripts/derive_confusable_pairs.py và kết quả chạy thử: top-ranked pairs
    trước khi lọc gần như toàn là biến thể chính tả của cùng 1 thuốc).
    """
    a_l, b_l = a.lower(), b.lower()
    return Levenshtein.distance(a_l, b_l) <= 1


def rank_confusable_pairs(
    vocab: list[str],
    min_len: int = 3,
    top_k: int | None = None,
    exclude_likely_typos: bool = True,
) -> list[ConfusablePair]:
    names = sorted({n for n in vocab if len(n) >= min_len}, key=str.lower)
    pairs: list[ConfusablePair] = []
    for a, b in itertools.combinations(names, 2):
        if a.lower() == b.lower():
            continue
        if exclude_likely_typos and looks_like_typo_of_same_word(a, b):
            continue
        ortho = normalized_edit_similarity(a.lower(), b.lower())
        phon = phonetic_similarity(a.lower(), b.lower())
        score = 0.5 * ortho + 0.5 * phon
        pairs.append(ConfusablePair(a, b, score, ortho, phon))
    pairs.sort(key=lambda p: -p.score)
    return pairs[:top_k] if top_k else pairs
