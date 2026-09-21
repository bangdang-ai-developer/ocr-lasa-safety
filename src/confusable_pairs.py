"""Infer look-alike/sound-alike (LASA) drug name pairs using a generic algorithm
(no ISMP/RxNorm — doesn't match the Bangladesh market).

Method: combines orthographic similarity (normalized edit distance) with
phonetic similarity (metaphone code + Jaro-Winkler), in the spirit of the
BI-SIM approach from Kondrak & Dorr, "Automatic identification of confusable
drug names" (Artificial Intelligence in Medicine, 2006).

The output is a RANKED list, not a list of "correct" pairs — a mandatory next
step is manual spot-checking of a top-N sample before official use, since
there is no independent ground truth for the Bangladesh market (circularity
risk; see docs/00-de-cuong-nghien-cuu.md, sections 4 and 7).
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
    """True if a/b are most likely ONE drug name recorded under two different
    labels (a typo/formatting inconsistency), rather than two genuinely
    DIFFERENT drug names.

    Heuristic: edit distance <= 1 between two sufficiently long strings. This
    is a deliberate trade-off: it may miss a few genuine LASA pairs that
    differ by just one character, in exchange for filtering out most of the
    labeling noise in RxHandBD (see scripts/derive_confusable_pairs.py; a
    test run showed that, before this filter, the top-ranked pairs were
    almost entirely spelling variants of the same drug).
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
