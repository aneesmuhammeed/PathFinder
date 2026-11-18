"""Standard ranking metrics over *graded* relevance (grade 0 = irrelevant; higher = better).

`grades` is the list of grades of the returned results, in rank order.
`all_grades` is the grade of EVERY document in the corpus (needed for the ideal ranking and exact recall).
Gain is linear (gain = grade) with a log2 position discount.
"""

import math
from collections.abc import Sequence


def dcg(grades: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(grades: Sequence[float], all_grades: Sequence[float], k: int = 10) -> float:
    ideal = dcg(sorted(all_grades, reverse=True), k)
    return dcg(grades, k) / ideal if ideal > 0 else 0.0


def precision_at_k(grades: Sequence[float], k: int = 10, threshold: float = 1) -> float:
    """Of the top k slots, the fraction holding a relevant result. Empty slots count as misses."""
    return sum(g >= threshold for g in grades[:k]) / k


def reciprocal_rank(grades: Sequence[float], threshold: float = 1) -> float:
    for i, g in enumerate(grades, start=1):
        if g >= threshold:
            return 1.0 / i
    return 0.0


def recall_at_k(grades: Sequence[float], all_grades: Sequence[float], k: int = 50, threshold: float = 1) -> float:
    total = sum(g >= threshold for g in all_grades)
    return sum(g >= threshold for g in grades[:k]) / total if total else 0.0


def summarize(grades: Sequence[float], all_grades: Sequence[float]) -> dict[str, float]:
    return {
        "ndcg@10": ndcg_at_k(grades, all_grades, 10),
        "p@10": precision_at_k(grades, 10),
        "mrr": reciprocal_rank(grades),
        "recall@50": recall_at_k(grades, all_grades, 50),
    }


def mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
