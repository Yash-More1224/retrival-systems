"""Q4.1 -- Ranking accuracy metrics: AUC, MRR, nDCG@5, nDCG@10.

Bugs fixed relative to the v1 draft (see SPEC.md Q4.1):
  - ndcg_at_k's ideal ranking is computed from the FULL label list, not
    labels[:k] -- truncating first can make IDCG smaller than it should be,
    inflating nDCG.
  - impression_auc returns None (not a crash) for single-class impressions --
    sklearn's roc_auc_score raises ValueError when all labels are 0 or all 1,
    which is common (most MIND impressions have exactly one positive).
    Callers must count and report how many impressions were skipped rather
    than silently averaging over fewer values than n_impressions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg(labels: list[int]) -> float:
    """labels must already be ordered by predicted score, descending."""
    return sum(l / np.log2(i + 2) for i, l in enumerate(labels))


def ndcg_at_k(labels: list[int], k: int) -> float:
    ideal = sorted(labels, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(labels[:k]) / idcg if idcg > 0 else 0.0


def mrr(labels: list[int]) -> float:
    for i, l in enumerate(labels, 1):
        if l == 1:
            return 1.0 / i
    return 0.0


def impression_auc(scores: list[float], labels: list[int]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def rank_by_score(candidate_ids: list[str], scores: list[float], labels: list[int]
                   ) -> tuple[list[str], list[float], list[int]]:
    """Sort candidates by score descending, breaking ties deterministically by
    candidate_id so results are exactly reproducible (SPEC.md Q4.0)."""
    order = sorted(range(len(candidate_ids)), key=lambda i: (-scores[i], candidate_ids[i]))
    ids = [candidate_ids[i] for i in order]
    sc = [scores[i] for i in order]
    lb = [labels[i] for i in order]
    return ids, sc, lb
