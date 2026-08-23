"""Q4.2 -- Beyond-accuracy metrics: intra-list diversity, novelty, coverage.

All operate on the top-`cutoff` of the RANKED candidate list per impression
(cutoff = config.eval.beyond_accuracy_cutoff, default 10) -- these metrics
are meaningless without a stated cutoff (SPEC.md Q4.2), so every caller must
pass one explicitly rather than relying on a hidden default.

Two ILD variants are reported (SPEC.md Q4.2), because they answer different
questions and can disagree:
  - categorical: fraction of pairs with a different `category` -- coarse,
    but works even for articles without an embedding.
  - embedding: mean pairwise cosine DISTANCE (1 - cos) -- the standard ILD,
    and the one that can reveal semantic retrieval's known redundancy
    problem (near-duplicate embeddings cluster together). Pairs where either
    article lacks an embedding are skipped, not treated as maximally
    diverse or maximally similar.

Novelty and popularity MUST be computed from TRAIN-split click counts only
(passed in, never recomputed here) -- using val/test counts would leak
future popularity into an offline metric (SPEC.md Q4.2).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def ild_categorical(categories: list[str]) -> float | None:
    n = len(categories)
    if n < 2:
        return None
    diffs = sum(1 for a, b in combinations(categories, 2) if a != b)
    return diffs / (n * (n - 1) / 2)


def ild_embedding(article_ids: list[str], embedding_lookup: dict[str, np.ndarray]) -> float | None:
    vecs = [embedding_lookup[a] for a in article_ids if a in embedding_lookup]
    if len(vecs) < 2:
        return None
    distances = []
    for a, b in combinations(vecs, 2):
        cos = float(np.dot(a, b))  # embeddings are pre-normalised (see build_embeddings.py)
        distances.append(1 - cos)
    return float(np.mean(distances))


def novelty(article_ids: list[str], train_click_count: dict[str, int], total_train_clicks: int) -> float | None:
    if not article_ids or total_train_clicks <= 0:
        return None
    # +1 smoothing: an article with zero train-split clicks would otherwise be
    # log2(0) = -inf "infinitely novel", which is a smoothing artifact, not a
    # meaningful signal -- see SPEC.md Q4.2.
    scores = [-np.log2((train_click_count.get(a, 0) + 1) / (total_train_clicks + 1)) for a in article_ids]
    return float(np.mean(scores))


def coverage(all_topk_lists: list[list[str]], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    recommended = set(a for lst in all_topk_lists for a in lst)
    return len(recommended) / catalog_size
