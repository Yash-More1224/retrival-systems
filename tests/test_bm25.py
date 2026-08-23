"""Regression test for the cross-machine reproducibility bug in top_k_indices
(found 2026-08-21 by comparing local vs. ada recall@K numbers for identical
seed/code/data -- see docs/ai_usage_log.md and bm25.py's top_k_indices docstring).
"""
from __future__ import annotations

import numpy as np

from src.retrieval.bm25 import top_k_indices


def test_top_k_ties_broken_by_ascending_index_deterministically():
    # Mirrors the real failure mode: one query, mostly-zero scores (sparse
    # single-title query), only a couple of genuinely distinguishing scores.
    scores = np.array([[0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0]])
    result = top_k_indices(scores, k=4)[0]
    # col 1 (5.0) then col 4 (3.0) are unambiguous; the remaining two slots are
    # a 4-way tie at 0.0 among cols {0, 2, 3, 5, 6} -- must deterministically
    # take the lowest-index ones (0, 2), not an arbitrary subset.
    assert list(result) == [1, 4, 0, 2]


def test_top_k_correctness_matches_naive_full_sort():
    rng = np.random.default_rng(0)
    scores = rng.random((5, 50))
    k = 10
    result = top_k_indices(scores, k)
    for i in range(scores.shape[0]):
        expected = sorted(range(50), key=lambda j: (-scores[i, j], j))[:k]
        assert list(result[i]) == expected


def test_top_k_repeated_calls_are_bit_identical():
    """The original bug was non-reproducible ONLY across different numpy/BLAS
    builds, so a same-process repeat call wouldn't have caught it -- this test
    exists mainly to document the invariant, not to reproduce the original bug
    (which required a different machine)."""
    scores = np.array([[0.0] * 100])
    r1 = top_k_indices(scores, 50)
    r2 = top_k_indices(scores, 50)
    assert np.array_equal(r1, r2)
    assert list(r1[0]) == list(range(50))  # all-tied -> ascending index order


def test_top_k_k_larger_than_catalog_is_clamped():
    scores = np.array([[1.0, 2.0, 3.0]])
    result = top_k_indices(scores, k=10)
    assert result.shape == (1, 3)
