"""Micro-integration test for the Q5 submission pipeline's core logic (BM25Index ->
score_bm25_batch -> scores_to_ranks -> validate_submission), using a tiny synthetic
catalog instead of the real feature store. Doesn't exercise src.submission.mind/ebnerd's
file I/O directly (those need the real data pipeline's directory layout -- see
src/config.py's REPO_ROOT-relative resolve_path), but does exercise every wiring point
that would otherwise only be caught by burning a real, rate-limited Codabench upload.
"""
from __future__ import annotations

import pandas as pd

from src.retrieval.bm25 import BM25Index
from src.retrieval.scoring import score_bm25_batch
from src.submission.format import scores_to_ranks, validate_submission


def test_bm25_scoring_to_submission_rows_end_to_end():
    article_ids = ["a1", "a2", "a3", "a4"]
    texts = [
        "breaking news election results",
        "sports football match highlights",
        "election results analysis",
        "weather forecast rain tomorrow",
    ]
    index = BM25Index.build(article_ids, texts, lang="english")

    imp = pd.DataFrame({
        "impression_id": ["100", "101"],
        "candidates": [["a1", "a2", "a3"], ["a2", "a4"]],
        "history_ids": [["a1"], []],  # second impression is cold-start (empty history)
    })
    article_title = dict(zip(article_ids, texts))
    train_click_count = {"a1": 5, "a2": 1, "a3": 0, "a4": 10}

    score_maps = score_bm25_batch(index, imp, article_title, n=5, popularity_by_id=train_click_count)
    assert len(score_maps) == 2

    rows = []
    for impression_id, candidates, score_map in zip(imp["impression_id"], imp["candidates"], score_maps):
        scores = [score_map[a] for a in candidates]  # every candidate is in-catalog here
        ranks = scores_to_ranks(scores)
        rows.append((impression_id, ranks))

    # Structural validity: every row is a permutation of 1..len(candidates)
    validate_submission(rows, expected_ids_in_order=["100", "101"])

    # Impression 101 is cold-start -> falls back to train_click_count popularity,
    # so a4 (count=10) must outrank a2 (count=1).
    _, ranks_101 = rows[1]
    a2_rank, a4_rank = ranks_101[0], ranks_101[1]  # candidates order was [a2, a4]
    assert a4_rank < a2_rank, "cold-start impression should rank by popularity, a4 should beat a2"

    # Impression 100 has real history ("a1") -> BM25 should favor a3 (shares "election
    # results" with a1's text) over a2 (shares nothing).
    _, ranks_100 = rows[0]
    a1_rank, a2_rank, a3_rank = ranks_100
    assert a3_rank < a2_rank, "a3 shares terms with the query and should outrank unrelated a2"


def test_uncovered_candidate_handled_not_crashed():
    """A candidate article missing from the scoring catalog (e.g. semantic method,
    embedding-uncovered article) must be handled explicitly, not crash a lookup."""
    article_ids = ["a1", "a2"]
    texts = ["some text here", "other text there"]
    index = BM25Index.build(article_ids, texts, lang="english")

    imp = pd.DataFrame({
        "impression_id": ["1"],
        "candidates": [["a1", "unknown_article"]],
        "history_ids": [["a1"]],
    })
    score_maps = score_bm25_batch(index, imp, dict(zip(article_ids, texts)), n=5, popularity_by_id={})

    scores = []
    for a in imp["candidates"].iloc[0]:
        s = score_maps[0].get(a)
        scores.append(float(s) if s is not None else -1e9)
    ranks = scores_to_ranks(scores)
    assert sorted(ranks) == [1, 2]
    assert ranks[1] == 2, "the uncovered candidate must be ranked last, not crash or rank first"
