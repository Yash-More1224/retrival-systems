"""Hand-checked correctness tests for Q4's metrics -- the difference between
"plausible" and "correct" per SPEC.md Q4.1. Also regression-guards the two
bugs the v1 draft had (truncated-ideal-DCG, unhandled single-class AUC).
"""
from __future__ import annotations

import math

import pytest

from src.eval.beyond_accuracy import coverage, ild_categorical, novelty
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import dcg, impression_auc, mrr, ndcg_at_k, rank_by_score


def test_mrr_basic():
    assert mrr([0, 1, 0]) == pytest.approx(0.5)
    assert mrr([1, 0, 0]) == pytest.approx(1.0)
    assert mrr([0, 0, 0]) == pytest.approx(0.0)


def test_ndcg_matches_hand_computed_example():
    # SPEC.md Q4.1's worked example: labels [0,1,0] (already score-ordered) -> nDCG@5 ~ 0.6309
    assert ndcg_at_k([0, 1, 0], 5) == pytest.approx(1 / math.log2(3), abs=1e-4)


def test_ndcg_ideal_uses_full_label_list_not_truncated_k():
    """Regression test for the v1 bug: computing IDCG from labels[:k] instead of the
    full label list first (then truncated) silently inflates nDCG when a relevant
    item exists beyond position k."""
    labels = [1, 0, 0, 1]  # one relevant item ranked 1st, another ranked 4th
    k = 2
    result = ndcg_at_k(labels, k)

    dcg_at_2 = 1.0 / math.log2(2) + 0.0 / math.log2(3)  # = 1.0
    correct_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # ideal ranking is [1,1,0,0], truncated to 2
    buggy_idcg = 1.0 / math.log2(2)  # what you'd get sorting labels[:2]=[1,0] instead

    assert result == pytest.approx(dcg_at_2 / correct_idcg)
    assert result != pytest.approx(dcg_at_2 / buggy_idcg)


def test_ndcg_zero_relevant_is_zero():
    assert ndcg_at_k([0, 0, 0], 5) == 0.0


def test_dcg_empty_is_zero():
    assert dcg([]) == 0.0


def test_impression_auc_single_class_returns_none():
    """Regression test: sklearn's roc_auc_score raises on single-class input;
    callers must be able to detect and skip this, not crash."""
    assert impression_auc([0.9, 0.5, 0.1], [0, 0, 0]) is None
    assert impression_auc([0.9, 0.5, 0.1], [1, 1, 1]) is None


def test_impression_auc_perfect_ranking():
    assert impression_auc([0.9, 0.5, 0.1], [1, 0, 0]) == pytest.approx(1.0)


def test_impression_auc_worst_ranking():
    assert impression_auc([0.1, 0.5, 0.9], [1, 0, 0]) == pytest.approx(0.0)


def test_rank_by_score_orders_descending_with_deterministic_tiebreak():
    ids, scores, labels = rank_by_score(["b", "a", "c"], [1.0, 2.0, 1.0], [0, 1, 1])
    # "a" (score 2.0) first; "b" and "c" tie at 1.0, broken by candidate_id ascending
    assert ids == ["a", "b", "c"]
    assert labels == [1, 0, 1]


def test_ild_categorical_all_same_is_zero():
    assert ild_categorical(["news", "news", "news"]) == pytest.approx(0.0)


def test_ild_categorical_all_different_is_one():
    assert ild_categorical(["a", "b", "c"]) == pytest.approx(1.0)


def test_ild_categorical_single_item_is_none():
    assert ild_categorical(["a"]) is None


def test_novelty_smoothing_avoids_log_zero():
    # article with 0 train clicks must not produce -inf
    result = novelty(["unseen_article"], {}, total_train_clicks=1000)
    assert result is not None
    assert math.isfinite(result)


def test_novelty_popular_article_is_less_novel():
    popular = novelty(["a"], {"a": 900}, total_train_clicks=1000)
    rare = novelty(["b"], {"b": 1}, total_train_clicks=1000)
    assert rare > popular


def test_coverage_basic():
    lists = [["a", "b"], ["b", "c"]]
    assert coverage(lists, catalog_size=10) == pytest.approx(3 / 10)


def test_coverage_empty_catalog_is_zero_not_crash():
    assert coverage([["a"]], catalog_size=0) == 0.0


def test_bootstrap_ci_is_seeded_and_reproducible():
    values = [0.1, 0.5, 0.9, 0.3, 0.7] * 10
    r1 = bootstrap_ci(values, n_boot=200, seed=42)
    r2 = bootstrap_ci(values, n_boot=200, seed=42)
    assert r1 == r2


def test_bootstrap_ci_filters_none_and_handles_all_none():
    mean, lo, hi = bootstrap_ci([0.5, None, 0.7, None], n_boot=100)
    assert mean == pytest.approx(0.6)
    mean_nan, _, _ = bootstrap_ci([None, None], n_boot=100)
    assert math.isnan(mean_nan)
