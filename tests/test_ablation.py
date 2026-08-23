"""Q9.1 -- Serving-feature ablation tests.

Unit-level tests exercise src.eval.run_ablation's scoring logic directly with
synthetic data (fast, no pipeline dependency). The integration-style test
checks the actual ablation run's results/ebnerd_ablation.json (if present)
shows the expected pattern: each config should score at least as well as the
last, and the deliberate post-interaction leakage should score (near-)perfectly,
demonstrating "looks excellent, is worthless" is a measured fact, not a claim.
"""
from __future__ import annotations

import json

import pytest

from src.config import load_config, resolve_path
from src.eval.run_ablation import LEAK_BONUS, _augmented_scores, _popularity_feature


def test_clean_config_ignores_features():
    """clean must depend only on the base score -- identical popularity/labels
    inputs must not change its output (it's the deployable-system baseline)."""
    base = [0.1, 0.9, 0.3]
    candidates = ["a1", "a2", "a3"]
    stats_a = {"a1": {"total_pageviews": 1000}, "a2": {}, "a3": {"total_pageviews": 0}}
    stats_b = {"a1": {"total_pageviews": 0}, "a2": {"total_pageviews": 999999}, "a3": {}}
    out_a = _augmented_scores("clean", base, candidates, [0, 1, 0], stats_a)
    out_b = _augmented_scores("clean", base, candidates, [0, 1, 0], stats_b)
    assert out_a == out_b


def test_popularity_feature_handles_missing_and_nan_stats():
    """A candidate with no article_stats entry, or explicit NaN fields, must
    contribute 0 (not crash / not propagate NaN) -- regression test for the
    `x or 0` vs. NaN bug found while building this (NaN is truthy in Python,
    so `stat or 0` silently returns NaN instead of falling back)."""
    import math
    candidates = ["known", "missing", "nan_stats"]
    stats = {
        "known": {"total_inviews": 100, "total_pageviews": 50, "total_read_time": 10, "sentiment_score": 0.5},
        "nan_stats": {"total_inviews": float("nan"), "total_pageviews": None, "total_read_time": float("nan"),
                      "sentiment_score": float("nan")},
    }
    vals = _popularity_feature(candidates, stats)
    assert len(vals) == 3
    assert all(not math.isnan(v) for v in vals), f"NaN leaked into popularity feature: {vals}"
    assert vals[1] == 0.0  # "missing" has no entry at all
    assert vals[2] == 0.0  # "nan_stats" has entries, but all NaN/None


def test_post_interaction_config_guarantees_true_click_ranks_first():
    """plus_post_interaction's whole point is a deliberate, extreme leak: the
    true-click candidate must always score strictly higher than every other
    candidate, regardless of how bad the base retrieval score was."""
    base = [0.9, 0.1, 0.5]  # candidate 0 (index 0) has the BEST base score...
    candidates = ["a1", "a2", "a3"]
    labels = [0, 1, 0]  # ...but candidate 1 (index 1) is the true click
    out = _augmented_scores("plus_post_interaction", base, candidates, labels, {})
    true_click_idx = labels.index(1)
    assert out[true_click_idx] == max(out)
    assert out[true_click_idx] - max(s for i, s in enumerate(out) if i != true_click_idx) >= LEAK_BONUS - 1.0


def test_configs_are_cumulative_not_independent():
    """plus_post_interaction is base + popularity + leak (cumulative), not a
    fresh base + leak -- SPEC.md Q9.1 frames the three rows as additive steps."""
    base = [0.5, 0.5]
    candidates = ["a1", "a2"]
    labels = [1, 0]
    stats = {"a1": {"total_pageviews": 100}, "a2": {"total_pageviews": 0}}
    pop_only = _augmented_scores("plus_popularity", base, candidates, labels, stats)
    leak = _augmented_scores("plus_post_interaction", base, candidates, labels, stats)
    # the non-true-click candidate's score should match between the two configs
    # (leak only adds to the true click), confirming plus_popularity's contribution carries through
    assert leak[1] == pytest.approx(pop_only[1])


def test_ablation_results_show_expected_leakage_pattern():
    """Integration check on the real run's output, if present: each config
    should not score WORSE than the previous one (more signal, even leaky
    signal, should not hurt AUC/MRR here), and the post-interaction config
    must be (near-)perfect -- the measured demonstration of Q9.1's point."""
    cfg = load_config()
    path = resolve_path(cfg, "results_dir") / "ebnerd_ablation.json"
    if not path.exists():
        pytest.skip(f"{path} not found -- run `python -m src.eval.run_ablation` first")

    results = json.loads(path.read_text())
    for method in ["bm25", "semantic"]:
        clean_auc = results[method]["clean"]["auc"]["mean"]
        pop_auc = results[method]["plus_popularity"]["auc"]["mean"]
        leak_auc = results[method]["plus_post_interaction"]["auc"]["mean"]
        assert clean_auc <= pop_auc + 1e-9, f"{method}: +popularity scored worse than clean"
        assert pop_auc <= leak_auc + 1e-9, f"{method}: +post_interaction scored worse than +popularity"
        assert leak_auc > 0.99, f"{method}: post-interaction leakage AUC={leak_auc:.4f}, expected near-perfect"
