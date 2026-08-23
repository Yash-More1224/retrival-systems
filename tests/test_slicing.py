"""Regression test for Q4.3's head/tail slicing bug: on both real datasets, over
90% of articles have ZERO train-split clicks, so the naive 90th-percentile
threshold is exactly 0 -- a `>=` comparison then classifies every article
(including all the zero-click ones) as 'head', silently collapsing the tail
slice to n=0. See src/eval/slicing.py's compute_head_threshold docstring.
"""
from __future__ import annotations

import numpy as np

from src.eval.slicing import compute_head_threshold, head_or_tail


def test_head_tail_not_degenerate_when_90pct_of_catalog_has_zero_clicks():
    # Mirrors the real MIND/EB-NeRD distribution (~90% zero-click articles). Needs
    # enough zeros that np.quantile's linear interpolation lands exactly on 0 (not
    # just close to it) -- see compute_head_threshold's docstring for why that
    # exact-0 case is what makes '>=' degenerate.
    counts = np.array([0] * 920 + list(range(1, 81)))  # 1000 articles, 80 with clicks
    threshold = compute_head_threshold(counts, top_pct=0.10)
    assert threshold == 0.0  # confirms the degenerate case actually triggers here

    train_click_count = {f"a{i}": c for i, c in enumerate(counts)}
    labels = [head_or_tail(aid, train_click_count, threshold) for aid in train_click_count]

    n_head = labels.count("head")
    n_tail = labels.count("tail")
    assert n_tail > 0, "every article was classified as head -- the >= degeneracy regressed"
    assert n_head == 80  # exactly the nonzero-click articles
    assert n_head + n_tail == 1000


def test_head_or_tail_none_article_id_returns_none():
    assert head_or_tail(None, {}, 5.0) is None
