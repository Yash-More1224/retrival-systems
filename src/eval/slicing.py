"""Q4.3 -- Slicing: cold-start vs. warm users (required), head vs. tail articles.

Both slices are computed from TRAIN-split-only statistics where relevant
(head/tail threshold from train_click_count), never from val/test, for the
same leakage reason as beyond_accuracy.py's novelty (SPEC.md Q4.2/Q9).
"""
from __future__ import annotations

import numpy as np


def cold_start_or_warm(history_len: int, threshold: int) -> str:
    return "cold_start" if history_len < threshold else "warm"


def compute_head_threshold(train_click_counts: np.ndarray, top_pct: float) -> float:
    """Click-count value at the (1 - top_pct) percentile -- e.g. top_pct=0.10
    returns the 90th-percentile click count, so articles ABOVE it are 'head'.

    IMPORTANT: both MIND and EB-NeRD are heavily click-skewed -- over 90% of
    articles get zero train-split clicks (verified 2026-08-20: 90.19% MIND,
    90.54% EB-NeRD). That means the 90th-percentile click count is exactly 0,
    so a naive `count >= threshold` comparison classifies EVERY article as
    'head' (0 >= 0), silently collapsing the tail slice to n=0. head_or_tail
    below uses a strict `>` for exactly this reason -- with `>`, 'head' comes
    out to ~9.5-9.8% of the catalog, i.e. actually the top decile, not 100%.
    """
    if len(train_click_counts) == 0:
        return 0.0
    return float(np.quantile(train_click_counts, 1 - top_pct))


def head_or_tail(article_id: str | None, train_click_count: dict[str, int], threshold: float) -> str | None:
    """Strict '>', not '>=' -- see compute_head_threshold's docstring for why '>='
    degenerates to classifying 100% of articles as 'head' on this data."""
    if article_id is None:
        return None
    return "head" if train_click_count.get(article_id, 0) > threshold else "tail"


def first_clicked_article(candidates: list[str], labels: list[int]) -> str | None:
    """SPEC.md Q4.3: 'slice by whether the CLICKED article is head or tail' -- an
    impression can have >1 click (EB-NeRD), so this picks the first by candidate-list
    order, documented and consistent rather than arbitrary."""
    for a, l in zip(candidates, labels):
        if l == 1:
            return a
    return None
