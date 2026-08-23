"""Q2.0 -- Retrieval pools and recall@K, shared by BM25 (Q2) and semantic (Q3).

Candidate generation and in-impression ranking are different tasks (see
SPEC.md Q2.0): here we retrieve top-K from the WHOLE catalog against
ground-truth clicks, under two clearly-labelled pools, rather than the
~20-item impression list Q4/Q5 rank. Reporting recall@200 over the full
catalog gives a small, honest number -- that's expected, not a bug.

Pool A -- full catalog: every article, minus those published after the
  impression (EB-NeRD only; MIND has no published_time), minus articles
  already in the user's own history.
Pool B -- active pool: articles that appeared in ANY impression's
  `article_ids_inview` during the target split -- a legitimate restriction
  (an article nobody could be shown cannot be clicked).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def active_pool_b(splits_dir: Path, dataset: str, split: str) -> set[str]:
    """Union of every candidate article_id shown in any impression of this split."""
    imp = pd.read_parquet(splits_dir / dataset / split / "impressions.parquet", columns=["candidates"])
    active: set[str] = set()
    for cands in imp["candidates"]:
        active.update(cands)
    return active


def eligibility_mask(article_ids: list[str], published_time: np.ndarray, impression_times) -> np.ndarray:
    """(n_queries, n_docs) bool mask: True where article is eligible (published_time is
    null, or <= the impression's timestamp). MIND has no published_time -> all True.
    `impression_times` is any array-like of per-query pd.Timestamp/np.datetime64.
    """
    imp_times = pd.to_datetime(pd.Series(list(impression_times))).to_numpy()
    pub = pd.Series(published_time)
    if pub.isna().all():
        return np.ones((len(imp_times), len(article_ids)), dtype=bool)
    pub_arr = pub.to_numpy()
    # broadcast: (n_queries, 1) vs (1, n_docs)
    null_mask = pd.isna(pub_arr)[None, :]
    not_future = pub_arr[None, :] <= imp_times[:, None]
    return null_mask | not_future


def exclude_history_inplace(mask: np.ndarray, article_id_to_idx: dict[str, int],
                             history_ids_per_row: list[list[str]]) -> None:
    """Set mask[row, col] = False for every article already in that row's click history.
    Mutates mask in place. Vectorized via a single fancy-index assignment rather than a
    per-row python loop over columns.
    """
    rows, cols = [], []
    for i, hist in enumerate(history_ids_per_row):
        for a in hist:
            j = article_id_to_idx.get(a)
            if j is not None:
                rows.append(i)
                cols.append(j)
    if rows:
        mask[np.asarray(rows), np.asarray(cols)] = False


def recall_at_k_batch(topk_ids: list[list[str]], ground_truth: list[set[str]]) -> list[float]:
    """Per-impression recall@K = |clicked ∩ topK| / |clicked|. len(topk_ids) must equal
    len(ground_truth); K is implicit in len(topk_ids[i])."""
    out = []
    for cands, truth in zip(topk_ids, ground_truth):
        if not truth:
            out.append(None)
            continue
        hit = len(set(cands) & truth)
        out.append(hit / len(truth))
    return out
