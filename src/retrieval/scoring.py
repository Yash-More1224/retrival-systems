"""Shared impression-candidate scoring, used by BOTH the Q4 eval harness
(src/eval/run_eval.py) and the Q5 Codabench submission generators
(src/submission/{mind,ebnerd}.py) -- factored out so both consumers score
impressions identically rather than risking two subtly different
implementations of the same "rank this impression's candidates" logic.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.retrieval.bm25 import BM25Index, build_query_text
from src.retrieval.semantic import SemanticIndex, build_user_embedding


def best_n(results_dir, dataset: str, method: str, default_n: int) -> int:
    """N selected by Q2/Q3's val-sweep (results/<dataset>_<method>_val_sweep.json),
    falling back to a config default if that file doesn't exist yet."""
    sweep_path = results_dir / f"{dataset}_{method}_val_sweep.json"
    if sweep_path.exists():
        return int(json.loads(sweep_path.read_text())["best_n"])
    return default_n


def _bm25_score_matrix(index: BM25Index, batch: pd.DataFrame, article_title: dict, n: int,
                        popularity_by_id: dict) -> np.ndarray:
    query_texts = [build_query_text(list(h), article_title, n) for h in batch["history_ids"]]
    is_empty = np.array([not qt.strip() for qt in query_texts])
    S = index.score_batch(query_texts)
    if is_empty.any():
        pop_row = np.array([popularity_by_id.get(a, 0.0) for a in index.article_ids], dtype=np.float32)
        S[is_empty] = pop_row
    return S


def score_bm25_batch(index: BM25Index, batch: pd.DataFrame, article_title: dict, n: int,
                      popularity_by_id: dict) -> list[dict[str, float]]:
    """Returns one {article_id: score} dict per row in batch, over index's FULL catalog.
    Cold-start (empty query) rows fall back to train-split popularity (SPEC.md Q2.3)."""
    S = _bm25_score_matrix(index, batch, article_title, n, popularity_by_id)
    return [dict(zip(index.article_ids, row)) for row in S]


def _covered_candidate_ids(batch: pd.DataFrame, id_to_col: dict) -> list:
    seen = set()
    covered = []
    for candidates in batch["candidates"]:
        for cid in candidates:
            if cid not in seen and cid in id_to_col:
                seen.add(cid)
                covered.append(cid)
    return covered


def score_bm25_batch_candidates(index: BM25Index, batch: pd.DataFrame, article_title: dict, n: int,
                                 popularity_by_id: dict, id_to_col: dict[str, int]) -> list[dict[str, float | None]]:
    """Same scoring as score_bm25_batch, but restricted throughout to each BATCH's own
    union of candidate article IDs instead of the full catalog: both the score matrix
    (query x candidate-columns of W, not query x every doc) and the returned dict (each
    row's own ~10-30 candidates, not every catalog article). score_bm25_batch's dict(zip(
    article_ids, row)) over a full-catalog score matrix is fine for MIND's ~30K articles /
    73K impressions, but at ebnerd_testset's 125K articles / 13.5M impressions the dict
    construction alone was estimated at 70+ hours, and even after fixing that, the FULL
    dense score matrix itself (score_bm25_batch's underlying computation) was still an
    estimated ~10 hours: real batches only ever touch ~250-300 unique candidates out of
    125,541 (found 2026-08-23 -- see src/submission/ebnerd.py). Since matrix columns are
    independent, restricting to a subset of columns before the matmul is numerically
    IDENTICAL to computing the full matrix and slicing after -- this is a pure speedup,
    not an approximation. A candidate missing from id_to_col maps to None."""
    query_texts = [build_query_text(list(h), article_title, n) for h in batch["history_ids"]]
    is_empty = np.array([not qt.strip() for qt in query_texts])
    Qmat = index.vectorizer.transform(query_texts).astype(np.float32)

    covered_ids = _covered_candidate_ids(batch, id_to_col)
    cols = [id_to_col[cid] for cid in covered_ids]
    W_sub = index.W[cols, :]  # (n_covered, n_terms) -- CSR row-slice, cheap
    S = np.asarray((Qmat @ W_sub.T).todense())  # (batch_size, n_covered)
    if is_empty.any():
        pop_row = np.array([popularity_by_id.get(cid, 0.0) for cid in covered_ids], dtype=np.float32)
        S[is_empty] = pop_row

    local_col = {cid: j for j, cid in enumerate(covered_ids)}
    return [
        {cid: (float(row[local_col[cid]]) if cid in local_col else None) for cid in candidates}
        for row, candidates in zip(S, batch["candidates"])
    ]


def _semantic_score_matrix(index: SemanticIndex, batch: pd.DataFrame, embedding_lookup: dict,
                            n: int, dim: int, popularity_by_id: dict) -> np.ndarray:
    query_vecs = np.zeros((len(batch), dim), dtype=np.float32)
    is_empty = np.zeros(len(batch), dtype=bool)
    for i, h in enumerate(batch["history_ids"]):
        vec, cold = build_user_embedding(list(h), embedding_lookup, n, dim)
        query_vecs[i] = vec
        is_empty[i] = cold
    S = index.score_dense(query_vecs)
    if is_empty.any():
        pop_row = np.array([popularity_by_id.get(a, 0.0) for a in index.article_ids], dtype=np.float32)
        S[is_empty] = pop_row
    return S


def score_semantic_batch(index: SemanticIndex, batch: pd.DataFrame, embedding_lookup: dict,
                          n: int, dim: int, popularity_by_id: dict) -> list[dict[str, float]]:
    """Same contract as score_bm25_batch, but only over the EMBEDDING-COVERED subset of
    the catalog (see build_embeddings.py) -- callers must handle candidates missing from
    the returned dict explicitly, not assume every candidate has a score."""
    S = _semantic_score_matrix(index, batch, embedding_lookup, n, dim, popularity_by_id)
    return [dict(zip(index.article_ids, row)) for row in S]


def score_semantic_batch_candidates(index: SemanticIndex, batch: pd.DataFrame, embedding_lookup: dict,
                                     n: int, dim: int, popularity_by_id: dict,
                                     id_to_col: dict[str, int]) -> list[dict[str, float | None]]:
    """Candidates-only counterpart to score_semantic_batch, restricted to the batch's own
    candidate columns for the matmul itself (not just the output dict) -- see
    score_bm25_batch_candidates' docstring for why this exists and why it's numerically
    exact, not approximate."""
    query_vecs = np.zeros((len(batch), dim), dtype=np.float32)
    is_empty = np.zeros(len(batch), dtype=bool)
    for i, h in enumerate(batch["history_ids"]):
        vec, cold = build_user_embedding(list(h), embedding_lookup, n, dim)
        query_vecs[i] = vec
        is_empty[i] = cold

    covered_ids = _covered_candidate_ids(batch, id_to_col)
    cols = [id_to_col[cid] for cid in covered_ids]
    sub_embeddings = index.embeddings[cols]  # (n_covered, dim)
    S = query_vecs @ sub_embeddings.T  # (batch_size, n_covered)
    if is_empty.any():
        pop_row = np.array([popularity_by_id.get(cid, 0.0) for cid in covered_ids], dtype=np.float32)
        S[is_empty] = pop_row

    local_col = {cid: j for j, cid in enumerate(covered_ids)}
    return [
        {cid: (float(row[local_col[cid]]) if cid in local_col else None) for cid in candidates}
        for row, candidates in zip(S, batch["candidates"])
    ]
