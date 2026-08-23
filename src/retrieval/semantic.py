"""Q3.2/3.3 -- ANN index (FAISS IndexFlatIP) + user representation for semantic retrieval.

Embeddings are PROVIDED (see build_embeddings.py), L2-normalised, so inner
product == cosine similarity.

Two access paths, both backed by the same vectors:
  - `SemanticIndex.faiss_search` uses the real FAISS index -- this is what
    satisfies "build an ANN index" and is what you'd use in a serving path
    with no eligibility filtering.
  - `SemanticIndex.score_dense` returns the full (n_queries, n_docs) inner
    product matrix via a plain matmul. IndexFlatIP is EXACT brute-force
    search under the hood (no approximation), so this is numerically
    identical to what faiss_search would return if it returned every
    document instead of just the top-k -- it's needed because Pool A/B
    recall@K (see SPEC.md Q2.0/Q3.4) requires masking candidates by
    eligibility/history/pool membership BEFORE picking top-K, which FAISS's
    own top-k search can't do without a per-query IDSelector. This mirrors
    how run_bm25_eval.py uses the sparse matmul for masked scoring while the
    CSR matrix itself is "the inverted index" for the unmasked case.

FAISS is optional at runtime: if the `faiss` package isn't installed, falls
back to a pure-numpy brute-force index with the identical interface (this is
explicitly a listed alternative in SPEC.md Q3.2, not a hack) -- so this
module still works for local development without the faiss-cpu wheel
installed, while ada (which has requirements.txt installed) gets the real
FAISS backend.
"""
from __future__ import annotations

import time

import numpy as np

from src.retrieval.bm25 import top_k_indices

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SemanticIndex:
    def __init__(self, article_ids: list[str], embeddings: np.ndarray):
        assert embeddings.dtype == np.float32
        self.article_ids = article_ids
        self.embeddings = embeddings  # (n_docs, dim), L2-normalised
        self.backend = "faiss" if FAISS_AVAILABLE else "numpy"

        if FAISS_AVAILABLE:
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            if embeddings.shape[0]:
                self._faiss_index.add(embeddings)

    def faiss_search(self, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices), each (n_queries, k), via the real ANN index
        (or its numpy-brute-force stand-in). For a plain, unmasked top-K demo/timing --
        NOT what recall@K pool evaluation uses (see module docstring)."""
        k = min(k, len(self.article_ids)) or 1
        if FAISS_AVAILABLE:
            return self._faiss_index.search(query_vecs, k)
        # Deterministic tie-break (see bm25.top_k_indices' docstring for why
        # argpartition alone is NOT cross-machine reproducible for ties, which
        # are common here too: many articles can share an identical cosine
        # score, e.g. when an embedding is genuinely absent/zero-adjacent).
        scores = query_vecs @ self.embeddings.T
        idx = top_k_indices(scores, k)
        row = np.arange(scores.shape[0])[:, None]
        return scores[row, idx], idx

    def score_dense(self, query_vecs: np.ndarray) -> np.ndarray:
        """Full (n_queries, n_docs) cosine-similarity matrix (see module docstring)."""
        return (query_vecs.astype(np.float32) @ self.embeddings.T)

    def benchmark_latency(self, n_queries: int = 200, k: int = 100, seed: int = 0) -> dict:
        """Query latency using the real ANN path -- cited in SPEC.md Q6's scale analysis."""
        if len(self.article_ids) == 0:
            return {"n_queries": 0, "k": k, "backend": self.backend, "total_sec": 0.0, "qps": 0.0}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.article_ids), size=min(n_queries, len(self.article_ids)))
        queries = self.embeddings[idx]
        t0 = time.time()
        self.faiss_search(queries, k)
        elapsed = time.time() - t0
        return {"n_queries": len(idx), "k": k, "backend": self.backend,
                "total_sec": elapsed, "qps": len(idx) / elapsed if elapsed > 0 else float("inf")}


def build_user_embedding(history_ids: list[str], embedding_lookup: dict[str, np.ndarray],
                          n: int, dim: int) -> tuple[np.ndarray, bool]:
    """Mean-pool the last n clicked articles' embeddings, L2-normalised.
    Returns (vector, is_cold_start) -- is_cold_start True means no covered history
    article was found (empty history, or every clicked article lacks an embedding --
    see build_embeddings.py's coverage note) and the caller should use the popularity
    fallback, same convention as run_bm25_eval.py's cold-start handling.
    """
    recent = history_ids[-n:] if n > 0 else []
    vecs = [embedding_lookup[a] for a in recent if a in embedding_lookup]
    if not vecs:
        return np.zeros(dim, dtype=np.float32), True
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.astype(np.float32), False
