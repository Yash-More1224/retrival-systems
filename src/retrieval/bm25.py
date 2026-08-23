"""Q2.1 -- Sparse BM25 inverted index (scipy CSR), not rank_bm25.

`rank_bm25.BM25Okapi.get_scores` loops over every document in pure Python for
every single query; with tens of thousands of evaluation impressions x tens
of thousands of articles this does not finish in reasonable time (see
SPEC.md Q2.1). Here, the doc-side BM25 weight for every (doc, term) pair is
precomputed once into a CSR matrix W; scoring a batch of queries is then a
single sparse matmul `Qmat @ W.T`, whose *sparsity pattern* (W.T, indexed by
term) is exactly a term -> postings inverted index.

sklearn's CountVectorizer is used only for the tokenize -> vocabulary ->
term-count-matrix bookkeeping (a standard, uncontroversial tool for that);
the actual BM25 weighting/scoring math below is hand-rolled, which is the
part that matters for the "build an inverted index, retrieve by BM25
scoring" requirement.
"""
from __future__ import annotations

import functools
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from src.retrieval.tokenize import tokenize


def _identity(x):
    return x


class BM25Index:
    def __init__(self, article_ids: list[str], vectorizer: CountVectorizer, W: sp.csr_matrix,
                 k1: float, b: float, doc_len: np.ndarray, avg_len: float):
        self.article_ids = article_ids
        self.vectorizer = vectorizer
        self.W = W  # (n_docs, n_terms) CSR, doc-side BM25 weight per (doc, term)
        self.k1 = k1
        self.b = b
        self.doc_len = doc_len
        self.avg_len = avg_len

    @classmethod
    def build(cls, article_ids: list[str], texts: list[str], lang: str,
              k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        # functools.partial over a module-level function (not a lambda closure) so the
        # vectorizer -- and therefore the whole index -- is picklable (see save()/load()).
        tok = functools.partial(tokenize, lang=lang)
        vectorizer = CountVectorizer(tokenizer=tok, preprocessor=_identity, token_pattern=None, lowercase=False)
        X = vectorizer.fit_transform(texts)  # (n_docs, n_terms) raw term counts, CSR

        n_docs = X.shape[0]
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avg_len = doc_len.mean() if n_docs else 0.0
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        Xc = X.tocoo()
        tf = Xc.data.astype(np.float64)
        row, col = Xc.row, Xc.col
        denom_norm = 1 - b + b * (doc_len[row] / avg_len if avg_len > 0 else 0.0)
        weight = idf[col] * (tf * (k1 + 1)) / (tf + k1 * denom_norm)

        W = sp.csr_matrix((weight.astype(np.float32), (row, col)), shape=X.shape)
        return cls(article_ids, vectorizer, W, k1, b, doc_len, float(avg_len))

    def score_batch(self, query_texts: list[str]) -> np.ndarray:
        """Returns a dense (n_queries, n_docs) float32 score matrix.

        Query-term weight is raw term count within the query (so a term
        appearing twice in a concatenated click-history query counts double)
        -- a standard, documented BM25 variant; see SPEC.md Q2.1.
        """
        Qmat = self.vectorizer.transform(query_texts).astype(np.float32)
        S = Qmat @ self.W.T  # sparse (n_queries, n_docs)
        return np.asarray(S.todense())

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(out_dir / "bm25_W.npz", self.W)
        with open(out_dir / "bm25_meta.pkl", "wb") as f:
            pickle.dump({
                "article_ids": self.article_ids,
                "vectorizer": self.vectorizer,
                "k1": self.k1,
                "b": self.b,
                "doc_len": self.doc_len,
                "avg_len": self.avg_len,
            }, f)

    @classmethod
    def load(cls, out_dir: Path) -> "BM25Index":
        W = sp.load_npz(out_dir / "bm25_W.npz")
        with open(out_dir / "bm25_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(meta["article_ids"], meta["vectorizer"], W, meta["k1"], meta["b"],
                    meta["doc_len"], meta["avg_len"])


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """scores: (n_queries, n_docs). Returns (n_queries, k) column indices, sorted
    descending by score within each row, with a deterministic tie-break by
    ascending column index.

    BUG FIX (found 2026-08-21 by comparing local vs. ada recall@K numbers for
    the same seed/data/code -- see docs/ai_usage_log.md): the original version
    used np.argpartition to select the top-k, then argsort only within that
    slice. argpartition makes NO guarantee about *which* tied elements end up
    in the partition -- for a sparse single-title query (N=1), the vast
    majority of the catalog scores exactly 0.0 (no shared terms), so with
    k up to 200 there often aren't 200 nonzero-scoring docs, and argpartition
    arbitrarily -- differently per numpy/BLAS build -- picks which zero-score
    docs fill the rest. That silently changed recall@K (and therefore the
    N-sweep's best-N selection) between machines even with the same seed.
    A full stable argsort has no such ambiguity: ties are always broken by
    ascending original index, identically on every platform.
    """
    k = min(k, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k]


def build_query_text(history_ids: list[str], article_title: dict[str, str], n: int) -> str:
    """Concatenate the titles of the last n clicked articles (SPEC.md Q2.3)."""
    recent = history_ids[-n:] if n > 0 else []
    titles = [article_title[a] for a in recent if a in article_title]
    return " ".join(titles)
