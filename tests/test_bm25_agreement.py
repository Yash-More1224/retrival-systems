"""Sanity check for the hand-rolled sparse BM25 (src/retrieval/bm25.py) against the
reference `rank_bm25.BM25Okapi` implementation on a small subsample (SPEC.md Q2.1:
rank_bm25 is fine for a correctness check, just not for full-scale scoring).

Requires the real feature store (`make data`); skips if not present or if rank_bm25
isn't installed (it's a dev-only reference dependency, not required at eval time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.retrieval.bm25 import BM25Index, build_query_text
from src.retrieval.tokenize import DATASET_LANG, tokenize

N_DOCS_SUBSAMPLE = 500
N_QUERIES = 20


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_bm25_scores_agree_with_rank_bm25(dataset):
    rank_bm25 = pytest.importorskip("rank_bm25")

    cfg = load_config()
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    articles_path = feature_store_dir / dataset / "articles.parquet"
    if not articles_path.exists():
        pytest.skip(f"{articles_path} not found -- run `make data` first")

    articles = pd.read_parquet(articles_path).head(N_DOCS_SUBSAMPLE).reset_index(drop=True)
    lang = DATASET_LANG[dataset]
    texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    article_ids = articles["article_id"].tolist()

    ours = BM25Index.build(article_ids, texts, lang)

    corpus_tokens = [tokenize(t, lang=lang) for t in texts]
    reference = rank_bm25.BM25Okapi(corpus_tokens, k1=ours.k1, b=ours.b)

    rng = np.random.default_rng(0)
    query_idx = rng.choice(len(texts), size=min(N_QUERIES, len(texts)), replace=False)
    max_abs_diff = 0.0
    for i in query_idx:
        query_tokens = corpus_tokens[i][:5] or corpus_tokens[i]  # a short, realistic query
        query_text = " ".join(query_tokens)
        if not query_text:
            continue

        our_scores = ours.score_batch([query_text])[0]
        ref_scores = np.asarray(reference.get_scores(query_tokens))

        assert our_scores.shape == ref_scores.shape
        max_abs_diff = max(max_abs_diff, float(np.abs(our_scores - ref_scores).max()))

    # rank_bm25's IDF formula differs slightly in sign convention from ours for very common
    # terms (both are standard Okapi BM25 variants; see SPEC.md Q2.1), so use a generous
    # but still meaningful tolerance rather than exact equality.
    assert max_abs_diff < 1.0, f"{dataset}: max |our_score - rank_bm25_score| = {max_abs_diff:.4f}, expected < 1.0"
