"""Rigor addition -- measure the `rank_bm25` claim in bm25.py's module docstring instead
of just asserting it.

bm25.py says rank_bm25.BM25Okapi.get_scores loops over every document in pure Python per
query and "does not finish in reasonable time" at this project's scale. That was never
actually timed. Here we time both scorers, head-to-head, on the SAME tokenized corpus and
the SAME query set (a real dataset's articles + a sample of real click-history queries from
run_bm25_eval.py's own query construction), then extrapolate the measured per-query rate to
the full test-split query count -- rank_bm25 itself is only run on a bounded sample (a few
hundred queries) so this finishes in seconds rather than actually waiting out the slow path
at full scale.

Usage:
    python -m src.eval.bench_bm25_alternatives --datasets mind ebnerd --n-queries 200

Writes:
    results/<dataset>_bm25_alternatives_bench.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import load_config, resolve_path, seed_everything
from src.retrieval.bm25 import BM25Index, build_query_text
from src.retrieval.tokenize import DATASET_LANG, tokenize


def _sample_queries(splits_dir, dataset: str, article_title: dict, n_queries: int, seed: int) -> list[str]:
    imp = pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet", columns=["history_ids"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imp), size=min(n_queries, len(imp)), replace=False)
    histories = imp.iloc[idx]["history_ids"].tolist()
    queries = [build_query_text(list(h), article_title, n=5) for h in histories]
    return [q for q in queries if q.strip()]  # drop empty-history (cold-start) queries -- both scorers no-op on these anyway


def run_dataset(cfg: dict, dataset: str, n_queries: int) -> dict:
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    lang = DATASET_LANG[dataset]

    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet")
    texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    article_ids = articles["article_id"].tolist()
    article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))
    n_docs = len(article_ids)

    queries = _sample_queries(splits_dir, dataset, article_title, n_queries, cfg["seed"])
    print(f"{dataset}: {n_docs} docs, {len(queries)} sampled non-empty queries")

    # --- our hand-rolled CSR index: build + batched score ---
    t0 = time.perf_counter()
    index = BM25Index.build(article_ids, texts, lang)
    t_build_ours = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = index.score_batch(queries)
    t_score_ours = time.perf_counter() - t0

    # --- rank_bm25: build (tokenizes corpus once, same as ours) + per-query Python loop ---
    tokenized_corpus = [tokenize(t, lang=lang) for t in texts]
    t0 = time.perf_counter()
    bm25_alt = BM25Okapi(tokenized_corpus)
    t_build_alt = time.perf_counter() - t0

    tokenized_queries = [tokenize(q, lang=lang) for q in queries]
    t0 = time.perf_counter()
    for tq in tokenized_queries:
        _ = bm25_alt.get_scores(tq)
    t_score_alt = time.perf_counter() - t0

    qps_ours = len(queries) / t_score_ours if t_score_ours > 0 else float("inf")
    qps_alt = len(queries) / t_score_alt if t_score_alt > 0 else float("inf")

    # Extrapolate rank_bm25's measured per-query rate to the real test-split query count,
    # rather than actually running it at full scale.
    test_imp_n = len(pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet", columns=["timestamp"]))
    projected_alt_seconds_full_test = test_imp_n / qps_alt if qps_alt > 0 else float("inf")
    projected_ours_seconds_full_test = test_imp_n / qps_ours if qps_ours > 0 else float("inf")

    result = {
        "dataset": dataset,
        "n_docs": n_docs,
        "n_queries_sampled": len(queries),
        "ours_csr_matmul": {
            "build_seconds": t_build_ours,
            "score_seconds": t_score_ours,
            "queries_per_second": qps_ours,
        },
        "rank_bm25": {
            "build_seconds": t_build_alt,
            "score_seconds": t_score_alt,
            "queries_per_second": qps_alt,
        },
        "speedup_score_only": t_score_alt / t_score_ours if t_score_ours > 0 else float("inf"),
        "full_test_split_n_impressions": test_imp_n,
        "projected_full_test_seconds": {
            "ours_csr_matmul": projected_ours_seconds_full_test,
            "rank_bm25": projected_alt_seconds_full_test,
        },
    }
    print(f"  ours:      {t_score_ours:.4f}s for {len(queries)} queries ({qps_ours:.1f} q/s)")
    print(f"  rank_bm25: {t_score_alt:.4f}s for {len(queries)} queries ({qps_alt:.1f} q/s)")
    print(f"  speedup (scoring only): {result['speedup_score_only']:.1f}x")
    print(f"  projected full test split ({test_imp_n} impressions): "
          f"ours={projected_ours_seconds_full_test:.1f}s, "
          f"rank_bm25={projected_alt_seconds_full_test / 60:.1f}min")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        result = run_dataset(cfg, dataset, args.n_queries)
        out_path = results_dir / f"{dataset}_bm25_alternatives_bench.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"{dataset}: wrote {out_path}")


if __name__ == "__main__":
    main()
