"""Q3 -- Semantic candidate generation: load provided embeddings, sweep user-history
length N on val, report recall@K on test under Pool A/Pool B with bootstrap CIs, and
benchmark real FAISS query latency.

Usage:
    python -m src.retrieval.build_embeddings --datasets mind ebnerd   # once, or after a rebuild
    python -m src.retrieval.run_semantic_eval --datasets mind ebnerd

Writes:
    results/<dataset>_semantic_val_sweep.json
    results/<dataset>_semantic_test.json

Mirrors run_bm25_eval.py's structure and pool/CI machinery (src/retrieval/candidates.py,
src/eval/bootstrap.py) so the two are directly comparable for Q3.5's lexical-vs-semantic
analysis. The retrievable catalog here is the embedding-COVERED subset only (see
build_embeddings.py's coverage note) -- narrower than BM25's full-text catalog for MIND in
particular, and that's reported explicitly (`catalog_coverage`) rather than left implicit,
since it directly affects how a lexical-vs-semantic recall comparison should be read.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.eval.bootstrap import bootstrap_ci
from src.retrieval.bm25 import top_k_indices
from src.retrieval.candidates import (
    active_pool_b,
    eligibility_mask,
    exclude_history_inplace,
    recall_at_k_batch,
)
from src.retrieval.semantic import SemanticIndex, build_user_embedding

BATCH_SIZE = 512


def _load_embeddings(feature_store_dir, dataset) -> tuple[list[str], np.ndarray, dict]:
    out_dir = feature_store_dir / dataset
    ids = np.load(out_dir / "embeddings_ids.npy", allow_pickle=True).tolist()
    embeddings = np.load(out_dir / "embeddings.npy")
    meta = json.loads((out_dir / "embeddings.meta.json").read_text())
    return ids, embeddings, meta


def _evaluate(index: SemanticIndex, article_id_to_idx: dict, published_time: np.ndarray,
              popularity: np.ndarray, imp: pd.DataFrame, pool_b: set[str],
              embedding_lookup: dict, n: int, dim: int, ks: list[int]) -> dict:
    article_ids = index.article_ids
    pool_b_bool = np.array([a in pool_b for a in article_ids], dtype=bool)
    k_max = max(ks)

    per_impression = {("A", k): [] for k in ks}
    per_impression.update({("B", k): [] for k in ks})
    n_empty_query = 0
    n_total = 0

    for start in range(0, len(imp), BATCH_SIZE):
        batch = imp.iloc[start:start + BATCH_SIZE]
        history_lists = [list(h) for h in batch["history_ids"]]

        query_vecs = np.zeros((len(batch), dim), dtype=np.float32)
        is_empty = np.zeros(len(batch), dtype=bool)
        for i, hist in enumerate(history_lists):
            vec, cold = build_user_embedding(hist, embedding_lookup, n, dim)
            query_vecs[i] = vec
            is_empty[i] = cold
        n_empty_query += int(is_empty.sum())
        n_total += len(batch)

        S = index.score_dense(query_vecs)
        if is_empty.any():
            S[is_empty] = popularity  # cold-start fallback: rank by train-split popularity

        ground_truth = [
            {a for a, l in zip(cands, labels) if l == 1}
            for cands, labels in zip(batch["candidates"], batch["labels"])
        ]

        mask_a = eligibility_mask(article_ids, published_time, batch["timestamp"])
        exclude_history_inplace(mask_a, article_id_to_idx, history_lists)
        scores_a = np.where(mask_a, S, -np.inf)
        topk_a = top_k_indices(scores_a, k_max)
        ids_a = [[article_ids[j] for j in row] for row in topk_a]

        mask_b = np.tile(pool_b_bool, (len(batch), 1)).copy()
        exclude_history_inplace(mask_b, article_id_to_idx, history_lists)
        scores_b = np.where(mask_b, S, -np.inf)
        topk_b = top_k_indices(scores_b, k_max)
        ids_b = [[article_ids[j] for j in row] for row in topk_b]

        for k in ks:
            per_impression[("A", k)].extend(recall_at_k_batch([r[:k] for r in ids_a], ground_truth))
            per_impression[("B", k)].extend(recall_at_k_batch([r[:k] for r in ids_b], ground_truth))

    result = {"n_impressions": n_total, "n_empty_query_coldstart": n_empty_query, "n": n, "recall": {}}
    for (pool, k), values in per_impression.items():
        mean, lo, hi = bootstrap_ci(values)
        result["recall"][f"{pool}@{k}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_scored": sum(v is not None for v in values)}
    return result


def run_dataset(cfg: dict, dataset: str) -> None:
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    ks = cfg["retrieval"]["semantic"]["top_k"]

    article_ids, embeddings, emb_meta = _load_embeddings(feature_store_dir, dataset)
    dim = embeddings.shape[1]
    embedding_lookup = dict(zip(article_ids, embeddings))
    article_id_to_idx = {a: i for i, a in enumerate(article_ids)}

    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet",
                                columns=["article_id", "published_time", "train_click_count"])
    articles = articles[articles["article_id"].isin(article_id_to_idx)].set_index("article_id")
    articles = articles.loc[article_ids]  # align to embedding row order
    published_time = articles["published_time"].to_numpy()
    popularity = articles["train_click_count"].to_numpy().astype(np.float32)

    index = SemanticIndex(article_ids, embeddings)
    latency = index.benchmark_latency(k=max(ks))
    print(f"{dataset}: {index.backend} backend, catalog {emb_meta['n_covered']}/{emb_meta['n_total']} "
          f"({emb_meta['coverage_pct']}%) covered, latency {latency['qps']:.0f} qps @k={max(ks)}")

    val_imp = pd.read_parquet(splits_dir / dataset / "val" / "impressions.parquet")
    val_pool_b = active_pool_b(splits_dir, dataset, "val")

    print(f"{dataset}: sweeping user-history N on val ({len(val_imp)} impressions)...")
    sweep = {}
    for n in [1, 5, 10, 20]:
        res = _evaluate(index, article_id_to_idx, published_time, popularity, val_imp, val_pool_b,
                         embedding_lookup, n, dim, ks)
        sweep[str(n)] = res
        mid_k = ks[len(ks) // 2]
        headline = res["recall"].get(f"A@{mid_k}", {}).get("mean")
        print(f"  N={n}: recall A@{mid_k} = {headline:.4f}" if headline is not None else f"  N={n}: (no data)")

    def score_for_selection(res: dict) -> float:
        mid_k = ks[len(ks) // 2]
        a = res["recall"][f"A@{mid_k}"]["mean"] or 0.0
        b = res["recall"][f"B@{mid_k}"]["mean"] or 0.0
        return (a + b) / 2

    best_n = int(max(sweep, key=lambda n: score_for_selection(sweep[n])))
    print(f"{dataset}: best N = {best_n} (selected by mean of A/B recall@{ks[len(ks) // 2]} on val)")

    (results_dir / f"{dataset}_semantic_val_sweep.json").write_text(
        json.dumps({"sweep": sweep, "best_n": best_n, "embedding_meta": emb_meta, "latency": latency}, indent=2)
    )

    test_imp = pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet")
    test_pool_b = active_pool_b(splits_dir, dataset, "test")
    print(f"{dataset}: final eval on test ({len(test_imp)} impressions) with N={best_n}...")
    test_result = _evaluate(index, article_id_to_idx, published_time, popularity, test_imp, test_pool_b,
                             embedding_lookup, int(best_n), dim, ks)
    test_result["embedding_meta"] = emb_meta
    test_result["latency"] = latency
    (results_dir / f"{dataset}_semantic_test.json").write_text(json.dumps(test_result, indent=2))
    print(f"{dataset}: wrote results/{dataset}_semantic_test.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    for dataset in args.datasets:
        run_dataset(cfg, dataset)


if __name__ == "__main__":
    main()
