"""Q2 -- BM25 candidate generation: build the index, sweep query-history length N
on val, then report recall@K on test under Pool A and Pool B with bootstrap CIs.

Usage:
    python -m src.retrieval.run_bm25_eval --datasets mind ebnerd

Writes:
    results/<dataset>_bm25_val_sweep.json
    results/<dataset>_bm25_test.json
    feature_store/<dataset>/bm25/  (cached index; --force to rebuild)

Cold-start handling (Q2.3): an impression whose query text ends up empty
(empty click history, or a history of articles missing from the catalog)
falls back to a train-split popularity ranking rather than an all-zero BM25
score row, and is additionally reported as its own slice count in the output
-- see SPEC.md Q2.3 and Q4.3.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.eval.bootstrap import bootstrap_ci
from src.retrieval.bm25 import BM25Index, build_query_text, top_k_indices
from src.retrieval.candidates import (
    active_pool_b,
    eligibility_mask,
    exclude_history_inplace,
    recall_at_k_batch,
)
from src.retrieval.tokenize import DATASET_LANG

BATCH_SIZE = 256


def _load_or_build_index(feature_store_dir, dataset, articles: pd.DataFrame, force: bool) -> BM25Index:
    index_dir = feature_store_dir / dataset / "bm25"
    if not force and (index_dir / "bm25_W.npz").exists():
        print(f"{dataset}: loading cached BM25 index from {index_dir}")
        return BM25Index.load(index_dir)

    lang = DATASET_LANG[dataset]
    texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    article_ids = articles["article_id"].tolist()
    print(f"{dataset}: building BM25 index over {len(article_ids)} articles ({lang})")
    index = BM25Index.build(article_ids, texts, lang)
    index.save(index_dir)
    return index


def _evaluate(index: BM25Index, articles: pd.DataFrame, imp: pd.DataFrame, pool_b: set[str],
              n: int, ks: list[int]) -> dict:
    article_ids = index.article_ids
    article_id_to_idx = {a: i for i, a in enumerate(article_ids)}
    article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))
    published_time = articles["published_time"].to_numpy()
    popularity = articles["train_click_count"].to_numpy().astype(np.float32)
    pool_b_bool = np.array([a in pool_b for a in article_ids], dtype=bool)

    k_max = max(ks)
    per_impression = {("A", k): [] for k in ks}
    per_impression.update({("B", k): [] for k in ks})
    n_empty_query = 0
    n_total = 0

    for start in range(0, len(imp), BATCH_SIZE):
        batch = imp.iloc[start:start + BATCH_SIZE]
        query_texts = [
            build_query_text(list(h), article_title, n) for h in batch["history_ids"]
        ]
        is_empty = np.array([not qt.strip() for qt in query_texts])
        n_empty_query += int(is_empty.sum())
        n_total += len(batch)

        S = index.score_batch(query_texts)
        if is_empty.any():
            S[is_empty] = popularity  # cold-start fallback: rank by train-split popularity

        history_lists = [list(h) for h in batch["history_ids"]]
        ground_truth = [
            {a for a, l in zip(cands, labels) if l == 1}
            for cands, labels in zip(batch["candidates"], batch["labels"])
        ]

        # Pool A: eligible-by-publish-time (EB-NeRD only) minus own history.
        mask_a = eligibility_mask(article_ids, published_time, batch["timestamp"])
        exclude_history_inplace(mask_a, article_id_to_idx, history_lists)
        scores_a = np.where(mask_a, S, -np.inf)
        topk_a = top_k_indices(scores_a, k_max)
        ids_a = [[article_ids[j] for j in row] for row in topk_a]

        # Pool B: restricted to articles shown in some impression in this split, minus history.
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


def run_dataset(cfg: dict, dataset: str, force_index: bool) -> None:
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    ks = cfg["retrieval"]["bm25"]["top_k"]

    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet")
    index = _load_or_build_index(feature_store_dir, dataset, articles, force_index)

    val_imp = pd.read_parquet(splits_dir / dataset / "val" / "impressions.parquet")
    val_pool_b = active_pool_b(splits_dir, dataset, "val")

    print(f"{dataset}: sweeping query-history N on val ({len(val_imp)} impressions)...")
    sweep = {}
    for n in [1, 5, 10, 20]:
        res = _evaluate(index, articles, val_imp, val_pool_b, n, ks)
        sweep[str(n)] = res
        headline = res["recall"].get(f"A@{ks[1] if len(ks) > 1 else ks[0]}", {}).get("mean")
        print(f"  N={n}: recall A@{ks[1] if len(ks) > 1 else ks[0]} = {headline:.4f}" if headline is not None else f"  N={n}: (no data)")

    def score_for_selection(res: dict) -> float:
        key = f"A@{ks[len(ks) // 2]}"
        key_b = f"B@{ks[len(ks) // 2]}"
        a = res["recall"][key]["mean"] or 0.0
        b = res["recall"][key_b]["mean"] or 0.0
        return (a + b) / 2

    best_n = int(max(sweep, key=lambda n: score_for_selection(sweep[n])))
    print(f"{dataset}: best N = {best_n} (selected by mean of A/B recall@{ks[len(ks) // 2]} on val)")

    (results_dir / f"{dataset}_bm25_val_sweep.json").write_text(
        json.dumps({"sweep": sweep, "best_n": best_n}, indent=2)
    )

    test_imp = pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet")
    test_pool_b = active_pool_b(splits_dir, dataset, "test")
    print(f"{dataset}: final eval on test ({len(test_imp)} impressions) with N={best_n}...")
    test_result = _evaluate(index, articles, test_imp, test_pool_b, int(best_n), ks)
    (results_dir / f"{dataset}_bm25_test.json").write_text(json.dumps(test_result, indent=2))
    print(f"{dataset}: wrote results/{dataset}_bm25_test.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-index", action="store_true", help="rebuild the BM25 index even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    for dataset in args.datasets:
        run_dataset(cfg, dataset, args.force_index)


if __name__ == "__main__":
    main()
