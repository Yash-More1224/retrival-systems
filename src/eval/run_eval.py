"""Q4.5 -- Offline evaluation harness: run Q4.1/4.2 metrics, sliced (Q4.3), with
bootstrap CIs (Q4.4), on BOTH BM25 and semantic ranking of each impression's own
candidate list.

This is a DIFFERENT task from Q2/Q3's recall@K: there, top-K was retrieved from
the whole catalog against ground-truth clicks (candidate generation). Here,
every candidate already in the impression (~20 items) is SCORED and RANKED
against each other -- the task AUC/MRR/nDCG and both Codabench leaderboards
actually measure (SPEC.md Q4.0).

Usage:
    python -m src.eval.run_eval --datasets mind ebnerd --split test

Writes:
    results/<dataset>_<method>_eval.json   for method in {bm25, semantic}

Query-history length N is read from each method's Q2/Q3 val-sweep result
(results/<dataset>_<method>_val_sweep.json's "best_n"), falling back to the
config default if that file doesn't exist yet -- so Q4 stays consistent with
whatever N Q2/Q3 already selected rather than re-tuning independently.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.eval.beyond_accuracy import coverage, ild_categorical, ild_embedding, novelty
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import impression_auc, mrr, ndcg_at_k, rank_by_score
from src.eval.slicing import (
    cold_start_or_warm,
    compute_head_threshold,
    first_clicked_article,
    head_or_tail,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.scoring import best_n as _best_n
from src.retrieval.scoring import score_bm25_batch as _score_bm25_batch
from src.retrieval.scoring import score_semantic_batch as _score_semantic_batch
from src.retrieval.semantic import SemanticIndex

BATCH_SIZE = 256


def _run_method(cfg: dict, dataset: str, method: str, imp: pd.DataFrame, articles: pd.DataFrame,
                 embedding_lookup: dict, feature_store_dir, results_dir) -> dict:
    ks = cfg["eval"]["ndcg_ks"]
    cutoff = cfg["eval"]["beyond_accuracy_cutoff"]
    cold_threshold = cfg["eval"]["cold_start_threshold"]

    train_click_count = dict(zip(articles["article_id"], articles["train_click_count"]))
    total_train_clicks = int(articles["train_click_count"].sum())
    head_threshold = compute_head_threshold(articles["train_click_count"].to_numpy(), cfg["eval"]["head_tail_top_pct"])
    article_category = dict(zip(articles["article_id"], articles["category"]))

    if method == "bm25":
        index = BM25Index.load(feature_store_dir / dataset / "bm25")
        n = _best_n(results_dir, dataset, method, cfg["retrieval"]["bm25"]["query_history_n"])
        article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))
    else:
        ids = np.load(feature_store_dir / dataset / "embeddings_ids.npy", allow_pickle=True).tolist()
        embeddings = np.load(feature_store_dir / dataset / "embeddings.npy")
        index = SemanticIndex(ids, embeddings)
        n = _best_n(results_dir, dataset, method, cfg["retrieval"]["semantic"]["user_history_n"])
        dim = embeddings.shape[1]

    records = []
    n_skipped_auc = 0
    n_uncovered_candidates = 0

    for start in range(0, len(imp), BATCH_SIZE):
        batch = imp.iloc[start:start + BATCH_SIZE]
        if method == "bm25":
            score_maps = _score_bm25_batch(index, batch, article_title, n, train_click_count)
        else:
            score_maps = _score_semantic_batch(index, batch, embedding_lookup, n, dim, train_click_count)

        for row, score_map in zip(batch.itertuples(index=False), score_maps):
            candidates, labels = list(row.candidates), list(row.labels)
            scores = []
            for a in candidates:
                s = score_map.get(a)
                if s is None:
                    n_uncovered_candidates += 1
                    s = -1e9  # not in this method's retrievable catalog -> ranked last, not dropped
                scores.append(float(s))

            ranked_ids, ranked_scores, ranked_labels = rank_by_score(candidates, scores, labels)
            auc = impression_auc(scores, labels)
            if auc is None:
                n_skipped_auc += 1

            top_ids = ranked_ids[:cutoff]
            rec = {
                "mrr": mrr(ranked_labels),
                "auc": auc,
                "ild_categorical": ild_categorical([article_category.get(a, "") for a in top_ids]),
                "ild_embedding": ild_embedding(top_ids, embedding_lookup),
                "novelty": novelty(top_ids, train_click_count, total_train_clicks),
                "top_ids": top_ids,
                "cold_warm": cold_start_or_warm(row.history_len, cold_threshold),
                "head_tail": head_or_tail(first_clicked_article(candidates, labels), train_click_count, head_threshold),
            }
            for k in ks:
                rec[f"ndcg{k}"] = ndcg_at_k(ranked_labels, k)
            records.append(rec)

    n_total_articles = len(articles)
    result = {
        "n_impressions": len(records),
        "n_skipped_auc_single_class": n_skipped_auc,
        "n_uncovered_candidates": n_uncovered_candidates,
        "n": n,
        "overall": _aggregate(records, ks, n_total_articles),
        "slices": {
            "cold_start": _aggregate([r for r in records if r["cold_warm"] == "cold_start"], ks, n_total_articles),
            "warm": _aggregate([r for r in records if r["cold_warm"] == "warm"], ks, n_total_articles),
            "head": _aggregate([r for r in records if r["head_tail"] == "head"], ks, n_total_articles),
            "tail": _aggregate([r for r in records if r["head_tail"] == "tail"], ks, n_total_articles),
        },
    }
    return result


def _aggregate(records: list[dict], ks: list[int], catalog_size: int) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for key in ["mrr", "auc", "ild_categorical", "ild_embedding", "novelty"] + [f"ndcg{k}" for k in ks]:
        mean, lo, hi = bootstrap_ci([r[key] for r in records])
        out[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    out["coverage"] = coverage([r["top_ids"] for r in records], catalog_size)
    return out


def run_dataset(cfg: dict, dataset: str, split: str) -> None:
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    imp = pd.read_parquet(splits_dir / dataset / split / "impressions.parquet")
    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet")

    emb_ids_path = feature_store_dir / dataset / "embeddings_ids.npy"
    if emb_ids_path.exists():
        ids = np.load(emb_ids_path, allow_pickle=True).tolist()
        vecs = np.load(feature_store_dir / dataset / "embeddings.npy")
        embedding_lookup = dict(zip(ids, vecs))
    else:
        embedding_lookup = {}

    for method in ["bm25", "semantic"]:
        print(f"{dataset}/{method}: evaluating {split} ({len(imp)} impressions)...")
        result = _run_method(cfg, dataset, method, imp, articles, embedding_lookup, feature_store_dir, results_dir)
        out_path = results_dir / f"{dataset}_{method}_eval.json"
        out_path.write_text(json.dumps(result, indent=2))
        o = result["overall"]
        k0 = cfg["eval"]["ndcg_ks"][0]
        ndcg_k0 = o[f"ndcg{k0}"]["mean"]
        print(f"  n={o['n']}, AUC={o['auc']['mean']:.4f}, MRR={o['mrr']['mean']:.4f}, "
              f"nDCG@{k0}={ndcg_k0:.4f}, coverage={o['coverage']:.4f} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    for dataset in args.datasets:
        run_dataset(cfg, dataset, args.split)


if __name__ == "__main__":
    main()
