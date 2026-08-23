"""Q9.1 -- Serving-feature ablation: report ranking metrics with and without
features that are unavailable at serving time.

EB-NeRD only. MIND's unified-schema columns for these features
(total_inviews/total_pageviews/total_read_time/sentiment_score on articles,
next_read_time/next_scroll_percentage on impressions) are all null/absent for
MIND (see clean_mind.py) -- the raw MINDsmall/MINDlarge files simply do not
carry post-interaction or corpus-aggregate engagement stats, so this ablation
is not constructible there. That is itself the honest answer, not a gap to
paper over.

Three cumulative configurations per method (bm25, semantic), run on the SAME
labeled offline split src.eval.run_eval already scores (data/splits/ebnerd/
<split>/impressions.parquet -- NOT the Codabench test sets, which are blind
and have no labels, so this exercise is impossible there regardless of
scale):

  1. clean               -- base retrieval score only. The deployable system.
  2. plus_popularity     -- + corpus-aggregate article stats (total_inviews,
                             total_pageviews, total_read_time, sentiment_score)
                             as a re-ranking prior. Unavailable at serving
                             time because they are aggregated over the WHOLE
                             corpus period, including the future relative to
                             any given impression.
  3. plus_post_interaction -- + a large bonus on whichever candidate is the
                             TRUE click for that impression (label == 1): the
                             most direct possible post-interaction feature,
                             since the click outcome IS the prediction target.
                             Expected to look excellent and be worthless.

NOTE on next_read_time/next_scroll_percentage: these columns exist in
impressions.parquet but are scoped to the WHOLE impression (one value per
impression row), not per-candidate. Every candidate in an impression would
receive the identical bonus, which is a no-op for AUC/MRR/nDCG (metrics
computed from each impression's own RELATIVE candidate ranking) -- adding
them cannot move any of these metrics, regardless of weight. This is reported
as a finding, not silently skipped: see the printed note in run_dataset().

Usage:
    python -m src.eval.run_ablation --split test

Writes:
    results/ebnerd_ablation.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import impression_auc, mrr, ndcg_at_k, rank_by_score
from src.retrieval.bm25 import BM25Index
from src.retrieval.scoring import best_n, score_bm25_batch, score_semantic_batch
from src.retrieval.semantic import SemanticIndex

BATCH_SIZE = 256
CONFIGS = ["clean", "plus_popularity", "plus_post_interaction"]
POPULARITY_WEIGHT = 1.0  # weight of the (min-max normalised) popularity bonus vs. the base score
LEAK_BONUS = 100.0  # deliberately huge -- guarantees the true click sorts first


def _minmax(xs: list[float]) -> np.ndarray:
    arr = np.asarray(xs, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _num(x, default: float = 0.0) -> float:
    """x or 0 doesn't catch NaN (NaN is truthy in Python) -- pandas leaves these
    engagement/sentiment fields as NaN, not None, for articles with no recorded stat."""
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _popularity_feature(candidates: list[str], article_stats: dict) -> list[float]:
    """Sum of log1p(corpus-aggregate engagement stats) + sentiment -- exactly the
    features SPEC.md Q9.1 names as unavailable at serving time (aggregated over
    the whole corpus period, including the future relative to any impression)."""
    vals = []
    for a in candidates:
        s = article_stats.get(a, {})
        v = (
            np.log1p(_num(s.get("total_inviews")))
            + np.log1p(_num(s.get("total_pageviews")))
            + np.log1p(_num(s.get("total_read_time")))
            + _num(s.get("sentiment_score"))
        )
        vals.append(v)
    return vals


def _augmented_scores(config: str, base_scores: list[float], candidates: list[str],
                       labels: list[int], article_stats: dict) -> list[float]:
    base = _minmax(base_scores)
    if config == "clean":
        return list(base)
    combined = base + POPULARITY_WEIGHT * _minmax(_popularity_feature(candidates, article_stats))
    if config == "plus_popularity":
        return list(combined)
    leak = np.array([LEAK_BONUS if lbl == 1 else 0.0 for lbl in labels])
    return list(combined + leak)


def _run_config(config: str, method: str, imp: pd.DataFrame, index, n: int, dim,
                 article_title: dict, embedding_lookup: dict, train_click_count: dict,
                 article_stats: dict) -> dict:
    records = []
    n_skipped_auc = 0
    for start in range(0, len(imp), BATCH_SIZE):
        batch = imp.iloc[start:start + BATCH_SIZE]
        if method == "bm25":
            score_maps = score_bm25_batch(index, batch, article_title, n, train_click_count)
        else:
            score_maps = score_semantic_batch(index, batch, embedding_lookup, n, dim, train_click_count)

        for row, score_map in zip(batch.itertuples(index=False), score_maps):
            candidates, labels = list(row.candidates), list(row.labels)
            base_scores = [float(score_map.get(a)) if score_map.get(a) is not None else -1e9 for a in candidates]
            scores = _augmented_scores(config, base_scores, candidates, labels, article_stats)

            _, _, ranked_labels = rank_by_score(candidates, scores, labels)
            auc = impression_auc(scores, labels)
            if auc is None:
                n_skipped_auc += 1
            records.append({
                "mrr": mrr(ranked_labels), "auc": auc,
                "ndcg5": ndcg_at_k(ranked_labels, 5), "ndcg10": ndcg_at_k(ranked_labels, 10),
            })

    out = {"n": len(records), "n_skipped_auc_single_class": n_skipped_auc}
    for key in ["mrr", "auc", "ndcg5", "ndcg10"]:
        mean, lo, hi = bootstrap_ci([r[key] for r in records])
        out[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    return out


def run_dataset(cfg: dict, split: str) -> None:
    dataset = "ebnerd"
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    imp = pd.read_parquet(splits_dir / dataset / split / "impressions.parquet")
    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet")
    train_click_count = dict(zip(articles["article_id"], articles["train_click_count"]))
    article_stats = {
        row.article_id: {
            "total_inviews": row.total_inviews, "total_pageviews": row.total_pageviews,
            "total_read_time": row.total_read_time, "sentiment_score": row.sentiment_score,
        }
        for row in articles.itertuples(index=False)
    }

    n_with_next_read_time = int(imp["next_read_time"].notna().sum()) if "next_read_time" in imp.columns else 0
    print(f"ebnerd: {n_with_next_read_time}/{len(imp)} impressions have next_read_time -- "
          f"present in the data but impression-scoped (one value per impression, not per "
          f"candidate), so it cannot move within-impression ranking metrics (AUC/MRR/nDCG) "
          f"at any weight; not used as a ranking feature for that reason (see module docstring)")

    emb_ids_path = feature_store_dir / dataset / "embeddings_ids.npy"
    if emb_ids_path.exists():
        ids = np.load(emb_ids_path, allow_pickle=True).tolist()
        vecs = np.load(feature_store_dir / dataset / "embeddings.npy")
        embedding_lookup = dict(zip(ids, vecs))
    else:
        embedding_lookup = {}

    results: dict = {}
    for method in ["bm25", "semantic"]:
        if method == "bm25":
            index = BM25Index.load(feature_store_dir / dataset / "bm25")
            n = best_n(results_dir, dataset, method, cfg["retrieval"]["bm25"]["query_history_n"])
            article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))
            dim = None
        else:
            ids = np.load(emb_ids_path, allow_pickle=True).tolist()
            vecs = np.load(feature_store_dir / dataset / "embeddings.npy")
            index = SemanticIndex(ids, vecs)
            n = best_n(results_dir, dataset, method, cfg["retrieval"]["semantic"]["user_history_n"])
            article_title = None
            dim = vecs.shape[1]

        results[method] = {}
        for config in CONFIGS:
            print(f"ebnerd/{method}/{config}: evaluating {split} ({len(imp)} impressions)...")
            r = _run_config(config, method, imp, index, n, dim, article_title, embedding_lookup,
                             train_click_count, article_stats)
            results[method][config] = r
            print(f"  AUC={r['auc']['mean']:.4f}  MRR={r['mrr']['mean']:.4f}  "
                  f"nDCG@5={r['ndcg5']['mean']:.4f}  nDCG@10={r['ndcg10']['mean']:.4f}")

    out_path = results_dir / "ebnerd_ablation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    run_dataset(cfg, args.split)


if __name__ == "__main__":
    main()
