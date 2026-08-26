"""Rigor addition -- measure the tokenizer/stemming choice instead of just asserting it.

design_note.tex (S3) claimed Danish text uses "language-specific tokenisation and
stemming", but src/retrieval/tokenize.py's `stem` parameter defaults to False, and no
caller (bm25.py, run_bm25_eval.py, the two submission scripts) ever passed stem=True --
stemming was never actually applied by the shipped pipeline. This script builds a second,
stemmed BM25 index per dataset and compares recall@K against the real (unstemmed) index
already used everywhere else, at the SAME best_n already selected on val (see
results/<dataset>_bm25_val_sweep.json) so the tokenizer is the only variable changed.

Usage:
    python -m src.eval.run_tokenizer_ablation --datasets mind ebnerd

Writes:
    results/<dataset>_tokenizer_ablation.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.retrieval.bm25 import BM25Index
from src.retrieval.candidates import active_pool_b
from src.retrieval.run_bm25_eval import _evaluate
from src.retrieval.tokenize import DATASET_LANG


def run_dataset(cfg: dict, dataset: str) -> dict:
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    results_dir = resolve_path(cfg, "results_dir")
    ks = cfg["retrieval"]["bm25"]["top_k"]
    lang = DATASET_LANG[dataset]

    sweep_path = results_dir / f"{dataset}_bm25_val_sweep.json"
    best_n = int(json.loads(sweep_path.read_text())["best_n"]) if sweep_path.exists() else int(cfg["retrieval"]["bm25"]["query_history_n"])

    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet")
    texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    article_ids = articles["article_id"].tolist()

    val_imp = pd.read_parquet(splits_dir / dataset / "val" / "impressions.parquet")
    val_pool_b = active_pool_b(splits_dir, dataset, "val")

    print(f"{dataset}: building unstemmed (current-pipeline) and stemmed indices, N={best_n}...")
    idx_unstemmed = BM25Index.build(article_ids, texts, lang, stem=False)
    idx_stemmed = BM25Index.build(article_ids, texts, lang, stem=True)

    res_unstemmed = _evaluate(idx_unstemmed, articles, val_imp, val_pool_b, best_n, ks)
    res_stemmed = _evaluate(idx_stemmed, articles, val_imp, val_pool_b, best_n, ks)

    result = {
        "dataset": dataset,
        "lang": lang,
        "n_used": best_n,
        "unstemmed_current_pipeline": res_unstemmed["recall"],
        "stemmed": res_stemmed["recall"],
    }
    for k in ks:
        for pool in ["A", "B"]:
            key = f"{pool}@{k}"
            u = res_unstemmed["recall"][key]["mean"]
            s = res_stemmed["recall"][key]["mean"]
            delta = None if (u is None or s is None) else s - u
            print(f"  {key}: unstemmed={u:.4f} stemmed={s:.4f} delta={delta:+.4f}" if delta is not None
                  else f"  {key}: unstemmed={u} stemmed={s}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        result = run_dataset(cfg, dataset)
        out_path = results_dir / f"{dataset}_tokenizer_ablation.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"{dataset}: wrote {out_path}")


if __name__ == "__main__":
    main()
