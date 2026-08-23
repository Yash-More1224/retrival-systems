"""Q1.4 -- Materialize the feature store: article features + user features.

Article features (title, abstract, body, category, entities, embeddings-to-come)
live at the dataset level -- one table per dataset, not per split, since the
catalog itself doesn't change across splits (only *eligibility* at retrieval
time does, which is Q2/Q3's job, not the feature store's).

Also attaches `train_click_count` per article: how many times it was clicked
in the TRAIN split only. This is precomputed here because it's a static
article-level feature several downstream consumers need (novelty in Q4.2,
head/tail slicing in Q4.3) and re-deriving it from scratch in every consumer
risks accidentally including val/test clicks, which would be leakage
(see SPEC.md Q4.2's "popularity counted on the train split only" note).

User features are per split: one row per user with their click history as of
that split, embedded (not re-derived) -- this is exactly what clean_ebnerd.py
/ clean_mind.py already produced per impression; here we just dedupe to one
row per user.

Every artifact gets a sidecar `<name>.meta.json` recording the config hash it
was built from, so `build_pipeline.py --force` knows what's stale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path


def _config_hash(cfg: dict) -> str:
    path = Path(cfg["_config_path"])
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _write_meta(path: Path, config_hash: str, **extra) -> None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps({"config_hash": config_hash, **extra}, indent=2))


def _train_click_counts(splits_dir: Path, dataset: str) -> pd.Series:
    train = pd.read_parquet(splits_dir / dataset / "train" / "impressions.parquet")
    counts: dict[str, int] = {}
    for cands, labels in zip(train["candidates"], train["labels"]):
        for article_id, label in zip(cands, labels):
            if label == 1:
                counts[article_id] = counts.get(article_id, 0) + 1
    return pd.Series(counts, name="train_click_count")


def build_article_features(interim_dir: Path, splits_dir: Path, feature_store_dir: Path,
                            dataset: str, interim_subdir: str, config_hash: str) -> None:
    articles = pd.read_parquet(interim_dir / interim_subdir / "articles.parquet")
    click_counts = _train_click_counts(splits_dir, dataset)
    articles = articles.merge(
        click_counts.rename_axis("article_id").reset_index(), on="article_id", how="left"
    )
    articles["train_click_count"] = articles["train_click_count"].fillna(0).astype(int)

    out_dir = feature_store_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "articles.parquet"
    articles.to_parquet(out_path, index=False)
    _write_meta(out_path, config_hash, n_articles=len(articles),
                n_with_nonzero_train_clicks=int((articles["train_click_count"] > 0).sum()))
    print(f"{dataset}: wrote {len(articles)} article feature rows -> {out_path}")


def build_user_features(splits_dir: Path, feature_store_dir: Path, dataset: str, config_hash: str) -> None:
    out_dir = feature_store_dir / dataset / "users"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        imp = pd.read_parquet(splits_dir / dataset / split / "impressions.parquet")
        users = (
            imp[["user_id", "history_ids", "history_times", "history_len"]]
            .drop_duplicates(subset="user_id", keep="first")
            .reset_index(drop=True)
        )
        out_path = out_dir / f"{split}.parquet"
        users.to_parquet(out_path, index=False)
        _write_meta(out_path, config_hash, n_users=len(users),
                    n_cold_start=int((users["history_len"] == 0).sum()))
        print(f"{dataset}/{split}: wrote {len(users)} user feature rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--ebnerd-scale", choices=["demo", "small"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    interim_dir = resolve_path(cfg, "interim_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    config_hash = _config_hash(cfg)

    if "mind" in args.datasets:
        build_article_features(interim_dir, splits_dir, feature_store_dir, "mind", "mind", config_hash)
        build_user_features(splits_dir, feature_store_dir, "mind", config_hash)
    if "ebnerd" in args.datasets:
        scale = args.ebnerd_scale or cfg["ebnerd"]["scale"]
        build_article_features(interim_dir, splits_dir, feature_store_dir, "ebnerd", f"ebnerd_{scale}", config_hash)
        build_user_features(splits_dir, feature_store_dir, "ebnerd", config_hash)


if __name__ == "__main__":
    main()
