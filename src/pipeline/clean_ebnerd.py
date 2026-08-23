"""Q1.2 -- Parse raw EB-NeRD files into the unified schema.

Reads ebnerd_{demo,small}.zip (already downloaded), extracts them, and writes:

  data/interim/ebnerd_<scale>/articles.parquet
  data/interim/ebnerd_<scale>/impressions.parquet

Schema real columns verified by directly reading the parquet files (see
SPEC.md "Verified dataset facts") -- this is NOT the schema the assignment
PDF's Part 0 implies, so don't rebuild this from the assignment text alone.

Key mapping decisions (documented here because they're not obvious from the
raw column names):
  - candidates = article_ids_inview; labels = membership of
    article_ids_clicked within article_ids_inview, in inview order.
  - history comes from a SEPARATE history.parquet per source split ("train"
    or "validation"), one row per user, joined onto behaviors by user_id.
    It is NOT one row per interaction.
  - `article_id` on behaviors.parquet is the context article being read when
    the impression fired (often null) -- it is NOT a candidate and NOT used.
  - next_read_time / next_scroll_percentage / total_inviews / total_pageviews /
    total_read_time / sentiment_score are kept but quarantined as
    serving-time-unavailable (Q9.1 ablation) -- never used by the default
    retrieval/ranking path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.pipeline.unzip_utils import extract_clean

SOURCE_FOLDERS = {"train_raw": "train", "validation_raw": "validation"}


def _load_articles(articles_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(articles_parquet)
    out = pd.DataFrame({
        "article_id": df["article_id"].astype(str),
        "dataset": "ebnerd",
        "title": df["title"].fillna(""),
        "abstract": df["subtitle"].fillna(""),
        "body": df["body"].fillna(""),
        "category": df["category_str"].fillna(""),
        "subcategory": df["subcategory"].apply(lambda xs: [str(x) for x in xs] if xs is not None else []),
        "entities": df["ner_clusters"].apply(lambda xs: list(xs) if xs is not None else []),
        "published_time": df["published_time"],
        "total_inviews": df["total_inviews"],
        "total_pageviews": df["total_pageviews"],
        "total_read_time": df["total_read_time"],
        "sentiment_score": df["sentiment_score"],
    })
    return out


def _load_history(history_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(history_parquet)
    return pd.DataFrame({
        "user_id": df["user_id"].astype(str),
        "history_ids": df["article_id_fixed"].apply(lambda xs: [str(x) for x in xs] if xs is not None else []),
        "history_times": df["impression_time_fixed"].apply(lambda xs: list(xs) if xs is not None else []),
    })


def _labels_for_impression(inview: list, clicked: list) -> list[int] | None:
    """Returns None (impression should be dropped) if any clicked id is missing from inview."""
    clicked_set = set(clicked) if clicked is not None else set()
    if not clicked_set.issubset(set(inview)):
        return None
    return [1 if a in clicked_set else 0 for a in inview]


def _load_behaviors(behaviors_parquet: Path, history_parquet: Path, source_split: str) -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(behaviors_parquet)
    hist = _load_history(history_parquet)

    df["user_id"] = df["user_id"].astype(str)
    df["candidates"] = df["article_ids_inview"].apply(lambda xs: [str(x) for x in xs])
    clicked_str = df["article_ids_clicked"].apply(lambda xs: [str(x) for x in xs] if xs is not None else [])

    labels = [
        _labels_for_impression(cands, clicks)
        for cands, clicks in zip(df["candidates"], clicked_str)
    ]
    df["labels"] = labels
    n_before = len(df)
    df = df[df["labels"].notna()].copy()
    n_dropped = n_before - len(df)

    df = df.merge(hist, on="user_id", how="left")
    df["history_ids"] = df["history_ids"].apply(lambda x: x if isinstance(x, list) else [])
    df["history_times"] = df["history_times"].apply(lambda x: x if isinstance(x, list) else [])
    df["history_len"] = df["history_ids"].apply(len)

    df["impression_id"] = df["impression_id"].astype(str)
    df["source_split"] = source_split
    df["dataset"] = "ebnerd"
    df = df.rename(columns={"impression_time": "timestamp"})
    # impression_id is verified globally unique across train/validation for
    # EB-NeRD (unlike MIND -- see clean_mind.py), but row_uid is kept for
    # schema parity so downstream code can join on the same column for both
    # datasets without a per-dataset special case.
    df["row_uid"] = "ebnerd:" + df["source_split"] + ":" + df["impression_id"]

    return df[[
        "impression_id", "row_uid", "dataset", "source_split", "user_id", "timestamp",
        "candidates", "labels", "history_ids", "history_times", "history_len",
        "next_read_time", "next_scroll_percentage",
    ]], n_dropped


def clean_ebnerd(raw_dir: Path, interim_dir: Path, scale: str) -> None:
    assert scale in ("demo", "small")
    zip_name = f"ebnerd_{scale}.zip"
    extract_dir = extract_clean(raw_dir / zip_name, raw_dir / f"ebnerd_{scale}")

    articles = _load_articles(extract_dir / "articles.parquet")

    frames, total_dropped = [], 0
    for source_split, folder in SOURCE_FOLDERS.items():
        split_dir = extract_dir / folder
        behaviors_df, n_dropped = _load_behaviors(
            split_dir / "behaviors.parquet", split_dir / "history.parquet", source_split
        )
        frames.append(behaviors_df)
        total_dropped += n_dropped
        print(f"ebnerd_{scale}/{folder}: {len(behaviors_df)} impressions kept, {n_dropped} dropped (clicked article not in inview list)")

    impressions = pd.concat(frames, ignore_index=True)

    print(f"ebnerd_{scale}: {len(articles)} articles, {len(impressions)} impressions total, "
          f"{total_dropped} dropped for label/candidate mismatch")

    out_dir = interim_dir / f"ebnerd_{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(out_dir / "articles.parquet", index=False)
    impressions.to_parquet(out_dir / "impressions.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--scale", choices=["demo", "small"], default=None,
                         help="overrides config.ebnerd.scale")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    scale = args.scale or cfg["ebnerd"]["scale"]
    clean_ebnerd(resolve_path(cfg, "raw_dir"), resolve_path(cfg, "interim_dir"), scale)


if __name__ == "__main__":
    main()
