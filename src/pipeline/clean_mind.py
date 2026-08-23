"""Q1.2 -- Parse raw MIND-small files into the unified schema.

Reads MINDsmall_train.zip and MINDsmall_dev.zip (already downloaded by
download.py), extracts them, and writes two interim tables:

  data/interim/mind/articles.parquet    -- one row per news_id
  data/interim/mind/impressions.parquet -- one row per impression, with the
                                            user's click history embedded

`source_split` ("train_raw" | "dev_raw") is kept on every impression row so
that split.py can implement SPEC.md's Q1.3 policy: train/val carved out of
MINDsmall_train by time, test = MINDsmall_dev in full.

Column layout (see SPEC.md "Unified Schema"):
  articles:    article_id, dataset, title, abstract, body, category,
               subcategory, entities, entity_wikidata_ids, published_time,
               total_inviews, total_pageviews, total_read_time, sentiment_score  (all null for MIND)
  impressions: impression_id, row_uid, dataset, source_split, user_id, timestamp,
               candidates, labels, history_ids, history_times, history_len

`entity_wikidata_ids` (e.g. "Q41") is kept separately from the human-readable
`entities` labels because Q3 (semantic retrieval) uses it to look up MIND's
provided entity_embedding.vec, which is keyed by WikidataId -- this is how
MIND's "provided" article embeddings are built without training/running our
own model, mirroring EB-NeRD's provided word2vec document vectors (see
SPEC.md Q3.1 and src/retrieval/build_embeddings.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.pipeline.unzip_utils import extract_clean

NEWS_COLS = [
    "article_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
BEHAVIORS_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # e.g. "11/11/2019 9:05:58 AM"


def _extract_entity_field(raw: str, field: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    values = []
    for item in items:
        v = item.get(field)
        if v:
            values.append(v)
    return values


def _load_news(news_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(news_tsv, sep="\t", header=None, names=NEWS_COLS, dtype=str)
    df["entities"] = (
        df["title_entities"].apply(lambda r: _extract_entity_field(r, "Label"))
        + df["abstract_entities"].apply(lambda r: _extract_entity_field(r, "Label"))
    ).apply(lambda es: sorted(set(es)))
    # WikidataId (e.g. "Q41"), kept separately from the human-readable `entities` labels
    # above because Q3 needs it to look up MIND's provided entity_embedding.vec, which is
    # keyed by WikidataId, not by label. Order/duplicates preserved (not deduped) since
    # mean-pooling downstream can legitimately weight a repeated entity mention more.
    df["entity_wikidata_ids"] = (
        df["title_entities"].apply(lambda r: _extract_entity_field(r, "WikidataId"))
        + df["abstract_entities"].apply(lambda r: _extract_entity_field(r, "WikidataId"))
    )
    df["abstract"] = df["abstract"].fillna("")
    df["title"] = df["title"].fillna("")
    return df[["article_id", "category", "subcategory", "title", "abstract", "entities", "entity_wikidata_ids"]]


def _load_behaviors(behaviors_tsv: Path, source_split: str) -> pd.DataFrame:
    df = pd.read_csv(behaviors_tsv, sep="\t", header=None, names=BEHAVIORS_COLS, dtype=str)
    df["timestamp"] = pd.to_datetime(df["time"], format=TIME_FORMAT)
    df["history_ids"] = df["history"].apply(
        lambda h: h.split() if isinstance(h, str) and h.strip() else []
    )
    df["history_len"] = df["history_ids"].apply(len)
    df["history_times"] = [None] * len(df)  # MIND has no per-click timestamps

    cand_ids, labels = [], []
    for imp in df["impressions"]:
        pairs = [p.rsplit("-", 1) for p in imp.split()]
        cand_ids.append([p[0] for p in pairs])
        labels.append([int(p[1]) for p in pairs])
    df["candidates"] = cand_ids
    df["labels"] = labels
    df["source_split"] = source_split
    df["dataset"] = "mind"
    # impression_id resets to 1 independently in MINDsmall_train.tsv and
    # MINDsmall_dev.tsv -- it is NOT globally unique across the two raw files
    # (verified: 156965/156965 unique within train_raw, 73152/73152 within
    # dev_raw, but they overlap with each other). row_uid is the safe key for
    # any future cross-split join/dedup; impression_id itself is kept as-is
    # because Q5 submissions must reference the exact original id.
    df["row_uid"] = "mind:" + df["source_split"] + ":" + df["impression_id"]

    return df[[
        "impression_id", "row_uid", "dataset", "source_split", "user_id", "timestamp",
        "candidates", "labels", "history_ids", "history_times", "history_len",
    ]]


def clean_mind(raw_dir: Path, interim_dir: Path) -> None:
    # Unlike the EB-NeRD zips, MINDsmall_{train,dev}.zip already contain a
    # wrapping MINDsmall_{train,dev}/ folder internally, so we extract
    # directly into raw_dir rather than into a same-named subdirectory
    # (which would otherwise double-nest the folder).
    train_dir = extract_clean(raw_dir / "MINDsmall_train.zip", raw_dir, raw_dir / "MINDsmall_train")
    dev_dir = extract_clean(raw_dir / "MINDsmall_dev.zip", raw_dir, raw_dir / "MINDsmall_dev")

    news = pd.concat(
        [_load_news(train_dir / "news.tsv"), _load_news(dev_dir / "news.tsv")],
        ignore_index=True,
    ).drop_duplicates(subset="article_id", keep="first")

    articles = pd.DataFrame({
        "article_id": news["article_id"],
        "dataset": "mind",
        "title": news["title"],
        "abstract": news["abstract"],
        "body": "",
        "category": news["category"],
        "subcategory": news["subcategory"],
        "entities": news["entities"],
        "entity_wikidata_ids": news["entity_wikidata_ids"],
        "published_time": pd.NaT,
        "total_inviews": pd.NA,
        "total_pageviews": pd.NA,
        "total_read_time": pd.NA,
        "sentiment_score": pd.NA,
    })

    behaviors = pd.concat(
        [
            _load_behaviors(train_dir / "behaviors.tsv", "train_raw"),
            _load_behaviors(dev_dir / "behaviors.tsv", "dev_raw"),
        ],
        ignore_index=True,
    )

    n_missing_article = articles["title"].eq("").sum()
    print(f"mind: {len(articles)} articles ({n_missing_article} with empty title), "
          f"{len(behaviors)} impressions "
          f"({(behaviors['source_split'] == 'train_raw').sum()} train_raw, "
          f"{(behaviors['source_split'] == 'dev_raw').sum()} dev_raw)")

    out_dir = interim_dir / "mind"
    out_dir.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(out_dir / "articles.parquet", index=False)
    behaviors.to_parquet(out_dir / "impressions.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    clean_mind(resolve_path(cfg, "raw_dir"), resolve_path(cfg, "interim_dir"))


if __name__ == "__main__":
    main()
