"""Sanity checks on the temporal split beyond the leakage boundary itself:
row_uid uniqueness (both within and across splits), and every split non-empty.
Complements test_no_leakage.py's timestamp-ordering checks.

NOTE on impression_id vs row_uid: raw `impression_id` is NOT a safe global key
for MIND -- MINDsmall_train.tsv and MINDsmall_dev.tsv each number impressions
independently starting from 1 (verified: 156965/156965 unique within
train_raw, 73152/73152 within dev_raw, but train_raw and dev_raw overlap with
each other; e.g. impression_id "1" exists in both files). That's expected raw
-dataset behaviour, not a pipeline bug -- see clean_mind.py's row_uid comment.
row_uid ("<dataset>:<source_split>:<impression_id>") is globally unique by
construction and is what this test actually checks; impression_id is only
required to be unique WITHIN its own split (needed so per-split code, e.g.
BM25/eval indexing, can safely use it as a row key).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config, resolve_path

DATASETS = ["mind", "ebnerd"]
SPLIT_NAMES = ["train", "val", "test"]


def _load_split(splits_dir, dataset, split):
    path = splits_dir / dataset / split / "impressions.parquet"
    if not path.exists():
        pytest.skip(f"{path} not found -- run `make data` first")
    return pd.read_parquet(path, columns=["impression_id", "row_uid"])


@pytest.mark.parametrize("dataset", DATASETS)
def test_impression_id_unique_within_each_split(dataset):
    cfg = load_config()
    splits_dir = resolve_path(cfg, "splits_dir")
    for split in SPLIT_NAMES:
        df = _load_split(splits_dir, dataset, split)
        assert df["impression_id"].is_unique, (
            f"{dataset}/{split}: impression_id is not unique within this split "
            f"({len(df)} rows, {df['impression_id'].nunique()} unique)"
        )


@pytest.mark.parametrize("dataset", DATASETS)
def test_row_uid_globally_unique_across_splits(dataset):
    cfg = load_config()
    splits_dir = resolve_path(cfg, "splits_dir")
    uids_by_split = {s: set(_load_split(splits_dir, dataset, s)["row_uid"]) for s in SPLIT_NAMES}

    for a, b in [("train", "val"), ("val", "test"), ("train", "test")]:
        overlap = uids_by_split[a] & uids_by_split[b]
        assert not overlap, f"{dataset}: {len(overlap)} row_uid(s) appear in both {a} and {b}"


@pytest.mark.parametrize("dataset,split", [(d, s) for d in DATASETS for s in SPLIT_NAMES])
def test_split_is_non_empty(dataset, split):
    cfg = load_config()
    splits_dir = resolve_path(cfg, "splits_dir")
    df = _load_split(splits_dir, dataset, split)
    assert len(df) > 0, f"{dataset}/{split} is empty -- check split boundaries in config/default.yaml"
