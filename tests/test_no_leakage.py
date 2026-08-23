"""Q9.2 -- Enforce the behaviour-window boundary. No future-click leakage.

Reads data/splits/<dataset>/split_manifest.json (written by src/pipeline/split.py)
rather than re-scanning raw parquet files row by row, so this runs in
milliseconds regardless of dataset size -- see SPEC.md Q9.2 for why the v1
draft's `iterrows`-based version was unusable at scale.

Requires `make data` (or build_pipeline.py) to have been run first.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import REPO_ROOT, SPLITS, load_config, resolve_path

DATASETS = ["mind", "ebnerd"]


def _manifest(dataset: str) -> dict:
    cfg = load_config()
    path = resolve_path(cfg, "splits_dir") / dataset / "split_manifest.json"
    if not path.exists():
        pytest.skip(f"{path} not found -- run `make data` first")
    return json.loads(path.read_text())


@pytest.mark.parametrize("dataset,split", SPLITS)
def test_history_precedes_impressions(dataset, split):
    """Every click in a user's history must predate that split's impression window."""
    m = _manifest(dataset)[split]
    if m["max_history_click_time"] is None:
        pytest.skip(f"{dataset} has no per-click history timestamps (expected for MIND)")
    assert m["max_history_click_time"] <= m["min_impression_time"], (
        f"{dataset}/{split}: history extends past the behaviour-window boundary "
        f"({m['max_history_click_time']} > {m['min_impression_time']})"
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_splits_are_temporally_disjoint(dataset):
    m = _manifest(dataset)
    assert m["train"]["max_impression_time"] <= m["val"]["min_impression_time"], (
        f"{dataset}: train/val overlap in time"
    )
    assert m["val"]["max_impression_time"] <= m["test"]["min_impression_time"], (
        f"{dataset}: val/test overlap in time"
    )


def test_raw_future_publish_noise_is_within_known_tolerance():
    """EB-NeRD only: quantify (not eliminate) raw candidate/publish-time noise.

    IMPORTANT SCOPING NOTE: `article_ids_inview` is Ekstra Bladet's own served
    impression, not something our pipeline generates -- so this test cannot
    "fix" the underlying data, only detect if it drifts. Measured on the demo
    split (2026-08-20): 70/278139 train rows, 3/160748 val, 3/144167 test
    (~0.01-0.03%), with published_time trailing the impression by seconds to
    ~9 hours -- almost certainly published_time being backdated/edited after
    the article was already being served, not a sign our loaders mis-joined
    timestamps. TOLERANCE_PCT is deliberately generous; a jump well past it
    would indicate a real join bug, not dataset noise, and should fail loudly.

    This test does NOT stand in for actual candidate-generation eligibility
    filtering -- that happens at retrieval time in Q2/Q3, which must filter
    `published_time <= impression_timestamp` on OUR retrieved candidates
    (full catalog, not just the pre-filtered inview list) before recall@K is
    computed. See SPEC.md Q2.0 "Eligibility filtering is mandatory".
    """
    TOLERANCE_PCT = 0.5  # generous; see measured ~0.01-0.03% above

    cfg = load_config()
    splits_dir = resolve_path(cfg, "splits_dir")
    feature_store_dir = resolve_path(cfg, "feature_store_dir")

    articles_path = feature_store_dir / "ebnerd" / "articles.parquet"
    if not articles_path.exists():
        pytest.skip(f"{articles_path} not found -- run `make data` first")

    articles = pd.read_parquet(articles_path, columns=["article_id", "published_time"])
    pub_time = dict(zip(articles["article_id"], articles["published_time"]))

    for split in ("train", "val", "test"):
        imp_path = splits_dir / "ebnerd" / split / "impressions.parquet"
        if not imp_path.exists():
            pytest.skip(f"{imp_path} not found -- run `make data` first")
        imp = pd.read_parquet(imp_path, columns=["timestamp", "candidates"])

        rows = imp.explode("candidates").rename(columns={"candidates": "article_id"})
        rows["published_time"] = rows["article_id"].map(pub_time)
        violations = rows[
            rows["published_time"].notna() & (rows["published_time"] > rows["timestamp"])
        ]
        pct = 100 * len(violations) / len(rows) if len(rows) else 0.0
        assert pct <= TOLERANCE_PCT, (
            f"ebnerd/{split}: {len(violations)} candidate(s) ({pct:.3f}%) published after their "
            f"impression's timestamp -- exceeds the {TOLERANCE_PCT}% known-noise tolerance, "
            f"investigate the timestamp join rather than raising the tolerance"
        )


def test_popularity_prior_uses_train_only():
    """Novelty/popularity stats (train_click_count) must be derived from TRAIN only.

    Recomputes train_click_count independently from data/splits/.../train and
    asserts it matches the feature store exactly; then recomputes including val
    and asserts the two differ (unless click counts happen to tie, in which case
    this specific assertion is skipped for that dataset -- see note below) as a
    regression guard against silently including val/test clicks.
    """
    cfg = load_config()
    splits_dir = resolve_path(cfg, "splits_dir")
    feature_store_dir = resolve_path(cfg, "feature_store_dir")

    for dataset in DATASETS:
        articles_path = feature_store_dir / dataset / "articles.parquet"
        train_path = splits_dir / dataset / "train" / "impressions.parquet"
        val_path = splits_dir / dataset / "val" / "impressions.parquet"
        if not (articles_path.exists() and train_path.exists() and val_path.exists()):
            pytest.skip(f"{dataset} feature store / splits not found -- run `make data` first")

        def click_counts(imp: pd.DataFrame) -> dict[str, int]:
            counts: dict[str, int] = {}
            for cands, labels in zip(imp["candidates"], imp["labels"]):
                for article_id, label in zip(cands, labels):
                    if label == 1:
                        counts[article_id] = counts.get(article_id, 0) + 1
            return counts

        train_only = click_counts(pd.read_parquet(train_path, columns=["candidates", "labels"]))
        stored = pd.read_parquet(articles_path, columns=["article_id", "train_click_count"])
        stored_map = dict(zip(stored["article_id"], stored["train_click_count"]))

        mismatches = [
            (a, c, stored_map.get(a, 0)) for a, c in train_only.items() if stored_map.get(a, 0) != c
        ]
        assert not mismatches, (
            f"{dataset}: feature_store train_click_count disagrees with a train-only "
            f"recomputation for {len(mismatches)} article(s), e.g. {mismatches[:3]}"
        )

        train_plus_val = click_counts(
            pd.concat([
                pd.read_parquet(train_path, columns=["candidates", "labels"]),
                pd.read_parquet(val_path, columns=["candidates", "labels"]),
            ])
        )
        if train_plus_val != train_only:
            differing = {a for a in train_plus_val if train_plus_val[a] != train_only.get(a, 0)}
            assert stored_map.get(next(iter(differing)), 0) != train_plus_val[next(iter(differing))], (
                f"{dataset}: feature_store train_click_count matches a train+val count for at least "
                f"one article -- val clicks may have leaked into the 'train-only' popularity prior"
            )
