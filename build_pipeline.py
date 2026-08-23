#!/usr/bin/env python
"""Q1.5 -- One-command rebuild: raw files -> ready-to-use feature store.

    python build_pipeline.py --datasets mind ebnerd --config config/default.yaml

Runs, in order: download -> clean -> split -> feature_store, for each
requested dataset. Every step is idempotent (see each module's docstring),
so re-running this is always safe and is how `make data` reproduces the
pipeline end to end.
"""
from __future__ import annotations

import argparse

from src.config import load_config, resolve_path, seed_everything
from src.pipeline import clean_ebnerd, clean_mind, download, feature_store, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--ebnerd-scale", choices=["demo", "small"], default=None)
    parser.add_argument("--include-testset", action="store_true",
                         help="also download ebnerd_testset.zip (1.5GB, needed only for Q5)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])

    raw_dir = resolve_path(cfg, "raw_dir")
    interim_dir = resolve_path(cfg, "interim_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    ebnerd_scale = args.ebnerd_scale or cfg["ebnerd"]["scale"]

    print("=== [1/4] download ===")
    files = {}
    if "mind" in args.datasets:
        files.update(download.MIND_FILES)
    if "ebnerd" in args.datasets:
        files.update(download.EBNERD_FILES)
        if args.include_testset:
            files.update(download.EBNERD_TESTSET_FILES)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "MANIFEST.json"
    import json
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in files.items():
        download.ensure_file(name, url, raw_dir, manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print("\n=== [2/4] clean ===")
    if "mind" in args.datasets:
        clean_mind.clean_mind(raw_dir, interim_dir)
    if "ebnerd" in args.datasets:
        clean_ebnerd.clean_ebnerd(raw_dir, interim_dir, ebnerd_scale)

    print("\n=== [3/4] split ===")
    if "mind" in args.datasets:
        split.split_mind(interim_dir, splits_dir, cfg["mind_split"]["val_days_from_train_tail"])
    if "ebnerd" in args.datasets:
        split.split_ebnerd(interim_dir, splits_dir, ebnerd_scale, cfg["ebnerd_split_boundaries"])

    print("\n=== [4/4] feature_store ===")
    config_hash = feature_store._config_hash(cfg)
    if "mind" in args.datasets:
        feature_store.build_article_features(interim_dir, splits_dir, feature_store_dir, "mind", "mind", config_hash)
        feature_store.build_user_features(splits_dir, feature_store_dir, "mind", config_hash)
    if "ebnerd" in args.datasets:
        feature_store.build_article_features(
            interim_dir, splits_dir, feature_store_dir, "ebnerd", f"ebnerd_{ebnerd_scale}", config_hash
        )
        feature_store.build_user_features(splits_dir, feature_store_dir, "ebnerd", config_hash)

    print("\nDone. Feature store ready at", feature_store_dir)


if __name__ == "__main__":
    main()
