"""Q1.3 -- Temporal train/val/test split. Never random (see SPEC.md Q1.3).

MIND-small: MINDsmall_train and MINDsmall_dev are already time-disjoint.
  train = MINDsmall_train minus its last `val_days_from_train_tail` day(s)
  val   = that last day of MINDsmall_train
  test  = all of MINDsmall_dev

EB-NeRD: the provided train/ and validation/ folders are already a clean
temporal partition with matched history windows (see SPEC.md "Verified
dataset facts"). val and test are carved out of the validation/ folder by
the fixed boundaries in config/default.yaml -- both inherit that folder's
own history.parquet (joined in clean_ebnerd.py), whose click window already
ends before validation/ impressions begin, so no leakage is introduced by
splitting inside that window.

Writes:
  data/splits/<dataset>/<split>/impressions.parquet
  data/splits/<dataset>/split_manifest.json

The manifest is what tests/test_no_leakage.py and tests/test_split_disjoint.py
check against, and what the design note's split-description table is
generated from -- see SPEC.md Q1.3 and Q9.2.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path


def _flatten_history_times(series: pd.Series) -> list:
    """Flatten the per-row history_times lists into one list of timestamps.

    NOTE: parquet list<> columns round-trip through pandas as numpy.ndarray,
    not native list -- `isinstance(x, list)` silently drops every row after a
    parquet read/write cycle (this bit us once: EB-NeRD's leakage test was
    passing vacuously because every history_times entry looked like "None" to
    an isinstance(list) check). Use a duck-typed length/iteration check instead.
    """
    out = []
    for lst in series:
        if lst is None:
            continue
        try:
            length = len(lst)
        except TypeError:
            continue
        if length == 0:
            continue
        out.extend(t for t in lst if t is not None)
    return out


def _describe_split(df: pd.DataFrame) -> dict:
    # history_times entries are numpy.datetime64 (parquet list<timestamp> round-trip
    # -- see _flatten_history_times' docstring), which has no .isoformat(); wrap in
    # pd.Timestamp first.
    hist_times = _flatten_history_times(df["history_times"])
    return {
        "n_rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "min_impression_time": df["timestamp"].min().isoformat() if len(df) else None,
        "max_impression_time": df["timestamp"].max().isoformat() if len(df) else None,
        "min_history_click_time": pd.Timestamp(min(hist_times)).isoformat() if hist_times else None,
        "max_history_click_time": pd.Timestamp(max(hist_times)).isoformat() if hist_times else None,
    }


def split_mind(interim_dir: Path, splits_dir: Path, val_days: int) -> dict:
    imp = pd.read_parquet(interim_dir / "mind" / "impressions.parquet")

    train_raw = imp[imp["source_split"] == "train_raw"]
    dev_raw = imp[imp["source_split"] == "dev_raw"]

    val_cutoff = train_raw["timestamp"].max() - timedelta(days=val_days)
    train = train_raw[train_raw["timestamp"] <= val_cutoff]
    val = train_raw[train_raw["timestamp"] > val_cutoff]
    test = dev_raw

    return _write_splits("mind", {"train": train, "val": val, "test": test}, splits_dir)


def split_ebnerd(interim_dir: Path, splits_dir: Path, scale: str, boundaries: dict) -> dict:
    imp = pd.read_parquet(interim_dir / f"ebnerd_{scale}" / "impressions.parquet")

    def between(lo, hi):
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        return imp[(imp["timestamp"] >= lo) & (imp["timestamp"] < hi)]

    train = between(boundaries["train_start"], boundaries["train_end"])
    val = between(boundaries["val_start"], boundaries["val_end"])
    test = between(boundaries["test_start"], boundaries["test_end"])

    return _write_splits("ebnerd", {"train": train, "val": val, "test": test}, splits_dir)


def _write_splits(dataset: str, splits: dict[str, pd.DataFrame], splits_dir: Path) -> dict:
    manifest = {}
    out_dir = splits_dir / dataset
    for name, df in splits.items():
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(split_dir / "impressions.parquet", index=False)
        manifest[name] = _describe_split(df)
        print(f"{dataset}/{name}: {manifest[name]['n_rows']} impressions, "
              f"{manifest[name]['n_users']} users, "
              f"{manifest[name]['min_impression_time']} -> {manifest[name]['max_impression_time']}")

    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--ebnerd-scale", choices=["demo", "small"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    interim_dir = resolve_path(cfg, "interim_dir")
    splits_dir = resolve_path(cfg, "splits_dir")

    if "mind" in args.datasets:
        split_mind(interim_dir, splits_dir, cfg["mind_split"]["val_days_from_train_tail"])
    if "ebnerd" in args.datasets:
        scale = args.ebnerd_scale or cfg["ebnerd"]["scale"]
        split_ebnerd(interim_dir, splits_dir, scale, cfg["ebnerd_split_boundaries"])


if __name__ == "__main__":
    main()
