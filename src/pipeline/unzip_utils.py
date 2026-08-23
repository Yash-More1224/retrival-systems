"""Shared zip-extraction helper.

Every EB-NeRD zip (and MIND's, to a lesser extent) carries __MACOSX/ AppleDouble
junk. Extracting naively pollutes data/raw/ with unusable ._* files.

Note the two datasets' zips are laid out differently:
  - MINDsmall_{train,dev}.zip already contain a wrapping folder internally
    (e.g. "MINDsmall_train/news.tsv"), so extract_root should be raw_dir itself.
  - ebnerd_{demo,small}.zip have NO wrapping folder (members are
    "articles.parquet", "train/behaviors.parquet", ...), so extract_root
    should be a dataset-specific subdirectory, e.g. raw_dir / "ebnerd_demo".
That's why extract_clean takes `expect_dir` separately from `extract_root`:
idempotency is checked against the directory the CALLER expects the content
to end up in, not against extract_root (which for MIND is raw_dir itself and
is never empty).
"""
from __future__ import annotations

import zipfile
from pathlib import Path


def extract_clean(zip_path: Path, extract_root: Path, expect_dir: Path | None = None, force: bool = False) -> Path:
    """Extract zip_path into extract_root, skipping __MACOSX/ and AppleDouble (._*) entries.

    expect_dir: the directory the extracted content should end up under, used
    only to decide whether extraction can be skipped (already done). Defaults
    to extract_root if not given.

    Returns expect_dir (or extract_root if expect_dir is None).
    """
    check_dir = expect_dir if expect_dir is not None else extract_root
    if check_dir.exists() and any(check_dir.iterdir()) and not force:
        return check_dir

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if "__MACOSX" in member or name.startswith("._"):
                continue
            zf.extract(member, extract_root)
    return check_dir
