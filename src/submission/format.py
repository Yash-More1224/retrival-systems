"""Q5 -- Shared Codabench submission format: both MIND and RecSys2024/EB-NeRD use
the same line format (verified against the official guidelines pages and the
downloaded MIND sample, 2026-08-20 -- see SPEC.md Q5):

    impression_id [rank_1,rank_2,...,rank_n]

Ranks are continuous integers 1..n aligned to the ORIGINAL candidate order
(MIND: the impressions field's order; EB-NeRD: article_ids_inview's order),
1 = most likely clicked, comma-separated with NO spaces inside the brackets.
The zip must contain exactly one file at its root -- no parent folder, no
__MACOSX/.

Both competitions also require every row to appear in original file order
and every impression_id in the test file to be present exactly once -- see
validate_submission, which checks this BEFORE zipping so a malformed
submission is caught locally rather than burning a rate-limited attempt
(MIND: 1/day; EB-NeRD: 5/day -- see SPEC.md Q5).
"""
from __future__ import annotations

import zipfile
from pathlib import Path


def scores_to_ranks(scores: list[float]) -> list[int]:
    """Higher score -> rank 1. Output is aligned to the INPUT order (not sorted) --
    ranks[i] is the rank of scores[i]. Deterministic tie-break by original position,
    consistent with src.retrieval.bm25.top_k_indices' tie-break convention."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, 1):
        ranks[idx] = rank
    return ranks


def format_line(impression_id: str, ranks: list[int]) -> str:
    return f"{impression_id} [{','.join(str(r) for r in ranks)}]"


def write_submission_file(rows: list[tuple[str, list[int]]], path: Path) -> None:
    """rows: list of (impression_id, ranks), in the EXACT order they must appear
    (original test-file order -- callers must not sort/shuffle before calling this)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for impression_id, ranks in rows:
            f.write(format_line(impression_id, ranks) + "\n")


def validate_submission(rows: list[tuple[str, list[int]]], expected_ids_in_order: list[str]) -> None:
    """Raises AssertionError with a specific message on the first violation found.
    Call this BEFORE zipping -- see module docstring."""
    actual_ids = [r[0] for r in rows]
    assert actual_ids == list(expected_ids_in_order), (
        f"Row order/coverage mismatch: {len(actual_ids)} rows produced, "
        f"{len(expected_ids_in_order)} expected. First mismatch at index "
        f"{next((i for i, (a, e) in enumerate(zip(actual_ids, expected_ids_in_order)) if a != e), 'length differs')}."
    )
    for impression_id, ranks in rows:
        n = len(ranks)
        assert sorted(ranks) == list(range(1, n + 1)), (
            f"impression {impression_id}: ranks {ranks} are not a valid permutation of 1..{n}"
        )


def write_submission_streaming(row_iter, path: Path, expected_total: int | None = None) -> int:
    """Streams (impression_id, ranks) pairs straight to disk instead of via
    write_submission_file + validate_submission, which both require the full
    `rows` list in memory -- fine for MIND's ~73K impressions, but EB-NeRD's
    real ebnerd_testset has 13.5M, and materializing that many (impression_id,
    ranks) tuples as Python objects OOM-killed this pipeline (2026-08-23, see
    src/submission/ebnerd.py). Row order/coverage is guaranteed by construction
    here (the caller must stream its source file in its own on-disk order,
    never resorting -- that IS "original file order", so there's no separate
    expected-order list to check against); each row's ranks are still checked
    as a valid permutation as it's written, so a bad row fails fast rather
    than silently producing a malformed submission. Returns the row count so
    the caller can assert full coverage against expected_total (e.g. the test
    file's row count from parquet metadata).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for impression_id, ranks in row_iter:
            m = len(ranks)
            assert sorted(ranks) == list(range(1, m + 1)), (
                f"impression {impression_id}: ranks {ranks} are not a valid permutation of 1..{m}"
            )
            f.write(format_line(impression_id, ranks) + "\n")
            n += 1
    if expected_total is not None:
        assert n == expected_total, f"wrote {n} rows but expected {expected_total} (test file row count)"
    return n


def zip_submission(txt_path: Path, zip_path: Path) -> None:
    """Zips txt_path alone at the archive root (no parent folder, no __MACOSX/),
    as required by both competitions' guidelines."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_path.name)
