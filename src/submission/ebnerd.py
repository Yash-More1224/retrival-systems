"""Q5 -- Generate the EB-NeRD/RecSys2024 Codabench submission.

`ebnerd_testset.zip` (1.5GB, URL confirmed live -- see SPEC.md Q1.1) is a SEPARATE,
unlabeled dataset -- not demo/small's train/validation folders. Its directory layout was
confirmed remotely via HTTP range requests on the zip's (uncompressed) central directory,
WITHOUT downloading the whole archive (2026-08-21):

    ebnerd_testset/articles.parquet          (~150MB)
    ebnerd_testset/test/history.parquet      (~1.15GB uncompressed)
    ebnerd_testset/test/behaviors.parquet    (~568MB uncompressed)

The actual COLUMN SCHEMAS were verified directly once the file finished downloading
locally (2026-08-23, via pyarrow.parquet.ParquetFile(path).schema_arrow -- NOT
.schema, which exposes Parquet's flattened nested-list leaf names instead of the
top-level columns pandas sees). All previously-assumed columns are present. Real
scale, also only knowable once downloaded: articles=125,541 rows, history=807,677
users, **behaviors=13,536,710 impressions** -- ~185x MIND's ~73K and far larger than
this module was originally written assuming. Two real bugs followed directly from
that scale, both fixed here:

  1. OOM: the original version read history.parquet/behaviors.parquet whole into
     pandas (unrestricted columns) and built the full `imp` DataFrame before any
     batching -- killed by the OOM killer at ~10.5GB resident on a 15GB machine.
     Fixed by (a) reading only the columns actually used (pyarrow column pruning,
     avoids ever materializing e.g. history's 3 unused list-columns or behaviors'
     8 unused scalar columns) and (b) streaming behaviors.parquet in row-batches
     via pyarrow's iter_batches instead of loading all 13.5M rows at once.
  2. Runtime: src.retrieval.scoring's score_bm25_batch/score_semantic_batch build a
     full {article_id: score} dict over the ENTIRE catalog for every row -- fine at
     MIND's ~30K articles / 73K impressions, but at 125K articles / 13.5M impressions
     that's an estimated 70+ hours of pure dict construction, separate from the
     actual scoring math. Fixed by using score_*_batch_candidates instead, which
     scores the same way but only extracts each row's own ~10-30 candidates via a
     precomputed article_id -> column-index map (see src/retrieval/scoring.py).

Submission rows are streamed straight to disk (write_submission_streaming) instead
of being accumulated as a 13.5M-element Python list -- see its docstring in
src/submission/format.py. Row order is exactly behaviors.parquet's own on-disk
order (the required "original file order"), since nothing here ever re-sorts it.

The test catalog is almost certainly disjoint from demo/small's article set (different,
presumably later time window) -- so this builds a FRESH BM25 index over the test
articles.parquet rather than reusing feature_store/ebnerd/bm25 (which only covers
demo/small). Semantic scoring reuses the provided word2vec vectors (already downloaded;
its document_vector.parquet has 125,541 rows -- much larger than demo's 11,777 -- so it
likely already covers the wider corpus including test articles; coverage is checked and
reported, not assumed).

Usage:
    python -m src.submission.ebnerd --method bm25
    python -m src.submission.ebnerd --method semantic

Writes:
    predictions/ebnerd_predictions.txt
    predictions/ebnerd_predictions.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import load_config, resolve_path, seed_everything
from src.pipeline.download import EBNERD_TESTSET_FILES, ensure_file
from src.pipeline.unzip_utils import extract_clean
from src.retrieval.bm25 import BM25Index
from src.retrieval.scoring import best_n, score_bm25_batch_candidates, score_semantic_batch_candidates
from src.retrieval.semantic import SemanticIndex
from src.submission.format import scores_to_ranks, write_submission_streaming, zip_submission

BATCH_SIZE = 256
PRINT_EVERY_BATCHES = 500  # ~128K impressions/print at BATCH_SIZE=256 -- 13.5M rows would be too noisy at every batch

ARTICLES_REQUIRED_COLS = ["article_id", "title"]
HISTORY_REQUIRED_COLS = ["user_id", "article_id_fixed"]
BEHAVIORS_REQUIRED_COLS = ["impression_id", "user_id", "impression_time", "article_ids_inview"]


def _assert_columns_names(available: list[str], required: list[str], name: str) -> None:
    missing = [c for c in required if c not in available]
    assert not missing, (
        f"{name}: expected columns {required} but missing {missing} -- actual columns: "
        f"{available}. ebnerd_testset's real schema differs from the assumption documented "
        f"in this module's docstring; inspect and fix before proceeding (do NOT guess-map columns)."
    )


def _ensure_testset(raw_dir) -> Path:
    manifest_path = raw_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    name, url = next(iter(EBNERD_TESTSET_FILES.items()))
    ensure_file(name, url, raw_dir, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Unlike ebnerd_demo.zip/ebnerd_small.zip, ebnerd_testset.zip's members ARE already
    # wrapped in an "ebnerd_testset/" folder (verified via a remote zip-directory listing,
    # 2026-08-21 -- same pattern as the MIND zips, which is what caused the double-nesting
    # bug fixed in clean_mind.py). Extract into raw_dir itself, not raw_dir/"ebnerd_testset",
    # or this doubles up into raw_dir/ebnerd_testset/ebnerd_testset/...
    return extract_clean(raw_dir / name, raw_dir, raw_dir / "ebnerd_testset")


def _load_articles(extract_dir: Path) -> pd.DataFrame:
    """Keeps article_id in its native int32 form (NOT .astype(str)) -- see
    module docstring's memory-blowup finding: at this scale (116.8M history
    references alone), converting IDs to Python str objects costs ~55 bytes
    each vs ~4 bytes for a raw numpy int32, a >10x blowup. Every ID-typed
    column in this module (article_id, user_id, impression_id) stays native
    throughout; the final submission line still renders correctly since
    Python's f-string formatting stringifies ints/numpy scalars on its own
    (see format.format_line) -- no .astype(str) anywhere is actually needed.
    """
    path = extract_dir / "articles.parquet"
    available = pq.ParquetFile(path).schema_arrow.names
    _assert_columns_names(available, ARTICLES_REQUIRED_COLS, "ebnerd_testset/articles.parquet")
    cols = [c for c in ARTICLES_REQUIRED_COLS + ["subtitle"] if c in available]
    articles = pd.read_parquet(path, columns=cols)
    if "subtitle" not in articles.columns:
        articles["subtitle"] = ""
    return articles


def _load_history_by_user(extract_dir: Path) -> dict[int, np.ndarray]:
    path = extract_dir / "test" / "history.parquet"
    available = pq.ParquetFile(path).schema_arrow.names
    _assert_columns_names(available, HISTORY_REQUIRED_COLS, "ebnerd_testset/test/history.parquet")
    history = pd.read_parquet(path, columns=HISTORY_REQUIRED_COLS)  # prunes 3 unused list-columns -- see module docstring
    # article_id_fixed cells are already compact numpy int32 arrays (pyarrow-backed) --
    # keep them as-is rather than rebuilding as Python lists (see _load_articles docstring).
    empty = np.array([], dtype=np.int32)
    return {
        uid: (xs if xs is not None else empty)
        for uid, xs in zip(history["user_id"], history["article_id_fixed"])
    }


def _iter_behavior_batches(extract_dir: Path, history_by_user: dict[int, np.ndarray], batch_size: int):
    """Yields (batch_df, total_rows) pairs, streaming behaviors.parquet instead of
    materializing all 13.5M test impressions at once -- see module docstring."""
    path = extract_dir / "test" / "behaviors.parquet"
    pf = pq.ParquetFile(path)
    available = pf.schema_arrow.names
    _assert_columns_names(available, BEHAVIORS_REQUIRED_COLS, "ebnerd_testset/test/behaviors.parquet")
    total_rows = pf.metadata.num_rows
    empty = np.array([], dtype=np.int32)
    for record_batch in pf.iter_batches(batch_size=batch_size, columns=BEHAVIORS_REQUIRED_COLS):
        batch = record_batch.to_pandas()
        batch["candidates"] = batch["article_ids_inview"]
        batch["history_ids"] = batch["user_id"].map(lambda u: history_by_user.get(u, empty))
        yield batch, total_rows


def generate(cfg: dict, method: str) -> None:
    raw_dir = resolve_path(cfg, "raw_dir")
    results_dir = resolve_path(cfg, "results_dir")
    predictions_dir = resolve_path(cfg, "predictions_dir")

    print("ebnerd: ensuring ebnerd_testset.zip is downloaded (1.5GB -- this may take a while)...")
    extract_dir = _ensure_testset(raw_dir)
    articles = _load_articles(extract_dir)
    history_by_user = _load_history_by_user(extract_dir)
    print(f"ebnerd: loaded {len(articles)} test articles, {len(history_by_user)} users with history")

    train_click_count: dict[int, float] = {}  # no train-split popularity crosses into the disjoint test catalog
    if method == "bm25":
        print("ebnerd: building a FRESH BM25 index over the test catalog (not reusing demo/small's)...")
        texts = (articles["title"].fillna("") + " " + articles["subtitle"].fillna("")).tolist()
        index = BM25Index.build(articles["article_id"].tolist(), texts, lang="danish")
        n = best_n(results_dir, "ebnerd", "bm25", cfg["retrieval"]["bm25"]["query_history_n"])
        article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))
        embedding_lookup = dim = None
    else:
        word2vec_path = raw_dir / "Ekstra_Bladet_word2vec" / "Ekstra_Bladet_word2vec" / "document_vector.parquet"
        assert word2vec_path.exists(), f"{word2vec_path} not found -- run build_embeddings.py first"
        doc_vecs = pd.read_parquet(word2vec_path)  # article_id is int32 here too -- kept native, see _load_articles
        vec_by_id = dict(zip(doc_vecs["article_id"], doc_vecs["document_vector"]))
        ids, vecs = [], []
        for a in articles["article_id"]:
            v = vec_by_id.get(a)
            if v is not None:
                ids.append(a)
                vecs.append(v)
        coverage_pct = round(100 * len(ids) / len(articles), 2) if len(articles) else 0.0
        print(f"ebnerd/semantic: word2vec coverage of test catalog = {len(ids)}/{len(articles)} ({coverage_pct}%)")
        mat = np.stack(vecs).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = (mat / norms).astype(np.float32)
        index = SemanticIndex(ids, mat)
        n = best_n(results_dir, "ebnerd", "semantic", cfg["retrieval"]["semantic"]["user_history_n"])
        dim = mat.shape[1]
        embedding_lookup = dict(zip(index.article_ids, index.embeddings))
        article_title = None

    id_to_col = {aid: i for i, aid in enumerate(index.article_ids)}
    total_impressions = pq.ParquetFile(extract_dir / "test" / "behaviors.parquet").metadata.num_rows
    print(f"ebnerd: {total_impressions} test impressions to score")

    def row_iter():
        nonlocal_n_uncovered = 0
        batches_done = 0
        total_rows = None
        for batch, total_rows in _iter_behavior_batches(extract_dir, history_by_user, BATCH_SIZE):
            if method == "bm25":
                score_maps = score_bm25_batch_candidates(index, batch, article_title, n, train_click_count, id_to_col)
            else:
                score_maps = score_semantic_batch_candidates(index, batch, embedding_lookup, n, dim,
                                                               train_click_count, id_to_col)
            for impression_id, candidates, score_map in zip(batch["impression_id"], batch["candidates"], score_maps):
                scores = []
                for a in candidates:
                    s = score_map.get(a)
                    if s is None:
                        nonlocal_n_uncovered += 1
                        s = -1e9
                    scores.append(float(s))
                ranks = scores_to_ranks(scores)
                yield impression_id, ranks
            batches_done += 1
            if batches_done % PRINT_EVERY_BATCHES == 0:
                print(f"  ...scored ~{batches_done * BATCH_SIZE}/{total_rows} impressions "
                      f"({nonlocal_n_uncovered} uncovered-candidate scores so far)")
        row_iter.n_uncovered = nonlocal_n_uncovered
        row_iter.total_rows = total_rows

    txt_path = predictions_dir / "ebnerd_predictions.txt"
    n_written = write_submission_streaming(row_iter(), txt_path, expected_total=total_impressions)
    print(f"ebnerd/{method}: validated and wrote {n_written} rows (n={n}, "
          f"{getattr(row_iter, 'n_uncovered', '?')} uncovered-candidate scores)")

    # Codabench's guidelines require the file INSIDE the zip to be named
    # exactly "predictions.txt" (plural) -- see SPEC.md Q5 -- even though our
    # on-disk copy is named ebnerd_predictions.txt for clarity alongside mind's.
    codabench_txt = predictions_dir / "predictions.txt"
    shutil.copyfile(txt_path, codabench_txt)
    zip_path = predictions_dir / "ebnerd_predictions.zip"
    zip_submission(codabench_txt, zip_path)
    codabench_txt.unlink()
    print(f"ebnerd/{method}: wrote {txt_path} and {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "semantic"], default="semantic",
                         help="semantic is the default: it had marginally higher AUC/MRR/nDCG than bm25 for EB-NeRD in Q4 eval")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    generate(cfg, args.method)


if __name__ == "__main__":
    main()
