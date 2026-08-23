"""Q5 -- Generate the MIND Codabench submission.

CORRECTED 2026-08-23: the original version of this module scored
data/splits/mind/test (== all of MINDsmall_dev), on the DOCUMENTED but
WRONG assumption that the assignment provided no separate MIND test file.
That submission failed Codabench scoring with a candidate-count mismatch
(some impression had 22 submitted ranks vs 16 expected) -- verified NOT to
be a bug in our own pipeline (all 73,152 MINDsmall_dev rows were checked
byte-for-byte against the raw MINDsmall_dev.tsv with zero mismatches), so
the only explanation left was that Codabench's actual reference file
differs from the public MINDsmall_dev.zip. The real official MIND test set
is MINDlarge_test (from https://msnews.github.io/, the original MIND
challenge's blind, unlabeled test split) -- confirmed by manually finding
it on that site (2026-08-23) after this failure. It is a SEPARATE, larger,
disjoint catalog from MINDsmall (2,370,727 impressions, 120,961 articles,
vs MINDsmall_dev's 73,152 / ~51K), with NO labels in its impressions field
(just candidate IDs, e.g. "N101071 N15647 ...", no "-0"/"-1" suffix).

This module now mirrors src/submission/ebnerd.py's approach for the same
reason: build a FRESH BM25/semantic index over MINDlarge_test's own
news.tsv (not feature_store/mind's MINDsmall-built index), and stream
behaviors.tsv in batches with candidates-only scoring (score_*_batch_candidates)
rather than materializing the whole file / building full-catalog score dicts
-- both real bugs found and fixed while building ebnerd.py at 13.5M-impression
scale; applied here preemptively rather than rediscovering them at MIND's
2.37M-impression scale.

Usage:
    python -m src.submission.mind --method bm25
    python -m src.submission.mind --method semantic

Writes:
    predictions/mind_predictions.txt
    predictions/mind_prediction.zip   (contains prediction.txt at the root --
                                        the exact filename Codabench's guidelines
                                        require, NOT mind_predictions.txt)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.pipeline.clean_mind import _load_news
from src.pipeline.unzip_utils import extract_clean
from src.retrieval.bm25 import BM25Index
from src.retrieval.scoring import best_n, score_bm25_batch_candidates, score_semantic_batch_candidates
from src.retrieval.semantic import SemanticIndex
from src.submission.format import scores_to_ranks, write_submission_streaming, zip_submission

BATCH_SIZE = 256
PRINT_EVERY_BATCHES = 200  # ~51K impressions/print at BATCH_SIZE=256


def _ensure_test_extracted(raw_dir: Path) -> Path:
    zip_path = raw_dir / "MINDlarge_test.zip"
    assert zip_path.exists(), (
        f"{zip_path} not found -- download it from https://msnews.github.io/ "
        "(the official MIND challenge site) and place it at that path"
    )
    # Same wrapping-folder pattern as MINDsmall's zips (verified via `unzip -l`,
    # 2026-08-23): members are under a MINDlarge_test/ folder already.
    return extract_clean(zip_path, raw_dir, raw_dir / "MINDlarge_test")


def _iter_behavior_batches(behaviors_tsv: Path, batch_size: int):
    """Streams behaviors.tsv in batches instead of loading all 2.37M rows at
    once -- see module docstring. impressions field has NO -label suffix
    (blind test), so candidates are just a whitespace-split list of IDs."""
    total_rows = sum(1 for _ in open(behaviors_tsv))
    reader = pd.read_csv(
        behaviors_tsv, sep="\t", header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
        dtype=str, chunksize=batch_size, usecols=["impression_id", "history", "impressions"],
    )
    for batch in reader:
        batch["history_ids"] = batch["history"].apply(lambda h: h.split() if isinstance(h, str) and h.strip() else [])
        batch["candidates"] = batch["impressions"].apply(lambda s: s.split())
        yield batch, total_rows


def _load_entity_vectors(vec_path: Path) -> dict[str, np.ndarray]:
    vecs: dict[str, np.ndarray] = {}
    with open(vec_path) as f:
        for line in f:
            # rstrip() (not just "\n"): every line has a trailing tab before the
            # newline, otherwise np.asarray(..., dtype=float32) crashes on ''.
            parts = line.rstrip().split("\t")
            vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vecs


def generate(cfg: dict, method: str) -> None:
    raw_dir = resolve_path(cfg, "raw_dir")
    results_dir = resolve_path(cfg, "results_dir")
    predictions_dir = resolve_path(cfg, "predictions_dir")

    print("mind: ensuring MINDlarge_test.zip is extracted...")
    extract_dir = _ensure_test_extracted(raw_dir)
    articles = _load_news(extract_dir / "news.tsv")
    print(f"mind: loaded {len(articles)} test articles from MINDlarge_test/news.tsv")

    train_click_count: dict[str, float] = {}  # no train-split popularity for this disjoint catalog (see docstring)
    if method == "bm25":
        print("mind: building a FRESH BM25 index over MINDlarge_test's catalog (not reusing MINDsmall's)...")
        texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
        index = BM25Index.build(articles["article_id"].tolist(), texts, lang="english")
        n = best_n(results_dir, "mind", "bm25", cfg["retrieval"]["bm25"]["query_history_n"])
        article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))
        embedding_lookup = dim = None
    else:
        entity_vecs = _load_entity_vectors(extract_dir / "entity_embedding.vec")
        dim = next(iter(entity_vecs.values())).shape[0] if entity_vecs else 100
        ids, vecs = [], []
        for article_id, entity_ids in zip(articles["article_id"], articles["entity_wikidata_ids"]):
            matched = [entity_vecs[e] for e in entity_ids if e in entity_vecs]
            if matched:
                ids.append(article_id)
                vecs.append(np.mean(matched, axis=0))
        coverage_pct = round(100 * len(ids) / len(articles), 2) if len(articles) else 0.0
        print(f"mind/semantic: entity-embedding coverage of test catalog = {len(ids)}/{len(articles)} ({coverage_pct}%)")
        mat = np.stack(vecs).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = (mat / norms).astype(np.float32)
        index = SemanticIndex(ids, mat)
        n = best_n(results_dir, "mind", "semantic", cfg["retrieval"]["semantic"]["user_history_n"])
        embedding_lookup = dict(zip(index.article_ids, index.embeddings))
        article_title = None

    id_to_col = {aid: i for i, aid in enumerate(index.article_ids)}

    def row_iter():
        n_uncovered = 0
        batches_done = 0
        for batch, total_rows in _iter_behavior_batches(extract_dir / "behaviors.tsv", BATCH_SIZE):
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
                        n_uncovered += 1
                        s = -1e9
                    scores.append(float(s))
                ranks = scores_to_ranks(scores)
                yield impression_id, ranks
            batches_done += 1
            if batches_done % PRINT_EVERY_BATCHES == 0:
                print(f"  ...scored ~{batches_done * BATCH_SIZE}/{total_rows} impressions "
                      f"({n_uncovered} uncovered-candidate scores so far)")
        row_iter.n_uncovered = n_uncovered

    total_impressions = sum(1 for _ in open(extract_dir / "behaviors.tsv"))
    txt_path = predictions_dir / "mind_predictions.txt"
    n_written = write_submission_streaming(row_iter(), txt_path, expected_total=total_impressions)
    print(f"mind/{method}: validated and wrote {n_written} rows (n={n}, "
          f"{getattr(row_iter, 'n_uncovered', '?')} uncovered-candidate scores)")

    # Codabench's guidelines require the file INSIDE the zip to be named
    # exactly "prediction.txt" (singular) -- confirmed from the downloaded
    # sample -- even though our on-disk copy is named mind_predictions.txt
    # for clarity alongside the ebnerd one.
    codabench_txt = predictions_dir / "prediction.txt"
    shutil.copyfile(txt_path, codabench_txt)
    zip_path = predictions_dir / "mind_prediction.zip"
    zip_submission(codabench_txt, zip_path)
    codabench_txt.unlink()
    print(f"mind/{method}: wrote {txt_path} and {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "semantic"], default="bm25",
                         help="bm25 is the default: it had higher AUC/MRR than semantic for MIND in Q4 eval")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    generate(cfg, args.method)


if __name__ == "__main__":
    main()
