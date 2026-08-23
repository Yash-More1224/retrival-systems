"""Q3.1 -- Load PROVIDED article embeddings for both datasets (no training,
no running a transformer -- per explicit instruction, avoid computing our
own embeddings).

  EB-NeRD: Ekstra_Bladet_word2vec.zip -> document_vector.parquet, one
    300-dim vector per article_id. Verified 100% coverage of ebnerd_demo's
    catalog (2026-08-20) -- used directly, no fallback needed in practice,
    but coverage is still checked and logged rather than assumed.

  MIND: has no provided per-article embedding. What IS provided is
    entity_embedding.vec -- 100-dim TransE-style vectors keyed by WikidataId
    (e.g. "Q41"), one per knowledge-graph entity mentioned in some article's
    title/abstract (see MIND's `title_entities`/`abstract_entities` columns,
    carried through clean_mind.py as `entity_wikidata_ids`). We build a
    per-article embedding by mean-pooling the entity vectors for the
    entities mentioned in that article. This is a real "provided embedding",
    just at entity granularity rather than document granularity -- not a
    trained/fine-tuned model.

    Coverage is NOT total: verified 86.7% of MINDsmall_train articles
    mention >=1 entity (44484/51282), and 97% of mentioned WikidataIds have
    a vector in entity_embedding.vec. The remaining ~13-14% of articles
    (zero entities, or only entities missing from the .vec file) have no
    embedding and are DROPPED from the semantic index and reported as a
    coverage gap (see SPEC.md Q3.1: "never silently zero-filled") -- this is
    itself a legitimate design-note finding (entity-embedding coverage is a
    real limitation of the "provided embeddings only" choice for MIND,
    unlike EB-NeRD's ~universal word2vec coverage).

Writes:
    feature_store/<dataset>/embeddings.npy        (float32, L2-normalised, covered articles only)
    feature_store/<dataset>/embeddings_ids.npy    (article_id per row, aligned to embeddings.npy)
    feature_store/<dataset>/embeddings.meta.json  (dim, n_total, n_covered, coverage_pct, source)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.pipeline.unzip_utils import extract_clean


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def build_ebnerd_embeddings(raw_dir, feature_store_dir) -> dict:
    zip_path = raw_dir / "Ekstra_Bladet_word2vec.zip"
    extract_dir = extract_clean(zip_path, raw_dir / "Ekstra_Bladet_word2vec")
    vec_path = extract_dir / "Ekstra_Bladet_word2vec" / "document_vector.parquet"
    doc_vecs = pd.read_parquet(vec_path)
    doc_vecs["article_id"] = doc_vecs["article_id"].astype(str)
    vec_by_id = dict(zip(doc_vecs["article_id"], doc_vecs["document_vector"]))

    articles = pd.read_parquet(feature_store_dir / "ebnerd" / "articles.parquet", columns=["article_id"])
    ids, vecs = [], []
    for a in articles["article_id"]:
        v = vec_by_id.get(a)
        if v is not None:
            ids.append(a)
            vecs.append(v)

    n_total, n_covered = len(articles), len(ids)
    mat = _l2_normalize(np.stack(vecs).astype(np.float32)) if vecs else np.zeros((0, 300), dtype=np.float32)
    dim = mat.shape[1] if mat.shape[0] else 300

    out_dir = feature_store_dir / "ebnerd"
    np.save(out_dir / "embeddings.npy", mat)
    np.save(out_dir / "embeddings_ids.npy", np.asarray(ids, dtype=object))
    meta = {
        "source": "provided: Ekstra_Bladet_word2vec/document_vector.parquet",
        "dim": dim, "n_total": n_total, "n_covered": n_covered,
        "coverage_pct": round(100 * n_covered / n_total, 2) if n_total else 0.0,
    }
    (out_dir / "embeddings.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ebnerd: embeddings coverage {n_covered}/{n_total} ({meta['coverage_pct']}%), dim={dim}")
    return meta


def _load_entity_vectors(raw_dir) -> dict[str, np.ndarray]:
    vecs: dict[str, np.ndarray] = {}
    for split_dir in ("MINDsmall_train", "MINDsmall_dev"):
        path = raw_dir / split_dir / "entity_embedding.vec"
        with open(path) as f:
            for line in f:
                # rstrip() (not just "\n"): every line in this file has a trailing tab
                # before the newline, which otherwise leaves an empty trailing field and
                # np.asarray(..., dtype=float32) crashes on ''.
                parts = line.rstrip().split("\t")
                entity_id, values = parts[0], parts[1:]
                vecs[entity_id] = np.asarray(values, dtype=np.float32)
    return vecs


def build_mind_embeddings(raw_dir, feature_store_dir) -> dict:
    entity_vecs = _load_entity_vectors(raw_dir)
    dim = next(iter(entity_vecs.values())).shape[0] if entity_vecs else 100

    articles = pd.read_parquet(feature_store_dir / "mind" / "articles.parquet",
                                columns=["article_id", "entity_wikidata_ids"])
    ids, vecs = [], []
    for article_id, entity_ids in zip(articles["article_id"], articles["entity_wikidata_ids"]):
        matched = [entity_vecs[e] for e in entity_ids if e in entity_vecs]
        if matched:
            ids.append(article_id)
            vecs.append(np.mean(matched, axis=0))

    n_total, n_covered = len(articles), len(ids)
    mat = _l2_normalize(np.stack(vecs).astype(np.float32)) if vecs else np.zeros((0, dim), dtype=np.float32)

    out_dir = feature_store_dir / "mind"
    np.save(out_dir / "embeddings.npy", mat)
    np.save(out_dir / "embeddings_ids.npy", np.asarray(ids, dtype=object))
    meta = {
        "source": "provided: entity_embedding.vec, mean-pooled over title+abstract entities",
        "dim": dim, "n_total": n_total, "n_covered": n_covered,
        "coverage_pct": round(100 * n_covered / n_total, 2) if n_total else 0.0,
    }
    (out_dir / "embeddings.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"mind: embeddings coverage {n_covered}/{n_total} ({meta['coverage_pct']}%), dim={dim} "
          f"-- {n_total - n_covered} articles have no matched entity vector and are excluded "
          f"from semantic retrieval (see build_embeddings.py docstring)")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    raw_dir = resolve_path(cfg, "raw_dir")
    feature_store_dir = resolve_path(cfg, "feature_store_dir")

    if "ebnerd" in args.datasets:
        build_ebnerd_embeddings(raw_dir, feature_store_dir)
    if "mind" in args.datasets:
        build_mind_embeddings(raw_dir, feature_store_dir)


if __name__ == "__main__":
    main()
