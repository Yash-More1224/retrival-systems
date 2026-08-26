"""Rigor addition -- the recall-vs-latency curve design_note.tex explicitly lists as NOT
done ("FAISS exact search only: no ANN (HNSW/IVF) recall-vs-latency curve was run at
production scale"). SemanticIndex only ever builds IndexFlatIP (exact, brute-force); this
compares it against two real approximate indexes -- IndexIVFFlat (inverted-file, tunable via
nprobe = how many of the nlist clusters to actually search) and IndexHNSWFlat (graph-based,
tunable via efSearch = how wide the graph search beam is) -- at demo/small scale, which is
what's locally available; production-scale (10x+ catalog) is extrapolated qualitatively in
the design note, not measured here.

"Recall" here means agreement with the exact index's own top-K (a standard way to evaluate
an ANN index: how often does the approximate search return the same top-K docs the exact
search would have), not agreement with ground-truth clicks -- that's what Q3's recall@K
already measures for the exact index elsewhere.

Usage:
    python -m src.eval.bench_ann_recall_latency --datasets mind ebnerd

Writes:
    results/<dataset>_ann_recall_latency.json
"""
from __future__ import annotations

import argparse
import json
import time

import faiss
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path, seed_everything
from src.retrieval.semantic import build_user_embedding

K = 100
N_QUERIES = 500


def _sample_query_vecs(splits_dir, dataset: str, embedding_lookup: dict, dim: int, n: int,
                        n_queries: int, seed: int) -> np.ndarray:
    imp = pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet", columns=["history_ids"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imp), size=min(n_queries * 3, len(imp)), replace=False)  # oversample, cold-starts get dropped
    vecs = []
    for h in imp.iloc[idx]["history_ids"]:
        v, is_cold = build_user_embedding(list(h), embedding_lookup, n, dim)
        if not is_cold:
            vecs.append(v)
        if len(vecs) >= n_queries:
            break
    return np.stack(vecs).astype(np.float32)


def _time_search(index, queries: np.ndarray, k: int, nprobe: int | None = None, ef_search: int | None = None):
    if nprobe is not None:
        index.nprobe = nprobe
    if ef_search is not None:
        index.hnsw.efSearch = ef_search
    t0 = time.perf_counter()
    _, ids = index.search(queries, k)
    elapsed = time.perf_counter() - t0
    return ids, elapsed


def _recall_vs_exact(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    per_query = [
        len(set(a.tolist()) & set(e.tolist())) / len(set(e.tolist()))
        for a, e in zip(approx_ids, exact_ids) if len(set(e.tolist())) > 0
    ]
    return float(np.mean(per_query)) if per_query else float("nan")


def run_dataset(cfg: dict, dataset: str) -> dict:
    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")

    out_dir = feature_store_dir / dataset
    article_ids = np.load(out_dir / "embeddings_ids.npy", allow_pickle=True).tolist()
    embeddings = np.load(out_dir / "embeddings.npy").astype(np.float32)
    n_docs, dim = embeddings.shape
    embedding_lookup = dict(zip(article_ids, embeddings))

    sweep_path = resolve_path(cfg, "results_dir") / f"{dataset}_semantic_val_sweep.json"
    best_n = int(json.loads(sweep_path.read_text())["best_n"]) if sweep_path.exists() else 1

    queries = _sample_query_vecs(splits_dir, dataset, embedding_lookup, dim, best_n, N_QUERIES, cfg["seed"])
    print(f"{dataset}: {n_docs} docs (dim={dim}), {len(queries)} real query vectors (N={best_n})")

    # --- exact baseline ---
    exact = faiss.IndexFlatIP(dim)
    exact.add(embeddings)
    exact_ids, exact_time = _time_search(exact, queries, K)
    exact_qps = len(queries) / exact_time if exact_time > 0 else float("inf")
    print(f"  exact (IndexFlatIP):  {exact_time*1000:.2f}ms total, {exact_qps:.0f} q/s, recall=1.0 (is the ground truth)")

    runs = {"exact_flat": {"latency_ms_total": exact_time * 1000, "qps": exact_qps, "recall_vs_exact": 1.0}}

    # --- IVF: cluster the catalog into nlist cells, search only nprobe of them ---
    nlist = max(4, min(256, int(round(n_docs ** 0.5))))
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(embeddings)
    ivf.add(embeddings)
    ivf_runs = {}
    for nprobe in sorted(set([1, 2, 4, 8, 16, min(32, nlist), nlist])):
        ids, t = _time_search(ivf, queries, K, nprobe=nprobe)
        qps = len(queries) / t if t > 0 else float("inf")
        recall = _recall_vs_exact(ids, exact_ids)
        ivf_runs[str(nprobe)] = {"latency_ms_total": t * 1000, "qps": qps, "recall_vs_exact": recall}
        print(f"  IVF nlist={nlist} nprobe={nprobe:>3d}: {t*1000:7.2f}ms, {qps:8.0f} q/s, recall_vs_exact={recall:.4f}")
    runs["ivf_flat"] = {"nlist": nlist, "by_nprobe": ivf_runs}

    # --- HNSW: graph search, no training step, tunable via efSearch ---
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(embeddings)
    hnsw_runs = {}
    for ef in [16, 32, 64, 128, 256]:
        ids, t = _time_search(hnsw, queries, K, ef_search=ef)
        qps = len(queries) / t if t > 0 else float("inf")
        recall = _recall_vs_exact(ids, exact_ids)
        hnsw_runs[str(ef)] = {"latency_ms_total": t * 1000, "qps": qps, "recall_vs_exact": recall}
        print(f"  HNSW M=32 efSearch={ef:>3d}: {t*1000:7.2f}ms, {qps:8.0f} q/s, recall_vs_exact={recall:.4f}")
    runs["hnsw"] = {"M": 32, "efConstruction": 40, "by_efSearch": hnsw_runs}

    return {"dataset": dataset, "n_docs": n_docs, "dim": dim, "n_queries": len(queries), "k": K, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"], choices=["mind", "ebnerd"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seed_everything(cfg["seed"])
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        result = run_dataset(cfg, dataset)
        out_path = results_dir / f"{dataset}_ann_recall_latency.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"{dataset}: wrote {out_path}")


if __name__ == "__main__":
    main()
