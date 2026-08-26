"""Rigor addition -- measure the OpenBLAS thread-count claim in design_note.tex S7 instead
of just asserting it ("this machine's numpy build caps OpenBLAS at 2 threads, compounding
[the full-catalog matmul cost]"). That claim is about the SEMANTIC full-catalog dense
matmul (SemanticIndex.score_dense, `query_vecs @ embeddings.T` -- see src/retrieval/
semantic.py), which is the operation actually measured at 367 impressions/sec in S7 point 3
-- that's --mode semantic here. --mode bm25 is included too, but as a DIFFERENT, separate
finding: bm25.py's scoring is a SPARSE matmul (`Qmat @ W.T`, then `.todense()`), which uses
scipy's own sparse-matrix routines rather than a dense BLAS GEMM call, so there's no strong
prior it should scale with BLAS thread count the same way -- worth checking empirically
rather than assuming the same mechanism applies to both.

OpenBLAS's thread count is read from the OPENBLAS_NUM_THREADS / OMP_NUM_THREADS environment
variables at *import* time, so each thread-count value needs its own fresh process --
re-importing numpy in the same process after the fact has no effect. This script is both
the orchestrator (spawns one subprocess per thread count) and, via --worker, the thing that
actually gets spawned: it loads the index for one dataset/mode and times the scoring call
over a fixed batch of real queries, repeated for a stable average, printing one JSON line to
stdout for the parent to parse.

Usage:
    python -m src.eval.bench_thread_scaling --dataset mind --mode semantic --threads 1 2 4 8

Writes:
    results/<dataset>_thread_scaling_<mode>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

N_QUERIES = 512
N_REPEATS = 5


def _worker_bm25(dataset: str, cfg: dict):
    import pandas as pd
    from src.config import resolve_path
    from src.retrieval.bm25 import BM25Index, build_query_text

    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    index = BM25Index.load(feature_store_dir / dataset / "bm25")
    articles = pd.read_parquet(feature_store_dir / dataset / "articles.parquet")
    article_title = dict(zip(articles["article_id"], articles["title"].fillna("")))

    imp = pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet", columns=["history_ids"])
    import numpy as np
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(imp), size=min(N_QUERIES, len(imp)), replace=False)
    queries = [build_query_text(list(h), article_title, n=5) for h in imp.iloc[idx]["history_ids"]]
    queries = [q for q in queries if q.strip()] or ["placeholder"]

    def call():
        index.score_batch(queries)
    return call, len(queries), len(index.article_ids)


def _worker_semantic(dataset: str, cfg: dict):
    """Times SemanticIndex.score_dense -- the full-catalog dense matmul S7's OpenBLAS
    claim is actually about (not the candidates-only-restricted version scoring.py uses
    in the real pipeline post-optimization)."""
    import numpy as np
    import pandas as pd
    from src.config import resolve_path
    from src.retrieval.semantic import SemanticIndex, build_user_embedding

    feature_store_dir = resolve_path(cfg, "feature_store_dir")
    splits_dir = resolve_path(cfg, "splits_dir")
    out_dir = feature_store_dir / dataset
    article_ids = np.load(out_dir / "embeddings_ids.npy", allow_pickle=True).tolist()
    embeddings = np.load(out_dir / "embeddings.npy").astype(np.float32)
    dim = embeddings.shape[1]
    index = SemanticIndex(article_ids, embeddings)
    embedding_lookup = dict(zip(article_ids, embeddings))

    imp = pd.read_parquet(splits_dir / dataset / "test" / "impressions.parquet", columns=["history_ids"])
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(imp), size=min(N_QUERIES * 2, len(imp)), replace=False)
    vecs = []
    for h in imp.iloc[idx]["history_ids"]:
        v, is_cold = build_user_embedding(list(h), embedding_lookup, 1, dim)
        if not is_cold:
            vecs.append(v)
        if len(vecs) >= N_QUERIES:
            break
    queries = np.stack(vecs).astype(np.float32)

    def call():
        index.score_dense(queries)
    return call, len(queries), len(article_ids)


def _worker(dataset: str, mode: str) -> None:
    # Imports deliberately deferred to inside _worker: OpenBLAS reads its thread-count env
    # vars once, at numpy's first import, so this must run in a subprocess that had the env
    # var set BEFORE the interpreter even started -- not after re-importing in-process.
    from src.config import load_config, seed_everything

    cfg = load_config()
    seed_everything(cfg["seed"])

    call, n_queries, n_docs = (_worker_bm25 if mode == "bm25" else _worker_semantic)(dataset, cfg)

    call()  # warm-up: first call pays any one-off lazy-init cost
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)

    print(json.dumps({
        "mode": mode,
        "n_queries": n_queries,
        "n_docs": n_docs,
        "times_seconds": times,
        "median_seconds": sorted(times)[len(times) // 2],
        "queries_per_second_median": n_queries / sorted(times)[len(times) // 2],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mind", choices=["mind", "ebnerd"])
    parser.add_argument("--mode", default="semantic", choices=["bm25", "semantic"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.dataset, args.mode)
        return

    from src.config import load_config, resolve_path

    results_dir = resolve_path(load_config(), "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for n_threads in args.threads:
        env = dict(os.environ)
        env["OPENBLAS_NUM_THREADS"] = str(n_threads)
        env["OMP_NUM_THREADS"] = str(n_threads)
        env["MKL_NUM_THREADS"] = str(n_threads)
        cmd = [sys.executable, "-m", "src.eval.bench_thread_scaling",
               "--dataset", args.dataset, "--mode", args.mode, "--worker"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"threads={n_threads}: FAILED\n{proc.stderr}", file=sys.stderr)
            continue
        # worker prints exactly one JSON line; tolerate any incidental stdout noise around it
        line = next(l for l in proc.stdout.splitlines() if l.strip().startswith("{"))
        data = json.loads(line)
        runs[str(n_threads)] = data
        print(f"threads={n_threads}: median={data['median_seconds']*1000:.2f}ms "
              f"({data['queries_per_second_median']:.0f} q/s)")

    baseline = runs.get("1", {}).get("queries_per_second_median")
    for n_threads, data in runs.items():
        if baseline:
            data["speedup_vs_1_thread"] = data["queries_per_second_median"] / baseline

    out = {"dataset": args.dataset, "mode": args.mode, "runs": runs}
    out_path = results_dir / f"{args.dataset}_thread_scaling_{args.mode}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
