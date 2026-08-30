# Lexical & Semantic Retrieval on EB-NeRD and MIND

CS4.406 Assignment 1 — Yash More (2024114004)

Two retrieval systems built and measured on two news recommendation datasets: a lexical one
(BM25, implemented from scratch as a sparse matrix) and a semantic one (provided article
embeddings + FAISS). Both are evaluated offline with an in-repo metrics harness and were
submitted to the two Codabench leaderboards.

## Start here

| What | Where |
|---|---|
| **Design note (main deliverable, 4 pages)** | [`docs/design_note.pdf`](docs/design_note.pdf) |
| All measured results (JSON, one file per experiment) | [`results/`](results/) |
| Codabench leaderboard screenshots | [`docs/screenshots/`](docs/screenshots/) |
| Full design rationale + requirements traceability | [`SPEC.md`](SPEC.md) |
| AI usage log | [`docs/ai_usage_log.md`](docs/ai_usage_log.md) |

Leaderboard scores: **MIND 0.5851 AUC**, **EB-NeRD 0.514 AUC**.

## What is not in this archive

Three directories are excluded because of size. Everything in them is regenerable from the
code here, and no result depends on having them present to *read*:

| Excluded | Size | Regenerate with |
|---|---|---|
| `data/` (raw + interim + splits) | 6.3 GB | `make data` |
| `feature_store/` (indices, embeddings) | 95 MB | `make data` then `make retrieval` |
| `predictions/` (Codabench submission files) | 1.3 GB | `make submit` |

The `results/*.json` files **are** included — they are small and are the evidence behind every
number in the design note, so the reported numbers can be checked without re-running anything.

## Running it

Python 3.12. No GPU required; everything runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test          # 40 passed, 22 skipped, ~4s — no data download needed
```

`make test` is the fastest way to confirm the checkout is sound. The 40 that run cover the
metrics implementations, the BM25 scorer, the slicing logic and submission formatting, all on
synthetic fixtures.

The 22 skips are expected in this archive and are not failures. The leakage and
split-boundary tests (`test_no_leakage.py`, `test_split_disjoint.py`, `test_bm25_agreement.py`)
read the real `data/splits/` and `feature_store/` directories, which are excluded here for
size, so they skip with an explicit `run make data first` message rather than silently
passing. After `make data` all 62 run, and the full suite is 59 passed / 3 skipped.

To rebuild everything from scratch:

```bash
make data          # download -> clean -> temporal split -> feature store
make retrieval     # BM25 index + N-sweep; embeddings + semantic eval
make eval          # ranking metrics, beyond-accuracy, slices
make ablation      # serving-feature leakage ablation (EB-NeRD only)
make submit        # regenerate Codabench prediction files
```

`make data` is idempotent and safe to re-run. It downloads MIND-small and EB-NeRD demo (see
`config/default.yaml`), parses both into one unified schema, splits each **temporally, never
randomly** (see SPEC.md Q1.3), and materialises the feature store.

Expect `make data` to take a while on first run, and `make submit` considerably longer — it
scores 13.5M EB-NeRD impressions and 2.37M MIND impressions against their official test
catalogs.

## Pipeline stages

**Q1 — Data pipeline** (`src/pipeline/`, `build_pipeline.py`). Both datasets parsed to a
common schema. Splits are drawn by time; `tests/test_no_leakage.py` asserts on every split
that the latest click timestamp in any user's history precedes the earliest impression
timestamp in that split.

**Q2 — Lexical retrieval** (`src/retrieval/{tokenize,bm25,candidates,run_bm25_eval}.py`).
BM25 built as a sparse article × vocabulary weight matrix, so scoring a whole query batch is
one sparse matmul and the matrix's non-zero pattern *is* the inverted index. Sweeps query
history length N on validation, reports recall@{50,100,200} on test under two candidate pools
with bootstrap CIs.

```bash
python -m src.retrieval.run_bm25_eval --datasets mind ebnerd
```

**Q3 — Semantic retrieval** (`src/retrieval/{build_embeddings,semantic,run_semantic_eval}.py`).
**Provided embeddings only, no training or model inference**: EB-NeRD's shipped word2vec
document vectors, MIND's shipped entity embeddings mean-pooled per article. Uses FAISS
`IndexFlatIP` when `faiss-cpu` is available, otherwise an equivalent numpy brute-force
fallback — both exact, neither approximate.

```bash
python -m src.retrieval.build_embeddings --datasets mind ebnerd
python -m src.retrieval.run_semantic_eval --datasets mind ebnerd
```

**Q4 — Evaluation harness** (`src/eval/{metrics,beyond_accuracy,slicing,run_eval}.py`). Scores
each impression's *own* candidate list, which is a different task from Q2/Q3's catalog-wide
recall@K (SPEC.md Q4.0). Reports AUC, MRR, nDCG@{5,10} with bootstrap 95% CIs, plus
beyond-accuracy metrics (intra-list diversity in both categorical and embedding variants,
novelty, coverage) over each impression's top-10, plus all of the above sliced by
cold-start/warm users and head/tail articles.

```bash
python -m src.eval.run_eval --datasets mind ebnerd --split test
```

**Q5 — Codabench submission** (`src/submission/{format,mind,ebnerd}.py`). Generates a ranked
prediction for every impression in the official test set and validates it structurally (full
coverage, original row order, valid rank permutations) *before* zipping, since MIND allows
only 1 upload/day and EB-NeRD 5/day.

```bash
python -m src.submission.mind --method bm25       # BM25 won on MIND in Q4
python -m src.submission.ebnerd --method semantic  # semantic won on EB-NeRD in Q4
```

Both build a fresh index over their own official test catalog rather than reusing the
demo-scale feature store. MIND's official test set is `MINDlarge_test` (from
https://msnews.github.io/, downloaded separately), not `MINDsmall_dev` — an earlier version
assumed the latter and Codabench rejected it with a candidate-count mismatch. Each script
streams its behaviour file in batches and scores only each impression's own candidates rather
than the whole catalog per row; at EB-NeRD's real scale the naive approach ran out of memory
and would have taken 70+ hours (see the design note's "Where It Breaks at Scale").

Note the archive filenames are dictated by Codabench, not chosen: MIND's zip must contain
`prediction.txt` (singular), EB-NeRD's must contain `predictions.txt` (plural).

**Q9.1 — Serving-feature ablation** (`src/eval/run_ablation.py`). EB-NeRD only, because MIND's
raw data carries none of these columns. Three cumulative feature configurations quantify how
much of an apparently strong result would be leakage rather than retrieval quality.

```bash
python -m src.eval.run_ablation --split test
```

## Measurement scripts

These back the quantitative claims in the design note about *tool and index choices*, rather
than about accuracy. Each writes its own `results/*.json`.

```bash
python -m src.eval.bench_bm25_alternatives --datasets mind ebnerd   # this BM25 vs rank_bm25
python -m src.eval.bench_ann_recall_latency --datasets mind ebnerd  # exact vs IVF vs HNSW
python -m src.eval.run_tokenizer_ablation --datasets mind ebnerd    # stemming on/off
python -m src.eval.bench_thread_scaling --dataset mind --mode semantic  # BLAS thread scaling
```

Headline findings: the sparse-matrix BM25 is ~782× faster than `rank_bm25` on MIND; IVF and
HNSW are both >10× faster than exact search at >90% agreement with its top-100, though exact
search is fast enough at this scale to not need them; stemming helps EB-NeRD (Danish
compounds) and slightly hurts MIND.


## Repository layout

```
retrival-systems/
├── config/default.yaml          # all hyperparameters + seeds in one place
├── src/
│   ├── config.py                # load yaml, seed everything
│   ├── pipeline/                # Q1 — download, clean, temporal split, feature store
│   ├── retrieval/               # Q2/Q3 — tokenize, bm25, semantic, candidates, evals
│   ├── eval/                    # Q4/Q9.1 — metrics, slicing, bootstrap, ablation, benchmarks
│   └── submission/              # Q5 — format validation, mind, ebnerd
├── tests/                       # 59 tests incl. test_no_leakage.py (Q9.2)
├── results/                     # measured JSON, one file per experiment — INCLUDED
├── docs/
│   ├── design_note.pdf          # Q6 deliverable (4 pages)
│   ├── screenshots/             # Q7.3 Codabench leaderboard screenshots
│   └── ai_usage_log.md          # Q7.4
├── build_pipeline.py            # Q1.5 — one-command entry point
├── Makefile, requirements.txt, pytest.ini
├── SPEC.md, README.md
├── data/                        # EXCLUDED — 6.3 GB, regenerate with `make data`
├── feature_store/               # EXCLUDED — 95 MB, regenerate with `make data`+`make retrieval`
└── predictions/                 # EXCLUDED — 1.3 GB, regenerate with `make submit`
```

A note on SPEC.md: it was written *before* implementation, as the plan and requirements
traceability document, and a few paths in its own layout section drifted during the build. The
tree above is the accurate one. In particular the serving-feature ablation ended up in
`src/eval/run_ablation.py` rather than a separate `src/ablation/` package, the leaderboard
screenshots are in `docs/screenshots/` rather than `docs/leaderboard/`, and `predictions/` grew
far past the size at which committing it was sensible, so it is gitignored and excluded here.

## AI usage

See [`docs/ai_usage_log.md`](docs/ai_usage_log.md).
