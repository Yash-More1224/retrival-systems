# Lexical & Semantic Retrieval on EB-NeRD and MIND

CS4.406 Assignment 1, Part I. Full design rationale, verified dataset schemas, and the
requirements traceability table live in [SPEC.md](SPEC.md) — read that first.

## Quickstart (on the remote GPU node)

All development and execution happens on the remote GPU node (`ada`), not on this checkout
directly. Sync the code over, then run the same commands there. From this machine's terminal
(NOT from `ada`) -- `data/raw/`, `feature_store/`, and `.venv/` are gitignored and large, so
don't sync the whole repo blindly; sync the code directories explicitly instead:

```bash
scp -r retrival-systems yash.more@ada:~/ire/a1
```

Then on `ada`:
```bash
cd ~/ire/a1/retrival-systems
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data   # Q1: download -> clean -> temporal split -> feature store
make test   # Q9.2: leakage / split-boundary tests
```

`make data` is idempotent — safe to re-run. It downloads MIND-small and EB-NeRD demo
(config default; see `config/default.yaml: ebnerd.scale`) if not already present in
`data/raw/`, parses both into a unified schema, splits each temporally (never randomly —
see SPEC.md Q1.3), and materializes the feature store under `feature_store/`.

To also fetch the EB-NeRD Codabench test set (1.5GB, needed only for Q5, not for offline
dev/eval):
```bash
python -m src.pipeline.download --datasets ebnerd --include-testset
```

To build final numbers on `ebnerd_small` instead of the default `ebnerd_demo`:
```bash
python build_pipeline.py --ebnerd-scale small
```

Then Q2 (BM25 candidate generation — builds/caches the index, sweeps query-history length N
on val, reports recall@{50,100,200} on test under Pool A/Pool B with bootstrap CIs):
```bash
python -m src.retrieval.run_bm25_eval --datasets mind ebnerd
```
Writes `results/<dataset>_bm25_val_sweep.json` (full N-sweep, for review) and
`results/<dataset>_bm25_test.json` (final numbers). See SPEC.md Q2.0 for why recall@K is
reported under two pools rather than one number.

Then Q3 (semantic candidate generation — **provided embeddings only**, no training/model
inference: EB-NeRD's shipped word2vec document vectors, MIND's shipped entity embeddings
mean-pooled per article; see SPEC.md Q3.1):
```bash
python -m src.retrieval.build_embeddings --datasets mind ebnerd   # once, or after a rebuild
python -m src.retrieval.run_semantic_eval --datasets mind ebnerd
```
Writes `feature_store/<dataset>/embeddings{,_ids}.npy` + `embeddings.meta.json` (coverage),
and `results/<dataset>_semantic_{val_sweep,test}.json` in the same format as Q2's BM25 results
for direct comparison (Q3.5). Uses a real FAISS `IndexFlatIP` if `faiss-cpu` is installed
(it's in `requirements.txt`, so this is automatic on `ada`), otherwise an equivalent numpy
brute-force fallback — both are exact, not approximate.

Then Q4 (evaluation harness — scores each impression's OWN candidate list, a different task
from Q2/Q3's catalog-wide recall@K; see SPEC.md Q4.0). Requires Q2/Q3 to have been run first
(reads their `best_n` from the val-sweep results; falls back to config defaults otherwise):
```bash
python -m src.eval.run_eval --datasets mind ebnerd --split test
```
Writes `results/<dataset>_<method>_eval.json` for `method` in `{bm25, semantic}`: AUC, MRR,
nDCG@{5,10} with bootstrap 95% CIs, beyond-accuracy (intra-list diversity — categorical AND
embedding variants, novelty, coverage) over each impression's top-10, and the same four
metrics sliced by cold-start/warm and head/tail articles (SPEC.md Q4.3). Note: on both
datasets over 90% of articles have zero train-split clicks, which makes the head/tail
threshold land exactly at 0 — `src/eval/slicing.py` handles this explicitly (see its
docstring); worth restating in the design note as a real property of the data, not an
implementation quirk to gloss over.

Then Q5 (Codabench submission). Generates a ranked prediction for every impression in the
official test set, validates it structurally (full coverage, original row order, valid rank
permutations, see `src/submission/format.py`) before zipping, since MIND allows only 1
upload/day and EB-NeRD 5/day:
```bash
python -m src.submission.mind --method bm25        # bm25 had higher AUC/MRR for MIND in Q4
python -m src.submission.ebnerd --method semantic   # semantic had marginally higher AUC/MRR/nDCG for EB-NeRD
```
Both build a **fresh** index over their own official test catalog rather than reusing the
demo/small-built `feature_store/`. MIND's official test set is `MINDlarge_test` (from
https://msnews.github.io/, downloaded separately, 2.37M impressions), not MINDsmall_dev: an
earlier version of this pipeline assumed MINDsmall_dev was the test set (the assignment
provides no separate MIND test file), which Codabench's own scoring rejected with a
candidate-count mismatch, since its real reference is `MINDlarge_test`. EB-NeRD downloads
`ebnerd_testset.zip` (1.5GB) on first run if not already present. Both scripts stream their
behaviour file in batches and score only each impression's own candidates (not the whole
catalog per row) rather than materialising everything in memory at once: at EB-NeRD's real
scale (13.5M impressions, 125K articles) the naive approach OOM'd and would have taken
70+ hours; see `docs/design_note.tex` §7 for the full story. Writes
`predictions/mind_prediction.zip` (containing `prediction.txt`, the exact filename Codabench's
MIND guidelines require) and `predictions/ebnerd_predictions.zip` (containing
`ebnerd_predictions.txt`). See SPEC.md Q5 for the verified format details.

Then Q9.1 (serving-feature ablation, EB-NeRD only, since MIND's raw data has none of these
columns). Reports the same AUC/MRR/nDCG on the offline test split with three cumulative
feature configurations, quantifying how much of an apparently strong result would be leakage:
```bash
python -m src.eval.run_ablation --split test
```
Writes `results/ebnerd_ablation.json`.

## Status

- [x] Q1 — Reproducible data pipeline (`src/pipeline/`, `build_pipeline.py`)
- [x] Q2 — Lexical candidate generation (BM25) (`src/retrieval/{tokenize,bm25,candidates,run_bm25_eval}.py`)
- [x] Q3 — Semantic candidate generation (`src/retrieval/{build_embeddings,semantic,run_semantic_eval}.py`) — provided embeddings only, per user instruction
- [x] Q4 — Offline evaluation harness (`src/eval/{metrics,beyond_accuracy,slicing,run_eval}.py`)
- [x] Q5 — Codabench submission (`src/submission/{format,mind,ebnerd}.py`) — uploaded to both leaderboards (MIND scored on MINDlarge_test; EB-NeRD submitted, scoring)
- [x] Q6 — Design note (`docs/design_note.tex` / `.pdf`, 3/4 pages — leaderboard screenshots still to be inserted once EB-NeRD scoring completes)
- [x] Q9.1 — Serving-feature ablation (`src/eval/run_ablation.py`, `results/ebnerd_ablation.json`) — EB-NeRD only, MIND has no such features in its raw data
- [x] Q9.2 — Behaviour-window / no-leakage tests (`tests/test_no_leakage.py`)

## Repository layout

See SPEC.md's "Repository layout" section for the full tree and what each path is for.
`data/`, `feature_store/`, and dependency/venv artifacts are gitignored (see `.gitignore`);
`results/` and `predictions/` are small and are committed intentionally.

## AI usage

See `docs/ai_usage_log.md`.
