# Assignment 1 — Lexical & Semantic Retrieval on EB-NeRD and MIND
## Implementation Plan (v2 — revised against verified dataset schemas)

**Course**: CS4.406: Information Retrieval & Extraction
**Due**: August 27, 2026 (at Quiz-1)
**Type**: Individual
**Datasets**: MIND-small + EB-NeRD demo/small
**Submission**: GitHub Classroom (code) + Moodle (design note) + Codabench (predictions, both leaderboards)

---

## What changed in v2 (read this first)

v1 was structurally sound but built on an assumed EB-NeRD schema. I unzipped `ebnerd_demo.zip`
and read the real parquet schemas. Findings that change the design:

| # | Issue in v1 | Reality | Impact |
|---|-------------|---------|--------|
| 1 | EB-NeRD articles have `abstract` + embeddings | Columns are `title`, `subtitle`, `body`; **no embeddings in `articles.parquet`** — they ship separately in `Ekstra_Bladet_word2vec.zip` → `document_vector.parquet` | Q1, Q3 |
| 2 | `behaviors.parquet` has `article_id` as the candidate | Candidates are `article_ids_inview`; labels are `article_ids_clicked`; `article_id` is the *context* article being read (often null) | Q1, Q4, Q5 |
| 3 | `history.parquet` is one row per interaction | One row **per user**, with parallel fixed-length arrays: `article_id_fixed`, `impression_time_fixed`, `read_time_fixed`, `scroll_percentage_fixed` | Q1 |
| 4 | Demo/small have a test set | They have **only `train/` and `validation/`**. Codabench needs `ebnerd_testset.zip` (separate download) | **Q5 — blocker** |
| 5 | Recall@K unqualified | BM25/ANN retrieve from a 100K+ catalog while ground truth sits inside a ~20-item impression. Naive recall@200 ≈ near-zero | **Q2, Q3 — biggest trap** |
| 6 | `rank_bm25` for scoring | `BM25Okapi.get_scores` is a pure-Python loop over all docs, called once per impression → tens of hours on MIND | **Q2 — feasibility blocker** |
| 7 | Q9 anti-gaming = leakage test only | Q9 has **two** clauses; "metrics with and without serving-unavailable features" was missing entirely, and the dataset hands us the exact features to ablate | **Q9 — graded ("ablation rigour")** |
| 8 | Leakage test compared article publish times | The behaviour-window boundary is about **click times vs. impression time**; also `iterrows` over millions of rows won't run | Q9 |
| 9 | — | `next_read_time`, `next_scroll_percentage`, `total_inviews`, `total_pageviews`, `sentiment_score` are present and are textbook serving-time-unavailable features | Q9 |
| 10 | Both leaderboards use MIND's format | RecSys/EB-NeRD expects **ranks**, zipped; MIND expects **scores**. Different formats | Q5 |
| 11 | nDCG / bootstrap / AUC snippets | Three correctness bugs (ideal-DCG over truncated list; unseeded bootstrap; `roc_auc_score` raises on single-class impressions) | Q4 |

Everything below is the corrected plan.

---

## Verified dataset facts

### EB-NeRD demo (confirmed by reading the parquet files)

```
ebnerd_demo/
├── articles.parquet        11,777 rows
├── train/
│   ├── behaviors.parquet   24,724 rows   2023-05-18 07:00 → 2023-05-25 06:59
│   └── history.parquet      1,590 rows (one per user)  clicks 2023-04-27 → 2023-05-18 06:59
└── validation/
    ├── behaviors.parquet   25,356 rows   2023-05-25 07:00 → 2023-06-01 06:59
    └── history.parquet      1,562 rows (one per user)  clicks ending 2023-05-25 06:59
```

**`articles.parquet` columns**
`article_id:int32`, `title`, `subtitle`, `body`, `last_modified_time`, `premium:bool`,
`published_time:ts`, `image_ids:list<int64>`, `article_type`, `url`,
`ner_clusters:list<str>`, `entity_groups:list<str>`, `topics:list<str>`,
`category:int16`, `subcategory:list<int16>`, `category_str`,
`total_inviews:int32`, `total_pageviews:int32`, `total_read_time:float`,
`sentiment_score:float`, `sentiment_label`

**`behaviors.parquet` columns**
`impression_id:uint32`, `article_id:int32` (context article, nullable), `impression_time:ts`,
`read_time:float`, `scroll_percentage:float`, `device_type:int8`,
`article_ids_inview:list<int32>` ← **candidates**, `article_ids_clicked:list<int32>` ← **labels**,
`user_id:uint32`, `is_sso_user`, `gender`, `postcode`, `age`, `is_subscriber`,
`session_id:uint32`, `next_read_time:float`, `next_scroll_percentage:float`

**`history.parquet` columns**
`user_id:uint32`, `impression_time_fixed:list<ts>`, `scroll_percentage_fixed:list<float>`,
`article_id_fixed:list<int32>`, `read_time_fixed:list<float>`

> [!IMPORTANT]
> **The behaviour-window boundary is already baked into the dataset.** Each split ships its *own*
> history file whose click timestamps end exactly where that split's impressions begin
> (train history ends 05-18 06:59:51; train impressions begin 05-18 07:00:03). This is precisely
> the invariant Q9 asks you to assert, and it gives you a free, exact, runnable test.
> It also means: **do not naively re-split by time** — if you move a boundary you must rebuild
> the history window for the new boundary, or you introduce leakage.

**`Ekstra_Bladet_word2vec.zip`** → `Ekstra_Bladet_word2vec/document_vector.parquet` (~150 MB).
One row per `article_id` with a document vector. Verify dimensionality and coverage against
`articles.parquet` on first load — do not assume every article has a vector.

### MIND-small

```
MINDsmall_train/  behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec
MINDsmall_dev/    behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec
```

- `news.tsv` (no header, tab-separated): `news_id`, `category`, `subcategory`, `title`,
  `abstract`, `url`, `title_entities` (JSON), `abstract_entities` (JSON)
- `behaviors.tsv` (no header): `impression_id`, `user_id`, `time`, `history` (space-separated
  news_ids, **may be empty** — these are the cold-start users), `impressions`
  (space-separated `newsid-label`)
- `time` is a US-format string like `11/11/2019 9:05:58 AM` — parse with an explicit format
  string, never `dayfirst` inference.
- MINDsmall_train and MINDsmall_dev are already **time-disjoint** (train precedes dev).
- **MIND has no body text** — `title + abstract` only, which is exactly what Q2 asks for.

> [!NOTE]
> **Verify these three things before writing loaders** (I did not confirm them from the files):
> exact column count of `news.tsv`, the observed time ranges of train/dev, and whether
> `MINDsmall_dev` labels are all present (they are, for MIND-small — unlike MINDlarge_test).

---

## Repository layout

```
retrival-systems/
├── config/
│   └── default.yaml             # all hyperparameters + seeds in one place
├── data/                        # gitignored
│   ├── raw/                     # zips + extracted originals
│   ├── interim/                 # unified-schema parquet
│   └── splits/{mind,ebnerd}/{train,val,test}/
├── feature_store/               # gitignored
│   ├── {mind,ebnerd}/articles.parquet
│   ├── {mind,ebnerd}/users.parquet
│   ├── {mind,ebnerd}/embeddings.npy + article_ids.npy
│   ├── {mind,ebnerd}/bm25.npz          # sparse CSR index
│   └── {mind,ebnerd}/faiss.index
├── src/
│   ├── config.py                # load yaml, seed everything
│   ├── pipeline/
│   │   ├── download.py          # Q1.1 — idempotent, checksummed
│   │   ├── clean_mind.py        # Q1.2
│   │   ├── clean_ebnerd.py      # Q1.2
│   │   ├── split.py             # Q1.3 — temporal
│   │   └── feature_store.py     # Q1.4
│   ├── retrieval/
│   │   ├── tokenize.py          # shared, language-aware
│   │   ├── bm25.py              # Q2 — sparse inverted index
│   │   ├── semantic.py          # Q3 — embeddings + FAISS
│   │   └── candidates.py        # eligibility / recall@K driver
│   ├── eval/
│   │   ├── metrics.py           # Q4.1–4.2
│   │   ├── slicing.py           # Q4.3
│   │   ├── bootstrap.py         # Q4.4
│   │   └── run_eval.py          # Q4.5 — produces results/*.json
│   ├── ablation/
│   │   └── serving_features.py  # Q9.1 — with/without leaky features
│   └── submission/
│       ├── mind.py              # Q5 — scores format
│       └── ebnerd.py            # Q5 — ranks format, zipped
├── tests/
│   ├── test_no_leakage.py       # Q9.2 — REQUIRED
│   ├── test_split_disjoint.py
│   └── test_metrics.py          # hand-checked metric values
├── results/                     # small JSON/CSV — COMMITTED
├── predictions/                 # submission files — COMMITTED (they're small)
├── docs/
│   ├── design_note.pdf          # Q6 deliverable
│   ├── leaderboard/             # Q7.3 screenshots
│   └── ai_usage_log.md          # Q7.4 — REQUIRED, easy to forget
├── build_pipeline.py            # Q1.5 — one-command entry point
├── Makefile
├── requirements.txt
└── README.md
```

> [!IMPORTANT]
> **Housekeeping in the existing repo:**
> - The five `.zip` files currently sit at repo root. Move them to `data/raw/`. `.gitignore`
>   already has `*.zip`, so they are untracked — good.
> - Add `feature_store/`, `results/*.tmp`, `.venv/`, `*.npy`, `*.index`, `.ipynb_checkpoints/`
>   to `.gitignore`. The assignment names `*.zip *.pt *.ckpt __pycache__/ data/` as the floor,
>   not the ceiling.
> - `SPEC.md` in the repo is a **stale copy of plan v1**. Replace it with this file or delete it —
>   two diverging plans in one repo is a self-inflicted wound.
> - `README.md` is currently empty and staged. It is a graded deliverable (Q7.1).

---

## Q1 — Reproducible Data Pipeline

### Requirements
1. Download raw files for MIND-small **and** EB-NeRD demo/small
2. Clean + parse into a unified schema (articles, behaviors/impressions, click history)
3. Temporal train/val/test split — **never random**
4. Feature store: article features (title, abstract, body, category, entities, embeddings) +
   user features (click history, recency)
5. One-command rebuild from raw files

### 1.1 Download

`download.py` must be idempotent and must actually be able to run from a clean checkout —
"I already have the zips" does not satisfy "one command rebuilds everything from raw files".

```python
FILES = {
  "ebnerd_demo.zip":    "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
  "ebnerd_small.zip":   "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
  "ebnerd_testset.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
  "Ekstra_Bladet_word2vec.zip":
      "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
  "MINDsmall_train.zip":
      "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
  "MINDsmall_dev.zip":
      "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
}
```
Skip if present and the SHA256 matches a recorded manifest; otherwise fetch with resume.
Record sizes + hashes in `data/raw/MANIFEST.json` and commit that manifest (it's tiny and it's
what makes the build verifiable).

> [!NOTE]
> **`ebnerd_testset.zip` URL confirmed** (2026-08-20, `curl -I`): `HTTP/1.1 200`,
> `Content-Length: 1631004285` (1.5GB, matches the Codabench page's stated size), direct S3
> download, no registration wall. The URL in `FILES` above is correct and downloadable as-is.

> [!CAUTION]
> **`yjw1029/MIND` is a GATED HuggingFace dataset** (confirmed 2026-08-20: `curl -I` on the
> zip URL returns `401`, `x-error-code: GatedRepo`). Two separate things are required, and
> `hf auth login` alone satisfies neither:
> 1. Visit https://huggingface.co/datasets/yjw1029/MIND and click "Agree and access
>    repository" with the same account used to log in — access approval isn't instant login,
>    and can take a few minutes to propagate.
> 2. The download request needs an actual `Authorization: Bearer <token>` header attached.
>    Plain `urllib.request.urlopen(url)` never sends this on its own even when logged in via
>    the CLI — this is what caused the first real 401 on `ada` (2026-08-20), not a missing
>    login. `download.py`'s `_hf_token()` reads `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env vars
>    or `~/.cache/huggingface/token` (where `hf auth login` writes it) and attaches it as a
>    Bearer header for any `huggingface.co` URL. `download_file` also raises a specific,
>    actionable `RuntimeError` on a 401 against a HF URL rather than a bare traceback.

Also unzip with `__MACOSX/` excluded — every EB-NeRD zip carries AppleDouble junk.

### 1.2 Unified schema

Keep the two datasets in *separate* files with *identical column names*. Do not concatenate
them into one table — the ID spaces collide (MIND `N12345` vs EB-NeRD `int32`) and every
downstream index would need a dataset filter.

```python
# articles/<dataset>.parquet
article_id:      str          # cast EB-NeRD int32 -> str for a uniform key
title:           str
abstract:        str          # MIND: abstract;  EB-NeRD: subtitle
body:            str          # MIND: "" (absent);  EB-NeRD: body
category:        str          # MIND: category;  EB-NeRD: category_str
subcategory:     str
entities:        list[str]    # MIND: title_entities+abstract_entities WikidataId/label
                              # EB-NeRD: ner_clusters
published_time:  datetime|null  # MIND has none -> null
# --- serving-time-UNAVAILABLE, quarantined for the Q9 ablation, never used by default ---
total_inviews:   int|null
total_pageviews: int|null
total_read_time: float|null
sentiment_score: float|null

# impressions/<dataset>/<split>.parquet
impression_id:   str
user_id:         str
timestamp:       datetime
candidates:      list[str]    # MIND: impressions ids;  EB-NeRD: article_ids_inview
labels:          list[int]    # aligned 1:1 with candidates
# --- serving-time-UNAVAILABLE ---
next_read_time:  float|null   # EB-NeRD only
next_scroll_pct: float|null   # EB-NeRD only

# users/<dataset>/<split>.parquet   (one row per user per split)
user_id:         str
history_ids:     list[str]    # chronological, oldest first
history_times:   list[datetime]|null   # EB-NeRD: impression_time_fixed; MIND: null
history_len:     int          # -> cold-start slice
```

**Parsing notes that will bite you:**
- MIND `history` is a *space-separated string* and is **empty for a real fraction of rows**.
  Empty history ⇒ no BM25 query and no user embedding. Decide and **document** the fallback
  (recommendation: fall back to a popularity prior, and report those impressions as their own
  slice — this *is* your cold-start slice).
- MIND has one history string per behaviour row, with no timestamps. EB-NeRD has one history
  row per user with timestamps. Normalise to the same shape; carry `history_times` as null for
  MIND and make every consumer null-safe.
- EB-NeRD `article_ids_clicked` may contain >1 clicked article per impression. `labels` must
  be built by set-membership against `article_ids_inview`, not by positional assumption.
- Occasionally a clicked id is not in `article_ids_inview`. Count these, log the count, and
  drop the impression. Silent dropping is the kind of thing that makes a pipeline unreproducible.
- EB-NeRD `category`/`subcategory` are integer codes; use `category_str` for the human-readable
  value (needed for intra-list diversity in Q4).

### 1.3 Temporal split

> [!IMPORTANT]
> Never random. And for EB-NeRD, never a naive global re-split either — history windows are
> tied to split boundaries (see the verified-facts box above).

**MIND-small** — train and dev are already time-disjoint.
- `train` = `MINDsmall_train` minus its last day
- `val`   = last day of `MINDsmall_train`
- `test`  = all of `MINDsmall_dev`

This keeps a real temporal gap between val and test and mirrors the leaderboard setup.

**EB-NeRD demo/small** — the provided `train/` and `validation/` are already a clean
temporal partition with matched history windows.
- `train` = provided `train/` (impressions 05-18 → 05-25, history 04-27 → 05-18)
- `val`   = provided `validation/`, impressions **05-25 07:00 → 05-29 07:00**
- `test`  = provided `validation/`, impressions **05-29 07:00 → 06-01 07:00**

Splitting *within* the validation week is safe because both halves share the same history file,
whose window ends 05-25 — strictly before both. This preserves the boundary invariant for free.
The true Codabench test set is `ebnerd_testset.zip` and is used only for Q5, never for
offline metrics.

Emit a `splits/<dataset>/split_manifest.json` recording, per split: row count, user count,
min/max impression time, and min/max history click time. This manifest is what
`test_no_leakage.py` and the design note both read from.

**Alternatives considered** (for the design note):
| Option | Verdict |
|---|---|
| Fixed-day boundary (chosen) | Interpretable, matches production and the leaderboard; boundary is a documented constant |
| Ratio-based (last 10% by time) | Works with uneven date ranges, but the boundary drifts if upstream data changes → less reproducible |
| Leave-last-impression-per-user | Creates a *non-contiguous* test set; every test impression has a different cutoff, so a global catalog eligibility filter becomes impossible. Rejected |

### 1.4 Feature store

| Artifact | Format | Rebuild trigger |
|---|---|---|
| Article text/meta | Parquet, one file per dataset | source hash change |
| Article embeddings | `embeddings.npy` (float32, L2-normalised) + `article_ids.npy` (row→id) | model or corpus change |
| BM25 index | `bm25.npz` — scipy CSR + vocab pickle | tokenizer or corpus change |
| ANN index | `faiss.index` | embeddings change |
| User features | Parquet per split | split change |

Plain Parquet + `.npy` + FAISS binaries. SQLite/HDF5 buy nothing at this scale and cost
portability. Every artifact gets a sidecar `.meta.json` with the config hash it was built from;
`build_pipeline.py --force` ignores them and rebuilds.

### 1.5 One-command rebuild

```makefile
.PHONY: all data retrieval eval ablation submit test clean
all: data retrieval eval ablation submit

data:        ; python build_pipeline.py --datasets mind ebnerd --config config/default.yaml
retrieval:   ; python -m src.retrieval.run_bm25_eval     --datasets mind ebnerd
             ; python -m src.retrieval.build_embeddings  --datasets mind ebnerd
             ; python -m src.retrieval.run_semantic_eval --datasets mind ebnerd
eval:        ; python -m src.eval.run_eval      --datasets mind ebnerd --split test
ablation:    ; python -m src.ablation.serving_features --datasets mind ebnerd
submit:      ; python -m src.submission.mind ; python -m src.submission.ebnerd
test:        ; pytest -q tests/
clean:       ; rm -rf data/interim data/splits feature_store results
```

README must state literally: `make all` reproduces every number in the design note.
Pin every dependency version and set `PYTHONHASHSEED`, `numpy` and `torch` seeds from
`config/default.yaml`.

---

## Q2 — Lexical Candidate Generation (BM25)

### Requirements
1. Build an **inverted index** over article text (title + abstract)
2. Construct a query from the user's click history
3. Retrieve top-K candidates by BM25
4. Report **recall@K** for K ∈ {50, 100, 200}

### 2.0 The recall@K trap — resolve this before writing any code

> [!CAUTION]
> This is the single most important correction to v1. **Candidate generation and ranking are
> two different tasks and v1 conflated them.**
>
> - **Q2/Q3 (candidate generation)** retrieve top-K from the *whole article catalog*
>   (11,777 EB-NeRD demo / ~50K MIND). Ground truth = the articles the user actually clicked.
> - **Q4/Q5 (ranking)** score *only the ~20 articles already in the impression's candidate set*
>   and order them. This is what AUC/MRR/nDCG and both leaderboards measure.
>
> If you compute recall@200 over a 50K catalog you will get a small number. **That is the
> correct answer and you should report it**, not quietly swap in the impression set to make the
> number look good. That swap would make recall@200 ≈ 1.0 and mean nothing.

Report recall@K under **two clearly-labelled retrieval pools**:

- **Pool A — full catalog.** Every article, minus those published after the impression
  timestamp, minus articles already in the user's history. The honest upper-funnel number.
- **Pool B — active pool.** Articles that appeared in *any* `article_ids_inview` /impression
  list during the target split. This is a legitimate, standard candidate-generation restriction
  (an article nobody could be shown cannot be clicked) and will give far higher recall.

Two rows per (dataset, method, K) in the results table. The gap between A and B is genuinely
interesting and is good design-note material.

**Eligibility filtering is mandatory for Pool A on EB-NeRD**: `published_time` exists, so
recommending an article published *after* the impression is a leakage bug. MIND has no publish
time — say so explicitly and note it as a known limitation, don't paper over it.

### 2.1 BM25 implementation

> [!CAUTION]
> **Do not use `rank_bm25` for scoring at this scale.** `BM25Okapi.get_scores(q)` is a Python
> loop over all N documents per query. With tens of thousands of evaluation impressions × tens
> of thousands of articles, this runs for hours to days. v1's plan would have stalled here.

**Chosen: hand-rolled sparse BM25 (scipy CSR).** This is *more* faithful to "build an inverted
index" than calling a library, it's ~40 lines, and it's a single sparse matmul per query batch.

```python
# Build once: CSR matrix W of shape (n_docs, vocab), W[d,t] = BM25 doc-side weight
#   idf[t]  = ln(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
#   W[d,t]  = idf[t] * (tf * (k1+1)) / (tf + k1*(1 - b + b*len_d/avg_len))
# Score a batch of Q queries (Q x vocab binary/count matrix) with:  S = Qmat @ W.T
# Top-K via np.argpartition on each row of S.  Batch 512 queries at a time.
```
k1 = 1.5, b = 0.75 (report as chosen constants). The CSR matrix *is* the inverted index — the
`.T` view gives you term→postings directly, which is worth one sentence in the design note.

> [!CAUTION]
> **Cross-machine reproducibility bug, found 2026-08-21: `np.argpartition` is not
> deterministic across numpy/BLAS builds for tied scores.** The original `top_k_indices`
> used `argpartition` to select the top-K, then `argsort` only within that slice.
> `argpartition` makes no guarantee about *which* tied elements land in the partition — for
> a sparse query (e.g. N=1, a single article's title), the vast majority of the catalog
> scores exactly `0.0` (no shared terms), so with K up to 200 there often aren't 200
> nonzero-scoring docs, and which zero-score docs fill the rest is arbitrary and
> platform-dependent. This was caught by literally comparing a local run against the `ada`
> run for the same seed/code/data: EB-NeRD BM25's val-sweep gave meaningfully different
> recall@100/200 at N=1 (e.g. B@200: 0.2665 local vs. 0.1508 on `ada`), which flipped the
> automatic best-N selection (N=1 locally, N=20 on `ada`) and cascaded into different Q4
> eval numbers downstream. **Fixed** with a full stable `np.argsort` (ties broken by
> ascending index, identical on every platform) instead of `argpartition`+partial `argsort`
> — see `bm25.py`'s `top_k_indices` docstring and `tests/test_bm25.py`. The same pattern
> existed in `semantic.py`'s numpy-fallback `faiss_search` and was fixed the same way.
> **Lesson for the design note**: "reproducible" isn't proven by a script exiting 0 on one
> machine — it needs verification across environments, which is exactly what surfaced this.

*Fallbacks if you're short on time:* `bm25s` (fast, sparse, drop-in) is acceptable.
`rank_bm25` is acceptable **only** for a small sanity subsample — use it to assert your sparse
implementation agrees to within float tolerance on 100 queries. That agreement check is itself
a nice test to have in `tests/`.

*Rejected:* Elasticsearch/OpenSearch — needs a running server, breaks one-command reproduce.

### 2.2 Tokenisation

| Dataset | Language | Approach |
|---|---|---|
| MIND | English | lowercase → strip punctuation → whitespace split → NLTK English stopwords → optional Porter stemming |
| EB-NeRD | **Danish** | lowercase → strip punctuation (keep `æøå`!) → whitespace split → NLTK Danish stopwords → optional Snowball Danish stemmer |

> [!NOTE]
> Unicode-normalise (NFC) and use a `\w` regex with `re.UNICODE`, not `str.split()` — a naive
> ASCII strip will mangle `æ ø å` and silently wreck Danish retrieval. This is a concrete,
> reportable dataset-difference observation for Q6.

Ship the stemming on/off comparison as a cheap ablation — it's one config flag and it feeds
"ablation rigour" in the grading criteria.

### 2.3 Query construction

**Chosen: concatenate the titles of the last N=5 clicked articles.**
Run N ∈ {1, 5, 10, 20} as an ablation on the *validation* split, pick the best, and report the
sweep. Fix N before ever touching test.

Empty history (MIND, and any user with no clicks) ⇒ no query. Fall back to a popularity prior
computed **only from the train split** and flag those impressions as the cold-start slice.

*Alternatives:* TF-IDF-weighted or recency-weighted term concatenation (worth one ablation row);
last-click-only (loses history signal — but is a useful baseline showing how much history helps).

### 2.4 Recall@K reporting

Per impression: `recall@K = |clicked ∩ topK| / |clicked|`, then average over impressions.
Report **macro** (mean of per-impression recall) as the headline; note that micro
(total hits / total clicked) differs when impressions have different click counts, and pick one
consistently. Report for K ∈ {50, 100, 200} × {MIND, EB-NeRD} × {Pool A, Pool B}, each with a
bootstrap 95% CI from Q4.4.

---

## Q3 — Semantic Candidate Generation (Embeddings)

### Requirements
1. Compute or load article embeddings for both datasets
2. Build an ANN index (FAISS/ScaNN/brute force)
3. User representation from click history → retrieve top-K
4. Report recall@K for K ∈ {50, 100, 200}
5. **Compare lexical vs. semantic — which is better, on which slices?**

### 3.1 Embeddings

> [!IMPORTANT]
> **Decision (2026-08-20, explicit user instruction): use PROVIDED embeddings only, for both
> datasets. Do not train or run a model (no `sentence-transformers`, no BERT/XLM-RoBERTa) —
> too costly/time-consuming for this assignment's scope.** This changes the original plan
> below for MIND, since MIND has no provided per-article embedding the way EB-NeRD does.

| Dataset | Chosen | Why | Coverage (verified 2026-08-20) |
|---|---|---|---|
| EB-NeRD | Provided `Ekstra_Bladet_word2vec/document_vector.parquet` (300-dim) | Danish-native, exactly what the challenge ships, zero compute | **100%** of `ebnerd_demo`'s catalog |
| MIND | Provided `entity_embedding.vec` (100-dim TransE), mean-pooled per article over its `title_entities`+`abstract_entities` WikidataIds | The only provided embedding MIND ships; zero compute, zero training | **86.7%** of articles have ≥1 entity (44484/51282 in MINDsmall_train); of mentioned WikidataIds, 97% have a vector |

**MIND's coverage gap is real and must be reported, not hidden**: ~13-14% of MIND articles
have no matched entity vector (no entities mentioned, or entities missing from the .vec file)
and are **excluded from the semantic retrievable catalog entirely** — narrower than BM25's
catalog, which covers every article with any title/abstract text. This asymmetry (EB-NeRD:
~universal coverage at document granularity; MIND: partial coverage at entity granularity) is
itself a legitimate Q6 finding about what "provided embeddings" costs you, not a limitation to
paper over. `build_embeddings.py` logs exact coverage counts; `run_semantic_eval.py` carries
`embedding_meta` (coverage %) into every results file so recall@K numbers are never read
without that context.

~~Original plan (superseded): `sentence-transformers/all-MiniLM-L6-v2` for MIND,
`paraphrase-multilingual-MiniLM-L12-v2` as an EB-NeRD contrast run, TF-IDF+SVD baseline for
both.~~ Not implemented per the instruction above. If time permits later, the MiniLM contrast
run remains a good design-note addition (isolates language effect vs. model-quality effect on
the lexical/semantic gap) — but it is not blocking and was explicitly deprioritized.

Verify embedding **coverage**: articles with no vector are dropped from the index and the
count is reported (`build_embeddings.py`), never silently zero-filled — a zero vector is
maximally similar to nothing and would quietly distort every cosine score.

L2-normalise all embeddings at build time; store as float32
(`feature_store/<dataset>/embeddings.npy` + `embeddings_ids.npy` + `embeddings.meta.json`).

### 3.2 ANN index

**Chosen: FAISS `IndexFlatIP`** over L2-normalised vectors (= exact cosine). At 12K–50K
articles exact search is milliseconds and removes recall-loss-from-approximation as a
confound. Build `IndexHNSWFlat` as a **second** index and report the recall-vs-latency
tradeoff against the exact index — that measured tradeoff is your 10× scale evidence for Q6,
and it's much stronger than speculating about it.

*Alternatives:* ScaNN (better Pareto frontier, heavier install); brute-force numpy (fine, but
FAISS is the same effort and scales).

### 3.3 User representation

**Chosen: mean-pool of the last N=5 clicked article embeddings, re-normalised.**

```python
V = emb[[idx[a] for a in history[-N:] if a in idx]]
if len(V) == 0: fallback_to_popularity_prior()
u = V.mean(axis=0); u /= (np.linalg.norm(u) + 1e-12)
```

Sweep N ∈ {1, 5, 10, 20} on validation, same as BM25, so the two methods are compared at their
own best N rather than at an arbitrary shared one.

*Ablation (recommended, cheap):* exponential recency decay
`w_i = exp(-λ·Δt)` using EB-NeRD's `impression_time_fixed`. EB-NeRD has the timestamps and MIND
does not — so this ablation runs on EB-NeRD only, and *that asymmetry is itself a reportable
dataset difference.*

### 3.4 Lexical vs. semantic comparison (Q3.5 — explicitly graded)

Same test split, same pools, same K values, same CIs. Break out by:
- **cold-start (history_len < 5) vs. warm (≥ 5)** — required
- **head (top-10% of articles by train-split click count) vs. tail**
- **MIND vs. EB-NeRD** (English vs. Danish; abstract vs. subtitle)

Hypotheses to state up front and then test (state them *before* seeing results; being wrong
is fine and reporting it is worth marks):
- BM25 wins on named-entity/breaking-news queries where exact term match matters.
- Semantic wins on topical drift, paraphrase, and Danish morphology (compounding hurts BM25).
- Semantic degrades harder on cold-start (a 1-article mean-pool is a noisy user vector).
- Both collapse on the tail; BM25's tail recall should be less bad than semantic's.

Add a **hybrid** row: normalise both score lists per impression (z-score or min-max) and fuse
with `α·BM25 + (1−α)·cosine`, α swept on validation. Reciprocal Rank Fusion is a good
alternative that needs no score normalisation. Not strictly required, but it's the natural
"so what" of the comparison and it's a couple dozen lines.

---

## Q4 — Offline Evaluation Harness

### Requirements
1. Metrics: AUC, MRR, nDCG@5, nDCG@10
2. Beyond-accuracy: intra-list diversity, novelty, coverage
3. At least one slice
4. Bootstrap 95% CI for each metric
5. **Run on both BM25 and embedding-based results**

### 4.0 What gets scored

Per impression, score every article in `candidates` (the inview list), sort descending, and
evaluate against `labels`. BM25 → score each candidate against the history query; semantic →
cosine(user vector, article vector). This is the ranking task, distinct from Q2/Q3 recall.
Ties must be broken deterministically (by article_id) or your numbers won't reproduce.

### 4.1 Accuracy metrics — with the v1 bugs fixed

```python
def dcg(labels):                       # labels ordered by predicted score
    return sum(l / np.log2(i + 2) for i, l in enumerate(labels))

def ndcg_at_k(labels, k):
    """BUG FIX (v1): the ideal ranking must come from the FULL label list, not labels[:k]."""
    ideal = sorted(labels, reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(labels[:k]) / idcg if idcg > 0 else 0.0

def mrr(labels):
    for i, l in enumerate(labels, 1):
        if l == 1:
            return 1.0 / i
    return 0.0

def impression_auc(scores, labels):
    """BUG FIX (v1): roc_auc_score RAISES on single-class impressions.
    A large share of impressions have exactly one positive and some have zero/all — skip them
    and report how many were skipped."""
    if len(set(labels)) < 2:
        return None
    return roc_auc_score(labels, scores)
```

> [!NOTE]
> AUC is computed **per impression and then averaged** (standard MIND protocol) — never as one
> global pooled AUC. Log the count of skipped single-class impressions in the results JSON;
> an unreported skip rate silently changes what the mean is over.

Validate all four metrics against hand-computed values in `tests/test_metrics.py`
(e.g. labels `[0,1,0]` with descending scores → MRR = 0.5, nDCG@5 = 1/log2(3) ≈ 0.6309).
This costs ten minutes and is the difference between "correct" and "plausible."

### 4.2 Beyond-accuracy metrics

Define the evaluated list explicitly: **the top-10 of the ranked candidate set** per impression.
State the cutoff — these metrics are meaningless without one.

**Intra-list diversity (ILD)** — mean pairwise distance within the top-10.
Report **two variants**, because they answer different questions:
- *Categorical*: fraction of pairs with differing `category` (v1's version — keep it, it's
  interpretable and works on both datasets).
- *Embedding*: mean pairwise cosine **distance** `1 − cos(e_i, e_j)`. This is the standard ILD
  and it's the one that reveals semantic retrieval's known redundancy problem — expect semantic
  to lose here, which is a genuinely interesting result to report against its recall win.

**Novelty** — mean self-information over the top-10:
`novelty = mean(-log2(clicks[a] / total_clicks))`, with popularity counted **on the train split
only** (using test-split popularity is leakage). Apply add-one smoothing for unseen articles
and say so.

**Coverage** — `|union of all top-10 lists| / |eligible catalog|`. Fix the denominator per
dataset and per pool and state it; coverage over Pool A and over Pool B are different numbers
and mixing them is meaningless.

### 4.3 Slicing

| Slice | Definition | Notes |
|---|---|---|
| Cold-start vs. warm | `history_len < 5` vs. `≥ 5` | **Required.** MIND has genuinely empty histories; report `history_len == 0` as its own bucket |
| Head vs. tail articles | Article in top-10% by **train-split** click count | Impression-level assignment: slice by whether the *clicked* article is head or tail |
| Dataset | MIND vs. EB-NeRD | Free — you're running both anyway |
| Session position | EB-NeRD `session_id`: first impression in session vs. later | Optional; EB-NeRD only |

Report every metric × every method × every slice, each with a CI. Include the **n per slice** in
every table — a slice with 40 impressions and a huge CI must not be read as a finding.

### 4.4 Bootstrap CIs — with the v1 bugs fixed

```python
def bootstrap_ci(per_impression_values, n_boot=1000, ci=0.95, seed=42):
    """BUG FIX (v1): unseeded -> not reproducible. Also: resample INDICES of
    per-impression metric values, then aggregate; don't resample raw records."""
    rng = np.random.default_rng(seed)          # seeded
    x = np.asarray([v for v in per_impression_values if v is not None], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100*(1-ci)/2, 100*(1+ci)/2])
    return float(x.mean()), float(lo), float(hi)
```
1000 resamples, seed in config, report as `mean [lo, hi]`. Bootstrap over **impressions**
(the independent unit), not over individual candidates.

### 4.5 Output contract

`run_eval.py` writes `results/{dataset}_{method}_{split}.json` with every metric × slice ×
CI × n, plus the config hash. The design note's tables are **generated from these JSONs**, never
retyped by hand — retyped numbers drift from the code and that's exactly the reproducibility
failure the assignment is testing for.

---

## Q5 — Codabench Submission

### Requirements
- MIND predictions → https://www.codabench.org/competitions/13967/
- EB-NeRD predictions → https://www.codabench.org/competitions/2469/
- Leaderboard **screenshots** in the design note

> [!NOTE]
> **Both formats confirmed** (2026-08-20). Registered on both competitions; read the official
> submission guidelines for each and unzipped the MIND sample file (`sample/prediction.zip`) to
> check it. The two formats are near-identical but not quite — see the exact contracts below.

> [!NOTE]
> **Implemented 2026-08-21** (`src/submission/{format,mind,ebnerd}.py`). Two decisions worth
> recording: (1) MIND's submission reuses `data/splits/mind/test` (== all of MINDsmall_dev)
> since the assignment gives no separate MIND test file — if Codabench rejects it for
> missing/unexpected impression_ids, that would mean a different file (e.g. MINDlarge_test)
> is actually expected. (2) EB-NeRD's submission builds a **fresh** BM25 index over
> `ebnerd_testset`'s own article catalog rather than reusing `feature_store/ebnerd/bm25` —
> the test articles are almost certainly disjoint from demo/small's (different time window),
> and scoring against the wrong catalog would silently produce near-zero-coverage garbage.
> `ebnerd_testset.zip`'s directory structure was confirmed via a remote HTTP-range listing of
> the zip's central directory (no 1.5GB download needed for that) — and this caught the
> **same double-nesting bug as the original MIND zips** before it ever ran: unlike
> `ebnerd_demo.zip`/`ebnerd_small.zip`, `ebnerd_testset.zip`'s members ARE wrapped in an
> `ebnerd_testset/` folder. The actual column *schemas* inside (`articles.parquet`,
> `test/history.parquet`, `test/behaviors.parquet`) could not be verified the same way (each
> member is DEFLATE-compressed, so a byte-range read can't land on a coherent parquet footer)
> — `ebnerd.py` asserts required columns explicitly and fails loudly if the demo/small-schema
> assumption is wrong, rather than guessing. **Not yet verified against the real file.**

**MIND** (verified from official guidelines + inspected `sample/prediction.zip`): zip containing
exactly one file, `prediction.txt`, at the zip root (**no folder, no `__MACOSX`**). One line per
impression:
```
ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
```
Ranks are continuous integers 1..N (1 = most likely clicked), listed **in the original candidate
order from `behaviors.tsv`**, comma-separated with **no spaces** inside the brackets, e.g.
`24481 [4,1,3,2]`. Confirmed sample line: `3 [6,2,5,3,1,4,7,8]`.
**Row order must match the original impressions file — do not sort or shuffle.**
**Rate limit: at most 1 submission per day.**

**EB-NeRD / RecSys 2024** (verified from official guidelines): zip containing exactly one file,
`predictions.txt`, at the zip root (no folder, no `__MACOSX`). One line per impression:
```
impression_id [rank_1,rank_2,...,rank_n]
```
Ranks are continuous integers 1..n aligned to `article_ids_inview` in its original order, same
comma-format as MIND. Example given in the guidelines: `139350 [3,2,4,1]` for
`article_ids_inview = [9798759, 9798604, 9777339, 9798829]` means article `9798829` is ranked
1st. Scored on AUC/MRR/nDCG; **only 50% of the test set is scored during the competition**
(same fixed seed per submission — expect `auc*`-style marked metrics until the deadline).
**Rate limit: at most 5 submissions per day, and scoring can take hours** — budget for this,
don't submit for the first time on the last day. Zip name itself can be anything
(`GIVE_ME_A_NAME.zip`); only the internal filename (`predictions.txt`) is fixed.

```python
def to_ranks(scores):
    """Higher score -> rank 1. Deterministic tie-break by original candidate-list position."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for r, i in enumerate(order, 1):
        ranks[i] = r
    return ranks

def write_submission(path_txt, rows):
    """rows: list of (impression_id, ranks) in ORIGINAL impression-file order."""
    with open(path_txt, "w") as f:
        for imp_id, ranks in rows:
            f.write(f"{imp_id} [{','.join(str(r) for r in ranks)}]\n")
    # zip path_txt alone at the archive root — no parent folder, no __MACOSX
```

**Coverage requirement:** the submission must contain **every** impression id in the test file,
in the file's original order. Any impression your model can't score (empty history, unseen
articles) still needs a row — emit the popularity-prior ranking. Assert
`submitted_ids == test_ids_in_order` (not just set-equality) before zipping; a missing or
reordered row is graded as a malformed submission, not a lower score.

**EB-NeRD test set:** `ebnerd_testset.zip` (1.5GB, URL confirmed live — see §1.1), no labels.
Its `article_ids_inview` lists are larger than in demo/small, and scoring is asynchronous and
slow server-side — submit early, not last-minute.

Given the 1/day (MIND) and 5/day (EB-NeRD) caps, don't burn attempts on debugging format issues
on the real endpoint: validate structurally offline first (row count, id coverage, rank
permutation validity per line) before every upload. Submit BM25 and the best semantic/hybrid run
if attempts allow; screenshot each and record the exact commit SHA that produced each submission
file in `predictions/README.md`. Grading is never on rank — but "which of my systems the
leaderboard preferred, and did it agree with my offline harness?" is a strong design-note
paragraph. Offline/online disagreement is a finding, not a failure.

---

## Q6 — Design Note (≤ 4 pages)

### Structure (page budget)

```
1. Introduction & task framing                              0.25 pp
2. Data pipeline: unified schema, temporal split,
   behaviour-window boundary, feature store                 0.60 pp
3. Lexical retrieval: sparse BM25 inverted index, Danish
   tokenisation, query construction, Recall@K (both pools)  0.75 pp
4. Semantic retrieval: embedding choices, FAISS exact vs.
   HNSW, user representation, Recall@K                      0.75 pp
5. Lexical vs. semantic: slice-level comparison + hybrid    0.50 pp
6. Evaluation harness: metrics, beyond-accuracy, CIs,
   Q9 serving-feature ablation, leaderboard screenshots     0.60 pp
7. Where it breaks at 10x                                   0.40 pp
8. Limitations + references                                 0.15 pp
```

### Design choices table (fill from actual results)

| Choice | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Temporal split | Fixed-day boundary; EB-NeRD's native windows preserved | Ratio-based; leave-last-out | Reproducible; keeps history/impression boundary intact |
| BM25 | Hand-rolled scipy CSR | `rank_bm25`, `bm25s`, Elasticsearch | `rank_bm25` is O(N) Python per query — infeasible; ES breaks one-command repro |
| Tokenisation | Language-specific + Danish stemmer | Shared English pipeline | `æøå` and Danish compounding |
| MIND embeddings | Provided `entity_embedding.vec`, mean-pooled per article | all-MiniLM-L6-v2, TF-IDF+SVD | No training/compute (explicit user instruction); only provided option MIND ships — costs coverage (86.7%) |
| EB-NeRD embeddings | Provided Word2Vec `document_vector.parquet` | Multilingual MiniLM, XLM-R fine-tune | Native Danish, free, 100% coverage, zero compute |
| ANN | FAISS IndexFlatIP (numpy brute-force fallback if unavailable) | ScaNN, HNSW | Exact at this scale removes a confound; HNSW comparison deprioritized alongside the embedding-model contrast run |
| User repr | Mean-pool last N (N swept on val) | Recency decay, last-click | Simple, strong; decay reported as ablation |

### "Where it breaks at 10×" — ground this in measurements, not adjectives

Grading explicitly names **scale analysis**. Measure, then extrapolate. Take actual timings and
peak RSS from your runs (demo → small is already ~2× articles and ~8× behaviours, so you have
two real points to extrapolate from — use them):

| Component | Breaks because | Evidence to cite | Fix at 10× |
|---|---|---|---|
| BM25 CSR matrix | nnz grows linearly; dense score row is `n_docs × float64` per query | measured nnz + peak RSS on demo vs. small | Block over doc shards; top-K merge; or a real inverted index with WAND/BlockMax skipping |
| Per-impression scoring | Python loop over impressions dominates | measured impressions/sec | Vectorise into batched matmuls; multiprocess by impression shard |
| FAISS IndexFlatIP | Exact search is O(N·d) per query; index is `N×d×4` bytes | measured index size + QPS | IVF-PQ / HNSW — you already have the HNSW recall-vs-latency curve |
| Embedding computation | One-off but linear in corpus | measured articles/sec | GPU batching; incremental — embed only new articles |
| Article catalog in RAM | Whole corpus held as pandas | measured RSS | Parquet row-group streaming; column pruning |
| Cold-start / news churn | Embeddings stale for articles published after index build | count of test-split articles absent from the train-split index | Incremental index updates; content-based cold-start path |

Also address the *data* dimension: EB-NeRD full is ~600M impressions vs. demo's 25K — 4 orders
of magnitude. Say which components survive that and which need to be replaced outright, not
merely tuned. Being specific about what you'd *throw away* reads far better than "we would add
more sharding."

---

## Q7–Q9 — Deliverables & Policies (do not treat as an afterthought)

### Q7 Deliverables
- [ ] **Code** on GitHub Classroom: pipeline, retrieval, eval harness, prediction files,
      `README.md` with a one-command reproduce
- [ ] **Design note** ≤ 4 pages → Moodle (PDF; check the page count *after* adding figures)
- [ ] **Leaderboard screenshots** from *both* Codabench competitions → `docs/leaderboard/`,
      embedded in the design note
- [ ] **AI usage log** → `docs/ai_usage_log.md`: all prompts, chat-history exports, and an
      explicit per-file/per-function marking of AI-generated vs. human-written code

> [!IMPORTANT]
> **Start `docs/ai_usage_log.md` now and append as you go.** It asks for *all prompts and chat
> history exports*. Reconstructing that on day 12 is impossible, and it is an explicitly listed
> deliverable. A simple convention: date-stamped entries with the prompt, the tool, what you kept,
> and what you rewrote — plus a header comment in each generated file marking its provenance.

### Q8 Git commit policy
- Frequent, meaningful commits — commit at each sub-step, not one "final" dump. The commit
  history is visible evidence of process.
- `.gitignore` must cover `*.zip *.pt *.ckpt __pycache__/ data/` plus `feature_store/ *.npy
  *.index .venv/`. Audit with `git count-objects -vH` before pushing.
- **No force-pushes after the deadline.**

### Q9 Anti-gaming — both clauses, and neither is optional

**9.1 "Report metrics with and without features unavailable at serving time."**

> [!CAUTION]
> v1 omitted this clause entirely. It is a required *ablation table*, and "ablation rigour" is
> a named grading criterion. EB-NeRD hands you the perfect concrete instance.

Features present in the data that a live serving system **cannot** have at ranking time:

| Feature | Where | Why unavailable |
|---|---|---|
| `total_inviews`, `total_pageviews`, `total_read_time` | `articles.parquet` | Aggregated over the *whole* corpus period, including the future relative to any given impression |
| `sentiment_score`, `sentiment_label` | `articles.parquet` | Derived offline over the full corpus; usable only if computed at publish time |
| `next_read_time`, `next_scroll_percentage` | `behaviors.parquet` | Describe what happened **after** the impression — direct label leakage |
| `read_time`, `scroll_percentage` | `behaviors.parquet` | Post-interaction outcome of the current impression |

Deliverable: a table with three rows per dataset —
1. **Clean (default)**: content + history only. This is the headline system.
2. **+ popularity features** (`total_inviews`/`total_pageviews` as a re-ranking prior).
3. **+ post-interaction features** (`next_read_time` etc.) — the deliberate "cheating" upper bound.

Expect row 3 to look excellent and be worthless. Reporting that gap *is* the exercise: it
quantifies how much of an apparently strong result is leakage. State plainly that only row 1 is
a deployable system, and that rows 2–3 exist to measure the temptation.

**9.2 "Enforce the behaviour-window boundary — no future-click leakage. Include a test."**

v1's test was not runnable (it compared article publish times, not click times, and used
`iterrows` over millions of rows). Working version:

```python
# tests/test_no_leakage.py
import json, pytest
from src.config import SPLITS   # [(dataset, split), ...]

@pytest.mark.parametrize("dataset,split", SPLITS)
def test_history_precedes_impressions(dataset, split):
    """Q9: every click in a user's history must predate that split's impression window."""
    m = json.load(open(f"data/splits/{dataset}/split_manifest.json"))[split]
    if m["max_history_click_time"] is not None:      # MIND has no history timestamps
        assert m["max_history_click_time"] <= m["min_impression_time"], (
            f"{dataset}/{split}: history extends past the behaviour-window boundary"
        )

@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_splits_are_temporally_disjoint(dataset):
    m = json.load(open(f"data/splits/{dataset}/split_manifest.json"))
    assert m["train"]["max_impression_time"] <= m["val"]["min_impression_time"]
    assert m["val"]["max_impression_time"]   <= m["test"]["min_impression_time"]

def test_no_future_articles_in_candidates():
    """EB-NeRD only: never retrieve an article published after the impression."""
    # vectorised: join top-K candidate ids to published_time, assert <= impression_time
    ...

def test_popularity_prior_uses_train_only():
    """Novelty/popularity stats must not be fitted on val/test."""
    ...
```

Vectorised, runs in seconds, and asserts the actual invariant. The manifest from Q1.3 is what
makes this cheap — which is why Q1.3 emits it.

---

## Phased schedule

```
Day 0  (do this first — unblocks everything)
  - Register on BOTH Codabench competitions; download both starting kits / sample submissions
  - Confirm the ebnerd_testset.zip URL from recsys.eb.dk/dataset/
  - Create docs/ai_usage_log.md and start logging
  - Replace stale SPEC.md; write README skeleton; extend .gitignore

Day 1-2  Q1 — pipeline
  - download.py + MANIFEST.json; clean_mind.py + clean_ebnerd.py -> unified schema
  - split.py + split_manifest.json; feature_store.py
  - tests/test_no_leakage.py + test_split_disjoint.py GREEN  <- gate; do not proceed until green
  - `make data` works from a clean checkout

Day 3-4  Q2 — BM25
  - Sparse CSR index; agreement check vs. rank_bm25 on a 100-query subsample
  - Pool A / Pool B recall@{50,100,200}, both datasets
  - Sweep N and stemming on VALIDATION only

Day 5-6  Q3 — semantic
  - Load EB-NeRD word2vec (check coverage); embed MIND with MiniLM
  - Multilingual MiniLM on EB-NeRD; TF-IDF+SVD baseline
  - FAISS flat + HNSW (record recall-vs-latency); recall@K; hybrid fusion sweep

Day 7-8  Q4 — eval harness
  - Metrics + hand-checked unit tests; beyond-accuracy; seeded bootstrap
  - Slices (cold/warm, head/tail); run over BM25 + semantic + hybrid -> results/*.json

Day 9    Q9.1 ablation + Q5 submissions
  - Serving-feature ablation table (3 rows x 2 datasets)
  - Generate + validate both submission files; submit; screenshot; record commit SHAs

Day 10-11  Q6 — design note
  - Tables auto-generated from results/*.json; scale analysis from measured timings

Day 12   Cleanup
  - README one-command reproduce verified on a FRESH clone
  - .gitignore audit; finalise AI usage log; final commits (no force-push after deadline)
```

The Day 1–2 test gate matters: every number produced after a leakage bug has to be recomputed.

---

## Dependencies

```
numpy>=1.24,<2.0
scipy>=1.11
pandas>=2.0
pyarrow>=14.0
polars>=0.20            # EB-NeRD parquet + list columns; far faster than pandas here
scikit-learn>=1.3       # CountVectorizer for BM25 tokenize/vocab bookkeeping only
faiss-cpu>=1.7.4
nltk>=3.8.1             # English + Danish stopwords, Snowball Danish stemmer
# NOTE: sentence-transformers/torch deliberately NOT included -- embeddings are provided
# (see Q3.1), not computed. Re-add only if the MiniLM contrast run is picked back up.
rank-bm25>=0.2.2        # reference implementation for the correctness check ONLY
tqdm>=4.65
pyyaml>=6.0
pytest>=7.4
matplotlib>=3.7         # design-note figures
```

Pin exact versions in the final `requirements.txt` (`pip freeze`) and record the Python version
in the README. `polars` is not currently installed — either add it or commit to pandas +
pyarrow throughout, but don't mix idioms across the pipeline.

---

## Answers to v1's open questions

1. **Code location** — Yes, inside `retrival-systems/`. It's already the git repo. **Repo stays
   local for now** — do not push to GitHub Classroom until explicitly told to.
2. **EB-NeRD scale** — Develop on **demo** (11.7K articles, 24.7K train impressions — minutes per
   run). Produce **final reported numbers on `ebnerd_small`**, since demo's 1,590 users make
   slice-level CIs uselessly wide. Both are downloaded already. Report which is which in every table.
3. **MIND embeddings** — Yes, MiniLM on ~50K MIND-small articles is ~5–10 min on GPU, ~20–40 min
   on CPU. Cache to `.npy`, gate the rebuild on a config hash, and it's a one-time cost.
4. **Codabench format** — **Resolved** (2026-08-20). See the confirmed §Q5 contracts, based on
   the official guidelines pages and the inspected MIND sample zip.
5. **EB-NeRD starter code** — Read `ebnerd-benchmark` for the schema conventions and the official
   metric implementations (worth cross-checking your AUC/nDCG against), but **write your own
   loaders**. The starter code pulls in a heavy dependency tree and Q1 is graded on *your*
   pipeline. Cite it in the design note as consulted.

---

## Execution environment: remote GPU (`ada`)

Heavy compute (embedding generation, FAISS builds on `small`, full eval sweeps) runs on a
remote GPU node, `ada`, over SSH — not on this machine.

- **Workflow**: develop/edit locally in `retrival-systems/`; `scp`/`rsync` the `src/`, `config/`,
  `tests/`, `Makefile`, `requirements.txt` (code only, never `data/` or `feature_store/`) to
  `ada`; run there; pull back only the small outputs (`results/*.json`, `predictions/*.zip`,
  `feature_store/*.meta.json`) — not the multi-GB indexes/embeddings themselves.
- **Implication for Q1.5 ("one command rebuilds everything")**: the rebuild command must be
  identical on both machines and must not hardcode local paths. Data can either be downloaded
  independently on `ada` (re-run `download.py` there — it's idempotent, see §1.1) or synced
  explicitly; document whichever you pick as an explicit step in `README.md`, since "runs on my
  machine" alone doesn't satisfy reproducibility if grading happens elsewhere.
- **Implication for `docs/ai_usage_log.md`**: if you run interactive tools on `ada` too, note
  which prompts/sessions happened there vs. locally, since `/export` only captures this VS Code
  session.
- I don't have direct SSH access to `ada`. I'll write/edit code locally and give you the exact
  `scp`/`rsync` and remote run commands to execute yourself; tell me `ada`'s hostname/alias and
  whether it's already in your `~/.ssh/config` when we get to Q1.5, or I'll ask then.

---

## Requirements traceability

| # | Assignment requirement | Where addressed | Risk |
|---|---|---|---|
| Q1.1 | Download both datasets | `download.py` + MANIFEST | — (ebnerd_testset URL confirmed live) |
| Q1.2 | Clean → unified schema | §Q1.2 (verified real columns) | — |
| Q1.3 | Temporal split, never random | §Q1.3 + `split_manifest.json` | — |
| Q1.4 | Feature store (articles + users) | §Q1.4 | — |
| Q1.5 | One-command rebuild | `make data` / `build_pipeline.py` | verify on fresh clone |
| Q2.1 | Inverted index over title+abstract | Sparse CSR (§2.1) | — |
| Q2.2 | Query from click history | §2.3, N swept | empty-history fallback needed |
| Q2.3 | Top-K by BM25 | §2.1 | perf — CSR, not rank_bm25 |
| Q2.4 | Recall@K, K∈{50,100,200} | §2.4, Pools A & B | — |
| Q3.1 | Embeddings, both datasets | §3.1 | word2vec coverage unverified |
| Q3.2 | ANN index | FAISS flat + HNSW (§3.2) | — |
| Q3.3 | User repr → top-K | §3.3 | — |
| Q3.4 | Recall@K, K∈{50,100,200} | §2.4 shared harness | — |
| Q3.5 | **Lexical vs. semantic, by slice** | §3.4 | — |
| Q4.1 | AUC, MRR, nDCG@5, nDCG@10 | §4.1 (3 bugs fixed) | single-class AUC handling |
| Q4.2 | Diversity, novelty, coverage | §4.2, top-10 cutoff | define denominators |
| Q4.3 | ≥1 slice | §4.3 (four slices) | per-slice n must be reported |
| Q4.4 | Bootstrap 95% CI | §4.4 (seeded) | — |
| Q4.5 | Run on BM25 **and** embeddings | `run_eval.py`, both + hybrid | — |
| Q5 | Submit to **both** leaderboards | §Q5 | rate limits: 1/day MIND, 5/day EB-NeRD — submit early |
| Q5 | Leaderboard screenshots | `docs/leaderboard/` | — |
| Q6 | Design note ≤ 4 pages | §Q6 page budget | 4 pp is tight — budget early |
| Q7.1 | Code + README one-command | README, Makefile | fresh-clone check |
| Q7.2 | Design note → Moodle | `docs/design_note.pdf` | — |
| Q7.3 | Screenshots, both competitions | `docs/leaderboard/` | — |
| Q7.4 | **AI usage log** | `docs/ai_usage_log.md` | **start Day 0 — unreconstructable later** |
| Q8 | Frequent commits, no large files | per-substep commits, .gitignore | — |
| Q8 | No force-push after deadline | policy | — |
| Q9.1 | **Metrics with/without serving-unavailable features** | §9.1 ablation table | **was missing in v1** |
| Q9.2 | Leakage test | `tests/test_no_leakage.py` | — |

---

## The five things most likely to cost you marks

1. **Skipping the Q9.1 serving-feature ablation** — absent from v1, explicitly required, and
   directly tied to the named grading criterion "ablation rigour."
2. **Not securing `ebnerd_testset.zip` and both submission formats early** — Q5 is mandatory and
   demo/small contain no test set. This is the only hard external dependency in the assignment.
3. **Conflating candidate-generation recall with in-impression ranking** — produces either an
   embarrassingly low number you can't explain or a suspiciously perfect one you can't justify.
4. **Using `rank_bm25` for full evaluation** — it will not finish, and you'll discover that on day 4.
5. **Leaving the AI usage log until the end** — a listed deliverable that cannot be reconstructed
   retroactively.
