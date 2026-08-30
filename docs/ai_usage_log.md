# AI Usage Log

Required deliverable (Q7.4): all prompts, chat-history exports, and a marking of AI-generated
vs. human-written code. Convention:

- Full conversation transcripts are exported via Claude Code's `/export` command and stored
  under `docs/ai_usage_log/session-YYYY-MM-DD.md` (one per work session).
- This file is the index: date, tool, what was generated, what was kept vs. rewritten.
- Generated files carry no special header by default; substantial hand-edits after generation
  are the norm and are summarized here rather than tracked line-by-line.

## Sessions

### 2026-08-20 — Planning + Q1 (data pipeline)

**Tool**: Claude Code (Sonnet 5), VS Code extension.

**What was AI-generated**:
- `implementation_plan.md` / `SPEC.md` — full assignment plan, written by Claude after reading
  `Assignment1_v1.pdf` and directly inspecting the raw dataset files (unzipped `ebnerd_demo.zip`,
  read parquet schemas with `pyarrow`, unzipped MIND zips, read `.tsv` headers) to correct several
  incorrect assumptions in an earlier human-drafted plan (v1).
- `src/config.py`, `src/pipeline/{download,clean_mind,clean_ebnerd,split,feature_store,unzip_utils}.py`,
  `build_pipeline.py`, `Makefile`, `config/default.yaml`, `pytest.ini`, `requirements.txt`,
  `.gitignore`, `README.md`, `tests/{test_no_leakage,test_split_disjoint}.py` — Q1 pipeline,
  written by Claude based on the SPEC and the verified schemas above.

**What was verified/confirmed by the user, not the AI**:
- Codabench registration on both competitions (MIND, RecSys 2024/EB-NeRD).
- Official submission format guidelines for both competitions (pasted into chat by the user,
  cross-checked by Claude against the user's downloaded `sample/prediction.zip`).
- `ebnerd_testset.zip` URL liveness confirmed by Claude via `curl -I` after the user pointed at
  the dataset page.

**Not yet run**: the pipeline code above has not been executed anywhere (this machine has no
Python env for this project; execution happens on the remote GPU node `ada`). "Tests passing"
claims should not be trusted until `make data && make test` has actually been run there.

**Human review status**: pending — code has been written but not yet reviewed/run by the user.

### 2026-08-20 — Q1 implementation + local smoke test

**Tool**: Claude Code (Sonnet 5), VS Code extension.

Wrote the Q1 pipeline code (`src/pipeline/*.py`, `build_pipeline.py`, `tests/*.py`,
`Makefile`, `.gitignore`, `README.md`, `docs/ai_usage_log.md`, `pytest.ini`) and then, since
this machine happened to already have pandas/pyarrow/pytest installed, ran it end-to-end
locally against the real downloaded data as a self-check before handing off to `ada` (the
primary dev/test environment per user instruction) — to avoid burning a slow scp/ssh round
trip on bugs a 30-second local run could catch. Found and fixed two genuine bugs this way,
not caught by code review alone:

1. **Zip extraction double-nesting** (`unzip_utils.py`): MIND's zips contain an internal
   wrapping folder (`MINDsmall_train/news.tsv`); EB-NeRD's don't. The original
   `extract_clean` used one code path for both, silently double-nesting MIND's output
   (`data/raw/MINDsmall_train/MINDsmall_train/news.tsv`) and later failing with
   `FileNotFoundError`. Fixed by separating `extract_root` from `expect_dir`.
2. **`isinstance(x, list)` false-negative after a parquet round-trip**: parquet `list<>`
   columns come back from `pd.read_parquet` as `numpy.ndarray`, not `list`. `split.py`'s
   history-time flattening used `isinstance(x, list)` and silently treated every EB-NeRD
   user's click-history timestamps as absent, making `max_history_click_time` null for
   every split and causing the leakage boundary test to skip instead of actually check
   anything on the one dataset (EB-NeRD) where it matters. This was silent — no exception,
   just vacuously-passing (skipped) tests — and would not have been caught without running
   the real pipeline end to end and inspecting the manifest output.

3. **Same `isinstance(x, list)` round-trip issue also affected the manifest writer**:
   `_describe_split`'s `min()`/`max()` over the (correctly, once fixed) flattened history
   timestamps returned `numpy.datetime64`, which has no `.isoformat()` -- surfaced
   immediately as a hard `AttributeError` once bug #2 was fixed (rather than silently, since
   this path only runs when there's data to describe). Fixed by wrapping in `pd.Timestamp(...)`.

Also found two behaviors in the *data itself* (not pipeline bugs) via the same run, and
adjusted tests to assert the right thing instead of a wrong invariant:
- MIND's `impression_id` resets to 1 independently in `MINDsmall_train.tsv` vs.
  `MINDsmall_dev.tsv` (verified: unique within each raw file, overlapping across files).
  Added a `row_uid` column (`"<dataset>:<source_split>:<impression_id>"`) for safe
  cross-split joins; kept raw `impression_id` untouched for Q5 submission fidelity.
- EB-NeRD's raw `article_ids_inview` occasionally lists an article whose `published_time`
  is up to ~9 hours after the impression (0.01-0.03% of candidate rows in the demo split)
  — almost certainly `published_time` being edited/backdated after the article was already
  served, not a join bug on our side. Test changed from a strict zero-tolerance assertion to
  a documented tolerance check, with an explicit note that real eligibility filtering belongs
  in Q2/Q3's candidate generation, not this test.

After all three fixes: `make data` runs clean end to end on both datasets (demo scale), and
`pytest tests/` passes **17/20**, with 3 legitimate skips (MIND has no per-click history
timestamps, so `test_history_precedes_impressions[mind-*]` is inapplicable there by design
— see SPEC.md). Confirmed the EB-NeRD boundary check is now actually exercising real data,
not skipping vacuously: `split_manifest.json` shows e.g. train's
`max_history_click_time = 2023-05-18T06:59:51` <= `min_impression_time = 2023-05-18T07:00:03`.

**Human review status**: pending. Local run was for a syntax/logic sanity check only —
per user instruction, real dev/testing of record happens on `ada`; this was not a substitute
for that.

### 2026-08-20 — Q2 implementation (BM25) + local smoke test

**Tool**: Claude Code (Sonnet 5), VS Code extension.

Wrote `src/retrieval/{tokenize,bm25,candidates,run_bm25_eval}.py`, `src/eval/bootstrap.py`
(pulled forward from Q4 since Q2.4 needs it), and `tests/test_bm25_agreement.py`. Design:
hand-rolled sparse BM25 as a scipy CSR matrix (not `rank_bm25`, which is a pure-Python
per-query loop -- see SPEC.md Q2.1), sklearn's `CountVectorizer` used only for
tokenize->vocabulary bookkeeping. Ran the full pipeline locally against the real feature
store from the Q1 run (both datasets, demo-scale EB-NeRD) as a pre-`ada` sanity check and
found two real bugs:

1. **Unpicklable index**: `CountVectorizer`'s tokenizer/preprocessor were lambda closures,
   so `BM25Index.save()` crashed on `pickle.dump`. Fixed with a module-level `_identity`
   function and `functools.partial(tokenize, lang=lang)`.
2. (Investigated, not a bug) The first EB-NeRD run took 2m21s and looked stalled; profiling
   isolated it to a one-time NLTK stopwords corpus download (network fetch + unzip), not a
   pipeline defect -- subsequent runs with the corpus cached took under a second to build
   the same index.

Also observed a genuine, slightly counterintuitive IR result worth carrying into the design
note (Q6): the automatic N-sweep (query = last N clicked titles) picked N=1 for EB-NeRD
because Pool B (the "already shown" active-candidate pool) recall@100 is *highest* at N=1
(0.0844) and *drops* at N=5/10/20 (~0.06-0.07), even though Pool A (full-catalog) recall
increases monotonically with N as expected. Read as: a longer concatenated-history query
broadens full-catalog recall but dilutes precision within the smaller, harder-to-search
active pool. MIND showed the more expected pattern (N=20 best on both pools). Not touched
as a "bug" -- documented instead.

**Performance note for Q6's scale analysls**: the local MIND run (65,238 articles, ~194k
impression-rows across the N-sweep + test) took ~10 minutes single-threaded on this
(non-`ada`) machine, dominated by `np.argpartition` over the full dense score row per query
per pool. EB-NeRD demo (11,777 articles) took ~1 minute for the equivalent work. This
densify-then-argpartition approach is the first thing to replace at 10x scale (true sparse
top-k, or GPU-batched scoring) -- worth citing with these numbers in Q6's "where it breaks
at 10x" section rather than guessing.

**Final local verification**: `pytest tests/` -> 17 passed, 5 skipped (3 legitimate MIND
history-timestamp skips as before, plus 2 new `rank_bm25` agreement-check skips since
`rank_bm25` isn't installed on this machine -- will run for real on `ada`, where it's in
`requirements.txt`). `results/{mind,ebnerd}_bm25_test.json` inspected by hand: numbers are
sane (Pool A recall small as expected for full-catalog retrieval, Pool B meaningfully
higher, cold-start/empty-history impressions correctly routed to the popularity fallback:
2214/73152 for MIND, 0/11798 for EB-NeRD).

**Human review status**: pending — as with Q1, this was a local sanity pass, not the dev/test
run of record (that happens on `ada` per user instruction).

### 2026-08-20 — Q3 implementation (semantic retrieval, provided embeddings only)

**Tool**: Claude Code (Sonnet 5), VS Code extension.

User explicitly instructed: use PROVIDED article embeddings only, avoid training or running a
model (e.g. sentence-transformers/BERT) to compute embeddings -- "costly and time consuming".
This changes SPEC.md Q3.1's original plan (`all-MiniLM-L6-v2` for MIND), since MIND ships no
per-article embedding the way EB-NeRD does. Before implementing, inspected the raw files to
find a genuinely-provided option for MIND rather than assuming one existed:

- Confirmed `Ekstra_Bladet_word2vec/document_vector.parquet` (300-dim) covers **100%** of
  `ebnerd_demo`'s 11,777 articles -- used directly for EB-NeRD.
- MIND has no document-level embedding, but does ship `entity_embedding.vec` (100-dim TransE
  vectors keyed by WikidataId, e.g. "Q41"). Verified 86.7% of MINDsmall_train articles mention
  >=1 entity and 97% of mentioned WikidataIds resolve to a vector -> built MIND's article
  embedding as the mean-pool of its title+abstract entities' vectors. This required amending
  `clean_mind.py` to also carry `entity_wikidata_ids` (the raw JSON's `WikidataId` field) since
  the existing `entities` column only kept human-readable `Label` strings, useless for this
  lookup. Ran final coverage check after implementing: **87.03%** (56774/65238) -- the ~13%
  gap (no entities, or entities missing from the .vec file) is real and is excluded from the
  semantic-retrievable catalog, logged explicitly rather than zero-filled.

Updated `SPEC.md` Q3.1, its design-choices table, and `requirements.txt` (dropped
`sentence-transformers`/`torch`, no longer needed) to reflect this decision as the doc of
record, rather than leaving a stale plan that contradicts what was actually built.

Wrote `src/retrieval/build_embeddings.py`, `src/retrieval/semantic.py` (FAISS `IndexFlatIP`
with an automatic numpy brute-force fallback when the `faiss` package isn't installed --
mathematically identical since IndexFlatIP is exact, not approximate), and
`src/retrieval/run_semantic_eval.py` (mirrors `run_bm25_eval.py`'s N-sweep/Pool-A-B/bootstrap-CI
structure for direct comparability, per Q3.5). Ran end to end locally on both datasets (no
`faiss` installed here, so this also exercised the numpy fallback path) and found one real bug:

1. **`entity_embedding.vec` parsing crash**: every line in the file has a trailing tab before
   the newline; splitting on `\t` after only stripping `\n` left an empty trailing field, and
   `np.asarray(values, dtype=np.float32)` raised `ValueError: could not convert string to
   float: ''`. Fixed by using `line.rstrip()` (strips the trailing tab too) instead of
   `line.rstrip("\n")`.

**Final local verification**: `pytest tests/` still 17 passed, 5 skipped (unchanged --
Q3 didn't add new tests; the two rank_bm25-agreement skips and three MIND-history-timestamp
skips are the same as before). `results/{mind,ebnerd}_semantic_test.json` inspected by hand:
sane recall numbers, `embedding_meta` correctly carries the coverage percentages into every
result file, cold-start counts look right (15108/73152 MIND -- much higher than BM25's
2214/73152, because "no covered history article" is a strictly bigger set than "no history at
all"; 0/11798 EB-NeRD, consistent with its 100% embedding coverage). Numpy-backend query
latency: ~1140 qps on MIND (56,774 x 100-dim), ~2580 qps on EB-NeRD (11,777 x 300-dim) --
useful data points for Q6's scale analysis even though `ada` will use the real FAISS backend.

Also re-ran `build_pipeline.py` and `run_bm25_eval.py` for both datasets after the
`clean_mind.py` schema change (new `entity_wikidata_ids` column) to keep the feature store and
cached BM25 index consistent with the current code -- confirmed EB-NeRD's BM25 numbers are
byte-for-byte reproducible against the earlier Q2 run (same N-sweep, same recall values).

**Human review status**: pending — local sanity pass only, not the dev/test run of record.

### 2026-08-20 — Q4 implementation (evaluation harness)

**Tool**: Claude Code (Sonnet 5), VS Code extension.

Wrote `src/eval/{metrics,beyond_accuracy,slicing,run_eval}.py` and `tests/{test_metrics,
test_slicing}.py` per SPEC.md Q4, including the three metric bug fixes already documented
there (truncated-ideal nDCG, unseeded bootstrap, single-class AUC crash) plus 18 hand-checked
unit tests (e.g. the worked nDCG example from SPEC.md, a regression test proving the ideal-DCG
bug fix actually changes the result: `ndcg_at_k([1,0,0,1], 2)` must NOT equal what truncating
labels before computing IDCG would give).

`run_eval.py` scores each impression's OWN candidate list (the ranking task Q4/Q5/both
leaderboards actually measure) rather than reusing Q2/Q3's candidate-generation recall@K --
these are different tasks operating over different universes (SPEC.md Q4.0), so this is fresh
computation, not a reuse of Q2/Q3's per-impression scores.

Ran the full harness locally on both datasets/both methods (bm25, semantic) and found one real
bug, caught only by inspecting the actual slice sizes rather than trusting the code:

1. **Head/tail slicing was completely degenerate**: over 90% of articles in BOTH datasets have
   ZERO train-split clicks (90.19% MIND, 90.54% EB-NeRD -- verified by direct computation, not
   assumed). The "top 10% by click count" threshold (90th percentile) therefore landed exactly
   on 0, and the original `train_click_count >= threshold` comparison classified EVERY article
   (including all the zero-click ones, since 0 >= 0) as "head" -- silently collapsing the tail
   slice to n=0 on the first real run. This would have gone completely unnoticed if I'd only
   checked that the code ran without crashing; it only surfaced by explicitly printing
   `slices["tail"]["n"]` and seeing 0. Fixed with a strict `>` comparison (documented in
   `slicing.py`'s docstring with the exact measured skew), which gives head ~9.5-9.8% of the
   catalog as intended. Added `tests/test_slicing.py` as a permanent regression guard, using a
   synthetic distribution deliberately shaped to trigger the same interpolation edge case
   (needed 920/1000 zeros, not just 90/100, for `np.quantile`'s linear interpolation to land
   exactly on 0 rather than a value merely close to it -- verified empirically before writing
   the test, not assumed).

**Final local verification**: `pytest tests/` -> 37 passed, 5 skipped (same 5 legitimate skips
as Q2/Q3; 19 new tests all passing). `results/{mind,ebnerd}_{bm25,semantic}_eval.json` inspected
by hand: slice sizes now sane on both datasets (MIND: cold_start=10306, warm=62846, head=7788,
tail=65364; EB-NeRD: head=141, tail=11657, cold_start=0 matching its 100% history coverage from
Q1), `n_uncovered_candidates` correctly nonzero only for MIND's semantic method (573874 --
consistent with its 87.03% embedding coverage from Q3) and zero everywhere else, AUC/MRR/nDCG
values in a plausible range for simple content-similarity rankers (AUC 0.50-0.55, i.e. modestly
better than random -- expected, since neither BM25 nor mean-pooled-entity/word2vec similarity
is a click-likelihood model).

**Human review status**: pending — local sanity pass only, not the dev/test run of record.

### 2026-08-21 — Fixed real `ada` failure: MIND download 401

User ran `make data` on `ada` (the actual dev/test run, not a local sanity pass) and hit
`HTTPError 401` downloading `MINDsmall_train.zip` from HuggingFace, despite having run
`hf auth login`. Diagnosed with `curl -I` on the URL: `x-error-code: GatedRepo` --
`yjw1029/MIND` is a gated dataset, which requires BOTH clicking "Agree and access
repository" on the dataset's page (separate from CLI login, not done yet at diagnosis time)
AND the download request actually carrying an `Authorization: Bearer <token>` header. The
real bug: `download.py` used plain `urllib.request.urlopen(url)`, which never attaches a
token even when one is stored locally by `hf auth login` -- so this would have 401'd even
after clicking "Agree and access repository", making it a code bug independent of the
access-approval step. Fixed `src/pipeline/download.py`: added `_hf_token()` (reads
`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env vars or `~/.cache/huggingface/token`) and attached
it as a Bearer header for `huggingface.co` URLs; also added a specific actionable
`RuntimeError` on a HF 401 instead of a bare traceback. Documented in `SPEC.md` Q1.1 since
this affects anyone re-running the pipeline from scratch, not just this session.

**Human review status**: fix not yet re-verified on `ada` (requires the user to complete the
access-approval step there first) -- this is a real bug, not a hypothetical, since it
surfaced on the actual dev/test run of record.

### 2026-08-21 — Fixed real cross-machine reproducibility bug (found by diffing local vs. ada results)

**Tool**: Claude Code (Sonnet 5), VS Code extension.

User finished running the full pipeline (Q1-Q4) on `ada` and asked what's next. Rather than
just asking for a summary, pulled `results/*.json` and the split/embedding manifests back
from `ada` via `scp` (consistent with the newly-agreed auto-sync behavior in
`~/.claude/CLAUDE.md`) and diffed them against the local smoke-test numbers before answering
"what's next" -- this is what surfaced the bug below; it would NOT have been caught by only
reading `ada`'s console output.

MIND's numbers matched local exactly (AUC 0.5544, MRR 0.3061 for BM25). EB-NeRD's BM25 numbers
did not: local picked N=1 for its query-history-length sweep, `ada` picked N=20, and the
underlying val-sweep recall@100/200 values at N=1 genuinely differed between the two runs
(e.g. B@200: 0.2665 local vs. 0.1508 on ada) despite identical seed/code/data. Root-caused to
`src/retrieval/bm25.py`'s `top_k_indices`: it used `np.argpartition` to select the top-K, then
`argsort` only within that slice. `argpartition` gives no guarantee about *which* tied
elements land in the partition, and for a sparse N=1 query (a single article's title), most of
the catalog scores exactly `0.0` -- with K up to 200 and far fewer than 200 nonzero-scoring
docs, which zero-score docs fill the rest is arbitrary and can differ across numpy/BLAS
builds/platforms. Fixed with a full stable `np.argsort` (deterministic ascending-index
tie-break). Same latent bug existed in `semantic.py`'s numpy-fallback `faiss_search` (not on
`run_semantic_eval.py`'s critical path, but fixed for consistency and because it's the same
class of bug). Added `tests/test_bm25.py` (4 tests) as a permanent regression guard, including
one that reconstructs the exact failure shape (mostly-zero scores, few real winners).

Documented in `SPEC.md` Q2.1 and synced the three changed files
(`bm25.py`, `semantic.py`, `tests/test_bm25.py`) to `ada` myself via `scp` per the
newly-established auto-sync default, since the context (active ada-workflow session,
known destination path) was already established.

**Why this matters beyond just fixing a bug**: Q1 explicitly requires a "reproducible"
pipeline, and this is a concrete demonstration that a script exiting cleanly on one machine
does not prove reproducibility -- it needed a second environment and an actual diff to catch.
Worth a sentence in the design note (Q6) as a genuine finding, not just a changelog entry.

**Human review status**: fix pushed to `ada`; NOT yet re-verified there. User needs to re-run
`run_bm25_eval.py`, `run_semantic_eval.py`, and `run_eval.py` (in that order, since Q4 reads
Q2/Q3's `best_n`) on `ada` for the fix to take effect in the results of record.

**Update, same day**: user re-ran all three on `ada`. Pulled `results/*.json` back and
confirmed the fix worked: EB-NeRD BM25 now picks N=20 (same as MIND), matching what a
deterministic tie-break should converge to, rather than the platform-dependent N=1 from
before. All slice sizes and other numbers otherwise unchanged, as expected (this bug only
affected recall@K/best-N selection, not Q4's own ranking, which already had deterministic
tie-breaking).

### 2026-08-21 — Q5 implementation (Codabench submission)

**Tool**: Claude Code (Sonnet 5), VS Code extension.

Wrote `src/retrieval/scoring.py` (extracted `run_eval.py`'s private scoring helpers into a
shared, public module so Q4's eval and Q5's submission generators score impressions through
the exact same code path -- avoids two subtly different reimplementations of the same logic),
`src/submission/{format,mind,ebnerd}.py`, and `tests/{test_submission_format,
test_submission_pipeline}.py` (13 new tests, all passing; the pipeline test exercises the real
`BM25Index -> score_bm25_batch -> scores_to_ranks -> validate_submission` path end to end on a
tiny synthetic catalog, including the cold-start-popularity-fallback and
uncovered-candidate-handling branches, specifically because a caught bug here is free and a
caught bug on the actual rate-limited Codabench endpoint (MIND: 1/day, EB-NeRD: 5/day) is not).

`ebnerd_testset.zip` (1.5GB, needed for the EB-NeRD submission) had not been downloaded yet.
Rather than pull the whole thing locally just to inspect it, listed its directory structure
remotely via HTTP range requests against the zip's central directory (no download needed for
that part) and found a real, would-have-shipped bug this way:

1. **Same double-nesting bug as `clean_mind.py`, caught before ever running**:
   `ebnerd_testset.zip`'s members ARE wrapped in an `ebnerd_testset/` folder internally
   (`ebnerd_testset/articles.parquet`, `ebnerd_testset/test/behaviors.parquet`, ...) --
   unlike `ebnerd_demo.zip`/`ebnerd_small.zip`, which have no such wrapping folder. My first
   draft of `_ensure_testset` used the demo/small extraction pattern (`extract_root =
   raw_dir/"ebnerd_testset"`), which would have silently double-nested the output exactly
   like the original MIND bug (Q1). Fixed to match `clean_mind.py`'s pattern: extract into
   `raw_dir` itself, with `expect_dir=raw_dir/"ebnerd_testset"` for the idempotency check.

Could NOT verify the actual column schemas of `articles.parquet`/`history.parquet`/
`behaviors.parquet` inside the zip the same way (each member is DEFLATE-compressed, so a
byte-range read doesn't land on a coherent parquet footer without decompressing from the
member's start) -- documented this limitation explicitly in `ebnerd.py`'s docstring rather
than silently assuming demo/small's schema carries over. The loader asserts required columns
by name and raises a specific, actionable `AssertionError` (not a generic `KeyError` or silent
`NaN` propagation) if the assumption is wrong -- this still needs to be verified for real on
`ada` once the user runs `python -m src.submission.ebnerd`, since it's the one part of Q5 that
couldn't be checked without the actual 1.5GB file.

Also documented (not a bug, a design decision): MIND's submission reuses `data/splits/mind/
test` since the assignment provides no separate MIND test file, and EB-NeRD's submission
builds a FRESH BM25 index over the test catalog rather than reusing `feature_store/ebnerd/
bm25`, since the test articles are almost certainly disjoint from demo/small's (different time
window) -- reusing the wrong index would have produced near-zero-coverage garbage silently.

**Human review status**: pending. Local verification limited to unit/pipeline tests on
synthetic data (couldn't smoke-test against the real `ebnerd_testset.zip` locally -- 1.5GB,
and the real dev/test environment is `ada` per user instruction anyway). User needs to run
both submission scripts on `ada`, inspect the generated `predictions/*.zip` files, and only
then upload to Codabench given the rate limits.

### 2026-08-22/23 -- Workflow pivot to local execution, real-scale Q5 bugs, MIND test-set correction, Q6 + Q9.1

User decided to run everything CPU-only on this local machine instead of `ada` (no GPU work in
this assignment), and scp'd `data/raw`, `data/interim`, `data/splits`, `feature_store`, and
`predictions` back from `ada`. Verified the transfer (sizes, file counts, and sha256 on 6 key
files) before proceeding. Three real, sequential production incidents followed while actually
running Q5 for real, plus a MIND correctness bug found via Codabench itself, plus new work
(Q6, Q9.1) -- summarized here; full technical detail is in `SPEC.md`/`docs/design_note.tex`.

**1. `ebnerd_testset.zip` download kept stalling.** A single TCP connection to the S3 bucket
went idle (`ESTABLISHED`, 0 bytes/sec) repeatedly, reproduced independently from two networks
(this machine and `ada`), while a same-machine test to a different host hit 1.5MB/s easily --
pointed at loss/RTT capping one TCP flow, not a saturated link. Added resume (`Range` header)
and auto-reconnect (read timeout + retry) to `src/pipeline/download.py`'s `download_file`,
then parallelised large downloads across up to 24 simultaneous range-requested connections
(`_parallel_download`, with a byte-interval-based resume scheme so the connection count can
change between runs without losing progress). Raised effective throughput roughly 30x.

**2. EB-NeRD's real test scale (13.5M impressions, 125K articles, confirmed via `pyarrow`
schema inspection once downloaded) OOM-killed the original `src/submission/ebnerd.py`** at
~10.5GB RSS on this 15GB machine, since it read `behaviors.parquet`/`history.parquet` whole
(many unused columns) before any batching. Fixed with column-pruned, batched parquet reads
(`pyarrow.parquet.ParquetFile.iter_batches`) and native int32/uint32 IDs instead of blanket
`.astype(str)` (the string conversion alone was costing ~6.4GB on the history data). Peak RSS
dropped to 1.7-1.9GB.

**3. Two further performance bugs found by actually timing the fixed pipeline, not assumed.**
`score_bm25_batch`/`score_semantic_batch` (`src/retrieval/scoring.py`) build a full
`{article_id: score}` dict over the ENTIRE catalog per impression: fine for MIND's ~30K
articles/73K impressions, estimated at 70+ hours of pure dict construction at EB-NeRD-test
scale. Then, even after restricting dict output to each batch's own ~250-300 unique
candidates, the underlying full-catalog score MATRIX computation itself was still ~10 hours
(measured: 367 impressions/sec). Added `score_bm25_batch_candidates`/
`score_semantic_batch_candidates`, which restrict the matmul/sparse-matmul itself to only
each batch's candidate columns (via a precomputed `article_id -> column index`) -- verified
numerically identical to the full computation (max diff 1.8e-7, pure float32 rounding) since
matrix columns are independent, not an approximation. This alone dropped the run to 49 minutes
(4629 impressions/sec). Both fixes are in the shared `scoring.py` module the Q4 harness also
uses, so Q4's own numbers are unaffected (same math, only the extraction/computation path for
large-batch cases changed) -- did not touch `score_bm25_batch`/`score_semantic_batch` (the
non-`_candidates` originals), used by `run_eval.py` and reused by `mind.py`. Applied all three
`ebnerd.py` fixes preemptively when writing MIND's corrected submission script (below); it ran
cleanly first try.

**4. MIND's Q5 submission was WRONG, caught by Codabench's own scoring, not by us.** The
original `src/submission/mind.py` scored `data/splits/mind/test` (== all of MINDsmall_dev),
on the documented assumption that the assignment provided no separate MIND test file. That
upload failed Codabench scoring with `IndexError: boolean index did not match indexed array
... dimension is 22 but corresponding boolean dimension is 16`. Diagnosed by comparing our
submission against the raw `MINDsmall_dev.tsv` line-by-line for all 73,152 rows (zero
mismatches, ruling out a bug in our own pipeline) before concluding Codabench's reference must
differ from the public file. Asked the user to check the competition page; they found and
downloaded `MINDlarge_test.zip` from https://msnews.github.io/ (the actual MIND challenge's
blind test set, 2.37M impressions, 121K articles, genuinely disjoint from MINDsmall). Rewrote
`mind.py` to build a fresh index over `MINDlarge_test`'s own catalog and stream its
`behaviors.tsv`, reusing `clean_mind.py`'s `_load_news` and applying the same
candidates-only-scoring and streaming-write fixes from (2)/(3) preemptively -- clean run,
0.83GB RSS, 2019 impressions/sec, ~20 minutes, 32 uncovered candidates out of ~35M
candidate-scores. Re-verified structurally (single root-level file in the zip, line count
matches the source file exactly, impression IDs sequential) before the user re-uploaded.
MIND's leaderboard submission subsequently succeeded.

**5. Q6 (design note).** Drafted `docs/design_note.tex` (LaTeX, per user's stated preference
for that format over an HTML-print or Markdown workflow) following SPEC.md's exact
page-budget structure, grounded entirely in `results/*.json` numbers and the real incidents
above (not extrapolated projections) for the "where it breaks at 10x" section. Compiled
locally with `pdflatex` (available on this machine) to verify the actual page count (3 of the
4-page budget) rather than guessing. Left an explicit placeholder for the leaderboard
screenshots (EB-NeRD's evaluation was still processing) instead of fabricating a result.

**6. Q9.1 (serving-feature ablation).** New `src/eval/run_ablation.py`, EB-NeRD only (MIND's
unified-schema columns for these features are entirely null; not constructible there, not
merely skipped). Three cumulative configurations on the same labeled offline test split Q4
already scores: clean (base retrieval score), +popularity (adds a re-ranking prior from
`total_inviews`/`total_pageviews`/`total_read_time`/`sentiment_score`, corpus-aggregate
features unavailable at serving time), +post-interaction (adds a large bonus directly to the
true-click candidate -- the most direct possible post-interaction feature, since the click
outcome is exactly the label being predicted). Found and fixed a NaN bug while building this:
`stat.get(field) or 0` doesn't catch NaN (NaN is truthy in Python), so pandas' NaN-filled
engagement fields silently propagated into `roc_auc_score` and crashed it; needed an explicit
NaN check. Result: clean and +popularity both under 0.6 AUC on both methods; +post-interaction
scores exactly 1.0000 on every metric, on both methods, by construction -- the measured
demonstration SPEC.md Q9.1 asks for. Five new tests in `tests/test_ablation.py`, including one
that reads the real `results/ebnerd_ablation.json` and asserts the expected ordering
(clean <= +popularity <= +post-interaction, with +post-interaction > 0.99 AUC). Also noted, as
a documented finding rather than a silent omission: EB-NeRD's `next_read_time`/
`next_scroll_percentage` columns exist but are scoped to the whole impression (one value per
impression, not per candidate), so they cannot move within-impression ranking metrics
(AUC/MRR/nDCG) at any weight -- not every leaky column is exploitable via simple re-ranking.
Also found and fixed, while building this, a pre-existing (unrelated) bug in
`run_bm25_eval.py`/`run_semantic_eval.py`: `best_n` was written to the val-sweep JSON as a
Python dict key (a string, since JSON keys are strings) rather than cast to `int`, which
crashed `src/retrieval/semantic.py`'s `n > 0` comparison the first time `ebnerd.py`'s
duplicated (not shared-helper) JSON-parsing path was actually exercised at real scale; fixed
at the source (`int(max(...))`) and routed `ebnerd.py`/`mind.py` through the existing shared
`scoring.best_n()` helper instead of re-deriving it inline.

**Human review status**: all of the above verified by the user actually running the scripts
and reporting real Codabench outcomes (MIND upload failure, then success), not just local
smoke tests. Q9.1's ablation numbers and the design note's content are grounded in files
committed to `results/` and `docs/`, checkable directly.

### 2026-08-24 -- Leaderboard screenshots inserted into the design note

User added `screenshots/MIND-leaderboard.png` and `screenshots/eb-nerd_screenshot.png`
(both 949px wide, 96px/41px tall -- narrow leaderboard-row crops, not full-page captures) and
asked for them to replace the placeholder text in `docs/design_note.tex` \S6 (Evaluation
harness), which previously read "[Leaderboard screenshots ... to be inserted here once
EB-NeRD's evaluation completes]" since EB-NeRD scoring was still pending when \S6 was
originally written. Replaced the placeholder with both images (`\includegraphics`,
`0.85\linewidth` each) directly under the sentence introducing them. Recompiled with
`pdflatex` and reverified via `pdfinfo`: still 3 of the 4-page budget, no new overfull-hbox
warnings. Visually checked the rendered page (rasterized with `pdftoppm`, read back as an
image) rather than trusting a clean compile alone -- confirmed both images render at a
readable size in the right place. Noted for the user: the EB-NeRD screenshot shows
`ebnerd_predictions.zip ... Submitted` with no score column filled in yet, while the MIND one
shows a completed score row -- consistent with EB-NeRD's evaluation still being in progress as
of this screenshot; flagged rather than assumed finished, since a stale "pending" screenshot in
a submitted design note would misrepresent the deliverable.

User also asked which file is the canonical AI usage log and where the current session's full
transcript is saved (this file's stated convention says transcripts should live under
`docs/ai_usage_log/session-YYYY-MM-DD.md`, but no such per-session export files exist yet --
this index file has been maintained by hand-summarizing each session instead, which is what
generated the "2026-08-24" entry above). Answered by pointing at this file (`ai_usage_log.md`)
as the index, and at Claude Code's own on-disk session transcript
(`~/.claude/projects/-home-yash-more-Downloads-31-ire-a1/5cbe3e82-c366-4a84-a62f-4c8b7fd0d1e6.jsonl`,
~11MB, spans this entire multi-day session including a mid-session `/compact`) as the raw
record, rather than fabricating a location. If a verbatim per-session export is wanted to fully
satisfy the file's own stated convention, it should be produced with Claude Code's `/export`
command and saved to `docs/ai_usage_log/session-2026-08-24.md`; not done automatically here
since it wasn't explicitly requested.

**Human review status**: image placement and page count verified by the user's own screenshot
files and a re-rendered page image, not just a clean LaTeX compile.

### 2026-08-25/26 -- EB-NeRD submission filename bug, final screenshots, requirements check

User reported the EB-NeRD Codabench submission had failed with `FileNotFoundError:
/app/input/res/predictions.txt`. Diagnosed by inspecting the actual zip contents
(`unzip -l predictions/ebnerd_predictions.zip`): it contained `ebnerd_predictions.txt` at the
root, not `predictions.txt`. Cross-checked against `SPEC.md`'s EB-NeRD submission-format
section and `src/submission/mind.py`, which already handles the equivalent MIND requirement
(Codabench needs the exact internal filename `prediction.txt`, singular) by copying to a
temp file with the required name before zipping, then deleting it. `src/submission/ebnerd.py`
was missing that step entirely -- it zipped `ebnerd_predictions.txt` directly. Fixed by adding
the same copy-rename-zip-delete pattern (`shutil.copyfile` to a temp `predictions.txt`, zip
that, `unlink()` after). Also repackaged the *existing* `ebnerd_predictions.zip` directly from
the already-computed `ebnerd_predictions.txt` (703MB) rather than rerunning the multi-hour
scoring pipeline, since only the archive's internal filename was wrong, not the predictions
themselves.

User then renamed `screenshots/MIND-leaderboard.png` to `screenshots/mind_screenshot.png` and
replaced `screenshots/eb-nerd_screenshot.png` with an updated capture (now showing a completed
score row, AUC 0.514 -- resolving the "still pending" state flagged in the 2026-08-24 entry
above) and asked for `docs/design_note.tex` to be updated and recompiled. Updated the
`\includegraphics` path for the renamed MIND file; the EB-NeRD path was already correct since
the filename didn't change. Recompiled with `pdflatex` (two passes, to settle the hyperref
bookmark outline); output unchanged at 3 of the 4-page budget.

User separately asked whether the assignment requirements were complete. Checked the working
tree against `Assignment1_v1.pdf` and this repo's own `SPEC.md` Q7-Q9 checklist: Q1-Q4 pipeline/
retrieval/eval code and tests all present and passing (`pytest -q`: 59 passed, 3 skipped); Q5
now has completed scores on both leaderboards (MIND AUC 0.5851, EB-NeRD AUC 0.514); Q6 design
note compiles clean with both screenshots embedded; Q9 leakage test and serving-feature
ablation table both present. Flagged three gaps to the user rather than silently fixing them:
today's changes (the `ebnerd.py` fix, renamed/updated screenshots, recompiled design note) were
sitting uncommitted; this AI usage log itself was stale (last entry 2026-08-24); and LaTeX
build junk (`design_note.aux/.log/.out`) was untracked but not gitignored. User asked for all
three to be closed out, which is what produced this entry, the `.gitignore` addition
(`docs/*.aux`, `docs/*.log`, `docs/*.out`), and the commit that follows.

**Human review status**: the `ebnerd.py` fix was verified by inspecting the repackaged zip's
contents directly (`unzip -l`, confirming the single root-level `predictions.txt`) rather than
just trusting the code change; actual Codabench rescoring outcome (whether the resubmitted zip
is accepted) is the user's to confirm once submitted, not verifiable from this environment.

### 2026-08-26 -- Rigor additions per professor's grading note (tool/index choices, alternatives, measured throughput)

Professor's message on grading emphasized "rigorously analyse your approach including
tool/db/index choices, their impact on engineering metrics, how they compare to alternatives,
... optimizations improving latency/throughputs" over leaderboard rank. Reviewing the design
note against that bar found four claims that were asserted rather than measured. User asked for
all four to be closed out in one sitting, combined with walking through the underlying concepts
(BM25, then semantic/FAISS) so the user can defend each choice in the graded viva, not just cite
a number. All four experiments are real runs against local labeled val/test splits (no Codabench
resubmission involved -- confirmed with the user first, since they asked whether this required
re-submitting to the leaderboard).

1. **`rank_bm25` vs. the hand-rolled CSR scorer** (`src/eval/bench_bm25_alternatives.py`, new):
   `bm25.py`'s docstring claimed rank_bm25 "does not finish in reasonable time" without ever
   timing it. Measured head-to-head on the same tokenized corpus/queries: 782x speedup on MIND
   (196 queries: 73.0s vs 0.09s), 497x on EB-NeRD; extrapolated to MIND's full 73K-impression
   test split, rank_bm25 would need ~7.6 hours vs. 35s for the CSR matmul. Written into
   `design_note.tex` S3, replacing the unmeasured claim.

2. **Stemming ablation** (`src/eval/run_tokenizer_ablation.py`, new): reading `tokenize.py`
   surfaced that `stem` defaults to `False` and no caller (bm25.py, either submission script)
   ever passes `stem=True` -- stemming was never actually applied, contradicting
   `design_note.tex`'s existing claim of "language-specific tokenisation and stemming." Added a
   `stem` kwarg to `BM25Index.build` (default unchanged) and measured recall@K with it on vs.
   off, at each dataset's already-selected best N, on the val split. Stemming genuinely helps
   EB-NeRD (B@200 recall 0.0444->0.0604, +36% relative) and genuinely hurts MIND slightly
   (B@200 0.1405->0.1372, -2.3%) -- consistent with Danish's heavier compounding morphology.
   Asked the user whether to adopt stemming into the live EB-NeRD submission (a real improvement)
   or just correct the false claim and report the finding; user chose report-only, given the
   deadline and EB-NeRD's 5/day Codabench rate limit plus multi-hour scoring time. Design note
   corrected accordingly, predictions left unchanged.

3. **FAISS ANN recall-vs-latency curve** (`src/eval/bench_ann_recall_latency.py`, new): the
   design note's own Limitations section already flagged this as not done. Built `IndexIVFFlat`
   (nlist=sqrt(n_docs), swept nprobe) and `IndexHNSWFlat` (M=32, swept efSearch) against the
   existing `IndexFlatIP` exact baseline, measuring recall against the exact index's own top-100
   (not ground-truth clicks -- a standard ANN-evaluation convention) and QPS, using real user
   query vectors sampled from the test split's click histories. IVF nprobe=8 reaches 91-92%
   recall at 13-20x exact's throughput; HNSW efSearch=64 reaches 93-99% recall at 15-19x.
   Written into design_note.tex S4; the Limitations line was updated from "no curve was run" to
   "measured at demo/small scale only, not at 10x+ production scale."

4. **OpenBLAS thread-count sweep** (`src/eval/bench_thread_scaling.py`, new): S7 claimed "this
   machine's numpy build caps OpenBLAS at 2 threads" (from `numpy.__config__.show()` reporting
   `MAX_THREADS=2` in the build config) without ever sweeping actual thread counts. First
   attempt mistakenly benchmarked BM25's *sparse* matmul; realized this doesn't test the actual
   claim (S7's 367 impressions/sec figure is about the *semantic* full-catalog *dense* matmul,
   `SemanticIndex.score_dense`), so generalized the script with a --mode flag and measured both.
   Semantic (dense): throughput keeps improving past 2 threads on both a wrapped benchmark and
   an independent bare-numpy sanity check (180->95->52->52ms at 1/2/4/8 threads, plateauing at
   4) -- the original "capped at 2 threads" claim does not hold on this machine and was
   retracted in the design note, with a note that the printed `MAX_THREADS` build field is not
   an enforced runtime ceiling. BM25 (sparse): flat throughput across 1-8 threads on both
   datasets -- confirmed as a genuinely different finding (scipy's sparse matmul doesn't route
   through the same threaded dense-BLAS GEMM path), kept in the design note as a contrast to the
   semantic result rather than discarded as a mistake. User asked for the correction to
   replace (not merely append to) the original unverified claim.

Design note grew from 3 to 4 pages after these additions (still within the 4-page assignment
cap); tightened `\parskip` and cut some verbosity in two paragraphs (the "Extrapolating past
13.5M" paragraph in S7, and the OpenBLAS correction paragraph) so S8/references landed back on
page 3, leaving page 4 as a short 3-line references-only tail rather than a mostly-blank page.
Full pytest suite re-run after the `bm25.py` `stem` parameter addition: 59 passed, 3 skipped,
unchanged from before.

**Human review status**: all four benchmark scripts were actually executed against real local
data (not synthetic/mocked) and their printed output cross-checked against what got written into
`design_note.tex` before compiling; the thread-scaling correction was independently re-verified
with a bare-numpy benchmark outside the pipeline wrapper specifically because it contradicts
previously-written content, not just accepted on the first measurement. The decision not to
adopt stemming into the live submission, and to correct (not merely append to) the thread-count
claim, were both explicit user choices via AskUserQuestion, not unilateral calls.

### 2026-08-26 (later) -- Pre-deadline review: tone/voice pass, verified a remembered number, structural rewrite

User flagged the deadline was ~4 hours out and asked for a review pass rather than new content:
tone (individual-assignment voice), one number they wanted double-checked against memory, and a
structural change (bullets instead of dense paragraphs) applied uniformly across the whole note.

1. **First-person voice**: this is individual work, but the note was written in "we/our" plural
   throughout (a holdover from earlier drafting). Grepped for whole-word `\b(we|our|us)\b`
   rather than trusting a visual scan (`grep -noiE`) to find every instance without false
   positives from words like "source"/"hour" -- found and fixed exactly 7 occurrences across
   S1/S3(x2)/S4/S5/S7(x2). Re-grepped after editing to confirm zero remained.

2. **Verified, not assumed, a remembered statistic**: user recalled MIND's cold-start-impression
   percentage as "13%" and asked to check it against the note's stated "14%". Recomputed directly
   from `results/mind_bm25_eval.json`'s slice counts (10306 cold-start / 73152 total = 14.09%)
   rather than trusting either the note or the user's memory blindly -- the note's existing 14%
   was correct; the user's recollection was the one slightly off. No change made, since the
   claim already matched the underlying data.

3. **Section 1 shortened**: cut restating what the assignment already asks for (the professor
   set the assignment, doesn't need it re-explained), keeping only what's actually
   informative -- the parameterised-not-duplicated pipeline decision and why it matters given
   the two datasets' differences.

4. **Whole-document restructure to bullet points**: every section's previously-dense paragraphs
   (S1 intro, S2 pipeline, S3 BM25 including the newly-added rank_bm25/stemming rigor content,
   S4 semantic including the ANN curve, S5's dataset-property/cold-start/head-tail findings, S6's
   metrics/slicing list plus the Q9.1 ablation writeup, S7's breakage narrative including the
   OpenBLAS correction and 10x extrapolation) rewritten as `itemize` lists, one distinct claim
   per bullet, so each point is individually scannable rather than requiring a full paragraph
   read to find one fact. S7's existing `enumerate` (the three breakages) was left as-is since it
   was already itemized. User explicitly said not to worry about the page count from this
   change ("we'll fix it later on") -- grew from 4 to a still-4-page document (page 4 now has
   more trailing whitespace than before the bullet conversion), left as is per that instruction
   rather than pre-emptively tightened.

Recompiled with `pdflatex` (two passes) and visually verified all four pages via `pdftoppm`
rasterization rather than trusting a clean compile alone -- confirmed bullets render correctly,
no text overflowing page margins, one new overfull-hbox warning (21.5pt, in the Q9.1 ablation
table's surrounding text) is not visible in the rendered output. Full pytest suite re-run since
this preceded a commit, even though only `docs/` changed: 59 passed, 3 skipped, unchanged.

**Human review status**: the "13% vs 14%" question was answered by direct recomputation from
the results file the claim is grounded in, not by re-reading the design note's own prose (which
would just be checking the claim against itself); the we/our/us removal was verified complete by
re-grepping after editing, not assumed from the edit list alone.

### 2026-08-27 -- Full rewrite into real ACL format, prose style, new Observations section

User asked for the design note to be rewritten in the genuine ACL conference format (they
supplied the official template zip), converted from bullet points back to flowing research-paper
prose, with simpler vocabulary, numbered tables/figures with captions, no section-sign (S)
references, and a new "Observations" section. Explicitly said not to worry about page limits.

**Getting the real ACL style.** `acl.sty` was not installed locally and only `acl_format.tex`
(the template body, not the style) was in `docs/`. Tried fetching `acl.sty` from the
acl-org/acl-style-files GitHub repo; the guessed paths 404'd, and querying the repo tree via the
GitHub API confirmed the file sits at the repo root rather than under `latex/`. While that was
in progress the user supplied the official zip directly, which was used instead --
`acl.sty` and `acl_natbib.bst` extracted into `docs/`. Verified the style compiles on this
machine with a throwaway minimal document before touching the real one, specifically checking
that `times`, `microtype`, `inconsolata` and `T1` encoding were all present and that Danish
characters (aeoaa) render, since the note discusses Danish tokenisation. Used the `preprint`
option rather than `review` (review adds line numbers and anonymises; this is a named final
submission).

**Numbers re-pulled from results, not copied from the old draft.** The new Observations section
makes claims about slices and beyond-accuracy metrics that the previous draft only partly
covered, so all of them were read directly out of `results/{mind,ebnerd}_{bm25,semantic}_eval.json`
rather than transcribed. This surfaced that the eval JSONs keep beyond-accuracy under
`overall`, not a separate `beyond_accuracy` key (a first attempt at extraction returned empty
dicts and was corrected rather than reported as "no data"). Notable findings written up in
Observations that had not previously been stated anywhere: MIND BM25 improves from 0.520 AUC
cold-start to 0.560 warm while semantic barely moves (0.527 -> 0.538) and is actually better
than BM25 on cold-start specifically; MIND semantic scores 0.476 AUC on head articles, i.e.
below random, while managing 0.543 on tail; and BM25 has both higher accuracy and higher
diversity/coverage on MIND, which runs against the usual assumption that those trade off.

**Structure**: added an Abstract (ACL format supports it, and the user explicitly allowed it),
promoted the Q9.1 serving-feature ablation from a subsection to its own numbered section, added
the new Observations section, and converted all six tables and the leaderboard figure to proper
`table*`/`table`/`figure*` floats with `\caption` and `\label`, cross-referenced in text via
`\ref` so numbering stays correct automatically. Replaced every `\S` reference with
"Section~\ref{...}". Wide tables span both columns; the ANN and ablation tables fit in one.

**Verification**: compiled twice (for cross-reference resolution), then rasterised every page
with `pdftoppm` and read them back as images rather than trusting a clean compile. First build
was 8 pages with two overfull-hbox warnings; the 58pt one was the eight-column beyond-accuracy
table genuinely exceeding the text block (fixed with `\small`), and the references had spilled
onto a nearly-empty page 8 (fixed by replacing the bulleted list with a compact ACL-style
hanging-indent block, bringing it to 7 pages). Final grep confirmed zero remaining `\S`
references, zero "we/our", and none of the jargon the user flagged ("headroom"). Test suite
re-run since this preceded a commit even though only `docs/` changed: 59 passed, 3 skipped.

**Human review status**: every page was visually inspected as a rendered image, not just
compiled; all statistics in the new Observations section were read out of the results JSONs at
write time rather than carried over from the previous draft, and the one extraction bug (wrong
JSON key) was caught and fixed rather than silently producing an empty table.

### 2026-08-27/29 -- Design-note refinement, 4-page condensation, repo cleanup, submission packaging

**Tool**: Claude Code (Opus 5), VS Code extension.

Continues directly from the ACL-format rewrite above. No new pipeline code was written in this
period; all work was on the design note, the README, and preparing the submission archive.

**1. User's own edits, preserved verbatim.** The user hand-edited `design_note.tex` between
sessions: moved the screenshots into `docs/screenshots/` and updated the `\includegraphics`
paths, added bold inline paragraph-lead labels (`\textbf{Background:}`, `\textbf{Two Tasks:}`,
`\textbf{Datasets:}` and so on) to Section 1 only, and commented out three explanatory sentences
with `%`. They asked for the same labelling pattern to be extended to the remaining sections and
explicitly said to keep their own edits as-is. Added roughly 29 further labels across Data
Pipeline, Lexical Retrieval, Semantic Retrieval, Evaluation Harness, Observations,
Serving-Feature Ablation and Where It Breaks at Scale, leaving Limitations unlabelled since a
single label there would just restate the section heading. Verified by grep afterwards that the
screenshot path, all three commented-out sentences, and an incomplete sentence the user had left
mid-edit (`...while MIND structurally cannot. The`) were all still intact and unmodified. That
dangling fragment was flagged to the user rather than silently fixed, since the instruction was
to preserve their edits.

**2. Layout changes, each traced to the actual mechanism rather than guessed.** The user asked
how to reduce page margins; `acl.sty` loads `geometry` internally at line 103 with
`margin=2.5cm`, so the fix is to call the `\geometry{...}` command after the package is already
loaded rather than re-`\usepackage` it. Applied at 1.5cm, later 1.2cm. Separately, the user
asked to reclaim the large gap under the title: `acl.sty` reserves a fixed `\titlebox` of
`11\baselineskip` with stretchy `fil` glue between the title, author block and body, so the gap
is leftover box height rather than a fixed skip; reduced to `7\baselineskip`. Also raised
`\topfraction`/`\dbltopfraction` and lowered `\textfraction`, because the default float limits
were forcing the wide `table*`/`figure*` floats onto pages of their own.

**3. Table consolidation.** On the user's request the two separate recall@K tables (BM25 and
semantic, identical column headers) were merged into one four-row table with a dataset-separating
midrule, and every `\ref` updated from the two old labels to a single `tab:recall`. The user then
noticed the Slicing paragraph in the Evaluation Harness section described cold/warm and head/tail
comparisons that appeared nowhere as a table -- the sliced numbers existed only as scattered prose
in Observations. Confirmed this was correct, then built the missing table directly from the
`slices` blocks of `results/{mind,ebnerd}_{bm25,semantic}_eval.json` and cross-referenced it from
the Slicing paragraph and from both Observations paragraphs that cite those numbers. For the
4-page version this was merged again with the beyond-accuracy table into a single float with
`Overall` and `By slice` sub-blocks, since the two share identical row labels and three
span-both-column floats were competing for the same page tops.

**4. Removals, all user-directed.** The References section was removed (four hand-formatted
entries, no bibtex). Figure 1 (the two leaderboard screenshots) was removed; rather than dropping
the evidence, the scores were kept inline as prose (MIND 0.5851, EB-NeRD 0.514) with a pointer to
the `docs/screenshots/` folder, per the user's instruction to mention that a separate folder holds
them. Also converted the word "percent" to `\%` throughout on request.

**5. 4-page condensation (`design_note-4-pages.tex`, later renamed).** The user asked for a
version containing all the same points but under a hard 4-page cap, and asked to see the cut plan
before any file was written. Measured the existing document first (4,241 words of prose across 6
pages, plus five tables and one figure taking roughly 1.5 pages) to derive the required reduction
rather than estimating it, then presented a section-by-section plan naming specific redundancies:
the BM25 three-corrections prose restating what the formula and its variable glossary already
say; the `rank_bm25` result stated four different ways; Pool A/B defined identically in both the
prose and the table caption; the Recall Comparison paragraph duplicating the merged caption; the
thread-count paragraph framed as a retraction of an earlier draft that does not exist in a new
document. The user approved the plan and asked to keep the ANN table. Final result is 4 pages with
prose down to roughly 2,700 words, about a 36% cut, with every measured number, all four rigor
findings and all tables retained. Reaching exactly 4 pages took several compile-measure-trim
iterations; each was verified by `pdfinfo` page count and by rasterising the trailing pages with
`pdftoppm` to see how much was actually spilling, rather than guessing at word counts.

**6. Which note is final.** The user decided the 4-page version is the deliverable. Renamed it to
`design_note.pdf` and the previous 6-page one to `design_note-old.pdf`. The `.tex` files were
renamed to match as well, which was not explicitly requested but was flagged to the user as
necessary: leaving the 6-page source named `design_note.tex` would mean any future
`pdflatex design_note.tex` silently overwrites the 4-page `design_note.pdf` with 6-page output.
Recompiled after renaming to confirm the renamed source still produces 4 pages.

**7. Repository cleanup before submission.** The user asked what was stale and whether to run
`make clean`. Answered no, with the reason: `make clean` is `rm -rf data/interim data/splits
feature_store results`, which would delete `results/`, the committed JSON evidence behind every
number in the design note. Instead surveyed the tree and identified genuine leftovers, which the
user then approved removing individually: the ACL template zip (already extracted), LaTeX
`.aux/.log/.out` artifacts, `.pytest_cache/`, six `__pycache__/` directories, `docs/acl_format.tex`
(the reference template, untracked from git and deleted), and `submissions/`. That last one was
checked before deleting rather than assumed redundant: it held `scoring_result.zip` at 0 bytes and
`prediction_result.zip` containing only an 84-byte `metadata` file with all-null fields, i.e.
Codabench output stubs from a run that never completed, timestamped 40 minutes before the MIND
predictions were even generated. Also removed `src/ablation/`, an empty directory left over from
SPEC.md's original plan (the ablation actually lives in `src/eval/run_ablation.py`).

**8. README rewritten for a zip submission.** The user said the submission is a zip rather than a
git push, which changes what the README needs to do. Two factual errors were found and corrected
rather than carried forward. First, the layout section claimed "`results/` and `predictions/` are
small and are committed intentionally" -- `predictions/` is 1.3 GB and is explicitly gitignored,
with a comment in `.gitignore` saying so, so the README directly contradicted the repo. Second,
the status list still read "EB-NeRD submitted, scoring" and "design note, 3/4 pages, leaderboard
screenshots still to be inserted once EB-NeRD scoring completes", both long since resolved.
Beyond the errors, the quickstart opened with an `scp` to the user's personal `ada` account, which
is meaningless to a grader unzipping the archive, so it was reframed around a local venv and
`make test`. Added a "Start here" table, an explicit table of what is excluded from the archive
and how to regenerate each part, and a section documenting the four rigor benchmark scripts, which
had not been mentioned in the README at all despite being what the professor's grading note asked
for. Verified every relative link resolves, every Makefile target referenced exists, and every
documented command parses -- the last check caught that three of the four benchmark scripts take
`--datasets` (plural, nargs) rather than the `--dataset` that had been written.

**9. A false claim caught by actually running the suite in the packaged copy.** After assembling
`submit/`, running `pytest` there gave 40 passed / 22 skipped, not the 59 passed / 3 skipped the
README asserted. The 22 skips are the tests that read `data/splits/` and `feature_store/`, both
deliberately excluded from the archive for size; they skip cleanly with an explicit
`run make data first` message rather than failing or vacuously passing. This mattered because the
README specifically claimed the leakage and split-boundary guarantees were covered "on synthetic
fixtures", when `test_no_leakage.py` -- the explicitly required Q9.2 test -- is among the skips.
Rewrote that section to state the real counts, name the three files that skip and why, and note
that all 62 run after `make data`. Both numbers were confirmed by running the suite in each
directory rather than reasoning about it.

**10. Viva preparation.** The professor announced that assignment work will be probed in a viva.
Produced `docs/viva_questions.md`, a 66-question bank with answer space, grounded in the actual
implementation by re-reading `metrics.py`, `tokenize.py`, `bm25.py`, `semantic.py`, `slicing.py`,
`beyond_accuracy.py`, `candidates.py` and `split.py` rather than writing generic IR questions.
The user has not answered them yet; the plan is for them to answer and for the answers to then be
reviewed and corrected. This file is intentionally NOT included in the submission archive, since
it is study material rather than an assignment artifact.

**Scope note on this entry**: it was written into `submit/docs/ai_usage_log.md` only, at the
user's instruction. The copy at `retrival-systems/docs/ai_usage_log.md` therefore remains stale
as of this date, ending at the 2026-08-27 entry.

**Human review status**: all layout and page-count claims were verified by compiling and then
rasterising pages with `pdftoppm` and reading them back as images, not by trusting a clean
compile; the 4-page target was confirmed with `pdfinfo` rather than estimated. The two README
factual errors and the test-count discrepancy were found by checking the repo and running the
suite, not by re-reading the README against itself. Every deletion was surveyed and reported
before being run, and each was individually approved by the user; `submissions/` in particular
was inspected for contents first, and `make clean` was recommended against rather than run. All
structural decisions in this period -- which design note is final, removing the figure and
references, keeping the ANN table, what to exclude from the archive -- were explicit user choices,
not unilateral ones.
