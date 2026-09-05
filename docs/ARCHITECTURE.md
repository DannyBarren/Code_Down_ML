# Architecture — Code_Down_ML

Written from the code in this repository. If this file and the code ever
disagree, the code is right — fix this file.

## Big picture

Code_Down_ML is a Streamlit app that assigns account codes to transactions.
A user uploads an Excel/CSV export, and the app fills the blank cells of one
canonical column — **`New Account`** — using a layered decision cascade. Every
automated fill records the engine that made it, a confidence score, and a
plain-language rationale. Rows the engine is unsure about go to a review queue;
human approvals become the memory that makes the next file faster.

Layers, in the priority order the engine applies them per row:

1. **Seed** — a value already present in the upload (or typed by the user) is
   kept as-is. Seeds are protected and never overwritten.
2. **Keyword rules** — deterministic phrase → code rules, persisted in SQLite.
   When a rule disagrees with learned memory, the rule wins and the rationale
   says so.
3. **Learned memory** — an exact text-signature lookup built from past
   approvals. Pairings the user rejected (`blocked_mappings`) never fire.
4. **Similarity** — embeddings + cosine distance to the nearest seed rows;
   DBSCAN clusters look-alike transactions into groups. When the nearest seeds
   disagree on the account, the row is sent to review with the vote split
   instead of being auto-filled; a weak majority caps the fill at
   filled-for-review.
5. **Trainable ML (optional)** — LogReg (always available) or SetFit
   (optional), layered on top of the similarity decision.

Confidence gating (defaults in `config.yaml`): `>= auto_apply_cutoff` (0.85) →
auto-filled; `>= review_cutoff` (0.55) → filled but flagged for review; below →
left blank and sent to the review queue.

## File-by-file

### Entry points

- **`main.py`** — Streamlit entry point and thin router. Forces the PyTorch-only
  Hugging Face backend env vars, relaunches itself under `streamlit run` when
  invoked as `python main.py`, gates on core dependencies (shows the setup
  screen when missing), boots shared services, renders the sidebar, and routes
  between the landing view, the spreadsheet view, and the management panels.
- **`cli.py`** — console script (`code-down`, via `pip install -e .`) that
  shells out to `streamlit run main.py`.
- **`smoke_test.py`** — end-to-end smoke test with no Streamlit: empty file,
  no-seeds file, column variations, the full sample workflow, the spreadsheet
  pipeline, candidate rules, rule reliability, reset controls.

### `src/` — engines and business logic (no Streamlit imports)

- **`config.py`** — Pydantic-validated config loaded from `config.yaml`.
  `is_demo()` / `demo_reset_enabled()` read the demo env vars.
  `select_data_dir()` picks the first writable data directory
  (`FILLDOWN_DATA_DIR` / `HF_DATA_DIR` → `./data/` → temp dir). Path helpers
  resolve the DB, work dir, log file, and model store.
- **`data_loader.py`** — reads `.xlsx`/`.xls`/`.csv` (encoding-tolerant),
  resolves messy real-world headers to the canonical `New Account` column
  (creating it when absent), normalizes codes, and builds `_sim_text`: the
  combined, cleaned transaction text used for similarity and ML. Also builds
  `_base_sig` (the same text without Rule Notes) so notes re-attach across
  uploads, and `summarize_upload()` for the coded-vs-blank readout.
- **`rules_manager.py`** — CRUD + matching for keyword rules. Matching is
  punctuation/case/spacing tolerant, supports comma-separated phrases (OR),
  `contains`/`exact`/`starts_with`/`ends_with`/`fuzzy`/`regex` match types,
  row-wide search by default, and optional column scoping with tolerant header
  resolution. Fuzzy matching is token-aware: it tolerates vendor misspellings
  but never matches a token inside a longer word (an address containing
  "shawnee" does not fire a "Shaw Media" rule). `apply_rules_to_dataframe()`
  is the deterministic blank-only fill path with a full per-row audit
  (`protected_existing` / `keyword_rule` / `no_rule_match`), and
  `near_miss_suggestions()` surfaces rows that almost matched a rule after a
  strict run (suggested rule edits — never auto-filled). Also hosts the
  **account knowledge base**: per-account profiles (keywords, sample texts,
  usage counts) reinforced on every rule save, rule run and human approval,
  plus `enrich_results()` which appends that learned context to rationales.
  `record_human_approval()` is the single learning path every approval flows
  through (learned mapping + training example + profile reinforcement).
- **`fill_down_engine.py`** — the orchestration core. `FillDownEngine.run()`
  embeds and clusters all rows, then decides each row in priority order:
  seed → rule → learned mapping → similarity, with the ML prediction layered
  onto the similarity decision (`_combine_with_ml`) according to the effective
  mode (`assist` / `hybrid` / `prefer_ml` / `similarity_only`). ML is strictly
  additive: disabled, untrained, or erroring ML falls back to pure similarity.
  Also exposes `run_selected_rules()` for the deterministic rule-only path.
- **`similarity.py`** — the embedding layer. Uses `sentence-transformers`
  (`all-MiniLM-L6-v2`) when installed and enabled; otherwise falls back to a
  scikit-learn TF-IDF char n-gram (3–5) vectorizer. Groups transactions with
  DBSCAN on cosine distance (`eps` derived from the similarity threshold).
- **`ml_classifier.py`** — the trainable layer. `LogRegModel` (TF-IDF +
  Logistic Regression, always available) and `SetFitModel` (optional few-shot
  transformer). `ModelManager` trains both, versions them under
  `data/models/`, tracks them in `registry.json`, picks the active model by
  held-out accuracy, and resolves `auto` mode from the number of approved
  examples (`hybrid_min` / `ml_primary_min`). Prediction never raises — no
  model means "no prediction", and the engine falls back to similarity.

#### Model storage after 1.1

Models are scoped per client, with the global store as the fallback:

```
data/models/registry.json                 # global registry
data/models/logreg/vN.joblib              # global LogReg versions
data/models/setfit/vN/                    # global SetFit versions
data/models/clients/<client_id>/registry.json
data/models/clients/<client_id>/logreg/vN.joblib
data/models/clients/<client_id>/setfit/vN/
```

`<client_id>` is a slugified client/project name (no path separators).
Training a client manager reads only that client's `training_data` plus the
shared pool; training the global manager reads everything (unchanged).
Prediction order when a client is active: **client model** (used when it
returns a label with confidence ≥ `confidence.review_cutoff`) → **global
model** (same cutoff) → **similarity** (the existing engine fallback). A
missing, untrained or corrupt model simply yields "no prediction" — nothing
ever raises. When no client is selected, only the global model is consulted.
- **`review_queue.py`** — builds the editable review table from flagged results
  (`FILLED_REVIEW`, `NEEDS_REVIEW`), sorted least-confident first, and applies
  the user's approvals back onto the working dataframe through the single
  learning path.
- **`spreadsheet_helpers.py`** — the pure, testable logic behind the
  spreadsheet UI. Owns the `work_df` conventions (internal columns are
  `_`-prefixed: `_confidence`, `_engine`, `_action`, `_why`, `_suggested`,
  `_select`, `_group`), the run modes (`run_full`, `run_selected_rules_audited`,
  `run_rules_plus_memory`, `run_rules_hybrid`, `run_learned_memory`),
  reset/un-run, bulk approve, manual-edit commits, rule candidate mining
  (keyword-source ordered: Name → Memo → Description → Payee → …, note-like
  columns excluded), live rule preview with blank-vs-protected counts, the
  review-workspace helpers (`review_rows_df`, `apply_review_table`,
  `group_review_summary`, `approve_similarity_group`, `reject_rows`,
  `apply_suggested_to_rows`, `recode_rows`), filters/search/pagination, and
  undo/redo snapshots.
- **`dependencies.py`** — stdlib-only dependency probes and the streamed
  in-app pip installer (installs only missing packages, pins
  `transformers>=4.41,<5` for SetFit compatibility, CPU torch index).
- **`exporter.py`** — three export strategies: write `New Account` back into a
  copy of the original workbook (formatting preserved, plus a "Code Down
  Summary" sheet), write a clean workbook from scratch, or emit a tidy CSV
  (internal columns stripped, codes normalized, optional `New Account (Base)`
  column).

### `ui/` — Streamlit views

- **`landing.py`** — dashboard: optional client/project name, file upload,
  sample-data loader, recent runs, demo controls.
- **`spreadsheet.py`** — the main workspace: sticky toolbar (run modes ordered
  strict-first, approve, undo/redo, export), filter bar (quick views, global
  search, engine and confidence filters), paginated `st.data_editor` with
  auto-save, post-run metrics and rule-run audit panels (including near-miss
  suggestions), and the export dialog.
- **`review.py`** — the Review workspace: only rows needing a human, sorted
  least-confident first; similarity-group cards with one-click group approve
  (split groups are never one-click approved); bulk approve visible /
  above-confidence; apply-suggested / recode / reject for selected rows; a
  batched editable table (one Apply click per review session, not per row);
  promote-to-rule on any row.
- **`panels.py`** — Rules, Models, and History management modals, including
  human-reviewed rule creation from pre-filled rows, suggested rules mined from
  coded rows, the account-knowledge view, and model training.
- **`rule_creation.py`** — the rule-builder dialog with live match preview, the
  selection strip, the inline "create rule?" prompt after a manual edit, and
  the seed-suggestion banner.
- **`sidebar.py`** — navigation, engine status, advanced tuning (similarity
  threshold, confidence cutoffs, similarity columns, semantic toggle).
- **`common.py`** — cached bootstrap (config, logging, storage, model manager,
  rules), session-state defaults, undo/redo stacks, demo-mode helpers and
  banner, sample-data loading, and shared CSS.
- **`setup.py`** — dependency status screens and the one-click in-app
  installer.
- **`guide.py`** — self-contained in-app user guide (full-page takeover).

### `utils/`, `models/`, `data/`

- **`utils/storage.py`** — the SQLite layer (stdlib `sqlite3`, thread-safe).
  Tables: `rules`, `learned_mappings`, `blocked_mappings`, `rule_notes`,
  `training_data`, `run_history`, `account_profiles`. `learned_mappings` and
  `training_data` carry a `client_id` (`''` = shared/default client) for
  per-client isolation; databases created before that column existed are
  migrated in place without losing rows. `blocked_mappings` records rejected
  signature → code pairings so they are never re-suggested. `reset_all()`
  backs demo-mode resets.
- **`utils/account_codes.py`** — account-code parsing and normalization
  (`6100 a` / `6100-A` / `6100.0` → `6100A`), base-account and sub-account
  helpers.
- **`utils/sample_data.py`** — generates the neutral sample dataset (fictional
  vendors across common spend categories) used by the dashboard button and the
  smoke test.
- **`utils/demo_utils.py`** — Fresh Demo Mode: wipes DB rows, trained models,
  logs, and work files on startup when demo reset is enabled. Never runs
  off-demo.
- **`utils/logging_setup.py`** — structlog configuration (console + rotating
  file), quiets noisy third-party loggers.
- **`models/schemas.py`** — Pydantic models: `KeywordRule`, `FillResult`,
  `FillAction`, `FillSource`, `RuleRunResult`, `LearnedMapping`,
  `TrainingExample`, `TransactionGroup`, `ModelStatus`, `RunSummary`.
- **`data/sample_transactions.csv` / `.xlsx`** — the tracked sample ledger.
  Fictional vendors only; safe to ship.

## How a row gets a code (full path)

1. **Load.** `data_loader.load_dataframe()` reads the file, resolves or creates
   `New Account`, normalizes existing codes, and builds `_sim_text`.
   `spreadsheet_helpers.build_work_df()` adds `Rule Notes` and the `_`-prefixed
   meta columns, and re-attaches any previously saved notes via `_base_sig`.
2. **Run.** The toolbar offers four modes (recommended first):
   - *Run Rules — Strict ★* → `run_selected_rules_audited()` — blank rows only,
     only the enabled rules' codes, full audit. No similarity, no ML.
   - *Rules + Memory* → `run_rules_plus_memory()` — strict rules, then learned
     exact-match memory. Deterministic, blank-only, audited.
   - *Run Rules + Similarity* → `run_rules_hybrid()` — strict rules first, then
     the rule-filled and pre-coded rows become seeds for a similarity pass.
   - *Full Intelligent Run* → `FillDownEngine.run()` (the full cascade).
3. **Decide.** Inside the engine, each blank row stops at the first confident
   layer: rule match (0.99) → learned exact match (0.97) → similarity score
   (best cosine × seed agreement), optionally overridden or confirmed by the ML
   prediction per the active mode. Confidence cutoffs decide between
   auto-filled, filled-for-review, and left-blank-for-review.
4. **Explain.** Every decision lands in the meta columns (`_engine`,
   `_confidence`, `_why`) and in per-run audit structures, so the UI can show
   exactly why each code was chosen — including rows a rule matched but
   couldn't touch because they were already coded.

## Review writes memory

Approving a row (review queue, bulk approve, or a manual edit in the grid)
does three things:

1. Writes the normalized code into `New Account` and marks the row as
   user-owned (`engine = manual` / seed), which protects it from future runs.
2. Upserts a **learned mapping** (`_sim_text` signature → code) — instant
   exact-match memory used by layer 3 on the next file.
3. Adds a **training example** to `training_data` — the labelled set the
   LogReg/SetFit models train on from the Models panel.

Rule Notes are persisted separately, keyed by `_base_sig`, so they re-attach to
the same transactions on future uploads and fold into `_sim_text` (and thus
into similarity and ML training).

## Environment variables

| Variable | Read in | Default | Purpose |
| --- | --- | --- | --- |
| `FILLDOWN_DATA_DIR` | `src/config.py` | unset | Writable root for DB / models / logs / cache. |
| `HF_DATA_DIR` | `src/config.py` | unset | Same, for Hugging Face persistent storage (`/data`). |
| `FILLDOWN_DB_PATH` | `ui/common.py` | unset | Override the SQLite path (tests use a throwaway DB). |
| `FILLDOWN_DEMO` | `src/config.py` | `0` | `1` = public-demo behavior (banner, sample auto-load). |
| `FILLDOWN_DEMO_RESET` | `src/config.py` | `1` in demo | `0` keeps demo data across restarts. |
| `FILLDOWN_MAX_ROWS` | `ui/common.py` | `0` | Soft per-upload row cap (`0` = unlimited). |
| `SPACE_ID` / `HF_SPACE_ID` | `src/config.py` | unset | Presence implies demo mode (HF Spaces). |

The Dockerfile sets `FILLDOWN_DEMO=1`, `FILLDOWN_DATA_DIR=/data`,
`FILLDOWN_MAX_ROWS=20000`, and `HF_HOME=/data/.huggingface`.

## What stays out of git

Enforced by `.gitignore` / `.dockerignore`:

- `.env`, `.venv/`, `__pycache__/`, `*.pyc` — local environment.
- `*.db`, `*.sqlite`, `data/*.db-journal` — the runtime SQLite knowledge base.
- `data/models/`, `checkpoints/`, `*.bin`, `*.pt`, `*.safetensors`,
  `*.joblib`, `*.pkl` — trained model artifacts. Models are retrained from
  approvals; weights are never committed.
- `data/work/`, `data/*.log`, `data/uploads/` — runtime scratch and logs.
- `.streamlit/secrets.toml` — local secrets.
- `DEVELOPMENT ONLY*.csv`, `*Transaction Detail*.csv`, `.~lock.*#` — real
  client exports and editor lock files. Never commit these.

Tracked data is limited to the neutral sample ledger
(`data/sample_transactions.*`) and `data/.gitkeep`.
