# Code_Down_ML

Transaction classification, automated. Upload a GL or transaction export, code a
few rows (or write a few keyword rules), and Code_Down_ML propagates the right
**account code** to every similar transaction. Every automated fill carries a
confidence score and a plain-language reason. Anything uncertain goes to a review
queue. Existing codes are never overwritten.

Spreadsheet-first. Your data stays in an editable grid. Every decision is
auditable. Nothing is a black box.

Runs locally. No login. No external service required.

---

## The problem

Property managers and bookkeepers spend hours every month coding AppFolio,
QuickBooks, and spreadsheet exports onto a chart of accounts. The work is
repetitive, but it is not "train a model and walk away":

- Rows the accountant already coded are **ground truth**. Overwriting them is a
  defect, not a feature.
- A vendor that posts to two accounts is a **collision**, not a coin-flip.
- The next file should get faster because last month's approvals stuck — without
  leaking Client A's memory into Client B's book.
- A fill with no *why* will not survive an audit.

Code_Down_ML is a local, spreadsheet-first cascade: deterministic keyword rules,
then exact learned memory, then similarity, then optional ML. Uncertain rows
go to a human. Approvals write memory. The coded book is the product.

---

## Architecture

```mermaid
flowchart TB
    Upload["Upload CSV / Excel<br/>AppFolio or QuickBooks shape"] --> Loader["data_loader<br/>normalize headers, protect seeds"]
    Loader --> WorkDF["work_df spreadsheet"]
    WorkDF --> Modes{"Run mode"}
    Modes -->|"Strict ★"| RulesOnly["Keyword rules only"]
    Modes -->|"Rules + Memory"| RulesMem["Rules, then exact learned mappings"]
    Modes -->|"Rules + Similarity"| Hybrid["Rules, then TF-IDF / embeddings"]
    Modes -->|"Full Intelligent"| Cascade["Rules → memory → similarity → ML"]
    RulesOnly --> Gate["Confidence gate<br/>auto ≥ 0.85 · review ≥ 0.55"]
    RulesMem --> Gate
    Hybrid --> Gate
    Cascade --> Gate
    Gate -->|auto-filled| Grid["Spreadsheet grid"]
    Gate -->|uncertain| Review["Review workspace"]
    Review -->|approve / recode / reject| Learn["record_human_approval"]
    Learn --> SQLite[("SQLite: rules, mappings,<br/>training_data, profiles")]
    Learn --> Models["Per-client LogReg / optional SetFit"]
    SQLite --> Cascade
    Models --> Cascade
    Grid --> Export["Excel / CSV export<br/>New Account only"]
    Grid --> Insights["Insights — read-only coaching metrics"]
```

Entry point is `main.py` (Streamlit). Engines and storage live under `src/` and
`utils/` with no Streamlit imports, so the same cascade is what `pytest` and
`python smoke_test.py` exercise. Runtime artifacts (SQLite, models, logs) stay
under `FILLDOWN_DATA_DIR` and are gitignored.

---

## How a row gets a code

For each blank row, the engine walks a layered cascade and stops at the first
confident answer. Each fill records which layer decided it, the confidence, and
why.

```mermaid
flowchart TD
    A[Blank row] --> B{Keyword rule matches?}
    B -- yes --> R1[Fill · confidence 0.99 · engine: rules]
    B -- no --> C{Exact text approved before?}
    C -- yes --> R2[Fill · confidence 0.97 · engine: learned]
    C -- no --> D[Similarity: nearest seed rows by embedding cosine]
    D --> E{Trained ML model active?}
    E -- no --> F[Similarity decision stands]
    E -- yes --> G[ML combined per mode: assist / hybrid / prefer_ml]
    F --> H{Confidence}
    G --> H
    H -- ">= auto-fill cutoff (0.85)" --> AF[Auto-filled]
    H -- "review cutoff (0.55) – auto-fill" --> FR[Filled · flagged for review]
    H -- "< review cutoff" --> NQ[Left blank · Review Queue]

    S[Row already coded in upload] --> K[Kept as seed · protected · never overwritten]
```

The layers, in priority order:

1. **Seeds.** Any code already in the file (or typed by you) is kept as-is and
   becomes ground truth the other layers learn from.
2. **Keyword rules.** Deterministic phrase → code mappings you write. Match
   types: `contains`, `exact`, `starts_with`, `ends_with`, `fuzzy`, `regex`.
   Multiple comma-separated phrases per rule (OR logic). Row-wide by default;
   optionally scoped to named columns. Priority-ordered. Stored in SQLite and
   reused across files.
3. **Learned memory.** Exact text-signature matches to codes you approved
   before.
4. **Similarity.** Blank rows are embedded and compared against seed rows.
   Confidence = best cosine similarity weighted by agreement among the similar
   seeds. DBSCAN groups look-alike transactions for visibility.
5. **Trainable ML (optional).** A trained model layers on top of the similarity
   decision when one exists and the mode allows it. High-confidence agreement
   auto-fills; disagreement between ML and similarity is sent to review.

Confidence cutoffs and the similarity threshold live in `config.yaml` and are
adjustable live from the sidebar.

---

## Engines

Built in (install and go):

- **TF-IDF similarity** — char n-gram (3–5) vectorizer from scikit-learn. Fast.
  No downloads.
- **Logistic Regression** — TF-IDF + LogReg pipeline trained on your approvals.
  The always-available ML baseline.

Optional (`pip install -r requirements-semantic.txt`):

- **Semantic matching** — `sentence-transformers` embeddings
  (`all-MiniLM-L6-v2`, ~80 MB, downloads on first use). Matches on meaning, not
  spelling.
- **SetFit** — few-shot transformer classifier. The most accurate trainable
  layer. Trains on CPU.

Every optional engine degrades gracefully: missing packages never crash the app,
it just falls back to TF-IDF + LogReg.

The ML layer has four modes (`auto`, `similarity_only`, `hybrid`, `prefer_ml`).
`auto` is progressive: it leans on similarity when few approved examples exist
and shifts toward the trained model as approvals accumulate (thresholds in
`config.yaml`: `hybrid_min`, `ml_primary_min`).

---

## Design trade-offs

| Choice | What we took | What we gave up |
| --- | --- | --- |
| **Strict-first run modes** | Accountants can fill nothing a rule didn't match | Coverage requires writing (or ingesting) rules |
| **Never overwrite seeds** | Uploaded codes stay sacred in every mode | The engine cannot "fix" a bad seed for you |
| **TF-IDF as the default** | `pip install` stays small; CI stays fast; no model download | Spelling and paraphrase misses that embeddings would catch |
| **Optional semantic / SetFit** | Best-accuracy path exists without forcing torch on every install | Those engines are not in CI; their accuracy is **not measured** here |
| **Client-scoped SQLite + models** | Switching the dashboard client name isolates memory | This is a single-user local app, not a multi-tenant service |
| **Confidence cutoffs 0.85 / 0.55** | High-confidence auto-fill, everything else is review | Tunable, but defaults are conservative on purpose |
| **No auth on this branch** | Clone and run | Do not put it on a public URL as-is (see `demo/railway-live` for a gated live mode) |
| **File ingest, no live GL APIs** | Works on an export you already have | No AppFolio / QuickBooks sync |

---

## How to run

Requires **Python 3.11+**.

```bash
git clone https://github.com/DannyBarren/Code_Down_ML.git
cd Code_Down_ML

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # core engine (pinned)
# optional, heavier: semantic matching + SetFit
# pip install -r requirements-semantic.txt

streamlit run main.py
```

Open <http://localhost:8501>. On the dashboard, click **Load sample dataset**
(neutral, fictional vendors), then **Run Rules — Strict ★ (recommended)** in
the toolbar for a deterministic, fully audited first pass. Rows no rule covers
can then be picked up with **Full Intelligent Run** and finished in the
**Review** workspace.

`python main.py` also works — it relaunches itself under Streamlit. If core
packages are missing, the app shows a setup screen with a one-click installer
instead of crashing.

Environment variables are listed in [`.env.example`](.env.example). The app
reads the process environment; it does not load a `.env` file automatically.

```bash
# throwaway database (keeps shipped data/ clean)
FILLDOWN_DB_PATH=/tmp/demo.db streamlit run main.py

# public-demo mode (banner + optional wipe on start — not for real books)
FILLDOWN_DEMO=1 streamlit run main.py
```

---

## Run modes

Four modes in the spreadsheet toolbar, ordered by trust (recommended first):

| Mode | What it uses | When |
| --- | --- | --- |
| **Run Rules — Strict ★ (recommended)** | Only your enabled keyword rules, blank rows only | Deterministic and auditable. Fills nothing a rule didn't match. |
| **Rules + Memory** | Strict rules, then exact matches you approved before | Deterministic coverage from past approvals. |
| **Run Rules + Similarity** | Rules first, then similarity spread from rule-filled + coded rows | Propagate rule codes to look-alike transactions. |
| **Full Intelligent Run** | Rules + learned memory + ML + similarity, whole file | Broadest coverage in one click. Can fill rows no rule matched — review those. |

Strict mode never leaks across accounts. Existing values are never overwritten
in any mode. When the nearest seed rows disagree on a code, the row goes to
review with the vote split instead of being silently filled. **Reset run**
un-runs the last pass and keeps your rules, notes, and manual edits.

---

## Review and learning

- The **Review workspace** (sidebar) shows only rows that need a human, least
  confident first, with the code, engine, confidence and a plain-English *why*
  on every row. Similarity groups approve in one click (split groups never
  do). Bulk actions approve everything visible or above a confidence cutoff;
  selected rows can be re-coded or rejected in bulk. Rejecting blocks the bad
  pairing from being suggested again — and never learns it.
- Approving a row writes the code, protects it, and records three things: a
  learned mapping (instant exact-match memory), a training example (feeds the
  ML models) and a reinforcement of the account's knowledge profile.
- **Rule Notes** is a per-row free-text column. Notes fold into the similarity
  text and persist across uploads via a stable row signature. They are never
  mined for rule keywords — suggestions come from Name, then Memo, then
  Description.
- The **Account Knowledge** base learns what each account code looks like —
  keywords, sample transactions, usage count — and adds that context to match
  rationales.
- **Models** panel: engine status, a memory inspector (mappings, examples,
  last train, held-out accuracy, delete-a-bad-mapping), and a quiet "train
  now" reminder after every 25 new approvals. **History** panel: past runs.
- Rules, learned mappings, training data and account profiles are scoped per
  client (the dashboard's client/project name), so switching clients never
  leaks one book's memory into another's.

---

## Export

- **Excel** — reopens your original workbook and writes only the `New Account`
  column back. Formatting, formulas, column order, and other sheets stay intact.
  Adds a **Code Down Summary** tab.
- **CSV** — tidy, internal columns stripped, codes normalized, optional derived
  `New Account (Base)` column. Ready to import into accounting/ERP tools.

---

## Configuration

All thresholds and column mappings live in **`config.yaml`**. Most are also
adjustable from the sidebar's **Advanced tuning** section.

- `similarity` — embedding model, similarity threshold (0.72), clustering,
  numeric-token stripping.
- `confidence` — auto-fill cutoff (0.85), review cutoff (0.55), rule/learned
  match confidences.
- `ml` — enable/disable, mode, confidence cutoff, SetFit settings, progressive
  thresholds, model store directory, `auto_train_every` (approvals between
  train reminders, default 25).
- `columns` — header variants recognized as the target `New Account` column, the
  text columns used for similarity (vendor-first: Name, Memo, Description…),
  amount and date columns, plus `keyword_source` / `keyword_source_exclude`
  controlling which columns rule suggestions are mined from (notes are never
  mined unless explicitly opted in).
- `account_glossary` — optional NARPM-style code → name/keywords mapping that
  enriches rationales. Empty by default; never required.
- `storage` / `logging` — SQLite path, work dir, log file.

Runtime data (SQLite DB, trained models, caches, logs) goes under a writable
data directory: `FILLDOWN_DATA_DIR` if set, else `./data/`, else a temp dir.
The app never crashes on a read-only filesystem.

See [`.env.example`](.env.example) for the full list. The ones operators set:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FILLDOWN_DATA_DIR` | `data/` | Writable root for DB / models / logs. |
| `FILLDOWN_DB_PATH` | under data dir | Override the SQLite DB path (used by tests). |
| `FILLDOWN_DEMO` | `0` | `1` enables public-demo behavior (banner, sample auto-load, fresh-state reset). |
| `FILLDOWN_DEMO_RESET` | `1` in demo | `0` keeps data across restarts in demo mode. |
| `FILLDOWN_MAX_ROWS` | `0` | Soft per-upload row cap (`0` = unlimited). |

There is no authentication gate. The app opens straight to the dashboard.

---

## Docker

```bash
docker build -t code-down-ml .
docker run -p 8501:8501 code-down-ml
```

The image builds on the core requirements (TF-IDF + LogReg), exposes port 8501,
and sets `FILLDOWN_DEMO=1` with the data dir at `/data` — mount a volume there
to persist rules and learning.

---

## Project structure

```
code_down_ML/
├── main.py                     # Streamlit entry point (thin router)
├── cli.py                      # `code-down` console script (pip install -e .)
├── config.yaml                 # thresholds + column mapping
├── requirements.txt            # core engine, pinned
├── requirements-dev.txt        # core + pytest (what CI installs)
├── requirements-semantic.txt   # optional semantic AI + SetFit
├── pyproject.toml              # packaging + pytest config
├── .env.example                # documented environment variables
├── Dockerfile                  # container build (port 8501)
├── .streamlit/config.toml      # theme + server settings
├── src/                        # engines and business logic
│   ├── config.py               #   typed config, env vars, data-dir selection
│   ├── data_loader.py          #   read/normalize files, resolve New Account
│   ├── ingest_profiles.py      #   AppFolio/QuickBooks export-shape detection
│   ├── insights.py             #   read-only coaching metrics (pure functions)
│   ├── rules_manager.py        #   keyword rules, matching, account knowledge
│   ├── fill_down_engine.py     #   the decision cascade + confidence scoring
│   ├── similarity.py           #   embeddings (semantic or TF-IDF) + grouping
│   ├── ml_classifier.py        #   LogReg + SetFit training, registry, predict
│   ├── review_queue.py         #   build/apply the human review queue
│   ├── spreadsheet_helpers.py  #   work_df logic: runs, audit, reset, filters
│   ├── dependencies.py         #   dependency probes + in-app pip installer
│   └── exporter.py             #   Excel/CSV export
├── ui/                         # Streamlit views
│   ├── landing.py              #   dashboard: client name, upload, sample data
│   ├── spreadsheet.py          #   main grid, toolbar, run modes, export, append
│   ├── review.py               #   review workspace: groups, bulk approve/reject
│   ├── insights.py             #   read-only coaching metrics page
│   ├── panels.py               #   Rules / Models / History modals
│   ├── rule_creation.py        #   rule builder dialog + suggestion banners
│   ├── sidebar.py              #   nav, engine status, advanced tuning
│   ├── common.py               #   bootstrap, session state, demo mode, CSS
│   ├── setup.py                #   dependency installer screens
│   └── guide.py                #   in-app user guide
├── utils/
│   ├── storage.py              #   SQLite: rules, memory, notes, training, runs
│   ├── account_codes.py        #   account-code parsing and normalization
│   ├── sample_data.py          #   neutral sample dataset generator
│   ├── demo_utils.py           #   demo-mode data reset
│   └── logging_setup.py        #   structlog configuration
├── models/schemas.py           # pydantic models
├── data/sample_transactions.*  # neutral sample ledger (fictional vendors)
└── tests/                      # pytest suite + smoke_test.py
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                   # unit + integration + headless UI tests
python smoke_test.py     # end-to-end smoke test without Streamlit
```

The suite covers the data loader, rules and matching, the decision cascade, the
ML layer, review and export, the spreadsheet helpers, and end-to-end UI flows
via Streamlit's `AppTest`. Tests use a throwaway SQLite database
(`FILLDOWN_DB_PATH`) and never touch shipped data.

---

## Measured results

Numbers below come from commands run in this repo on **2026-10-19** (Python
3.12.3, pinned core stack: pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1,
streamlit 1.66.0). Anything not produced by those commands is marked
**not measured**.

| Metric | Result | Command |
| --- | --- | --- |
| Pytest (unit + integration + headless AppTest) | **224 passed in 10.80s** | `pytest` |
| Smoke test (no Streamlit) | **all checks passed** | `python smoke_test.py` |
| Neutral sample ledger, Full Intelligent, TF-IDF, default example rules only | 220 rows · 16 groups · **142 auto-filled** · 0 filled-for-review · 0 needs-review · **68 left blank** (10 rows were already seeded; 142/210 blanks filled) | `python smoke_test.py` (sample workflow on `data/sample_transactions.xlsx`) |
| Precision / recall / F1 vs a labeled production ledger | **not measured** | — |
| Held-out LogReg / SetFit accuracy | **not measured** | tests assert class identity on synthetic data; they do not record a score |
| Semantic embeddings / SetFit | **not measured** | optional stack is not installed in CI |

These sample-file counts are **coverage on a fictional 220-row book**, not
accuracy. Default example rules (`Cloud Hosting`, `SaaS Tools`) plus TF-IDF
similarity from the 10 seeded rows produced the 142 auto-fills. Do not read
that as production quality.

---

## Known limitations

- **No authentication** on this branch. Fine for local use; not a public SaaS.
- **Single-user SQLite.** No concurrent writers, no hosted multi-tenant ACL.
- **Optional semantic / SetFit** are heavier, not in CI, and unmeasured here.
- **TF-IDF is n-gram based.** Vendor typos and paraphrase often need a keyword
  rule or the optional embedding backend.
- **Similarity needs seeds or rule-fills.** A file with no coded rows and no
  matching rules stays mostly blank — by design.
- **No live accounting APIs.** AppFolio / QuickBooks are ingest *shapes*, not
  connectors. Insights are read-only over the file you uploaded.
- **Excel export writes `New Account` (plus a summary tab).** Other sheets and
  formatting are preserved; the engine does not rewrite the rest of the book.
- **`FILLDOWN_DEMO=1` can wipe the knowledge base on start.** Do not enable
  demo reset against a real client's SQLite file.
- **Published accuracy on real client ledgers is intentionally absent.** Client
  books stay out of git.

---

## License

MIT — see [LICENSE](LICENSE). Copyright 2026 Danny Barren.
