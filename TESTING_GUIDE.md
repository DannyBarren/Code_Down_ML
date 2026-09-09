# Testing Guide — Code_Down_ML

This guide covers (1) the automated test suite, (2) a manual test matrix for the
spreadsheet-dominant UI, and (3) a one-page **Demo Script** for the presentation.

---

## 1 · Environment

```bash
# From the project root
python3 -m venv .venv            # create an isolated environment
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # core engine
pip install pytest               # test runner
# Optional (heavy): best-accuracy semantic AI + SetFit
# pip install -r requirements-semantic.txt
```

> The app and tests run fine on the **core** stack alone (TF-IDF matcher). The
> semantic/SetFit add-ons are optional and the app falls back automatically.

---

## 2 · Automated tests (run after every change)

```bash
pytest tests/ -q          # unit + integration + headless UI e2e
python smoke_test.py      # non-Streamlit end-to-end smoke (happy path + edges)
```

What's covered:

| Suite | File | Focus |
|-------|------|-------|
| Data loader | `tests/test_data_loader.py` | header resolution, code normalisation, `_sim_text` |
| Engine | `tests/test_fill_down_engine.py` | seed/rule/learned/similarity decision priority |
| ML layer | `tests/test_ml_classifier.py` | hybrid/prefer-ml modes, fallbacks |
| Review/export | `tests/test_review_and_export.py` | apply reviews, CSV/Excel exports |
| Rules / codes | `tests/test_rules_manager.py`, `tests/test_account_codes.py` | matching, suffix normalization |
| **Spreadsheet helpers** | `tests/test_spreadsheet_helpers.py` | `work_df` build, Rule-Notes folding + persistence, runs, manual edits, bulk approve, rule preview, filters, pagination, undo/redo |
| **Production hardening** | `tests/test_production_hardening.py` | keyword source order (never Notes), honest rule previews, token-aware fuzzy, seed-disagreement → review, reject-never-learns, group approve blanks-only, bulk-recode protection, client isolation + migration, learned-memory mode, strict-mode purity, reset/export regression locks |
| **UI e2e** | `tests/test_e2e_v22.py` | headless `AppTest`: landing → load sample → spreadsheet → full run → filters/pagination → modal panels → export modal → undo |
| **Review inbox** | `tests/test_review_inbox.py` | Keep / Not this from the Review page with no spreadsheet `_select`; reject never learns; seeds sacred |
| **Review inbox e2e** | `tests/test_review_inbox_e2e.py` | headless `AppTest` on the inbox itself: renders after Strict and with the leftover table shown, "Queue is clear", Keep codes + learns, Not this leaves blank and never learns, coded rows survive a pile Keep, split piles refuse one-click Keep, "Always code this vendor" does not raise |
| **Ingest profiles** | `tests/test_ingest_profiles.py` | AppFolio / QuickBooks detection, append dedupe, append never overwrites, Notes never mined |
| **Insights** | `tests/test_insights.py` | read-only cards; never writes `New Account`; survives a missing Amount column |
| **Live instance** | `tests/test_live_session.py` | password gate, fail-closed data dir, workspace save/resume, Start fresh is session-only, live wins over demo, durable client name, row cap refuses instead of truncating |
| **Demo reset** | `tests/test_demo_reset.py` | the demo wipe stays on demo and can never run on live |

> The e2e modules and the bootstrap honour `FILLDOWN_DB_PATH`, so tests
> use a throwaway SQLite file and never touch the shipped `data/fill_down.db`.

Expected result: **all tests pass** and `smoke_test.py` prints
`ALL SMOKE TESTS PASSED ✅`.

---

## 3 · Manual test matrix (browser)

Launch with `streamlit run main.py`. Tip: use a clean DB with
`FILLDOWN_DB_PATH=/tmp/demo.db streamlit run main.py`.

### Happy path
1. **Landing** → enter a client name → **Load sample data & start**. ✅ App opens
   directly in the spreadsheet; metrics + progress bar populate. A banner
   offers to ingest the coded example rows as exact rules.
2. Click **Run Rules — Strict ★** (the recommended button). ✅ Only blank rows
   fill, only with your rules' codes; the audit panel shows protected / filled
   / still-blank plus near-miss suggestions; the grid filter jumps to the
   rule-filled rows.
3. Click **Rules + Memory**. ✅ Deterministic: rules, then exact matches you
   approved before. No similarity, no AI.
4. Click **Full Intelligent Run**. ✅ Codes fill in; **Confidence** bars and
   **How decided** appear; rows whose nearest seeds disagree land in review
   instead of being silently filled.
5. **Review inbox** (sidebar → Review, or the CTA after a run). ✅ Three verbs
   on one card — **Keep** the suggestion, **Fix** it by typing an account, or
   **Not this**. Least-sure first. Look-alike piles offer one-click Keep or
   Not this for the whole pile; a pile whose rows disagree refuses one-click
   Keep. Keep and Fix are remembered for this client; **Not this never
   learns** — it blocks that pairing and leaves the row blank. Already-coded
   rows inside a pile are protected, never overwritten. When nothing is left,
   the page says **"Queue is clear"** and offers Export.
   After a Strict run there are no suggestions to Keep, so the queue is blank
   rows: turn on **"no suggestion yet"** (or use the "Review N blank rows →"
   CTA) and Fix them from the same card.
   The leftover table is optional — above 20 leftovers it is off by default so
   the next-leftover path stays fast; toggle it on to tick Keep/Pick in bulk.
6. Tick a few **✓** boxes in the grid → **Approve Selected**. ✅ Rows resolve;
   a toast reports "Approved N · learned N · queued for next model train".
   This is a bulk shortcut only — Review decisions never require it.
7. **✨ Create Rule from Selection** (select a vendor's rows first). ✅ Keyword +
   Rule Notes pre-fill; the live preview shows *blank* fills vs already-coded
   matches separately; a rule that fills 0 blank rows needs explicit
   confirmation to save.
8. **📤 Export** (toolbar). ✅ A modal opens *over* the grid; three downloads
   build with client + date in the filenames; open the Excel — original
   formatting kept + a **Code Down Summary** tab (run mode, per-engine counts,
   timestamp); the exported CSV has no `_`-prefixed columns.
9. **⚙️ Rules / 🧠 Models / 🕘 History** (sidebar). ✅ Each opens as a **modal**;
   the spreadsheet stays visible underneath and is never replaced. The Models
   modal hosts the **memory inspector** (mappings, examples, last train,
   accuracy, delete-a-mapping) and the **train reminder** banner after 25 new
   approvals.
10. **Suggested rules (seeding workflow).** Code a few blank rows by typing a
   Target Account. ✅ A **💡 banner** appears ("turn coded rows into N rules").
   Open **⚙️ Rules** → the **Suggested rules** table lists keyword → code with a
   "rows it would fill" count. Tick some → **Create selected rule(s)**. ✅ Rules
   are created; running them fills the look-alike rows. Suggestions are mined
   from Name → Memo → Description — never from Notes.
11. **Engine status.** Sidebar **🧩 Engine status** shows three lines —
   **Core / Semantic / SetFit** — with **⬇️ Install missing AI engines** when any
   optional engine is absent. ✅ The terminal prints a startup banner listing
   exactly what loaded.

### Edge cases
- **Missing columns:** upload a CSV with **no** `New Account` column. ✅ A
  Target Account column is created automatically; a Rule Notes column appears.
- **No seeds:** upload a file with all codes blank. ✅ Run still works via rules /
  learned memory; unmatched rows stay blank (no crash).
- **Large file:** load 5k–20k rows. ✅ Pagination controls appear; switching
  pages and filters stays responsive; edits on one page survive page changes.
- **Rule Notes persistence:** add a note, run, re-upload the *same* file. ✅ The
  note re-appears on the matching row (matched by base signature).
- **Undo/redo:** after a run or bulk approve, click **↩️ Undo** then **↪️ Redo**.
  ✅ Target Account / Rule Notes revert and re-apply.
- **Refresh:** reload the browser tab. ✅ Rules, learned mappings, saved Rule
  Notes and run history persist (they live in SQLite). *In-session undo history
  and the loaded file are not restored — re-open the file to continue.*

---

## 4 · One-page Demo Script (≈5 minutes)

**Setup (before the room):** `FILLDOWN_DB_PATH=/tmp/demo.db streamlit run main.py`,
browser open on the dashboard, sidebar visible.

1. **The problem (15s).** "Classifying a transaction export into the right account
   codes is hours of manual copy-paste in Excel. Watch this."
2. **Land + load (20s).** Type a client name → **Load sample data & start**.
   "One screen — a real spreadsheet. 220 transactions, a few already coded as
   examples."
3. **One click (30s).** **Run Rules — Strict ★**. "Deterministic — only my
   rules, only blank rows, and the audit shows exactly which rule filled which
   row." Then **Full Intelligent Run** for the rest: "It grouped look-alike
   transactions and filled the **Target Account** codes — with a confidence bar
   and a plain-English reason on every row."
4. **Stay in control (45s).** Sidebar → **Review**. "It only asks about what
   it's unsure of — least sure first, and look-alikes are stacked into one
   pile." **Keep** a whole pile in one click; **Fix** one by typing the
   account; **Not this** on one you disagree with. "Notice 'Examples learned'
   just went up — it's getting smarter. And 'Not this' is never learned from,
   so a wrong guess can't teach it the wrong thing."
5. **Teach a rule (45s).** Select a vendor's rows → **✨ Create Rule from
   Selection**. "Pre-filled keyword, and a **live preview**: this rule will
   touch *N* rows." Save. "That rule is now permanent for every future file."
6. **Rule Notes = data asset (30s).** Add a Rule Note on a tricky row. "These
   hints improve the matching **and** are remembered next time we get a similar
   file — turning clean-up into compounding intelligence."
7. **Deliver (30s).** **📤 Export** → Excel keeps their formatting + a summary
   tab; CSV is ready to import. "Audit-friendly, and done in minutes."
8. **Close (15s).** "Reliable from day one with the built-in matcher; an optional
   AI model trains on your approvals and takes over as confidence grows. You are
   always in control."

**If asked about reliability:** "190+ automated tests, including a headless run
of the real UI, plus an end-to-end smoke test — all green."
