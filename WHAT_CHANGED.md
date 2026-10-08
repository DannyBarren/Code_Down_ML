# What changed — ingest per GL, collisions, sheet picker, auto-approve, 36→44

Branch: `cursor/demo-railway-live-ingest-collisions-8d34` → PR into
`demo/railway-live` only. Nothing touches `main`.

## Verification pass — 2026-10-08

The 2026-09-19 commits on this branch were re-checked against the brief from
a clean environment rather than taken on trust. Everything below was
confirmed to be real: the suite collects and passes, the six behaviours are
implemented, `utils/account_codes.py` is byte-identical to the target branch,
`models/schemas.py` still has exactly the five `FillAction` values (collisions
ride `NEEDS_REVIEW` + `engine_used="collision"`), `tests/fixtures/
chrysalis_like.csv` is untouched so the `protected_existing == 187` lock
holds, no real client ledger is tracked, and the claimed 271 → 305 test
counts are accurate.

One real gap was found and fixed: `FillDownEngine.run` had a learned-memory
carve-out that let a collision-keyed blank row auto-fill in the memory modes.
The brief forbids that in every mode, so it was removed (see Edit B). That is
the only behavioural change in this pass.

## Where this started

`origin/demo/railway-live-opus-reliability` (`e548093`, PR #7 draft) was
ahead of `demo/railway-live` (`55b593c`) and green (271 tests vs 253), so
this branch is based on it, as the brief allowed. PR #4, #5 and #7 are left
alone.

## Edit A — Ingest per GL account, not per row

`RulesManager.ingest_rules_per_account()` (and the kept wrapper
`ingest_existing_new_account_as_rules()`, which now returns rules touched):

* Groups coded rows by canonical `New Account`.
* One rule per account. Phrases = every distinct Name on that account, then
  every distinct Memo / Memo/Description term. Numeric-only tokens are
  dropped from phrases (`Harbor Sign Co 4` → `Harbor Sign Co`,
  `weekly lawn service #20` → `weekly lawn service`) so next month's invoice
  number still matches; commas inside a Name become spaces so the OR split
  cannot cut a vendor in two.
* `match_type="contains"`, `priority=10`, `fields` scoped to the Name + Memo
  columns that exist (Notes can never participate). No Name/Memo column →
  falls back to the first non-note text column, as before.
* Re-ingest merges into the existing rule for that account + client (same
  id, phrase union). Prefers an already-ingested rule, else the oldest
  non-regex rule for that code; regex rules are never widened.
* Everything is persisted; only the UI caps what it displays.
* **Name-first matching** (`_first_matching_rule`, used by `match_row`,
  `apply_rules_to_dataframe` and `run_selected_rules_audited`): among rules
  scoped to Name+Memo, a Name hit beats an earlier Memo-only hit. Row-wide
  manual rules keep plain priority order. A noisy Memo can no longer steal a
  row whose vendor Name belongs to another account.
* Banner copy rewritten ("one rule per account, every vendor name already
  on it"); button is **Ingest as rules**; the flash reports rules
  created/updated, phrases added, and collisions found.

## Edit B — Collisions escalate, marked, never auto-fill

* Collision key = normalised Name, else normalised Memo (numeric-only tokens
  dropped). `detect_collisions()` maps key → set(codes); `len > 1` collides.
  Only accountant-owned codes vote (seeds / manual / rule fills, never an
  engine guess).
* Detected at ingest (book rows + the client's existing rule phrases) and at
  match time (stored keys ∪ in-book coded rows, so a book that was never
  ingested still escalates). Stored in a new client-scoped
  `collision_keys` table (`upsert/get/list/delete/count/clear_collisions`).
* Gate runs in every path: Strict (`apply_rules_to_dataframe`,
  `run_selected_rules_audited`, `run_rules_only`), Rules+Memory, Hybrid and
  Full (`FillDownEngine.run`, before keyword rules). No new `FillAction`:
  `NEEDS_REVIEW` + `engine_used="collision"` + rationale
  `Collision: this vendor/memo also codes to 6326 and 6760.01 …`.
  `FillResult.status == "collision"`, counted in `pure_rule_audit`.
* **No exceptions.** Nothing fills a collision-keyed blank row — not a rule,
  not learned memory, not similarity, not ML, not at 0.99. An earlier draft
  of this branch let an exact learned-memory match fill one in the memory
  modes; that carve-out was removed, because the brief says collisions never
  auto-fill in Strict, fuzzy, regex or Full Intelligent. When she *has*
  decided that exact row before, the code is appended to the why-text
  ("You coded this exact transaction '6760.01' before.") as context for her
  Fix, but the cell still stays blank.
* Review card: `:red-background[Collision]` badge + "This vendor already
  maps to more than one account: 6326 and 6760.01." above the unchanged
  Keep / Fix / Not this. Keep is disabled (no suggestion). Fix learns via
  `record_human_approval`. Not this blocks **every** competing code for the
  row's signature and never learns. Collision rows never join a one-click
  pile.
* The shipped seed-disagreement test (`Acme Landscaping` coded 6300 and
  6310 in the same book) is by definition also a collision; the rationale
  still says "disagree" and names both codes, so that lock holds.

## Edit C — Auto-approve above a user-set threshold (default 85%)

* Threshold lives in `st.session_state["auto_approve_threshold"]`, seeded
  from `config.confidence.auto_apply_cutoff` (0.85, unchanged in
  `config.yaml`). The Review expander and the sidebar slider share it and
  are kept in sync from the `on_change` callback;
  `sync_auto_approve_threshold()` pushes it into the config before each run,
  so the engine's existing `AUTO_FILLED` write path honours it.
* After Full and Rules + Similarity runs, `auto_approve_learn()` records each
  `AUTO_FILLED` row at or above the threshold through
  `record_human_approval` (same path as Keep; `approved_by="auto"`,
  `engine_used="auto:<engine>"`). Seeds, manual codes, learned replays and
  collisions are skipped; seed-disagreement rows are `FILLED_REVIEW` /
  `NEEDS_REVIEW` so they never qualify. Existing New Account values are
  never overwritten.
* Flash: "Auto-approved N row(s) at or above X%. Review still has M."
* Strict is unchanged: it fills only from rules that are already shared
  memory in SQLite, so re-learning them adds nothing and its determinism
  lock stays intact.

## Edit D — Excel two-tab bug

* `data_loader.sheet_headers()` / `score_transaction_headers()` /
  `recommend_transaction_sheet()`: header row only, scored by
  `ingest_profiles.detect_profile` (3) or Name/Payee/Memo + Amount/Date (2)
  or just a date/amount shape (1). A single top score wins; a tie between
  data-like sheets or no candidate → `None`.
* `ui/landing.py` keeps the one selectbox. Multi-sheet files preselect the
  recommended sheet with the caption "This file has N tabs. We picked the
  transaction sheet — change it if that is wrong." Ambiguous → placeholder,
  Load disabled until a sheet is chosen. Single sheet / CSV unchanged. Works
  locally and in `FILLDOWN_LIVE=1` (the same disabled-button gate as the
  client-name rule).
* Exporter untouched (the "Code Down Summary" output tab already carries a
  descriptive name; the bug was inbound).

## Edit E — Recall miss 36 vs 44 (synthetic only)

* `tests/fixtures/chrysalis_recall_like.csv` — new 44-row fictional coded
  book. `tests/fixtures/chrysalis_like.csv` was **not** extended: two
  existing tests lock `protected_existing == 187` on it and the brief says
  not to weaken passing assertions.
* `tests/test_recall_fixtures.py` builds 44 blank twins including the
  previously-missed shapes (memo-only "Lease Fee" / "leasefee" / "LeaseFee" /
  "August lease fee — unit 12", `6760.01`, "Smith, Jones & Co", numeric
  suffixes, invoice numbers, "CloudNine Hosting", a Name with a Memo that
  belongs to another account) and asserts **44 of 44** after ingest +
  Strict, again under fuzzy, and the lease-fee twins under a regex rule.
* `tests/fixtures/swann_like.csv` — `6760.01` and `6326` distinct, a Notes
  column full of account-like numbers (ignored), one vendor coded both ways
  (escalates as a collision).
* `normalize_code("6760.01") == "6760.01"` is locked in
  `tests/test_ingest_per_account.py`. `utils/account_codes.py` unchanged.

## Edit F — Two users, one memory

No split. `docs/RAILWAY.md` gained the sheet-picker, ingest, collision and
threshold steps in the smoke, `collision_keys` in the volume table, and a
short "Two accountants, one memory" section. One URL, one password, same
client-name spelling, one `/data` volume — unchanged.

## Not done, on purpose

* No `FillAction.COLLISION` enum, no schema changes in `models/schemas.py`.
* No exporter change.
* `chrysalis_like.csv` not extended (see Edit E).
* Fix on a collision does not delete the stored collision key, so the vendor
  keeps escalating until she gives it a single mapping. Deleting keys
  automatically would let one Fix silence a real two-account vendor.
* Strict runs do not call `auto_approve_learn` (see Edit C).

## Tests

Counts below were re-run from a clean checkout of this branch on 2026-10-08
(`pip install -r requirements.txt pytest`, Python 3.12.3):

* `demo/railway-live` (`55b593c`, the PR target): **253 passed**.
* `demo/railway-live-opus-reliability` (`e548093`, PR #7, this branch's base):
  **271 passed**.
* This branch: **305 passed**, `python smoke_test.py` green (51 checks).
* New files: `tests/test_ingest_per_account.py`, `tests/test_collisions.py`,
  `tests/test_excel_sheets.py`, `tests/test_recall_fixtures.py`; two
  AppTest cases added to `tests/test_review_inbox_e2e.py` (collision badge +
  Fix; threshold slider sync).
* Two existing assertions were updated to the new spec, not weakened:
  `test_ingest_prefers_name_then_memo` (both coded rows share `6322`, so
  one rule now holds all three phrases — Name first — instead of two rules;
  the Notes lock is kept and extended to `rule.fields`), and
  `test_ingest_prefilled_creates_exact_high_priority_rules` (ingested rules
  are `contains`, priority 10 still asserted). Every other pre-existing
  assertion is untouched.
