# Acceptance — full-app reliability pass

Branch `demo/railway-live-opus-reliability`, cut from `demo/railway-live` @ `55b593c`.
Target is `demo/railway-live`. Nothing on this branch goes to `main`.

This file records what was actually run and what was actually observed. It is
not a plan.

---

## 1. Commands run

```bash
python -m pytest -q --tb=short
python smoke_test.py
```

| When | pytest | smoke |
|------|--------|-------|
| Baseline at `55b593c` | **253 passed, 0 failed, 0 skipped** | `ALL SMOKE TESTS PASSED ✅` |
| After the fixes + new tests | **271 passed, 0 failed, 0 skipped** | `ALL SMOKE TESTS PASSED ✅` |

No test was skipped, weakened, or deleted. The 18 new tests are additive:
11 in `tests/test_review_inbox_e2e.py` (new file), 7 in
`tests/test_live_session.py`.

Run locally, not via Actions: CI only triggers on PRs into `main`, and this
branch never targets `main`. Core deps only — no torch, no
sentence-transformers, no SetFit.

---

## 2. The 8-click path

Run against a real live instance, not a mock:

```bash
rm -rf /tmp/filldown-opus-live && mkdir -p /tmp/filldown-opus-live
FILLDOWN_LIVE=1 FILLDOWN_DEMO=0 FILLDOWN_DEMO_RESET=0 \
FILLDOWN_AUTH_PASSWORD=testpass FILLDOWN_DATA_DIR=/tmp/filldown-opus-live \
FILLDOWN_MAX_ROWS=50000 streamlit run main.py
```

**Result: passed, with one honest deviation at step 5.**

| # | Click | Observed |
|---|-------|----------|
| 0 | Password gate | "Hosted client instance" gate appeared; `testpass` unlocked it |
| 1 | Client name | `Northwind Trading Co.` accepted, no error |
| 2 | Load sample | 220 records; chip read `Generic export · Matching: TF-IDF · ML not trained` |
| 3 | Run Rules — Strict ★ | "Rules filled 22 blank row(s) using only your rules' codes. 10 already coded (protected); 188 still blank." |
| 4 | Review CTA | Button read **"Review 188 blank rows →"**; page opened on "0 leftover(s) need a decision · 188 still blank" |
| 5 | **Fix**, not Keep | Card offered no Keep — see the deviation below. Typed `6322`, clicked Fix → "Fixed row 4 to 6322. Remembered for next time." Queue advanced |
| 6 | Not this | "Not this — row 17 left blank, not remembered." Queue advanced |
| 7 | Insights | Coverage 33/220 · 15%, filled-by-automation 22, uncoded spend $244,648.84, spend-by-account and top-vendor tables. No error box |
| 8 | Export | Three downloads offered: "Excel (keeps your formatting)", "CSV (ready to import)", "Excel (all columns + notes)" |

No traceback, no red crash box, no stuck spinner.

### The deviation at step 5, and why it is correct

Strict is rules-only and makes no suggestions, so after a Strict run there is
nothing to **Keep** — the queue is blank rows, and the verb that applies is
**Fix**. This is the documented Strict behaviour ("she can still reach blank
leftovers and Fix them"), not a defect. Keep and Not this appear as soon as a
run produces suggestions (Rules + Memory, Similarity, Full).

**Tell Amanda this before the call**: on a Strict-first pass her verb is Fix.

---

## 3. Live Resume across a restart

**Result: passed.** The process was killed (`kill <pid>`) and restarted with
the same env — a real restart, not a browser reload.

| Check | Observed |
|-------|----------|
| Gate after restart | Password page reappeared. The instance did not stay open |
| Saved-workspace notice | "One saved workspace: Northwind Trading Co. — sample_transactions.xlsx (220 rows). Resume restores that client name." |
| Resume | "Restored 'sample_transactions.xlsx' (220 rows)." → spreadsheet, 33/220 coded · 15% |
| **The book is back** | Row 4 (Acme Supplies / Inventory restock) reads **6322** — the fix made before the restart |
| **Memory is back** | Memory inspector: `6322`, scope `Northwind Trading Co.`, text `acme supplies \| inventory restock \| general supplies purchase \| operating expens…` |
| Insights | Still render, real numbers, no error |
| Rejection held | Row 17, rejected before the restart, was not re-suggested by the later similarity run |
| **Start fresh is session-only** | Dialog promised rules/memory/models are kept. After confirming with the typed client name: file unloaded, session files gone, and the `6322` mapping was **still in the memory inspector** |
| Demo wipe cannot run on live | `/tmp/filldown-opus-live` survived the restart — `maybe_auto_reset` returns early when `is_live()` |

On-disk state before the restart:

```
/tmp/filldown-opus-live/fill_down.db
/tmp/filldown-opus-live/sessions/northwind_trading_co/meta.json
/tmp/filldown-opus-live/sessions/northwind_trading_co/original.bin
/tmp/filldown-opus-live/sessions/northwind_trading_co/work_df.pkl
```

`safe_client_id("Northwind Trading Co.")` → `northwind_trading_co`, which is
the folder Resume read back.

### Inverse: local with no env

`streamlit run main.py` with no FILLDOWN vars: no password page, no hosted
banner, nothing auto-loaded, sample is opt-in, 220 rows load clean.

---

## 4. P0 found and fixed

**Live silently truncated an oversized upload and called it a demo.**

`Dockerfile.railway` bakes `FILLDOWN_MAX_ROWS=50000`. `load_into_session`
truncated any longer file to the cap, flashed *"Demo limit: showing the first
50,000 rows"*, and carried on. On a hosted client instance that is three
failures at once: transactions vanish from the book, Export then writes the
short book back, and the notice calls a paid client instance a demo.

Fixed: live refuses the file and explains why, so nothing is dropped. The
public demo still previews a large file — that is what the demo is for — and a
cap set on a local run no longer describes itself as a demo limit.

Proven in the browser on a deliberately capped instance
(`FILLDOWN_MAX_ROWS=50`, 220-row upload):

> This file has 220 rows and this instance is capped at 50. Nothing was
> loaded, so no transactions were dropped. Split the export into smaller
> periods and use Add another export to build the month up, or ask your admin
> to raise FILLDOWN_MAX_ROWS.

The app stayed on the landing page with nothing loaded. Locked by four tests
in `tests/test_live_session.py`.

---

## 5. P1 found and fixed

**Client scope could split between the run and the learning path.** Runs,
approvals, rule creation, the memory inspector, run history and export
filenames read the landing widget key `client_name` directly, while Review
reads the durable `_live_client_name`. `seed_client_name_widget()` papers over
the difference today, so this was latent rather than live — but any path that
renders before it, or any Streamlit version that drops the key, would write a
learned mapping into the shared scope while Review looked at the client scope,
and the client's memory would appear to vanish. Every client-scoped read now
goes through `current_client_id()`.

---

## 6. Other surfaces checked

Verified by direct probe or in the browser; no defect found, so no code
changed.

- **Ingest** — CSV and XLSX; missing `New Account` column is created; empty
  file and garbage bytes raise `DataLoadError` (browser showed *"Could not
  read that file: The file was read but contains no rows of data."*, no
  traceback); odd columns still load; detection never fails a load; the
  AppFolio fixture's Notes text never reaches `_sim_text`.
- **Run modes** — Strict fills blanks only, writes no similarity suggestion,
  and leaves seeds untouched; Rules + Memory fills exact signatures only; full
  run never overwrites a seed or a manual code; undo restores.
- **Rules** — lower priority number wins; a disabled rule does not fire; rule
  prefill never mines Notes.
- **Memory** — reject clears the row, writes no learned mapping, blocks the
  pairing, and the blocked pairing is not refilled on the next run; client A's
  mapping is invisible to client B; a missing per-client model does not raise.
- **Insights** — coverage runs, never writes `New Account`, survives a book
  with no Amount column.
- **Append** — in the browser, appending a 5-row export (2 duplicates, 3 new)
  onto the 220-row book gave *"Added 3 new row(s) · skipped 2 duplicate(s) ·
  protected 2 coded."* Total went 220 → **223**, not 225. The coded count held
  at 32. The 3 new rows arrived uncoded.
- **Export** — CSV strips `_`-prefixed columns and keeps `New Account`; the
  preserving Excel export writes a **Code Down Summary** tab into a copy of
  the original workbook.
- **Concurrency** — twelve back-to-back approvals against the live SQLite file
  did not raise `database is locked` (WAL + `busy_timeout` are on).
- **Widget landmines** — no `st.session_state[widget_key]` write after
  instantiation; `_do_start_fresh_live` never calls `clear_rules()` or
  `clear_learned_mappings()`; `select_data_dir()` refuses to fall through to
  `/tmp` on live; `Dockerfile.railway` carries no `FILLDOWN_DEMO=1`, no
  `USER 1000`, and no auto-wipe.

---

## 7. Residual risk

1. **Keep cannot be demoed on the bundled sample.** The sample's rows are
   exact text duplicates of its seeds, so TF-IDF cosine is 1.0 and the
   confidence of 100% is honest, not inflated. Everything auto-fills above the
   `auto_apply_cutoff` and the Review queue stays empty — the live run filled
   138 rows at "138 high, 0 medium, 0 low". Keep is proven by
   `tests/test_review_inbox_e2e.py` against a staged queue, but it was never
   clicked in a browser. Real AppFolio text varies and should produce a real
   queue. Tuning thresholds to manufacture a demo queue is a product decision,
   so it was left alone. **If Amanda wants to see Keep on the call, use a real
   client export, not the sample.**
2. **Not tested against a real Railway restart.** A local `kill` + relaunch on
   a `/tmp` data dir stands in for it. Volume permissions, `RAILWAY_RUN_UID=0`
   and the `/data` mount can only be confirmed on the service itself
   (`docs/RAILWAY.md` §§4–8).
3. **No real AppFolio or QuickBooks export was used.** Detection was exercised
   with the sanitized fixture and the sample; the chip on the sample honestly
   reads `Generic export`.
4. **Append does not re-check the row cap.** A book already near
   `FILLDOWN_MAX_ROWS` can be grown past it by repeated appends. Nothing is
   lost or truncated — the cap is an upload guard — but the ceiling is not
   enforced end to end.
5. **The 7 pre-existing `value=` + `key=` widget pairs were left alone.** None
   of their keys is written from session state, so none can raise the
   duplicate-default warning today. Changing them is style, not a fix.

## 8. Out of scope, found but not done

- Routing a share of high-confidence similarity fills into Review so the
  accountant sees a sample of what was auto-applied. That is a new product
  behaviour.
- A row-cap ceiling on append (see residual risk 4).
- `FILLDOWN_MAX_ROWS=50000` on Railway is a guess at a safe ceiling; whether
  it matches Amanda's real monthly volume is a question for her, not a code
  change.

No live URL is recorded here. The hosted address is not in git.
