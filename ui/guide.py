"""User Guide — a fully self-contained, additive help page.

This module is intentionally isolated: it imports only Streamlit, renders into
the main area, and never touches the app's data, services, session logic or
other views. It is shown as a full-page takeover (like the installer screen) and
returns to the app via the "Back" button. Nothing else in the app is affected.

One page, strict-first, written for accountants. Works in both light and dark
themes (uses only theme-aware Streamlit widgets).
"""

from __future__ import annotations

import streamlit as st

GUIDE_FLAG = "show_guide"


def open_guide() -> None:
    """Callback: open the guide (safe to use as ``on_click``)."""
    st.session_state[GUIDE_FLAG] = True


def close_guide() -> None:
    """Callback: return to the app."""
    st.session_state[GUIDE_FLAG] = False


def is_open() -> bool:
    return bool(st.session_state.get(GUIDE_FLAG))


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render_guide() -> None:
    """Render the complete guide as a single page."""
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("User Guide — Code_Down_ML")
        st.caption("Code a week of transactions in minutes, and trust every "
                   "fill.")
    with top_r:
        st.write("")
        st.button("← Back to app", width="stretch", key="guide_back_top",
                  on_click=close_guide)

    st.divider()
    _the_loop()
    _run_modes()
    _review()
    _learning()
    _export()
    _rules_that_matter()

    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Code_Down_ML • MIT License • © 2026 Danny Barren")
    with c2:
        st.button("← Back to app", width="stretch", key="guide_back_bottom",
                  on_click=close_guide)


def _the_loop() -> None:
    st.subheader("The weekly loop")
    st.markdown(
        """
1. **Upload** the client's AppFolio / QuickBooks export on the dashboard. Type
   the client name first — rules and memory are kept per client.
2. **Ingest coded examples** if the file has them. The banner offers to turn
   pre-filled rows into exact-match rules in one click.
3. **Run Rules — Strict** (the recommended button). Only your rules run, only
   blank rows are filled, and every fill names the rule that made it.
4. **Review the leftovers** in the Review workspace — grouped, least confident
   first, one click per group.
5. **Export.** Excel writes only the `New Account` column back into your
   original workbook (formatting intact, plus a summary tab); the CSV is clean
   and import-ready.

Next week, same client: the rules and remembered codes are already loaded, so
coverage jumps before you touch anything.
        """)


def _run_modes() -> None:
    st.subheader("The four run buttons, in order of trust")
    st.markdown(
        """
- **Run Rules — Strict ★ (recommended).** Deterministic. Fills a blank row
  only when one of your keyword rules matches, with only that rule's code.
  Already-coded rows are never touched. The audit panel shows exactly which
  rule filled which row — and which rows no rule covered.
- **Rules + Memory.** Strict rules, then exact matches you have approved
  before. Still deterministic — no guessing.
- **Rules + Similarity.** Rules first, then codes spread to look-alike
  transactions with a confidence score. Broader; review the medium/low
  confidence fills.
- **Full Intelligent Run.** Adds the trained AI model on top. It can fill rows
  no rule matched — that's why it's not the default. Anything the engines
  disagree on, or the seeds split on, goes to Review instead of being silently
  filled.

**Reset run** (toolbar → Reset) un-does the last automation pass and keeps
your rules, Rule Notes and hand-typed codes.
        """)


def _review() -> None:
    st.subheader("Review: where the leftovers get decided")
    st.markdown(
        """
The **Review** workspace (sidebar) shows only rows that need you, least
confident first. Each row tells you the code, the engine, the confidence and a
one-line why — *"Matched rule #14 'Cunningham' (contains) → 6322"*.

- **Groups.** Look-alike transactions cluster. A group with one suggested code
  approves in one click. A split group (two codes) is never one-click
  approved — you pick.
- **Bulk.** Approve everything visible, or everything above a confidence
  cutoff. Select rows in the grid to apply suggestions, re-code, or reject
  them in bulk.
- **The table.** Tick **Approve** on as many rows as you like, fix a code by
  typing over it, then click **Apply approved** once. No per-row waiting.
- **Reject** leaves the row blank and blocks that suggestion from coming back.

Every approval is remembered immediately — as an exact-match mapping and as
training data for the model.
        """)


def _learning() -> None:
    st.subheader("How it learns (and how to stay in charge)")
    st.markdown(
        """
- **Learned memory** is exact: approve "Cunningham Communications → 6322"
  once, and the next identical transaction codes itself.
- **Rules always win** over memory. If a newer rule disagrees with an old
  memory, the row's *why* says so.
- **Re-code freely.** Your latest decision replaces the old mapping.
- **The Models panel** shows what the tool remembers for this client, when it
  last trained, and its held-out accuracy. Delete a bad mapping there and it
  stops firing immediately. After every 25 new approvals a banner offers a
  one-second retrain — training is never silent.
- **Similarity leads early.** The model starts voting alongside similarity at
  ~300 approvals and leads after ~1,500. That's deliberate: it earns trust
  before it leads.
        """)


def _export() -> None:
    st.subheader("Export")
    st.markdown(
        """
- **Excel (keeps your formatting)** re-opens your original workbook and writes
  only the `New Account` column back. Formulas, column widths and other sheets
  stay intact. A **Code Down Summary** tab records the run mode, the counts by
  engine and the timestamp.
- **CSV (ready to import)** is tidy: internal helper columns stripped, codes
  normalized, plus a base-account column.

Filenames carry the client and date, so exports self-identify in your
downloads folder.
        """)


def _rules_that_matter() -> None:
    st.subheader("The rules that protect your work")
    st.markdown(
        """
- A code that came in the file, or one you typed, is **never overwritten** —
  not by rules, not by similarity, not by the AI, not by bulk actions, not by
  reset.
- Rule suggestions are mined from **Name, then Memo, then Description** —
  never from Notes or Rule Notes.
- A rule that would fill **zero blank rows** can't be saved without you
  confirming it — the live preview shows blank matches and already-coded
  matches separately before you commit.
- Everything the automation does is auditable per row: code, engine,
  confidence, and why.
        """)
