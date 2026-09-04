"""User Guide — a fully self-contained, additive help page.

This module is intentionally isolated: it imports only Streamlit, renders into
the main area, and never touches the app's data, services, session logic or
other views. It is shown as a full-page takeover (like the installer screen) and
returns to the app via the "Back" button. Nothing else in the app is affected.

Works in both light and dark themes (uses only theme-aware Streamlit widgets).
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
    """Render the complete, non-technical user guide (full width)."""
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("📖 User Guide — Code_Down_ML")
        st.caption("Transaction Classification · v1.0")
    with top_r:
        st.write("")
        st.button("← Back to app", width="stretch", key="guide_back_top",
                  on_click=close_guide)

    st.divider()

    tabs = st.tabs([
        "Welcome",
        "Quick Start",
        "How it Works",
        "Review & Learning",
        "Exports",
        "Tips & Troubleshooting",
        "Why It Matters",
    ])

    with tabs[0]:
        _welcome()
    with tabs[1]:
        _quick_start()
    with tabs[2]:
        _how_it_works()
    with tabs[3]:
        _review_and_learning()
    with tabs[4]:
        _exports()
    with tabs[5]:
        _tips_and_troubleshooting()
    with tabs[6]:
        _business_value()

    st.divider()
    _legal()

    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            "**Questions or ideas?** Open an issue on the GitHub repository: "
            "https://github.com/DannyBarren/Code_Down_ML")
    with c2:
        st.button("← Back to app", width="stretch", key="guide_back_bottom",
                  on_click=close_guide)


def _legal() -> None:
    st.subheader("License ⚖️")
    st.markdown(
        """
**Copyright © 2026 Danny Barren.** Released under the **MIT License** — see the
`LICENSE` file in the repository.

- You may use, copy, modify and distribute this software.
- The software is provided **"as-is"**, without warranty of any kind.
- The bundled sample data is fictional and provided **"as-is"** for evaluation.
        """)
    st.caption("Code_Down_ML • v1.0 • MIT License")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _welcome() -> None:
    st.subheader("Welcome 👋")
    st.markdown(
        """
This tool takes one of the most tedious data tasks in any business —
classifying every transaction to the right account or category — and does it for
you in seconds, with you staying firmly in control.

You seed a few transactions with the correct **account / category code**, and the
tool propagates those codes to every similar transaction, scores its confidence,
explains its reasoning, and flags anything uncertain for a quick human review.

**What it does for you**

- ⏱️ **Saves hours** every cycle — no more manual search‑copy‑paste in Excel.
- 🎯 **Classifies accurately** — it proposes a code *and* a confidence score for every row.
- 🧠 **Learns from you** — every code you approve makes the next file faster and smarter.
- 🔁 **Repeatable & auditable** — consistent results you can stand behind.

**The big idea:** the more data you run through it, the smarter it gets — and the
richer the insights you can surface about your operation.
        """)
    st.info(
        "You're always in charge. The tool *suggests*; you *approve*. Nothing is "
        "finalized until you say so.")


def _quick_start() -> None:
    st.subheader("Quick Start (about 5 minutes) ⚡")
    st.markdown(
        """
1. **Name the client / project** *(optional)* — on the dashboard, type a name.
   This labels your exports and run history.
2. **Upload the file** — drag in any transaction export (`.xlsx` or `.csv`). No
   file handy? Click **Load sample dataset** to try it instantly.
3. **Run the automation** — open the spreadsheet and choose:
   - **Run Selected Rules** for a safe, predictable pass, **or**
   - **Full Intelligent Run** to let the tool code everything it confidently can.
4. **Review & approve** — scan the results. High-confidence rows are ready;
   anything uncertain is flagged for a quick look. Approve what's correct.
5. **Download** — click **Export** for a clean Excel (formatting preserved, plus a
   summary tab) or a ready-to-import CSV.

That's it — a file that used to take an hour is done in minutes.
        """)
    st.caption("📸 Screenshot placeholder: the dashboard with the project name "
               "field and the upload box.")


def _how_it_works() -> None:
    st.subheader("How the matching works (in plain English) 🛠️")

    st.markdown("#### Rule-driven runs — safe and predictable")
    st.markdown(
        """
A **rule** is a simple shortcut you control: *"whenever you see this word, use
this code."* For example, **`Cloud Hosting → 6100`**.

- Rules run **first** and are **deterministic** — same input, same result.
- A rule only ever touches rows it actually matches; it **never overwrites** a
  value you typed or a code you already approved.
- Rules are saved and reused on **every future file** — so you teach the tool a
  recurring vendor once, and benefit forever.
        """)
    st.success("Think of rules as your safety net: precise, repeatable, and fully "
               "under your control.")

    st.markdown("#### Intelligent Mode — smarter matching when you want it")
    st.markdown(
        """
When you choose **Full Intelligent Run**, the tool works through a sensible
hierarchy, strongest evidence first:

1. **Your rules** (exact shortcuts you defined).
2. **Learned memory** (codes you've approved before for look‑alike transactions).
3. **Similarity** (groups transactions that read alike and applies your seeds).
4. **Trained model** (an optional AI layer that improves as it learns).

Every result comes with a **confidence score** and a short reason, so you can see
*why* a code was chosen. Low-confidence rows are politely set aside for review
rather than guessed.
        """)
    st.caption("📸 Screenshot placeholder: the spreadsheet with filled codes, "
               "confidence chips, and the toolbar.")


def _review_and_learning() -> None:
    st.subheader("Review Queue & how the tool gets smarter 📈")
    st.markdown(
        """
**The Review Queue** gathers the rows the tool wasn't fully sure about so you can
make the call quickly:

- Confirm a suggested code with a click, or
- Correct it — type the right code, and the tool remembers.

**Every approval teaches the tool.** Approved codes become **learned memory** and
**training examples**, so the next file is faster and more accurate.

#### Models & learning progress
- The app quietly trains a model in the background as approvals accumulate.
- You can watch progress on the **Models** screen (examples learned, accuracy,
  which engine is active).
- If the model ever underperforms, it **automatically falls back** to the proven
  built-in matcher — you never get worse results by enabling it.

#### Rule Notes — capture the "why"
Add a short **Rule Note** on a tricky transaction (e.g. *"quarterly maintenance
contract"*). These hints improve matching **and** are remembered next time a
similar file comes through.
        """)
    st.caption("📸 Screenshot placeholder: the Review Queue and the Models "
               "progress screen.")


def _exports() -> None:
    st.subheader("Exports — hand off clean, finished work 📤")
    st.markdown(
        """
When you're happy with the classification, click **Export**:

- **Excel (`.xlsx`)** — keeps the original formatting and adds a tidy
  **summary tab** (totals coded, auto-filled, reviewed). Great for records and
  for client-facing review.
- **CSV** — a clean, ready-to-import file for your accounting or ERP system.

Internal helper columns are stripped automatically, so what you download is
polished and ready to use.
        """)
    st.info("Tip: the Excel summary tab is a quick, professional way to show "
            "exactly what was done.")


def _tips_and_troubleshooting() -> None:
    st.subheader("Tips for best results 💡")
    st.markdown(
        """
- **Seed a few codes first.** Code a handful of clear transactions before running
  full automation — it gives the tool strong examples to learn from.
- **Build rules for recurring vendors.** One good rule (e.g. `SaaS Tools → 6100`)
  pays off on every future file.
- **Use the search and filters.** Jump straight to "needs review" rows, or search
  a vendor across the whole file.
- **Approve generously but carefully.** Approvals are how the tool learns — the
  more consistent you are, the smarter it gets.
- **Add Rule Notes** on ambiguous transactions to capture context.
        """)

    st.subheader("Troubleshooting common questions 🧩")
    with st.expander("A rule \"applied to 0 rows\" — why?"):
        st.markdown(
            "The keyword may be too specific, or it's looking only in selected "
            "columns. Try a shorter, common part of the vendor name and leave the "
            "column restriction off so it searches the whole row.")
    with st.expander("Some rows were left blank after a run"):
        st.markdown(
            "The tool leaves a row blank rather than guess when it isn't "
            "confident. Code one example, then re-run — similarity will carry it "
            "to the look‑alikes. Or add a rule for that vendor.")
    with st.expander("My codes look slightly different (e.g. a letter suffix)"):
        st.markdown(
            "Codes are automatically normalized to a consistent form (e.g. "
            "`6100` or `6100A`), so they stay consistent across files.")
    with st.expander("Will it overwrite work I've already done?"):
        st.markdown(
            "No. Values you typed or approved are protected. Rules and automation "
            "only fill blanks or refine the tool's own earlier guesses.")
    with st.expander("Did my data leave my computer / the demo?"):
        st.markdown(
            "Processing happens within the app. In the public demo, data is not "
            "permanently saved across sessions unless persistent storage is "
            "enabled (you'll see a banner noting this).")


def _business_value() -> None:
    st.subheader("Why this matters for your business 🚀")
    st.markdown(
        """
This tool isn't just a time-saver — it's a **data flywheel** for your
operation.

- **Consistency compounds.** Each file you run trains better models, so
  classification gets faster and more accurate across everything you process.
- **More data, better insight.** The more transactions flow through the tool, the
  richer the picture you can build — spend patterns, vendor trends, and category
  breakdowns that help you understand and improve the business.
- **A premium, repeatable process.** Clean, auditable, consistently coded data
  delivered quickly positions you as the proactive, insight-driven operator.
- **Lower cost to serve.** Hours saved per file means you can handle more volume
  without adding headcount.

> The strategic takeaway: every file you run makes the next one cheaper to
> produce **and** more valuable. Volume and consistency turn routine data clean-up
> into compounding intelligence.
        """)
    st.success(
        "The more data the tool sees, the better the automation **and** the "
        "insights you can act on.")
