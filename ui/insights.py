"""Insights — read-only coaching metrics from the coded book.

One page, four cards, two small tables. Everything is computed by pure
functions in ``src/insights.py``; this module only renders. Insights never
write ``New Account`` — they describe the coded book so the coaching
conversation (and the next rules to write) are obvious.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import insights as ins
from ui import common
from ui.common import services


def render_insights() -> None:
    config, storage, rules, mm, logger = services()
    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    if work is None or loaded is None:
        st.info("No file loaded yet.")
        if st.button("Go to upload", type="primary"):
            st.session_state["view"] = "landing"
            st.rerun()
        return

    client = common.current_client_id() or ""
    head = st.columns([3.4, 2])
    head[0].markdown("### Insights" + (f" — {client}" if client else ""))
    head[0].caption(f"{loaded.source_name} · read-only · computed from the "
                    "coded rows in this file")
    if head[1].button("← Back to spreadsheet", width="stretch",
                      key="insights_back"):
        st.session_state["view"] = "spreadsheet"
        st.rerun()

    common.incentive(
        "Coded books are how we coach the client — the more they share, the "
        "sharper this gets.")

    coverage = ins.coding_coverage(work, loaded)
    residual = ins.uncoded_residual(work, loaded, config)
    runs = storage.list_runs(limit=200)
    tput = ins.throughput(runs)
    client_id = common.current_client_id()
    n_memory = len(storage.list_learned_mappings(client_id=client_id))

    # ---- Four cards -------------------------------------------------------
    cards = st.columns(4)
    cards[0].metric(
        "Coding coverage",
        f"{coverage['coded']:,} of {coverage['total']:,}",
        delta=f"{coverage['pct_coded']:.0%} coded", delta_color="off",
        help="Rows with a New Account code out of all rows in this file.")
    cards[1].metric(
        "Filled by automation", f"{coverage['auto_filled']:,}",
        delta=f"{coverage['human']:,} approved/seeded by you",
        delta_color="off",
        help="Codes the engines filled vs codes that came from you "
             "(uploaded or typed).")
    if residual["blank_spend"] is not None:
        cards[2].metric(
            "Uncoded spend", f"${residual['blank_spend']:,.2f}",
            delta=f"{residual['blank_rows']:,} blank rows", delta_color="off",
            help="Spend still without a code — the residual to clear.")
    else:
        cards[2].metric(
            "Still uncoded", f"{residual['blank_rows']:,} rows",
            help="Rows still without a code (no Amount column in this file).")
    cards[3].metric(
        "Rows coded across runs", f"{tput['rows_coded_all_runs']:,}",
        delta=f"{n_memory:,} codes remembered", delta_color="off",
        help="Total rows coded across all recorded runs for this book, and "
             "the exact-match codes remembered for next time.")

    # ---- Two tables -------------------------------------------------------
    mix = ins.expense_mix(work, loaded, config)
    vendors = ins.vendor_concentration(work, loaded, config)

    left, right = st.columns(2)
    with left:
        st.markdown("**Spend by account**")
        if mix is None:
            st.caption("No Amount column in this file — the mix appears once "
                       "amounts are present.")
        elif mix.empty:
            st.caption("No coded rows with amounts yet.")
        else:
            st.dataframe(mix, width="stretch", hide_index=True,
                         height=min(360, 60 + 35 * len(mix)))
    with right:
        st.markdown("**Top vendors** — and what's still uncoded")
        if vendors is None:
            st.caption("No Name/Payee column in this file.")
        else:
            _render_vendor_table(vendors, residual)

    if residual["top_vendors"] is not None and not residual["top_vendors"].empty:
        st.caption("The uncoded vendors above are the next rules worth "
                   "writing — code one row for each and Strict will carry it.")

    _render_insights_export(coverage, mix, vendors, residual, tput, loaded)


def _render_vendor_table(vendors: dict, residual: dict) -> None:
    """Vendor concentration merged with the uncoded residual per vendor."""
    by_count = vendors["by_count"].copy()
    by_spend = vendors.get("by_spend")
    table = by_count
    if by_spend is not None:
        table = table.merge(by_spend, on="Vendor", how="outer")
    uncoded = residual.get("top_vendors")
    if uncoded is not None and not uncoded.empty \
            and "Uncoded spend" in uncoded.columns:
        table = table.merge(
            uncoded.rename(columns={"Uncoded spend": "Uncoded"}),
            on="Vendor", how="outer")
    for col in ("Rows", "Spend", "Uncoded"):
        if col in table.columns:
            table[col] = table[col].fillna(0)
    if "Spend" in table.columns:
        table = table.sort_values("Spend", ascending=False)
    st.dataframe(table.head(10), width="stretch", hide_index=True,
                 height=min(360, 60 + 35 * min(len(table), 10)))


def _render_insights_export(coverage, mix, vendors, residual, tput,
                            loaded) -> None:
    """Optional tiny CSV of the insight numbers — never blocks the page."""
    rows = [
        ("File", loaded.source_name),
        ("Total rows", coverage["total"]),
        ("Coded rows", coverage["coded"]),
        ("Coverage %", f"{coverage['pct_coded']:.1%}"),
        ("Filled by automation", coverage["auto_filled"]),
        ("Approved/seeded by you", coverage["human"]),
        ("Blank rows", coverage["blank"]),
        ("Uncoded spend", residual["blank_spend"]
         if residual["blank_spend"] is not None else "—"),
        ("Rows coded across runs", tput["rows_coded_all_runs"]),
    ]
    summary = pd.DataFrame(rows, columns=["Metric", "Value"])
    parts = [summary]
    if mix is not None and not mix.empty:
        parts = [summary, pd.DataFrame([("", "")]), mix]
    out = pd.concat(parts, ignore_index=True)
    import io
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    st.download_button("Download insights (CSV)", data=buf.getvalue(),
                       file_name="insights.csv", mime="text/csv",
                       key="insights_csv")
