"""The Review workspace — where leftovers get decided fast.

A first-class, full-width view that shows ONLY the rows that need a human
(``filled_review`` / ``needs_review``, optionally ``no_match``), sorted
least-confident first and grouped by similarity cluster. Built for speed:

* one click approves a whole similarity group at its consensus code,
* bulk actions approve everything visible / above a confidence cutoff,
* the editable table batches per-row decisions into a single Apply click
  (Streamlit has no reliable per-key handling, so every control here mutates
  many rows per rerun instead of one),
* every approval flows through the single learning path (memory + training
  data + account knowledge), and every reject *blocks* the bad pairing.

All mutations go through ``src/spreadsheet_helpers`` and honour the hard
invariant: existing codes (seeds / manual edits) are never overwritten.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import spreadsheet_helpers as sh
from ui import common
from ui import rule_creation as rc
from ui.common import services


def render_review() -> None:
    config, storage, rules, mm, logger = services()
    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    if work is None or loaded is None:
        st.info("No file loaded yet.")
        if st.button("Go to upload", type="primary"):
            st.session_state["view"] = "landing"
            st.rerun()
        return

    client_id = st.session_state.get("client_name") or None
    counts = sh.summary_counts(work, loaded)

    head = st.columns([3.4, 2])
    client = st.session_state.get("client_name") or ""
    head[0].markdown(f"### Review" + (f" — {client}" if client else ""))
    head[0].caption(
        f"{loaded.source_name} · {counts['review_pending']:,} row(s) need your "
        f"eyes · {counts['blank']:,} still blank")
    if head[1].button("← Back to spreadsheet", width="stretch"):
        st.session_state["view"] = "spreadsheet"
        st.rerun()

    include_no_match = st.toggle(
        "Also show rows with no suggestion at all",
        value=bool(st.session_state.get("review_include_no_match", False)),
        key="review_include_no_match",
        help="No-match rows have no suggested code — include them to code "
             "everything from one screen.")

    table = sh.review_rows_df(work, loaded, include_no_match=include_no_match)

    if table.empty:
        st.success(
            f"Nothing left to review. {counts['filled']:,} of "
            f"{counts['total']:,} rows are coded"
            + (f" ({counts['blank']:,} still blank — code one example or add a "
               "rule and run again)" if counts["blank"] else "")
            + ". Export when you're ready.")
        c1, c2, _ = st.columns([1.4, 1.4, 3])
        if c1.button("Open spreadsheet", width="stretch", key="rev_empty_grid"):
            st.session_state["view"] = "spreadsheet"
            st.rerun()
        if c2.button("Export", width="stretch", key="rev_empty_export"):
            st.session_state["view"] = "spreadsheet"
            st.session_state["export_open"] = True
            st.session_state["panel"] = None
            st.rerun()
        return

    _render_group_cards(work, loaded, storage, config, client_id)
    _render_bulk_bar(work, loaded, storage, config, client_id, table)
    _render_review_table(work, loaded, storage, config, client_id, table)

    # The rule-creation dialog can be opened from "Promote to rule".
    if st.session_state.get("rule_panel_open"):
        rc.rule_creation_dialog(work, loaded, config, rules, storage)


# --------------------------------------------------------------------------- #
# Similarity groups — first-class one-click objects
# --------------------------------------------------------------------------- #
def _render_group_cards(work, loaded, storage, config, client_id) -> None:
    groups = sh.group_review_summary(
        work, loaded, high_cutoff=float(config.confidence.auto_apply_cutoff))
    if not groups:
        return

    st.subheader(f"Similarity groups ({len(groups)})")
    st.caption("Look-alike transactions cluster together. When the whole group "
               "points at one code, approve it in one click. Split groups are "
               "never one-click approved — pick the code row by row below.")
    for g in groups[:15]:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.6])
            title = (f"**Group #{g['group_id']}** · {g['rows']} row(s) · "
                     f"suggested **{g['suggested'] or '—'}** · "
                     f"{g['high_conf']} high confidence")
            c1.markdown(title)
            if g["sample"]:
                c1.caption(g["sample"])
            if g["split"]:
                split_txt = ", ".join(f"{code} ×{n}"
                                      for code, n in g["codes"].items())
                c2.button(f"Split: {split_txt}", width="stretch",
                          key=f"rev_group_split_{g['group_id']}",
                          disabled=True,
                          help="This group's rows suggest different codes. "
                               "Review them individually below.")
            else:
                if c2.button(
                        f"Approve group → {g['suggested']}",
                        type="primary", width="stretch",
                        key=f"rev_group_approve_{g['group_id']}",
                        help="Fill every blank row in this group with the "
                             "suggested code and learn from them. Existing "
                             "codes are never overwritten."):
                    common.push_undo()
                    out = sh.approve_similarity_group(
                        work, loaded, storage, config, g["group_id"],
                        client_id=client_id)
                    n = int(out["approved"])
                    st.toast(f"Approved {n} · learned {n} · queued for next "
                             "model train")
                    common.set_flash(
                        f"Group #{g['group_id']}: approved {n} row(s) at "
                        f"{out['code']} — learned for next time.")
                    _bump()
                    common.persist_workspace()
                    st.rerun()


# --------------------------------------------------------------------------- #
# Bulk action bar
# --------------------------------------------------------------------------- #
def _render_bulk_bar(work, loaded, storage, config, client_id, table) -> None:
    visible = [int(r) for r in table["row"].tolist()]
    selected = [i for i in sh.selected_indices(work) if i in set(visible)]

    with st.container(border=True):
        st.markdown("**Bulk actions** — every action is one explicit click; "
                    "existing codes are never overwritten.")
        b1, b2, b3, b4 = st.columns([1.6, 2.2, 1.8, 1.6])

        if b1.button(f"Approve all visible ({len(visible):,})",
                     width="stretch", key="rev_approve_visible",
                     help="Approve every row currently in the review table at "
                          "its current/suggested code."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=visible, client_id=client_id)
            st.toast(f"Approved {out['applied']} · learned {out['learned']} · "
                     "queued for next model train")
            _bump()
            common.persist_workspace()
            st.rerun()

        cutoff = b2.slider("Confidence ≥", 0.0, 1.0, 0.85, 0.05,
                           key="rev_conf_cutoff",
                           help="Approve only rows at or above this "
                                "confidence.")
        confident = [int(r) for r in table.loc[
            table["Confidence"].astype(float) >= cutoff, "row"].tolist()]
        if b2.button(f"Approve {len(confident):,} row(s) ≥ {cutoff:.2f}",
                     width="stretch", key="rev_approve_conf",
                     disabled=not confident):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=confident, client_id=client_id)
            st.toast(f"Approved {out['applied']} · learned {out['learned']} · "
                     "queued for next model train")
            _bump()
            common.persist_workspace()
            st.rerun()

        if b3.button(f"Apply suggested to selected ({len(selected):,})",
                     width="stretch", key="rev_apply_suggested",
                     disabled=not selected,
                     help="For each row selected in the spreadsheet grid, "
                          "write its suggested code (blank rows only)."):
            common.push_undo()
            out = sh.apply_suggested_to_rows(work, loaded, storage, selected,
                                             client_id=client_id)
            st.toast(f"Approved {out['applied']} · learned {out['learned']} · "
                     "queued for next model train")
            _bump()
            common.persist_workspace()
            st.rerun()

        if b4.button(f"Reject selected ({len(selected):,})",
                     width="stretch", key="rev_reject",
                     disabled=not selected,
                     help="Leave the selected rows blank and do NOT learn the "
                          "suggestion — the same pairing won't be proposed "
                          "again."):
            common.push_undo()
            out = sh.reject_rows(work, loaded, storage, selected,
                                 client_id=client_id)
            common.set_flash(
                f"Rejected {out['rejected']} row(s) — left blank, not learned.")
            _bump()
            common.persist_workspace()
            st.rerun()

        # Recode selected to a typed account.
        r1, r2, r3 = st.columns([2, 1.4, 3])
        recode_val = r1.text_input(
            "Recode selected to account", key="rev_recode_code",
            placeholder="e.g. 6322",
            label_visibility="collapsed")
        overwrite = r2.checkbox(
            "Also re-code automation fills", key="rev_recode_overwrite",
            help="Off: only blank selected rows are coded (existing codes are "
                 "sacred). On: rows the automation filled may be re-coded too "
                 "— your own seeds and manual edits are still never touched.")
        if r3.button(f"Recode {len(selected):,} selected row(s)",
                     width="stretch", key="rev_recode_apply",
                     disabled=not (selected and recode_val.strip())):
            common.push_undo()
            out = sh.recode_rows(work, loaded, storage, selected, recode_val,
                                 client_id=client_id,
                                 overwrite_engine=overwrite)
            msg = (f"Re-coded {out['recoded']} row(s) to "
                   f"{recode_val.strip()} · learned {out['learned']}.")
            if out["skipped_protected"]:
                msg += (f" Skipped {out['skipped_protected']} already-coded "
                        "row(s) (protected).")
            st.toast(f"Re-coded {out['recoded']} · learned {out['learned']}")
            common.set_flash(msg)
            _bump()
            common.persist_workspace()
            st.rerun()


# --------------------------------------------------------------------------- #
# The editable review table (batched decisions, one Apply click)
# --------------------------------------------------------------------------- #
def _render_review_table(work, loaded, storage, config, client_id,
                         table) -> None:
    st.subheader(f"Rows to review ({len(table):,})")
    st.caption(
        "Least confident first. Edit the **New Account** cell to correct a "
        "suggestion, tick **Approve**, then click **Apply approved** once — "
        "no per-row waiting. Approving writes the code, protects it, and "
        "teaches the tool (memory + training example).")

    display = table.drop(columns=["row"])
    edited = st.data_editor(
        display, width="stretch", hide_index=True, key="review_table_editor",
        height=min(600, 80 + 35 * len(display)),
        column_config={
            "Approve": st.column_config.CheckboxColumn(
                "Approve", width="small"),
            "New Account": st.column_config.TextColumn(
                "New Account", help="The code to apply when approved."),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "Why": st.column_config.TextColumn("Why", width="large"),
        },
        disabled=[c for c in display.columns
                  if c not in ("Approve", "New Account")],
    )

    a1, a2, a3 = st.columns([1.8, 1.8, 3])
    n_approved = int(edited["Approve"].astype(bool).sum()) \
        if "Approve" in edited.columns else 0
    if a1.button(f"Apply approved ({n_approved:,})", type="primary",
                 width="stretch", key="rev_apply_table",
                 disabled=n_approved == 0,
                 help="Write the approved codes, protect them, and learn from "
                      "them. Nothing is applied for rows you didn't tick."):
        common.push_undo()
        merged = edited.copy()
        merged["row"] = table["row"].to_numpy()
        out = sh.apply_review_table(work, loaded, storage, merged,
                                    client_id=client_id)
        st.toast(f"Approved {out['applied']} · learned {out['learned']} · "
                 "queued for next model train")
        common.set_flash(
            f"Review applied: {out['applied']} code(s) written, "
            f"{out['learned']} learned"
            + (f", {out['cleared']} cleared" if out["cleared"] else "") + ".")
        _bump()
        common.persist_workspace()
        st.rerun()

    # Promote an approved row to a reusable rule.
    options = {
        int(r): f"row {int(r) + 1} — {lbl}"
        for r, lbl in zip(table["row"], _row_labels(table))}
    pick = a2.selectbox("Promote a row to a rule", [None] + list(options),
                        key="rev_promote_pick", format_func=lambda v:
                        "Promote to rule…" if v is None else options[v],
                        label_visibility="collapsed")
    if a3.button("Promote to rule", width="stretch", key="rev_promote_btn",
                 disabled=pick is None,
                 help="Turn this row's code into a reusable keyword rule, "
                      "pre-filled with the best short phrase from Name, then "
                      "Memo."):
        row_idx = int(pick)
        code = str(work.at[row_idx, loaded.new_account_col] or "").strip() \
            or str(work.at[row_idx, sh.SUGGESTED_COL] or "").strip()
        prefill = sh.rule_creation_prefill(work, [row_idx], loaded,
                                           rules_manager=rules, config=config)
        if code:
            prefill["code"] = code
        rc.open_rule_panel(prefill)
        st.rerun()


def _row_labels(table) -> list:
    """Short human labels for rows in the review table."""
    text_cols = [c for c in ("Name", "Payee", "Memo", "Description")
                 if c in table.columns]
    labels = []
    for _, row in table.iterrows():
        parts = [str(row[c]).strip() for c in text_cols
                 if str(row[c]).strip()]
        sug = str(row.get("Suggested", "")).strip()
        labels.append(" | ".join(parts)[:60] + (f" → {sug}" if sug else ""))
    return labels


def _bump() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
