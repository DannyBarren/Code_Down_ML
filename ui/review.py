"""The Review inbox — leftovers get a yes / fix / no decision.

A first-class, full-width view that shows ONLY the rows that need a human
(``filled_review`` / ``needs_review``, optionally ``no_match``), sorted
least-confident first and grouped by similarity cluster.

Built so an accountant never has to go back to the spreadsheet to tick rows:

* the next-up card is the least-confident leftover — one click Approves,
  Fixes, or marks Not this (reject + block, never learns),
* look-alike piles still one-click approve; split groups refuse; Not this
  rejects the whole pile,
* the table batches per-row ticks into a single Save click (Streamlit has no
  reliable per-key handler),
* every approval flows through the single learning path (memory + training
  + account knowledge), and every reject *blocks* the bad pairing.

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
    head[0].markdown("### Review" + (f" — {client}" if client else ""))
    head[0].caption(
        f"{loaded.source_name} · {counts['review_pending']:,} need a decision"
        f" · {counts['blank']:,} still blank")
    if head[1].button("← Back to spreadsheet", width="stretch"):
        st.session_state["view"] = "spreadsheet"
        st.rerun()

    include_no_match = bool(st.session_state.get("review_include_no_match", False))
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

    st.caption(
        "You only see what the rules would not stake their name on. "
        "**Looks right** codes it and remembers. **Not this** leaves it blank "
        "and will not suggest that pairing again. Existing codes are never "
        "overwritten.")

    _render_next_up(work, loaded, storage, config, client_id, table)
    _render_group_cards(work, loaded, storage, config, client_id)
    _render_bulk_bar(work, loaded, storage, config, client_id, table)
    _render_review_table(work, loaded, storage, config, client_id, table)

    if st.session_state.get("rule_panel_open"):
        rc.rule_creation_dialog(work, loaded, config, rules, storage)


# --------------------------------------------------------------------------- #
# Next up — one leftover, three verbs
# --------------------------------------------------------------------------- #
def _render_next_up(work, loaded, storage, config, client_id, table) -> None:
    top = table.iloc[0]
    idx = int(top["row"])
    suggested = str(top.get("Suggested", "") or "").strip()
    current = str(top.get("New Account", "") or "").strip()
    code = current or suggested
    why = str(top.get("Why", "") or "").strip() or "No confident match yet."
    engine = str(top.get("Engine", "") or "").strip()
    conf = float(top.get("Confidence", 0) or 0)
    label = _row_label_from_series(top)

    with st.container(border=True):
        st.markdown("**Next up** — least sure first")
        st.markdown(label or f"Row {idx + 1}")
        meta = (
            f"Suggested **{code or '—'}**"
            + (f" · {engine}" if engine else "")
            + f" · {conf:.0%} sure"
        )
        st.caption(meta)
        st.caption(why)

        a1, a2, a3 = st.columns([1.6, 1.4, 2.4])
        approve_label = f"Looks right → {code}" if code else "Looks right"
        if a1.button(approve_label, type="primary", width="stretch",
                     key="rev_next_approve", disabled=not code,
                     help="Write this code, protect it, and remember it "
                          "for this client."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=[idx], client_id=client_id)
            st.toast(f"Approved {out['applied']} · remembered {out['learned']}")
            common.set_flash(
                f"Approved row {idx + 1} as {code} — remembered for next time.")
            _bump()
            st.rerun()

        if a2.button("Not this", width="stretch", key="rev_next_reject",
                     help="Leave blank and do not learn this pairing. "
                          "The same suggestion will not come back."):
            common.push_undo()
            out = sh.reject_rows(work, loaded, storage, [idx],
                                 client_id=client_id)
            st.toast(f"Left blank · blocked {out['blocked']} pairing(s)")
            common.set_flash(
                f"Rejected row {idx + 1} — left blank, not remembered.")
            _bump()
            st.rerun()

        fix = a3.text_input(
            "Fix to account", key="rev_next_fix",
            placeholder="e.g. 6322",
            label_visibility="collapsed")
        f1, f2 = st.columns([1.4, 3])
        if f1.button("Apply fix", width="stretch", key="rev_next_fix_btn",
                     disabled=not str(fix).strip(),
                     help="Code this leftover to the account you typed. "
                          "Already-coded seeds are never overwritten."):
            common.push_undo()
            out = sh.recode_rows(work, loaded, storage, [idx], fix,
                                 client_id=client_id)
            msg = (f"Re-coded {out['recoded']} to {str(fix).strip()} · "
                   f"remembered {out['learned']}.")
            if out["skipped_protected"]:
                msg += f" Skipped {out['skipped_protected']} protected."
            st.toast(msg)
            common.set_flash(msg)
            _bump()
            st.rerun()
        if f2.button("Always code this vendor this way",
                     width="stretch", key="rev_next_promote",
                     disabled=not (code or str(fix).strip()),
                     help="Turn this vendor into a reusable keyword rule "
                          "from Name, then Memo — never Notes."):
            chosen = str(fix).strip() or code
            prefill = sh.rule_creation_prefill(
                work, [idx], loaded,
                rules_manager=services()[2], config=config)
            if chosen:
                prefill["code"] = chosen
            rc.open_rule_panel(prefill)
            st.rerun()


# --------------------------------------------------------------------------- #
# Similarity groups — first-class one-click objects
# --------------------------------------------------------------------------- #
def _render_group_cards(work, loaded, storage, config, client_id) -> None:
    groups = sh.group_review_summary(
        work, loaded, high_cutoff=float(config.confidence.auto_apply_cutoff))
    if not groups:
        return

    st.subheader(f"Same-looking piles ({len(groups)})")
    st.caption("Look-alike transactions cluster together. When the whole pile "
               "points at one code, approve it in one click. Split piles are "
               "never one-click approved — pick the code on Next up or below.")
    for g in groups[:15]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3.6, 1.6, 1.4])
            title = (f"**Pile #{g['group_id']}** · {g['rows']} row(s) · "
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
                          help="This pile's rows suggest different codes. "
                               "Decide them one at a time.")
                c3.button("Not this pile", width="stretch",
                          key=f"rev_group_reject_disabled_{g['group_id']}",
                          disabled=True,
                          help="Split piles stay in the list so you can "
                               "pick the right code per row.")
            else:
                if c2.button(
                        f"Looks right → {g['suggested']}",
                        type="primary", width="stretch",
                        key=f"rev_group_approve_{g['group_id']}",
                        help="Fill every blank row in this pile with the "
                             "suggested code and remember them. Existing "
                             "codes are never overwritten."):
                    common.push_undo()
                    out = sh.approve_similarity_group(
                        work, loaded, storage, config, g["group_id"],
                        client_id=client_id)
                    n = int(out["approved"])
                    st.toast(f"Approved {n} · remembered {n}")
                    common.set_flash(
                        f"Pile #{g['group_id']}: approved {n} row(s) at "
                        f"{out['code']} — remembered for next time.")
                    _bump()
                    st.rerun()
                if c3.button(
                        "Not this pile",
                        width="stretch",
                        key=f"rev_group_reject_{g['group_id']}",
                        help="Leave these leftovers blank and block the "
                             "suggested pairing so it does not come back."):
                    common.push_undo()
                    indices = list(g.get("indices") or [])
                    out = sh.reject_rows(work, loaded, storage, indices,
                                         client_id=client_id)
                    st.toast(
                        f"Left {out['rejected']} blank · "
                        f"blocked {out['blocked']}")
                    common.set_flash(
                        f"Pile #{g['group_id']}: rejected {out['rejected']} "
                        f"row(s) — left blank, not remembered.")
                    _bump()
                    st.rerun()


# --------------------------------------------------------------------------- #
# Bulk action bar — Review-page rows only (never spreadsheet ticks)
# --------------------------------------------------------------------------- #
def _render_bulk_bar(work, loaded, storage, config, client_id, table) -> None:
    visible = [int(r) for r in table["row"].tolist()]
    cutoff = float(config.confidence.auto_apply_cutoff)
    confident = [int(r) for r in table.loc[
        table["Confidence"].astype(float) >= cutoff, "row"].tolist()]
    with_code = []
    for _, row in table.iterrows():
        if str(row.get("New Account", "") or "").strip() or str(
                row.get("Suggested", "") or "").strip():
            with_code.append(int(row["row"]))

    with st.container(border=True):
        st.markdown("**Quick actions** — these use the leftovers on this page, "
                    "not ticks on the spreadsheet.")
        b1, b2, b3 = st.columns([1.8, 2.0, 1.6])

        if b1.button(f"Looks right on all leftovers ({len(with_code):,})",
                     width="stretch", key="rev_approve_visible",
                     disabled=not with_code,
                     help="Approve every leftover that already has a "
                          "suggested or typed code."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=with_code, client_id=client_id)
            st.toast(f"Approved {out['applied']} · remembered {out['learned']}")
            _bump()
            st.rerun()

        if b2.button(
                f"Looks right on high-confidence ({len(confident):,} ≥ "
                f"{cutoff:.0%})",
                width="stretch", key="rev_approve_conf",
                disabled=not confident,
                help="Approve leftovers at or above the auto-fill cutoff."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=confident, client_id=client_id)
            st.toast(f"Approved {out['applied']} · remembered {out['learned']}")
            _bump()
            st.rerun()

        if b3.button(f"Not this — all leftovers ({len(visible):,})",
                     width="stretch", key="rev_reject_visible",
                     help="Leave every leftover on this page blank and do "
                          "not remember the suggestions."):
            common.push_undo()
            out = sh.reject_rows(work, loaded, storage, visible,
                                 client_id=client_id)
            st.toast(
                f"Left {out['rejected']} blank · blocked {out['blocked']}")
            common.set_flash(
                f"Rejected {out['rejected']} leftover(s) — left blank, "
                "not remembered.")
            _bump()
            st.rerun()

        with st.expander("More options", expanded=False):
            st.toggle(
                "Also show rows with no suggestion at all",
                value=include_no_match_value(),
                key="review_include_no_match",
                help="No-suggestion rows have no proposed code — include "
                     "them to code everything from this screen.")
            r1, r2 = st.columns([2, 1.6])
            recode_val = r1.text_input(
                "Fix leftovers to account", key="rev_recode_code",
                placeholder="e.g. 6322")
            if r2.button(f"Apply fix to {len(visible):,} leftovers",
                         width="stretch", key="rev_recode_apply",
                         disabled=not recode_val.strip()):
                common.push_undo()
                out = sh.recode_rows(work, loaded, storage, visible,
                                     recode_val, client_id=client_id)
                msg = (f"Re-coded {out['recoded']} · remembered "
                       f"{out['learned']}.")
                if out["skipped_protected"]:
                    msg += (f" Skipped {out['skipped_protected']} already-"
                            "coded (protected).")
                st.toast(msg)
                common.set_flash(msg)
                _bump()
                st.rerun()


def include_no_match_value() -> bool:
    return bool(st.session_state.get("review_include_no_match", False))


# --------------------------------------------------------------------------- #
# The editable review table (batched decisions, one Save click)
# --------------------------------------------------------------------------- #
def _render_review_table(work, loaded, storage, config, client_id,
                         table) -> None:
    st.subheader(f"All leftovers ({len(table):,})")
    st.caption(
        "Least sure first. Tick **Looks right**, correct the account if "
        "needed, then click **Save my decisions** once. Saving writes the "
        "code, protects it, and teaches the tool.")

    keep = [c for c in table.columns if c not in
            ("row", "Current", "Engine", "Action")]
    display = table[keep]
    # Accountant labels on the tick column.
    display = display.rename(columns={"Approve": "Looks right"})

    edited = st.data_editor(
        display, width="stretch", hide_index=True, key="review_table_editor",
        height=min(600, 80 + 35 * len(display)),
        column_config={
            "Looks right": st.column_config.CheckboxColumn(
                "Looks right", width="small"),
            "New Account": st.column_config.TextColumn(
                "Account", help="The code to apply when you save."),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "Why": st.column_config.TextColumn("Why", width="large"),
        },
        disabled=[c for c in display.columns
                  if c not in ("Looks right", "New Account")],
    )

    a1, a2, a3 = st.columns([1.8, 1.8, 3])
    n_approved = int(edited["Looks right"].astype(bool).sum()) \
        if "Looks right" in edited.columns else 0
    if a1.button(f"Save my decisions ({n_approved:,})", type="primary",
                 width="stretch", key="rev_apply_table",
                 disabled=n_approved == 0,
                 help="Write the ticked codes, protect them, and remember "
                      "them. Unticked rows are left alone."):
        common.push_undo()
        merged = edited.rename(columns={"Looks right": "Approve"}).copy()
        merged["row"] = table["row"].to_numpy()
        out = sh.apply_review_table(work, loaded, storage, merged,
                                    client_id=client_id)
        st.toast(f"Saved {out['applied']} · remembered {out['learned']}")
        common.set_flash(
            f"Review saved: {out['applied']} code(s) written, "
            f"{out['learned']} remembered"
            + (f", {out['cleared']} cleared" if out["cleared"] else "") + ".")
        _bump()
        st.rerun()

    options = {
        int(r): f"row {int(r) + 1} — {lbl}"
        for r, lbl in zip(table["row"], _row_labels(table))}
    pick = a2.selectbox("Always code this vendor…", [None] + list(options),
                        key="rev_promote_pick", format_func=lambda v:
                        "Always code this vendor…" if v is None else options[v],
                        label_visibility="collapsed")
    if a3.button("Turn into a rule", width="stretch", key="rev_promote_btn",
                 disabled=pick is None,
                 help="Reusable keyword rule, pre-filled from Name, then Memo."):
        row_idx = int(pick)
        code = str(work.at[row_idx, loaded.new_account_col] or "").strip() \
            or str(work.at[row_idx, sh.SUGGESTED_COL] or "").strip()
        prefill = sh.rule_creation_prefill(work, [row_idx], loaded,
                                           rules_manager=services()[2],
                                           config=config)
        if code:
            prefill["code"] = code
        rc.open_rule_panel(prefill)
        st.rerun()


def _row_labels(table) -> list:
    labels = []
    for _, row in table.iterrows():
        labels.append(_row_label_from_series(row))
    return labels


def _row_label_from_series(row) -> str:
    text_cols = [c for c in ("Name", "Payee", "Vendor", "Memo", "Description",
                             "Amount", "Date")
                 if c in getattr(row, "index", [])]
    parts = [str(row[c]).strip() for c in text_cols if str(row[c]).strip()]
    sug = str(row["Suggested"]).strip() if "Suggested" in getattr(
        row, "index", []) else ""
    return " | ".join(parts)[:80] + (f" → {sug}" if sug else "")


def _bump() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
