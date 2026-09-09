"""The Review workspace — leftovers get a Keep / Fix / Not this decision.

A first-class, full-width view that shows ONLY the rows that need a human
(``filled_review`` / ``needs_review``, optionally ``no_match``), sorted
least-confident first and grouped by similarity cluster.

The accountant loop is three verbs:

* **Keep** — accept the suggested code, protect it, teach memory.
* **Fix** — type the right account and learn that instead.
* **Not this** — leave the row blank and *block* the bad pairing so it
  does not come back.

Streamlit has no reliable per-row button inside ``st.data_editor``, so the
table still batches ticks into one Apply click. Group cards and the inbox
card at the top are one-click. All mutations go through
``src/spreadsheet_helpers`` and honour the hard invariant: existing codes
(seeds / manual edits) are never overwritten.
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
        f"{loaded.source_name} · {counts['review_pending']:,} leftover(s) "
        f"need a decision · {counts['blank']:,} still blank")
    if head[1].button("← Back to spreadsheet", width="stretch"):
        st.session_state["view"] = "spreadsheet"
        st.rerun()

    include_no_match = st.toggle(
        "Also show rows with no suggestion yet",
        value=bool(st.session_state.get("review_include_no_match", False)),
        key="review_include_no_match",
        help="Include blank rows the run could not suggest a code for, so you "
             "can type one from this screen.")

    table = sh.review_rows_df(work, loaded, include_no_match=include_no_match)

    if table.empty:
        st.success(
            f"Queue is clear. {counts['filled']:,} of "
            f"{counts['total']:,} rows are coded"
            + (f" ({counts['blank']:,} still blank — turn on ‘no suggestion "
               f"yet’ above, or add a rule and run again)"
               if counts["blank"] else "")
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

    _render_inbox_card(work, loaded, storage, config, client_id, table)
    _render_group_cards(work, loaded, storage, config, client_id)
    _render_decide_bar(work, loaded, storage, config, client_id, table)
    _render_review_table(work, loaded, storage, config, client_id, table)

    if st.session_state.get("rule_panel_open"):
        rc.rule_creation_dialog(work, loaded, config, rules, storage)


# --------------------------------------------------------------------------- #
# Next leftover — one pile, three verbs
# --------------------------------------------------------------------------- #
def _render_inbox_card(work, loaded, storage, config, client_id, table) -> None:
    groups = sh.group_review_summary(
        work, loaded, high_cutoff=float(config.confidence.auto_apply_cutoff))
    consensus = next((g for g in groups if not g["split"] and g["suggested"]),
                     None)

    with st.container(border=True):
        st.markdown("**Next leftover**")
        if consensus is not None:
            st.markdown(
                f"{consensus['rows']} look-alike row(s) → "
                f"**{consensus['suggested']}**"
                + (f" · {consensus['high_conf']} look solid"
                   if consensus["high_conf"] else ""))
            if consensus["sample"]:
                st.caption(consensus["sample"])
            k, f, n = st.columns(3)
            if k.button(f"Keep as {consensus['suggested']}", type="primary",
                        width="stretch", key="rev_inbox_keep",
                        help="Accept this code for every blank row in the "
                             "pile. Existing codes are never overwritten."):
                common.push_undo()
                out = sh.approve_similarity_group(
                    work, loaded, storage, config, consensus["group_id"],
                    client_id=client_id)
                common.set_flash(
                    f"Kept {out['approved']} row(s) as {out['code']}. "
                    "Remembered for next time.")
                _bump()
                st.rerun()
            fix_code = f.text_input(
                "Fix to account", key="rev_inbox_fix_code",
                placeholder="e.g. 6322", label_visibility="collapsed")
            if f.button("Fix these", width="stretch", key="rev_inbox_fix",
                        disabled=not str(fix_code or "").strip(),
                        help="Write this account on the blank rows in the "
                             "pile and remember it. Seeds stay protected."):
                common.push_undo()
                out = sh.recode_rows(
                    work, loaded, storage, list(consensus["indices"]),
                    fix_code, client_id=client_id)
                msg = (f"Fixed {out['recoded']} row(s) to "
                       f"{str(fix_code).strip()}.")
                if out["skipped_protected"]:
                    msg += f" Left {out['skipped_protected']} already-coded."
                common.set_flash(msg)
                _bump()
                st.rerun()
            if n.button("Not this", width="stretch", key="rev_inbox_not",
                        help="Leave these blank and do not learn the "
                             "suggestion — it will not be proposed again."):
                common.push_undo()
                out = sh.reject_rows(
                    work, loaded, storage, list(consensus["indices"]),
                    client_id=client_id)
                common.set_flash(
                    f"Not this — {out['rejected']} row(s) left blank, "
                    "not remembered.")
                _bump()
                st.rerun()
            return

        # No consensus pile: decide the least-confident single row.
        top = table.iloc[0]
        idx = int(top["row"])
        suggested = str(top.get("Suggested") or top.get("New Account") or "").strip()
        label_bits = [str(top[c]).strip() for c in
                      ("Name", "Payee", "Memo", "Description")
                      if c in table.columns and str(top.get(c, "")).strip()]
        st.markdown(
            (" | ".join(label_bits)[:90] if label_bits else f"Row {idx + 1}")
            + (f" → **{suggested}**" if suggested else " — no suggestion yet"))
        why = str(top.get("Why") or "").strip()
        if why:
            st.caption(why)
        k, f, n = st.columns(3)
        if k.button(
                f"Keep as {suggested}" if suggested else "Keep",
                type="primary", width="stretch", key="rev_inbox_keep_row",
                disabled=not suggested,
                help="Accept the suggested code on this row and remember it."):
            common.push_undo()
            out = sh.apply_suggested_to_rows(
                work, loaded, storage, [idx], client_id=client_id)
            if out["applied"] == 0 and suggested:
                out = sh.approve_rows(work, loaded, storage, config,
                                      indices=[idx], client_id=client_id)
            common.set_flash(
                f"Kept row {idx + 1} as {suggested}. Remembered for next time.")
            _bump()
            st.rerun()
        fix_code = f.text_input(
            "Fix to account", key="rev_inbox_fix_row_code",
            placeholder="e.g. 6322", label_visibility="collapsed")
        if f.button("Fix this", width="stretch", key="rev_inbox_fix_row",
                    disabled=not str(fix_code or "").strip()):
            common.push_undo()
            out = sh.recode_rows(work, loaded, storage, [idx], fix_code,
                                 client_id=client_id)
            common.set_flash(
                f"Fixed row {idx + 1} to {str(fix_code).strip()}."
                if out["recoded"] else
                "That row already has a protected code — it was left alone.")
            _bump()
            st.rerun()
        if n.button("Not this", width="stretch", key="rev_inbox_not_row",
                    help="Leave this row blank and block the suggestion."):
            common.push_undo()
            out = sh.reject_rows(work, loaded, storage, [idx],
                                 client_id=client_id)
            common.set_flash(
                f"Not this — row {idx + 1} left blank, not remembered.")
            _bump()
            st.rerun()


# --------------------------------------------------------------------------- #
# Similarity groups — one-click Keep / Not this
# --------------------------------------------------------------------------- #
def _render_group_cards(work, loaded, storage, config, client_id) -> None:
    groups = sh.group_review_summary(
        work, loaded, high_cutoff=float(config.confidence.auto_apply_cutoff))
    if not groups:
        return

    st.subheader(f"Look-alike piles ({len(groups)})")
    st.caption("Same-looking transactions stay together. Keep the pile when "
               "the code is right. Not this leaves them blank and will not "
               "learn the suggestion. Split piles need a row-by-row pick.")
    for g in groups[:15]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3.4, 1.5, 1.3])
            title = (f"**{g['rows']} rows** · suggested "
                     f"**{g['suggested'] or '—'}**"
                     + (f" · {g['high_conf']} look solid"
                        if g["high_conf"] else ""))
            c1.markdown(title)
            if g["sample"]:
                c1.caption(g["sample"])
            if g["split"]:
                split_txt = ", ".join(f"{code} ×{n}"
                                      for code, n in g["codes"].items())
                c2.button(f"Split: {split_txt}", width="stretch",
                          key=f"rev_group_split_{g['group_id']}",
                          disabled=True,
                          help="This pile suggests different codes. Decide "
                               "them in the table below.")
                c3.button("Not this pile", width="stretch",
                          key=f"rev_group_not_disabled_{g['group_id']}",
                          disabled=True)
            else:
                if c2.button(
                        f"Keep as {g['suggested']}",
                        type="primary", width="stretch",
                        key=f"rev_group_approve_{g['group_id']}",
                        help="Fill every blank row in this pile with the "
                             "suggested code and remember it. Existing codes "
                             "are never overwritten."):
                    common.push_undo()
                    out = sh.approve_similarity_group(
                        work, loaded, storage, config, g["group_id"],
                        client_id=client_id)
                    common.set_flash(
                        f"Kept {out['approved']} row(s) as {out['code']}. "
                        "Remembered for next time.")
                    _bump()
                    st.rerun()
                if c3.button(
                        "Not this", width="stretch",
                        key=f"rev_group_not_{g['group_id']}",
                        help="Leave this pile blank and block the pairing."):
                    common.push_undo()
                    out = sh.reject_rows(
                        work, loaded, storage, list(g["indices"]),
                        client_id=client_id)
                    common.set_flash(
                        f"Not this — {out['rejected']} row(s) left blank, "
                        "not remembered.")
                    _bump()
                    st.rerun()


# --------------------------------------------------------------------------- #
# Decide bar — actions use THIS page's Pick ticks, not the spreadsheet
# --------------------------------------------------------------------------- #
def _render_decide_bar(work, loaded, storage, config, client_id, table) -> None:
    visible = [int(r) for r in table["row"].tolist()]
    picked = _picked_indices(table)

    with st.container(border=True):
        st.markdown("**Decide several at once** — ticks live on this page. "
                    "Existing codes are never overwritten.")
        b1, b2, b3 = st.columns([1.8, 1.8, 1.8])

        if b1.button(f"Keep all leftovers ({len(visible):,})",
                     width="stretch", key="rev_approve_visible",
                     help="Keep the suggested code on every leftover in this "
                          "list."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=visible, client_id=client_id)
            common.set_flash(
                f"Kept {out['applied']} · remembered {out['learned']}.")
            _bump()
            st.rerun()

        high_cut = float(config.confidence.auto_apply_cutoff)
        confident = [int(r) for r in table.loc[
            table["Confidence"].astype(float) >= high_cut, "row"].tolist()]
        if b2.button(
                f"Keep the solid ones ({len(confident):,})",
                width="stretch", key="rev_approve_conf",
                disabled=not confident,
                help=f"Keep leftovers at or above {high_cut:.0%} confidence."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=confident, client_id=client_id)
            common.set_flash(
                f"Kept {out['applied']} · remembered {out['learned']}.")
            _bump()
            st.rerun()

        if b3.button(f"Not this on picked ({len(picked):,})",
                     width="stretch", key="rev_reject_picked",
                     disabled=not picked,
                     help="Leave picked rows blank and do not remember the "
                          "suggestion. Tick Pick in the table first."):
            common.push_undo()
            out = sh.reject_rows(work, loaded, storage, picked,
                                 client_id=client_id)
            common.set_flash(
                f"Not this — {out['rejected']} row(s) left blank, "
                "not remembered.")
            _bump()
            st.rerun()

        with st.expander("Fix picked rows to a different account"):
            r1, r2 = st.columns([2.2, 1.6])
            recode_val = r1.text_input(
                "Account code", key="rev_recode_code",
                placeholder="e.g. 6322")
            if r2.button(f"Fix {len(picked):,} picked row(s)",
                         width="stretch", key="rev_recode_apply",
                         disabled=not (picked and str(recode_val).strip())):
                common.push_undo()
                out = sh.recode_rows(work, loaded, storage, picked, recode_val,
                                     client_id=client_id)
                msg = (f"Fixed {out['recoded']} row(s) to "
                       f"{str(recode_val).strip()} · remembered "
                       f"{out['learned']}.")
                if out["skipped_protected"]:
                    msg += (f" Left {out['skipped_protected']} already-coded "
                            "row(s) alone.")
                common.set_flash(msg)
                _bump()
                st.rerun()


# --------------------------------------------------------------------------- #
# Editable table — Pick / Keep / account, one Apply
# --------------------------------------------------------------------------- #
def _render_review_table(work, loaded, storage, config, client_id,
                         table) -> None:
    st.subheader(f"Leftovers ({len(table):,})")
    st.caption(
        "Least sure first. Tick **Keep** on rows that look right (or type a "
        "better **Account**), tick **Pick** for Not this / Fix above, then "
        "**Apply keeps**. One click writes the codes and remembers them.")

    display = table.drop(columns=[c for c in
                                  ("row", "Current", "Engine", "Action")
                                  if c in table.columns]).copy()
    display.insert(0, "Pick", False)
    if "Approve" in display.columns:
        display = display.rename(columns={"Approve": "Keep",
                                          "New Account": "Account"})
    elif "New Account" in display.columns:
        display = display.rename(columns={"New Account": "Account"})
        display["Keep"] = False

    # Preferred column order for an accountant, not an engineer.
    preferred = ["Pick", "Keep", "Account", "Suggested", "Confidence", "Why"]
    rest = [c for c in display.columns if c not in preferred]
    display = display[[c for c in preferred if c in display.columns] + rest]

    edited = st.data_editor(
        display, width="stretch", hide_index=True, key="review_table_editor",
        height=min(600, 80 + 35 * len(display)),
        column_config={
            "Pick": st.column_config.CheckboxColumn(
                "Pick", width="small",
                help="Mark rows for Not this or Fix above."),
            "Keep": st.column_config.CheckboxColumn(
                "Keep", width="small",
                help="Accept the Account on this row."),
            "Account": st.column_config.TextColumn(
                "Account", help="The code to write when you Apply keeps."),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "Why": st.column_config.TextColumn("Why", width="large"),
        },
        disabled=[c for c in display.columns
                  if c not in ("Pick", "Keep", "Account")],
    )

    # Persist Pick ticks so the decide bar above can read them next rerun.
    _store_picks(table, edited)

    a1, a2, a3 = st.columns([1.8, 1.8, 3])
    n_keep = int(edited["Keep"].astype(bool).sum()) if "Keep" in edited.columns else 0
    if a1.button(f"Apply keeps ({n_keep:,})", type="primary",
                 width="stretch", key="rev_apply_table",
                 disabled=n_keep == 0,
                 help="Write the kept accounts, protect them, and remember "
                      "them. Unticked rows are left alone."):
        common.push_undo()
        merged = edited.rename(columns={"Keep": "Approve",
                                        "Account": "New Account"}).copy()
        merged["row"] = table["row"].to_numpy()
        out = sh.apply_review_table(work, loaded, storage, merged,
                                    client_id=client_id)
        common.set_flash(
            f"Kept {out['applied']} · remembered {out['learned']}"
            + (f" · cleared {out['cleared']}" if out.get("cleared") else "")
            + ".")
        _bump()
        st.rerun()

    options = {
        int(r): f"row {int(r) + 1} — {lbl}"
        for r, lbl in zip(table["row"], _row_labels(table))}
    pick = a2.selectbox("Turn a leftover into a reusable rule", [None] + list(options),
                        key="rev_promote_pick", format_func=lambda v:
                        "Always code this vendor…" if v is None else options[v],
                        label_visibility="collapsed")
    if a3.button("Always code this vendor", width="stretch",
                 key="rev_promote_btn",
                 disabled=pick is None,
                 help="Turn this row into a keyword rule, pre-filled from "
                      "Name then Memo — never Notes."):
        row_idx = int(pick)
        code = str(work.at[row_idx, loaded.new_account_col] or "").strip() \
            or str(work.at[row_idx, sh.SUGGESTED_COL] or "").strip()
        prefill = sh.rule_creation_prefill(work, [row_idx], loaded,
                                           rules_manager=rules, config=config)
        if code:
            prefill["code"] = code
        rc.open_rule_panel(prefill)
        st.rerun()


def _store_picks(table, edited) -> None:
    """Remember which leftover rows are Picked so the decide bar can use them."""
    if edited is None or edited.empty or "Pick" not in edited.columns:
        st.session_state["review_picked_idx"] = []
        return
    flags = edited["Pick"].astype(bool).tolist()
    rows = table["row"].tolist()
    st.session_state["review_picked_idx"] = [
        int(r) for r, flag in zip(rows, flags) if flag]


def _picked_indices(table) -> list:
    """Leftover row indices ticked Pick on this page (not the spreadsheet)."""
    stored = st.session_state.get("review_picked_idx") or []
    visible = set(int(r) for r in table["row"].tolist())
    return [i for i in stored if i in visible]


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
    st.session_state["review_picked_idx"] = []
