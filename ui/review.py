"""The Review inbox — leftovers get a Keep / Fix / Not this decision.

Shows ONLY rows that need a human, least-sure first. The accountant loop is
three verbs:

* **Keep** — accept the suggested account, protect it, remember it.
* **Fix** — type the right account; seeds / manual codes stay sacred.
* **Not this** — leave blank and block the pairing so it is not suggested again.

Look-alike piles Keep or Not this in one click. Split piles never one-click
Keep; they may Not this. All mutations go through ``src/spreadsheet_helpers``
and honour: existing codes are never overwritten; reject never learns.
"""

from __future__ import annotations

import streamlit as st

from src import spreadsheet_helpers as sh
from ui import common
from ui import rule_creation as rc
from ui.common import services

_MAX_EDITOR_ROWS = 40


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

    client_id = common.current_client_id()
    counts = sh.summary_counts(work, loaded)

    head = st.columns([3.4, 2])
    client = client_id or ""
    head[0].markdown("### Review" + (f" — {client}" if client else ""))
    head[0].caption(
        f"{loaded.source_name} · {counts['review_pending']:,} leftover(s) "
        f"need a decision · {counts['blank']:,} still blank")
    if head[1].button("← Back to spreadsheet", width="stretch"):
        st.session_state["view"] = "spreadsheet"
        st.rerun()

    st.session_state.setdefault("review_include_no_match", False)
    include_no_match = st.toggle(
        "Also show rows with no suggestion yet",
        key="review_include_no_match",
        help="Include blank rows the run could not suggest a code for, so you "
             "can type one from this screen.")

    table = sh.review_rows_df(work, loaded, include_no_match=include_no_match)

    if table.empty:
        _render_empty_queue(counts)
        return

    leftover = sh.next_leftover(
        work, loaded, table,
        high_cutoff=float(config.confidence.auto_apply_cutoff))
    skip_gid = (leftover["group"]["group_id"]
                if leftover and leftover["kind"] == "group" else None)

    toast = st.session_state.pop("review_toast", None)
    if toast:
        st.toast(toast)

    _render_inbox_card(work, loaded, storage, config, client_id, leftover)
    _render_group_cards(work, loaded, storage, config, client_id,
                        skip_group_id=skip_gid)

    # A 60–200 row data_editor makes Keep / Fix / Not this feel stuck after
    # Strict. Next leftover is the accountant path; the table is optional.
    st.session_state.setdefault("review_show_leftover_table", len(table) <= 20)
    show_table = True
    if len(table) > 20:
        show_table = st.toggle(
            f"Show leftover table ({len(table):,})",
            key="review_show_leftover_table",
            help="Tick Keep or Pick on many rows at once. Leave this off so "
                 "Next leftover stays fast.")
    if show_table:
        visible = table.head(_MAX_EDITOR_ROWS)
        if len(table) > _MAX_EDITOR_ROWS:
            st.caption(
                f"Showing the {_MAX_EDITOR_ROWS} least-sure of "
                f"{len(table):,}. Use Next leftover for the rest.")
        edited = _render_review_table(work, loaded, visible)
        _render_decide_bar(work, loaded, storage, config, client_id, visible,
                           edited)
        _render_apply_and_promote(work, loaded, storage, config, client_id,
                                  rules, visible, edited)
    else:
        _render_promote_only(work, loaded, rules, config, table)

    if st.session_state.get("rule_panel_open"):
        rc.rule_creation_dialog(work, loaded, config, rules, storage)


def _render_empty_queue(counts: dict) -> None:
    blanks = int(counts.get("blank") or 0)
    extra = ""
    if blanks and not st.session_state.get("review_include_no_match"):
        extra = (f" ({blanks:,} still blank — turn on ‘no suggestion yet’ "
                 "above, or add a rule and run again)")
    elif blanks:
        extra = f" ({blanks:,} still blank — add a rule and run again)"
    st.success(
        f"Queue is clear. {counts['filled']:,} of "
        f"{counts['total']:,} rows are coded{extra}. Export when you're ready.")
    c1, c2, _ = st.columns([1.4, 1.4, 3])
    if c1.button("Open spreadsheet", width="stretch", key="rev_empty_grid"):
        st.session_state["view"] = "spreadsheet"
        st.rerun()
    if c2.button("Export", width="stretch", key="rev_empty_export"):
        st.session_state["view"] = "spreadsheet"
        st.session_state["export_open"] = True
        st.session_state["panel"] = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Next leftover — one pile or one row, three verbs
# --------------------------------------------------------------------------- #
def _render_inbox_card(work, loaded, storage, config, client_id,
                       leftover) -> None:
    if leftover is None:
        return

    with st.container(border=True):
        st.markdown("**Next leftover**")
        if leftover["kind"] == "group":
            _render_inbox_group(work, loaded, storage, config, client_id,
                                leftover["group"])
        else:
            _render_inbox_row(work, loaded, storage, config, client_id,
                              leftover)


def _render_inbox_group(work, loaded, storage, config, client_id, g) -> None:
    st.markdown(
        f"{g['rows']} look-alike row(s) → **{g['suggested']}**"
        + (f" · {g['high_conf']} look solid" if g["high_conf"] else ""))
    if g.get("sample"):
        st.caption(g["sample"])
    k, f, n = st.columns(3)
    if k.button(f"Keep as {g['suggested']}", type="primary",
                width="stretch", key="rev_inbox_keep",
                help="Accept this code for every blank row in the pile. "
                     "Existing codes are never overwritten."):
        common.push_undo()
        out = sh.approve_similarity_group(
            work, loaded, storage, config, g["group_id"],
            client_id=client_id)
        _done(f"Kept {out['approved']} row(s) as {out['code']}. "
              "Remembered for next time.")
    with f.form("rev_inbox_fix_form"):
        fix_code = st.text_input(
            "Fix to account", placeholder="e.g. 6322",
            label_visibility="collapsed")
        if st.form_submit_button("Fix", width="stretch",
                                 help="Write this account on the blank rows "
                                      "in the pile and remember it. Seeds "
                                      "stay protected."):
            if not str(fix_code or "").strip():
                st.warning("Type an account first.")
            else:
                common.push_undo()
                out = sh.recode_rows(
                    work, loaded, storage, list(g["indices"]),
                    fix_code, client_id=client_id)
                msg = (f"Fixed {out['recoded']} row(s) to "
                       f"{str(fix_code).strip()}.")
                if out["skipped_protected"]:
                    msg += f" Left {out['skipped_protected']} already-coded."
                if out["recoded"]:
                    msg += " Remembered for next time."
                _done(msg)
    if n.button("Not this", width="stretch", key="rev_inbox_not",
                help="Leave these blank and do not remember the suggestion."):
        common.push_undo()
        out = sh.reject_similarity_group(
            work, loaded, storage, g["group_id"], client_id=client_id)
        _done(f"Not this — {out['rejected']} row(s) left blank, "
              "not remembered.")


def _render_inbox_row(work, loaded, storage, config, client_id,
                      leftover) -> None:
    rec = leftover["record"]
    idx = int(leftover["row"])
    suggested = str(rec.get("Suggested") or rec.get("New Account")
                    or rec.get("Account") or "").strip()
    label_bits = [str(rec[c]).strip() for c in
                  ("Name", "Payee", "Memo", "Description")
                  if c in rec.index and str(rec.get(c, "")).strip()]
    st.markdown(
        f"Row {idx + 1} · "
        + (" | ".join(label_bits)[:90] if label_bits else "leftover")
        + (f" → **{suggested}**" if suggested else " — no suggestion yet"))
    why = str(rec.get("Why") or "").strip()
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
        _done(f"Kept row {idx + 1} as {suggested}. Remembered for next time.")
    with f.form("rev_inbox_fix_row_form"):
        fix_code = st.text_input(
            "Fix to account", placeholder="e.g. 6322",
            label_visibility="collapsed")
        if st.form_submit_button("Fix", width="stretch"):
            if not str(fix_code or "").strip():
                st.warning("Type an account first.")
            else:
                common.push_undo()
                out = sh.recode_rows(work, loaded, storage, [idx], fix_code,
                                     client_id=client_id)
                if out["recoded"]:
                    _done(f"Fixed row {idx + 1} to {str(fix_code).strip()}. "
                          "Remembered for next time.")
                else:
                    _done("That row already has a protected code — "
                          "it was left alone.")
    if n.button("Not this", width="stretch", key="rev_inbox_not_row",
                help="Leave this row blank and block the suggestion."):
        common.push_undo()
        out = sh.reject_rows(work, loaded, storage, [idx],
                             client_id=client_id)
        _done(f"Not this — row {idx + 1} left blank, not remembered.")


# --------------------------------------------------------------------------- #
# Look-alike piles — Keep / Not this (split piles: Not this only)
# --------------------------------------------------------------------------- #
def _render_group_cards(work, loaded, storage, config, client_id,
                        skip_group_id=None) -> None:
    groups = sh.group_review_summary(
        work, loaded, high_cutoff=float(config.confidence.auto_apply_cutoff))
    if skip_group_id is not None:
        groups = [g for g in groups if g["group_id"] != skip_group_id]
    if not groups:
        return

    st.subheader(f"Look-alike piles ({len(groups)})")
    st.caption("Same-looking transactions stay together. Keep the pile when "
               "the code is right. Not this leaves them blank and will not "
               "remember the suggestion. Split piles are never one-click Keep.")
    for g in groups[:15]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3.4, 1.5, 1.3])
            title = (f"**{g['rows']} rows** · suggested "
                     f"**{g['suggested'] or '—'}**"
                     + (f" · {g['high_conf']} look solid"
                        if g["high_conf"] else ""))
            c1.markdown(title)
            if g.get("sample"):
                c1.caption(g["sample"])
            if g["split"]:
                split_txt = ", ".join(f"{code} ×{n}"
                                      for code, n in g["codes"].items())
                c2.button(f"Split: {split_txt}", width="stretch",
                          key=f"rev_group_split_{g['group_id']}",
                          disabled=True,
                          help="This pile suggests different codes. Decide "
                               "them with Keep / Fix on the table below.")
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
                    _done(f"Kept {out['approved']} row(s) as {out['code']}. "
                          "Remembered for next time.")
            if c3.button(
                    "Not this", width="stretch",
                    key=f"rev_group_not_{g['group_id']}",
                    help="Leave this pile blank and block every suggestion "
                         "in it. Seeds stay."):
                common.push_undo()
                out = sh.reject_similarity_group(
                    work, loaded, storage, g["group_id"],
                    client_id=client_id)
                _done(f"Not this — {out['rejected']} row(s) left blank, "
                      "not remembered.")


# --------------------------------------------------------------------------- #
# Table first, then decide bar (same-run Pick ticks)
# --------------------------------------------------------------------------- #
def _render_review_table(work, loaded, table):
    st.subheader(f"Leftovers ({len(table):,})")
    st.caption(
        "Least sure first. Tick **Keep** on rows that look right (or type a "
        "better **Account**), tick **Pick** for Not this / Fix below, then "
        "**Apply keeps**.")

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

    preferred = ["Pick", "Keep", "Account", "Suggested", "Confidence", "Why"]
    rest = [c for c in display.columns if c not in preferred]
    display = display[[c for c in preferred if c in display.columns] + rest]

    return st.data_editor(
        display, width="stretch", hide_index=True,
        key="review_table_editor_inbox",
        height=min(600, 80 + 35 * len(display)),
        column_config={
            "Pick": st.column_config.CheckboxColumn(
                "Pick", width="small",
                help="Mark rows for Not this or Fix below."),
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


def _picks_from_edited(table, edited) -> list:
    """Review-page Pick ticks from this run's editor return value."""
    if edited is None or edited.empty or "Pick" not in edited.columns:
        return []
    flags = edited["Pick"].astype(bool).tolist()
    rows = table["row"].tolist()
    return [int(r) for r, flag in zip(rows, flags) if flag]


def _render_decide_bar(work, loaded, storage, config, client_id, table,
                       edited) -> None:
    picked = _picks_from_edited(table, edited)
    visible = [int(r) for r in table["row"].tolist()]

    with st.container(border=True):
        st.markdown("**Decide picked rows** — ticks live on this page. "
                    "Existing codes are never overwritten.")
        b1, b2, b3 = st.columns([1.6, 2.2, 1.6])
        if b1.button(f"Keep ({len(picked):,})", type="primary",
                     width="stretch", key="rev_keep_picked",
                     disabled=not picked,
                     help="Keep the Account / suggestion on picked leftovers."):
            common.push_undo()
            out = sh.approve_rows(work, loaded, storage, config,
                                  indices=picked, client_id=client_id)
            _done(f"Kept {out['applied']}. Remembered for next time.")
        with b2.form("rev_recode_form"):
            fix_code = st.text_input(
                "Fix to account", placeholder="e.g. 6322",
                label_visibility="collapsed")
            if st.form_submit_button(f"Fix ({len(picked):,})",
                                     width="stretch",
                                     disabled=not picked):
                if not str(fix_code or "").strip():
                    st.warning("Type an account first.")
                else:
                    common.push_undo()
                    out = sh.recode_rows(work, loaded, storage, picked,
                                         fix_code, client_id=client_id)
                    msg = (f"Fixed {out['recoded']} row(s) to "
                           f"{str(fix_code).strip()}.")
                    if out["skipped_protected"]:
                        msg += (f" Left {out['skipped_protected']} "
                                "already-coded alone.")
                    if out["recoded"]:
                        msg += " Remembered for next time."
                    _done(msg)
        if b3.button(f"Not this ({len(picked):,})", width="stretch",
                     key="rev_reject_picked",
                     disabled=not picked,
                     help="Leave picked rows blank and do not remember the "
                          "suggestion."):
            common.push_undo()
            out = sh.reject_rows(work, loaded, storage, picked,
                                 client_id=client_id)
            _done(f"Not this — {out['rejected']} row(s) left blank, "
                  "not remembered.")

        with st.expander("More leftovers at once"):
            if st.button(f"Keep all leftovers ({len(visible):,})",
                         width="stretch", key="rev_approve_visible",
                         help="Keep the suggested code on every leftover "
                              "in this list."):
                common.push_undo()
                out = sh.approve_rows(work, loaded, storage, config,
                                      indices=visible, client_id=client_id)
                _done(f"Kept {out['applied']}. Remembered for next time.")


def _render_apply_and_promote(work, loaded, storage, config, client_id,
                              rules, table, edited) -> None:
    a1, a2, a3 = st.columns([1.8, 1.8, 3])
    n_keep = int(edited["Keep"].astype(bool).sum()) \
        if edited is not None and "Keep" in edited.columns else 0
    if a1.button(f"Apply keeps ({n_keep:,})", type="primary",
                 width="stretch", key="rev_apply_table",
                 disabled=n_keep == 0,
                 help="Write the kept accounts, protect them, and remember "
                      "them. Unticked rows are left alone."):
        common.push_undo()
        merged = edited.copy()
        merged["row"] = table["row"].to_numpy()
        out = sh.apply_review_table(work, loaded, storage, merged,
                                    client_id=client_id)
        extra = f" · cleared {out['cleared']}" if out.get("cleared") else ""
        _done(f"Kept {out['applied']}. Remembered for next time.{extra}")

    options = {
        int(r): f"row {int(r) + 1} — {lbl}"
        for r, lbl in zip(table["row"], _row_labels(table))}
    pick = a2.selectbox(
        "Turn a leftover into a reusable rule", [None] + list(options),
        key="rev_promote_pick",
        format_func=lambda v:
        "Always code this vendor…" if v is None else options[v],
        label_visibility="collapsed")
    if a3.button("Always code this vendor", width="stretch",
                 key="rev_promote_btn",
                 disabled=pick is None,
                 help="Turn this row into a keyword rule, pre-filled from "
                      "Name then Memo — never Notes."):
        _open_vendor_rule(work, loaded, rules, config, int(pick))


def _render_promote_only(work, loaded, rules, config, table) -> None:
    """Rule promote without the leftover grid (keeps Next leftover snappy)."""
    options = {
        int(r): f"row {int(r) + 1} — {lbl}"
        for r, lbl in zip(table["row"], _row_labels(table))}
    c1, c2 = st.columns([2.2, 3])
    pick = c1.selectbox(
        "Turn a leftover into a reusable rule", [None] + list(options),
        key="rev_promote_pick",
        format_func=lambda v:
        "Always code this vendor…" if v is None else options[v],
        label_visibility="collapsed")
    if c2.button("Always code this vendor", width="stretch",
                 key="rev_promote_btn",
                 disabled=pick is None,
                 help="Turn this row into a keyword rule, pre-filled from "
                      "Name then Memo — never Notes."):
        _open_vendor_rule(work, loaded, rules, config, int(pick))


def _open_vendor_rule(work, loaded, rules, config, row_idx: int) -> None:
    code = str(work.at[row_idx, loaded.new_account_col] or "").strip() \
        or str(work.at[row_idx, sh.SUGGESTED_COL] or "").strip()
    prefill = sh.rule_creation_prefill(work, [row_idx], loaded,
                                       rules_manager=rules, config=config)
    if code:
        prefill["code"] = code
    rc.open_rule_panel(prefill)
    st.rerun()


def _row_labels(table) -> list:
    text_cols = [c for c in ("Name", "Payee", "Memo", "Description")
                 if c in table.columns]
    labels = []
    for _, row in table.iterrows():
        parts = [str(row[c]).strip() for c in text_cols
                 if str(row[c]).strip()]
        sug = str(row.get("Suggested", "")).strip()
        labels.append(" | ".join(parts)[:60] + (f" → {sug}" if sug else ""))
    return labels


def _done(message: str) -> None:
    common.persist_workspace()
    _bump()
    common.set_flash(message)
    st.session_state["review_toast"] = message
    st.rerun()


def _bump() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    st.session_state["review_picked_idx"] = []
