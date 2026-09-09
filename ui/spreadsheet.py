"""The dominant Spreadsheet workspace — the heart of v2.2.

Design notes on editor <-> work_df sync (the part most likely to drift):

* ``work_df`` (in session state) is the single source of truth.
* The editor is rendered for the current *page* of the current *filter*. Its
  return value is committed straight back into ``work_df`` by row index on every
  rerun (auto-save), so manual edits, selections and Rule Notes never get lost.
* Programmatic mutations (runs, bulk actions, undo/redo, applying a rule) bump a
  ``data_version`` counter. The editor's widget ``key`` includes that counter,
  the page and the filter — so after any programmatic change the grid reloads
  cleanly from ``work_df`` instead of replaying stale edit deltas.
"""

from __future__ import annotations

import streamlit as st

from src import spreadsheet_helpers as sh
from src.data_loader import RULE_NOTES_COL
from src.exporter import (
    export_dataframe,
    export_preserving_original,
    export_csv,
)
from src.similarity import semantic_backend_available
from ui import common
from ui import rule_creation as rc
from ui.common import services

FILTER_MODES = ["All", "Review only", "High confidence", "Blanks only",
                "Filled only", "Filled by rules", "Filled by memory"]
PAGE_SIZES = [100, 250, 500, 1000, 1500, 2000, 5000]
EDITOR_HEIGHT = 720


def _bump() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1


def render_spreadsheet() -> None:
    config, storage, rules, mm, logger = services()
    st.session_state.setdefault("data_version", 0)

    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    if work is None or loaded is None:
        st.info("No file loaded yet.")
        if st.button("Go to upload", type="primary"):
            st.session_state["view"] = "landing"
            st.rerun()
        return

    # A run may ask to switch the quick-view filter (must happen before the
    # filter widgets are instantiated below).
    pending_mode = st.session_state.pop("_pending_filter_mode", None)
    if pending_mode in FILTER_MODES:
        st.session_state["filter_mode"] = pending_mode
        st.session_state["page"] = 0

    counts = sh.summary_counts(work, loaded)

    _render_header(loaded, counts)
    rc.render_memory_bootstrap_banner(work, loaded, rules, storage, config)
    rc.render_ingest_banner(work, loaded, rules, storage, config)
    rc.render_seed_suggestion_banner(work, loaded, rules)
    from ui import panels
    panels.render_auto_train_banner()
    _render_run_metrics(work, loaded, counts)
    _render_rule_audit(work, loaded)

    actions = _render_toolbar(work, loaded, counts, config)

    if counts["selected"]:
        rc.render_selection_strip(work, loaded, rules, counts, config)

    # ---- Filter bar (global search + quick views) -> mask over ALL records.
    mask = _render_filter_bar(work, loaded, config)
    page_size = int(st.session_state.get("page_size", 1500))
    total_filtered = int(mask.sum())
    n_pages = max(1, (total_filtered + page_size - 1) // page_size)
    page = max(0, min(int(st.session_state.get("page", 0)), n_pages - 1))
    st.session_state["page"] = page

    _render_page_controls(page, n_pages, total_filtered, len(work))

    page_df, _, _ = sh.page_slice(work, mask, page, page_size)

    # ---- The editor.
    edited = _render_editor(page_df, work, loaded)

    # ---- Commit manual edits immediately (auto-save) with undo support.
    _commit_with_undo(work, edited, loaded, storage, config, rules)

    rc.render_target_account_prompt(work, loaded)

    # ---- Handle deferred toolbar actions (after edits are safely committed).
    _handle_actions(actions, work, loaded, config, rules, storage, mm, logger)

    # ---- Overlays (only one modal per run; confirmations take precedence so a
    # destructive action is never hidden behind another dialog).
    if st.session_state.get("confirm_action"):
        _confirm_dialog(work, loaded, config, rules, storage)
    elif st.session_state.get("export_open") and not st.session_state.get("panel"):
        _export_dialog(work, loaded, counts)
    elif st.session_state.get("rule_panel_open"):
        rc.rule_creation_dialog(work, loaded, config, rules, storage)


# --------------------------------------------------------------------------- #
# Confirmation dialog for destructive reset actions
# --------------------------------------------------------------------------- #
def _dismiss_confirm() -> None:
    st.session_state["confirm_action"] = None


def _do_clear_spreadsheet() -> None:
    """on_click handler — runs before widgets re-instantiate, so it may safely
    reset widget-keyed state."""
    st.session_state["confirm_action"] = None
    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    config = common.services()[0]
    if work is None or loaded is None:
        return
    common.push_undo()
    n = sh.clear_spreadsheet_values(work, loaded, config)
    _bump()
    common.persist_workspace()
    common.set_flash(f"Cleared Target Account and Rule Notes on {n:,} row(s).")


def _do_reset_run() -> None:
    """on_click handler — un-run the automation so the user can re-run cleanly.

    Clears everything the engine filled (rules / similarity / learned / AI) but
    keeps saved rules, Rule Notes, upload codes and manual edits. Undoable.
    """
    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    config = common.services()[0]
    if work is None or loaded is None:
        return
    common.push_undo()
    n = sh.reset_automation(work, loaded, config)
    # Drop the now-stale result panels.
    st.session_state.pop("show_run_metrics", None)
    st.session_state.pop("last_rule_audit", None)
    st.session_state.pop("last_result", None)
    _bump()
    common.persist_workspace()
    common.set_flash(
        f"Reset the run — cleared {n:,} auto-filled row(s). Your rules, Rule "
        "Notes and codes from the upload are kept. Adjust and run again.")


def _do_start_fresh() -> None:
    """on_click handler — wipes persisted memory and unloads the file.

    Local / public-demo only. Live mode uses ``_do_start_fresh_live`` so an
    ungated click can never call ``clear_rules`` / ``clear_learned_mappings``.
    """
    if common.live_mode():
        return
    _, storage, rules, *_ = common.services()
    rules.clear_rules()
    storage.clear_learned_mappings()
    storage.clear_rule_notes()
    common.reset_file_session()
    common.set_flash("Cleared rules, learned mappings and Rule Notes. "
                     "Upload a file to begin again.")


def _do_start_fresh_live() -> None:
    """Live Start fresh: delete this client's *session files* only, then unload.

    Does not touch SQLite rules, mappings, training, or models.
    """
    typed = (st.session_state.get("fresh_confirm_name") or "").strip()
    client = (common.current_client_id() or "").strip()
    st.session_state["confirm_action"] = None
    if not client or typed != client:
        common.set_flash("Start fresh cancelled — client name did not match.")
        return
    common.delete_live_session(client)
    common.reset_file_session()
    common.set_flash(
        "Unloaded the file and deleted this client's saved workspace. "
        "Rules, learned memory, and models were kept.")


def _do_unload_file() -> None:
    """Clear session state. Does not delete the saved workspace or SQLite."""
    st.session_state["confirm_action"] = None
    common.reset_file_session()
    common.set_flash("File unloaded. Saved workspace and memory were kept.")


def _do_unload_and_delete_session() -> None:
    """Clear session state and delete this client's session folder only."""
    st.session_state["confirm_action"] = None
    client = (common.current_client_id() or "").strip()
    if client:
        common.delete_live_session(client)
    common.reset_file_session()
    common.set_flash(
        "File unloaded and this client's saved workspace was deleted. "
        "Rules and memory were kept.")


@st.dialog("Please confirm", on_dismiss=_dismiss_confirm)
def _confirm_dialog(work, loaded, config, rules, storage) -> None:
    kind = st.session_state.get("confirm_action")
    if kind == "clear_spreadsheet":
        counts = sh.summary_counts(work, loaded)
        st.warning(
            f"This clears every **Target Account** and **Rule Notes** value in "
            f"this file ({counts['filled']:,} coded, {counts['with_notes']:,} "
            f"with notes). Your original uploaded columns are kept. Confidence "
            f"and review flags are reset. This cannot be undone.")
        st.caption("Your saved keyword rules and learned mappings are not "
                   "affected — only this file's editable values.")
        c1, c2 = st.columns([1.4, 1])
        c1.button("Clear spreadsheet data", type="primary", width="stretch",
                  key="confirm_clear_sheet_yes", on_click=_do_clear_spreadsheet)
        c2.button("Cancel", width="stretch", key="confirm_clear_sheet_no",
                  on_click=_dismiss_confirm)

    elif kind == "start_fresh":
        if common.live_mode():
            client = (common.current_client_id() or "").strip()
            st.warning(
                "**Start fresh** on this hosted instance deletes this "
                "client's *saved workspace files* and unloads the file. "
                "Rules, learned memory, training examples, and models "
                "are kept.")
            st.text_input(
                "Type the client name to confirm",
                key="fresh_confirm_name",
                placeholder=client or "client name")
            typed = (st.session_state.get("fresh_confirm_name") or "").strip()
            c1, c2 = st.columns([1.4, 1])
            c1.button(
                "Delete this client's workspace files", type="primary",
                width="stretch", key="confirm_fresh_live_yes",
                disabled=not client or typed != client,
                on_click=_do_start_fresh_live)
            c2.button("Cancel", width="stretch", key="confirm_fresh_live_no",
                      on_click=_dismiss_confirm)
        else:
            n_rules = len(rules.list_rules())
            n_maps = len(storage.list_learned_mappings())
            n_notes = storage.count_rule_notes()
            st.warning(
                "**Start fresh** permanently deletes everything below and unloads "
                "the current file:")
            st.markdown(
                f"- All keyword rules (**{n_rules}**)\n"
                f"- All learned mappings (**{n_maps}**)\n"
                f"- All saved Rule Notes (**{n_notes}**)\n"
                f"- The currently loaded file")
            st.caption("Trained models and run history are kept. This cannot be "
                       "undone.")
            c1, c2 = st.columns([1.4, 1])
            c1.button("Yes, start fresh", type="primary", width="stretch",
                      key="confirm_fresh_yes", on_click=_do_start_fresh)
            c2.button("Cancel", width="stretch", key="confirm_fresh_no",
                      on_click=_dismiss_confirm)

    elif kind == "unload_file":
        st.info(
            "Unload this file from the browser session. Rules, learned "
            "memory, and models are not touched.")
        delete_saved = st.checkbox(
            "Also delete this client's saved workspace on the volume",
            key="unload_delete_saved",
            value=False)
        c1, c2 = st.columns([1.4, 1])
        if delete_saved:
            c1.button("Unload and delete saved workspace", type="primary",
                      width="stretch", key="confirm_unload_delete",
                      on_click=_do_unload_and_delete_session)
        else:
            c1.button("Unload file", type="primary", width="stretch",
                      key="confirm_unload_keep", on_click=_do_unload_file)
        c2.button("Cancel", width="stretch", key="confirm_unload_no",
                  on_click=_dismiss_confirm)


# --------------------------------------------------------------------------- #
# Post-run performance metrics (full intelligent run)
# --------------------------------------------------------------------------- #
def _clear_run_metrics() -> None:
    st.session_state.pop("show_run_metrics", None)


def _render_run_metrics(work, loaded, counts: dict) -> None:
    """Clear, at-a-glance results after a Full Intelligent Run."""
    if not st.session_state.get("show_run_metrics"):
        return

    cstats = sh.confidence_stats(work, loaded)
    breakdown = sh.engine_breakdown(work)
    total = counts["total"] or 1
    coded_pct = counts["filled"] / total

    with st.expander("Last run — performance summary", expanded=True):
        top = st.columns(5)
        top[0].metric("Transactions", f"{counts['total']:,}")
        top[1].metric("Coded", f"{counts['filled']:,}",
                      delta=f"{coded_pct:.0%} of file", delta_color="off")
        top[2].metric("Auto-coded", f"{counts['auto_filled']:,}",
                      help="Filled at high confidence — ready to use.")
        top[3].metric("Needs review", f"{counts['review_pending']:,}",
                      help="Filled below the auto cutoff or left for you to check.")
        top[4].metric("Still blank", f"{counts['blank']:,}")

        mid = st.columns(5)
        mid[0].metric("Avg confidence",
                      f"{cstats['avg_confidence'] * 100:.0f}%",
                      help="Average confidence across coded rows.")
        mid[1].metric("High confidence (≥85%)", f"{cstats['high_confidence']:,}")
        mid[2].metric("Low confidence (<55%)", f"{cstats['low_confidence']:,}",
                      help="Worth a quick look — filter to 'Review only'.")
        mid[3].metric("Distinct accounts", f"{counts['distinct_accounts']:,}")
        mins = common.minutes_saved(counts)
        mid[4].metric("Est. time saved",
                      f"{mins // 60}h {mins % 60}m" if mins >= 60
                      else f"{mins}m",
                      help="Rough estimate at ~15 seconds saved per auto-coded row.")

        if breakdown:
            import pandas as pd
            bt = pd.DataFrame([{
                "How it was decided": b["engine"],
                "Rows": b["rows"],
                "Avg confidence": f"{b['avg_confidence'] * 100:.0f}%",
            } for b in breakdown])
            st.caption("How the codes were decided:")
            st.dataframe(bt, width="stretch", hide_index=True,
                         height=min(60 + 35 * len(bt), 260))

        b = st.columns([2.2, 1.4, 1.4])
        _render_review_cta(b[0], counts, key="metrics_goto_review")
        b[1].button("Reset run (un-run)", key="reset_run_from_metrics",
                    on_click=_do_reset_run,
                    help="Don't like these results? Clear all auto-filled codes "
                         "(keeps your rules and notes) and run again.")
        b[2].button("Dismiss summary", key="dismiss_run_metrics",
                    on_click=_clear_run_metrics)


# --------------------------------------------------------------------------- #
# Rule-run audit (deterministic, primary path)
# --------------------------------------------------------------------------- #
def _clear_rule_audit() -> None:
    st.session_state.pop("last_rule_audit", None)


def _render_rule_audit(work, loaded) -> None:
    """Results panel for the most recent rule run (strict or hybrid).

    Two complementary views in tabs:
      * **Audit — how it matched**: metrics, confidence breakdown and the
        per-row rule/similarity explanation tables.
      * **Raw matched rows**: a clean spreadsheet of ONLY the rows that were
        filled this run, using the original uploaded columns, with a download.
    """
    audit = st.session_state.get("last_rule_audit")
    if not audit or "keyword_filled" not in audit:
        return

    scoped = audit.get("scoped")
    mode = audit.get("mode")
    is_pure = mode in ("pure", "rules+memory")
    kw_n = int(audit.get("keyword_filled", 0))
    sem_n = int(audit.get("semantic_filled", 0))
    mem_n = int(audit.get("memory_filled", 0))
    total = kw_n + sem_n + mem_n

    if mode == "rules+memory":
        label = ("Last run — rules + memory (selected rows)" if scoped
                 else "Last run — rules + memory")
    elif is_pure:
        label = ("Last run — strict rules (selected rows)" if scoped
                 else "Last run — strict rules")
    else:
        label = ("Last run — rules + similarity (selected rows + whole-file "
                 "spread)" if scoped else "Last run — rules + similarity")

    with st.expander(label, expanded=True):
        tab_audit, tab_raw = st.tabs(
            ["Audit — how it matched", f"Raw matched rows ({total:,})"])
        with tab_audit:
            _render_rule_audit_details(audit, is_pure)
        with tab_raw:
            _render_matched_raw(work, loaded, audit)

        counts = sh.summary_counts(work, loaded)
        b = st.columns([2.2, 1.4, 1.4])
        _render_review_cta(b[0], counts, key="audit_goto_review")
        b[1].button("Reset run (un-run)", key="reset_run_from_audit",
                    on_click=_do_reset_run,
                    help="Don't like these results? Clear all auto-filled codes "
                         "(keeps your rules and notes) and run again.")
        b[2].button("Dismiss results", key="dismiss_rule_audit",
                    on_click=_clear_rule_audit)


def _render_rule_audit_details(audit: dict, is_pure: bool) -> None:
    """The detailed audit view (metrics + confidence + explanation tables)."""
    import pandas as pd

    kw_results = audit.get("keyword_results", [])
    sem_results = audit.get("semantic_results", [])
    kw_n = int(audit.get("keyword_filled", 0))
    sem_n = int(audit.get("semantic_filled", 0))
    pbm = int(audit.get("protected_but_matched", 0))
    total = kw_n + sem_n

    if is_pure:
        mem_n = int(audit.get("memory_filled", 0))
        st.caption(
            "Deterministic: only your keyword rules"
            + (" and remembered exact matches" if audit.get("mode")
               == "rules+memory" else "")
            + " ran, filling blank rows only. No similarity, no other "
            "accounts. Use **Full Intelligent Run** for broader matching.")
        m = st.columns(6)
        m[0].metric("Filled by rules", f"{kw_n:,}")
        m[1].metric("Filled by memory", f"{mem_n:,}",
                    help="Exact matches to codes you approved before.")
        m[2].metric("Protected (already coded)",
                    f"{int(audit.get('protected_existing', 0)):,}",
                    delta=(f"{pbm} matched a rule" if pbm else None),
                    delta_color="off")
        left_blank = int(audit.get("left_blank", 0))
        m[3].metric("Still blank", f"{left_blank:,}")
        m[4].metric("Distinct codes used",
                    f"{len({r.proposed_value for r in kw_results}):,}")
        blanks = kw_n + mem_n + left_blank
        m[5].metric("Success rate on blanks",
                    f"{(kw_n + mem_n) / blanks:.0%}" if blanks else "—",
                    help="Share of blank rows this run filled.")
    else:
        st.caption(
            "Rules match exact phrases, then spread those codes to similar "
            "blank transactions across the file — each with a confidence score.")
        m = st.columns(5)
        m[0].metric("Filled (total)", f"{total:,}")
        m[1].metric("By keyword rule", f"{kw_n:,}",
                    help="Direct, deterministic phrase matches — very high "
                         "confidence.")
        m[2].metric("By semantic match", f"{sem_n:,}",
                    help="Spread from your rule + coded rows to similar "
                         "transactions.")
        m[3].metric("Protected (already coded)",
                    f"{int(audit.get('protected_existing', 0)):,}",
                    delta=(f"{pbm} matched a rule" if pbm else None),
                    delta_color="off")
        m[4].metric("Still blank", f"{int(audit.get('left_blank', 0)):,}")

    st.caption("Confidence breakdown for this run:")
    c = st.columns(3)
    c[0].metric("High (≥85%)", f"{int(audit.get('conf_high', 0)):,}",
                help="Safe to accept quickly.")
    c[1].metric("Medium", f"{int(audit.get('conf_medium', 0)):,}",
                help="A quick glance is worth it.")
    c[2].metric("Low (<55%) — review", f"{int(audit.get('conf_low', 0)):,}",
                help="Focus your review here. Use the Confidence filter set "
                     "to 'Low' to see just these rows.")

    if kw_results:
        table = pd.DataFrame([{
            "Row": r.row_index + 1,
            "New Account": r.proposed_value,
            "Confidence": f"{r.confidence * 100:.0f}%",
            "Matched rule": r.matched_rule_name,
            "Pattern": r.matched_pattern,
            "Why": r.rationale,
        } for r in kw_results])
        st.caption(f"Filled by direct keyword rule ({len(kw_results)}):")
        st.dataframe(table, width="stretch", hide_index=True,
                     height=min(240, 60 + 35 * len(table)))

    if sem_results:
        stab = pd.DataFrame([{
            "Row": int(s["row_index"]) + 1,
            "New Account": s["code"],
            "Confidence": f"{float(s['confidence']) * 100:.0f}%",
            "How decided": sh.friendly_engine(str(s.get("engine", ""))),
            "Why": s.get("rationale", ""),
        } for s in sem_results])
        st.caption(
            f"Filled by semantic match from your rule + coded seeds "
            f"({len(sem_results)}):")
        st.dataframe(stab, width="stretch", hide_index=True,
                     height=min(240, 60 + 35 * len(stab)))

    if not kw_results and not sem_results:
        st.info("No blank rows were filled in this run.")

    # Near misses: rows no rule fired on that almost matched — suggestions to
    # widen a rule, never auto-filled.
    near_misses = audit.get("near_misses") or []
    if near_misses:
        with st.expander(f"Near misses — {len(near_misses)} rule(s) almost "
                         "matched leftover rows", expanded=False):
            st.caption(
                "These blank rows share most of a rule's words but didn't "
                "match. Widen the rule (add the suggested phrase) if they "
                "should be covered — nothing here was filled.")
            for nm in near_misses:
                rows_n = len(nm.get("rows", []))
                extras = ", ".join(nm.get("suggested_tokens") or [])
                st.markdown(
                    f"- Rule **'{nm['rule_keyword']}'** → "
                    f"**{nm['account_code']}** nearly matched "
                    f"**{rows_n}** blank row(s)"
                    + (f" — consider adding: `{extras}`" if extras else ""))

    if pbm:
        matched_protected = audit.get("protected_results", [])
        st.info(
            f"Your rules also matched {pbm} row(s) that are **already coded** — "
            "the matching works; those protected cells fill blank rows on future "
            "files.")
        if matched_protected:
            ptab = pd.DataFrame([{
                "Row": r.row_index + 1,
                "Existing New Account": r.original_value,
                "Matched rule": r.matched_rule_name,
                "Pattern": r.matched_pattern,
            } for r in matched_protected[:200]])
            st.dataframe(ptab, width="stretch", hide_index=True, height=200)


def _render_matched_raw(work, loaded, audit: dict) -> None:
    """Clean spreadsheet of only the rows filled this run, in original columns."""
    filled_rows = audit.get("filled_rows", []) or []
    raw = sh.matched_records_view(work, loaded, filled_rows)
    if raw.empty:
        st.info("No rows were filled in this run, so there's nothing to show "
                "here yet.")
        return

    st.caption(
        f"The **{len(raw):,}** transaction(s) this run filled, shown with your "
        "original spreadsheet columns (no audit columns). This is the real data "
        "for the matched records.")
    st.dataframe(raw, width="stretch", hide_index=True,
                 height=min(520, 80 + 35 * len(raw)))

    from pathlib import Path
    stem = Path(loaded.source_name).stem
    try:
        csv_bytes = export_csv(raw, loaded.new_account_col, add_base_column=False)
        xlsx_bytes = export_dataframe(raw, fmt="xlsx",
                                      sheet_name="Matched rows")
        d1, d2, _ = st.columns([1.6, 1.6, 3])
        d1.download_button(
            "Download matched rows (CSV)", data=csv_bytes,
            file_name=f"matched_{stem}.csv", mime="text/csv",
            width="stretch", key="dl_matched_csv")
        d2.download_button(
            "Download matched rows (Excel)", data=xlsx_bytes,
            file_name=f"matched_{stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet",
            width="stretch", key="dl_matched_xlsx")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not build the matched-rows download: {exc}")


# --------------------------------------------------------------------------- #
# Header + metrics + incentive
# --------------------------------------------------------------------------- #
def _engine_status_text() -> str:
    """Compact, honest 'what's loaded' chip for the spreadsheet header."""
    from src.ml_classifier import setfit_available
    from src.similarity import semantic_backend_available

    config, _, _, mm, _ = services()
    sem_ok, _ = semantic_backend_available()
    sf_ok, _ = setfit_available()
    matching = ("semantic AI" if (sem_ok and config.similarity.use_embeddings)
                else "TF-IDF")
    scoped_mm = common.model_manager_for(common.current_client_id())
    if mm.has_model() or scoped_mm.has_model():
        ml = "SetFit + LogReg" if sf_ok else "LogReg"
    else:
        ml = "not trained — using rules + similarity"
    return f"Matching: {matching} · ML: {ml}"


def _render_header(loaded, counts: dict) -> None:
    """Compact, single-strip header so the grid owns the viewport."""
    client = common.current_client_id() or ""
    title = "Spreadsheet" + (f" — {client}" if client else "")
    head = st.columns([3.2, 2.8])
    head[0].markdown(f"### {title}")
    from src.ingest_profiles import profile_chip
    head[0].caption(f"{loaded.source_name} · {counts['total']:,} transactions · "
                    f"{profile_chip(getattr(loaded, 'source_profile', ''))} · "
                    + _engine_status_text())

    pct = (counts["filled"] / counts["total"]) if counts["total"] else 0.0
    head[1].progress(pct, text=f"{counts['filled']:,}/{counts['total']:,} "
                               f"coded · {pct:.0%}")
    head[1].caption(
        f"Filled {counts['filled']:,}  ·  Auto {counts['auto_filled']:,}  ·  "
        f"Review {counts['review_pending']:,}  ·  Blank {counts['blank']:,}  ·  "
        f"Examples {counts['seeds']:,}  ·  Selected {counts['selected']:,}")

    note = common.value_note(counts)
    if note:
        head[0].caption(note)


# --------------------------------------------------------------------------- #
# Toolbar (sticky) + automation controls
# --------------------------------------------------------------------------- #
def _render_toolbar(work, loaded, counts: dict, config) -> dict:
    st.markdown("<div class='pc-toolbar'></div>", unsafe_allow_html=True)
    actions = {}

    # Recommended path first, left to right: Strict -> + Memory -> +
    # Similarity -> Full Intelligent (broadest, visually secondary).
    r1 = st.columns([2.5, 1.9, 2.1, 2.1])
    actions["run_rules"] = r1[0].button(
        "Run Rules — Strict ★", type="primary", width="stretch",
        key="tb_run_rules",
        help="RECOMMENDED. Deterministic: fills blank rows ONLY where your "
             "keyword rules match, using ONLY those rules' target account "
             "codes. No semantic similarity, no other accounts, no learned "
             "memory. Exactly what your rules say. Never overwrites rows that "
             "already have a value.")
    actions["run_memory"] = r1[1].button(
        "Rules + Memory", width="stretch", key="tb_run_memory",
        help="Strict rules first, then exact matches you approved before "
             "(learned memory). Deterministic — no similarity, no AI. Never "
             "overwrites existing values.")
    actions["run_rules_hybrid"] = r1[2].button(
        "Run Rules + Similarity", width="stretch", key="tb_run_rules_hybrid",
        help="Hybrid: applies your keyword rules, then spreads those codes to "
             "similar blank rows via semantic similarity (with confidence "
             "scores). Broader than strict rules, narrower than the Full "
             "Intelligent Run.")
    actions["run_full"] = r1[3].button(
        "Full Intelligent Run", width="stretch", key="tb_run_full",
        help="Uses similarity and ML on top of rules + memory. Can fill rows "
             "no rule matched — review those in the Review workspace. Never "
             "overwrites existing values.")

    sel_count = counts["selected"]
    r2 = st.columns([2.2, 1.9, 1.8, 1.5, 0.9, 0.9, 1.2, 1.6])
    actions["create_rule"] = r2[0].button(
        "Create Rule from Selection", width="stretch", key="tb_create_rule",
        type="primary" if sel_count else "secondary",
        disabled=sel_count == 0,
        help="Select rows with the Sel checkbox to build a rule. You're also "
             "prompted automatically after typing a Target Account code.")
    actions["approve"] = r2[1].button(
        f"Approve Selected ({sel_count})", width="stretch",
        key="tb_approve", disabled=sel_count == 0,
        help="Confirm the codes on the selected rows and add them to training.")
    actions["select_all"] = r2[2].button(
        "Select all in view", width="stretch", key="tb_select_all",
        help="Select every row that matches the current search and filters "
             "(across all pages).")
    actions["clear_sel"] = r2[3].button("Clear selection", width="stretch",
                                        key="tb_clear_sel")
    actions["undo"] = r2[4].button(
        "Undo", width="stretch", key="tb_undo",
        disabled=not st.session_state.get("undo_stack"))
    actions["redo"] = r2[5].button(
        "Redo", width="stretch", key="tb_redo",
        disabled=not st.session_state.get("redo_stack"))
    actions["export"] = r2[6].button(
        "Export", width="stretch", key="tb_export",
        help="Download the finished file (Excel preserves formatting; CSV is "
             "ready to import).")
    with r2[7].popover("Reset", use_container_width=True):
        _render_reset_controls()

    if common.live_mode():
        r3 = st.columns([2.0, 2.0, 1.6, 3.2])
    else:
        r3 = st.columns([2.2, 2.2, 4])
    with r3[0].popover("Automation settings", use_container_width=True):
        _render_automation_controls(config)
    with r3[1].popover("Add another export", use_container_width=True):
        _render_append_control(work, loaded, config)
    review_col = r3[3] if common.live_mode() else r3[2]
    if common.live_mode():
        actions["save_workspace"] = r3[2].button(
            "Save workspace", width="stretch", key="tb_save_workspace",
            help="Write the current coded workbook to this server's "
                 "persistent volume for this client.")
    _render_review_cta(review_col, counts, key="tb_goto_review")
    return actions


def _open_review(*, include_blanks: bool = False) -> None:
    if include_blanks:
        st.session_state["review_include_no_match"] = True
    st.session_state["view"] = "review"
    st.rerun()


def _render_review_cta(slot, counts: dict, *, key: str) -> None:
    """Loud button after a run: leftovers first, then unmatched blanks."""
    pending = int(counts.get("review_pending") or 0)
    blanks = int(counts.get("blank") or 0)
    if pending:
        if slot.button(
                f"Review {pending:,} rows that need you →",
                type="primary", width="stretch", key=key,
                help="Open Review: Keep / Fix / Not this, least-sure first."):
            _open_review()
    elif blanks:
        if slot.button(
                f"Review {blanks:,} blank rows →",
                type="primary", width="stretch", key=key,
                help="Open Review and show rows with no suggestion yet so "
                     "you can type an account from this screen."):
            _open_review(include_blanks=True)


def _render_append_control(work, loaded, config) -> None:
    """Throughout-the-month ingest: append next week's export onto this book.

    File-drop only — the accountant uploads the export they already downloaded.
    Duplicates (same transaction signature + date + amount) are skipped and
    existing codes are never overwritten.
    """
    _config, storage, *_ = services()
    st.caption("Append another AppFolio / QuickBooks export onto this file. "
               "Duplicates are skipped; existing codes are never touched.")
    up = st.file_uploader("Choose a file (.xlsx or .csv)", key="append_uploader",
                          type=["xlsx", "xlsm", "xls", "csv"],
                          label_visibility="collapsed")
    if st.button("Append rows", type="primary", width="stretch",
                 key="append_apply", disabled=up is None):
        from pathlib import Path
        try:
            with st.spinner("Appending new rows…"):
                common.push_undo()
                new_work, audit = sh.append_export(
                    work, loaded, up.getvalue(), up.name, config, storage)
                st.session_state["work_df"] = new_work
            _bump()
            common.persist_workspace()
            common.set_flash(
                f"Added {audit['added']:,} new row(s) · skipped "
                f"{audit['skipped']:,} duplicate(s) · protected "
                f"{audit['protected']:,} coded.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not append that file: {exc}")


def _render_reset_controls() -> None:
    """Safe reset actions; each opens a confirmation dialog before doing anything."""
    st.markdown("**Re-run**")
    st.caption("Made a mistake or forgot a rule? Un-run the automation and "
               "start the matching over.")
    st.button(
        "Reset run (un-run automation)", width="stretch", key="reset_run_btn",
        on_click=_do_reset_run,
        help="Clears everything the automation filled (keyword rules, similarity "
             "and the AI model) but KEEPS your saved rules, Rule Notes and any "
             "codes that came from the upload or you typed yourself. Then adjust "
             "and run again. You can also Undo this.")
    st.divider()
    st.markdown("**Reset & clean up**")
    st.caption("Destructive actions — you'll be asked to confirm first.")
    if st.button("Clear spreadsheet data", width="stretch",
                 key="reset_clear_sheet",
                 help="Clear every Target Account and Rule Notes value in this "
                      "file. Your original columns are kept."):
        st.session_state["confirm_action"] = "clear_spreadsheet"
        st.session_state["export_open"] = False
        st.session_state["panel"] = None
        st.rerun()
    if common.live_mode():
        if st.button("Unload file", width="stretch", key="reset_unload_file",
                     help="Clear the spreadsheet from this browser session. "
                          "Rules and learned memory stay. Optionally delete "
                          "this client's saved workspace folder."):
            st.session_state["confirm_action"] = "unload_file"
            st.session_state["export_open"] = False
            st.session_state["panel"] = None
            st.rerun()
        if st.button("Start fresh (this client)…", width="stretch",
                     key="reset_start_fresh",
                     help="Requires typing the client name. Deletes this "
                          "client's saved workspace files only — not SQLite "
                          "rules or memory."):
            st.session_state["confirm_action"] = "start_fresh"
            st.session_state["export_open"] = False
            st.session_state["panel"] = None
            st.rerun()
    else:
        if st.button("Start fresh", width="stretch", key="reset_start_fresh",
                     help="Delete all rules, learned mappings and saved Rule Notes, "
                          "and unload the current file."):
            st.session_state["confirm_action"] = "start_fresh"
            st.session_state["export_open"] = False
            st.session_state["panel"] = None
            st.rerun()


def _render_automation_controls(config) -> None:
    """Intelligent-mode + semantic toggle, available right on the spreadsheet."""
    st.markdown("**Run options**")
    labels = list(common.ML_MODE_LABELS.values())
    keys = list(common.ML_MODE_LABELS.keys())
    cur = st.session_state.get("ml_mode", config.ml.mode)
    idx = keys.index(cur) if cur in keys else 0
    choice = st.selectbox(
        "Intelligent mode", labels, index=idx, key="tb_ml_mode_label",
        help="How the Full Intelligent Run decides codes.")
    st.session_state["ml_mode"] = keys[labels.index(choice)]

    available, _ = semantic_backend_available()
    config.similarity.use_embeddings = st.toggle(
        "Use Semantic AI matching",
        value=config.similarity.use_embeddings and available,
        disabled=not available, key="tb_semantic",
        help=("Matches on meaning rather than spelling." if available else
              "Optional engine not installed — TF-IDF is used."))
    if not available:
        st.caption("Semantic AI not installed — add it from the sidebar to "
                   "enable meaning-based matching.")


def _automation_summary(config) -> str:
    mode_label = common.ML_MODE_LABELS.get(
        st.session_state.get("ml_mode", config.ml.mode), "")
    mode_short = mode_label.split("—")[0].strip() or mode_label
    matching = "Semantic AI" if config.similarity.use_embeddings else "TF-IDF"
    return f"Mode: {mode_short} · Matching: {matching}"


# --------------------------------------------------------------------------- #
# Filter bar — global search across ALL records + quick views
# --------------------------------------------------------------------------- #
def _current_mask(work, loaded, config):
    """Build the active view mask from the search/filter widgets."""
    scope = st.session_state.get("search_scope", "All columns")
    cols = None if scope == "All columns" else [scope]
    engines = st.session_state.get("engine_filter") or None
    conf_level = st.session_state.get("conf_filter", "All")
    conf_range = sh.confidence_level_range(conf_level, config)
    mask = sh.build_view_mask(
        work, loaded,
        mode=st.session_state.get("filter_mode") or "All",
        query=st.session_state.get("search_query", ""),
        search_columns=cols,
        engines=engines,
        conf_range=conf_range,
        cutoff=float(config.confidence.auto_apply_cutoff))
    # A confidence filter is about *coded* rows — never surface blank rows (whose
    # stored confidence is 0) when the reviewer picks a confidence band.
    if conf_range is not None:
        na = work[loaded.new_account_col].astype(str).str.strip()
        mask &= (na != "")
    return mask


def _clear_filters() -> None:
    st.session_state["filter_mode"] = "All"
    st.session_state["conf_filter"] = "All"
    st.session_state["search_query"] = ""
    st.session_state["search_scope"] = "All columns"
    st.session_state["engine_filter"] = []
    st.session_state["page"] = 0


def _render_filter_bar(work, loaded, config):
    # Quick-view toggles: Show All / Review only / High confidence / Blanks / Filled.
    st.segmented_control(
        "Quick view", FILTER_MODES, key="filter_mode",
        selection_mode="single", label_visibility="collapsed")

    f = st.columns([3.1, 1.7, 2.1, 1.6, 1.1, 1.1])
    f[0].text_input(
        "Search", key="search_query", label_visibility="collapsed",
        placeholder="Search all records — vendor, memo, code, notes…")
    scope_opts = ["All columns"] + sh.searchable_columns(work, loaded)
    if st.session_state.get("search_scope") not in scope_opts:
        st.session_state["search_scope"] = "All columns"
    f[1].selectbox("Search in", scope_opts, key="search_scope",
                   label_visibility="collapsed")

    eng_opts = sh.available_engines(work)
    # Drop any stale selections no longer present so the widget never errors.
    st.session_state["engine_filter"] = [
        e for e in st.session_state.get("engine_filter", []) if e in eng_opts]
    f[2].multiselect("Decided by", eng_opts, key="engine_filter",
                     label_visibility="collapsed",
                     placeholder="Filter by how decided")
    f[3].selectbox("Confidence", list(sh.CONF_LEVELS), key="conf_filter",
                   label_visibility="collapsed",
                   help="Show only coded rows in a confidence band. Pick 'Low' "
                        "to focus review on the least certain fills.")
    f[4].selectbox("Rows/page", PAGE_SIZES, key="page_size",
                   label_visibility="collapsed")
    # Reset via on_click so widget-keyed state is changed *before* the widgets
    # are re-instantiated on the next run (Streamlit forbids mid-run changes).
    f[5].button("Clear filters", width="stretch", key="tb_clear_filters",
                on_click=_clear_filters)

    mask = _current_mask(work, loaded, config)

    # Reset to the first page whenever the view changes (keeps paging sane).
    sig = (st.session_state.get("filter_mode"),
           st.session_state.get("conf_filter"),
           st.session_state.get("search_query"),
           st.session_state.get("search_scope"),
           tuple(st.session_state.get("engine_filter") or ()),
           st.session_state.get("page_size"))
    if st.session_state.get("_view_sig") != sig:
        st.session_state["_view_sig"] = sig
        st.session_state["page"] = 0
    return mask


def _render_page_controls(page: int, n_pages: int, total_filtered: int,
                          total_all: int) -> None:
    c = st.columns([1, 1, 5, 2])
    if c[0].button("Previous", disabled=page <= 0, width="stretch", key="pg_prev"):
        st.session_state["page"] = max(0, page - 1)
        st.rerun()
    if c[1].button("Next", disabled=page >= n_pages - 1, width="stretch",
                   key="pg_next"):
        st.session_state["page"] = min(n_pages - 1, page + 1)
        st.rerun()
    if total_filtered != total_all:
        match_txt = f"**{total_filtered:,}** of {total_all:,} records match"
    else:
        match_txt = f"All **{total_all:,}** records"
    c[2].caption(f"{match_txt} · page {page + 1} / {n_pages}")


# --------------------------------------------------------------------------- #
# Editor
# --------------------------------------------------------------------------- #
def _render_editor(page_df, work, loaded):
    na_col = loaded.new_account_col
    cols = sh.editor_columns(loaded, work)
    view = page_df[cols].copy()

    # Show confidence as a true 0–100% value. The stored column is 0.0–1.0;
    # a plain "%.0f%%" printf on that renders 0.99 as "1%", so we scale the
    # *display* copy to 0–100 and format as a whole-number percentage.
    if sh.CONF_COL in view.columns:
        view[sh.CONF_COL] = (view[sh.CONF_COL].astype(float) * 100
                             ).round().clip(0, 100).astype(int)

    suggestions = _code_suggestions(work, na_col)
    column_config = {
        sh.SELECT_COL: st.column_config.CheckboxColumn(
            "Sel", help="Select rows for bulk actions and rule creation.",
            width="small"),
        na_col: st.column_config.TextColumn(
            sh.TARGET_ACCOUNT_LABEL,
            help="The account / category code. After you enter a code, you'll be "
                 "offered a one-click rule. Suggestions: "
                 + (", ".join(suggestions[:8]) if suggestions else "—")),
        RULE_NOTES_COL: st.column_config.TextColumn(
            "Rule Notes",
            help="Your hints about this transaction. These improve matching and "
                 "are remembered for future files."),
        sh.CONF_COL: st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=100, format="%d%%",
            help="How sure the tool is about this code, from 0% to 100%."),
        sh.ENGINE_COL: st.column_config.TextColumn("How decided"),
    }
    disabled = [c for c in view.columns
                if c not in (sh.SELECT_COL, na_col, RULE_NOTES_COL)]
    version = st.session_state.get("data_version", 0)
    page = st.session_state.get("page", 0)
    # The key must change whenever the displayed slice changes (page, filters,
    # search) so stale edit deltas are never replayed onto different rows.
    sig = st.session_state.get("_view_sig")
    key = f"grid_{version}_{page}_{abs(hash(sig)) & 0xffffffff}"
    return st.data_editor(
        view, width="stretch", height=EDITOR_HEIGHT, column_config=column_config,
        disabled=disabled, hide_index=True, key=key)


def _code_suggestions(work, na_col) -> list:
    """Most common existing codes — used as free-text hints."""
    vals = (work[na_col].astype(str).str.strip())
    vals = vals[vals != ""]
    return list(vals.value_counts().index[:12])


def _commit_with_undo(work, edited, loaded, storage, config, rules) -> None:
    pre = sh.snapshot(work)
    client_id = common.current_client_id()
    counts = sh.commit_editor_changes(work, edited, loaded, storage, config,
                                      client_id=client_id)
    if counts.get("targets") or counts.get("notes"):
        stack = st.session_state["undo_stack"]
        stack.append(pre)
        if len(stack) > 30:
            del stack[0]
        st.session_state["redo_stack"] = []
        common.persist_workspace()
    rc.queue_target_edit_prompts(counts.get("target_edits", []), rules)


# --------------------------------------------------------------------------- #
# Action handling
# --------------------------------------------------------------------------- #
def _handle_actions(actions, work, loaded, config, rules, storage, mm, logger) -> None:
    # Undo / redo.
    if actions.get("undo"):
        if common.undo():
            _bump()
            common.set_flash("Undid the last change.")
        st.rerun()
    if actions.get("redo"):
        if common.redo():
            _bump()
            common.set_flash("Redid the change.")
        st.rerun()

    # Selection helpers — "select all in view" respects search + filters and
    # spans every page of the current view, not just the one on screen.
    if actions.get("select_all"):
        mask = _current_mask(work, loaded, config)
        sh.set_selection(work, [int(i) for i in work.index[mask]], True)
        _bump()
        n = int(mask.sum())
        common.set_flash(f"Selected {n:,} row(s) matching the current view.")
        st.rerun()
    if actions.get("clear_sel"):
        sh.clear_selection(work)
        _bump()
        st.rerun()

    # Full intelligent run.
    if actions.get("run_full"):
        common.push_undo()
        progress = st.progress(0.0, text="Starting…")

        def cb(frac, msg):
            progress.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            with st.spinner("Running the full automation…"):
                result = sh.run_full(work, loaded, config, common.make_engine(),
                                     progress_cb=cb)
            st.session_state["last_result"] = result
            _record_run(result, storage)
            progress.progress(1.0, text="Done.")
            counts = sh.summary_counts(work, loaded)
            msg = (f"Filled {counts['filled']:,} transactions "
                   f"({counts['auto_filled']:,} automatically). "
                   f"{counts['review_pending']:,} rows that need you.")
            note = common.value_note(counts)
            if note:
                msg += f"  {note}."
            common.set_flash(msg)
            st.session_state["show_run_metrics"] = True
            _bump()
            common.persist_workspace()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"The run did not complete: {exc}")
            logger.error("run_failed", error=str(exc))

    # Strict rules run — deterministic, blank-only, fully audited. Applies ONLY
    # the enabled keyword rules and fills ONLY with their target account codes.
    # No semantic similarity, no learned memory, no cross-account influence — the
    # outcome is exactly what the rules say. Existing values are never overwritten.
    # A selection scopes the run to those rows; otherwise every blank row.
    if actions.get("run_rules"):
        if not rules.list_rules(enabled_only=True):
            common.set_flash(
                "No keyword rules yet. Create one from a selection, or build "
                "rules from your pre-filled rows in the Rules panel.")
            st.rerun()
        common.push_undo()
        sel = sh.selected_indices(work)
        indices = sel if sel else None
        client_id = common.current_client_id()
        n, results = sh.run_selected_rules_audited(
            work, rules, loaded, config, indices=indices, client_id=client_id)
        audit = sh.pure_rule_audit(results, config)
        audit["scoped"] = bool(sel)
        audit["applied"] = n
        # Near misses: rows no rule fired on that almost matched — suggested
        # rule edits, never auto-filled.
        try:
            from src.rules_manager import near_miss_suggestions
            audit["near_misses"] = near_miss_suggestions(
                results, work, rules.list_rules(enabled_only=True),
                list(loaded.text_columns))
        except Exception:  # noqa: BLE001 - suggestions are best-effort
            audit["near_misses"] = []
        st.session_state["last_rule_audit"] = audit
        _bump()
        msg = (f"Rules filled {n:,} blank row(s) using only your rules' codes. "
               f"{audit['protected_existing']:,} already coded (protected); "
               f"{audit['left_blank']:,} still blank.")
        pbm = audit.get("protected_but_matched", 0)
        if n == 0 and pbm:
            msg += (f"  Note: your rules matched {pbm} already-coded row(s) — "
                    "they work, but those cells already have a value.")
        if audit["left_blank"]:
            msg += " Open Review to finish the leftovers."
        common.set_flash(msg)
        # Show the impact immediately: jump the grid to the rule-filled rows.
        if n:
            st.session_state["_pending_filter_mode"] = "Filled by rules"
        common.persist_workspace()
        st.rerun()

    # Rules + Memory — strict rules, then exact matches approved before.
    # Deterministic (no similarity, no AI), blank-only, fully audited.
    if actions.get("run_memory"):
        client_id = common.current_client_id()
        if not rules.list_rules(enabled_only=True) \
                and not storage.get_learned_lookup(client_id=client_id):
            common.set_flash(
                "No rules or remembered codes yet. Create a rule or approve a "
                "few rows first — memory builds from your approvals.")
            st.rerun()
        common.push_undo()
        sel = sh.selected_indices(work)
        indices = sel if sel else None
        n, audit = sh.run_rules_plus_memory(
            work, rules, loaded, config, indices=indices, client_id=client_id)
        audit["applied"] = n
        st.session_state["last_rule_audit"] = audit
        _bump()
        common.set_flash(
            f"Filled {n:,} blank row(s): {audit['keyword_filled']:,} by rules, "
            f"{audit.get('memory_filled', 0):,} from remembered codes. "
            f"{audit['protected_existing']:,} already coded (protected); "
            f"{audit['left_blank']:,} still blank.")
        if n:
            st.session_state["_pending_filter_mode"] = "Filled only"
        common.persist_workspace()
        st.rerun()

    # Hybrid rules run — explicit, separate action. Applies keyword rules, then
    # spreads those codes to similar blank rows via semantic similarity. Broader
    # than strict rules; never overwrites existing values.
    if actions.get("run_rules_hybrid"):
        if not rules.list_rules(enabled_only=True):
            common.set_flash(
                "No keyword rules yet. Create one first, then use Run Rules + "
                "Similarity to spread the codes to look-alike transactions.")
            st.rerun()
        common.push_undo()
        sel = sh.selected_indices(work)
        indices = sel if sel else None
        client_id = common.current_client_id()
        progress = st.progress(0.0, text="Applying rules…")

        def hybrid_cb(frac, msg):
            progress.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            n, audit = sh.run_rules_hybrid(
                work, rules, loaded, config, common.make_hybrid_engine(),
                indices=indices, client_id=client_id, progress_cb=hybrid_cb)
            progress.progress(1.0, text="Done.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"The rule run did not complete: {exc}")
            logger.error("rule_run_failed", error=str(exc))
            return
        audit["scoped"] = bool(sel)
        audit["applied"] = n
        st.session_state["last_rule_audit"] = audit
        _bump()
        kw, sem = audit["keyword_filled"], audit["semantic_filled"]
        msg = (f"Filled {n:,} row(s): {kw:,} by keyword rules, {sem:,} by "
               f"semantic similarity. Confidence — {audit['conf_high']:,} high, "
               f"{audit['conf_medium']:,} medium, {audit['conf_low']:,} low "
               f"(review these). {audit['protected_existing']:,} already coded "
               "(protected).")
        pbm = audit.get("protected_but_matched", 0)
        if n == 0 and pbm:
            msg += (f"  Note: your rules matched {pbm} already-coded row(s) — "
                    "they work, but those cells already have a value.")
        common.set_flash(msg)
        common.persist_workspace()
        st.rerun()

    # Open the rule-creation dialog, pre-filled from the checkbox selection.
    if actions.get("create_rule"):
        sel = sh.selected_indices(work)
        if not sel:
            common.set_flash(
                "Select one or more rows with the Sel checkbox to create a rule.")
            st.rerun()
        prefill = sh.rule_creation_prefill(work, sel, loaded,
                                           rules_manager=rules, config=config)
        rc.open_rule_panel(prefill)
        st.rerun()

    # Open the export modal (closing any management panel first — one modal/run).
    if actions.get("export"):
        st.session_state["export_open"] = True
        st.session_state["panel"] = None
        st.rerun()

    # Approve selected.
    if actions.get("approve"):
        common.push_undo()
        client_id = common.current_client_id()
        out = sh.approve_rows(work, loaded, storage, config,
                              client_id=client_id)
        _bump()
        st.toast(f"Approved {out['applied']} · learned {out['learned']} · "
                 "queued for next model train")
        common.set_flash(
            f"Approved {out['applied']} transaction(s); learned "
            f"{out['learned']} mapping(s).")
        common.persist_workspace()
        st.rerun()

    if actions.get("save_workspace"):
        name = common.remember_client_name() or (common.current_client_id() or "")
        if not name:
            common.set_flash(
                "Set a client name on the dashboard before saving.")
        elif st.session_state.get("work_df") is None \
                or st.session_state.get("original_bytes") is None:
            common.set_flash("Load a file before saving the workspace.")
        elif common.persist_workspace():
            common.set_flash("Workspace saved to this server's persistent volume.")
        else:
            common.set_flash("Could not save the workspace to the volume.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Run history record
# --------------------------------------------------------------------------- #
def _record_run(result, storage) -> None:
    try:
        client = common.current_client_id() or ""
        if client:
            result.summary.file_name = f"{client}: {result.summary.file_name}"
        storage.add_run(result.summary)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def _close_export() -> None:
    st.session_state["export_open"] = False


@st.dialog("Export", width="large", on_dismiss=_close_export)
def _export_dialog(work, loaded, counts: dict) -> None:
    st.caption("Internal helper columns are stripped automatically; each Excel "
               "file includes a 'Code Down Summary' tab.")
    base_msg = (f"{counts['filled']:,} of {counts['total']:,} transactions are "
                "coded — ready for journal entries and clean reporting.")
    note = common.value_note(counts)
    common.incentive(base_msg + (f" {note}." if note else ""))

    config, *_ = services()
    na_col = loaded.new_account_col
    from pathlib import Path
    stem = Path(loaded.source_name).stem
    result = st.session_state.get("last_result")
    backend = result.backend if result else ""
    mode = result.mode if result else ""
    summary_df = common.export_summary_df(
        counts, loaded, backend, mode, work=work,
        audit=st.session_state.get("last_rule_audit"))

    # Filenames carry the client + date so exports self-identify.
    import re as _re
    from datetime import datetime as _dt
    client_slug = _re.sub(r"[^A-Za-z0-9]+", "_",
                          common.current_client_id() or "").strip("_")
    date_slug = _dt.now().strftime("%Y%m%d")
    prefix = "_".join(p for p in (client_slug, stem, date_slug) if p)

    ext = st.session_state.get("original_ext")
    orig = st.session_state.get("original_bytes")
    try:
        with st.spinner("Preparing your files…"):
            if orig and ext in (".xlsx", ".xlsm", ".xls"):
                xlsx = export_preserving_original(
                    orig, work, na_col, config,
                    extra_header_candidates=[loaded.original_new_account_header],
                    summary=summary_df)
                label = "Excel (keeps your formatting)"
            else:
                xlsx = export_dataframe(work, fmt="xlsx", summary=summary_df)
                label = "Excel (tidy)"
            qb_df = work.drop(columns=[RULE_NOTES_COL], errors="ignore")
            qb_bytes = export_csv(qb_df, na_col)
            clean_bytes = export_dataframe(work, fmt="xlsx", summary=summary_df)

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            label, data=xlsx, file_name=f"filled_{prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet", width="stretch")
        c2.download_button(
            "CSV (ready to import)", data=qb_bytes,
            file_name=f"filled_{prefix}_export.csv", mime="text/csv",
            width="stretch")
        c3.download_button(
            "Excel (all columns + notes)", data=clean_bytes,
            file_name=f"filled_{prefix}_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet", width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build the download: {exc}")

    if st.button("Close", key="export_close", width="stretch"):
        _close_export()
        st.rerun()
