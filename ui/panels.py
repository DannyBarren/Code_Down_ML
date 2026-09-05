"""Secondary management views: Rules, Models and History.

Reached from the sidebar; each offers a quick way back to the spreadsheet.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.schemas import KeywordRule
from src.data_loader import SIM_TEXT_COL
from src.ml_classifier import setfit_available
from src.similarity import semantic_backend_available
from ui import common
from ui.common import ML_MODE_LABELS, services

# Supported keyword-matching modes (order = friendliness / most common first).
MATCH_TYPES = ["contains", "exact", "starts_with", "ends_with", "fuzzy", "regex"]


def _close_panel() -> None:
    st.session_state["panel"] = None


def open_panel_dialog(panel: str) -> None:
    """Open the requested management view as a modal over the spreadsheet.

    Modals never replace the grid underneath — closing one (X or Close button)
    returns straight to the spreadsheet.
    """
    if panel == "rules":
        _rules_dialog()
    elif panel == "models":
        _models_dialog()
    elif panel == "history":
        _history_dialog()


@st.dialog("Rules", width="large", on_dismiss=_close_panel)
def _rules_dialog() -> None:
    page_rules()
    if st.button("Close", key="rules_close", width="stretch"):
        _close_panel()
        st.rerun()


@st.dialog("Models", width="large", on_dismiss=_close_panel)
def _models_dialog() -> None:
    page_models()
    if st.button("Close", key="models_close", width="stretch"):
        _close_panel()
        st.rerun()


@st.dialog("History", width="large", on_dismiss=_close_panel)
def _history_dialog() -> None:
    page_history()
    if st.button("Close", key="history_close", width="stretch"):
        _close_panel()
        st.rerun()


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _available_fields() -> list:
    config, *_ = services()
    loaded = st.session_state.get("loaded")
    if loaded is not None:
        return [c for c in loaded.df.columns if c != SIM_TEXT_COL]
    return config.columns.similarity_columns


def _render_ingest_prefilled(rules_manager) -> None:
    """Human-reviewed rule creation from already-coded ``New Account`` rows.

    Instead of silently inventing keywords, this shows the accountant the actual
    coded rows (grouped by account code, with sample transaction text) and lets
    them type exactly the keywords/phrases each rule should use. Nothing is
    created until the user enters phrases and clicks the create button — full
    human oversight over every rule.
    """
    from src.data_loader import summarize_upload

    config, *_ = services()
    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    if work is None or loaded is None:
        return
    stats = summarize_upload(work, loaded.new_account_col)
    if stats["prefilled"] <= 0:
        return

    from src import spreadsheet_helpers as sh
    cands = rules_manager.prefilled_ingest_candidates(
        work, text_cols=sh.mining_columns(work, loaded, config))
    if not cands:
        return

    with st.expander(
            f"Build rules from your already-coded rows ({stats['prefilled']:,} "
            f"coded · {len(cands)} account code(s))", expanded=False):
        st.caption(
            "These rows already have a New Account code. Review them below and "
            "type the keywords or phrases you want each rule to match — you're "
            "in full control. Separate multiple phrases with commas; the rule "
            "matches if ANY appears anywhere in the transaction. Leave a box "
            "empty to skip that code. Nothing is created until you click "
            "**Create reviewed rules**.")

        with st.form("ingest_reviewed_rules", clear_on_submit=False):
            entries: list = []
            for cand in cands:
                code = cand["code"]
                st.markdown(f"**{code}** — {cand['count']:,} coded row(s)"
                            + ("  ·  _already has a rule_" if cand["has_rule"]
                               else ""))
                if cand["samples"]:
                    st.caption("Examples: "
                               + "  •  ".join(cand["samples"][:3]))
                phrases = st.text_input(
                    f"Keywords / phrases for {code} (comma-separated)",
                    key=f"ingest_kw_{code}",
                    placeholder="Rent Signage, signage, yard sign, For Rent Sign",
                    label_visibility="collapsed")
                entries.append((code, phrases))
                st.divider()

            submitted = st.form_submit_button(
                "Create reviewed rules", type="primary")
            if submitted:
                client_id = st.session_state.get("client_name") or None
                created = 0
                for code, phrases in entries:
                    phrases = (phrases or "").strip().strip(",").strip()
                    if not phrases:
                        continue
                    rules_manager.create_rule(
                        phrases, code, match_type="contains",
                        priority=10, client_id=client_id,
                        notes="Reviewed from pre-filled New Account rows.")
                    created += 1
                common.set_flash(
                    f"Created {created} reviewed rule(s) from your coded rows."
                    if created else
                    "No rules created — enter at least one keyword/phrase to "
                    "build a rule.")
                st.rerun()


def _render_suggested_rules(rules_manager) -> None:
    """Mine the loaded sheet's already-coded rows for rules to create in bulk."""
    from src import spreadsheet_helpers as sh

    config, *_ = services()
    work = st.session_state.get("work_df")
    loaded = st.session_state.get("loaded")
    if work is None or loaded is None:
        return
    cands = sh.candidate_rules(work, loaded, rules_manager, config=config)
    if not cands:
        return

    with st.expander(f"Suggested rules from your coded rows ({len(cands)})",
                     expanded=False):
        st.caption(
            "These come from transactions you've already coded. Tick the ones to "
            "keep and create them all at once — each becomes permanent for every "
            "future file.")
        rows = [{
            "create": True,
            "keyword": c["keyword"],
            "account_code": c["account_code"],
            "what it means": c["description"],
            "rows it would fill": c["matches"],
        } for c in cands]
        edited = st.data_editor(
            pd.DataFrame(rows), width="stretch", hide_index=True,
            key="suggested_rules_editor", num_rows="fixed",
            disabled=["keyword", "account_code", "what it means",
                      "rows it would fill"],
            column_config={
                "create": st.column_config.CheckboxColumn("create?"),
                "rows it would fill": st.column_config.NumberColumn(width="small"),
            })
        chosen = [{"keyword": r["keyword"], "account_code": r["account_code"]}
                  for _, r in edited.iterrows() if bool(r["create"])]
        if st.button(f"Create {len(chosen)} selected rule(s)", type="primary",
                     width="stretch", key="create_suggested_rules",
                     disabled=not chosen):
            n = sh.create_rules_from_candidates(
                rules_manager, chosen, client_id=common.current_client_id())
            common.set_flash(
                f"Created {n} rule(s) from your coded rows. Run them from the "
                "spreadsheet toolbar.")
            st.rerun()


def _render_rules_confirm(rules_manager, total_rules: int) -> None:
    """Inline confirm strip for destructive rule actions (modal-safe).

    A nested ``st.dialog`` cannot be opened from inside the Rules modal, so we
    use a clearly styled inline confirmation that the user must accept.
    """
    pending = st.session_state.get("_rules_confirm")
    if not pending:
        return
    kind, ids = pending
    if kind == "delete_selected":
        n = len(ids)
        if n == 0:
            st.session_state.pop("_rules_confirm", None)
            return
        msg = (f"This will permanently delete <strong>{n} selected "
               f"rule{'s' if n != 1 else ''}</strong>. This cannot be undone.")
    else:  # clear_all
        msg = (f"This will permanently delete <strong>all {total_rules} "
               f"rule{'s' if total_rules != 1 else ''}</strong>. This cannot be "
               "undone.")

    st.markdown(f"<div class='pc-rule-prompt'>{msg} Continue?</div>",
                unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.4, 1, 4])
    if c1.button("Yes, delete", type="primary", key="rules_confirm_yes"):
        if kind == "delete_selected":
            removed = rules_manager.delete_rules(ids)
        else:
            removed = rules_manager.clear_rules()
        st.session_state.pop("_rules_confirm", None)
        common.set_flash(f"Deleted {removed} rule(s).")
        st.rerun()
    if c2.button("Cancel", key="rules_confirm_no"):
        st.session_state.pop("_rules_confirm", None)
        st.rerun()


def page_rules() -> None:
    config, storage, rules_manager, mm, _ = services()
    from utils.account_codes import describe, normalize_code

    st.write(
        "Rules are shortcuts you control: *whenever you see this word, use this "
        "code*. They run before everything else and are saved for every future "
        "file. Example: **Cloud Hosting → 6100**.")
    common.incentive(
        "Well-tuned rules are the fastest way to teach the tool your "
        "recurring vendors — they pay off on every future file.")

    _render_ingest_prefilled(rules_manager)
    _render_suggested_rules(rules_manager)

    with st.expander("Add a new rule", expanded=False):
        with st.form("add_rule", clear_on_submit=True):
            r1, r2 = st.columns(2)
            keyword = r1.text_input(
                "Keywords / phrases (comma-separated)",
                placeholder="Rent Signage, signage, yard sign, For Rent Sign",
                help="Enter multiple keywords or phrases separated by commas. "
                     "The rule matches if ANY of them appears anywhere in the "
                     "transaction (great for spelling variants of one vendor).")
            code = r2.text_input("Use this Target Account code", placeholder="6100")
            st.caption(
                "By default the rule searches the **whole transaction row**, so "
                "you don't need to pick a column. Separate multiple phrases with "
                "commas.")
            r3, r4 = st.columns(2)
            match_type = r3.selectbox(
                "How to match", MATCH_TYPES,
                help="contains · exact · starts_with · ends_with · fuzzy "
                     "(typo-tolerant) · regex")
            case_sensitive = r4.checkbox("Match case exactly", value=False)

            with st.expander("Advanced: restrict to specific columns only",
                             expanded=False):
                st.caption(
                    "Optional. By default the whole row is searched. Pick one or "
                    "more columns here only if you want to limit this rule to "
                    "them (e.g. match a vendor in 'Name' but not in a memo).")
                fields = st.multiselect(
                    "Search only in these columns",
                    options=_available_fields(),
                    label_visibility="collapsed")
            notes = st.text_input("Notes (optional)")
            submitted = st.form_submit_button("Add rule", type="primary")
            if submitted:
                if not keyword.strip() or not code.strip():
                    st.error("Please fill in both the keyword(s) and the code.")
                else:
                    rules_manager.add_rule(
                        keyword.strip(), code.strip(), match_type=match_type,
                        case_sensitive=case_sensitive, fields=fields,
                        notes=notes, client_id=common.current_client_id())
                    common.set_flash(
                        f"Added rule: '{keyword}' → {normalize_code(code)}.")
                    st.rerun()

    rules = rules_manager.list_rules()
    st.subheader(f"Your rules ({len(rules)})")
    if not rules:
        st.info("No rules yet. Add one above to get started.")
        return

    # Select-all / deselect-all set a flag the editor honours on its next render.
    sa1, sa2, _ = st.columns([1, 1, 4])
    if sa1.button("Select all rules", width="stretch", key="rules_select_all"):
        st.session_state["_rules_select_all"] = True
        st.rerun()
    if sa2.button("Deselect all", width="stretch", key="rules_deselect_all"):
        st.session_state["_rules_select_all"] = False
        st.rerun()

    preset = st.session_state.pop("_rules_select_all", None)
    df = pd.DataFrame([{
        "select": bool(preset) if preset is not None else False,
        "id": r.id, "keyword": r.keyword, "account_code": r.account_code,
        "what it means": describe(r.account_code) or "",
        "match_type": r.match_type, "case_sensitive": r.case_sensitive,
        "fields": ", ".join(r.fields), "enabled": r.enabled, "notes": r.notes,
    } for r in rules])
    # Force the editor to remount when select-all toggles so the preset sticks.
    editor_key = f"rules_editor_{int(bool(preset))}_{len(rules)}" \
        if preset is not None else "rules_editor"
    edited = st.data_editor(
        df, width="stretch", disabled=["id", "fields", "what it means"],
        column_config={
            "select": st.column_config.CheckboxColumn(
                "sel", help="Tick rules for bulk delete.", width="small"),
            "enabled": st.column_config.CheckboxColumn("on"),
            "case_sensitive": st.column_config.CheckboxColumn("case"),
            "match_type": st.column_config.SelectboxColumn(
                "match_type", options=MATCH_TYPES),
        }, key=editor_key, num_rows="fixed")

    selected_ids = [int(row["id"]) for _, row in edited.iterrows()
                    if bool(row["select"])]

    b1, b2, b3 = st.columns([1.4, 1.6, 1.6])
    if b1.button("Save changes", width="stretch", key="rules_save"):
        saved = 0
        for _, row in edited.iterrows():
            existing = next((r for r in rules if r.id == int(row["id"])), None)
            if existing is None:
                continue
            rules_manager.update_rule(KeywordRule(
                id=int(row["id"]), keyword=str(row["keyword"]),
                account_code=str(row["account_code"]),
                match_type=str(row["match_type"]),
                case_sensitive=bool(row["case_sensitive"]),
                fields=existing.fields, enabled=bool(row["enabled"]),
                notes=str(row["notes"]), priority=existing.priority,
                client_id=existing.client_id, created_at=existing.created_at))
            saved += 1
        common.set_flash(f"Saved {saved} rule(s).")
        st.rerun()
    if b2.button(f"Delete selected ({len(selected_ids)})", width="stretch",
                 key="rules_delete_selected", disabled=not selected_ids):
        st.session_state["_rules_confirm"] = ("delete_selected", selected_ids)
        st.rerun()
    if b3.button("Clear all rules", width="stretch", key="rules_clear_all",
                 type="secondary"):
        st.session_state["_rules_confirm"] = ("clear_all", [])
        st.rerun()

    _render_rules_confirm(rules_manager, len(rules))

    _render_account_knowledge(rules_manager)

    st.divider()
    st.subheader("What the tool remembers")
    mappings = storage.list_learned_mappings()
    notes_count = storage.count_rule_notes()
    st.caption(f"Learned mappings: {len(mappings)} · saved Rule Notes: {notes_count}")
    if mappings:
        mdf = pd.DataFrame([{
            "account_code": m.account_code, "times seen": m.hits,
            "transaction text": m.signature[:90], "last seen": m.last_seen,
        } for m in mappings])
        st.dataframe(mdf, width="stretch", height=240)
        c1, c2 = st.columns(2)
        if c1.button("Forget all remembered mappings"):
            storage.clear_learned_mappings()
            common.set_flash("Cleared remembered mappings.")
            st.rerun()
        if c2.button("Forget all saved Rule Notes"):
            storage.clear_rule_notes()
            common.set_flash("Cleared saved Rule Notes.")
            st.rerun()
    else:
        st.info("Nothing remembered yet. Approve some codes in the spreadsheet.")


def _render_account_knowledge(rules_manager) -> None:
    """🧠 Account Knowledge — what the tool has learned per GL account.

    Shows each account's learned keywords, sample transactions and how often it
    has been used, and lets the user rebuild this knowledge from every rule with
    one click. This is ProfitCoach's compounding advantage: the more it's used,
    the sharper (and more explainable) the coding gets.
    """
    profiles = rules_manager.list_account_profiles()

    st.divider()
    st.subheader("🧠 Account Knowledge")
    st.caption(
        "What the tool has learned about each account from your rules and "
        "matches — keywords, example transactions and how often it's used. This "
        "memory reinforces matching accuracy and explains *why* a code was "
        "chosen. It grows every time you add a rule or run the automation.")

    b1, b2, _ = st.columns([1.8, 1.4, 3])
    if b1.button("Refresh Learning from All Rules", type="primary",
                 width="stretch", key="kb_refresh"):
        n = rules_manager.refresh_learning_from_rules()
        common.set_flash(
            f"Refreshed account knowledge from your rules ({n} account(s)).")
        st.rerun()
    if profiles and b2.button("Clear knowledge", width="stretch",
                              key="kb_clear"):
        rules_manager.clear_account_profiles()
        common.set_flash("Cleared the account knowledge base.")
        st.rerun()

    if not profiles:
        st.info(
            "No account knowledge yet. Add a rule (e.g. **Cloud Hosting → 6100**) or "
            "run the automation and profiles will build automatically.")
        return

    kb = pd.DataFrame([{
        "account": p["account_code"],
        "base": p.get("base_account", ""),
        "scope": (p.get("client_id") or "shared"),
        "used": int(p.get("usage_count", 0)),
        "keywords": ", ".join(p.get("keywords", [])[:12]),
        "examples": len(p.get("sample_texts", [])),
        "notes": p.get("notes", ""),
    } for p in profiles])
    st.dataframe(
        kb, width="stretch", hide_index=True,
        height=min(360, 60 + 34 * len(kb)),
        column_config={
            "used": st.column_config.NumberColumn(
                "used", help="How many times this account has been matched."),
            "examples": st.column_config.NumberColumn(
                "examples", help="Sample transactions captured for this account."),
        })

    # Drill-in: show the learned sample transactions for one account.
    codes = [p["account_code"] for p in profiles]
    pick = st.selectbox("Inspect learned examples for account", codes,
                        key="kb_inspect")
    chosen = next((p for p in profiles if p["account_code"] == pick), None)
    if chosen:
        samples = chosen.get("sample_texts", [])
        if samples:
            st.caption(f"Example transactions learned for **{pick}**:")
            st.dataframe(pd.DataFrame({"transaction text": samples}),
                         width="stretch", hide_index=True,
                         height=min(240, 60 + 34 * len(samples)))
        else:
            st.caption("No example transactions captured for this account yet.")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def render_auto_train_banner() -> None:
    """Public wrapper so the spreadsheet view can show the train reminder."""
    config, storage, _rules, model_manager, logger = services()
    _render_auto_train_banner(config, storage, model_manager, logger)


def _render_auto_train_banner(config, storage, model_manager, logger) -> None:
    """Quiet 'train now — N new examples' reminder (never silent training).

    Appears once approvals since the last successful train reach
    ``ml.auto_train_every``. The button trains the fast LogReg baseline inline
    (small data, sub-second); the heavier SetFit model stays an explicit
    choice on the train button below.
    """
    every = int(getattr(config.ml, "auto_train_every", 0) or 0)
    if every <= 0:
        return
    new_examples = model_manager.new_examples_since_train()
    can_train, _ = model_manager.can_train()
    if new_examples < every or not can_train:
        return
    with st.container(border=True):
        c1, c2 = st.columns([3.4, 1.4])
        c1.markdown(f"**{new_examples:,} new approvals** since the last "
                    "training run.")
        c1.caption("Train now to put them to work — takes a second with the "
                   "built-in LogReg model.")
        if c2.button("Train now", type="primary", width="stretch",
                     key="auto_train_now"):
            try:
                with st.spinner("Training on your latest approvals…"):
                    results = model_manager.train_all(model_types=["logreg"])
                res = results.get("logreg", {})
                if "error" in res:
                    st.warning(f"Training skipped: {res['error']}")
                else:
                    acc = res.get("accuracy")
                    common.set_flash(
                        f"Trained LogReg on {res.get('n_examples', 0):,} "
                        f"examples across {res.get('n_labels', 0)} codes"
                        + (f" — held-out accuracy {acc:.0%}."
                           if acc is not None else ".")
                        + f" Active model: "
                          f"{model_manager.active_model_name() or 'none'}.")
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training did not complete: {exc}")
                logger.error("auto_train_failed", error=str(exc))


def _render_memory_inspector(storage, model_manager, client_id) -> None:
    """Memory inspector: what the tool remembers for this client.

    Counts, last train time, active model + held-out accuracy, and the top
    learned mappings — with the ability to delete a bad mapping so it stops
    firing on the next run.
    """
    st.divider()
    st.subheader("Memory inspector")
    scope = client_id or "shared"
    mappings = storage.list_learned_mappings(client_id=client_id)
    n_examples = storage.count_training_data(client_id=client_id)
    labels = storage.distinct_labels(client_id=client_id)
    n_blocked = storage.count_blocked_mappings()

    statuses = {s.name: s for s in model_manager.status_list()}
    active = model_manager.active_model_name()
    active_status = statuses.get(active) if active else None

    m = st.columns(4)
    m[0].metric("Remembered codes", f"{len(mappings):,}",
                help=f"Exact-match mappings in scope: {scope}.")
    m[1].metric("Training examples", f"{n_examples:,}",
                help=f"{len(labels):,} distinct account codes.")
    m[2].metric("Last trained",
                (model_manager.last_trained_at() or "never")[:19].replace(
                    "T", " "))
    m[3].metric("Active model accuracy",
                f"{active_status.accuracy:.0%}"
                if active_status and active_status.accuracy is not None
                else "—",
                help=f"Held-out accuracy of the active model "
                     f"({active or 'none yet'}).")
    if n_blocked:
        st.caption(f"{n_blocked} rejected pairing(s) are blocked from "
                   "re-suggesting.")

    if not mappings:
        st.info("Nothing remembered yet for this client. Approve codes in "
                "the spreadsheet or Review workspace.")
        return

    st.caption(f"Top {min(20, len(mappings))} remembered mappings "
               "(most-used first). Tick **del** and confirm to forget one — "
               "it stops firing on the next run.")
    top = mappings[:20]
    mdf = pd.DataFrame([{
        "del": False,
        "id": m.id,
        "account_code": m.account_code,
        "times seen": m.hits,
        "scope": m.client_id or "shared",
        "transaction text": m.signature[:80],
    } for m in top])
    edited = st.data_editor(
        mdf, width="stretch", hide_index=True, key="memory_inspector_editor",
        disabled=[c for c in mdf.columns if c != "del"],
        column_config={
            "del": st.column_config.CheckboxColumn("del", width="small"),
            "id": st.column_config.NumberColumn(width="small"),
        })
    to_delete = [int(r["id"]) for _, r in edited.iterrows() if bool(r["del"])]
    if st.button(f"Delete selected mapping(s) ({len(to_delete)})",
                 key="memory_delete", disabled=not to_delete):
        removed = sum(1 for mid in to_delete
                      if storage.delete_learned_mapping(mid))
        common.set_flash(f"Deleted {removed} mapping(s) — they will not fire "
                         "on the next run.")
        st.rerun()


def page_models() -> None:
    config, storage, rules, model_manager, logger = services()
    client_id = common.current_client_id()

    st.write(
        "As codes are approved, the application trains a model in the background "
        "that improves over time. If the model underperforms, it falls back to "
        "the TF-IDF + LogReg matcher automatically.")

    n_examples = model_manager.training_count()
    n_labels = len(storage.distinct_labels())
    avail = model_manager.available_model_types()

    c1, c2, c3 = st.columns(3)
    c1.metric("Examples learned", f"{n_examples:,}")
    c2.metric("Different codes seen", n_labels)
    c3.metric("Model in use", model_manager.active_model_name() or "none yet")

    st.markdown("#### Learning progress")
    hybrid_min = config.ml.hybrid_min
    primary_min = config.ml.ml_primary_min
    if n_examples < hybrid_min:
        target, label = hybrid_min, "Hybrid (model + similarity)"
    elif n_examples < primary_min:
        target, label = primary_min, "Prefer model"
    else:
        target, label = primary_min, "model leading"
    pct = min(n_examples / target, 1.0) if target else 1.0
    st.progress(pct, text=f"{n_examples:,} / {target:,} examples — next: {label}")
    st.caption(
        f"Similarity leads while you're building up approvals. The model "
        f"starts voting alongside similarity at ~{hybrid_min:,} approvals and "
        f"starts leading after ~{primary_min:,} — that's why similarity is "
        "still in charge early on.")

    can_train, reason = model_manager.can_train()
    if not model_manager.has_model() and can_train:
        st.info("Enough examples are available — select **Train the model** below.")
    elif not can_train:
        st.info(f"Continue approving codes in the spreadsheet. {reason}")
    else:
        st.success("Keep the mode on **Auto** — the model is used more as it "
                   "accumulates examples.")

    _render_auto_train_banner(config, storage, model_manager, logger)
    _render_memory_inspector(storage, model_manager, client_id)

    st.markdown("#### Decision mode")
    keys = list(ML_MODE_LABELS.keys())
    current = st.session_state.get("ml_mode", config.ml.mode)
    idx = keys.index(current) if current in keys else 0
    choice = st.radio("Mode", keys, index=idx,
                      format_func=lambda k: ML_MODE_LABELS[k],
                      label_visibility="collapsed")
    st.session_state["ml_mode"] = choice

    st.markdown("#### Engines available")
    a1, a2 = st.columns(2)
    a1.success("LogReg (built-in) — always ready")
    if avail.get("setfit"):
        a2.success("SetFit (advanced) — installed")
    else:
        a2.info("SetFit (advanced) — optional, not installed.")
    common.render_optional_setup()

    st.markdown("#### Train the model")
    if st.button("Train the model now", type="primary", width="stretch",
                 disabled=not can_train):
        prog = st.progress(0.0, text="Starting…")
        msgs: list = []
        box = st.empty()

        def cb(msg: str) -> None:
            msgs.append(msg)
            prog.progress(min(0.2 + 0.2 * len(msgs), 0.95), text=msg)
            box.code("\n".join(msgs[-8:]))

        try:
            with st.spinner("Teaching the model from your approvals…"):
                results = model_manager.train_all(progress_cb=cb)
            prog.progress(1.0, text="Done.")
            parts = []
            for name, res in results.items():
                friendly = "LogReg" if name == "logreg" else "SetFit"
                if "error" in res:
                    if "not installed" in res["error"].lower():
                        continue
                    parts.append(f"{friendly}: skipped")
                else:
                    acc = res.get("accuracy")
                    parts.append(f"{friendly}: " + (f"{acc:.0%} accurate"
                                 if acc is not None else "trained"))
            common.set_flash("Training complete. " + " · ".join(parts))
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training did not complete: {exc}")
            logger.error("train_failed", error=str(exc))

    st.markdown("#### Model comparison")
    statuses = model_manager.status_list()
    comp = pd.DataFrame([{
        "model": ("LogReg" if s.name == "logreg" else "SetFit"),
        "ready": "yes" if s.trained else ("optional" if not s.available else "—"),
        "accuracy": (f"{s.accuracy:.0%}" if s.accuracy is not None else "—"),
        "examples": str(s.n_examples) if s.trained else "—",
        "codes": str(s.n_labels) if s.trained else "—",
        "note": s.note,
    } for s in statuses]).astype(str)
    st.dataframe(comp, width="stretch", hide_index=True)

    st.divider()
    with st.expander("Training examples", expanded=False):
        examples = storage.list_training_data(limit=500)
        if examples:
            tdf = pd.DataFrame([{
                "code": e.label, "learned from": e.engine_used,
                "transaction text": e.text[:100],
                "when": str(e.timestamp)[:19].replace("T", " "),
            } for e in examples])
            st.dataframe(tdf, width="stretch", height=300)
            if st.button("Clear all training examples"):
                storage.clear_training_data()
                common.set_flash("Cleared all learned examples.")
                st.rerun()
        else:
            st.info("Nothing learned yet. Approve codes in the spreadsheet.")


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def page_history() -> None:
    config, storage, *_ = services()
    runs = storage.list_runs(limit=200)
    if not runs:
        st.info("No runs yet. Process a file and your history will appear here.")
        return

    df = pd.DataFrame([{
        "id": r.id, "when": r.run_at, "file": r.file_name, "rows": r.total_rows,
        "examples": r.seeds, "auto_filled": r.auto_filled,
        "filled_review": r.filled_review, "needs_review": r.needs_review,
        "left_blank": r.no_match, "groups": r.groups_found,
        "engine": r.embedding_backend, "mode": r.notes,
    } for r in runs])

    latest = runs[0]
    c = st.columns(4)
    c[0].metric("Total runs", len(runs))
    c[1].metric("Last file", latest.file_name or "—")
    c[2].metric("Last auto-filled", latest.auto_filled)
    c[3].metric("Last needing review", latest.needs_review)

    st.dataframe(df, width="stretch", height=380)
    chart_df = df.set_index("id")[["auto_filled", "filled_review",
                                    "needs_review", "left_blank"]]
    st.bar_chart(chart_df)
