"""Rule creation UX — dialog, selection strip, seed prompts.

``st.data_editor`` renders to a canvas and cannot expose per-cell right-click
text, so this module relies on three *reliable* triggers that always work:

1. **Checkbox selection** → selection strip + toolbar button. Selecting a single
   row exposes a per-column picker (the dependable "create a rule from this
   cell" path).
2. **Target Account inline prompt** — after typing a code, a one-click prompt
   offers to turn it into a reusable rule.
3. **Seeding banner** — coded rows surface candidate rules for bulk review.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from src import spreadsheet_helpers as sh
from src.data_loader import RULE_NOTES_COL
from src.rules_manager import RulesManager
from ui import common
from utils.account_codes import describe, normalize_code


def open_rule_panel(prefill: Optional[Dict] = None) -> None:
    """Open the rule-creation dialog with optional prefill."""
    st.session_state["rule_panel_open"] = True
    if prefill:
        st.session_state["rule_prefill"] = prefill
        st.session_state["rulepanel_keyword"] = str(prefill.get("keyword", ""))
        st.session_state["rulepanel_code"] = str(prefill.get("code", ""))
        st.session_state["rulepanel_notes"] = str(prefill.get("notes", ""))


def close_rule_panel() -> None:
    st.session_state["rule_panel_open"] = False


def queue_target_edit_prompts(
    target_edits: List[Dict],
    rules_manager,
) -> None:
    """After a Target Account edit, queue one inline 'create rule?' prompt."""
    if st.session_state.get("rule_prompt"):
        return
    dismissed = set(st.session_state.get("dismissed_rule_prompts", []))
    for edit in target_edits or []:
        kw = str(edit.get("keyword", "")).strip()
        code = str(edit.get("code", "")).strip()
        key = f"{kw.lower()}|{code}"
        if key in dismissed:
            continue
        if sh.rule_prompt_worthy(kw, code, rules_manager):
            st.session_state["rule_prompt"] = {
                "keyword": kw,
                "code": code,
                "row": int(edit.get("row", -1)),
            }
            return


def render_selection_strip(work, loaded, rules_manager, counts: dict,
                           config) -> None:
    """Compact action bar when rows are selected — fast path to rule creation.

    Selecting exactly one row reveals a per-column picker: the reliable,
    canvas-safe equivalent of "create a rule from this cell".
    """
    sel = sh.selected_indices(work)
    if not sel:
        return

    kw_cols = sh.rule_keyword_columns(work, loaded, config)
    prefill = sh.rule_creation_prefill(work, sel, loaded,
                                       rules_manager=rules_manager,
                                       config=config)
    keyword = str(prefill.get("keyword", ""))
    code = str(prefill.get("code", ""))

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1.2])
        c1.markdown(
            f"**{len(sel)} row(s) selected** — "
            f"keyword: `{keyword or '—'}`"
            + (f" → **{code}**" if code else ""))
        if len(sel) == 1 and kw_cols:
            focus = c2.selectbox(
                "Keyword from column",
                ["Best guess"] + kw_cols,
                key="rule_focus_column",
                label_visibility="collapsed",
                help="Use the best-guess keyword, or take it from one column "
                     "of this row.")
            if focus != "Best guess":
                focused_kw = sh.suggest_keyword_from_cell(
                    work, sel[0], focus, loaded, config)
                if focused_kw:
                    keyword = focused_kw
                    prefill["keyword"] = keyword
                    prefill["column"] = focus
                    prefill["fields"] = [focus]
        else:
            c2.caption("Tip: select one row to pull the keyword from a column.")

        if keyword:
            detail = sh.rule_preview_detail(
                work, keyword, "contains", False,
                list(prefill.get("fields") or []), loaded, config)
            c3.caption(
                f"Would fill **{detail['blank_matches']}** blank row(s)"
                + (f" · {detail['protected_matches']} already coded"
                   if detail["protected_matches"] else ""))

        if c4.button("Create rule", type="primary", width="stretch",
                     key="strip_create_rule"):
            open_rule_panel(prefill)
            st.rerun()


def render_target_account_prompt(work, loaded) -> None:
    """Inline Yes/No prompt after the user types a Target Account code."""
    prompt = st.session_state.get("rule_prompt")
    if not prompt:
        return

    kw = str(prompt.get("keyword", "")).strip()
    code = normalize_code(str(prompt.get("code", "")))
    if not kw or not code:
        st.session_state.pop("rule_prompt", None)
        return

    st.markdown(
        f"<div class='pc-rule-prompt'>Create a reusable rule for "
        f"<strong>{kw}</strong> → <strong>{code}</strong>?</div>",
        unsafe_allow_html=True,
    )
    y, n, _ = st.columns([1, 1, 6])
    if y.button("Yes, create rule", type="primary", key="rule_prompt_yes"):
        config, *_ = common.services()
        row = int(prompt.get("row", -1))
        indices = [row] if 0 <= row < len(work) else sh.selected_indices(work)
        prefill = sh.rule_creation_prefill(work, indices, loaded,
                                           config=config)
        prefill["keyword"] = kw
        prefill["code"] = code
        st.session_state.pop("rule_prompt", None)
        open_rule_panel(prefill)
        st.rerun()
    if n.button("Not now", key="rule_prompt_no"):
        dismissed = list(st.session_state.get("dismissed_rule_prompts", []))
        dismissed.append(f"{kw.lower()}|{code}")
        st.session_state["dismissed_rule_prompts"] = dismissed[-40:]
        st.session_state.pop("rule_prompt", None)
        st.rerun()


def render_memory_bootstrap_banner(work, loaded, rules_manager, storage,
                                   config) -> None:
    """Fresh-file bootstrap: what this client already knows + one-click memory.

    Shown once per file when no automation has run yet and the client has
    rules or remembered codes. "Apply remembered codes" runs the deterministic
    learned-memory-only pass (blank rows, exact signature match) — safe and
    auditable.
    """
    if st.session_state.get("memory_bootstrap_dismissed"):
        return
    if sh.ENGINE_COL not in work.columns:
        return
    # Only before any automation has run (engines are all seed/blank).
    engines = {str(e).strip() for e in work[sh.ENGINE_COL].unique()}
    if engines - {"", "seed"}:
        return

    client_id = st.session_state.get("client_name") or None
    n_rules = len(rules_manager.list_rules(enabled_only=True,
                                           client_id=client_id))
    n_memory = len(storage.list_learned_mappings(client_id=client_id))
    if not n_rules and not n_memory:
        return

    st.markdown(
        f"<div class='pc-seed-banner'>"
        f"<strong>{n_rules} rule{'s' if n_rules != 1 else ''} and "
        f"{n_memory:,} remembered code{'s' if n_memory != 1 else ''}</strong> "
        f"are ready for this client. Run <strong>Rules — Strict</strong> to "
        f"apply rules, or apply the remembered exact matches now."
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.8, 1.2, 4])
    if c1.button("Apply remembered codes", type="primary", width="stretch",
                 key="bootstrap_apply_memory",
                 help="Deterministic: fills blank rows whose exact transaction "
                      "text you approved before. Never overwrites anything."):
        common.push_undo()
        n = sh.run_learned_memory(work, loaded, storage, config,
                                  client_id=client_id)
        st.session_state["memory_bootstrap_dismissed"] = True
        st.session_state["data_version"] = \
            st.session_state.get("data_version", 0) + 1
        if n:
            common.set_flash(
                f"Applied {n:,} remembered code(s) to blank rows — exact "
                "matches you approved before.")
        else:
            common.set_flash(
                "No blank rows matched a remembered code exactly. Run Rules — "
                "Strict next, or Full Intelligent for broader matching.")
        st.rerun()
    if c2.button("Dismiss", width="stretch", key="bootstrap_dismiss"):
        st.session_state["memory_bootstrap_dismissed"] = True
        st.rerun()


def render_ingest_banner(work, loaded, rules_manager, storage, config) -> None:
    """One-click ingest of pre-filled rows as exact rules (Name → Memo → …)."""
    if st.session_state.get("ingest_banner_dismissed"):
        return
    from src.data_loader import summarize_upload
    stats = summarize_upload(work, loaded.new_account_col)
    if stats["prefilled"] <= 0:
        return

    st.markdown(
        f"<div class='pc-seed-banner'>"
        f"<strong>{stats['prefilled']:,} coded example"
        f"{'s' if stats['prefilled'] != 1 else ''}</strong> found in this "
        f"file. Turn them into exact-match rules in one click — they fill "
        f"identical transactions now and on every future file."
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.8, 1.2, 4])
    if c1.button("Ingest as exact rules", type="primary", width="stretch",
                 key="banner_ingest_exact",
                 help="Creates one high-priority exact-match rule per coded "
                      "row, keyed on Name, then Memo, then Description. "
                      "Review them anytime in the Rules panel."):
        client_id = st.session_state.get("client_name") or None
        source_cols = sh.mining_columns(work, loaded, config)
        created = rules_manager.ingest_existing_new_account_as_rules(
            work, client_id=client_id, source_text_cols=source_cols)
        st.session_state["ingest_banner_dismissed"] = True
        common.set_flash(
            f"Ingested {created} exact rule(s) from your coded rows. Run "
            "Rules — Strict to apply them."
            if created else
            "Those coded rows are already covered by existing rules.")
        st.rerun()
    if c2.button("Dismiss", width="stretch", key="banner_ingest_dismiss"):
        st.session_state["ingest_banner_dismissed"] = True
        st.rerun()


def render_seed_suggestion_banner(work, loaded, rules_manager) -> None:
    """Banner when coded rows yield candidate rules (upload / seeding)."""
    if st.session_state.get("hide_rule_suggestions"):
        return
    config, *_ = common.services()
    try:
        cands = sh.candidate_rules(work, loaded, rules_manager, config=config)
    except Exception:  # noqa: BLE001
        return
    if not cands:
        return

    n = len(cands)
    st.markdown(
        f"<div class='pc-seed-banner'>"
        f"<strong>{n} potential rule{'s' if n != 1 else ''}</strong> detected "
        f"from your seeded Target Account values. "
        f"Review and create them once — they apply to every future file."
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.4, 1.4, 4])
    if c1.button("Review and create", type="primary", width="stretch",
                 key="banner_review_create"):
        st.session_state["panel"] = "rules"
        st.session_state["export_open"] = False
        st.rerun()
    if c2.button("Dismiss", width="stretch", key="banner_dismiss_rules"):
        st.session_state["hide_rule_suggestions"] = True
        st.rerun()


@st.dialog("Create rule", width="large")
def rule_creation_dialog(work, loaded, config, rules, storage) -> None:
    """Modal rule builder with live preview."""
    prefill = st.session_state.get("rule_prefill", {})
    indices = list(prefill.get("indices") or sh.selected_indices(work))
    kw_cols = sh.rule_keyword_columns(work, loaded, config)

    st.caption(
        "A rule is organised around **one Target Account code**. Give it as many "
        "keywords, phrases and characteristics as you like — the rule matches if "
        "ANY of them appears. Matching runs across all transaction text (Name, "
        "Payee, Description, Memo) unless you narrow it to specific columns. When "
        "you run rules, exact matches fill first, then their codes intelligently "
        "spread to similar transactions with a confidence score.")

    st.markdown("**1. Target Account** — the code this rule assigns")
    t1, t2 = st.columns([2, 4])
    code = t1.text_input(
        "Target Account code",
        key="rulepanel_code",
        placeholder="e.g. 6100",
        help="Every keyword/phrase in this rule maps to this one account code.")
    if code.strip():
        norm = normalize_code(code)
        desc = describe(norm)
        t2.caption(f"This rule codes matching transactions to **{norm}**"
                   + (f" — {desc}" if desc else "") + ".")

    st.markdown("**2. Keywords & phrases** — what to look for")
    c1, c3 = st.columns([4, 2])
    keyword = c1.text_input(
        "When you see any of these words or phrases",
        key="rulepanel_keyword",
        placeholder="Rent Signage, signage, yard sign, For Rent Sign",
        help="Enter multiple keywords or phrases separated by commas. The rule "
             "matches if ANY of them appears anywhere in the transaction. Add "
             "every spelling variation you can think of.")
    match_type = c3.selectbox(
        "Match type",
        ["contains", "exact", "starts_with", "ends_with", "fuzzy", "regex"],
        key="rulepanel_match",
        help="contains · exact · starts_with · ends_with · fuzzy (typos) · regex")

    # Live feedback so users can see their comma-separated phrases were parsed.
    _phrases = [p.strip() for p in (keyword or "").split(",") if p.strip()]
    if len(_phrases) > 1:
        st.caption(f"Matches any of **{len(_phrases)}** phrases: "
                   + " · ".join(f"“{p}”" for p in _phrases[:8])
                   + (" …" if len(_phrases) > 8 else ""))
    else:
        st.caption("Tip: add more variations separated by commas, e.g. "
                   "`Rent Signage, signage, yard sign`.")

    default_fields = list(prefill.get("fields") or [])
    focus_default = str(prefill.get("column", ""))
    if focus_default and focus_default not in default_fields:
        default_fields.append(focus_default)
    default_fields = [f for f in default_fields if f in kw_cols]

    st.markdown("**3. Options**")
    c5, c6 = st.columns([3, 1.4])
    notes = c5.text_input(
        "Rule notes (optional)",
        key="rulepanel_notes",
        placeholder=str(prefill.get("notes_placeholder", "")
                      or "Optional hint for the matcher"))
    case_sensitive = c6.checkbox("Case sensitive", value=False,
                                 key="rulepanel_case")

    with st.expander("Advanced: restrict to specific columns only",
                     expanded=bool(default_fields)):
        st.caption(
            "By default the rule searches the **whole transaction row**, so you "
            "don't need to choose a column. Pick one or more columns here only "
            "to limit this rule to them.")
        fields: List[str] = st.multiselect(
            "Search only in these columns",
            kw_cols,
            default=default_fields,
            key="rulepanel_focus",
            label_visibility="collapsed")

    keyword = keyword.strip()
    blank_matches = 0
    if keyword:
        detail = sh.rule_preview_detail(
            work, keyword, match_type, case_sensitive, fields, loaded, config)
        blank_matches = int(detail["blank_matches"])
        protected = int(detail["protected_matches"])
        st.markdown(
            f"**Live preview — fills {blank_matches} blank row(s)**"
            + (f" · matches {protected} already-coded row(s) "
               "*(those are never overwritten)*" if protected else ""))
        sample = detail["samples"]
        if not sample.empty:
            st.dataframe(sample, width="stretch", hide_index=True, height=160)
        if code.strip():
            norm = normalize_code(code)
            desc = describe(norm)
            st.caption(
                f"Will set Target Account to **{norm}**"
                + (f" ({desc})" if desc else ""))
    else:
        st.caption("Enter a keyword to see the live match preview.")

    # Guard: a rule that fills zero blank rows is usually a typo — make the
    # user confirm before saving it.
    confirm_zero = True
    if keyword and blank_matches == 0:
        st.warning("This rule matches **0 blank rows** in the current file, "
                   "so it would fill nothing here right now.")
        confirm_zero = st.checkbox(
            "Save it anyway — it will apply to future files",
            value=False, key="rulepanel_confirm_zero")

    b1, b2, b3 = st.columns([2, 2, 1])
    create = b1.button(
        "Create rule and apply", type="primary", width="stretch",
        disabled=not (keyword and code.strip() and confirm_zero),
        key="rulepanel_create_btn")
    stamp_notes = b2.checkbox(
        "Stamp rule notes on matched rows",
        value=bool(str(prefill.get("notes", "")).strip()),
        key="rulepanel_applynotes")
    if b3.button("Cancel", width="stretch", key="rulepanel_cancel"):
        close_rule_panel()
        st.rerun()

    if create:
        _save_rule_from_dialog(
            work, loaded, config, rules, storage,
            keyword, code.strip(), match_type, case_sensitive,
            fields, notes.strip(), stamp_notes)


def _save_rule_from_dialog(work, loaded, config, rules, storage,
                           keyword, code, match_type, case_sensitive,
                           fields, notes, stamp_notes) -> None:
    common.push_undo()
    rules.add_rule(
        keyword, code, match_type=match_type,
        case_sensitive=case_sensitive, fields=fields, notes=notes)

    if stamp_notes and notes:
        from models.schemas import KeywordRule
        rule = KeywordRule(
            keyword=keyword, account_code="0", match_type=match_type,
            case_sensitive=case_sensitive, fields=fields)
        for i in range(len(work)):
            sim = work.iloc[i][sh.SIM_TEXT_COL] if sh.SIM_TEXT_COL in work.columns else ""
            if RulesManager._rule_matches(rule, sim, work.iloc[i]):
                work.at[work.index[i], RULE_NOTES_COL] = notes
        from src.data_loader import recompute_sim_text
        recompute_sim_text(work, loaded.text_columns, config)
        sh._persist_changed_notes(work, storage)

    applied = sh.run_rules_only(work, rules, loaded, config, overwrite=False)
    close_rule_panel()
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    common.set_flash(
        f"Created rule '{keyword}' → {normalize_code(code)} and filled "
        f"{applied} matching row(s). Saved for future files.")
    st.rerun()
