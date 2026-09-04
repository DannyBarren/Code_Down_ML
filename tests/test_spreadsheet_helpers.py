"""Tests for the Spreadsheet-Dominant v2.2 working-dataframe logic.

These cover the parts most likely to drift: work_df construction, Rule Notes
folding + persistence, full/rules-only runs, manual edits via the grid,
bulk approve, live rule preview, filters/metrics, pagination and undo/redo.
"""

from __future__ import annotations

import pandas as pd

from src import spreadsheet_helpers as sh
from src.data_loader import RULE_NOTES_COL, SIM_TEXT_COL, BASE_SIG_COL
from src.fill_down_engine import FillDownEngine


def _engine(config, rules, storage):
    return FillDownEngine(config, rules, learned_lookup=storage.get_learned_lookup())


def test_build_work_df_has_internal_columns(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Office Depot", "Description": "paper", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    for col in (RULE_NOTES_COL, SIM_TEXT_COL, BASE_SIG_COL, sh.SELECT_COL,
                sh.CONF_COL, sh.ENGINE_COL, sh.ACTION_COL, sh.WHY_COL,
                sh.SUGGESTED_COL):
        assert col in work.columns
    # Seed row is pre-marked.
    assert work.iloc[0][sh.ACTION_COL] == "kept_seed"
    assert work.iloc[0][sh.CONF_COL] == 1.0


def test_rule_notes_fold_into_sim_text(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Weird Vendor", "Description": "special", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, RULE_NOTES_COL] = "landscaping quarterly"
    sh.recompute_sim_text(work, loaded.text_columns, config)
    assert "landscaping" in work.iloc[0][SIM_TEXT_COL]
    # Base signature excludes the note.
    assert "landscaping" not in work.iloc[0][BASE_SIG_COL]


def test_clear_spreadsheet_values_wipes_outputs_keeps_columns(
        make_loaded, storage, rules, config):
    rules.seed_default_rules()
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    sh.run_full(work, loaded, config, _engine(config, rules, storage))
    work.at[0, RULE_NOTES_COL] = "monthly screening vendor"
    sh.recompute_sim_text(work, loaded.text_columns, config)

    na = loaded.new_account_col
    n = sh.clear_spreadsheet_values(work, loaded, config)

    assert n == len(work)
    # Outputs are wiped.
    assert (work[na] == "").all()
    assert (work[RULE_NOTES_COL] == "").all()
    assert (work[sh.ENGINE_COL] == "").all()
    assert (work[sh.CONF_COL] == 0.0).all()
    assert not work[sh.SELECT_COL].any()
    # Original client columns are preserved.
    assert list(work["Name"]) == ["Cred Hub", "Cred Hub"]
    assert list(work["Description"]) == ["fee", "fee"]
    # Cleared notes no longer leak into the similarity text.
    assert "screening" not in work.iloc[0][SIM_TEXT_COL]


def test_full_run_syncs_work_df(make_loaded, storage, rules, config):
    rules.seed_default_rules()
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
        {"Name": "Mystery", "Description": "zzz", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    result = sh.run_full(work, loaded, config, _engine(config, rules, storage))
    # Row 1 gets filled (rule or similarity), meta synced.
    assert str(work.iloc[1]["New Account"]).strip() == "6618S"
    assert work.iloc[1][sh.ENGINE_COL] != ""
    assert work.iloc[1][sh.ACTION_COL] != ""
    assert len(result.results) == 3


def test_run_rules_only_fills_blanks(make_loaded, storage, rules, config):
    rules.seed_default_rules()
    loaded = make_loaded([
        {"Name": "Cloud Hosting", "Description": "hosting", "New Account": ""},
        {"Name": "Nothing", "Description": "noise", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    n = sh.run_rules_only(work, rules, loaded, config)
    assert n == 1
    assert work.iloc[0]["New Account"] == "6100"
    assert work.iloc[1]["New Account"] == ""


def test_commit_editor_changes_target_and_notes(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "X", "Description": "y", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    edited = work.loc[[0]].copy()
    edited.at[0, "New Account"] = "6618 s"       # should normalize -> 6618S
    edited.at[0, RULE_NOTES_COL] = "monthly fee"
    counts = sh.commit_editor_changes(work, edited, loaded, storage, config)
    assert counts["targets"] == 1 and counts["notes"] == 1
    assert work.iloc[0]["New Account"] == "6618S"
    assert "monthly fee" in work.iloc[0][SIM_TEXT_COL]
    # Manual edit was learned.
    assert storage.get_learned_lookup()  # non-empty
    # Note was persisted by base signature.
    assert storage.count_rule_notes() == 1


def test_rule_notes_persist_across_uploads(make_loaded, storage, config):
    records = [{"Name": "Acme Co", "Description": "service", "New Account": ""}]
    loaded1 = make_loaded(records)
    work1 = sh.build_work_df(loaded1, storage, config)
    edited = work1.loc[[0]].copy()
    edited.at[0, RULE_NOTES_COL] = "recurring janitorial"
    sh.commit_editor_changes(work1, edited, loaded1, storage, config)

    # Fresh upload of the same transaction text -> notes re-attach.
    loaded2 = make_loaded(records)
    work2 = sh.build_work_df(loaded2, storage, config)
    assert work2.iloc[0][RULE_NOTES_COL] == "recurring janitorial"
    assert "janitorial" in work2.iloc[0][SIM_TEXT_COL]


def test_clearing_notes_removes_persistence(make_loaded, storage, config):
    records = [{"Name": "Acme Co", "Description": "service", "New Account": ""}]
    loaded = make_loaded(records)
    work = sh.build_work_df(loaded, storage, config)
    e = work.loc[[0]].copy()
    e.at[0, RULE_NOTES_COL] = "temp note"
    sh.commit_editor_changes(work, e, loaded, storage, config)
    assert storage.count_rule_notes() == 1
    e2 = work.loc[[0]].copy()
    e2.at[0, RULE_NOTES_COL] = ""
    sh.commit_editor_changes(work, e2, loaded, storage, config)
    assert storage.count_rule_notes() == 0


def test_approve_rows_fills_from_suggestion(make_loaded, storage, rules, config):
    rules.seed_default_rules()
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee variant", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    sh.run_full(work, loaded, config, _engine(config, rules, storage))
    # Simulate a blank-but-suggested row.
    work.at[1, "New Account"] = ""
    work.at[1, sh.SUGGESTED_COL] = "6618S"
    out = sh.approve_rows(work, loaded, storage, config, indices=[1])
    assert out["applied"] == 1
    assert work.iloc[1]["New Account"] == "6618S"


def test_rule_preview_counts(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Home Depot", "Description": "supplies", "New Account": ""},
        {"Name": "The Home Depot", "Description": "hardware", "New Account": ""},
        {"Name": "Lowes", "Description": "materials", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    count, sample = sh.rule_preview(
        work, "home depot", "contains", False, [], loaded, config)
    assert count == 2
    assert len(sample) == 2


def test_filter_mask_and_summary(make_loaded, storage, rules, config):
    rules.seed_default_rules()
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
        {"Name": "zzz", "Description": "qqq", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    sh.run_full(work, loaded, config, _engine(config, rules, storage))
    counts = sh.summary_counts(work, loaded)
    assert counts["total"] == 3
    assert counts["seeds"] >= 1
    mask = sh.filter_mask(work, "Blanks only")
    assert mask.sum() == counts["blank"]


def test_summary_counts_reports_distinct_accounts(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "A", "Description": "x", "New Account": "6210S"},
        {"Name": "B", "Description": "x", "New Account": "6210S"},
        {"Name": "C", "Description": "x", "New Account": "6230S"},
        {"Name": "D", "Description": "x", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    counts = sh.summary_counts(work, loaded)
    assert counts["distinct_accounts"] == 2


def test_value_note_time_saved_and_insight():
    from ui import common
    # Nothing auto-coded yet but accounts exist -> only the insight shows.
    note = common.value_note({"auto_filled": 0, "distinct_accounts": 3})
    assert "account" in note and "saved" not in note
    # Auto-coded work yields a credible time-saved estimate.
    note2 = common.value_note({"auto_filled": 40, "distinct_accounts": 5})
    assert "saved" in note2 and "account" in note2
    assert common.minutes_saved({"auto_filled": 40}) == 10  # 40*15s = 600s
    # Clean slate stays empty (no clutter before any value is delivered).
    assert common.value_note({"auto_filled": 0, "distinct_accounts": 0}) == ""


def test_search_mask_searches_all_records(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Office Depot", "Description": "toner", "New Account": "6210S"},
        {"Name": "Home Depot", "Description": "lumber", "New Account": ""},
        {"Name": "City Water", "Description": "utility", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Single term, case-insensitive, spans the whole frame (not a page).
    m = sh.search_mask(work, "depot")
    assert m.sum() == 2
    # Multi-term AND: both must appear somewhere in the row.
    m2 = sh.search_mask(work, "office toner")
    assert m2.sum() == 1 and bool(m2.iloc[0])
    # Column-scoped search.
    m3 = sh.search_mask(work, "depot", columns=["Description"])
    assert m3.sum() == 0
    # Searching the Target Account code works too.
    m4 = sh.search_mask(work, "6210", columns=["New Account"])
    assert m4.sum() == 1
    # Empty query matches everything.
    assert sh.search_mask(work, "").all()


def test_build_view_mask_combines_filters(make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Office Depot", "Description": "toner", "New Account": "6210S"},
        {"Name": "Office Depot", "Description": "paper", "New Account": ""},
        {"Name": "Sparkle Clean", "Description": "janitorial", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Blanks only + search "office" -> just the blank Office Depot row.
    mask = sh.build_view_mask(
        work, loaded, mode="Blanks only", query="office")
    assert mask.sum() == 1
    assert work[mask].iloc[0]["Description"] == "paper"


def test_engine_filter_and_available_engines(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, sh.ENGINE_COL] = "seed"
    work.at[1, sh.ENGINE_COL] = "rules"
    engines = sh.available_engines(work)
    assert "Already coded by you" in engines and "Keyword rule" in engines
    mask = sh.engine_mask(work, ["Keyword rule"])
    assert mask.sum() == 1 and bool(mask.iloc[1])


def test_searchable_columns_excludes_internal(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Acme", "Description": "svc", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cols = sh.searchable_columns(work, loaded)
    assert "Name" in cols and "New Account" in cols
    assert not any(str(c).startswith("_") for c in cols)


def test_pagination_slices(make_loaded, storage, config):
    records = [{"Name": f"V{i}", "Description": "d", "New Account": ""}
               for i in range(25)]
    loaded = make_loaded(records)
    work = sh.build_work_df(loaded, storage, config)
    mask = sh.filter_mask(work, "All")
    page0, n_pages, total = sh.page_slice(work, mask, 0, 10)
    assert total == 25 and n_pages == 3 and len(page0) == 10
    page2, _, _ = sh.page_slice(work, mask, 2, 10)
    assert len(page2) == 5
    # Index labels are preserved for write-back.
    assert list(page2.index) == list(range(20, 25))


def test_undo_redo_snapshot(make_loaded, storage, config):
    loaded = make_loaded([{"Name": "X", "Description": "y", "New Account": ""}])
    work = sh.build_work_df(loaded, storage, config)
    snap = sh.snapshot(work)
    work.at[0, "New Account"] = "1234S"
    assert work.iloc[0]["New Account"] == "1234S"
    sh.restore(work, snap)
    assert work.iloc[0]["New Account"] == ""


def test_selection_helpers(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "A", "Description": "x", "New Account": ""},
        {"Name": "B", "Description": "y", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    sh.set_selection(work, [1], True)
    assert sh.selected_indices(work) == [1]
    sh.clear_selection(work)
    assert sh.selected_indices(work) == []


# --------------------------------------------------------------------------- #
# Candidate rules from already-coded rows (the proactive seeding workflow)
# --------------------------------------------------------------------------- #
def test_candidate_rules_from_coded_rows(make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Acme Landscaping", "Description": "lawn", "New Account": "6300S"},
        {"Name": "Acme Landscaping", "Description": "lawn", "New Account": "6300S"},
        {"Name": "Acme Landscaping", "Description": "lawn", "New Account": ""},
        {"Name": "Random Vendor", "Description": "misc", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cands = sh.candidate_rules(work, loaded, rules)
    kws = {c["keyword"]: c for c in cands}
    # A keyword from the coded "Acme Landscaping" rows is suggested -> 6300S.
    assert any(c["account_code"] == "6300S" for c in cands)
    # "acme" appears in all three Acme rows (incl. the blank one) = 3 matches.
    acme = next((c for c in cands if c["keyword"] == "acme"), None)
    assert acme is not None and acme["matches"] == 3


def test_candidate_rules_skip_existing_keyword(make_loaded, storage, rules, config):
    rules.add_rule("acme", "6300S")  # already covered
    loaded = make_loaded([
        {"Name": "Acme Co", "Description": "x", "New Account": "6300S"},
        {"Name": "Acme Co", "Description": "x", "New Account": "6300S"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cands = sh.candidate_rules(work, loaded, rules)
    assert all(c["keyword"] != "acme" for c in cands)


def test_candidate_rules_rejects_overbroad_keyword(
        make_loaded, storage, rules, config):
    # "expenses" appears in every row but across two different codes -> too
    # generic/ambiguous to suggest. "acme"/"globex" are specific (purity 1.0).
    loaded = make_loaded([
        {"Name": "Acme Co", "Description": "monthly expenses", "New Account": "6300S"},
        {"Name": "Acme Co", "Description": "monthly expenses", "New Account": "6300S"},
        {"Name": "Globex", "Description": "monthly expenses", "New Account": "6400S"},
        {"Name": "Globex", "Description": "monthly expenses", "New Account": "6400S"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cands = sh.candidate_rules(work, loaded, rules)
    assert all(c["keyword"] != "expenses" for c in cands)
    assert all(c["purity"] >= 0.7 for c in cands)


def test_rule_notes_persist_with_duplicate_signatures(
        make_loaded, storage, config):
    # Two identical transactions share a base signature. A note on one must
    # persist and not be clobbered by the blank duplicate.
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    edited = work.loc[[0]].copy()
    edited.at[0, RULE_NOTES_COL] = "resident screening"
    sh.commit_editor_changes(work, edited, loaded, storage, config)
    assert storage.count_rule_notes() == 1
    lookup = storage.get_rule_notes_lookup()
    assert "resident screening" in lookup.values()


def test_run_rules_only_matches_non_similarity_column(
        make_loaded, storage, rules, config):
    """A rule keyed on a Payee vendor (not a similarity column) fills blanks."""
    loaded = make_loaded([
        {"Payee": "Acme Plumbing", "Memo": "job", "New Account": ""},
        {"Payee": "Acme Plumbing", "Memo": "job", "New Account": ""},
        {"Payee": "Other Co", "Memo": "x", "New Account": ""},
    ])
    rules.add_rule("Acme Plumbing", "6300S")
    work = sh.build_work_df(loaded, storage, config)
    n = sh.run_rules_only(work, rules, loaded, config)
    assert n == 2
    assert work.iloc[0]["New Account"] == "6300S"
    assert work.iloc[2]["New Account"] == ""


def test_run_rules_only_override_protects_user_values(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Acme", "Description": "x", "New Account": "1111S"},  # seed
        {"Name": "Acme", "Description": "x", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Simulate an engine guess on row 1 that the rule should override.
    work.at[1, "New Account"] = "2222S"
    work.at[1, sh.ENGINE_COL] = "similarity"
    rules.add_rule("Acme", "6300S")
    n = sh.run_rules_only(work, rules, loaded, config, indices=[0, 1],
                          overwrite=True)
    assert n == 1                                   # only the engine guess changed
    assert work.iloc[0]["New Account"] == "1111S"   # seed protected
    assert work.iloc[1]["New Account"] == "6300S"   # guess overridden


def test_candidate_rules_from_non_similarity_column(
        make_loaded, storage, rules, config):
    """Seeds whose vendor is only in Payee still produce a candidate rule."""
    loaded = make_loaded([
        {"Date": "1/1", "Payee": "Bluewater Pools", "New Account": "6400S"},
        {"Date": "1/2", "Payee": "Bluewater Pools", "New Account": "6400S"},
        {"Date": "1/3", "Payee": "Bluewater Pools", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cands = sh.candidate_rules(work, loaded, rules)
    assert any(c["account_code"] == "6400S" for c in cands)
    # No date-like junk keywords leak in.
    assert all(any(ch.isalpha() for ch in c["keyword"]) for c in cands)
    assert all("/" not in c["keyword"] for c in cands)


def test_suggest_keyword_prefers_shared_specific_phrase(
        make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Office Depot", "Description": "paper", "New Account": ""},
        {"Name": "Office Depot", "Description": "toner", "New Account": ""},
        {"Name": "The Home Depot", "Description": "lumber", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Selecting the two Office Depot rows yields the specific shared phrase.
    kw = sh.suggest_keyword_from_rows(work, [0, 1], loaded)
    assert "office" in kw and "depot" in kw


def test_suggest_keyword_from_cell(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Office Depot", "Memo": "supplies", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    kw = sh.suggest_keyword_from_cell(work, 0, "Name", loaded)
    assert "office" in kw and "depot" in kw


def test_rule_creation_prefill_uses_target_account(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    pre = sh.rule_creation_prefill(work, [0, 1], loaded, rules_manager=rules)
    assert pre["code"] == "6618S"
    assert "cred" in str(pre["keyword"]).lower()


def test_rule_prompt_worthy_skips_existing_keyword(
        make_loaded, storage, rules, config):
    rules.add_rule("appfolio", "6520S")
    assert not sh.rule_prompt_worthy("appfolio", "6520S", rules)
    assert sh.rule_prompt_worthy("newvendor", "1234S", rules)


def test_single_row_keyword_is_reusable_not_overspecific(
        make_loaded, storage, config):
    """A single coded row yields a short vendor phrase that matches its siblings.

    Regression: previously a single row produced a 3-token keyword that only
    matched itself (the "applied to ~1 row" problem).
    """
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "resident screening", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "monthly bureau", "New Account": ""},
        {"Name": "Cred Hub", "Description": "annual report", "New Account": ""},
        {"Name": "Other Vendor", "Description": "misc", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    kw = sh.suggest_keyword_from_rows(work, [0], loaded)
    # Short, vendor-focused keyword.
    assert kw == "cred hub"
    count, _ = sh.rule_preview(work, kw, "contains", False, [], loaded, config)
    assert count == 3  # all three Cred Hub rows, not just the seeded one


def test_rule_keyword_columns_excludes_numeric(make_loaded, storage, config):
    loaded = make_loaded([
        {"Date": "2025-01-04", "Name": "AppFolio", "Amount": "1860.34",
         "Description": "software", "New Account": "6520S"},
        {"Date": "2025-02-04", "Name": "AppFolio", "Amount": "12.00",
         "Description": "software", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cols = sh.rule_keyword_columns(work, loaded)
    assert "Date" not in cols and "Amount" not in cols
    assert "Name" in cols
    # Vendor column comes first.
    assert cols[0] == "Name"


def test_commit_editor_returns_target_edits(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Acme Co", "Description": "service", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    edited = work.loc[[0]].copy()
    edited.at[0, "New Account"] = "6300S"
    out = sh.commit_editor_changes(work, edited, loaded, storage, config)
    assert out["targets"] == 1
    assert len(out["target_edits"]) == 1
    assert out["target_edits"][0]["code"] == "6300S"
    assert out["target_edits"][0]["keyword"]


def test_create_rules_from_candidates_persists_and_dedupes(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Bluewater Pools", "Description": "x", "New Account": "6400S"},
        {"Name": "Bluewater Pools", "Description": "x", "New Account": "6400S"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cands = sh.candidate_rules(work, loaded, rules)
    before = len(rules.list_rules())
    created = sh.create_rules_from_candidates(rules, cands)
    assert created == len(cands) and created > 0
    assert len(rules.list_rules()) == before + created
    # Re-running creates nothing (keywords now exist).
    assert sh.create_rules_from_candidates(rules, cands) == 0
