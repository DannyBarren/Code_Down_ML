"""Ingest coded rows as **one rule per GL account** (Name -> Memo, never Notes).

One GL code has many vendor names — that is the normal case. Ingest groups the
coded rows by canonical account and builds one rule per account whose phrase
list carries every distinct vendor Name, then every distinct Memo term.
Re-ingesting merges into the existing rule instead of spawning duplicates.
"""

from __future__ import annotations

import pandas as pd

from models.schemas import KeywordRule
from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL
from src.fill_down_engine import FillDownEngine
from src.rules_manager import RulesManager
from utils.account_codes import normalize_code


def _phrases_lower(rule: KeywordRule) -> list:
    return [p.lower() for p in rule.match_phrases()]


# --------------------------------------------------------------------------- #
# normalize_code: decimal sub-accounts survive (Swann uses 6760.01)
# --------------------------------------------------------------------------- #
def test_normalize_code_keeps_decimal_sub_accounts_and_plain_codes():
    assert normalize_code("6760.01") == "6760.01"
    assert normalize_code("6326") == "6326"
    # Excel-float artefacts still collapse; letter suffixes still canonicalise.
    assert normalize_code("6100.0") == "6100"
    assert normalize_code("6100 a") == "6100A"


# --------------------------------------------------------------------------- #
# Test 1 — six coded rows, one account -> ONE rule with every Name + Memo term
# --------------------------------------------------------------------------- #
def _six_row_book():
    return [
        {"Name": "Harbor Plumbing", "Memo": "repair call", "Amount": "120.00",
         "New Account": "6326"},
        {"Name": "Willow Tree Plumbing", "Memo": "repair call",
         "Amount": "88.50", "New Account": "6326"},
        {"Name": "Cedar Pipe Works", "Memo": "water heater swap",
         "Amount": "410.00", "New Account": "6326"},
        {"Name": "Harbor Plumbing", "Memo": "", "Amount": "60.00",
         "New Account": "6326"},
        {"Name": "Cedar Pipe Works", "Memo": "repair call",
         "Amount": "75.00", "New Account": "6326"},
        {"Name": "Northgate Office Supply", "Memo": "printer paper",
         "Amount": "42.00", "New Account": "6200"},
    ]


def test_ingest_builds_one_rule_per_account_with_all_names_then_memos(
        make_loaded, rules, config):
    loaded = make_loaded(_six_row_book())
    created = rules.ingest_existing_new_account_as_rules(
        loaded.df,
        source_text_cols=sh.mining_columns(loaded.df, loaded, config))
    # Two accounts in the book -> two rules touched.
    assert created == 2

    by_code = {r.account_code: r for r in rules.list_rules()}
    assert set(by_code) == {"6326", "6200"}
    rule = by_code["6326"]
    phrases = _phrases_lower(rule)
    # Every distinct vendor Name on 6326, exactly once each.
    assert phrases[:3] == ["harbor plumbing", "willow tree plumbing",
                           "cedar pipe works"]
    # ...then the two distinct Memo terms.
    assert set(phrases[3:]) == {"repair call", "water heater swap"}
    assert len(phrases) == 5
    # No extra rules for that account.
    assert sum(1 for r in rules.list_rules() if r.account_code == "6326") == 1
    # Ingested rules are contains + high priority, scoped to Name + Memo.
    assert rule.match_type == "contains"
    assert rule.priority == 10
    assert [f.lower() for f in rule.fields] == ["name", "memo"]


def test_reingest_merges_into_existing_rule_and_keeps_its_id(
        make_loaded, rules, config):
    loaded = make_loaded(_six_row_book())
    cols = sh.mining_columns(loaded.df, loaded, config)
    rules.ingest_existing_new_account_as_rules(loaded.df, client_id="acme",
                                               source_text_cols=cols)
    first = {r.account_code: r for r in rules.list_rules()}["6326"]

    # A later file adds a brand-new vendor to the same account.
    later = make_loaded([
        {"Name": "Riverstone Rooter", "Memo": "drain clearing",
         "Amount": "99.00", "New Account": "6326"},
    ])
    touched = rules.ingest_existing_new_account_as_rules(
        later.df, client_id="acme",
        source_text_cols=sh.mining_columns(later.df, later, config))
    assert touched == 1
    merged = [r for r in rules.list_rules() if r.account_code == "6326"]
    assert len(merged) == 1                       # merged, not duplicated
    assert merged[0].id == first.id               # same rule id
    phrases = _phrases_lower(merged[0])
    assert "riverstone rooter" in phrases
    assert "harbor plumbing" in phrases            # nothing lost
    assert "drain clearing" in phrases

    # Nothing new -> nothing touched (idempotent).
    assert rules.ingest_existing_new_account_as_rules(
        later.df, client_id="acme",
        source_text_cols=sh.mining_columns(later.df, later, config)) == 0


def test_ingest_report_persists_every_phrase_without_truncation(
        make_loaded, rules, config):
    names = [f"Vendor {chr(65 + i // 26)}{chr(65 + i % 26)} LLC"
             for i in range(80)]
    records = [{"Name": n, "Memo": f"invoice #{i}", "Amount": "1.00",
                "New Account": "6300"} for i, n in enumerate(names)]
    loaded = make_loaded(records)
    report = rules.ingest_rules_per_account(
        loaded.df, source_text_cols=sh.mining_columns(loaded.df, loaded, config))
    rule = rules.list_rules()[0]
    phrases = rule.match_phrases()
    assert report.rules_created == 1
    # All 80 vendor names persisted on the one rule — never truncated.
    assert phrases[:80] == names
    # Digit / "#" tokens are dropped so "invoice #3" and "invoice #7" collapse
    # to one reusable memo term.
    assert phrases[80:] == ["invoice"]
    assert report.phrases_added == 81


# --------------------------------------------------------------------------- #
# Test 3 — Name wins at match time; Memo only when Name is empty / missed
# --------------------------------------------------------------------------- #
def test_name_beats_noisy_memo_and_memo_only_rows_still_match(
        make_loaded, rules, storage, config):
    coded = make_loaded([
        {"Name": "Harbor Plumbing", "Memo": "repair call",
         "New Account": "6326"},
        {"Name": "Golden Gate Electric", "Memo": "lighting repair",
         "New Account": "6335"},
    ])
    rules.ingest_existing_new_account_as_rules(
        coded.df, source_text_cols=sh.mining_columns(coded.df, coded, config))

    blank_rows = [
        # Name belongs to 6335, but the Memo carries 6326's "repair call".
        {"Name": "Golden Gate Electric", "Memo": "repair call after storm",
         "New Account": ""},
        # Memo-only row (no Name): the Memo term on 6326's rule still hits.
        {"Name": "", "Memo": "water heater repair call", "New Account": ""},
        # Unknown vendor with an unrelated memo stays blank.
        {"Name": "Someone Else", "Memo": "misc", "New Account": ""},
    ]
    blank = make_loaded(blank_rows)
    engine = FillDownEngine(config, rules, learned_lookup={})
    df, results = engine.run_selected_rules(blank.df,
                                            text_columns=blank.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist() == ["6335", "6326", ""]
    # Strict on the work_df path agrees.
    again = make_loaded(blank_rows)
    work = sh.build_work_df(again, storage, config)
    applied, _ = sh.run_selected_rules_audited(work, rules, again, config)
    assert applied == 2
    assert work[NEW_ACCOUNT_COL].tolist() == ["6335", "6326", ""]


def test_name_first_matching_via_match_row(rules):
    rules.create_rule("Harbor Plumbing, repair call", "6326", priority=10,
                      fields=["Name", "Memo"])
    rules.create_rule("Golden Gate Electric, lighting repair", "6335",
                      priority=10, fields=["Name", "Memo"])
    row = pd.Series({"Name": "Golden Gate Electric",
                     "Memo": "repair call after storm", "New Account": ""})
    match = rules.match_row("golden gate electric | repair call after storm",
                            row=row)
    assert match is not None and match.account_code == "6335"
    memo_only = pd.Series({"Name": "", "Memo": "emergency repair call",
                           "New Account": ""})
    match = rules.match_row("emergency repair call", row=memo_only)
    assert match is not None and match.account_code == "6326"


# --------------------------------------------------------------------------- #
# Test 4 — Notes never feed keywords and are never an account field
# --------------------------------------------------------------------------- #
def test_notes_column_with_account_number_is_ignored(make_loaded, rules,
                                                     storage, config):
    loaded = make_loaded([
        {"Name": "Cook Multimedia", "Memo": "video package",
         "Notes": "4122 see coach", "New Account": "6622"},
        {"Name": "Cook Multimedia", "Memo": "video package 2",
         "Notes": "4122", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    cols = sh.mining_columns(work, loaded, config)
    assert "Notes" not in cols
    rules.ingest_existing_new_account_as_rules(work, source_text_cols=cols)
    for rule in rules.list_rules():
        assert rule.account_code == "6622"
        assert "notes" not in [f.lower() for f in rule.fields]
        for phrase in rule.match_phrases():
            assert "4122" not in phrase
            assert "coach" not in phrase.lower()
    # Candidate mining and per-cell suggestions never read Notes either.
    for cand in sh.candidate_rules(work, loaded, rules, config=config):
        assert "coach" not in cand["keyword"]
    from_notes = sh.suggest_keyword_from_cell(work, 0, "Notes", loaded, config)
    assert "coach" not in from_notes and "4122" not in from_notes
    assert from_notes == "cook multimedia"        # fell back to Name
    # The blank twin codes to 6622 (the seed), never to the number in Notes.
    applied, _ = sh.run_selected_rules_audited(work, rules, loaded, config)
    assert applied == 1
    assert work[NEW_ACCOUNT_COL].tolist() == ["6622", "6622"]


def test_ingest_falls_back_to_description_when_no_name_or_memo(make_loaded,
                                                               rules):
    loaded = make_loaded([
        {"Description": "Cred Hub screening", "New Account": "6618S"},
        {"Description": "Staples order", "New Account": "6200"},
    ])
    assert rules.ingest_existing_new_account_as_rules(loaded.df) == 2
    by_code = {r.account_code: r for r in rules.list_rules()}
    assert by_code["6618S"].match_phrases() == ["Cred Hub screening"]
    assert by_code["6618S"].fields == ["Description"]


def test_manager_helpers_identify_name_and_memo_columns():
    name_col, memo_col = RulesManager.resolve_name_memo_columns(
        ["Date", "Payee", "Memo/Description", "Amount", "Notes"])
    assert (name_col, memo_col) == ("Payee", "Memo/Description")
    assert RulesManager.resolve_name_memo_columns(["Date", "Amount"]) == (
        None, None)
