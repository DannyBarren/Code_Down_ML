"""Production-readiness tests for the rule-driven "New Account" workflow.

These lock in the client's non-negotiable guarantees:

* ``"New Account"`` is the one and only target column rules read from / write to.
* Rules work on a brand-new sheet with **zero** prior seeds, history or learning.
* A rule run **never overwrites** a row that already has a ``New Account`` value.
* Pre-filled rows can be ingested as high-priority exact-match rules.
* Every run is fully auditable (protected / filled / left-blank).
* One rule can carry several comma-separated phrases and several fields.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.schemas import RuleRunResult
from src.data_loader import NEW_ACCOUNT_COL, load_dataframe, summarize_upload
from src.fill_down_engine import FillDownEngine
from src.rules_manager import get_rule_application_summary


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _sample_records():
    """Realistic export: some rows already coded, some blank."""
    return [
        # Pre-filled (must be preserved exactly).
        {"Description": "CRED HUB RESIDENT SCREENING", "Payee": "Cred Hub",
         "New Account": "6618S"},
        {"Description": "STAPLES STORE #123 OFFICE", "Payee": "Staples",
         "New Account": "6200"},
        # Blank rows that should be filled by rules.
        {"Description": "Cred Hub monthly screening fee", "Payee": "Cred Hub",
         "New Account": ""},
        {"Description": "Payment to Staples for paper", "Payee": "Staples",
         "New Account": ""},
        # Blank row that no rule covers -> stays blank.
        {"Description": "Acme Landscaping quarterly", "Payee": "Acme",
         "New Account": ""},
    ]


def _engine(config, rules, storage) -> FillDownEngine:
    return FillDownEngine(config, rules,
                          learned_lookup=storage.get_learned_lookup())


# --------------------------------------------------------------------------- #
# Core guarantees
# --------------------------------------------------------------------------- #
def test_new_account_is_the_target_column(make_loaded):
    loaded = make_loaded(_sample_records())
    assert loaded.new_account_col == NEW_ACCOUNT_COL
    assert NEW_ACCOUNT_COL in loaded.df.columns


def test_summarize_upload_counts(make_loaded):
    loaded = make_loaded(_sample_records())
    stats = summarize_upload(loaded.df)
    assert stats == {"total": 5, "prefilled": 2, "blank": 3}


def test_close_variant_header_normalized_to_new_account(config):
    import io
    df = pd.DataFrame([
        {"Description": "Cred Hub fee", "New COA": "6618S"},
        {"Description": "Blank one", "New COA": ""},
    ])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    from src.data_loader import load_dataframe
    loaded = load_dataframe(buf.getvalue(), config, source_name="v.csv")
    assert NEW_ACCOUNT_COL in loaded.df.columns
    assert "New COA" not in loaded.df.columns


def test_missing_new_account_column_is_created(config):
    import io
    df = pd.DataFrame([{"Description": "Cred Hub fee"}])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    from src.data_loader import load_dataframe
    loaded = load_dataframe(buf.getvalue(), config, source_name="v.csv")
    assert NEW_ACCOUNT_COL in loaded.df.columns
    assert loaded.df[NEW_ACCOUNT_COL].tolist() == [""]


def test_rule_run_fills_blanks_and_protects_existing(make_loaded, rules, storage,
                                                     config):
    """The headline scenario: zero history, rules fill only the right blanks."""
    loaded = make_loaded(_sample_records())
    rules.create_rule("Cred Hub", "6618S", match_type="contains")
    rules.create_rule("Staples", "6200", match_type="contains")

    df, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)

    na = df[NEW_ACCOUNT_COL].tolist()
    # Pre-filled rows unchanged.
    assert na[0] == "6618S"
    assert na[1] == "6200"
    # Matching blanks filled.
    assert na[2] == "6618S"
    assert na[3] == "6200"
    # Non-matching blank stays blank.
    assert na[4] == ""

    # Audit is complete and accurate.
    assert len(results) == 5
    by_status = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)
    assert len(by_status["protected_existing"]) == 2
    assert len(by_status["keyword_rule"]) == 2
    assert len(by_status["no_rule_match"]) == 1
    filled = {r.row_index: r for r in by_status["keyword_rule"]}
    assert filled[2].proposed_value == "6618S"
    assert filled[2].matched_pattern.startswith("contains")


def test_summary_metrics(make_loaded, rules, storage, config):
    loaded = make_loaded(_sample_records())
    rules.create_rule("Cred Hub", "6618S")
    rules.create_rule("Staples", "6200")
    _, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    summary = get_rule_application_summary(results)
    assert summary["protected_existing"] == 2
    assert summary["filled_by_rules"] == 2
    assert summary["left_blank"] == 1
    assert summary["success_rate"] == 2 / 3


def test_never_overwrites_existing_even_if_a_rule_matches(make_loaded, rules,
                                                          storage, config):
    """A rule that would match a coded row must not change it."""
    loaded = make_loaded(_sample_records())
    # This rule would match row 0 (Cred Hub) but it's already 6618S.
    rules.create_rule("Cred Hub", "9999", match_type="contains")
    df, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist()[0] == "6618S"  # untouched
    protected = [r for r in results if r.status == "protected_existing"]
    assert any(r.original_value == "6618S" for r in protected)


def test_works_with_zero_prior_history(make_loaded, rules, storage, config):
    """No seeds needed by the engine, no learned data, no models — pure rules."""
    records = [
        {"Description": "Cred Hub fee", "New Account": ""},
        {"Description": "nothing relevant", "New Account": ""},
    ]
    loaded = make_loaded(records)
    assert not rules.list_rules()  # brand new
    rules.create_rule("Cred Hub", "6618S")
    df, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist() == ["6618S", ""]


# --------------------------------------------------------------------------- #
# Multi-phrase / multi-field / new match types
# --------------------------------------------------------------------------- #
def test_comma_separated_phrases_match_any(make_loaded, rules, storage, config):
    records = [
        {"Description": "credit hub screening", "New Account": ""},
        {"Description": "cred-hub inc fee", "New Account": ""},
        {"Description": "unrelated vendor", "New Account": ""},
    ]
    loaded = make_loaded(records)
    rules.create_rule("cred hub, credit hub, cred-hub", "6618S",
                      match_type="contains")
    df, _ = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist() == ["6618S", "6618S", ""]


def test_starts_with_and_ends_with(rules):
    from models.schemas import KeywordRule
    starts = KeywordRule(keyword="cred hub", account_code="6618S",
                         match_type="starts_with")
    ends = KeywordRule(keyword="fee", account_code="6200",
                       match_type="ends_with")
    assert rules.rule_matches(starts, "Cred Hub monthly screening")
    assert not rules.rule_matches(starts, "monthly cred hub screening")
    assert rules.rule_matches(ends, "monthly screening fee")
    assert not rules.rule_matches(ends, "fee schedule monthly")


def test_field_scoped_rule_only_matches_named_columns(make_loaded, rules,
                                                      storage, config):
    records = [
        {"Payee": "Cred Hub", "Description": "screening", "New Account": ""},
        {"Payee": "Other", "Description": "cred hub in memo", "New Account": ""},
    ]
    loaded = make_loaded(records)
    rules.create_rule("cred hub", "6618S", match_type="contains",
                      fields=["Payee"])
    df, _ = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist() == ["6618S", ""]


# --------------------------------------------------------------------------- #
# Ingest pre-filled rows as rules
# --------------------------------------------------------------------------- #
def test_ingest_prefilled_creates_exact_high_priority_rules(make_loaded, rules):
    loaded = make_loaded(_sample_records())
    created = rules.ingest_existing_new_account_as_rules(
        loaded.df, source_text_col="Description")
    assert created == 2  # two coded rows
    ingested = rules.list_rules()
    assert all(r.match_type == "exact" for r in ingested)
    assert all(r.priority == 10 for r in ingested)
    codes = {r.account_code for r in ingested}
    assert {"6618S", "6200"} <= codes


def test_ingested_rules_are_reusable_on_future_files(make_loaded, rules, storage,
                                                     config):
    """Ingest from one file, then those exact rows code automatically later."""
    first = make_loaded(_sample_records())
    rules.ingest_existing_new_account_as_rules(first.df,
                                               source_text_col="Description")
    # A new file with the *same* description text, but blank New Account.
    second = make_loaded([
        {"Description": "CRED HUB RESIDENT SCREENING", "New Account": ""},
        {"Description": "brand new vendor", "New Account": ""},
    ])
    df, _ = _engine(config, rules, storage).run_selected_rules(
        second.df, text_columns=second.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist() == ["6618S", ""]


def test_ingest_is_idempotent(make_loaded, rules):
    loaded = make_loaded(_sample_records())
    a = rules.ingest_existing_new_account_as_rules(loaded.df)
    b = rules.ingest_existing_new_account_as_rules(loaded.df)
    assert a == 2 and b == 0


# --------------------------------------------------------------------------- #
# apply_rules_to_dataframe directly + robustness
# --------------------------------------------------------------------------- #
def test_apply_rules_to_dataframe_returns_df_and_results(rules):
    rules.create_rule("Cred Hub", "6618S")
    df = pd.DataFrame([
        {"Description": "Cred Hub fee", "New Account": ""},
        {"Description": "already done", "New Account": "1000"},
        {"Description": "nothing", "New Account": ""},
    ])
    out, results = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6618S", "1000", ""]
    assert isinstance(results[0], RuleRunResult)
    assert results[1].status == "protected_existing"


def test_priority_orders_rules(rules):
    # A broad low-priority "contains" and a specific high-priority exact.
    rules.create_rule("hub", "1111", match_type="contains", priority=100)
    rules.create_rule("Cred Hub fee", "2222", match_type="exact", priority=10)
    df = pd.DataFrame([{"Description": "Cred Hub fee", "New Account": ""}])
    out, results = rules.apply_rules_to_dataframe(df)
    # The high-priority exact rule wins.
    assert out[NEW_ACCOUNT_COL].tolist() == ["2222"]


def test_bad_regex_rule_never_crashes(rules):
    rules.create_rule("(unclosed", "6618S", match_type="regex")
    df = pd.DataFrame([{"Description": "anything", "New Account": ""}])
    out, results = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == [""]  # no crash, no match


def test_string_dtype_columns_are_handled(rules):
    rules.create_rule("Cred Hub", "6618S")
    df = pd.DataFrame({
        "Description": pd.array(["Cred Hub fee", "other"], dtype="string"),
        "New Account": pd.array(["", ""], dtype="string"),
    })
    out, _ = rules.apply_rules_to_dataframe(df)
    assert list(out[NEW_ACCOUNT_COL]) == ["6618S", ""]


# --------------------------------------------------------------------------- #
# Field scoping robustness + audit visibility (the Chrysalis bug)
# --------------------------------------------------------------------------- #
def test_field_scoping_tolerates_messy_column_names(rules):
    """A rule scoped to 'memo / description' still matches column 'Memo/Description'."""
    rules.create_rule("Rent Signage", "6622", match_type="contains",
                      fields=["memo / description"])
    df = pd.DataFrame([
        {"Name": "ACME", "Memo/Description": "Rent Signage install",
         "New Account": ""},
        {"Name": "Other", "Memo/Description": "nothing", "New Account": ""},
    ])
    out, _ = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6622", ""]


def test_field_scoping_only_uses_named_columns(rules):
    """'Cook Multimedia' scoped to Name must not match when it's only in Memo."""
    rules.create_rule("Cook Multimedia", "6622", match_type="contains",
                      fields=["Name"])
    df = pd.DataFrame([
        {"Name": "Cook Multimedia", "Memo/Description": "x", "New Account": ""},
        {"Name": "Other", "Memo/Description": "cook multimedia here",
         "New Account": ""},
    ])
    out, _ = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6622", ""]


def test_protected_row_reports_matched_rule(rules):
    """When a rule matches an already-coded row, the audit proves it fired."""
    rules.create_rule("Cook Multimedia", "9999", match_type="contains",
                      fields=["Name"])
    df = pd.DataFrame([
        {"Name": "Cook Multimedia", "New Account": "6622"},  # already coded
    ])
    out, results = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6622"]  # never overwritten
    r = results[0]
    assert r.status == "protected_existing"
    assert r.matched_rule_name == "Cook Multimedia"
    summary = get_rule_application_summary(results)
    assert summary["protected_but_matched"] == 1
    assert summary["filled_by_rules"] == 0


def test_chrysalis_exact_rules_scenario(config, rules, storage):
    """Reproduce the client's exact rules on a sanitized fixture.

    ``tests/fixtures/chrysalis_like.csv`` is synthetic (fictional vendors) but
    keeps the real file's invariants: 187 pre-filled rows, and enough Cook
    Multimedia / Rent Signage rows that the rules provably fire on protected
    rows. All rows containing the phrases are already coded, so nothing is
    filled — but the rules must be shown to *match* those protected rows, and
    appended blank rows with the same phrases must fill correctly.
    """
    csv = (Path(__file__).resolve().parent / "fixtures" / "chrysalis_like.csv")
    if not csv.exists():
        pytest.skip("sanitized chrysalis_like fixture missing from checkout.")

    loaded = load_dataframe(str(csv), config, source_name="chrysalis_like.csv")
    # Append blank rows that resemble future transactions.
    loaded.df = pd.concat([loaded.df, pd.DataFrame([
        {"Name": "Cook Multimedia", "Memo/Description": "video",
         "New Account": ""},
        {"Name": "ACME", "Memo/Description": "Rent Signage banner",
         "New Account": ""},
        {"Name": "Zeta Multimedia", "Memo/Description": "promo",
         "New Account": ""},  # fuzzy target for the 'Mutlimedia' typo
    ])], ignore_index=True)

    # The exact three rules from the screenshot.
    rules.create_rule("Cook Multimedia", "6622", match_type="contains",
                      fields=["Name"])
    rules.create_rule("Rent Signage", "6622", match_type="contains",
                      fields=["Memo/Description"])
    rules.create_rule("Rent Signage, Mutlimedia", "6622", match_type="fuzzy",
                      fields=["Memo/Description", "Name"])

    df, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    summary = get_rule_application_summary(results)

    # Protection of the 187 pre-filled rows still holds.
    assert summary["protected_existing"] == 187
    # The rules provably fire on already-coded rows.
    assert summary["protected_but_matched"] >= 6
    # The three appended blank rows are filled by the rules.
    tail = df[NEW_ACCOUNT_COL].tolist()[-3:]
    assert tail == ["6622", "6622", "6622"]


# --------------------------------------------------------------------------- #
# Row-wide (full-row concatenation) default matching
# --------------------------------------------------------------------------- #
def test_default_matching_searches_whole_row(rules):
    """A rule with no field restriction matches text in ANY column."""
    rules.create_rule("Rent Signage", "6622")  # no fields -> row-wide default
    df = pd.DataFrame([
        {"Name": "ACME", "Memo/Description": "Rent Signage install",
         "New Account": ""},
        {"Name": "Other", "Memo/Description": "nothing", "New Account": ""},
    ])
    out, _ = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6622", ""]


def test_full_row_concatenation_matches_across_columns(rules):
    """The whole row is one string, so a phrase spanning two columns matches."""
    rules.create_rule("widget order", "7000")  # "widget" | "order" across cols
    df = pd.DataFrame([
        {"A": "super widget", "B": "order 5", "New Account": ""},
        {"A": "super", "B": "gadget 5", "New Account": ""},
    ])
    out, _ = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["7000", ""]


def test_comma_phrases_match_on_full_row(rules):
    """Comma-separated phrases (OR) work against the whole concatenated row."""
    rules.create_rule("yard sign, rent signage", "6622")
    df = pd.DataFrame([
        {"Name": "ACME", "Memo/Description": "install yard sign",
         "New Account": ""},
        {"Name": "BETA", "Memo/Description": "Rent Signage banner",
         "New Account": ""},
        {"Name": "GAMMA", "Memo/Description": "nothing", "New Account": ""},
    ])
    out, _ = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6622", "6622", ""]


def test_advanced_field_restriction_still_works(rules):
    """Backward compat: an explicit field list limits matching to those columns."""
    rules.create_rule("cook multimedia", "6622", match_type="contains",
                      fields=["Name"])
    df = pd.DataFrame([
        {"Name": "Cook Multimedia", "Memo/Description": "x", "New Account": ""},
        {"Name": "Other", "Memo/Description": "cook multimedia here",
         "New Account": ""},
    ])
    out, _ = rules.apply_rules_to_dataframe(df)
    assert out[NEW_ACCOUNT_COL].tolist() == ["6622", ""]


# --------------------------------------------------------------------------- #
# Strict (pure) rule mode: no cross-account leakage, no semantic propagation
# --------------------------------------------------------------------------- #
def test_pure_rule_run_does_not_leak_to_other_accounts(make_loaded, rules,
                                                       storage, config):
    """Strict rules fill ONLY their own target code — never a look-alike's."""
    records = [
        # A pre-coded row for 6800, unrelated to any rule.
        {"Description": "Acme Landscaping quarterly", "New Account": "6800"},
        # A blank row IDENTICAL to the 6800 seed — semantic would grab it, but
        # strict rules must leave it blank (no rule covers it).
        {"Description": "Acme Landscaping quarterly", "New Account": ""},
        # A blank row the rule genuinely matches.
        {"Description": "Cred Hub screening fee", "New Account": ""},
        {"Description": "nothing relevant", "New Account": ""},
    ]
    loaded = make_loaded(records)
    rules.create_rule("Cred Hub", "6618S")

    df, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)

    na = df[NEW_ACCOUNT_COL].tolist()
    assert na == ["6800", "", "6618S", ""]  # no 6800 leakage onto the blank
    # Nothing was filled with a code that isn't a rule's target.
    filled_codes = {r.proposed_value for r in results if r.status == "keyword_rule"}
    assert filled_codes == {"6618S"}


def test_pure_rules_and_full_run_behave_differently(make_loaded, rules, storage,
                                                    config):
    """Strict rules ≠ Full Intelligent Run: only the latter spreads by similarity."""
    rules.create_rule("Cred Hub", "6618S")
    recs = [
        {"Description": "Acme Landscaping quarterly", "New Account": "6800"},
        {"Description": "Acme Landscaping quarterly", "New Account": ""},
        {"Description": "Cred Hub screening fee", "New Account": ""},
    ]

    # Strict: the 6800 look-alike stays blank.
    pure_loaded = make_loaded(recs)
    pure_df, _ = _engine(config, rules, storage).run_selected_rules(
        pure_loaded.df, text_columns=pure_loaded.text_columns)
    assert pure_df[NEW_ACCOUNT_COL].tolist() == ["6800", "", "6618S"]

    # Full: similarity spreads the 6800 seed to the identical blank row.
    full_loaded = make_loaded(recs)
    engine = FillDownEngine(config, rules,
                            learned_lookup=storage.get_learned_lookup(),
                            model_manager=None, mode="similarity_only")
    result = engine.run(full_loaded)
    full_na = result.df[NEW_ACCOUNT_COL].tolist()
    assert full_na[1] == "6800"      # spread by similarity (broad matching)
    assert full_na[2] == "6618S"     # the rule still applies


def test_pure_rule_audit_reports_no_semantic(make_loaded, rules, storage, config):
    from src import spreadsheet_helpers as sh

    loaded = make_loaded([
        {"Description": "Cred Hub fee", "New Account": ""},
        {"Description": "Staples order", "New Account": ""},
    ])
    rules.create_rule("Cred Hub", "6618S")
    _, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)

    audit = sh.pure_rule_audit(results, config)
    assert audit["mode"] == "pure"
    assert audit["semantic_filled"] == 0
    assert audit["semantic_results"] == []
    assert audit["keyword_filled"] == 1
    assert audit["left_blank"] == 1


# --------------------------------------------------------------------------- #
# Human-reviewed ingestion (no silent auto-rules)
# --------------------------------------------------------------------------- #
def test_prefilled_ingest_candidates_show_data_without_creating(make_loaded,
                                                                rules):
    loaded = make_loaded(_sample_records())
    cands = rules.prefilled_ingest_candidates(loaded.df)
    codes = {c["code"] for c in cands}
    assert {"6618S", "6200"} <= codes
    for c in cands:
        assert c["count"] >= 1
        assert isinstance(c["samples"], list) and c["samples"]
    # Crucially, inspecting candidates must NOT create any rules.
    assert rules.list_rules() == []


def test_reviewed_ingest_uses_user_phrases(make_loaded, rules, storage, config):
    """Simulate the UI: the user types phrases, then rules are created from them."""
    loaded = make_loaded(_sample_records())
    cands = rules.prefilled_ingest_candidates(loaded.df)
    # The user types their own keywords per code (comma-separated OR logic).
    user_input = {"6618S": "cred hub, credit hub", "6200": "staples"}
    for c in cands:
        phrases = user_input.get(c["code"], "").strip()
        if phrases:
            rules.create_rule(phrases, c["code"], match_type="contains",
                              priority=10)
    assert len(rules.list_rules()) == 2

    second = make_loaded([
        {"Description": "credit hub monthly", "New Account": ""},
        {"Description": "Staples paper order", "New Account": ""},
        {"Description": "unrelated vendor", "New Account": ""},
    ])
    df, _ = _engine(config, rules, storage).run_selected_rules(
        second.df, text_columns=second.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist() == ["6618S", "6200", ""]


def test_chrysalis_row_wide_rules_fill_future_rows(config, rules, storage):
    """Row-wide rules (no field selection) code the appended future rows."""
    csv = (Path(__file__).resolve().parent / "fixtures" / "chrysalis_like.csv")
    if not csv.exists():
        pytest.skip("sanitized chrysalis_like fixture missing from checkout.")

    loaded = load_dataframe(str(csv), config, source_name="chrysalis_like.csv")
    loaded.df = pd.concat([loaded.df, pd.DataFrame([
        {"Name": "Cook Multimedia", "Memo/Description": "video",
         "New Account": ""},
        {"Name": "ACME", "Memo/Description": "Rent Signage banner",
         "New Account": ""},
    ])], ignore_index=True)

    # No fields — the row-wide default handles column placement automatically.
    rules.create_rule("Cook Multimedia", "6622")
    rules.create_rule("Rent Signage", "6622")

    df, results = _engine(config, rules, storage).run_selected_rules(
        loaded.df, text_columns=loaded.text_columns)
    summary = get_rule_application_summary(results)
    assert summary["protected_existing"] == 187
    assert df[NEW_ACCOUNT_COL].tolist()[-2:] == ["6622", "6622"]


def test_na_values_are_treated_as_blank_and_filled(rules):
    """A real StringDtype export with <NA> blanks is filled, not skipped."""
    rules.create_rule("Cred Hub", "6618S")
    df = pd.DataFrame({
        "Description": pd.array(["Cred Hub fee", "Cred Hub fee"], dtype="string"),
        "New Account": pd.array(["6200", pd.NA], dtype="string"),
    })
    out, results = rules.apply_rules_to_dataframe(df)
    assert str(out[NEW_ACCOUNT_COL].iloc[0]) == "6200"          # protected
    assert str(out[NEW_ACCOUNT_COL].iloc[1]) == "6618S"         # NA filled
    assert results[0].status == "protected_existing"
    assert results[1].status == "keyword_rule"
