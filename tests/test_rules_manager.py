"""Tests for keyword rules: CRUD, matching modes, code normalisation."""

import pandas as pd


def test_add_and_list(rules):
    r = rules.add_rule("Cred Hub", "6618s")
    assert r.id is not None
    assert r.account_code == "6618S"  # normalised
    assert len(rules.list_rules()) == 1


def test_contains_match(rules):
    rules.add_rule("cred hub", "6618S")
    match = rules.match_row("payment to cred hub monthly", row=None)
    assert match is not None
    assert match.account_code == "6618S"


def test_no_match(rules):
    rules.add_rule("cred hub", "6618S")
    assert rules.match_row("office depot supplies") is None


def test_exact_match(rules):
    rules.add_rule("appfolio", "6520S", match_type="exact")
    assert rules.match_row("appfolio") is not None
    assert rules.match_row("appfolio monthly") is None


def test_regex_match(rules):
    rules.add_rule(r"home\s*depot", "6730S", match_type="regex")
    assert rules.match_row("the home depot #455") is not None
    assert rules.match_row("homedepot online") is not None


def test_field_scoped_match(rules):
    rules.add_rule("statefarm", "6410S", fields=["Name"])
    row_hit = pd.Series({"Name": "StateFarm Insurance", "Memo": "x"})
    row_miss = pd.Series({"Name": "Other", "Memo": "statefarm in memo"})
    assert rules.match_row("statefarm insurance", row=row_hit) is not None
    assert rules.match_row("other statefarm", row=row_miss) is None


def test_update_and_delete(rules):
    r = rules.add_rule("x", "1000")
    r.account_code = "2000s"
    rules.update_rule(r)
    assert rules.list_rules()[0].account_code == "2000S"
    rules.delete_rule(r.id)
    assert rules.list_rules() == []


def test_delete_rules_bulk(rules):
    a = rules.add_rule("a", "1000")
    b = rules.add_rule("b", "2000")
    c = rules.add_rule("c", "3000")
    removed = rules.delete_rules([a.id, c.id])
    assert removed == 2
    remaining = rules.list_rules()
    assert [r.id for r in remaining] == [b.id]


def test_delete_rules_empty_is_noop(rules):
    rules.add_rule("a", "1000")
    assert rules.delete_rules([]) == 0
    assert len(rules.list_rules()) == 1


def test_clear_rules_removes_everything(rules):
    rules.add_rule("a", "1000")
    rules.add_rule("b", "2000")
    removed = rules.clear_rules()
    assert removed == 2
    assert rules.list_rules() == []
    # Clearing an empty table is safe.
    assert rules.clear_rules() == 0


def test_disabled_rule_skipped(rules):
    r = rules.add_rule("cred hub", "6618S")
    r.enabled = False
    rules.update_rule(r)
    assert rules.match_row("cred hub", rules=rules.list_rules(enabled_only=True)) is None


# --------------------------------------------------------------------------- #
# Robust matching (the production-rescue fixes)
# --------------------------------------------------------------------------- #
def test_contains_searches_all_row_columns(rules):
    """A vendor that lives only in a non-similarity column (Payee) still matches.

    ``combined_text`` deliberately omits the vendor; the engine must still find
    it by scanning the row's columns.
    """
    rules.add_rule("acme plumbing", "6300S")
    row = pd.Series({"Date": "2025-01-01", "Payee": "Acme Plumbing",
                     "Memo": "service call"})
    assert rules.match_row("service call", row=row) is not None


def test_multiword_matches_concatenated_source(rules):
    """A spaced keyword matches concatenated/punctuated source text."""
    rules.add_rule("Cred Hub", "6618S")
    assert rules.match_row("credhub screening", row=None) is not None
    assert rules.match_row("cred-hub inc fee", row=None) is not None


def test_fuzzy_match_tolerates_typo(rules):
    rules.add_rule("comcast", "6850S", match_type="fuzzy")
    assert rules.match_row("comcst business internet") is not None   # typo
    assert rules.match_row("verizon fios") is None


def test_case_sensitive_is_literal(rules):
    rules.add_rule("AppFolio", "6520S", case_sensitive=True)
    assert rules.match_row("paid AppFolio invoice") is not None
    assert rules.match_row("paid appfolio invoice") is None


def test_answer_column_not_searched(rules):
    """A rule must not match because its account code appears in New Account."""
    rules.add_rule("6618S", "9999")
    row = pd.Series({"Name": "Mystery Vendor", "New Account": "6618S"})
    # The keyword only appears in the answer column -> no match.
    assert rules.match_row("mystery vendor", row=row) is None
