"""Account-Level Knowledge Base: profiles that learn from rules + matches.

Covers canonicalisation, keyword extraction, profile upsert/merge, learning on
rule create/update, reinforcement from runs, per-client isolation, rationale
enrichment, and the refresh-from-rules backfill.
"""

from __future__ import annotations

import pandas as pd

from src import spreadsheet_helpers as sh
from src.fill_down_engine import FillDownEngine
from src.rules_manager import canonicalize_account, extract_keywords


# --------------------------------------------------------------------------- #
# Canonicalisation + keyword extraction
# --------------------------------------------------------------------------- #
def test_canonicalize_account_handles_suffix_variants():
    for variant in ("6618S", "6618 s", "6618-S", "6618_s", "6618.0S", "6618s"):
        assert canonicalize_account(variant) == "6618S"
    assert canonicalize_account("6618") == "6618"
    assert canonicalize_account("6618.0") == "6618"


def test_extract_keywords_splits_commas_and_tokens():
    kws = extract_keywords("Cred Hub, Rent Signage")
    assert "cred hub" in kws            # phrase preserved (normalised)
    assert "rent signage" in kws
    assert "cred" in kws and "hub" in kws  # significant tokens too
    # stop-words / short tokens dropped, no duplicates
    kws2 = extract_keywords("The Home Depot LLC")
    assert "the" not in kws2 and "llc" not in kws2
    assert len(kws2) == len(set(kws2))


# --------------------------------------------------------------------------- #
# Learning on rule create / update
# --------------------------------------------------------------------------- #
def test_create_rule_builds_account_profile(rules):
    rules.create_rule("Cred Hub", "6618S", notes="Resident screening.")
    prof = rules.get_account_profile("6618S")
    assert prof, "profile should be created for the rule's account"
    assert prof["account_code"] == "6618S"
    assert prof["base_account"] == "6618"
    assert "cred hub" in prof["keywords"]
    assert prof["notes"] == "Resident screening."
    # canonicalisation: a messy variant resolves to the same profile
    assert rules.get_account_profile("6618 s")["account_code"] == "6618S"


def test_create_rule_with_variant_code_merges_same_profile(rules):
    rules.create_rule("Cred Hub", "6618S")
    rules.create_rule("Screening", "6618 s")   # variant -> same canonical account
    prof = rules.get_account_profile("6618S")
    assert "cred hub" in prof["keywords"]
    assert "screening" in prof["keywords"]


def test_upsert_merges_keywords_and_increments_usage(rules):
    rules.upsert_account_profile("7000", new_keywords=["alpha"], sample_text="a")
    rules.upsert_account_profile("7000", new_keywords=["beta", "alpha"],
                                 sample_text="b", usage_increment=3)
    prof = rules.get_account_profile("7000")
    assert prof["keywords"] == ["alpha", "beta"]      # deduped, order preserved
    assert prof["usage_count"] == 3
    assert set(prof["sample_texts"]) == {"a", "b"}


def test_update_rule_syncs_new_keywords(rules):
    rule = rules.create_rule("Cred Hub", "6618S")
    rule.keyword = "Cred Hub, credit hub"
    rules.update_rule(rule)
    prof = rules.get_account_profile("6618S")
    assert "credit hub" in prof["keywords"]


# --------------------------------------------------------------------------- #
# Per-client isolation
# --------------------------------------------------------------------------- #
def test_per_client_profiles_are_isolated_from_shared(rules):
    rules.create_rule("Cred Hub", "6618S")                       # shared/global
    rules.create_rule("Acme Screening", "6618S", client_id="Acme")

    shared = rules.get_account_profile("6618S")                  # client_id=None
    acme = rules.get_account_profile("6618S", client_id="Acme")

    assert "cred hub" in shared["keywords"]
    assert "acme screening" in acme["keywords"]
    # isolation both ways
    assert "acme screening" not in shared["keywords"]
    assert "cred hub" not in acme["keywords"]


# --------------------------------------------------------------------------- #
# Reinforcement + rationale enrichment on a real run
# --------------------------------------------------------------------------- #
def _engine(config, rules):
    return FillDownEngine(config, rules, learned_lookup={})


def test_run_reinforces_usage_and_enriches_rationale(make_loaded, rules, config):
    rules.create_rule("Cred Hub", "6618S")
    loaded = make_loaded([
        {"Name": "Cred Hub", "Memo/Description": "resident screening",
         "New Account": ""},
        {"Name": "Cred Hub LLC", "Memo/Description": "screening", "New Account": ""},
        {"Name": "Unrelated", "Memo/Description": "misc", "New Account": ""},
    ])
    df, results = _engine(config, rules).run_selected_rules(loaded.df)

    filled = [r for r in results if r.status == "keyword_rule"]
    assert len(filled) == 2
    # rationale enriched with learned account patterns
    assert any("learned patterns for 6618S" in r.rationale for r in filled)

    prof = rules.get_account_profile("6618S")
    assert prof["usage_count"] >= 2          # both matched rows reinforced
    assert prof["sample_texts"]              # captured real transaction samples


def test_audited_run_reinforces_profile(make_loaded, rules, config, storage):
    rules.create_rule("gift, bonus", "6510")
    loaded = make_loaded([
        {"Name": "Gift Shop", "Memo/Description": "employee gift", "New Account": ""},
        {"Name": "Bonus Co", "Memo/Description": "annual bonus", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    n, results = sh.run_selected_rules_audited(work, rules, loaded, config)
    assert n == 2
    prof = rules.get_account_profile("6510")
    assert prof["usage_count"] >= 2


# --------------------------------------------------------------------------- #
# Refresh backfill + protection guarantees
# --------------------------------------------------------------------------- #
def test_refresh_learning_rebuilds_from_all_rules(rules, storage):
    # Create rules directly via storage to simulate imported rules with no profiles.
    from models.schemas import KeywordRule
    storage.add_rule(KeywordRule(keyword="Staples", account_code="6200"))
    storage.add_rule(KeywordRule(keyword="Cred Hub", account_code="6618S"))
    assert rules.list_account_profiles() == []      # none learned yet

    touched = rules.refresh_learning_from_rules()
    assert touched == 2
    assert rules.get_account_profile("6200")["keywords"]
    assert rules.get_account_profile("6618S")["keywords"]


def test_ingest_prefilled_populates_profiles(make_loaded, rules):
    loaded = make_loaded([
        {"Description": "Cred Hub screening", "New Account": "6618S"},
        {"Description": "Staples order", "New Account": "6200"},
    ])
    created = rules.ingest_existing_new_account_as_rules(loaded.df)
    assert created == 2
    assert rules.get_account_profile("6618S")["keywords"]
    assert rules.get_account_profile("6200")["keywords"]


def test_reset_all_clears_account_profiles(rules, storage):
    rules.create_rule("Cred Hub", "6618S")
    assert rules.list_account_profiles()
    storage.reset_all()
    assert rules.list_account_profiles() == []
