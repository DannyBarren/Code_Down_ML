"""Tests for the fill-down engine: propagation, precedence, confidence."""

from models.schemas import FillAction, FillSource
from src.fill_down_engine import FillDownEngine


def _run(config, rules, loaded, learned=None):
    engine = FillDownEngine(config, rules, learned_lookup=learned or {})
    return engine.run(loaded)


def test_seed_is_kept(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "screening fee", "New Account": "6618S"},
    ])
    res = _run(config, rules, loaded)
    r0 = res.results[0]
    assert r0.action == FillAction.KEPT_SEED
    assert r0.source == FillSource.SEED
    assert res.df.iloc[0][loaded.new_account_col] == "6618S"


def test_similarity_propagation(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "screening fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "screening fee", "New Account": ""},
        {"Name": "Office Depot", "Description": "paper", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    # Row 1 is identical to the seed -> should be filled with 6618S.
    assert res.df.iloc[1][loaded.new_account_col] == "6618S"
    # Row 2 is unrelated and has no seed -> left blank.
    assert res.df.iloc[2][loaded.new_account_col] == ""


def test_rule_takes_precedence(config, rules, make_loaded):
    rules.add_rule("office depot", "6310S")
    loaded = make_loaded([
        {"Name": "Office Depot", "Description": "paper", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    r0 = res.results[0]
    assert r0.source == FillSource.RULE
    assert res.df.iloc[0][loaded.new_account_col] == "6310S"
    assert r0.confidence >= 0.95


def test_learned_mapping_applied(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "Weird Vendor", "Description": "special", "New Account": ""},
    ])
    signature = loaded.df["_sim_text"].iloc[0]
    res = _run(config, rules, loaded, learned={signature: "9999S"})
    assert res.results[0].source == FillSource.LEARNED
    assert res.df.iloc[0][loaded.new_account_col] == "9999S"


def test_no_seeds_no_rules(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "A", "Description": "thing one", "New Account": ""},
        {"Name": "B", "Description": "thing two", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    assert res.summary.seeds == 0
    assert res.summary.total_filled == 0
    assert all(r.action == FillAction.NO_MATCH for r in res.results)


def test_summary_counts_consistent(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
        {"Name": "Random", "Description": "noise", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    s = res.summary
    total = s.seeds + s.auto_filled + s.filled_review + s.needs_review + s.no_match
    assert total == s.total_rows == len(loaded.df)
