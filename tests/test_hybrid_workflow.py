"""Tests for the hybrid "Run Selected Rules" workflow + confidence filtering.

The hybrid run:
  1. applies keyword rules to blank rows (deterministic, high confidence), then
  2. uses those rule fills + existing coded rows as seeds and spreads the codes
     to similar blank rows via semantic similarity (confidence from cosine),
  3. never overwriting a row that already has a value, and
  4. recording a clean audit (keyword vs semantic fills, protected, confidence
     buckets) for efficient review.
"""

from __future__ import annotations

from src import spreadsheet_helpers as sh
from src.fill_down_engine import FillDownEngine


def _hybrid_engine(config, rules, storage) -> FillDownEngine:
    """Rules + semantic similarity only (no ML) — matches the UI hybrid engine."""
    return FillDownEngine(
        config, rules,
        learned_lookup=storage.get_learned_lookup(),
        model_manager=None,
        mode="similarity_only",
    )


def _hybrid_records():
    return [
        # Already coded (must be preserved) — becomes a semantic seed.
        {"Description": "Acme Landscaping quarterly maintenance",
         "New Account": "6800"},
        # Identical text but blank -> no keyword hit, filled by SEMANTIC (cos=1).
        {"Description": "Acme Landscaping quarterly maintenance",
         "New Account": ""},
        # Blank, matches the keyword rule directly.
        {"Description": "Cred Hub resident screening fee", "New Account": ""},
        # Blank, unrelated -> stays blank.
        {"Description": "totally different widget order 999", "New Account": ""},
    ]


# --------------------------------------------------------------------------- #
# Hybrid run
# --------------------------------------------------------------------------- #
def test_hybrid_keyword_then_semantic(make_loaded, rules, storage, config):
    loaded = make_loaded(_hybrid_records())
    work = sh.build_work_df(loaded, storage, config)
    rules.create_rule("Cred Hub", "6618S", match_type="contains")

    n, audit = sh.run_rules_hybrid(
        work, rules, loaded, config, _hybrid_engine(config, rules, storage))

    na = work[loaded.new_account_col].tolist()
    # Existing coded row preserved; keyword + semantic fills applied; last blank.
    assert na == ["6800", "6800", "6618S", ""]

    assert audit["keyword_filled"] == 1
    assert audit["semantic_filled"] == 1
    assert audit["protected_existing"] == 1
    assert audit["left_blank"] == 1
    assert n == 2

    # Both fills are high confidence in this clean scenario.
    assert audit["conf_high"] == 2
    assert audit["conf_medium"] == 0
    assert audit["conf_low"] == 0

    # Provenance: the keyword-filled row reads "rules" (not "seed") at ~0.99,
    # and the semantic row is labelled a similarity match.
    assert work.at[2, sh.ENGINE_COL] == "rules"
    assert abs(float(work.at[2, sh.CONF_COL])
               - config.confidence.rule_match_confidence) < 1e-6
    assert work.at[1, sh.ENGINE_COL] == "similarity"


def test_hybrid_never_overwrites_but_reports_match(make_loaded, rules, storage,
                                                   config):
    loaded = make_loaded([
        {"Description": "Acme Landscaping quarterly", "New Account": "6800"},
        {"Description": "Acme Landscaping quarterly", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    rules.create_rule("Acme", "1111", match_type="contains")

    n, audit = sh.run_rules_hybrid(
        work, rules, loaded, config, _hybrid_engine(config, rules, storage))

    na = work[loaded.new_account_col].tolist()
    assert na[0] == "6800"                  # never overwritten
    assert na[1] == "1111"                  # blank filled by the keyword rule
    assert audit["keyword_filled"] == 1
    assert audit["protected_but_matched"] >= 1
    # The protected-but-matched row is captured for the audit table.
    assert audit["protected_results"]
    assert audit["protected_results"][0].matched_rule_name


def test_hybrid_returns_result_lists(make_loaded, rules, storage, config):
    loaded = make_loaded(_hybrid_records())
    work = sh.build_work_df(loaded, storage, config)
    rules.create_rule("Cred Hub", "6618S")
    _, audit = sh.run_rules_hybrid(
        work, rules, loaded, config, _hybrid_engine(config, rules, storage))

    # Keyword results are RuleRunResult objects; semantic are dicts w/ confidence.
    assert audit["keyword_results"][0].status == "keyword_rule"
    sem = audit["semantic_results"]
    assert sem and set(sem[0]) >= {"row_index", "code", "confidence", "engine"}
    assert 0.0 <= float(sem[0]["confidence"]) <= 1.0


# --------------------------------------------------------------------------- #
# Raw matched-records view
# --------------------------------------------------------------------------- #
def test_matched_records_view_shows_only_filled_rows_raw_columns(
        make_loaded, rules, storage, config):
    loaded = make_loaded([
        {"Name": "Gift Shop", "Memo/Description": "employee gift",
         "Amount": 50.0, "New Account": ""},
        {"Name": "Acme", "Memo/Description": "landscaping", "Amount": 12.0,
         "New Account": ""},
        {"Name": "Bonus Co", "Memo/Description": "annual bonus", "Amount": 99.0,
         "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    rules.create_rule("gift, bonus", "6510")

    _, results = sh.run_selected_rules_audited(work, rules, loaded, config)
    audit = sh.pure_rule_audit(results, config)

    raw = sh.matched_records_view(work, loaded, audit["filled_rows"])
    # Only the two matched rows (gift, bonus) — not the Acme row.
    assert len(raw) == 2
    assert set(raw["New Account"]) == {"6510"}
    # Original columns preserved; NO internal/meta columns leak in.
    assert "Amount" in raw.columns and "Name" in raw.columns
    assert not any(str(c).startswith("_") for c in raw.columns)
    assert "_confidence" not in raw.columns


def test_matched_records_view_empty_when_nothing_filled(make_loaded, rules,
                                                        storage, config):
    loaded = make_loaded([{"Name": "Acme", "New Account": ""}])
    work = sh.build_work_df(loaded, storage, config)
    raw = sh.matched_records_view(work, loaded, [])
    assert raw.empty
    assert not any(str(c).startswith("_") for c in raw.columns)


# --------------------------------------------------------------------------- #
# Reset / un-run automation
# --------------------------------------------------------------------------- #
def test_reset_automation_unruns_but_keeps_seeds_notes_rules(
        make_loaded, rules, storage, config):
    from src.data_loader import RULE_NOTES_COL

    loaded = make_loaded(_hybrid_records())
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, RULE_NOTES_COL] = "keep me"          # a Rule Note to preserve
    rules.create_rule("Cred Hub", "6618S")

    sh.run_rules_hybrid(
        work, rules, loaded, config, _hybrid_engine(config, rules, storage))
    assert work[loaded.new_account_col].tolist() == ["6800", "6800", "6618S", ""]

    cleared = sh.reset_automation(work, loaded, config)

    na = work[loaded.new_account_col].tolist()
    # Upload seed kept; the semantic + keyword fills are cleared; blank stays.
    assert na == ["6800", "", "", ""]
    assert cleared == 2                              # semantic row + rule row
    # Provenance reset for cleared rows; seed row untouched.
    assert work.at[0, sh.ENGINE_COL] == "seed"
    assert work.at[1, sh.ENGINE_COL] == ""
    assert work.at[2, sh.ENGINE_COL] == ""
    assert float(work.at[2, sh.CONF_COL]) == 0.0
    # Rules and Rule Notes are preserved so the user can adjust and re-run.
    assert len(rules.list_rules()) == 1
    assert work.at[0, RULE_NOTES_COL] == "keep me"


def test_reset_automation_preserves_manual_edits(make_loaded, storage, config):
    loaded = make_loaded([
        {"Description": "typed by hand", "New Account": ""},
        {"Description": "auto row", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Simulate a manual edit and an automation fill.
    work.at[0, loaded.new_account_col] = "5000"
    work.at[0, sh.ENGINE_COL] = "manual"
    work.at[1, loaded.new_account_col] = "6000"
    work.at[1, sh.ENGINE_COL] = "similarity"

    cleared = sh.reset_automation(work, loaded, config)
    assert work[loaded.new_account_col].tolist() == ["5000", ""]
    assert cleared == 1


def test_reset_automation_is_safe_when_nothing_ran(make_loaded, storage, config):
    loaded = make_loaded(_hybrid_records())
    work = sh.build_work_df(loaded, storage, config)
    before = work[loaded.new_account_col].tolist()
    cleared = sh.reset_automation(work, loaded, config)
    assert cleared == 0                              # nothing automated yet
    assert work[loaded.new_account_col].tolist() == before  # seed preserved


# --------------------------------------------------------------------------- #
# Confidence buckets + filtering
# --------------------------------------------------------------------------- #
def test_confidence_level_range(config):
    hi = config.confidence.auto_apply_cutoff
    lo = config.confidence.review_cutoff
    assert sh.confidence_level_range("All", config) is None
    assert sh.confidence_level_range("High", config) == (hi, 1.0)
    assert sh.confidence_level_range("Medium", config)[0] == lo
    assert sh.confidence_level_range("Low", config)[0] == 0.0


def test_confidence_buckets_exclude_blanks(make_loaded, storage, config):
    loaded = make_loaded([
        {"Description": "a", "New Account": "1"},
        {"Description": "b", "New Account": "2"},
        {"Description": "c", "New Account": "3"},
        {"Description": "d", "New Account": ""},   # blank -> no confidence
    ])
    work = sh.build_work_df(loaded, storage, config)
    work[sh.CONF_COL] = [0.95, 0.70, 0.30, 0.0]
    assert sh.confidence_buckets(work, loaded, config) == {
        "high": 1, "medium": 1, "low": 1}


def test_view_mask_confidence_low_band(make_loaded, storage, config):
    loaded = make_loaded([
        {"Description": "a", "New Account": "1"},
        {"Description": "b", "New Account": "2"},
        {"Description": "c", "New Account": "3"},
        {"Description": "d", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work[sh.CONF_COL] = [0.95, 0.70, 0.30, 0.0]

    rng = sh.confidence_level_range("Low", config)
    mask = sh.build_view_mask(work, loaded, conf_range=rng)
    # UI also drops blank rows when a confidence band is selected.
    na = work[loaded.new_account_col].astype(str).str.strip()
    mask &= (na != "")
    assert list(work.index[mask]) == [2]  # only the 0.30 coded row
