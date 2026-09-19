"""Collisions: one vendor, two GL accounts -> Review, never auto-fill.

A collision is not seed-disagreement. It is the same Name value (or the same
Memo value when Name is empty) coded to two or more different accounts in the
ingested book and/or the client's existing rules. On collision nothing fills —
not Strict, not fuzzy/regex, not Full Intelligent, not at 0.99. The row lands
in the Review inbox marked as a collision with both codes visible. Learning
only happens after the accountant decides (Keep / Fix); Not this blocks.
"""

from __future__ import annotations

import pandas as pd

from models.schemas import FillAction
from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL
from src.fill_down_engine import FillDownEngine
from src.review_queue import (
    REVIEW_ACTIONS,
    build_review_table,
    is_collision,
    learn_auto_approved,
)
from src.rules_manager import (
    RulesManager,
    collision_key_for_row,
    detect_collisions,
    phrase_key,
)


def _swann_like_book():
    """Fictional Swann-TD-style rows: one vendor coded two ways."""
    return [
        {"Name": "Lakeside Hardware", "Memo": "faucet cartridge",
         "Amount": "38.20", "New Account": "6326"},
        {"Name": "Lakeside Hardware", "Memo": "deck boards for unit 4",
         "Amount": "612.00", "New Account": "6760.01"},
        {"Name": "Pioneer Pest Control", "Memo": "quarterly treatment",
         "Amount": "95.00", "New Account": "6320"},
        {"Name": "Pioneer Pest Control", "Memo": "quarterly treatment",
         "Amount": "95.00", "New Account": "6320"},
    ]


def _blank_rows():
    return [
        {"Name": "Lakeside Hardware", "Memo": "shelf brackets",
         "Amount": "14.00", "New Account": ""},
        {"Name": "Pioneer Pest Control", "Memo": "quarterly treatment",
         "Amount": "95.00", "New Account": ""},
    ]


# --------------------------------------------------------------------------- #
# Detection primitives
# --------------------------------------------------------------------------- #
def test_collision_key_is_name_else_memo_and_ignores_numbers():
    row = pd.Series({"Name": "Lakeside Hardware #12", "Memo": "faucet"})
    assert collision_key_for_row(row, "Name", "Memo") == "lakeside hardware"
    memo_only = pd.Series({"Name": "", "Memo": "August lease fee - unit 12"})
    assert collision_key_for_row(memo_only, "Name", "Memo") == "august lease fee unit"
    assert phrase_key("LAKESIDE  Hardware") == "lakeside hardware"


def test_detect_collisions_only_flags_keys_with_two_codes():
    df = pd.DataFrame(_swann_like_book())
    found = detect_collisions(df, "Name", "Memo")
    assert found == {"lakeside hardware": {"6326", "6760.01"}}


def test_engine_guess_is_not_a_vote(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Acme", "Memo": "x", "New Account": "1111"},
        {"Name": "Acme", "Memo": "y", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[1, NEW_ACCOUNT_COL] = "2222"
    work.at[1, sh.ENGINE_COL] = "similarity"
    assert detect_collisions(work, "Name", "Memo") == {}


# --------------------------------------------------------------------------- #
# Test 2 — same Name, 6326 and 6760.01 -> Review as collision, both codes named
# --------------------------------------------------------------------------- #
def test_ingest_persists_collision_and_leaves_vendor_out_of_rules(
        make_loaded, rules, storage, config):
    coded = make_loaded(_swann_like_book())
    report = rules.ingest_rules_per_account(
        coded.df, client_id="swann",
        source_text_cols=sh.mining_columns(coded.df, coded, config))
    assert report.collisions == {"lakeside hardware": ["6326", "6760.01"]}
    assert storage.get_collision_lookup(client_id="swann") == {
        "lakeside hardware": {"6326", "6760.01"}}
    # The colliding vendor is on no rule; the memos still are.
    for rule in rules.list_rules(client_id="swann"):
        assert "lakeside hardware" not in [p.lower() for p in rule.match_phrases()]
    codes = {r.account_code for r in rules.list_rules(client_id="swann")}
    assert codes == {"6326", "6760.01", "6320"}
    # 6760.01 survives as its own account, distinct from 6326.
    assert "6760.01" in codes and "6326" in codes


def test_collision_rows_go_to_review_in_strict_with_both_codes(
        make_loaded, rules, storage, config):
    coded = make_loaded(_swann_like_book())
    rules.ingest_rules_per_account(
        coded.df, client_id="swann",
        source_text_cols=sh.mining_columns(coded.df, coded, config))

    later = make_loaded(_blank_rows())
    work = sh.build_work_df(later, storage, config)
    applied, results = sh.run_selected_rules_audited(
        work, rules, later, config, client_id="swann")
    # Pioneer fills; Lakeside stays blank and is marked as a collision.
    assert applied == 1
    assert work[NEW_ACCOUNT_COL].tolist() == ["", "6320"]
    assert work.at[0, sh.ACTION_COL] == FillAction.NEEDS_REVIEW.value
    assert work.at[0, sh.ENGINE_COL] == "collision"
    why = work.at[0, sh.WHY_COL]
    assert why.startswith("Collision:")
    assert "6326" in why and "6760.01" in why
    assert results[0].status == "collision"
    audit = sh.pure_rule_audit(results, config)
    assert audit["collisions"] == 1 and audit["left_blank"] == 1

    # It is in the Review inbox, with the badge data available to the card.
    table = sh.review_rows_df(work, later)
    assert table["row"].tolist() == [0]
    assert table.iloc[0]["Engine"] == sh.friendly_engine("collision")
    lookup = sh.collision_lookup_for(work, later, storage, client_id="swann")
    assert sh.is_collision_row(work, 0)
    assert sh.collision_codes_for_row(work, later, 0, lookup) == ["6326", "6760.01"]
    assert "6326 and 6760.01" in sh.collision_badge_text(["6760.01", "6326"])
    # A stray manual rule for the vendor changes nothing: still a collision.
    rules.create_rule("Lakeside Hardware", "6326", client_id="swann")
    fresh = sh.build_work_df(make_loaded(_blank_rows()), storage, config)
    applied, _ = sh.run_selected_rules_audited(fresh, rules, later, config,
                                               client_id="swann")
    assert fresh.at[0, NEW_ACCOUNT_COL] == "" and applied == 1


def test_collision_rows_go_to_review_in_full_intelligent_even_at_99(
        make_loaded, rules, storage, config):
    coded = make_loaded(_swann_like_book())
    rules.ingest_rules_per_account(
        coded.df, client_id="swann",
        source_text_cols=sh.mining_columns(coded.df, coded, config))
    # A rule for the vendor exists (0.99 confidence when it fires) — a
    # collision still wins.
    rules.create_rule("Lakeside Hardware", "6326", client_id="swann")

    later = make_loaded(_blank_rows())
    engine = FillDownEngine(
        config, rules, learned_lookup={}, client_id="swann",
        collision_lookup=storage.get_collision_lookup(client_id="swann"))
    res = engine.run(later)
    lakeside, pioneer = res.results
    assert lakeside.action == FillAction.NEEDS_REVIEW
    assert lakeside.engine_used == "collision"
    assert lakeside.proposed_value in (None, "")
    assert "6326" in lakeside.rationale and "6760.01" in lakeside.rationale
    assert res.df.iloc[0][later.new_account_col] == ""
    assert is_collision(lakeside)
    assert lakeside.action in REVIEW_ACTIONS
    # The unrelated vendor coded from its rule.
    assert pioneer.action == FillAction.AUTO_FILLED
    assert res.df.iloc[1][later.new_account_col] == "6320"
    # The classic review table carries it too.
    table = build_review_table(res.df, res.results, later.new_account_col,
                               later.text_columns)
    assert table["row"].tolist() == [0]
    assert table.iloc[0]["Engine"] == "collision"


def test_in_book_collision_is_detected_without_ingest(make_loaded, rules,
                                                      storage, config):
    """Coded rows in the current file disagree -> the blank twin escalates."""
    loaded = make_loaded(_swann_like_book() + _blank_rows())
    res = FillDownEngine(config, rules, learned_lookup={}).run(loaded)
    r = res.results[4]
    assert r.action == FillAction.NEEDS_REVIEW and r.engine_used == "collision"
    assert res.df.iloc[4][loaded.new_account_col] == ""
    # Pioneer (agreeing seeds) is unaffected by the other vendor's collision.
    assert res.df.iloc[5][loaded.new_account_col] == "6320"


def test_stored_collision_escalates_on_a_later_file(make_loaded, rules, storage,
                                                    config):
    """Stored colliding keys keep escalating until the accountant resolves."""
    storage.upsert_collision("lakeside hardware", ["6326", "6760.01"],
                             client_id="swann")
    later = make_loaded(_blank_rows())
    work = sh.build_work_df(later, storage, config)
    rules.create_rule("Lakeside", "6326", client_id="swann")
    n = sh.run_rules_only(work, rules, later, config, client_id="swann")
    assert n == 0
    assert work.at[0, NEW_ACCOUNT_COL] == ""
    assert work.at[0, sh.ENGINE_COL] == "collision"
    # Another client is not affected by this client's collision.
    other = sh.build_work_df(make_loaded(_blank_rows()), storage, config)
    rules.create_rule("Lakeside", "6326", client_id="other")
    assert sh.run_rules_only(other, rules, later, config,
                             client_id="other") == 1


# --------------------------------------------------------------------------- #
# Deciding a collision: Keep / Fix learn; Not this blocks, never learns
# --------------------------------------------------------------------------- #
def test_fix_on_collision_learns_and_not_this_blocks_without_learning(
        make_loaded, rules, storage, config):
    coded = make_loaded(_swann_like_book())
    rules.ingest_rules_per_account(
        coded.df, client_id="swann",
        source_text_cols=sh.mining_columns(coded.df, coded, config))
    later = make_loaded(_blank_rows() + [
        {"Name": "Lakeside Hardware", "Memo": "paint for unit 9",
         "Amount": "80.00", "New Account": ""}])
    work = sh.build_work_df(later, storage, config)
    sh.run_selected_rules_audited(work, rules, later, config, client_id="swann")
    assert work.at[0, sh.ENGINE_COL] == "collision"
    assert work.at[2, sh.ENGINE_COL] == "collision"

    # Fix row 0 -> 6760.01: learned through record_human_approval.
    out = sh.recode_rows(work, later, storage, [0], "6760.01", client_id="swann")
    assert out["recoded"] == 1 and out["learned"] == 1
    assert work.at[0, NEW_ACCOUNT_COL] == "6760.01"
    sig0 = str(work.at[0, sh.SIM_TEXT_COL]).strip()
    assert storage.get_learned_lookup(client_id="swann")[sig0] == "6760.01"

    # Not this on row 2: blocks BOTH competing codes, learns nothing.
    before = len(storage.list_learned_mappings(client_id="swann"))
    out = sh.reject_rows(work, later, storage, [2], client_id="swann")
    assert out["rejected"] == 1 and out["blocked"] == 2
    sig2 = str(work.at[2, sh.SIM_TEXT_COL]).strip()
    assert storage.get_blocked_lookup(client_id="swann")[sig2] == {"6326", "6760.01"}
    assert len(storage.list_learned_mappings(client_id="swann")) == before
    assert work.at[2, NEW_ACCOUNT_COL] == ""

    # Next month: the exact transaction she Fixed fills from memory (her
    # decision, not a guess); an unseen Lakeside row still escalates.
    again = make_loaded(_blank_rows())
    engine = FillDownEngine(
        config, rules, client_id="swann",
        learned_lookup=storage.get_learned_lookup(client_id="swann"),
        blocked_lookup=storage.get_blocked_lookup(client_id="swann"),
        collision_lookup=storage.get_collision_lookup(client_id="swann"))
    res = engine.run(again)
    assert res.results[0].engine_used == "learned"
    assert res.df.iloc[0][again.new_account_col] == "6760.01"
    unseen = make_loaded([{"Name": "Lakeside Hardware", "Memo": "new thing",
                           "Amount": "1.00", "New Account": ""}])
    res = engine.run(unseen)
    assert res.results[0].engine_used == "collision"
    assert res.df.iloc[0][unseen.new_account_col] == ""


def test_collision_rows_never_join_a_one_click_pile(make_loaded, storage,
                                                    config):
    loaded = make_loaded(_blank_rows())
    work = sh.build_work_df(loaded, storage, config)
    for i in (0, 1):
        work.at[i, sh.GROUP_COL] = 3
        work.at[i, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[0, sh.ENGINE_COL] = "collision"
    work.at[1, sh.ENGINE_COL] = "similarity"
    work.at[1, sh.SUGGESTED_COL] = "6320"
    (g,) = sh.group_review_summary(work, loaded)
    assert g["indices"] == [1]


# --------------------------------------------------------------------------- #
# Test 8 — auto-approve at the threshold; collisions never, seeds untouched
# --------------------------------------------------------------------------- #
def test_auto_approve_writes_confident_blank_rows_not_seeds_not_collisions(
        make_loaded, rules, storage, config):
    coded = make_loaded(_swann_like_book())
    rules.ingest_rules_per_account(
        coded.df, client_id="swann",
        source_text_cols=sh.mining_columns(coded.df, coded, config))
    rules.create_rule("Lakeside Hardware", "6326", client_id="swann")

    book = make_loaded([
        {"Name": "Pioneer Pest Control", "Memo": "quarterly treatment",
         "Amount": "95.00", "New Account": "6399"},           # seed (odd code)
        {"Name": "Pioneer Pest Control", "Memo": "quarterly treatment",
         "Amount": "95.00", "New Account": ""},               # confident blank
        {"Name": "Lakeside Hardware", "Memo": "shelf brackets",
         "Amount": "14.00", "New Account": ""},               # collision
    ])
    config.confidence.auto_apply_cutoff = 0.85
    engine = FillDownEngine(
        config, rules, learned_lookup={}, client_id="swann",
        collision_lookup=storage.get_collision_lookup(client_id="swann"))
    res = engine.run(book)
    seed, confident, collision = res.results
    assert seed.action == FillAction.KEPT_SEED
    assert res.df.iloc[0][book.new_account_col] == "6399"     # never overwritten
    assert confident.action == FillAction.AUTO_FILLED
    assert confident.confidence >= 0.85
    assert res.df.iloc[1][book.new_account_col] == "6320"
    assert collision.action == FillAction.NEEDS_REVIEW
    assert collision.engine_used == "collision"
    assert res.df.iloc[2][book.new_account_col] == ""

    # Auto-approved rows learn through the single learning path.
    out = learn_auto_approved(res.df, res.results, book.new_account_col,
                              0.85, storage, client_id="swann")
    assert out == {"auto_approved": 1, "learned": 1, "review": 1}
    lookup = storage.get_learned_lookup(client_id="swann")
    sig = str(res.df.iloc[1]["_sim_text"]).strip()
    assert lookup[sig] == "6320"
    assert storage.count_training_data(client_id="swann") == 1

    # Raising the threshold above the rule confidence approves nothing.
    assert learn_auto_approved(res.df, res.results, book.new_account_col,
                               0.995, storage, client_id="swann")["auto_approved"] == 0


def test_auto_approve_on_work_df_reports_audit_line(make_loaded, rules, storage,
                                                    config):
    rules.create_rule("Pioneer Pest Control", "6320")
    loaded = make_loaded(_blank_rows())
    work = sh.build_work_df(loaded, storage, config)
    storage.upsert_collision("lakeside hardware", ["6326", "6760.01"])
    engine = FillDownEngine(config, rules, learned_lookup={},
                            collision_lookup=storage.get_collision_lookup())
    sh.run_full(work, loaded, config, engine)
    out = sh.auto_approve_learn(work, loaded, storage, config, threshold=0.85)
    assert out["auto_approved"] == 1 and out["learned"] == 1
    assert out["review_pending"] == 1
    assert sh.auto_approve_message(out) == (
        "Auto-approved 1 row(s) at or above 85%. Review still has 1.")
    # Default threshold comes from config when no override is given.
    config.confidence.auto_apply_cutoff = 0.9
    assert sh.auto_approve_threshold(config) == 0.9
    assert sh.auto_approve_threshold(config, 0.7) == 0.7
