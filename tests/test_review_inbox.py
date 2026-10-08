"""Review-inbox UX locks — Keep / Fix / Not this stay on this page.

The UI rewrite must not weaken:
  * reject never learns and blocks the pairing
  * group Not this uses Review-page indices (not spreadsheet _select)
  * seeds / already-coded rows are never overwritten
  * review_rows_df sort + Approve defaults stay the same (other tests lock that)
  * Next leftover is least-sure consensus, not the biggest pile
  * blank no-suggestion rows can be Fix'd
  * Keep/Account editor labels map back to apply_review_table
"""

from __future__ import annotations

import pandas as pd

from models.schemas import FillAction
from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL


def _mark_review(work, idxs, *, group, suggested, conf=0.8, action=None):
    action = action or FillAction.NEEDS_REVIEW.value
    for i in idxs:
        work.at[i, sh.GROUP_COL] = group
        work.at[i, sh.ACTION_COL] = action
        work.at[i, sh.SUGGESTED_COL] = suggested
        work.at[i, sh.CONF_COL] = conf


def test_group_not_this_rejects_without_learning(make_loaded, storage, rules,
                                                 config):
    loaded = make_loaded([
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": "1111"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    _mark_review(work, range(3), group=5, suggested="6300S")
    work.at[2, sh.ENGINE_COL] = "manual"

    groups = sh.group_review_summary(work, loaded)
    assert groups and groups[0]["group_id"] == 5
    out = sh.reject_similarity_group(work, loaded, storage, 5)
    assert out["rejected"] == 3
    assert out["blocked"] >= 2
    assert work.at[0, NEW_ACCOUNT_COL] == ""
    assert work.at[1, NEW_ACCOUNT_COL] == ""
    assert work.at[2, NEW_ACCOUNT_COL] == "1111"  # seed / manual stays
    assert storage.list_learned_mappings() == []
    assert storage.count_training_data() == 0


def test_review_page_indices_reject_without_spreadsheet_select(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Odd Vendor", "Description": "misc", "New Account": ""},
        {"Name": "Other Co", "Description": "misc", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, sh.SUGGESTED_COL] = "6618S"
    work.at[0, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[1, sh.SUGGESTED_COL] = "6700S"
    work.at[1, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value

    # Spreadsheet selection is empty on purpose — Review uses its own indices.
    assert sh.selected_indices(work) == []
    table = sh.review_rows_df(work, loaded)
    picked = [int(table.iloc[0]["row"])]
    out = sh.reject_rows(work, loaded, storage, picked)
    assert out["rejected"] == 1
    assert work.at[picked[0], NEW_ACCOUNT_COL] == ""
    assert storage.list_learned_mappings() == []
    other = 1 if picked[0] == 0 else 0
    assert work.at[other, sh.SUGGESTED_COL] == (
        "6700S" if other == 1 else "6618S")


def test_inbox_keep_group_still_protects_coded_rows(make_loaded, storage,
                                                    rules, config):
    loaded = make_loaded([
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": "1111"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    _mark_review(work, range(3), group=8, suggested="6300S", conf=0.88)
    work.at[2, sh.ENGINE_COL] = "manual"

    out = sh.approve_similarity_group(work, loaded, storage, config, 8)
    assert out["approved"] == 2
    assert work.at[0, NEW_ACCOUNT_COL] == "6300S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6300S"
    assert work.at[2, NEW_ACCOUNT_COL] == "1111"


def test_split_group_is_never_one_click_keep(make_loaded, storage, rules,
                                             config):
    loaded = make_loaded([
        {"Name": "Split Co", "Description": "a", "New Account": ""},
        {"Name": "Split Co", "Description": "b", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    _mark_review(work, [0], group=9, suggested="6100")
    _mark_review(work, [1], group=9, suggested="6200")
    groups = sh.group_review_summary(work, loaded)
    split = next(g for g in groups if g["group_id"] == 9)
    assert split["split"] is True
    out = sh.approve_similarity_group(work, loaded, storage, config, 9)
    assert out["approved"] == 0
    assert out["split"]
    assert work.at[0, NEW_ACCOUNT_COL] == ""
    assert work.at[1, NEW_ACCOUNT_COL] == ""


def test_split_group_not_this_never_learns(make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Split Co", "Description": "a", "New Account": ""},
        {"Name": "Split Co", "Description": "b", "New Account": ""},
        {"Name": "Split Co", "Description": "c", "New Account": "1111"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    _mark_review(work, [0], group=4, suggested="6100")
    _mark_review(work, [1], group=4, suggested="6200")
    _mark_review(work, [2], group=4, suggested="6100")
    work.at[2, sh.ENGINE_COL] = "manual"

    out = sh.reject_similarity_group(work, loaded, storage, 4)
    assert out["rejected"] == 3
    assert work.at[0, NEW_ACCOUNT_COL] == ""
    assert work.at[1, NEW_ACCOUNT_COL] == ""
    assert work.at[2, NEW_ACCOUNT_COL] == "1111"
    assert storage.list_learned_mappings() == []
    assert storage.count_training_data() == 0


def test_apply_review_table_accepts_keep_account_labels(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Keep Me", "Description": "fee", "New Account": ""},
        {"Name": "Skip Me", "Description": "fee", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, sh.ACTION_COL] = FillAction.FILLED_REVIEW.value
    work.at[0, sh.SUGGESTED_COL] = "6322"
    work.at[1, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[1, sh.SUGGESTED_COL] = "6700"

    edited = pd.DataFrame([
        {"row": 0, "Keep": True, "Account": "6322", "Confidence": 0.9},
        {"row": 1, "Keep": False, "Account": "6700", "Confidence": 0.4},
    ])
    out = sh.apply_review_table(work, loaded, storage, edited)
    assert out["applied"] == 1
    assert out["learned"] == 1
    assert work.at[0, NEW_ACCOUNT_COL] == "6322"
    assert work.at[1, NEW_ACCOUNT_COL] == ""
    assert storage.count_training_data() == 1
    assert storage.list_learned_mappings()[0].account_code == "6322"


def test_blank_no_suggestion_row_can_be_fixed(make_loaded, storage, rules,
                                              config):
    """A Strict leftover (blank, no action) can be Fix'd from Review."""
    loaded = make_loaded([
        {"Name": "Unknown Vendor", "Description": "misc", "New Account": ""},
        {"Name": "Coded Co", "Description": "rent", "New Account": "4100"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # After Strict: unmatched blank has empty action, not no_match.
    assert str(work.at[0, sh.ACTION_COL] or "") == ""
    table = sh.review_rows_df(work, loaded, include_no_match=True)
    assert not table.empty
    assert 0 in set(int(r) for r in table["row"])

    out = sh.recode_rows(work, loaded, storage, [0], "6322")
    assert out["recoded"] == 1
    assert out["learned"] == 1
    assert work.at[0, NEW_ACCOUNT_COL] == "6322"
    assert work.at[1, NEW_ACCOUNT_COL] == "4100"
    assert storage.get_learned_lookup()  # memory written


def test_next_leftover_is_least_confident_consensus_pile(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Big Pile", "Description": "a", "New Account": ""},
        {"Name": "Big Pile", "Description": "b", "New Account": ""},
        {"Name": "Big Pile", "Description": "c", "New Account": ""},
        {"Name": "Shaky Pile", "Description": "x", "New Account": ""},
        {"Name": "Shaky Pile", "Description": "y", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    _mark_review(work, [0, 1, 2], group=1, suggested="6100", conf=0.92)
    _mark_review(work, [3, 4], group=2, suggested="6322", conf=0.41)

    nxt = sh.next_leftover(work, loaded)
    assert nxt is not None
    assert nxt["kind"] == "group"
    assert nxt["group"]["group_id"] == 2
    assert nxt["group"]["suggested"] == "6322"


def test_next_leftover_falls_back_to_least_sure_row(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Solo A", "Description": "a", "New Account": ""},
        {"Name": "Solo B", "Description": "b", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[0, sh.SUGGESTED_COL] = "6100"
    work.at[0, sh.CONF_COL] = 0.70
    work.at[1, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[1, sh.SUGGESTED_COL] = "6200"
    work.at[1, sh.CONF_COL] = 0.20
    table = sh.review_rows_df(work, loaded)
    nxt = sh.next_leftover(work, loaded, table)
    assert nxt["kind"] == "row"
    assert nxt["row"] == 1
