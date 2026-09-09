"""Review-inbox UX locks — Keep / Fix / Not this stay on this page.

The UI rewrite must not weaken:
  * reject never learns and blocks the pairing
  * group Not this uses Review-page indices (not spreadsheet _select)
  * seeds / already-coded rows are never overwritten
  * review_rows_df sort + Approve defaults stay the same (other tests lock that)
"""

from __future__ import annotations

from models.schemas import FillAction
from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL


def test_group_not_this_rejects_without_learning(make_loaded, storage, rules,
                                                 config):
    loaded = make_loaded([
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": "1111"},
    ])
    work = sh.build_work_df(loaded, storage, config)
    for i in range(3):
        work.at[i, sh.GROUP_COL] = 5
        work.at[i, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
        work.at[i, sh.SUGGESTED_COL] = "6300S"
        work.at[i, sh.CONF_COL] = 0.8
    work.at[2, sh.ENGINE_COL] = "manual"

    groups = sh.group_review_summary(work, loaded)
    assert groups and groups[0]["group_id"] == 5
    out = sh.reject_rows(work, loaded, storage, list(groups[0]["indices"]))
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
    # The other leftover is untouched.
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
    for i in range(3):
        work.at[i, sh.GROUP_COL] = 8
        work.at[i, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
        work.at[i, sh.SUGGESTED_COL] = "6300S"
        work.at[i, sh.CONF_COL] = 0.88
    work.at[2, sh.ENGINE_COL] = "manual"

    out = sh.approve_similarity_group(work, loaded, storage, config, 8)
    assert out["approved"] == 2
    assert work.at[0, NEW_ACCOUNT_COL] == "6300S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6300S"
    assert work.at[2, NEW_ACCOUNT_COL] == "1111"
