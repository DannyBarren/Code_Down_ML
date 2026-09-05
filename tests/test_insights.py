"""Insights: pure helpers on the working dataframe + never-writes guarantee."""

from __future__ import annotations

import pandas as pd

from src import insights as ins
from src import spreadsheet_helpers as sh


def _work(make_loaded, storage, config, records):
    loaded = make_loaded(records)
    return sh.build_work_df(loaded, storage, config), loaded


def _records():
    return [
        {"Name": "Bluebird Landscaping", "Memo": "weekly", "Amount": "120.00",
         "Date": "2026-01-05", "New Account": "6300"},
        {"Name": "Bluebird Landscaping", "Memo": "weekly", "Amount": "130.00",
         "Date": "2026-01-12", "New Account": "6300"},
        {"Name": "Cloud Nine Hosting", "Memo": "hosting", "Amount": "$49.00",
         "Date": "2026-01-06", "New Account": "6100"},
        {"Name": "Staples", "Memo": "paper", "Amount": "(25.00)",
         "Date": "2026-01-07", "New Account": ""},
        {"Name": "Falcon Courier", "Memo": "delivery", "Amount": "18.50",
         "Date": "2026-01-08", "New Account": ""},
    ]


def test_coding_coverage_splits_auto_human_seed(make_loaded, storage, config):
    work, loaded = _work(make_loaded, storage, config, _records())
    # Row 3 coded by a rule engine fill; row 4 approved by the user.
    work.at[3, loaded.new_account_col] = "6200"
    work.at[3, sh.ENGINE_COL] = "rules"
    work.at[4, loaded.new_account_col] = "6370"
    work.at[4, sh.ENGINE_COL] = "manual"

    cov = ins.coding_coverage(work, loaded)
    assert cov["total"] == 5
    assert cov["coded"] == 5
    assert cov["blank"] == 0
    assert cov["auto_filled"] == 1          # the rules fill
    assert cov["human"] == 4                # 3 seeds + 1 manual
    assert cov["seeds"] == 3
    assert cov["manual"] == 1
    assert cov["pct_coded"] == 1.0


def test_expense_mix_sums_by_account(make_loaded, storage, config):
    work, loaded = _work(make_loaded, storage, config, _records())
    mix = ins.expense_mix(work, loaded, config)
    assert mix is not None
    row6300 = mix[mix["Account"] == "6300"]
    assert float(row6300["Spend"].iloc[0]) == 250.00
    # Only coded rows count; the blank Staples row is excluded.
    assert set(mix["Account"]) == {"6300", "6100"}


def test_missing_amount_hides_mix_but_coverage_works(make_loaded, storage,
                                                     config):
    records = [{k: v for k, v in r.items() if k != "Amount"}
               for r in _records()]
    work, loaded = _work(make_loaded, storage, config, records)
    assert ins.expense_mix(work, loaded, config) is None
    residual = ins.uncoded_residual(work, loaded, config)
    assert residual["blank_spend"] is None
    assert residual["blank_rows"] == 2
    cov = ins.coding_coverage(work, loaded)
    assert cov["coded"] == 3 and cov["blank"] == 2
    # Vendor concentration still works by row count without amounts.
    vendors = ins.vendor_concentration(work, loaded, config)
    assert vendors is not None and vendors["by_spend"] is None
    assert vendors["by_count"].iloc[0]["Vendor"] == "Bluebird Landscaping"


def test_vendor_concentration_by_spend_and_count(make_loaded, storage, config):
    work, loaded = _work(make_loaded, storage, config, _records())
    vendors = ins.vendor_concentration(work, loaded, config)
    assert vendors["by_spend"].iloc[0]["Vendor"] == "Bluebird Landscaping"
    assert float(vendors["by_spend"].iloc[0]["Spend"]) == 250.00
    assert vendors["by_count"].iloc[0]["Rows"] == 2


def test_uncoded_residual_points_at_next_rules(make_loaded, storage, config):
    work, loaded = _work(make_loaded, storage, config, _records())
    residual = ins.uncoded_residual(work, loaded, config)
    assert residual["blank_rows"] == 2
    assert residual["blank_spend"] == -6.50      # -25.00 + 18.50
    top = residual["top_vendors"]
    assert list(top["Vendor"]) == ["Falcon Courier", "Staples"]


def test_parse_amount_edge_cases():
    assert ins.parse_amount("1,234.56") == 1234.56
    assert ins.parse_amount("$1,234.56") == 1234.56
    assert ins.parse_amount("(123.45)") == -123.45
    assert ins.parse_amount(42.5) == 42.5
    assert ins.parse_amount("") == 0.0
    assert ins.parse_amount("n/a") == 0.0
    assert ins.parse_amount(None) == 0.0


def test_throughput_from_run_history(storage):
    from models.schemas import RunSummary
    assert ins.throughput([])["runs"] == 0
    storage.add_run(RunSummary(file_name="week1.csv", total_rows=10,
                               auto_filled=6, filled_review=1,
                               needs_review=2, no_match=1))
    storage.add_run(RunSummary(file_name="week2.csv", total_rows=4,
                               auto_filled=3, needs_review=1))
    t = ins.throughput(storage.list_runs())
    assert t["runs"] == 2
    assert t["rows_coded_all_runs"] == 10
    assert t["last_run_file"] == "week2.csv"
    assert t["last_run_filled"] == 3


def test_insights_never_write_new_account(make_loaded, storage, config):
    work, loaded = _work(make_loaded, storage, config, _records())
    before = work[loaded.new_account_col].tolist()
    ins.coding_coverage(work, loaded)
    ins.expense_mix(work, loaded, config)
    ins.vendor_concentration(work, loaded, config)
    ins.uncoded_residual(work, loaded, config)
    assert work[loaded.new_account_col].tolist() == before
