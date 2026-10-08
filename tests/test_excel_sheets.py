"""Excel two-tab bug: never silently code the summary tab as transactions.

Amanda's ``.xlsx`` exports have two tabs; tab 0 is often a summary / chart of
accounts. The landing picker already exists — these tests lock the default it
should preselect (the transaction sheet) and the loads that must not happen.
The workbooks are built in the test (openpyxl via pandas), never client files.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.data_loader import (
    DataLoadError,
    list_excel_sheets,
    load_dataframe,
    recommend_transaction_sheet,
    score_transaction_headers,
    sheet_headers,
)


def _two_tab_workbook() -> bytes:
    summary = pd.DataFrame({
        "Summary": ["Repairs", "Improvements", "Total"],
        "Account Number": ["6326", "6760.01", ""],
        "Balance": [1234.50, 612.00, 1846.50],
    })
    detail = pd.DataFrame({
        "Date": ["2026-08-01", "2026-08-03", "2026-08-05"],
        "Name": ["Lakeside Hardware", "Pioneer Pest Control", ""],
        "Memo/Description": ["faucet cartridge", "quarterly treatment",
                             "August lease fee - unit 12"],
        "Amount": [38.20, 95.00, 250.00],
        "New Account": ["6326", "", ""],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        detail.to_excel(writer, index=False, sheet_name="Transaction Detail")
    return buf.getvalue()


def _workbook(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=name)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Test 5 — sheet 0 is Summary / Account Number; sheet 1 is AppFolio-like
# --------------------------------------------------------------------------- #
def test_two_tab_workbook_recommends_the_transaction_sheet_not_tab_zero():
    raw = _two_tab_workbook()
    sheets = list_excel_sheets(raw)
    assert sheets == ["Summary", "Transaction Detail"]

    heads = sheet_headers(raw, sheets)
    assert heads["Summary"] == ["Summary", "Account Number", "Balance"]
    assert "Memo/Description" in heads["Transaction Detail"]

    rec = recommend_transaction_sheet(raw, sheets)
    assert rec["sheet"] == "Transaction Detail"
    assert rec["index"] == 1                      # not 0 just because it is first
    assert rec["data_sheets"] == ["Transaction Detail"]
    assert rec["scores"] == {"Summary": 0, "Transaction Detail": 3}


def test_loading_the_recommended_sheet_reads_transactions(config):
    raw = _two_tab_workbook()
    rec = recommend_transaction_sheet(raw)
    loaded = load_dataframe(raw, config, source_name="book.xlsx",
                            sheet_name=rec["sheet"])
    assert len(loaded.df) == 3
    assert "Name" in loaded.df.columns and "Memo/Description" in loaded.df.columns
    assert loaded.source_profile == "appfolio_transaction_detail"
    assert loaded.df[loaded.new_account_col].tolist() == ["6326", "", ""]
    # 6760.01 in the summary tab was never read as a New Account.
    assert "6760.01" not in loaded.df[loaded.new_account_col].tolist()


def test_summary_tab_is_never_silently_coded(config):
    """Loading tab 0 by default would treat 'Account Number' as codes."""
    raw = _two_tab_workbook()
    summary = load_dataframe(raw, config, source_name="book.xlsx", sheet_name=0)
    # The summary tab has no Name / Memo, so its similarity text is junk and
    # it is NOT what the recommendation returns.
    assert "Name" not in summary.df.columns
    assert recommend_transaction_sheet(raw)["index"] != 0


def test_generic_name_memo_amount_sheet_is_picked_without_a_profile():
    raw = _workbook({
        "Chart of Accounts": pd.DataFrame({"Account": ["6326"],
                                           "Title": ["Repairs"]}),
        # Name + Date only: no profile resolves it (no memo / amount role),
        # but it still reads as row-level transactions, not a summary.
        "Ledger": pd.DataFrame({"Name": ["Acme"], "Date": ["2026-08-01"],
                                "Ref": ["1001"]}),
    })
    rec = recommend_transaction_sheet(raw)
    assert rec["sheet"] == "Ledger" and rec["index"] == 1
    assert rec["scores"]["Ledger"] == 2
    assert score_transaction_headers(["Name", "Date", "Ref"]) == 2
    # A full Payee/Memo/Date/Total row wins a profile outright.
    assert score_transaction_headers(["Payee", "Memo", "Date", "Total"]) == 3
    assert score_transaction_headers(["Summary", "Account Number"]) == 0


def test_two_data_sheets_is_ambiguous_and_none_matching_is_none():
    detail = pd.DataFrame({"Name": ["A"], "Memo/Description": ["m"],
                           "Amount": [1.0], "Date": ["2026-01-01"]})
    raw = _workbook({"July": detail, "August": detail})
    rec = recommend_transaction_sheet(raw)
    assert rec["sheet"] is None and rec["index"] is None
    assert rec["data_sheets"] == ["July", "August"]
    assert "2 sheets" in rec["reason"]

    none = _workbook({
        "Summary": pd.DataFrame({"Summary": ["x"], "Account Number": ["1"]}),
        "Notes": pd.DataFrame({"Notes": ["hello"]}),
    })
    rec = recommend_transaction_sheet(none)
    assert rec["sheet"] is None and rec["data_sheets"] == []


def test_single_sheet_and_csv_are_unchanged(config):
    one = _workbook({"Only": pd.DataFrame({"Summary": ["x"],
                                           "Account Number": ["1"]})})
    rec = recommend_transaction_sheet(one)
    assert rec["sheet"] == "Only" and rec["index"] == 0
    csv = b"Name,Memo,Amount,New Account\nAcme,x,1.00,\n"
    assert list_excel_sheets(csv) == []            # not an Excel file
    loaded = load_dataframe(csv, config, source_name="t.csv")
    assert len(loaded.df) == 1


def test_unreadable_file_degrades_gracefully(config):
    rec = recommend_transaction_sheet(b"not an excel file")
    assert rec["sheet"] is None and rec["data_sheets"] == []
    with pytest.raises(DataLoadError):
        load_dataframe(b"not an excel file", config, source_name="x.xlsx")
