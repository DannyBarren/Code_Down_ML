"""Export-shape ingest: source profiles + multi-file append.

Locks the throughout-the-month workflow: AppFolio / QuickBooks exports load
without renaming columns, and appending a later export never duplicates rows
or overwrites an existing ``New Account``.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL, load_dataframe
from src.ingest_profiles import detect_profile, profile_chip


def _csv_bytes(records) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(records).to_csv(buf, index=False)
    return buf.getvalue()


def _appfolio_records():
    return [
        {"Date": "2026-01-05", "Name": "Bluebird Landscaping",
         "Memo/Description": "weekly lawn service", "Amount": "120.00",
         "New Account": "6300"},
        {"Date": "2026-01-06", "Name": "Cloud Nine Hosting",
         "Memo/Description": "monthly hosting plan", "Amount": "49.00",
         "New Account": ""},
    ]


def _quickbooks_records():
    return [
        {"Txn Date": "2026-01-05", "Payee": "Bluebird Landscaping",
         "Memo": "weekly lawn service", "Amount": "120.00",
         "New Account": "6300"},
        {"Txn Date": "2026-01-06", "Payee": "Cloud Nine Hosting",
         "Memo": "monthly hosting plan", "Amount": "49.00",
         "New Account": ""},
    ]


# --------------------------------------------------------------------------- #
# Profile detection + header resolution
# --------------------------------------------------------------------------- #
def test_appfolio_headers_resolve_without_renaming(config):
    loaded = load_dataframe(_csv_bytes(_appfolio_records()), config,
                            source_name="af.csv")
    assert loaded.source_profile == "appfolio_transaction_detail"
    # The combined Memo/Description column feeds the similarity text.
    assert "Memo/Description" in loaded.text_columns
    assert "Name" in loaded.text_columns
    assert "lawn service" in loaded.df["_sim_text"].iloc[0]
    assert loaded.seed_count == 1


def test_quickbooks_headers_resolve_without_renaming(config):
    loaded = load_dataframe(_csv_bytes(_quickbooks_records()), config,
                            source_name="qb.csv")
    assert loaded.source_profile == "quickbooks_transaction_list"
    # Payee feeds the vendor signal even though the generic config says "Name".
    assert "Payee" in loaded.text_columns
    assert "bluebird landscaping" in loaded.df["_sim_text"].iloc[0]
    assert loaded.seed_count == 1


def test_unknown_headers_still_load_generic(config):
    loaded = load_dataframe(_csv_bytes([
        {"Supplier Name": "X", "Details": "y", "New Account": ""},
    ]), config, source_name="mystery.csv")
    assert loaded.source_profile == ""
    assert profile_chip(loaded.source_profile) == "Generic export"
    assert NEW_ACCOUNT_COL in loaded.df.columns


def test_profile_chip_labels():
    assert profile_chip("appfolio_transaction_detail") == \
        "Read as AppFolio transaction detail"
    assert profile_chip("quickbooks_transaction_list") == \
        "Read as QuickBooks transaction list"
    assert profile_chip("") == "Generic export"


def test_ambiguous_shape_stays_generic():
    # Name + Memo + Amount resolve for both profiles with no distinctive
    # header — a tie must not pick one arbitrarily.
    assert detect_profile(["Name", "Memo", "Amount"]) is None
    assert detect_profile(["Name", "Memo/Description", "Amount"]) is not None
    assert detect_profile(["Payee", "Memo", "Amount"]).key == \
        "quickbooks_transaction_list"


def test_notes_still_not_mined_after_appfolio_load(make_loaded, storage,
                                                   rules, config):
    records = _appfolio_records()
    for r in records:
        r["Notes"] = "coachnote zzz"
    loaded = load_dataframe(_csv_bytes(records), config, source_name="af.csv")
    assert loaded.source_profile == "appfolio_transaction_detail"
    work = sh.build_work_df(loaded, storage, config)
    assert "Notes" not in sh.mining_columns(work, loaded, config)
    cands = sh.candidate_rules(work, loaded, rules, config=config)
    assert all("coachnote" not in c["keyword"] for c in cands)


# --------------------------------------------------------------------------- #
# Multi-file append
# --------------------------------------------------------------------------- #
def test_append_does_not_duplicate_base_sig_rows(make_loaded, storage, config):
    loaded = load_dataframe(_csv_bytes(_appfolio_records()), config,
                            source_name="week1.csv")
    work = sh.build_work_df(loaded, storage, config)
    before = len(work)

    # Week 2 file: one identical row (same text/date/amount) + one new row.
    week2 = _appfolio_records()[:1] + [
        {"Date": "2026-01-12", "Name": "Summit Cleaning Crew",
         "Memo/Description": "janitorial visit", "Amount": "200.00",
         "New Account": ""},
    ]
    new_work, audit = sh.append_export(work, loaded, _csv_bytes(week2),
                                       "week2.csv", config, storage)
    assert audit["added"] == 1
    assert audit["skipped"] == 1
    assert len(new_work) == before + 1
    # The duplicate's text appears exactly once.
    sims = new_work["_base_sig"].astype(str)
    assert (sims == sims.iloc[0]).sum() == 1


def test_append_never_overwrites_existing_new_account(storage, config):
    loaded = load_dataframe(_csv_bytes(_appfolio_records()), config,
                            source_name="week1.csv")
    work = sh.build_work_df(loaded, storage, config)
    # The blank row got coded by the user since the first file arrived.
    work.at[1, NEW_ACCOUNT_COL] = "6100"
    work.at[1, sh.ENGINE_COL] = "manual"

    # Week 2 file carries the same rows, one now coded differently — the
    # existing values must win.
    week2 = [
        {"Date": "2026-01-05", "Name": "Bluebird Landscaping",
         "Memo/Description": "weekly lawn service", "Amount": "120.00",
         "New Account": "9999"},
        {"Date": "2026-01-06", "Name": "Cloud Nine Hosting",
         "Memo/Description": "monthly hosting plan", "Amount": "49.00",
         "New Account": "9999"},
    ]
    new_work, audit = sh.append_export(work, loaded, _csv_bytes(week2),
                                       "week2.csv", config, storage)
    assert audit["added"] == 0
    assert audit["skipped"] == 2
    assert audit["protected"] == 2
    assert new_work.at[0, NEW_ACCOUNT_COL] == "6300"   # seed kept
    assert new_work.at[1, NEW_ACCOUNT_COL] == "6100"   # manual edit kept


def test_append_brings_new_rows_with_their_codes_as_seeds(storage, config):
    loaded = load_dataframe(_csv_bytes(_appfolio_records()), config,
                            source_name="week1.csv")
    work = sh.build_work_df(loaded, storage, config)
    week2 = [
        {"Date": "2026-01-12", "Name": "Summit Cleaning Crew",
         "Memo/Description": "janitorial visit", "Amount": "200.00",
         "New Account": "6310"},
        {"Date": "2026-01-13", "Name": "Falcon Courier",
         "Memo/Description": "package delivery", "Amount": "18.50",
         "New Account": ""},
    ]
    new_work, audit = sh.append_export(work, loaded, _csv_bytes(week2),
                                       "week2.csv", config, storage)
    assert audit["added"] == 2
    # The coded new row arrives as a protected seed; the blank one stays blank.
    assert new_work.at[2, NEW_ACCOUNT_COL] == "6310"
    assert new_work.at[2, sh.ENGINE_COL] == "seed"
    assert new_work.at[3, NEW_ACCOUNT_COL] == ""
    # And a Strict run still works on the combined frame.
    rules = sh.RulesManager(storage)
    rules.create_rule("Falcon Courier", "6370")
    n, _ = sh.run_selected_rules_audited(new_work, rules, loaded, config)
    assert n == 1
    assert new_work.at[3, NEW_ACCOUNT_COL] == "6370"
    assert new_work.at[2, NEW_ACCOUNT_COL] == "6310"   # seed untouched
