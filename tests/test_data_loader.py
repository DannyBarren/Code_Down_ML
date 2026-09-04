"""Tests for the data loader: header resolution, normalisation, edge cases."""

import io

import pandas as pd
import pytest

from src.data_loader import (
    NEW_ACCOUNT_COL,
    SIM_TEXT_COL,
    DataLoadError,
    load_dataframe,
)


def _csv_bytes(records):
    df = pd.DataFrame(records)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_basic_load(config, make_loaded):
    loaded = make_loaded([
        {"Description": "CredHub fee", "Name": "Cred Hub", "Memo": "x",
         "New Account": "6618S"},
        {"Description": "Office stuff", "Name": "Office Depot", "Memo": "y",
         "New Account": ""},
    ])
    assert len(loaded.df) == 2
    assert loaded.new_account_col == NEW_ACCOUNT_COL
    assert loaded.seed_count == 1
    assert loaded.blank_count == 1
    assert SIM_TEXT_COL in loaded.df.columns


def test_seed_codes_normalised(make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "New Account": "6618 s"},
        {"Name": "Cred Hub", "New Account": "6618.0"},
    ])
    vals = list(loaded.df[loaded.new_account_col])
    assert vals[0] == "6618S"
    assert vals[1] == "6618"


def test_alternate_header_name(config):
    # "New COA" should be recognised as the New Account column.
    loaded = load_dataframe(
        _csv_bytes([{"Name": "X", "New COA": "1000"}]), config, "alt.csv")
    assert loaded.new_account_col == NEW_ACCOUNT_COL
    assert loaded.original_new_account_header.lower() == "new coa"
    assert loaded.seed_count == 1


def test_missing_new_account_created(config):
    loaded = load_dataframe(
        _csv_bytes([{"Name": "X", "Memo": "Y"}]), config, "no_na.csv")
    assert NEW_ACCOUNT_COL in loaded.df.columns
    assert loaded.seed_count == 0


def test_empty_file_raises(config):
    with pytest.raises(DataLoadError):
        load_dataframe(b"", config, "empty.csv")


def test_numeric_tokens_dropped_in_sim_text(make_loaded):
    loaded = make_loaded([
        {"Description": "Check 123456 payment", "Name": "Vendor", "New Account": ""},
    ])
    sim = loaded.df[SIM_TEXT_COL].iloc[0]
    assert "123456" not in sim
    assert "vendor" in sim
