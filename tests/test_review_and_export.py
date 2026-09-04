"""Tests for the review queue and exporters."""

import io

import pandas as pd

from src.exporter import (
    export_dataframe,
    export_preserving_original,
    export_csv,
)
from src.fill_down_engine import FillDownEngine
from src.review_queue import (
    apply_reviews,
    build_review_table,
    filter_by_confidence,
    mark_high_confidence,
)


def _run(config, rules, loaded):
    return FillDownEngine(config, rules, learned_lookup={}).run(loaded)


def test_build_and_filter_review_table(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub LLC", "Description": "monthly fee", "New Account": ""},
        {"Name": "Cred Hub Inc", "Description": "screening", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    table = build_review_table(
        res.df, res.results, loaded.new_account_col, loaded.text_columns)
    if not table.empty:
        filtered = filter_by_confidence(table, 0.0, 1.0)
        assert len(filtered) == len(table)
        assert "Confidence" in table.columns
        assert "Approve" in table.columns


def test_mark_high_confidence_and_apply(config, rules, storage, make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    table = build_review_table(
        res.df, res.results, loaded.new_account_col, loaded.text_columns,
        include_no_match=True)
    if not table.empty:
        approved = mark_high_confidence(table, 0.0)
        counts = apply_reviews(res.df, approved, loaded.new_account_col, storage)
        assert counts["applied"] >= 0
        # Codes written back are normalised.
        for v in res.df[loaded.new_account_col]:
            assert v == "" or v == v.strip()


def test_apply_normalises_codes(config, rules, storage, make_loaded):
    loaded = make_loaded([
        {"Name": "X", "Description": "y", "New Account": ""},
    ])
    res = _run(config, rules, loaded)
    table = pd.DataFrame([{"row": 0, "New Account": "6618 s", "Approve": True}])
    apply_reviews(res.df, table, loaded.new_account_col, storage)
    assert res.df.iloc[0][loaded.new_account_col] == "6618S"


def test_quickbooks_csv_has_base_column(config, rules, make_loaded):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
    ])
    res = _run(config, rules, loaded)
    raw = export_csv(res.df, loaded.new_account_col)
    out = pd.read_csv(io.BytesIO(raw))
    base_col = f"{loaded.new_account_col} (Base)"
    assert base_col in out.columns
    assert str(out[base_col].iloc[0]) == "6618"
    assert "_sim_text" not in out.columns


def test_export_dataframe_strips_internal(config, rules, make_loaded):
    loaded = make_loaded([{"Name": "X", "New Account": "1000"}])
    res = _run(config, rules, loaded)
    csv = export_dataframe(res.df, fmt="csv")
    assert b"_sim_text" not in csv


def test_export_xlsx_handles_nan_and_numeric_columns():
    """Regression: autosize must not call len() on float/NaN cells.

    Some pandas versions leave missing values as float NaN even after
    ``astype(str)``; the exporter must still build a valid xlsx.
    """
    df = pd.DataFrame({
        "Name": ["Acme", None, "Zeta"],       # object col with a None
        "Amount": [10.5, float("nan"), 3.0],  # numeric col with NaN
        "Num": [float("nan"), float("nan"), float("nan")],  # all-NaN column
        "New Account": ["6510", "", "6510"],
    })
    out = export_dataframe(df, fmt="xlsx")
    back = pd.read_excel(io.BytesIO(out))
    assert list(back.columns) == ["Name", "Amount", "Num", "New Account"]
    assert len(back) == 3


def test_export_preserving_original_roundtrip(config, rules, make_loaded):
    # Build an xlsx in memory to act as the "original".
    df = pd.DataFrame([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    original = buf.getvalue()

    from src.data_loader import load_dataframe
    loaded = load_dataframe(original, config, source_name="orig.xlsx")
    res = _run(config, rules, loaded)
    out = export_preserving_original(original, res.df, loaded.new_account_col, config)
    back = pd.read_excel(io.BytesIO(out))
    assert "New Account" in back.columns
    assert len(back) == 2
