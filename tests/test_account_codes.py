"""Tests for NARPM account-code parsing and the 'S' sub-account notation."""

from utils.account_codes import (
    base_account,
    describe,
    is_sub_account,
    normalize_code,
    parse_account_code,
    same_base,
)


def test_normalize_basic():
    assert normalize_code("6618S") == "6618S"
    assert normalize_code("6618s") == "6618S"
    assert normalize_code("  6618 s ") == "6618S"
    assert normalize_code("6618-S") == "6618S"
    assert normalize_code("6618_s") == "6618S"


def test_excel_float_codes():
    # Excel often turns 6618 into 6618.0
    assert normalize_code("6618.0") == "6618"
    assert normalize_code("6618.00") == "6618"


def test_base_and_suffix():
    parsed = parse_account_code("6618S")
    assert parsed.base == "6618"
    assert parsed.suffix == "S"
    assert parsed.is_sub_account
    assert base_account("6618S") == "6618"
    assert base_account("6618") == "6618"


def test_non_code_passthrough():
    assert normalize_code("Miscellaneous") == "Miscellaneous"
    assert base_account("Miscellaneous") == ""
    assert not is_sub_account("Miscellaneous")


def test_same_base():
    assert same_base("6618S", "6618")
    assert same_base("6618S", "6618P")
    assert not same_base("6618S", "6620S")
    assert not same_base("Misc", "Misc")  # no numeric base


def test_describe():
    assert describe("6618S") == "Sub-account S of account 6618"
    assert describe("6618") == "Account 6618"
    assert describe("Misc") is None


def test_empty_and_none():
    assert normalize_code(None) == ""
    assert normalize_code("") == ""
    assert base_account(None) == ""
