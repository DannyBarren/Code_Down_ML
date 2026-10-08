"""Recall miss 36 vs 44 — synthetic Chrysalis-like and Swann-like fixtures.

On Amanda's real books fuzzy and regex both recovered 36 of 44 expected rows.
The eight misses are encoded here as **fictional** fixture rows (never the
real ledgers):

* ``tests/fixtures/chrysalis_recall_like.csv`` — a compact 44-row coded book.
  Its blank "twins" (one per coded row) include the previously-missed shapes:
  memo-only "lease fee" rows and their variants (``leasefee`` / ``Lease Fee`` /
  ``August lease fee — unit 12``), a decimal sub-account (``6760.01``), a
  vendor with a comma in its name, numeric suffixes on Names, an invoice
  number in a memo-only row, concatenated words, and a Name whose Memo
  belongs to a different account.
* ``tests/fixtures/swann_like.csv`` — ``6760.01`` and ``6326`` as distinct
  accounts, a Notes column full of account-like numbers, and one vendor coded
  both ways (a collision).

The 44 assertion is against *this* fixture, never a real ledger.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.schemas import FillAction
from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL, load_dataframe
from src.fill_down_engine import FillDownEngine
from utils.account_codes import normalize_code

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name, config):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} fixture missing from checkout.")
    return load_dataframe(str(path), config, source_name=name)


def _engine(config, rules, storage):
    return FillDownEngine(config, rules,
                          learned_lookup=storage.get_learned_lookup(),
                          collision_lookup=storage.get_collision_lookup())


def _twin(name: str, memo: str) -> dict:
    return {"Date": "2026-09-30", "Name": name, "Memo/Description": memo,
            "Amount": "10.00", NEW_ACCOUNT_COL: ""}


# The 44 blank twins, in fixture order. The first 13 are the shapes that were
# previously missed (or close variants of them); the rest are plain repeats.
_MISSED_TWINS = [
    (_twin("", "August lease fee — unit 12"), "4122"),   # memo-only, noise
    (_twin("Maple Court Leasing", "leasefee"), "4122"),  # Name hits anyway
    (_twin("", "LeaseFee"), "4122"),                     # concatenated memo
    (_twin("Oakridge Tenant Services", ""), "4122"),     # Name only, no Memo
    (_twin("Pinecrest Roofing", "shingle repair unit 3"), "6760.01"),
    (_twin("Summit Gutter Guard", ""), "6760.01"),
    # Name present + a Memo that belongs to 6100 must NOT flip the code.
    (_twin("Harbor Plumbing", "monthly hosting plan"), "6326"),
    (_twin("Willow Tree Plumbing", "drain snake #77"), "6326"),
    (_twin("Smith Jones & Co", "quarterly audit"), "6385"),
    (_twin("Harbor Sign Co 11", "banner install 11"), "6622"),
    (_twin("cook multimedia", "video"), "6622"),
    (_twin("CloudNine Hosting", "hosting plan #43"), "6100"),
    (_twin("", "weekly lawn service #99"), "6300"),      # memo-only + invoice
]


def _all_twins(coded: pd.DataFrame) -> list:
    twins = list(_MISSED_TWINS)
    for _, row in coded.iloc[len(_MISSED_TWINS):].iterrows():
        twins.append((_twin(str(row["Name"]), "September invoice"),
                      normalize_code(str(row[NEW_ACCOUNT_COL]))))
    return twins


def _month_book(coded: pd.DataFrame, twins: list) -> pd.DataFrame:
    """The coded fixture plus its blank twins — a book mid-month."""
    return pd.concat([coded, pd.DataFrame([t for t, _ in twins])],
                     ignore_index=True)


# --------------------------------------------------------------------------- #
# Test 6 — Chrysalis-like: ingest + Strict recovers all 44 (fixture only)
# --------------------------------------------------------------------------- #
def test_chrysalis_like_ingest_then_strict_recovers_all_44_twins(
        config, rules, storage):
    loaded = _load("chrysalis_recall_like.csv", config)
    coded = loaded.df
    assert len(coded) == 44 and (coded[NEW_ACCOUNT_COL] != "").all()

    report = rules.ingest_rules_per_account(
        coded, client_id="chrysalis",
        source_text_cols=sh.mining_columns(coded, loaded, config))
    assert report.collisions == {}                     # a clean book
    by_code = {r.account_code: r for r in rules.list_rules()}
    # One rule per GL account, not one per coded row.
    assert len(by_code) == coded[NEW_ACCOUNT_COL].map(normalize_code).nunique()
    lease = [p.lower() for p in by_code["4122"].match_phrases()]
    assert "lease fee" in lease
    assert "6760.01" in by_code                        # decimal survives

    twins = _all_twins(coded)
    assert len(twins) == 44
    book = _month_book(coded, twins)
    df, results = _engine(config, rules, storage).run_selected_rules(
        book, text_columns=loaded.text_columns, client_id="chrysalis")

    got = df[NEW_ACCOUNT_COL].tolist()[44:]
    want = [code for _, code in twins]
    misses = [(t, w, g) for (t, _), w, g in zip(twins, want, got) if g != w]
    assert not misses, f"{len(misses)} of 44 missed: {misses}"
    # The 44 coded rows were protected, not rewritten.
    assert df[NEW_ACCOUNT_COL].tolist()[:44] == \
        coded[NEW_ACCOUNT_COL].map(normalize_code).tolist()
    assert sum(1 for r in results if r.status == "protected_existing") == 44


def test_chrysalis_like_lease_fee_under_fuzzy_match_type(config, rules, storage):
    loaded = _load("chrysalis_recall_like.csv", config)
    rules.ingest_rules_per_account(
        loaded.df, client_id="chrysalis",
        source_text_cols=sh.mining_columns(loaded.df, loaded, config))
    for rule in rules.list_rules():
        rule.match_type = "fuzzy"
        rules.update_rule(rule)

    twins = _all_twins(loaded.df)
    book = _month_book(loaded.df, twins)
    df, _ = _engine(config, rules, storage).run_selected_rules(
        book, text_columns=loaded.text_columns, client_id="chrysalis")
    got = df[NEW_ACCOUNT_COL].tolist()[44:]
    misses = [(t, w, g) for (t, w), g in zip(twins, got) if g != w]
    assert not misses, f"{len(misses)} of 44 missed under fuzzy: {misses}"


def test_chrysalis_like_lease_fee_under_regex_match_type(config, rules, storage):
    loaded = _load("chrysalis_recall_like.csv", config)
    rules.create_rule(r"lease\s*fee", "4122", match_type="regex",
                      fields=["Name", "Memo/Description"], priority=10,
                      client_id="chrysalis")
    # Only twins that carry the pattern somewhere (a Name-only twin like
    # "Oakridge Tenant Services" needs the ingested vendor rule, not regex).
    lease_twins = [t for t, code in _MISSED_TWINS if code == "4122"
                   and "lease" in (t["Name"] + t["Memo/Description"]).lower()]
    assert len(lease_twins) == 3
    book = _month_book(loaded.df, [(t, "4122") for t in lease_twins])
    df, _ = _engine(config, rules, storage).run_selected_rules(
        book, text_columns=loaded.text_columns, client_id="chrysalis")
    assert df[NEW_ACCOUNT_COL].tolist()[44:] == ["4122"] * len(lease_twins)
    # A regex rule scoped to Name + Memo cannot be fooled by other columns.
    assert df[NEW_ACCOUNT_COL].tolist()[:44] == \
        loaded.df[NEW_ACCOUNT_COL].map(normalize_code).tolist()


# --------------------------------------------------------------------------- #
# Test 7 — Swann-like: 6760.01 and 6326 are distinct; Notes never count
# --------------------------------------------------------------------------- #
def test_swann_like_recovers_6760_01_and_6326_as_distinct_accounts(
        config, rules, storage):
    assert normalize_code("6760.01") == "6760.01"
    assert normalize_code("6326") == "6326"

    loaded = _load("swann_like.csv", config)
    codes = set(loaded.df[NEW_ACCOUNT_COL].map(normalize_code))
    assert {"6760.01", "6326", "6320", "6300"} <= codes
    # Loading never collapsed the decimal sub-account into its parent.
    assert "6760" not in codes

    cols = sh.mining_columns(loaded.df, loaded, config)
    assert not any("note" in str(c).lower() for c in cols)
    report = rules.ingest_rules_per_account(loaded.df, client_id="swann",
                                            source_text_cols=cols)
    by_code = {r.account_code: r for r in rules.list_rules()}
    assert {"6760.01", "6326"} <= set(by_code)
    roof = [p.lower() for p in by_code["6760.01"].match_phrases()]
    plumb = [p.lower() for p in by_code["6326"].match_phrases()]
    assert {"pinecrest roofing", "summit gutter guard"} <= set(roof)
    assert {"harbor plumbing", "willow tree plumbing"} <= set(plumb)
    # The Notes column's account-like numbers never became phrases or fields.
    for rule in by_code.values():
        assert "6326" not in " ".join(rule.match_phrases())
        assert "6760.01" not in " ".join(rule.match_phrases())
        assert not any("note" in f.lower() for f in rule.fields)
    # Metro Repair Group was coded both ways -> a collision, not a rule phrase.
    assert list(report.collisions.values()) == [["6326", "6760.01"]]
    assert "metro repair group" not in roof + plumb

    later = load_dataframe(
        pd.DataFrame([
            {"Date": "2026-09-01", "Name": "Pinecrest Roofing",
             "Memo/Description": "ridge cap", "Amount": "300.00",
             "Notes": "6326", NEW_ACCOUNT_COL: ""},
            {"Date": "2026-09-02", "Name": "Willow Tree Plumbing",
             "Memo/Description": "", "Amount": "90.00",
             "Notes": "6760.01", NEW_ACCOUNT_COL: ""},
            {"Date": "2026-09-03", "Name": "Metro Repair Group",
             "Memo/Description": "roof patch", "Amount": "120.00",
             "Notes": "", NEW_ACCOUNT_COL: ""},
        ]).to_csv(index=False).encode(), config, source_name="swann_sept.csv")
    work = sh.build_work_df(later, storage, config)
    applied, _ = sh.run_selected_rules_audited(
        work, rules, later, config, client_id="swann")
    assert applied == 2
    assert work[NEW_ACCOUNT_COL].tolist() == ["6760.01", "6326", ""]
    # The collision vendor waits in Review naming both codes.
    assert work.at[2, sh.ACTION_COL] == FillAction.NEEDS_REVIEW.value
    assert work.at[2, sh.ENGINE_COL] == "collision"
    assert "6326" in work.at[2, sh.WHY_COL] and "6760.01" in work.at[2, sh.WHY_COL]
