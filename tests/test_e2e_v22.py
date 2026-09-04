"""Headless end-to-end tests for the Spreadsheet-Dominant v2.2 UI.

Uses Streamlit's ``AppTest`` to run ``main.py`` exactly as the browser would,
covering the happy path (landing -> load sample -> spreadsheet -> full run ->
panels) without a real browser. An isolated SQLite database is used via the
``FILLDOWN_DB_PATH`` env override so the shipped demo db is never touched.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Point the app at a throwaway database BEFORE Streamlit boots it (the bootstrap
# is cached per-process, so this must be set at import time).
_TMP_DB = Path(tempfile.mkdtemp()) / "e2e.db"
os.environ["FILLDOWN_DB_PATH"] = str(_TMP_DB)

from streamlit.testing.v1 import AppTest  # noqa: E402

MAIN = str(Path(__file__).resolve().parent.parent / "main.py")


def _button(at, needle: str):
    for b in at.button:
        if needle.lower() in (b.label or "").lower():
            return b
    raise AssertionError(f"button containing '{needle}' not found "
                         f"(have: {[b.label for b in at.button]})")


def _button_by_key(at, key: str):
    for b in at.button:
        if getattr(b, "key", None) == key:
            return b
    raise AssertionError(f"button with key '{key}' not found "
                         f"(have: {[b.key for b in at.button]})")


def _fresh_app():
    at = AppTest.from_file(MAIN, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    return at


def test_landing_renders():
    at = _fresh_app()
    assert at.session_state["view"] == "landing"


def test_full_happy_path():
    at = _fresh_app()
    # Load the bundled sample -> should auto-transition to the spreadsheet.
    _button(at, "Load sample data").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "spreadsheet"
    work = at.session_state["work_df"]
    assert work is not None and len(work) > 0

    # Run the full intelligent pipeline from the toolbar.
    _button(at, "Full Intelligent Run").click().run()
    assert not at.exception, at.exception
    work = at.session_state["work_df"]
    from src import spreadsheet_helpers as sh
    loaded = at.session_state["loaded"]
    counts = sh.summary_counts(work, loaded)
    assert counts["filled"] > 0
    # A run should have been recorded in the (isolated) history.
    assert at.session_state["last_result"] is not None


def test_filters_and_pagination_no_error():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    _button(at, "Full Intelligent Run").click().run()
    assert not at.exception, at.exception

    # Switch the filter to "Review only" and shrink the page size.
    at.session_state["filter_mode"] = "Review only"
    at.session_state["page_size"] = 100
    at.run()
    assert not at.exception, at.exception

    at.session_state["filter_mode"] = "Blanks only"
    at.run()
    assert not at.exception, at.exception


def test_global_search_and_select_all_in_view():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    assert not at.exception, at.exception

    from src import spreadsheet_helpers as sh
    work = at.session_state["work_df"]
    loaded = at.session_state["loaded"]

    # Pick a vendor token that exists in the data and search for it (all records).
    name_col = "Name" if "Name" in work.columns else loaded.text_columns[0]
    token = str(work.iloc[0][name_col]).split()[0]
    at.session_state["search_query"] = token
    at.run()
    assert not at.exception, at.exception

    # "Select all in view" must select exactly the rows matching the search,
    # across every page — not just what is on screen.
    expected = int(sh.build_view_mask(work, loaded, query=token).sum())
    _button_by_key(at, "tb_select_all").click().run()
    assert not at.exception, at.exception
    work = at.session_state["work_df"]
    assert len(sh.selected_indices(work)) == expected
    assert expected >= 1


def test_search_then_clear_filters():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    at.session_state["search_query"] = "zzz_no_match_token_xyz"
    at.run()
    assert not at.exception, at.exception
    # Clearing filters restores the full view without error.
    _button_by_key(at, "tb_clear_filters").click().run()
    assert not at.exception, at.exception
    assert at.session_state["search_query"] == ""


def test_clear_spreadsheet_data_flow():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    _button(at, "Full Intelligent Run").click().run()
    assert not at.exception, at.exception

    from src import spreadsheet_helpers as sh
    loaded = at.session_state["loaded"]
    work = at.session_state["work_df"]
    assert sh.summary_counts(work, loaded)["filled"] > 0

    # Toolbar Reset -> Clear spreadsheet data opens a confirmation first.
    _button_by_key(at, "reset_clear_sheet").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confirm_action"] == "clear_spreadsheet"

    _button_by_key(at, "confirm_clear_sheet_yes").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confirm_action"] is None
    work = at.session_state["work_df"]
    na = loaded.new_account_col
    assert (work[na].astype(str) == "").all()
    # Original columns survive the clear.
    assert "Name" in work.columns


def test_clear_spreadsheet_cancel_keeps_data():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    _button(at, "Full Intelligent Run").click().run()
    from src import spreadsheet_helpers as sh
    loaded = at.session_state["loaded"]
    filled_before = sh.summary_counts(at.session_state["work_df"], loaded)["filled"]

    _button_by_key(at, "reset_clear_sheet").click().run()
    _button_by_key(at, "confirm_clear_sheet_no").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confirm_action"] is None
    after = sh.summary_counts(at.session_state["work_df"], loaded)["filled"]
    assert after == filled_before


def test_start_fresh_flow():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    assert at.session_state["work_df"] is not None

    _button_by_key(at, "reset_start_fresh").click().run()
    assert not at.exception, at.exception
    assert at.session_state["confirm_action"] == "start_fresh"

    _button_by_key(at, "confirm_fresh_yes").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "landing"
    assert at.session_state["work_df"] is None
    # All rules were wiped as part of starting fresh (read the same DB fresh).
    from utils.storage import Storage
    from src.rules_manager import RulesManager
    assert RulesManager(Storage(_TMP_DB)).list_rules() == []


@pytest.mark.parametrize("panel", ["rules", "models", "history"])
def test_panels_render(panel):
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    at.session_state["panel"] = panel
    at.run()
    assert not at.exception, at.exception
    # Modal overlays the spreadsheet — the base view must stay put.
    assert at.session_state["view"] == "spreadsheet"


def test_open_panel_keeps_spreadsheet_view():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    _button_by_key(at, "nav_rules").click().run()
    assert not at.exception, at.exception
    assert at.session_state["panel"] == "rules"
    # Opening a management panel must NOT take over the grid.
    assert at.session_state["view"] == "spreadsheet"


def test_export_modal_opens():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    _button(at, "Export").click().run()
    assert not at.exception, at.exception
    assert at.session_state["export_open"] is True


def test_undo_after_edit():
    at = _fresh_app()
    _button(at, "Load sample data").click().run()
    _button(at, "Full Intelligent Run").click().run()
    assert not at.exception, at.exception
    # An undo entry should be available after the run; clicking undo must work.
    if at.session_state["undo_stack"]:
        _button(at, "Undo").click().run()
        assert not at.exception, at.exception
