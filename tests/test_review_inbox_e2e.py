"""Headless AppTest coverage for the Review inbox (Keep / Fix / Not this).

The inbox is the surface a conversion accountant clicks most, and its worst
failures are *render-time*: a NameError, a stale widget key, or a decide bar
that needs a second rerun. Unit tests on ``spreadsheet_helpers`` cannot see
those. These tests run ``main.py`` the way the browser does.

An isolated SQLite database is used via ``FILLDOWN_DB_PATH``. These tests
never depend on seeded keyword rules, because a sibling e2e module clears
them and pytest may run the files in either order.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Must be set before Streamlit boots the cached bootstrap. ``setdefault`` so a
# sibling e2e module that already chose a throwaway db keeps ownership.
os.environ.setdefault(
    "FILLDOWN_DB_PATH", str(Path(tempfile.mkdtemp()) / "review_e2e.db"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from models.schemas import FillAction  # noqa: E402
from src import spreadsheet_helpers as sh  # noqa: E402

MAIN = str(Path(__file__).resolve().parent.parent / "main.py")

SUGGESTED = "6300S"


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


def _has_key(at, key: str) -> bool:
    return any(getattr(b, "key", None) == key for b in at.button)


def _storage():
    """The very storage instance the running app uses (cached bootstrap)."""
    from ui.common import services
    return services()[1]


def _loaded_app():
    at = AppTest.from_file(MAIN, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    _button(at, "Load sample data").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "spreadsheet"
    return at


def _seed_review_queue(at, *, rows=3, group=77, conf=0.62,
                       action=FillAction.NEEDS_REVIEW.value):
    """Put deterministic leftovers in the book, then open Review.

    Engine output on the sample is all-or-nothing (auto-filled or no-match),
    so the review queue is staged directly. That keeps these tests about the
    inbox, not about similarity scoring.
    """
    work = at.session_state["work_df"]
    loaded = at.session_state["loaded"]
    na = loaded.new_account_col
    picked = []
    for i in work.index:
        if len(picked) >= rows:
            break
        if str(work.at[i, na] or "").strip():
            continue          # never stage a coded row: seeds stay sacred
        picked.append(int(i))
    assert len(picked) == rows, "sample should have blank rows to stage"
    for i in picked:
        work.at[i, sh.GROUP_COL] = group
        work.at[i, sh.ACTION_COL] = action
        work.at[i, sh.SUGGESTED_COL] = SUGGESTED
        work.at[i, sh.CONF_COL] = conf
        work.at[i, sh.WHY_COL] = "Similar rows suggest this code."
    at.session_state["view"] = "review"
    at.run()
    assert not at.exception, at.exception
    return picked


# --------------------------------------------------------------------------- #
# Render safety
# --------------------------------------------------------------------------- #
def test_review_renders_with_a_staged_queue():
    at = _loaded_app()
    _seed_review_queue(at)
    assert at.session_state["view"] == "review"


def test_review_renders_blank_only_queue_after_a_rules_only_run():
    """Strict leaves unmatched rows blank with no action; Review must still open."""
    at = _loaded_app()
    at.session_state["review_include_no_match"] = True
    at.session_state["view"] = "review"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "review"


def test_review_renders_with_the_leftover_table_shown():
    """Guards the data_editor schema/key: Pick + Keep + Account columns."""
    at = _loaded_app()
    _seed_review_queue(at)
    at.session_state["review_show_leftover_table"] = True
    at.run()
    assert not at.exception, at.exception


def test_empty_queue_shows_queue_is_clear():
    at = _loaded_app()
    at.session_state["review_include_no_match"] = False
    at.session_state["view"] = "review"
    at.run()
    assert not at.exception, at.exception
    blob = " ".join(s.value for s in at.success).lower()
    assert "queue is clear" in blob, [s.value for s in at.success]


def test_spreadsheet_cta_opens_review():
    at = _loaded_app()
    _button_by_key(at, "tb_goto_review").click().run()
    assert not at.exception, at.exception
    assert at.session_state["view"] == "review"


# --------------------------------------------------------------------------- #
# The three verbs
# --------------------------------------------------------------------------- #
def test_keep_the_next_leftover_codes_the_pile_and_learns():
    at = _loaded_app()
    picked = _seed_review_queue(at)
    loaded = at.session_state["loaded"]
    na = loaded.new_account_col
    before = len(_storage().list_learned_mappings())

    _button_by_key(at, "rev_inbox_keep").click().run()
    assert not at.exception, at.exception

    work = at.session_state["work_df"]
    assert [str(work.at[i, na]).strip() for i in picked] == [SUGGESTED] * len(picked)
    assert len(_storage().list_learned_mappings()) > before, "Keep must remember"
    # The decided rows leave the queue.
    assert sh.summary_counts(work, loaded)["review_pending"] == 0


def test_not_this_leaves_rows_blank_and_never_learns():
    at = _loaded_app()
    picked = _seed_review_queue(at, group=78)
    loaded = at.session_state["loaded"]
    na = loaded.new_account_col
    before = len(_storage().list_learned_mappings())

    _button_by_key(at, "rev_inbox_not").click().run()
    assert not at.exception, at.exception

    work = at.session_state["work_df"]
    assert all(str(work.at[i, na]).strip() == "" for i in picked)
    assert len(_storage().list_learned_mappings()) == before, \
        "reject must never write a learned mapping"


def test_keep_never_overwrites_an_already_coded_row():
    """Seeds and manual codes stay sacred even inside a kept pile."""
    at = _loaded_app()
    picked = _seed_review_queue(at, rows=3, group=79)
    work = at.session_state["work_df"]
    loaded = at.session_state["loaded"]
    na = loaded.new_account_col

    protected = picked[0]
    work.at[protected, na] = "1111"
    work.at[protected, sh.ENGINE_COL] = "manual"
    at.run()
    assert not at.exception, at.exception

    _button_by_key(at, "rev_inbox_keep").click().run()
    assert not at.exception, at.exception
    work = at.session_state["work_df"]
    assert str(work.at[protected, na]).strip() == "1111"
    for i in picked[1:]:
        assert str(work.at[i, na]).strip() == SUGGESTED


def test_split_pile_offers_not_this_but_never_one_click_keep():
    at = _loaded_app()
    picked = _seed_review_queue(at, rows=3, group=80)
    work = at.session_state["work_df"]
    work.at[picked[0], sh.SUGGESTED_COL] = "6100"   # make the pile disagree
    at.run()
    assert not at.exception, at.exception

    groups = sh.group_review_summary(work, at.session_state["loaded"])
    split = next(g for g in groups if g["group_id"] == 80)
    assert split["split"] is True
    # A split pile is never a one-click Keep, but it can still be rejected.
    assert not _has_key(at, "rev_group_approve_80")
    assert _has_key(at, "rev_group_not_80") or _has_key(at, "rev_inbox_not")


# --------------------------------------------------------------------------- #
# Promote to rule — the control that used to raise NameError
# --------------------------------------------------------------------------- #
def test_always_code_this_vendor_does_not_traceback():
    at = _loaded_app()
    picked = _seed_review_queue(at)
    at.session_state["rev_promote_pick"] = picked[0]
    at.run()
    assert not at.exception, at.exception

    _button_by_key(at, "rev_promote_btn").click().run()
    assert not at.exception, at.exception
    assert at.session_state["rule_panel_open"] is True


def test_promote_control_available_when_the_table_is_hidden():
    at = _loaded_app()
    picked = _seed_review_queue(at)
    at.session_state["review_show_leftover_table"] = False
    at.run()
    assert not at.exception, at.exception
    assert _has_key(at, "rev_promote_btn")
