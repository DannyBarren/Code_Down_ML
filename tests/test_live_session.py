"""Live client instance: session store, auth gate, data-dir fail-closed."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src import config as cfg_mod
from src.ml_classifier import safe_client_id
from ui import common
from utils import session_store
from utils.storage import Storage


NORTHWIND = "Northwind Trading Co."
NORTHWIND_SLUG = "northwind_trading_co"


@pytest.fixture(autouse=True)
def _clean_live_env(monkeypatch):
    for var in (
        "FILLDOWN_LIVE", "FILLDOWN_DEMO", "FILLDOWN_DEMO_RESET",
        "FILLDOWN_AUTH_PASSWORD", "FILLDOWN_DATA_DIR", "HF_DATA_DIR",
        "RAILWAY_ENVIRONMENT", "RAILWAY_VOLUME_MOUNT_PATH",
        "SPACE_ID", "HF_SPACE_ID",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# Detection + sample auto-load
# --------------------------------------------------------------------------- #
def test_safe_client_id_is_session_folder_name():
    assert safe_client_id(NORTHWIND) == NORTHWIND_SLUG
    assert session_store.session_dir("/data", NORTHWIND).name == NORTHWIND_SLUG


def test_demo_mode_false_when_live(monkeypatch):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.setenv("FILLDOWN_DEMO", "1")
    assert common.demo_mode() is False
    assert common.should_auto_load_sample() is False
    assert common.live_mode() is True


def test_should_auto_load_sample_only_in_public_demo(monkeypatch):
    assert common.should_auto_load_sample() is False
    monkeypatch.setenv("FILLDOWN_DEMO", "1")
    assert common.should_auto_load_sample() is True
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    assert common.should_auto_load_sample() is False


# --------------------------------------------------------------------------- #
# Auth helper
# --------------------------------------------------------------------------- #
def test_live_boot_blocked_when_password_missing(monkeypatch):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.delenv("FILLDOWN_AUTH_PASSWORD", raising=False)
    assert cfg_mod.live_boot_blocked() is True
    assert cfg_mod.auth_password() == ""


def test_live_boot_open_when_password_set(monkeypatch):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.setenv("FILLDOWN_AUTH_PASSWORD", "correct-horse-battery")
    assert cfg_mod.live_boot_blocked() is False


def test_local_auth_open(monkeypatch):
    monkeypatch.delenv("FILLDOWN_LIVE", raising=False)
    monkeypatch.delenv("FILLDOWN_AUTH_PASSWORD", raising=False)
    assert cfg_mod.is_live() is False
    assert cfg_mod.live_boot_blocked() is False


# --------------------------------------------------------------------------- #
# select_data_dir fail-closed
# --------------------------------------------------------------------------- #
def test_select_data_dir_live_no_temp_fallback(monkeypatch, tmp_path):
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("file, not a directory", encoding="utf-8")
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.setenv("FILLDOWN_DATA_DIR", str(blocked))
    with pytest.raises(cfg_mod.DataDirUnwritable) as exc:
        cfg_mod.select_data_dir()
    msg = str(exc.value).lower()
    assert "temporary" in msg
    assert "railway_run_uid" in msg


def test_select_data_dir_live_uses_configured_dir(monkeypatch, tmp_path):
    dest = tmp_path / "volume"
    dest.mkdir()
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.setenv("FILLDOWN_DATA_DIR", str(dest))
    assert cfg_mod.select_data_dir() == dest


def test_select_data_dir_local_still_falls_back_to_temp(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked-file"
    blocked.write_text("nope", encoding="utf-8")
    monkeypatch.delenv("FILLDOWN_LIVE", raising=False)
    monkeypatch.setenv("FILLDOWN_DATA_DIR", str(blocked))
    chosen = cfg_mod.select_data_dir()
    # Must not raise, and must not be the unwritable configured path.
    assert chosen != blocked


# --------------------------------------------------------------------------- #
# Session store
# --------------------------------------------------------------------------- #
def _tiny_frame():
    return pd.DataFrame({
        "Name": ["Acme"],
        "Memo": ["Widget"],
        "New Account": ["6100"],
    })


def test_session_store_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    work = _tiny_frame()
    raw = b"date,name,amount\n2026-01-01,Acme,10\n"
    dest = session_store.save_workspace(
        tmp_path, NORTHWIND,
        work_df=work,
        original_bytes=raw,
        original_ext=".csv",
        source_name="northwind.csv",
        sheet=0,
    )
    assert dest is not None
    assert dest.name == NORTHWIND_SLUG
    assert (dest / "work_df.pkl").is_file()
    assert (dest / "original.bin").is_file()

    loaded = session_store.load_workspace(tmp_path, NORTHWIND)
    assert loaded is not None
    assert loaded["client_name"] == NORTHWIND
    assert loaded["safe_client_id"] == NORTHWIND_SLUG
    assert loaded["source_name"] == "northwind.csv"
    assert loaded["original_ext"] == ".csv"
    assert loaded["original_bytes"] == raw
    assert list(loaded["work_df"]["New Account"]) == ["6100"]
    assert loaded["row_count"] == 1


def test_session_store_noop_when_not_live(monkeypatch, tmp_path):
    monkeypatch.delenv("FILLDOWN_LIVE", raising=False)
    dest = session_store.save_workspace(
        tmp_path, NORTHWIND,
        work_df=_tiny_frame(),
        original_bytes=b"abc",
        original_ext=".csv",
        source_name="x.csv",
    )
    assert dest is None
    assert session_store.load_workspace(tmp_path, NORTHWIND) is None
    assert session_store.workspace_exists(tmp_path, NORTHWIND) is False
    assert session_store.list_sessions(tmp_path) == []
    assert not (_tiny_sessions := list(tmp_path.rglob("work_df.pkl")))


def test_session_store_refuses_empty_client_name(monkeypatch, tmp_path):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    dest = session_store.save_workspace(
        tmp_path, "   ",
        work_df=_tiny_frame(),
        original_bytes=b"abc",
        original_ext=".csv",
        source_name="x.csv",
    )
    assert dest is None
    assert session_store.list_sessions(tmp_path) == []


def test_session_store_list_and_delete(monkeypatch, tmp_path):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    session_store.save_workspace(
        tmp_path, NORTHWIND,
        work_df=_tiny_frame(),
        original_bytes=b"abc",
        original_ext=".csv",
        source_name="x.csv",
    )
    listed = session_store.list_sessions(tmp_path)
    assert len(listed) == 1
    assert listed[0]["client_name"] == NORTHWIND
    assert listed[0]["slug"] == NORTHWIND_SLUG
    assert session_store.delete_workspace(tmp_path, NORTHWIND) is True
    assert session_store.list_sessions(tmp_path) == []


def test_storage_enables_wal(tmp_path):
    db = tmp_path / "wal.db"
    store = Storage(db)
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    store.close()
    assert str(mode).lower() == "wal"


# --------------------------------------------------------------------------- #
# Row cap — live refuses, never silently truncates the book
# --------------------------------------------------------------------------- #
def test_live_row_cap_refuses_instead_of_truncating(monkeypatch):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.setenv("FILLDOWN_MAX_ROWS", "50")
    msg = common.row_cap_refusal(60)
    assert msg, "live must refuse an oversized upload"
    low = msg.lower()
    assert "60" in msg and "50" in msg
    assert "nothing was loaded" in low
    assert "add another export" in low


def test_live_row_cap_allows_a_file_that_fits(monkeypatch):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.setenv("FILLDOWN_MAX_ROWS", "50")
    assert common.row_cap_refusal(50) is None
    assert common.row_cap_refusal(1) is None


def test_no_cap_configured_never_refuses(monkeypatch):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    monkeypatch.delenv("FILLDOWN_MAX_ROWS", raising=False)
    assert common.row_cap_refusal(10_000_000) is None


def test_demo_and_local_still_truncate_rather_than_refuse(monkeypatch):
    """The public demo previews a big file; only live refuses."""
    monkeypatch.setenv("FILLDOWN_MAX_ROWS", "50")
    monkeypatch.setenv("FILLDOWN_DEMO", "1")
    assert common.row_cap_refusal(60) is None
    monkeypatch.delenv("FILLDOWN_DEMO", raising=False)
    assert common.row_cap_refusal(60) is None


# --------------------------------------------------------------------------- #
# Durable client name drives every client-scoped write
# --------------------------------------------------------------------------- #
def test_client_scope_survives_a_dropped_landing_widget(monkeypatch):
    """Navigation can drop ``client_name``; memory must stay client-scoped.

    If ``current_client_id()`` ever returned None here, a run would write
    learned mappings into the shared scope and the client's memory would
    look lost.
    """
    class _Stub:
        session_state = {"_live_client_name": NORTHWIND}

    monkeypatch.setattr(common, "st", _Stub)
    assert common.current_client_id() == NORTHWIND
    assert common.remember_client_name() == NORTHWIND


def test_client_scope_prefers_the_typed_name_and_backfills_durable(monkeypatch):
    class _Stub:
        session_state = {"client_name": "Second Client",
                         "_live_client_name": NORTHWIND}

    monkeypatch.setattr(common, "st", _Stub)
    assert common.current_client_id() == "Second Client"
    # Switching clients must move the durable copy too, or Save would write
    # the previous client's session folder.
    assert _Stub.session_state["_live_client_name"] == "Second Client"


def test_no_client_name_is_the_shared_scope(monkeypatch):
    class _Stub:
        session_state = {}

    monkeypatch.setattr(common, "st", _Stub)
    assert common.current_client_id() is None


def test_session_bytes_survive_excel_like_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("FILLDOWN_LIVE", "1")
    buf = io.BytesIO()
    _tiny_frame().to_csv(buf, index=False)
    raw = buf.getvalue()
    session_store.save_workspace(
        tmp_path, NORTHWIND,
        work_df=_tiny_frame(),
        original_bytes=raw,
        original_ext=".csv",
        source_name="book.csv",
        sheet="Transactions",
    )
    loaded = session_store.load_workspace(tmp_path, NORTHWIND)
    assert loaded["sheet"] == "Transactions"
    assert loaded["original_bytes"] == raw
