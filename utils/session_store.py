"""Persist the working workbook for a hosted (live) client instance.

Used only when ``FILLDOWN_LIVE=1``. Local and public-demo runs never write
here — ``work_df`` stays in Streamlit session state exactly as before.

Layout under ``{data_dir}/sessions/{safe_client_id}/``:

* ``meta.json`` — client name, filename, extension, sheet, timestamp
* ``original.bin`` — the uploaded bytes (Excel export reopens these)
* ``work_df.pkl`` — pandas pickle of the coded working dataframe

The slugger is ``src.ml_classifier.safe_client_id`` — do not invent another.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.config import is_live
from src.ml_classifier import safe_client_id
from utils.logging_setup import get_logger

logger = get_logger("session_store")

META_NAME = "meta.json"
BYTES_NAME = "original.bin"
WORK_NAME = "work_df.pkl"


def _root(data_dir: Union[str, Path]) -> Path:
    return Path(data_dir) / "sessions"


def session_dir(data_dir: Union[str, Path], client_name: str) -> Path:
    """Folder for one client. Uses the shared ``safe_client_id`` slugger."""
    return _root(data_dir) / safe_client_id(client_name)


def save_workspace(
    data_dir: Union[str, Path],
    client_name: str,
    *,
    work_df: pd.DataFrame,
    original_bytes: bytes,
    original_ext: str,
    source_name: str,
    sheet: Any = 0,
) -> Optional[Path]:
    """Write one client's workspace. No-op (returns None) when not live.

    Refuses to save an empty client name so books never land in the shared
    ``client`` slug by accident.
    """
    if not is_live():
        return None
    name = (client_name or "").strip()
    if not name:
        return None
    if work_df is None or original_bytes is None:
        return None

    dest = session_dir(data_dir, name)
    dest.mkdir(parents=True, exist_ok=True)
    work_path = dest / WORK_NAME
    bytes_path = dest / BYTES_NAME
    meta_path = dest / META_NAME

    work_df.to_pickle(work_path)
    bytes_path.write_bytes(original_bytes)
    meta = {
        "client_name": name,
        "safe_client_id": safe_client_id(name),
        "source_name": source_name or "",
        "original_ext": original_ext or "",
        "sheet": sheet if isinstance(sheet, (int, str)) else 0,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(work_df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("workspace_saved", client=safe_client_id(name),
                rows=meta["row_count"], file=meta["source_name"])
    return dest


def load_workspace(data_dir: Union[str, Path],
                   client_name: str) -> Optional[Dict[str, Any]]:
    """Read one client's workspace. None when not live or nothing is stored."""
    if not is_live():
        return None
    name = (client_name or "").strip()
    if not name:
        return None
    dest = session_dir(data_dir, name)
    meta_path = dest / META_NAME
    work_path = dest / WORK_NAME
    bytes_path = dest / BYTES_NAME
    if not (meta_path.is_file() and work_path.is_file() and bytes_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        work_df = pd.read_pickle(work_path)
        original_bytes = bytes_path.read_bytes()
    except Exception:  # noqa: BLE001
        logger.warning("workspace_load_failed", path=str(dest))
        return None
    return {
        "client_name": meta.get("client_name") or name,
        "safe_client_id": meta.get("safe_client_id") or safe_client_id(name),
        "source_name": meta.get("source_name") or "",
        "original_ext": meta.get("original_ext") or "",
        "sheet": meta.get("sheet", 0),
        "saved_at": meta.get("saved_at") or "",
        "row_count": int(meta.get("row_count") or len(work_df)),
        "work_df": work_df,
        "original_bytes": original_bytes,
    }


def workspace_exists(data_dir: Union[str, Path], client_name: str) -> bool:
    if not is_live():
        return False
    dest = session_dir(data_dir, client_name or "")
    return (dest / META_NAME).is_file() and (dest / WORK_NAME).is_file() \
        and (dest / BYTES_NAME).is_file()


def list_sessions(data_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Metadata for every stored workspace (empty when not live)."""
    if not is_live():
        return []
    root = _root(data_dir)
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        meta_path = child / META_NAME
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta["slug"] = child.name
        out.append(meta)
    return out


def delete_workspace(data_dir: Union[str, Path], client_name: str) -> bool:
    """Remove one client's session folder. Does not touch SQLite. No-op off-live."""
    if not is_live():
        return False
    name = (client_name or "").strip()
    if not name:
        return False
    dest = session_dir(data_dir, name)
    if not dest.exists():
        return False
    shutil.rmtree(dest, ignore_errors=True)
    logger.info("workspace_deleted", client=safe_client_id(name))
    return True
