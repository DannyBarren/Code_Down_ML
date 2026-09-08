"""Load and normalise transaction files (Excel / CSV).

Responsibilities:
    * Read .xlsx / .xls / .csv into a pandas DataFrame.
    * Resolve the messy real-world headers to the logical columns the engine
      expects (via the column-mapping config).
    * Guarantee a ``New Account`` column exists and normalise its codes.
    * Build the combined text used for similarity grouping, robust to the
      Memo/Name/Description variations seen in real-world exports.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.config import Config
from utils.account_codes import normalize_code
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Canonical name we use internally for the target column regardless of the
# source header.
NEW_ACCOUNT_COL = "New Account"
SIM_TEXT_COL = "_sim_text"
# Free-text "Rule Notes" the user adds per transaction. The notes are
# folded into the similarity text (and therefore into ML training) and are
# persisted per row via a stable base signature (see ``BASE_SIG_COL``).
RULE_NOTES_COL = "Rule Notes"
# The transaction text *without* the Rule Notes — stable across uploads, so we
# can re-attach saved notes to the same rows on a future file.
BASE_SIG_COL = "_base_sig"

# Tokens that are pure noise for similarity: long digit runs (check numbers,
# dates, ids, amounts) and repeated whitespace.
_NUMERIC_TOKEN = re.compile(r"\b[\d][\d.,/:\-#]{3,}\b")
_NON_WORD = re.compile(r"[^\w\s&]+")
_WHITESPACE = re.compile(r"\s+")


class DataLoadError(Exception):
    """Raised when an uploaded file cannot be read or understood."""


@dataclass
class LoadedData:
    """Everything downstream stages need about a loaded file."""

    df: pd.DataFrame
    source_name: str
    new_account_col: str = NEW_ACCOUNT_COL
    original_new_account_header: str = NEW_ACCOUNT_COL
    text_columns: List[str] = field(default_factory=list)
    column_map: Dict[str, str] = field(default_factory=dict)
    original_columns: List[str] = field(default_factory=list)
    # Which known export shape the file matched (``""`` = generic export).
    source_profile: str = ""

    @property
    def seed_count(self) -> int:
        return int(self.df[self.new_account_col].apply(_has_value).sum())

    @property
    def blank_count(self) -> int:
        return len(self.df) - self.seed_count


def _has_value(v) -> bool:
    """True if a cell holds a meaningful (non-empty) value."""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    return str(v).strip() != ""


def _resolve_column(candidates: List[str], available: List[str]) -> Optional[str]:
    """Return the first available header matching one of ``candidates``.

    Matching is case-insensitive and ignores surrounding whitespace.
    """
    norm = {str(c).strip().lower(): c for c in available}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None


def load_dataframe(
    file: Union[str, Path, io.BytesIO, bytes],
    config: Config,
    source_name: Optional[str] = None,
    sheet_name: Optional[Union[str, int]] = 0,
) -> LoadedData:
    """Read a file into a normalised :class:`LoadedData`.

    ``file`` may be a path, raw bytes or a file-like object (e.g. a Streamlit
    upload). ``source_name`` is used for logging/history and to pick the reader
    when ``file`` is not a path.
    """
    df, resolved_name = _read_any(file, source_name, sheet_name)

    if df.empty:
        raise DataLoadError("The file was read but contains no rows of data.")

    df.columns = [str(c).strip() for c in df.columns]
    original_columns = list(df.columns)
    logger.info("file_loaded", file=resolved_name, rows=len(df),
                columns=len(original_columns))

    # Resolve / create the New Account column (remember the original header).
    na_col = _resolve_column(config.columns.new_account, original_columns)
    original_header = na_col or NEW_ACCOUNT_COL
    if na_col is None:
        na_col = NEW_ACCOUNT_COL
        df[na_col] = ""
        logger.info("new_account_column_created", name=na_col)
    elif na_col != NEW_ACCOUNT_COL:
        df = df.rename(columns={na_col: NEW_ACCOUNT_COL})
        na_col = NEW_ACCOUNT_COL

    # Normalise the New Account column: clean strings + canonical code form
    # (e.g. "6100 a" -> "6100A"), preserving blanks.
    df[na_col] = df[na_col].apply(
        lambda v: normalize_code(v) if _has_value(v) else ""
    )

    # Recognize known export shapes (AppFolio / QuickBooks) so their headers
    # feed similarity text without the user renaming anything. Detection never
    # fails a load — no winner means the generic resolver runs as before.
    from src.ingest_profiles import detect_profile, profile_similarity_candidates
    profile = detect_profile(original_columns)
    profile_key = profile.key if profile else ""
    if profile:
        logger.info("source_profile_detected", profile=profile.key,
                    file=resolved_name)

    # Determine which columns feed the similarity text.
    sim_cols = _select_text_columns(df, config, original_columns, na_col,
                                    profile=profile)
    if not sim_cols:
        raise DataLoadError(
            "Could not find any text columns to compare transactions. "
            "Please check the column mapping in config.yaml (similarity_columns)."
        )

    df[SIM_TEXT_COL] = _build_text(
        df, sim_cols,
        drop_numbers=config.similarity.drop_numeric_tokens,
        dedupe=config.similarity.dedupe_fields,
    )

    column_map = {logical: _resolve_column([logical], original_columns) or ""
                  for logical in config.columns.similarity_columns}

    return LoadedData(
        df=df,
        source_name=resolved_name,
        new_account_col=na_col,
        original_new_account_header=original_header,
        text_columns=sim_cols,
        column_map=column_map,
        original_columns=original_columns,
        source_profile=profile_key,
    )


def _select_text_columns(
    df: pd.DataFrame,
    config: Config,
    original_columns: List[str],
    na_col: str,
    profile=None,
) -> List[str]:
    """Pick the columns used to build similarity text, with sensible fallbacks.

    When a source profile won detection, its aliases are tried *first* (so e.g.
    a QuickBooks ``Payee`` column feeds the vendor signal even though the
    generic config list only names ``Name``), then the configured columns.
    """
    candidates: List[str] = []
    if profile is not None:
        from src.ingest_profiles import profile_similarity_candidates
        candidates.extend(profile_similarity_candidates(profile))
    candidates.extend(config.similarity_columns_effective())

    sim_cols = [_resolve_column([c], original_columns) for c in candidates]
    sim_cols = [c for c in sim_cols if c]
    if not sim_cols:
        sim_cols = [_resolve_column([c], original_columns)
                    for c in config.columns.text_columns]
        sim_cols = [c for c in sim_cols if c]
    if not sim_cols:
        sim_cols = [c for c in df.columns
                    if c not in (na_col, SIM_TEXT_COL) and df[c].dtype == object]
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered = []
    for c in sim_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _read_any(
    file: Union[str, Path, io.BytesIO, bytes],
    source_name: Optional[str],
    sheet_name: Optional[Union[str, int]],
) -> tuple[pd.DataFrame, str]:
    """Dispatch to the right pandas reader based on extension / content."""
    try:
        if isinstance(file, (str, Path)):
            path = Path(file)
            name = source_name or path.name
            if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
                return pd.read_excel(path, sheet_name=sheet_name, dtype=object), name
            if path.suffix.lower() == ".csv":
                return _read_csv(path), name
            raise DataLoadError(f"Unsupported file type: {path.suffix}")

        # bytes / file-like
        name = source_name or "uploaded_file"
        raw = file.read() if hasattr(file, "read") else file
        buffer = io.BytesIO(raw)
        if name.lower().endswith(".csv"):
            return _read_csv(buffer), name
        # Default to Excel for anything else.
        return pd.read_excel(buffer, sheet_name=sheet_name, dtype=object), name
    except DataLoadError:
        raise
    except ImportError as exc:
        raise DataLoadError(
            "A library needed to read this file is missing "
            f"({exc}). Try: pip install openpyxl"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface a friendly error
        logger.error("file_read_failed", error=str(exc))
        raise DataLoadError(
            f"Could not read the file. Please make sure it is a valid Excel or "
            f"CSV file. (Technical detail: {exc})"
        ) from exc


def _read_csv(source) -> pd.DataFrame:
    """Read CSV, tolerating common encodings."""
    last_err: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, dtype=object, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise DataLoadError(
        "Could not decode the CSV file. Try re-saving it as UTF-8 or as .xlsx."
    ) from last_err


def _clean_fragment(text: str, drop_numbers: bool) -> str:
    """Normalise a single field value into comparable text."""
    s = text.lower().strip()
    if drop_numbers:
        s = _NUMERIC_TOKEN.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


def _build_text(
    df: pd.DataFrame,
    cols: List[str],
    drop_numbers: bool = True,
    dedupe: bool = True,
) -> pd.Series:
    """Concatenate the chosen columns into one normalised text field per row.

    De-duplicates fragments so that, e.g., a vendor name repeated in both
    ``Name`` and ``Memo`` does not dominate the embedding/TF-IDF signal.
    """
    def row_text(row) -> str:
        fragments: List[str] = []
        seen: set[str] = set()
        for c in cols:
            val = row.get(c)
            if not _has_value(val):
                continue
            frag = _clean_fragment(str(val), drop_numbers)
            if not frag:
                continue
            if dedupe and frag in seen:
                continue
            seen.add(frag)
            fragments.append(frag)
        return " | ".join(fragments)

    return df.apply(row_text, axis=1)


def build_base_signature(
    df: pd.DataFrame,
    text_columns: List[str],
    config: Config,
) -> pd.Series:
    """The transaction-only similarity text (no Rule Notes).

    This is identical to what :func:`load_dataframe` puts in ``_sim_text`` for a
    file with no notes, and is what we key persisted Rule Notes on.
    """
    cols = [c for c in text_columns if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return _build_text(
        df, cols,
        drop_numbers=config.similarity.drop_numeric_tokens,
        dedupe=config.similarity.dedupe_fields,
    )


def recompute_sim_text(
    df: pd.DataFrame,
    text_columns: List[str],
    config: Config,
    rule_notes_col: str = RULE_NOTES_COL,
) -> pd.DataFrame:
    """Rebuild ``_base_sig`` and ``_sim_text`` in place.

    ``_sim_text`` is the base transaction text with any non-empty **Rule Notes**
    appended, so that the notes influence both similarity grouping and the
    labelled examples captured for ML training. Mutates and returns ``df``.
    """
    base = build_base_signature(df, text_columns, config)
    df[BASE_SIG_COL] = base

    if rule_notes_col in df.columns:
        notes = df[rule_notes_col].apply(
            lambda v: _clean_fragment(str(v), config.similarity.drop_numeric_tokens)
            if _has_value(v) else "")
        df[SIM_TEXT_COL] = [
            f"{b} | {n}".strip(" |") if n else b
            for b, n in zip(base.tolist(), notes.tolist())
        ]
    else:
        df[SIM_TEXT_COL] = base
    return df


def summarize_upload(
    df: pd.DataFrame,
    new_account_col: str = NEW_ACCOUNT_COL,
) -> Dict[str, int]:
    """Report pre-filled vs blank counts for the ``New Account`` column.

    This is the canonical readout after a load: how many rows already carry a
    code (which rules must preserve) versus how many are blank and eligible to be
    filled. If the column is somehow absent, every row is treated as blank.
    """
    total = int(len(df))
    if new_account_col not in df.columns:
        return {"total": total, "prefilled": 0, "blank": total}
    prefilled = int(df[new_account_col].apply(_has_value).sum())
    return {"total": total, "prefilled": prefilled, "blank": total - prefilled}


def list_excel_sheets(file: Union[str, Path, io.BytesIO, bytes]) -> List[str]:
    """Return sheet names for an Excel file (empty list for CSV)."""
    try:
        if isinstance(file, (str, Path)):
            if Path(file).suffix.lower() == ".csv":
                return []
            return pd.ExcelFile(file).sheet_names
        raw = file.read() if hasattr(file, "read") else file
        return pd.ExcelFile(io.BytesIO(raw)).sheet_names
    except Exception:  # noqa: BLE001
        return []
