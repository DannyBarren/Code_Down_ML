"""Build and apply the human-review queue.

The engine flags two kinds of rows for review:
    * FILLED_REVIEW – a value was applied but confidence is only moderate.
    * NEEDS_REVIEW  – a plausible suggestion exists but was not auto-applied.

This module turns those into a tidy editable table and applies the user's
corrections back onto the working dataframe. Every approval is recorded both as
a learned mapping (instant exact-match memory) **and** as a labelled training
example (so the ML models improve over time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from models.schemas import FillAction, FillResult
from src.data_loader import SIM_TEXT_COL
from utils.account_codes import normalize_code
from utils.logging_setup import get_logger
from utils.storage import Storage

logger = get_logger(__name__)

REVIEW_ACTIONS = (FillAction.FILLED_REVIEW, FillAction.NEEDS_REVIEW)

# Stable, ordered list of the non-text columns the review table always carries.
META_COLUMNS = ["Group", "Current", "Suggested", "New Account", "Confidence",
                "Engine", "Action", "Approve", "Why"]


@dataclass
class ReviewItem:
    row_index: int
    text: str
    current_value: str
    suggested_value: str
    confidence: float
    action: str
    source: str
    rationale: str


def build_review_table(
    df: pd.DataFrame,
    results: List[FillResult],
    na_col: str,
    text_columns: List[str],
    include_no_match: bool = False,
) -> pd.DataFrame:
    """Create a DataFrame of rows that need human attention.

    Includes a boolean ``Approve`` column the UI can toggle and an editable
    ``New Account`` column. Moderate-confidence fills are pre-approved.
    """
    rows = []
    for r in results:
        relevant = r.action in REVIEW_ACTIONS or (
            include_no_match and r.action == FillAction.NO_MATCH
        )
        if not relevant:
            continue
        src = df.iloc[r.row_index]
        record: Dict[str, object] = {"row": r.row_index}
        for c in text_columns:
            if c in df.columns:
                record[c] = src[c]
        record["Group"] = "" if r.group_id is None else f"#{r.group_id}"
        record["Current"] = src[na_col]
        record["Suggested"] = r.proposed_value or ""
        record["New Account"] = (
            src[na_col] if str(src[na_col]).strip() else (r.proposed_value or "")
        )
        record["Confidence"] = round(r.confidence, 3)
        record["Engine"] = r.engine_used
        record["Action"] = r.action.value
        record["Approve"] = r.action == FillAction.FILLED_REVIEW
        record["Why"] = r.rationale
        rows.append(record)

    columns_order = (
        ["row"]
        + [c for c in text_columns if c in df.columns]
        + META_COLUMNS
    )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table[columns_order]
    return table


def filter_by_confidence(
    table: pd.DataFrame, min_conf: float, max_conf: float = 1.0
) -> pd.DataFrame:
    """Return rows whose confidence is within ``[min_conf, max_conf]``."""
    if table.empty:
        return table
    mask = (table["Confidence"] >= min_conf) & (table["Confidence"] <= max_conf)
    return table[mask].reset_index(drop=True)


def mark_high_confidence(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return a copy of ``table`` with ``Approve`` set for confident rows."""
    if table.empty:
        return table
    out = table.copy()
    has_value = out["New Account"].astype(str).str.strip() != ""
    out["Approve"] = (out["Confidence"] >= threshold) & has_value
    return out


def apply_reviews(
    df: pd.DataFrame,
    edited_table: pd.DataFrame,
    na_col: str,
    storage: Storage,
    learn: bool = True,
    approved_by: str = "user",
) -> Dict[str, int]:
    """Apply the user's edited review table back onto ``df`` (in place).

    For every approved row with a value we:
      * write the value into the dataframe (canonicalised),
      * record a learned mapping (exact-match memory),
      * record a labelled training example (for the ML models).

    Returns counts of what changed.
    """
    applied = 0
    learned = 0
    trained = 0
    cleared = 0
    if edited_table is None or edited_table.empty:
        return {"applied": 0, "learned": 0, "trained": 0, "cleared": 0}

    na_idx = df.columns.get_loc(na_col)
    sig_idx = df.columns.get_loc(SIM_TEXT_COL) if SIM_TEXT_COL in df.columns else None

    for _, row in edited_table.iterrows():
        idx = int(row["row"])
        approved = bool(row.get("Approve", False))
        if not approved:
            continue

        raw_value = str(row.get("New Account", "")).strip()
        new_value = normalize_code(raw_value) if raw_value else ""
        current = str(df.iat[idx, na_idx]).strip()

        if new_value:
            if new_value != current:
                df.iat[idx, na_idx] = new_value
                applied += 1
            if sig_idx is not None:
                signature = str(df.iat[idx, sig_idx]).strip()
                if signature:
                    if learn:
                        storage.upsert_learned_mapping(signature, new_value)
                        learned += 1
                    # Capture as labelled training data for the ML models.
                    confidence = _safe_float(row.get("Confidence"), 1.0)
                    engine = str(row.get("Engine", "manual")) or "manual"
                    storage.add_training_example(
                        text=signature, label=new_value, confidence=confidence,
                        engine_used=engine, approved_by=approved_by)
                    trained += 1
        else:
            if current:
                df.iat[idx, na_idx] = ""
                cleared += 1

    logger.info("reviews_applied", applied=applied, learned=learned,
                trained=trained, cleared=cleared)
    return {"applied": applied, "learned": learned, "trained": trained,
            "cleared": cleared}


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
