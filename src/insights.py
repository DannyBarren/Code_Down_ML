"""Read-only coaching insights computed from the working dataframe.

Every function here is pure: it reads the coded rows (plus Amount / Date when
those columns exist) and returns plain data structures for the UI. Nothing in
this module ever writes ``New Account`` — insights describe the book, they
don't change it.

The point for ProfitCoach: coded books are how we coach the client — the more
they share, the sharper this gets.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.data_loader import _has_value
from src.spreadsheet_helpers import ENGINE_COL

# Engine labels that mean "a human owns this code" (upload seed / typed in).
_HUMAN_ENGINES = {"seed", "manual"}


# --------------------------------------------------------------------------- #
# Column + amount resolution (tolerant, profile-aware)
# --------------------------------------------------------------------------- #
def resolve_amount_col(df: pd.DataFrame, config=None) -> Optional[str]:
    """The amount column, tolerantly resolved (None when the file has none)."""
    candidates: List[str] = []
    if config is not None:
        candidates.extend(getattr(config.columns, "amount", []) or [])
    candidates.extend(["Amount", "Amount (USD)", "Debit", "Credit", "Total"])
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = norm.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    # Last resort: any header containing "amount".
    for col in df.columns:
        if "amount" in str(col).lower():
            return col
    return None


def resolve_name_col(df: pd.DataFrame) -> Optional[str]:
    """The vendor/payee column (None when the file has none)."""
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in ("Name", "Payee", "Vendor", "Supplier", "Payee Name"):
        hit = norm.get(cand.lower())
        if hit is not None:
            return hit
    return None


def parse_amount(value) -> float:
    """Parse messy money cells: '1,234.56', '$1,234.56', '(123.45)', floats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return 0.0 if pd.isna(value) else float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    try:
        val = float(s)
    except ValueError:
        return 0.0
    return -val if negative else val


# --------------------------------------------------------------------------- #
# 1. Coding coverage
# --------------------------------------------------------------------------- #
def coding_coverage(work_df: pd.DataFrame, loaded) -> Dict[str, object]:
    """Coded vs blank, split into automation-filled vs human-owned."""
    na_col = loaded.new_account_col
    na = work_df[na_col].astype(str).str.strip()
    coded_mask = na != ""
    engines = (work_df[ENGINE_COL].astype(str)
               if ENGINE_COL in work_df.columns
               else pd.Series([""] * len(work_df), index=work_df.index))

    human = coded_mask & engines.isin(_HUMAN_ENGINES)
    auto = coded_mask & ~engines.isin(_HUMAN_ENGINES)
    total = int(len(work_df))
    coded = int(coded_mask.sum())
    return {
        "total": total,
        "coded": coded,
        "blank": total - coded,
        "pct_coded": (coded / total) if total else 0.0,
        "auto_filled": int(auto.sum()),
        "human": int(human.sum()),
        "seeds": int((engines == "seed").sum()),
        "manual": int((engines == "manual").sum()),
    }


# --------------------------------------------------------------------------- #
# 2. Expense mix — spend by account code
# --------------------------------------------------------------------------- #
def expense_mix(work_df: pd.DataFrame, loaded, config=None,
                top: int = 10) -> Optional[pd.DataFrame]:
    """Sum of Amount per New Account over coded rows (None if no Amount)."""
    amount_col = resolve_amount_col(work_df, config)
    na_col = loaded.new_account_col
    if amount_col is None or na_col not in work_df.columns:
        return None
    coded = work_df[work_df[na_col].astype(str).str.strip() != ""]
    if coded.empty:
        return None
    amounts = coded[amount_col].map(parse_amount)
    mix = (amounts.groupby(coded[na_col].astype(str)).sum()
           .sort_values(ascending=False))
    out = mix.head(top).reset_index()
    out.columns = ["Account", "Spend"]
    out["Spend"] = out["Spend"].round(2)
    return out


# --------------------------------------------------------------------------- #
# 3. Vendor concentration
# --------------------------------------------------------------------------- #
def vendor_concentration(work_df: pd.DataFrame, loaded, config=None,
                         top: int = 10) -> Optional[Dict[str, pd.DataFrame]]:
    """Top vendors by spend and by row count (None if no vendor column)."""
    name_col = resolve_name_col(work_df)
    if name_col is None:
        return None
    names = (work_df[name_col].astype(str).str.strip())
    names = names[names != ""]
    if names.empty:
        return None

    by_count = names.value_counts().head(top).reset_index()
    by_count.columns = ["Vendor", "Rows"]

    amount_col = resolve_amount_col(work_df, config)
    by_spend = None
    if amount_col is not None:
        frame = pd.DataFrame({
            "vendor": work_df[name_col].astype(str).str.strip(),
            "amount": work_df[amount_col].map(parse_amount),
        })
        frame = frame[frame["vendor"] != ""]
        by_spend = (frame.groupby("vendor")["amount"].sum()
                    .sort_values(ascending=False).head(top).reset_index())
        by_spend.columns = ["Vendor", "Spend"]
        by_spend["Spend"] = by_spend["Spend"].round(2)
    return {"by_spend": by_spend, "by_count": by_count}


# --------------------------------------------------------------------------- #
# 4. Uncoded residual — where the next rules should come from
# --------------------------------------------------------------------------- #
def uncoded_residual(work_df: pd.DataFrame, loaded, config=None,
                     top: int = 10) -> Dict[str, object]:
    """Blank-row spend and the vendors with the most uncoded spend."""
    na_col = loaded.new_account_col
    blanks = work_df[work_df[na_col].astype(str).str.strip() == ""]
    amount_col = resolve_amount_col(work_df, config)
    name_col = resolve_name_col(work_df)

    blank_spend: Optional[float] = None
    top_vendors = None
    if amount_col is not None and not blanks.empty:
        blank_spend = round(float(blanks[amount_col].map(parse_amount).sum()),
                            2)
    if name_col is not None and not blanks.empty:
        names = blanks[name_col].astype(str).str.strip()
        names = names[names != ""]
        if not names.empty:
            if amount_col is not None:
                frame = pd.DataFrame({
                    "vendor": names,
                    "amount": blanks.loc[names.index, amount_col]
                    .map(parse_amount),
                })
                tv = (frame.groupby("vendor")["amount"].sum()
                      .sort_values(ascending=False).head(top).reset_index())
                tv.columns = ["Vendor", "Uncoded spend"]
                tv["Uncoded spend"] = tv["Uncoded spend"].round(2)
            else:
                tv = names.value_counts().head(top).reset_index()
                tv.columns = ["Vendor", "Uncoded rows"]
            top_vendors = tv
    return {
        "blank_rows": int(len(blanks)),
        "blank_spend": blank_spend,
        "top_vendors": top_vendors,
    }


# --------------------------------------------------------------------------- #
# 5. Throughput — from real run history, not a fake time series
# --------------------------------------------------------------------------- #
def throughput(runs: list) -> Dict[str, object]:
    """Rows coded across recorded runs (run_history), plus the latest run."""
    total_filled = sum(int(r.auto_filled) + int(r.filled_review) for r in runs)
    last = runs[0] if runs else None
    return {
        "runs": len(runs),
        "rows_coded_all_runs": int(total_filled),
        "last_run_file": (last.file_name or "—") if last else "—",
        "last_run_filled": (int(last.auto_filled) + int(last.filled_review))
        if last else 0,
        "last_run_review": int(last.needs_review) if last else 0,
    }
