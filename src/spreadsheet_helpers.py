"""Pure, testable logic behind the Spreadsheet-Dominant v2.2 UI.

This module is the single source of truth for the **working dataframe**
(``work_df``) that the spreadsheet view edits. Keeping it Streamlit-free means
the risky parts — dataframe/SQLite sync, runs, manual edits, bulk actions,
undo/redo, pagination and the live rule preview — are all unit-testable.

Column conventions
------------------
* The canonical target column keeps its internal name ``"New Account"`` (so the
  engine, exporter and existing tests are unchanged); the UI simply *labels* it
  **"Target Account"**.
* Every app-internal column is ``_``-prefixed so the exporter can strip them:
  ``_sim_text``, ``_base_sig``, ``_confidence``, ``_engine``, ``_action``,
  ``_why``, ``_suggested`` and ``_select``. Real client columns never start
  with an underscore.
* ``Rule Notes`` is a real (user-facing) column whose text is folded into
  ``_sim_text`` so it influences similarity grouping and ML training, and is
  persisted per-row via the stable ``_base_sig`` (transaction text only).
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from models.schemas import FillAction, KeywordRule
from src.config import Config
from src.data_loader import (
    BASE_SIG_COL,
    NEW_ACCOUNT_COL,
    RULE_NOTES_COL,
    SIM_TEXT_COL,
    LoadedData,
    _has_value,
    recompute_sim_text,
)
from src.fill_down_engine import EngineResult, FillDownEngine
from src.rules_manager import RulesManager, RuleMatch
from utils.account_codes import normalize_code
from utils.storage import Storage

# --------------------------------------------------------------------------- #
# Column constants
# --------------------------------------------------------------------------- #
TARGET_ACCOUNT_LABEL = "Target Account"   # display name for NEW_ACCOUNT_COL

SELECT_COL = "_select"
CONF_COL = "_confidence"
ENGINE_COL = "_engine"
ACTION_COL = "_action"
WHY_COL = "_why"
SUGGESTED_COL = "_suggested"

# All meta columns the spreadsheet maintains (besides _sim_text / _base_sig).
META_COLS = [SELECT_COL, CONF_COL, ENGINE_COL, ACTION_COL, WHY_COL, SUGGESTED_COL]
# Columns the user may edit directly in the grid.
EDITABLE_COLS = [SELECT_COL, NEW_ACCOUNT_COL, RULE_NOTES_COL]

# Plain-English names for the engine that produced each value.
ENGINE_FRIENDLY = {
    "seed": "Already coded by you",
    "rules": "Keyword rule",
    "learned": "Learned from past approvals",
    "similarity": "Looks like similar transactions",
    "ml+similarity": "AI + similarity agree",
    "none": "No confident match",
    "manual": "Edited by you",
}


def friendly_engine(engine: str) -> str:
    """Turn an internal engine label into user-friendly text."""
    if not engine:
        return ""
    if engine in ENGINE_FRIENDLY:
        return ENGINE_FRIENDLY[engine]
    if "vs similarity" in engine:
        return "AI and similarity disagree"
    if engine.startswith("ml:"):
        return f"AI model ({engine.split(':', 1)[1]})"
    return engine


# --------------------------------------------------------------------------- #
# Building the working dataframe
# --------------------------------------------------------------------------- #
def build_work_df(loaded: LoadedData, storage: Storage, config: Config) -> pd.DataFrame:
    """Create the authoritative ``work_df`` from a freshly loaded file.

    * Guarantees ``New Account`` and ``Rule Notes`` columns exist.
    * Re-attaches previously saved Rule Notes via the base signature.
    * Folds Rule Notes into ``_sim_text`` and seeds the meta columns.
    """
    df = loaded.df.copy().reset_index(drop=True)

    na_col = loaded.new_account_col
    if na_col not in df.columns:
        df[na_col] = ""
    df[na_col] = df[na_col].apply(
        lambda v: normalize_code(v) if _has_value(v) else "")

    # Rule Notes: keep any that came in the file, then fill blanks from storage.
    if RULE_NOTES_COL not in df.columns:
        df[RULE_NOTES_COL] = ""
    df[RULE_NOTES_COL] = df[RULE_NOTES_COL].apply(
        lambda v: str(v).strip() if _has_value(v) else "")
    _seed_saved_notes(df, loaded, storage, config)

    # Fold notes into _sim_text (and (re)build _base_sig).
    recompute_sim_text(df, loaded.text_columns, config)

    # Meta columns.
    df[SELECT_COL] = False
    df[CONF_COL] = 0.0
    df[ENGINE_COL] = ""
    df[ACTION_COL] = ""
    df[WHY_COL] = ""
    df[SUGGESTED_COL] = ""

    # Pre-mark seed rows (already coded) so the metrics/filters are correct
    # before the first run.
    seed_mask = df[na_col].apply(_has_value)
    df.loc[seed_mask, ACTION_COL] = FillAction.KEPT_SEED.value
    df.loc[seed_mask, CONF_COL] = 1.0
    df.loc[seed_mask, ENGINE_COL] = "seed"
    return df


def _seed_saved_notes(df: pd.DataFrame, loaded: LoadedData, storage: Storage,
                      config: Config) -> int:
    """Fill blank Rule Notes from the persisted {base_sig: notes} store."""
    from src.data_loader import build_base_signature

    lookup = storage.get_rule_notes_lookup()
    if not lookup:
        return 0
    base = build_base_signature(df, loaded.text_columns, config)
    filled = 0
    notes = df[RULE_NOTES_COL].tolist()
    for i, sig in enumerate(base.tolist()):
        if not notes[i] and sig in lookup and lookup[sig]:
            notes[i] = lookup[sig]
            filled += 1
    df[RULE_NOTES_COL] = notes
    return filled


def ensure_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotently guarantee all internal columns exist (defensive)."""
    if RULE_NOTES_COL not in df.columns:
        df[RULE_NOTES_COL] = ""
    defaults = {SELECT_COL: False, CONF_COL: 0.0, ENGINE_COL: "",
                ACTION_COL: "", WHY_COL: "", SUGGESTED_COL: ""}
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


# --------------------------------------------------------------------------- #
# Running the engine against the working dataframe
# --------------------------------------------------------------------------- #
def make_loaded_for_run(work_df: pd.DataFrame, loaded: LoadedData,
                        config: Config) -> LoadedData:
    """Wrap the current ``work_df`` as a :class:`LoadedData` for the engine.

    Recomputes ``_sim_text`` first so the latest Rule Notes are included.
    """
    df = work_df.copy().reset_index(drop=True)
    recompute_sim_text(df, loaded.text_columns, config)
    return LoadedData(
        df=df,
        source_name=loaded.source_name,
        new_account_col=loaded.new_account_col,
        original_new_account_header=loaded.original_new_account_header,
        text_columns=list(loaded.text_columns),
        column_map=dict(loaded.column_map),
        original_columns=list(loaded.original_columns),
    )


def run_full(work_df: pd.DataFrame, loaded: LoadedData, config: Config,
             engine: FillDownEngine, progress_cb=None) -> EngineResult:
    """Run the full intelligent pipeline and sync results into ``work_df``."""
    run_loaded = make_loaded_for_run(work_df, loaded, config)
    result = engine.run(run_loaded, progress_cb=progress_cb)
    apply_run_result(work_df, result, loaded)
    return result


def apply_run_result(work_df: pd.DataFrame, result: EngineResult,
                     loaded: LoadedData) -> None:
    """Merge an :class:`EngineResult` back into ``work_df`` (in place)."""
    na_col = loaded.new_account_col
    ensure_state_columns(work_df)

    # The engine wrote applied fills into result.df; copy the whole column
    # back positionally (both frames share a 0..n-1 RangeIndex).
    if na_col in result.df.columns and len(result.df) == len(work_df):
        work_df[na_col] = list(result.df[na_col])

    conf = work_df[CONF_COL].tolist()
    eng = work_df[ENGINE_COL].tolist()
    act = work_df[ACTION_COL].tolist()
    why = work_df[WHY_COL].tolist()
    sug = work_df[SUGGESTED_COL].tolist()
    for r in result.results:
        i = r.row_index
        if i < 0 or i >= len(work_df):
            continue
        conf[i] = round(float(r.confidence), 3)
        eng[i] = r.engine_used
        act[i] = r.action.value
        why[i] = r.rationale
        sug[i] = r.proposed_value or ""
    work_df[CONF_COL] = conf
    work_df[ENGINE_COL] = eng
    work_df[ACTION_COL] = act
    work_df[WHY_COL] = why
    work_df[SUGGESTED_COL] = sug


# --------------------------------------------------------------------------- #
# Rules-only run + live preview
# --------------------------------------------------------------------------- #
# Values the user owns directly — never overwritten by a rule, even on override.
_PROTECTED_ENGINES = {"seed", "manual"}


def run_rules_only(work_df: pd.DataFrame, rules_manager: RulesManager,
                   loaded: LoadedData, config: Config,
                   indices: Optional[List[int]] = None,
                   overwrite: bool = False) -> int:
    """Apply enabled keyword rules to the given rows.

    * ``overwrite=False`` (default): only fills **blank** rows.
    * ``overwrite=True``: also re-codes rows already filled by an engine guess
      (similarity / learned / ML / a previous rule), because a deterministic
      keyword rule is the most trustworthy signal. Rows the user owns — original
      file **seeds** and **manual** edits — are always preserved.

    Returns the number of cells actually changed.
    """
    recompute_sim_text(work_df, loaded.text_columns, config)
    na_col = loaded.new_account_col
    active = rules_manager.list_rules(enabled_only=True)
    if not active:
        return 0
    if indices is None:
        indices = list(range(len(work_df)))

    na = work_df[na_col].tolist()
    sim = work_df[SIM_TEXT_COL].tolist()
    conf = work_df[CONF_COL].tolist()
    eng = work_df[ENGINE_COL].tolist()
    act = work_df[ACTION_COL].tolist()
    why = work_df[WHY_COL].tolist()
    applied = 0
    for i in indices:
        if i < 0 or i >= len(work_df):
            continue
        existing = str(na[i]).strip()
        if existing:
            if not overwrite:
                continue
            # Protect values the user entered themselves.
            if str(eng[i]) in _PROTECTED_ENGINES:
                continue
        match: Optional[RuleMatch] = rules_manager.match_row(
            sim[i], row=work_df.iloc[i], rules=active)
        if not match:
            continue
        code = normalize_code(match.account_code)
        if existing and code == existing:
            continue  # no change — don't inflate the count
        na[i] = code
        conf[i] = config.confidence.rule_match_confidence
        eng[i] = "rules"
        act[i] = FillAction.AUTO_FILLED.value
        why[i] = f"Matched rule '{match.rule.keyword}' -> {code}."
        applied += 1
    work_df[na_col] = na
    work_df[CONF_COL] = conf
    work_df[ENGINE_COL] = eng
    work_df[ACTION_COL] = act
    work_df[WHY_COL] = why
    return applied


def run_selected_rules_audited(
    work_df: pd.DataFrame,
    rules_manager: RulesManager,
    loaded: LoadedData,
    config: Config,
    indices: Optional[List[int]] = None,
    client_id: Optional[str] = None,
) -> Tuple[int, list]:
    """Deterministic, **blank-only**, fully audited rule run — the primary path.

    Unlike :func:`run_rules_only` (which can re-code engine guesses on an explicit
    selection), this never touches a row that already has a ``New Account`` value,
    honouring the client's hard requirement. It returns ``(applied, results)``
    where ``results`` is a list of :class:`~models.schemas.RuleRunResult` with a
    status of ``protected_existing`` / ``keyword_rule`` / ``no_rule_match`` for
    every row considered — so the UI can show a complete audit table.
    """
    from models.schemas import RuleRunResult

    ensure_state_columns(work_df)
    recompute_sim_text(work_df, loaded.text_columns, config)
    na_col = loaded.new_account_col
    active = RulesManager._sorted_for_apply(
        rules_manager.list_rules(enabled_only=True))
    if indices is None:
        indices = list(work_df.index)

    results: list = []
    applied = 0
    for idx in indices:
        if idx not in work_df.index:
            continue
        try:
            ri = int(idx)
        except (TypeError, ValueError):
            ri = 0

        row = work_df.loc[idx]
        sim = str(work_df.at[idx, SIM_TEXT_COL] or "")
        matched = None
        for rule in active:
            try:
                if rules_manager._rule_matches(rule, sim, row):
                    matched = rule
                    break
            except Exception:  # noqa: BLE001 - never fail the whole run
                continue

        existing = str(work_df.at[idx, na_col] or "").strip()
        if existing:
            # Never overwrite. Record a rule that would have matched, so the
            # audit shows the rule fires even though the cell is already coded.
            rr = RuleRunResult(
                row_index=ri, original_value=existing, proposed_value=existing,
                engine_used=str(work_df.at[idx, ENGINE_COL] or "seed"),
                confidence=1.0, status="protected_existing",
                rationale="New Account already set — preserved.")
            if matched is not None:
                rr.matched_rule_name = matched.notes.strip() or matched.keyword
                rr.matched_pattern = f"{matched.match_type}: {matched.keyword}"
                rr.rationale = (f"Rule '{matched.keyword}' matches, but the "
                                f"existing value ({existing}) is preserved.")
            results.append(rr)
            continue

        if matched is not None:
            code = normalize_code(matched.account_code)
            work_df.at[idx, na_col] = code
            work_df.at[idx, CONF_COL] = config.confidence.rule_match_confidence
            work_df.at[idx, ENGINE_COL] = "rules"
            work_df.at[idx, ACTION_COL] = FillAction.AUTO_FILLED.value
            work_df.at[idx, WHY_COL] = f"Matched rule '{matched.keyword}' -> {code}."
            applied += 1
            results.append(RuleRunResult(
                row_index=ri, proposed_value=code, engine_used="rules",
                confidence=config.confidence.rule_match_confidence,
                status="keyword_rule",
                matched_rule_name=(matched.notes.strip() or matched.keyword),
                matched_pattern=f"{matched.match_type}: {matched.keyword}",
                rationale=f"Matched '{matched.keyword}' "
                          f"({matched.match_type}) -> {code}."))
        else:
            results.append(RuleRunResult(
                row_index=ri, engine_used="none", status="no_rule_match",
                rationale="No keyword rule matched this row."))

    # Reinforce + explain via the account knowledge base (best-effort).
    try:
        rules_manager.reinforce_from_rule_run(
            results, work_df, list(loaded.text_columns), client_id=client_id)
        rules_manager.enrich_results(results, client_id=client_id)
    except Exception:  # noqa: BLE001 - learning must never break a run
        pass
    return applied, results


def pure_rule_audit(results: list, config: Config) -> Dict[str, object]:
    """Build a results/audit payload for a **pure** (deterministic) rule run.

    Same shape the results panel consumes for a hybrid run, but with zero
    semantic fills — so the UI can render both modes with one component while
    making it explicit that only the selected keyword rules ran.
    """
    kw_fills = [r for r in results if r.status == "keyword_rule"]
    protected_results = [r for r in results
                         if r.status == "protected_existing" and r.matched_rule_name]
    protected_existing = sum(1 for r in results
                             if r.status == "protected_existing")
    left_blank = sum(1 for r in results if r.status == "no_rule_match")
    confs = [float(config.confidence.rule_match_confidence)] * len(kw_fills)
    buckets = _bucket_confidences(confs, config)
    return {
        "mode": "pure",
        "keyword_results": kw_fills,
        "protected_results": protected_results,
        "semantic_results": [],
        "filled_rows": [r.row_index for r in kw_fills],
        "keyword_filled": len(kw_fills),
        "semantic_filled": 0,
        "protected_existing": protected_existing,
        "protected_but_matched": len(protected_results),
        "left_blank": left_blank,
        "conf_high": buckets["high"],
        "conf_medium": buckets["medium"],
        "conf_low": buckets["low"],
    }


def matched_records_view(work_df: pd.DataFrame, loaded: LoadedData,
                         indices: List[int]) -> pd.DataFrame:
    """Raw transaction rows for the given indices, using the original columns.

    Strips every internal/meta helper column (anything ``_``-prefixed and
    ``_sim_text``) so the result shows exactly the uploaded spreadsheet columns
    (Date, Name, Memo/Description, Amount, …) plus New Account and Rule Notes,
    in the original order. Used for the "raw matched rows" view + its download.
    """
    valid = [i for i in (indices or []) if i in work_df.index]
    drop = [c for c in work_df.columns
            if str(c).startswith("_") or c == SIM_TEXT_COL]
    if not valid:
        return work_df.drop(columns=drop, errors="ignore").iloc[0:0]
    view = work_df.loc[valid].drop(columns=drop, errors="ignore")
    # Keep original upload order first, then any app-added columns (Rule Notes).
    orig = [c for c in loaded.original_columns if c in view.columns]
    extras = [c for c in view.columns if c not in orig]
    ordered = orig + extras
    return view[ordered] if ordered else view


def run_rules_hybrid(
    work_df: pd.DataFrame,
    rules_manager: RulesManager,
    loaded: LoadedData,
    config: Config,
    engine: FillDownEngine,
    indices: Optional[List[int]] = None,
    client_id: Optional[str] = None,
) -> Tuple[int, Dict[str, object]]:
    """Hybrid rule run: strict keyword rules first, then semantic propagation.

    Steps:
      1. Apply keyword rules to blank rows (deterministic, high confidence).
      2. Treat the rule-filled rows **plus** existing coded rows as
         high-confidence seeds.
      3. Run semantic similarity over the remaining blank rows in the current
         file, filling with a confidence score + rationale.

    Existing ``New Account`` values are never overwritten. Returns
    ``(total_filled, audit)`` where ``audit`` cleanly separates keyword fills,
    semantic fills, protected rows and confidence buckets for review.
    """
    na_col = loaded.new_account_col
    # Rows already coded *before* this run (true user/seed values).
    pre_coded = {int(i) for i in work_df.index
                 if str(work_df.at[i, na_col] or "").strip()}

    # 1) Keyword rules (blank-only, audited). These fills become seeds too.
    _, kw_results = run_selected_rules_audited(
        work_df, rules_manager, loaded, config, indices, client_id=client_id)
    kw_fills = [r for r in kw_results if r.status == "keyword_rule"]
    rule_idx = {r.row_index for r in kw_fills}
    protected_results = [r for r in kw_results
                         if r.status == "protected_existing" and r.matched_rule_name]
    protected_but_matched = len(protected_results)

    # 2/3) Semantic propagation seeded from (existing coded ∪ rule-filled) rows.
    run_loaded = make_loaded_for_run(work_df, loaded, config)
    result = engine.run(run_loaded)
    apply_run_result(work_df, result, loaded)

    # Re-stamp rule-filled rows: the engine saw them as seeds and relabelled
    # them "seed"; restore the truthful "keyword rule" provenance + confidence.
    kw_by_idx = {r.row_index: r for r in kw_fills}
    for i, r in kw_by_idx.items():
        if i in work_df.index:
            work_df.at[i, CONF_COL] = config.confidence.rule_match_confidence
            work_df.at[i, ENGINE_COL] = "rules"
            work_df.at[i, ACTION_COL] = FillAction.AUTO_FILLED.value
            work_df.at[i, WHY_COL] = r.rationale

    # Build the semantic-fill list from the engine's per-row decisions,
    # excluding rows that were protected or already filled by a keyword rule.
    semantic_fills: List[Dict[str, object]] = []
    left_blank = 0
    fill_actions = {FillAction.AUTO_FILLED.value, FillAction.FILLED_REVIEW.value}
    for fr in result.results:
        i = int(fr.row_index)
        if i in rule_idx or i in pre_coded:
            continue
        if fr.proposed_value and fr.action.value in fill_actions:
            semantic_fills.append({
                "row_index": i,
                "code": normalize_code(fr.proposed_value),
                "confidence": float(fr.confidence),
                "engine": fr.engine_used,
                "rationale": fr.rationale,
            })
        else:
            left_blank += 1

    # Confidence buckets across *this run's* fills (keyword + semantic).
    confs = ([float(config.confidence.rule_match_confidence)] * len(kw_fills)
             + [s["confidence"] for s in semantic_fills])
    buckets = _bucket_confidences(confs, config)

    audit: Dict[str, object] = {
        "mode": "hybrid",
        "keyword_results": kw_fills,
        "protected_results": protected_results,
        "semantic_results": semantic_fills,
        "filled_rows": ([r.row_index for r in kw_fills]
                        + [int(s["row_index"]) for s in semantic_fills]),
        "keyword_filled": len(kw_fills),
        "semantic_filled": len(semantic_fills),
        "protected_existing": len(pre_coded),
        "protected_but_matched": protected_but_matched,
        "left_blank": left_blank,
        "conf_high": buckets["high"],
        "conf_medium": buckets["medium"],
        "conf_low": buckets["low"],
        "scoped": bool(indices),
    }
    return len(kw_fills) + len(semantic_fills), audit


def _bucket_confidences(confs: List[float], config: Config) -> Dict[str, int]:
    """Split confidence values into high / medium / low using config cutoffs."""
    hi = config.confidence.auto_apply_cutoff      # e.g. 0.85
    lo = config.confidence.review_cutoff          # e.g. 0.55
    high = sum(1 for c in confs if c >= hi)
    medium = sum(1 for c in confs if lo <= c < hi)
    low = sum(1 for c in confs if c < lo)
    return {"high": high, "medium": medium, "low": low}


CONF_LEVELS = ("All", "High", "Medium", "Low")


def confidence_level_range(level: str, config: Config) -> Optional[Tuple[float, float]]:
    """Map a confidence-level label to a (low, high) inclusive range, or None."""
    hi = config.confidence.auto_apply_cutoff
    lo = config.confidence.review_cutoff
    if level == "High":
        return (hi, 1.0)
    if level == "Medium":
        return (lo, hi - 1e-9)
    if level == "Low":
        return (0.0, lo - 1e-9)
    return None


def confidence_buckets(work_df: pd.DataFrame, loaded: LoadedData,
                       config: Config) -> Dict[str, int]:
    """High / medium / low counts across the currently **coded** rows.

    Only rows that carry a New Account value are counted (blank rows have no
    confidence to review).
    """
    na_col = loaded.new_account_col
    na = work_df[na_col].astype(str).str.strip()
    filled = na != ""
    if CONF_COL not in work_df.columns:
        return {"high": 0, "medium": 0, "low": 0}
    confs = work_df.loc[filled, CONF_COL].astype(float).tolist()
    return _bucket_confidences(confs, config)


def rule_preview(work_df: pd.DataFrame, keyword: str, match_type: str,
                 case_sensitive: bool, fields: List[str], loaded: LoadedData,
                 config: Config, sample: int = 8) -> Tuple[int, pd.DataFrame]:
    """Count and sample the rows a prospective rule would match (live preview)."""
    keyword = (keyword or "").strip()
    if not keyword:
        return 0, pd.DataFrame()
    recompute_sim_text(work_df, loaded.text_columns, config)
    try:
        rule = KeywordRule(keyword=keyword, account_code="0",
                           match_type=match_type, case_sensitive=case_sensitive,
                           fields=list(fields or []))
    except Exception:  # noqa: BLE001 - invalid keyword/code -> no matches
        return 0, pd.DataFrame()

    sim = work_df[SIM_TEXT_COL].tolist()
    matched_idx: List[int] = []
    for i in range(len(work_df)):
        if RulesManager._rule_matches(rule, sim[i], work_df.iloc[i]):
            matched_idx.append(i)

    text_cols = [c for c in loaded.text_columns if c in work_df.columns]
    show = text_cols + [loaded.new_account_col]
    sample_df = work_df.iloc[matched_idx[:sample]][show].copy() \
        if matched_idx else pd.DataFrame(columns=show)
    return len(matched_idx), sample_df


# Logical columns users expect to keyword-match against, vendor-first so
# the default keyword favours the vendor name over a generic memo.
_RULE_COLUMN_CANDIDATES = ("Name", "Payee", "Vendor", "Description", "Memo")


def _column_has_text(work_df: pd.DataFrame, col: str, sample: int = 40) -> bool:
    """True if a column carries alphabetic content worth keyword-matching.

    Filters out pure number/date columns (Amount, Date, check #s) which only add
    noise to the keyword picker and never make good rule keywords.
    """
    series = work_df[col].dropna()
    if series.empty:
        return False
    hits = 0
    checked = 0
    for val in series.head(sample).tolist():
        text = str(val).strip()
        if not text:
            continue
        checked += 1
        if _keywords_from_text(text):
            hits += 1
    return checked > 0 and (hits / checked) >= 0.2


def rule_keyword_columns(work_df: pd.DataFrame, loaded: LoadedData) -> List[str]:
    """Text columns to mine for rule keywords, vendor-first.

    Resolves Name / Payee / Description / Memo (and similar) and drops columns
    that are essentially numeric/date (e.g. Amount, Date) so the keyword picker
    only ever offers useful, text-bearing columns.
    """
    available = [c for c in mining_columns(work_df, loaded)
                 if _column_has_text(work_df, c)]
    norm = {c.strip().lower(): c for c in available}
    out: List[str] = []
    for cand in _RULE_COLUMN_CANDIDATES:
        col = norm.get(cand.lower())
        if col and col not in out:
            out.append(col)
    for c in available:
        if c not in out:
            out.append(c)
    return out


def suggest_keyword_from_cell(work_df: pd.DataFrame, row_idx: int,
                              column: Optional[str], loaded: LoadedData) -> str:
    """Best keyword from one row's column (column-focused rule creation)."""
    if row_idx < 0 or row_idx >= len(work_df):
        return ""
    if column and column in work_df.columns:
        raw = str(work_df.iloc[row_idx][column] or "").strip()
        if raw:
            toks = _keywords_from_text(raw)
            if len(toks) >= 2:
                return " ".join(toks[:3])
            if toks:
                return toks[0]
            if len(raw) <= 40:
                return raw
            return " ".join(raw.split()[:3])
    return suggest_keyword_from_rows(work_df, [row_idx], loaded)


def rule_creation_prefill(
    work_df: pd.DataFrame,
    indices: List[int],
    loaded: LoadedData,
    *,
    column: Optional[str] = None,
    rules_manager: Optional[RulesManager] = None,
) -> Dict[str, object]:
    """Unified prefill payload for the rule-creation dialog."""
    indices = [int(i) for i in indices if 0 <= int(i) < len(work_df)]
    na_col = loaded.new_account_col
    keyword = ""
    if len(indices) == 1 and column:
        keyword = suggest_keyword_from_cell(work_df, indices[0], column, loaded)
    if not keyword:
        keyword = suggest_keyword_from_rows(work_df, indices, loaded)
    codes = [str(work_df.iloc[i][na_col]).strip()
             for i in indices if str(work_df.iloc[i][na_col]).strip()]
    code = ""
    if codes:
        from collections import Counter
        code = Counter(codes).most_common(1)[0][0]
    notes = suggested_notes_from_rows(work_df, indices)
    fields: List[str] = [column] if column else []
    return {
        "keyword": keyword,
        "code": code,
        "notes": notes,
        "fields": fields,
        "indices": indices,
        "column": column or "",
        "notes_placeholder": (
            "Optional hint for the matcher (e.g. recurring vendor, tax type)"
            if not notes else ""),
    }


def rule_prompt_worthy(keyword: str, code: str,
                       rules_manager: RulesManager) -> bool:
    """True when a quick 'create rule?' prompt is appropriate."""
    keyword = (keyword or "").strip()
    code = (code or "").strip()
    if not keyword or not code:
        return False
    if keyword.lower() in existing_rule_keywords(rules_manager):
        return False
    return True


def suggest_keyword_from_rows(work_df: pd.DataFrame, indices: List[int],
                              loaded: LoadedData) -> str:
    """Best-guess, *specific* keyword from the selected rows.

    Prefers the meaningful tokens shared by **every** selected row (so picking
    "Office Depot" rows yields ``"office depot"``, not the ambiguous ``"depot"``
    that would also hit Home Depot). Falls back to the most frequent token.
    """
    if not indices:
        return ""
    cols = rule_keyword_columns(work_df, loaded)
    if not cols:
        cols = mining_columns(work_df, loaded)
    if not cols:
        return ""

    # Single row: a reusable 1-2 token vendor phrase from the highest-priority
    # text column that yields tokens (avoids over-specific multi-word keywords
    # that would only match the one row).
    if len(indices) == 1:
        i = indices[0]
        if 0 <= i < len(work_df):
            for c in cols:
                toks = _keywords_from_text(str(work_df.iloc[i][c] or ""))
                if toks:
                    return " ".join(toks[:2])
        return ""

    per_row: List[List[str]] = []
    for i in indices:
        if 0 <= i < len(work_df):
            text = " ".join(str(work_df.iloc[i][c] or "") for c in cols)
            toks = _keywords_from_text(text)
            if toks:
                per_row.append(toks)
    if not per_row:
        return ""

    # Tokens common to all selected rows, kept in the first row's order.
    common = set(per_row[0])
    for toks in per_row[1:]:
        common &= set(toks)
    if common:
        ordered: List[str] = []
        for t in per_row[0]:
            if t in common and t not in ordered:
                ordered.append(t)
        return " ".join(ordered[:3])

    from collections import Counter
    counter = Counter(t for toks in per_row for t in toks)
    return counter.most_common(1)[0][0] if counter else ""


# Common function/finance words that make poor, over-generic rule keywords.
_KEYWORD_STOPWORDS = {
    "and", "the", "for", "with", "from", "inc", "llc", "ltd", "corp", "company",
    "payment", "payments", "invoice", "invoices", "expense", "expenses", "fee",
    "fees", "bill", "income", "charge", "charges", "misc", "various", "general",
    "other", "account", "monthly", "annual", "service", "services", "vendor",
}


# Alphabetic word tokens (letters and ampersand). Dropping anything with digits
# keeps dates/amounts/check-numbers/ids out of suggested keywords.
_WORD_RE = re.compile(r"[a-z][a-z&]+")


def _keywords_from_text(text: str) -> List[str]:
    """Lowercased alphabetic tokens worth using as a rule keyword.

    Drops numbers, dates, amounts, very short tokens and a small set of generic
    finance/function words so suggested rules key on vendor-specific terms.
    """
    out: List[str] = []
    for tok in _WORD_RE.findall(str(text or "").lower()):
        if len(tok) >= 3 and tok not in _KEYWORD_STOPWORDS:
            out.append(tok)
    return out


def existing_rule_keywords(rules_manager: RulesManager) -> set:
    """Lowercased set of keywords already covered by saved rules (any state)."""
    return {r.keyword.strip().lower()
            for r in rules_manager.list_rules() if (r.keyword or "").strip()}


def mining_columns(work_df: pd.DataFrame, loaded: LoadedData) -> List[str]:
    """Real text columns to mine for rule keywords.

    Broader than ``loaded.text_columns`` (the *similarity* columns) on purpose:
    a vendor often lives in a column that isn't configured for similarity
    (e.g. **Payee** or **Split**). Mining only the similarity columns is why
    seeded vendors sometimes produced no candidate rules. We include every
    original client column except the answer column and Rule Notes; numeric
    junk is dropped later by :func:`_keywords_from_text`.
    """
    cols: List[str] = []
    seen: set = set()
    ordered = list(loaded.text_columns) + list(loaded.original_columns or [])
    na_col = loaded.new_account_col
    for c in ordered:
        if c in seen or c not in work_df.columns:
            continue
        if c == na_col or c == RULE_NOTES_COL or str(c).startswith("_"):
            continue
        seen.add(c)
        cols.append(c)
    return cols


def candidate_rules(work_df: pd.DataFrame, loaded: LoadedData,
                    rules_manager: RulesManager, *, min_support: int = 1,
                    max_candidates: int = 25, min_purity: float = 0.7,
                    max_match_frac: float = 0.6) -> List[Dict]:
    """Suggest keyword rules from rows that already have a Target Account.

    The *primary* rule-creation workflow: users code a few rows
    (seeding), and we mine those coded rows for a representative keyword per
    account code that isn't already covered by an existing rule. Pure + fully
    deterministic so it can be unit-tested and previewed before anything is saved.

    A keyword is only suggested if it is **specific** to its account code:

    * ``min_purity`` — of the *coded* rows containing the keyword, at least this
      fraction must already carry the proposed code. This rejects generic words
      like "expenses" or "payment" that appear across many different accounts.
    * ``max_match_frac`` — a keyword matching more than this fraction of the whole
      sheet is treated as too broad (unless it is almost perfectly pure).

    Returns dicts with ``keyword``, ``account_code``, ``description``,
    ``support`` (coded rows of this code containing the keyword), ``matches``
    (rows in the whole sheet the rule would touch) and ``purity``.
    """
    from collections import Counter, defaultdict

    from utils.account_codes import describe

    na_col = loaded.new_account_col
    if na_col not in work_df.columns:
        return []
    text_cols = mining_columns(work_df, loaded)
    if not text_cols:
        return []

    na = work_df[na_col].astype(str).str.strip().tolist()
    coded_norm = [normalize_code(c) if c else "" for c in na]
    # Combined text per row, drawn from *all* text columns so a short primary
    # column (e.g. a one-letter Description) never starves the suggestions.
    combined = (work_df[text_cols].fillna("").astype(str)
                .agg(" ".join, axis=1).tolist())
    combined_lower = [c.lower() for c in combined]
    total_rows = len(work_df) or 1

    code_rows: Dict[str, List[int]] = defaultdict(list)
    for i, code in enumerate(coded_norm):
        if code:
            code_rows[code].append(i)

    used = set(existing_rule_keywords(rules_manager))
    candidates: List[Dict] = []
    for code, rows in code_rows.items():
        counter: Counter = Counter()
        for i in rows:
            for tok in set(_keywords_from_text(combined[i])):
                counter[tok] += 1
        chosen = None
        # Deterministic: highest support first, then alphabetical for stable ties.
        for tok, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if cnt < min_support:
                break
            if tok in used:
                continue
            # One pass computes total matches and the code purity of the token.
            total = coded_same = coded_any = 0
            for i, text in enumerate(combined_lower):
                if tok in text:
                    total += 1
                    if coded_norm[i]:
                        coded_any += 1
                        if coded_norm[i] == code:
                            coded_same += 1
            purity = (coded_same / coded_any) if coded_any else 1.0
            too_broad = total > max_match_frac * total_rows and purity < 0.95
            if purity < min_purity or too_broad:
                continue
            chosen = (tok, cnt, total, purity)
            break
        if not chosen:
            continue
        tok, support, total, purity = chosen
        used.add(tok)
        candidates.append({
            "keyword": tok,
            "account_code": code,
            "description": describe(code) or "",
            "support": int(support),
            "matches": int(total),
            "purity": round(float(purity), 3),
        })

    # Surface the most specific, highest-impact suggestions first.
    candidates.sort(key=lambda c: (-c["purity"], -c["matches"], c["keyword"]))
    return candidates[:max_candidates]


def create_rules_from_candidates(rules_manager: RulesManager,
                                 candidates: List[Dict],
                                 match_type: str = "contains") -> int:
    """Persist the chosen candidate rules (skipping any already-covered keyword)."""
    existing = existing_rule_keywords(rules_manager)
    created = 0
    for c in candidates:
        kw = str(c.get("keyword", "")).strip()
        code = str(c.get("account_code", "")).strip()
        if not kw or not code or kw.lower() in existing:
            continue
        rules_manager.add_rule(kw, code, match_type=match_type,
                               case_sensitive=False, fields=[],
                               notes="Auto-suggested from your coded rows.")
        existing.add(kw.lower())
        created += 1
    return created


def suggested_notes_from_rows(work_df: pd.DataFrame, indices: List[int]) -> str:
    """Pre-fill Rule Notes for a new rule from any notes already on the rows."""
    seen: List[str] = []
    for i in indices:
        if 0 <= i < len(work_df) and RULE_NOTES_COL in work_df.columns:
            note = str(work_df.iloc[i][RULE_NOTES_COL] or "").strip()
            if note and note not in seen:
                seen.append(note)
    return "; ".join(seen)


# --------------------------------------------------------------------------- #
# Manual edits committed from the grid (incl. pagination)
# --------------------------------------------------------------------------- #
def commit_editor_changes(work_df: pd.DataFrame, edited: pd.DataFrame,
                          loaded: LoadedData, storage: Storage,
                          config: Config) -> Dict[str, object]:
    """Reconcile an edited page back into ``work_df`` (in place).

    ``edited`` is the dataframe returned by ``st.data_editor`` for the current
    page; its index aligns with ``work_df``. Only the editable columns
    (selection, Target Account, Rule Notes) are trusted.

    Returns change counts plus ``target_edits`` — rows where the user newly
    typed a Target Account (used for the inline "create rule?" prompt).
    """
    ensure_state_columns(work_df)
    na_col = loaded.new_account_col
    changed_targets = 0
    changed_notes = 0
    notes_dirty = False
    target_edits: List[Dict[str, object]] = []

    for idx in edited.index:
        if idx not in work_df.index:
            continue
        # Selection (cheap, no side effects).
        if SELECT_COL in edited.columns:
            work_df.at[idx, SELECT_COL] = bool(edited.at[idx, SELECT_COL])

        # Rule Notes.
        if RULE_NOTES_COL in edited.columns:
            new_note = str(edited.at[idx, RULE_NOTES_COL] or "").strip()
            old_note = str(work_df.at[idx, RULE_NOTES_COL] or "").strip()
            if new_note != old_note:
                work_df.at[idx, RULE_NOTES_COL] = new_note
                changed_notes += 1
                notes_dirty = True

        # Target Account (canonical New Account).
        if na_col in edited.columns:
            raw = str(edited.at[idx, na_col] or "").strip()
            new_val = normalize_code(raw) if raw else ""
            old_val = str(work_df.at[idx, na_col] or "").strip()
            if new_val != old_val:
                work_df.at[idx, na_col] = new_val
                work_df.at[idx, ACTION_COL] = (
                    FillAction.KEPT_SEED.value if new_val else "")
                work_df.at[idx, CONF_COL] = 1.0 if new_val else 0.0
                work_df.at[idx, ENGINE_COL] = "manual" if new_val else ""
                work_df.at[idx, WHY_COL] = (
                    "Entered by you." if new_val else "")
                changed_targets += 1
                if new_val:
                    kw = suggest_keyword_from_rows(work_df, [int(idx)], loaded)
                    target_edits.append({
                        "row": int(idx),
                        "code": new_val,
                        "keyword": kw,
                    })

    # Recompute sim_text if notes changed, then persist notes + learn edits.
    if notes_dirty:
        recompute_sim_text(work_df, loaded.text_columns, config)
        _persist_changed_notes(work_df, storage)

    # Learn manual target edits (after any sim_text recompute).
    if changed_targets:
        _learn_filled_rows(work_df, loaded, storage,
                           only_engine="manual")
    return {
        "targets": changed_targets,
        "notes": changed_notes,
        "target_edits": target_edits,
    }


def _persist_changed_notes(work_df: pd.DataFrame, storage: Storage) -> None:
    """Upsert Rule Notes keyed by base signature.

    Identical transactions share a base signature, so we aggregate first and let
    a non-empty note win over blank duplicates. Writing per-row would otherwise
    let a blank duplicate row *delete* a note just saved on its sibling.
    """
    if BASE_SIG_COL not in work_df.columns:
        return
    by_sig: Dict[str, str] = {}
    for sig, note in zip(work_df[BASE_SIG_COL].tolist(),
                         work_df[RULE_NOTES_COL].tolist()):
        sig = str(sig).strip()
        if not sig:
            continue
        note = str(note or "").strip()
        if sig not in by_sig or (note and not by_sig[sig]):
            by_sig[sig] = note
    for sig, note in by_sig.items():
        storage.upsert_rule_note(sig, note)


def _learn_filled_rows(work_df: pd.DataFrame, loaded: LoadedData,
                       storage: Storage, indices: Optional[List[int]] = None,
                       only_engine: Optional[str] = None) -> int:
    """Record learned mappings + training examples for filled rows."""
    na_col = loaded.new_account_col
    if SIM_TEXT_COL not in work_df.columns:
        return 0
    if indices is None:
        indices = list(work_df.index)
    learned = 0
    for idx in indices:
        if idx not in work_df.index:
            continue
        if only_engine is not None and \
                str(work_df.at[idx, ENGINE_COL]) != only_engine:
            continue
        code = str(work_df.at[idx, na_col] or "").strip()
        sig = str(work_df.at[idx, SIM_TEXT_COL] or "").strip()
        if code and sig:
            storage.upsert_learned_mapping(sig, code)
            storage.add_training_example(
                text=sig, label=code, confidence=1.0,
                engine_used="manual_edit")
            learned += 1
    return learned


# --------------------------------------------------------------------------- #
# Bulk actions
# --------------------------------------------------------------------------- #
def selected_indices(work_df: pd.DataFrame) -> List[int]:
    if SELECT_COL not in work_df.columns:
        return []
    return [int(i) for i in work_df.index[work_df[SELECT_COL] == True]]  # noqa: E712


def set_selection(work_df: pd.DataFrame, indices: List[int], value: bool = True) -> None:
    ensure_state_columns(work_df)
    work_df.loc[work_df.index.isin(indices), SELECT_COL] = bool(value)


def clear_selection(work_df: pd.DataFrame) -> None:
    if SELECT_COL in work_df.columns:
        work_df[SELECT_COL] = False


def clear_spreadsheet_values(work_df: pd.DataFrame, loaded: LoadedData,
                             config: Config) -> int:
    """Wipe all Target Account + Rule Notes values and reset decision metadata.

    The original client columns are left untouched — only the app's editable
    outputs (codes, notes) and derived meta (confidence, engine, action, why,
    suggestion, selection) are reset to a clean slate. Returns the row count.
    """
    ensure_state_columns(work_df)
    na_col = loaded.new_account_col
    work_df[na_col] = ""
    work_df[RULE_NOTES_COL] = ""
    work_df[CONF_COL] = 0.0
    work_df[ENGINE_COL] = ""
    work_df[ACTION_COL] = ""
    work_df[WHY_COL] = ""
    work_df[SUGGESTED_COL] = ""
    work_df[SELECT_COL] = False
    # Rebuild _sim_text so the cleared notes no longer influence matching.
    recompute_sim_text(work_df, loaded.text_columns, config)
    return int(len(work_df))


def reset_automation(work_df: pd.DataFrame, loaded: LoadedData,
                     config: Config) -> int:
    """"Un-run" the automation so the user can adjust rules and run again.

    Clears every value the engine produced — keyword rules, semantic similarity,
    learned memory and the ML model — and resets the derived meta (confidence,
    engine, action, why, suggestion, selection) for those rows. Deliberately
    **preserves**:

      * codes that came in with the upload (``engine == "seed"``),
      * codes the user typed themselves (``engine == "manual"``),
      * all Rule Notes, and
      * every saved keyword rule / learned mapping (untouched here).

    Returns how many automation-filled rows were cleared.
    """
    ensure_state_columns(work_df)
    na_col = loaded.new_account_col

    na = work_df[na_col].tolist()
    conf = work_df[CONF_COL].tolist()
    eng = work_df[ENGINE_COL].astype(str).tolist()
    act = work_df[ACTION_COL].tolist()
    why = work_df[WHY_COL].tolist()
    sug = work_df[SUGGESTED_COL].tolist()

    cleared = 0
    for i in range(len(work_df)):
        engine_label = (eng[i] or "").strip()
        if engine_label in _PROTECTED_ENGINES:
            continue  # user-owned (upload seed or manual edit) — keep as-is
        had_output = (str(na[i]).strip() != ""
                      or str(sug[i]).strip() != "")
        na[i] = ""
        conf[i] = 0.0
        eng[i] = ""
        act[i] = ""
        why[i] = ""
        sug[i] = ""
        if had_output:
            cleared += 1

    work_df[na_col] = na
    work_df[CONF_COL] = conf
    work_df[ENGINE_COL] = eng
    work_df[ACTION_COL] = act
    work_df[WHY_COL] = why
    work_df[SUGGESTED_COL] = sug
    clear_selection(work_df)
    return cleared


def approve_rows(work_df: pd.DataFrame, loaded: LoadedData, storage: Storage,
                 config: Config, indices: Optional[List[int]] = None) -> Dict[str, int]:
    """Approve rows: fill from suggestion if blank, learn, and mark resolved.

    If ``indices`` is None, approves the current selection. Returns counts.
    """
    ensure_state_columns(work_df)
    recompute_sim_text(work_df, loaded.text_columns, config)
    na_col = loaded.new_account_col
    if indices is None:
        indices = selected_indices(work_df)
    applied = 0
    learned = 0
    for idx in indices:
        if idx not in work_df.index:
            continue
        value = str(work_df.at[idx, na_col] or "").strip()
        if not value:
            value = str(work_df.at[idx, SUGGESTED_COL] or "").strip()
        if not value:
            continue
        value = normalize_code(value)
        work_df.at[idx, na_col] = value
        work_df.at[idx, ACTION_COL] = FillAction.KEPT_SEED.value
        work_df.at[idx, CONF_COL] = 1.0
        if not str(work_df.at[idx, ENGINE_COL]).strip():
            work_df.at[idx, ENGINE_COL] = "manual"
        applied += 1
        sig = str(work_df.at[idx, SIM_TEXT_COL] or "").strip()
        if sig:
            storage.upsert_learned_mapping(sig, value)
            storage.add_training_example(
                text=sig, label=value, confidence=1.0, engine_used="approved")
            learned += 1
    clear_selection(work_df)
    return {"applied": applied, "learned": learned}


def set_target_for_rows(work_df: pd.DataFrame, indices: List[int], code: str,
                        loaded: LoadedData) -> int:
    """Bulk-set the Target Account for rows (used by 'apply suggestion')."""
    ensure_state_columns(work_df)
    na_col = loaded.new_account_col
    code = normalize_code(code) if str(code).strip() else ""
    count = 0
    for idx in indices:
        if idx in work_df.index:
            work_df.at[idx, na_col] = code
            work_df.at[idx, ACTION_COL] = (
                FillAction.KEPT_SEED.value if code else "")
            work_df.at[idx, CONF_COL] = 1.0 if code else 0.0
            work_df.at[idx, ENGINE_COL] = "manual" if code else ""
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Filters, metrics, pagination
# --------------------------------------------------------------------------- #
REVIEW_ACTIONS = {FillAction.FILLED_REVIEW.value, FillAction.NEEDS_REVIEW.value}


def filter_mask(work_df: pd.DataFrame, mode: str,
                high_conf_cutoff: float = 0.85) -> pd.Series:
    """Boolean mask for the toolbar quick-view filters."""
    na = work_df[NEW_ACCOUNT_COL].astype(str).str.strip()
    if mode == "Review only":
        return work_df[ACTION_COL].isin(REVIEW_ACTIONS)
    if mode == "High confidence":
        return work_df[CONF_COL].astype(float) >= high_conf_cutoff
    if mode == "Blanks only":
        return na == ""
    if mode == "Filled only":
        return na != ""
    return pd.Series(True, index=work_df.index)


def searchable_columns(work_df: pd.DataFrame, loaded: LoadedData) -> List[str]:
    """User-facing columns the global search should scan.

    Client text columns + Target Account + Rule Notes. Internal ``_`` columns
    are never searched.
    """
    na_col = loaded.new_account_col
    cols = list(mining_columns(work_df, loaded))
    for extra in (na_col, RULE_NOTES_COL):
        if extra in work_df.columns and extra not in cols:
            cols.append(extra)
    return cols


def search_mask(work_df: pd.DataFrame, query: str,
                columns: Optional[List[str]] = None) -> pd.Series:
    """Case-insensitive, all-records search across columns.

    Multiple whitespace-separated terms are AND-ed (every term must appear
    somewhere in the row's searched text). Operates on the *entire* dataframe,
    so results are never limited to the current page.
    """
    q = str(query or "").strip().lower()
    if not q:
        return pd.Series(True, index=work_df.index)
    if columns:
        cols = [c for c in columns if c in work_df.columns]
    else:
        cols = [c for c in work_df.columns if not str(c).startswith("_")
                and c != SIM_TEXT_COL]
    if not cols:
        return pd.Series(False, index=work_df.index)

    hay = work_df[cols[0]].astype(str).fillna("")
    for c in cols[1:]:
        hay = hay.str.cat(work_df[c].astype(str).fillna(""), sep=" \u0001 ")
    hay = hay.str.lower()

    mask = pd.Series(True, index=work_df.index)
    for term in q.split():
        mask &= hay.str.contains(re.escape(term), regex=True, na=False)
    return mask


def available_engines(work_df: pd.DataFrame) -> List[str]:
    """Sorted friendly 'how decided' labels present in the data (for filtering)."""
    if ENGINE_COL not in work_df.columns:
        return []
    vals = work_df[ENGINE_COL].astype(str).str.strip()
    vals = vals[vals != ""]
    return sorted({friendly_engine(v) for v in vals.unique()})


def engine_mask(work_df: pd.DataFrame, engines: List[str]) -> pd.Series:
    """Mask rows whose friendly engine label is in ``engines``."""
    if not engines or ENGINE_COL not in work_df.columns:
        return pd.Series(True, index=work_df.index)
    friendly = work_df[ENGINE_COL].astype(str).map(friendly_engine)
    return friendly.isin(set(engines))


def build_view_mask(work_df: pd.DataFrame, loaded: LoadedData, *,
                    mode: str = "All", query: str = "",
                    search_columns: Optional[List[str]] = None,
                    engines: Optional[List[str]] = None,
                    conf_range: Optional[Tuple[float, float]] = None,
                    cutoff: float = 0.85) -> pd.Series:
    """Combine quick-view mode + global search + engine + confidence filters.

    Single source of truth for what the grid shows, so pagination and
    "select all shown" act on the *whole* filtered set, not just one page.
    """
    mask = filter_mask(work_df, mode, cutoff)
    if query:
        mask &= search_mask(work_df, query, search_columns)
    if engines:
        mask &= engine_mask(work_df, engines)
    if conf_range and CONF_COL in work_df.columns:
        lo, hi = conf_range
        conf = work_df[CONF_COL].astype(float)
        mask &= (conf >= lo) & (conf <= hi)
    return mask


def summary_counts(work_df: pd.DataFrame, loaded: LoadedData) -> Dict[str, int]:
    """Live counts derived from the meta columns (kept in sync after edits)."""
    na_col = loaded.new_account_col
    action = work_df[ACTION_COL] if ACTION_COL in work_df.columns else pd.Series([], dtype=str)
    na = work_df[na_col].astype(str).str.strip()
    counts = {
        "total": len(work_df),
        "seeds": int((action == FillAction.KEPT_SEED.value).sum()),
        "auto_filled": int((action == FillAction.AUTO_FILLED.value).sum()),
        "filled_review": int((action == FillAction.FILLED_REVIEW.value).sum()),
        "needs_review": int((action == FillAction.NEEDS_REVIEW.value).sum()),
        "no_match": int((action == FillAction.NO_MATCH.value).sum()),
        "filled": int((na != "").sum()),
        "blank": int((na == "").sum()),
    }
    counts["review_pending"] = counts["filled_review"] + counts["needs_review"]
    counts["distinct_accounts"] = int(na[na != ""].nunique())
    counts["selected"] = len(selected_indices(work_df))
    counts["with_notes"] = int(
        (work_df[RULE_NOTES_COL].astype(str).str.strip() != "").sum()
        if RULE_NOTES_COL in work_df.columns else 0)
    return counts


def engine_breakdown(work_df: pd.DataFrame) -> List[Dict[str, object]]:
    """Per-engine counts + average confidence for the current sheet state.

    Powers the post-run "how were these coded" panel: e.g. how many rows came
    from keyword rules vs learned memory vs similarity vs the ML model.
    """
    if ENGINE_COL not in work_df.columns:
        return []
    eng = work_df[ENGINE_COL].astype(str)
    conf = work_df[CONF_COL].astype(float) if CONF_COL in work_df.columns \
        else pd.Series([0.0] * len(work_df), index=work_df.index)
    rows: List[Dict[str, object]] = []
    for label in eng[eng.str.strip() != ""].unique():
        mask = eng == label
        confs = conf[mask]
        rows.append({
            "engine": friendly_engine(label),
            "rows": int(mask.sum()),
            "avg_confidence": float(confs.mean()) if len(confs) else 0.0,
        })
    rows.sort(key=lambda r: -r["rows"])
    return rows


def confidence_stats(work_df: pd.DataFrame, loaded: LoadedData,
                     high_cutoff: float = 0.85) -> Dict[str, float]:
    """Confidence distribution across the **coded** rows (0.0–1.0 scale)."""
    na_col = loaded.new_account_col
    na = work_df[na_col].astype(str).str.strip()
    filled = na != ""
    conf = (work_df.loc[filled, CONF_COL].astype(float)
            if CONF_COL in work_df.columns else pd.Series([], dtype=float))
    n = int(filled.sum())
    return {
        "coded": n,
        "avg_confidence": float(conf.mean()) if n else 0.0,
        "high_confidence": int((conf >= high_cutoff).sum()) if n else 0,
        "low_confidence": int((conf < 0.55).sum()) if n else 0,
    }


def page_slice(work_df: pd.DataFrame, mask: pd.Series, page: int,
               page_size: int) -> Tuple[pd.DataFrame, int, int]:
    """Return (page_df, n_pages, total_filtered) for the filtered view."""
    filtered = work_df[mask]
    total = len(filtered)
    n_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, n_pages - 1))
    start = page * page_size
    return filtered.iloc[start:start + page_size].copy(), n_pages, total


def editor_columns(loaded: LoadedData, work_df: pd.DataFrame) -> List[str]:
    """Ordered list of columns to show in the editor."""
    na_col = loaded.new_account_col
    text_cols = [c for c in loaded.text_columns if c in work_df.columns]
    extra = [c for c in (loaded.original_columns or [])
             if c in work_df.columns and c not in text_cols
             and c != na_col and not str(c).startswith("_")
             and c != RULE_NOTES_COL]
    return ([SELECT_COL] + text_cols + extra
            + [na_col, RULE_NOTES_COL, CONF_COL, ENGINE_COL])


# --------------------------------------------------------------------------- #
# Undo / redo (snapshots of the mutable columns)
# --------------------------------------------------------------------------- #
_SNAPSHOT_COLS = [NEW_ACCOUNT_COL, RULE_NOTES_COL, CONF_COL, ENGINE_COL,
                  ACTION_COL, WHY_COL, SUGGESTED_COL, SIM_TEXT_COL]


def snapshot(work_df: pd.DataFrame) -> Dict[str, list]:
    """Capture the mutable columns so an action can be undone."""
    return {c: list(work_df[c]) for c in _SNAPSHOT_COLS if c in work_df.columns}


def restore(work_df: pd.DataFrame, snap: Dict[str, list]) -> None:
    """Restore a snapshot taken by :func:`snapshot` (in place)."""
    for col, values in snap.items():
        if col in work_df.columns and len(values) == len(work_df):
            work_df[col] = values


def clone_snapshot(snap: Dict[str, list]) -> Dict[str, list]:
    return copy.deepcopy(snap)
