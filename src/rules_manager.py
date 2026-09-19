"""Keyword-rule management and matching.

Rules give users a fast, deterministic override for the patterns they
already know (e.g. "Cloud Hosting" -> "6100"). They are persisted in SQLite via the
:class:`~utils.storage.Storage` layer and always take precedence over the
fuzzy similarity engine.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from models.schemas import KeywordRule, RuleRunResult
from src.data_loader import NEW_ACCOUNT_COL, RULE_NOTES_COL, _has_value
from utils.account_codes import base_account, normalize_code
from utils.logging_setup import get_logger
from utils.storage import Storage

logger = get_logger(__name__)

# Collapse anything that isn't a letter/digit to spaces, lower-cased. This makes
# matching robust to punctuation/spacing differences ("Cloud-Host", "CLOUD HOST").
_RULE_NONALNUM = re.compile(r"[^a-z0-9]+")
# Default similarity ratio a window must reach for a "fuzzy" rule to match.
_FUZZY_CUTOFF = 0.84

# --------------------------------------------------------------------------- #
# Account-Level Knowledge Base tuning
# --------------------------------------------------------------------------- #
# How much learned knowledge to keep per account (bounded so profiles stay
# lightweight and the UI stays readable).
_MAX_PROFILE_KEYWORDS = 60
_MAX_PROFILE_SAMPLES = 12
# Low-signal words dropped when extracting keywords from a rule pattern.
_KEYWORD_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "of", "to", "a", "an", "in",
    "on", "at", "by", "or", "llc", "inc", "co", "ltd", "corp", "company",
}


def canonicalize_account(code: str) -> str:
    """Canonical account code, e.g. ``"6618 s"`` / ``"6618-S"`` -> ``"6618S"``.

    Thin, well-named wrapper over :func:`utils.account_codes.normalize_code` so
    the knowledge base always keys accounts consistently (including the optional
    single-letter sub-account suffix like the NARPM-style ``S``).
    """
    return normalize_code(code)


def extract_keywords(pattern: str) -> List[str]:
    """Turn a rule pattern into a normalised list of keywords/phrases.

    * Splits on commas (each rule can hold several phrases, OR logic).
    * Normalises each phrase (lower-case, punctuation -> space, collapsed).
    * Also emits the individual significant tokens of multi-word phrases so a
      profile learns both ``"cred hub"`` and ``cred`` / ``hub``.
    * Drops empties, pure numbers and common stop-words; de-duplicates while
      preserving first-seen order.
    """
    out: List[str] = []
    seen: set = set()

    def _add(term: str) -> None:
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            out.append(term)

    for raw_phrase in str(pattern or "").split(","):
        phrase = _norm_space(raw_phrase)
        if not phrase:
            continue
        _add(phrase)
        tokens = phrase.split()
        if len(tokens) > 1:
            for tok in tokens:
                if len(tok) >= 3 and not tok.isdigit() \
                        and tok not in _KEYWORD_STOPWORDS:
                    _add(tok)
    return out


def _is_blank(value) -> bool:
    """True for None / NaN / pandas NA / empty-or-whitespace strings.

    Robust across object, string and StringDtype cells so the "never overwrite a
    non-blank New Account" guarantee holds regardless of dtype.
    """
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _norm_space(text: str) -> str:
    """Lower-case and collapse runs of non-alphanumerics to single spaces."""
    return _RULE_NONALNUM.sub(" ", str(text).lower()).strip()


def _norm_tight(text: str) -> str:
    """Lower-case and strip every non-alphanumeric (so 'Cloud Host' -> 'cloudhost')."""
    return _RULE_NONALNUM.sub("", str(text).lower())


def _norm_field(name: str) -> str:
    """Normalise a column name for tolerant matching.

    Lower-cases and removes *all* whitespace so a rule scoped to
    ``"Memo/Description"`` still resolves to a column typed as
    ``"Memo / Description"`` or ``"memo/description"`` in a different export.
    """
    return re.sub(r"\s+", "", str(name)).lower()


# --------------------------------------------------------------------------- #
# Name / Memo roles, phrase keys and collisions
# --------------------------------------------------------------------------- #
# Headers that carry the vendor name (the most distinctive signal on a ledger)
# and the ones that carry the memo / description (secondary — often noise).
NAME_LIKE_FIELDS: Tuple[str, ...] = (
    "Name", "Payee", "Vendor", "Payee Name", "Supplier", "Payee/Payer")
MEMO_LIKE_FIELDS: Tuple[str, ...] = (
    "Memo", "Memo/Description", "Memo / Description", "Description")
_NAME_LIKE_NORM = {_norm_field(f) for f in NAME_LIKE_FIELDS}
_MEMO_LIKE_NORM = {_norm_field(f) for f in MEMO_LIKE_FIELDS}
# Tokens that are pure numbers / ids / amounts ("#20", "12", "$1,234.00",
# "01/05/2026"). They are dropped from ingested phrases and collision keys so
# "weekly lawn service #20" and "weekly lawn service #21" are one term.
_NUMERIC_ONLY_TOKEN = re.compile(r"^[#$€£\-\d.,/:%()]+$")


def is_name_like(header: object) -> bool:
    """True for a vendor-name column (Name / Payee / Vendor ...)."""
    return _norm_field(str(header)) in _NAME_LIKE_NORM


def is_memo_like(header: object) -> bool:
    """True for a memo / description column."""
    return _norm_field(str(header)) in _MEMO_LIKE_NORM


def _is_note_header(header: object) -> bool:
    """Any header containing "note" (Notes / Rule Notes / Internal Notes)."""
    return "note" in str(header).strip().lower()


def phrase_display(text: object) -> str:
    """A reusable phrase from a cell: numeric-only tokens dropped, commas
    replaced (commas separate phrases inside a rule), whitespace collapsed.

    Case is preserved so the rule reads like the ledger. Falls back to the
    raw (comma-free) text when stripping numbers would leave nothing.
    """
    raw = "" if text is None else str(text).replace(",", " ").strip()
    if not raw:
        return ""
    toks = [t for t in raw.split() if not _NUMERIC_ONLY_TOKEN.match(t)]
    cleaned = " ".join(toks).strip()
    if not cleaned or not re.search(r"[A-Za-z]", cleaned):
        return " ".join(raw.split())
    return cleaned


def phrase_key(text: object) -> str:
    """Canonical comparison key for a phrase / collision (see phrase_display)."""
    return _norm_space(phrase_display(text))


def collision_key_for_row(row: pd.Series, name_col: Optional[str],
                          memo_col: Optional[str]) -> str:
    """The collision key of a row: normalised Name if present, else Memo."""
    if name_col is not None and name_col in row.index:
        key = phrase_key(row[name_col]) if _has_value(row[name_col]) else ""
        if key:
            return key
    if memo_col is not None and memo_col in row.index and _has_value(row[memo_col]):
        return phrase_key(row[memo_col])
    return ""


# The spreadsheet's per-row provenance column (see spreadsheet_helpers
# ENGINE_COL). When present, only rows the accountant *owns* — upload seeds and
# manual codes — count as coded votes; an engine guess is never a vote.
_ENGINE_META_COL = "_engine"
_OWNED_ENGINES = {"seed", "manual", ""}


def owned_code_rows(df: pd.DataFrame) -> List[object]:
    """Index labels of rows whose New Account the accountant owns."""
    if _ENGINE_META_COL not in df.columns:
        return list(df.index)
    eng = df[_ENGINE_META_COL].astype(str).str.strip()
    return list(df.index[eng.isin(_OWNED_ENGINES)])


def detect_collisions(
    df: pd.DataFrame,
    name_col: Optional[str],
    memo_col: Optional[str],
    na_col: str = NEW_ACCOUNT_COL,
    seed: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Set[str]]:
    """Keys (Name, else Memo) that code to **two or more** accounts.

    Looks at the coded rows of ``df`` that the accountant owns (seeds and
    manual codes — never an engine's own guess) and, optionally, a ``seed``
    mapping of key -> codes already known (e.g. the phrases of the client's
    existing rules). Returns only the colliding keys with their full code set.
    """
    key_codes: Dict[str, Set[str]] = {k: set(v) for k, v in (seed or {}).items()}
    if na_col in df.columns and (name_col is not None or memo_col is not None):
        for idx in owned_code_rows(df):
            code_raw = df.at[idx, na_col]
            if _is_blank(code_raw):
                continue
            code = normalize_code(str(code_raw).strip())
            if not code:
                continue
            key = collision_key_for_row(df.loc[idx], name_col, memo_col)
            if key:
                key_codes.setdefault(key, set()).add(code)
    return {k: v for k, v in key_codes.items() if len(v) > 1}


def format_codes(codes: Iterable[str]) -> str:
    """``"6326 and 6760.01"`` / ``"6326, 6760.01 and 7000"``."""
    items = sorted({str(c).strip() for c in codes if str(c).strip()})
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def collision_rationale(codes: Iterable[str]) -> str:
    """Plain-English why-text for a collision row (never auto-filled)."""
    return (f"Collision: this vendor/memo also codes to {format_codes(codes)} "
            "— your coded rows disagree, so it was not auto-filled. "
            "Choose the right account in Review.")


COLLISION_ENGINE = "collision"


@dataclass
class IngestReport:
    """What one ingest-per-account pass did."""

    rules_created: int = 0
    rules_updated: int = 0
    phrases_added: int = 0
    accounts: List[str] = field(default_factory=list)
    # key -> sorted colliding codes (persisted; excluded from rule phrases).
    collisions: Dict[str, List[str]] = field(default_factory=dict)
    skipped_collision_phrases: int = 0
    name_col: Optional[str] = None
    memo_col: Optional[str] = None

    @property
    def rules_touched(self) -> int:
        return self.rules_created + self.rules_updated


@dataclass
class RuleMatch:
    account_code: str
    rule: KeywordRule


def record_human_approval(
    storage: Storage,
    signature: str,
    account_code: str,
    *,
    confidence: float = 1.0,
    engine_used: str = "manual",
    approved_by: str = "user",
    client_id: Optional[str] = None,
) -> bool:
    """The single learning path for every human decision.

    One approval writes all three kinds of memory:
      1. a **learned mapping** (exact signature -> code; last human write wins),
      2. a **training example** for the ML models,
      3. a reinforcement of the account's knowledge profile.

    Returns False (and writes nothing) for blank signatures/codes. Never
    raises for the profile reinforcement — learning must not break approvals.
    """
    signature = (signature or "").strip()
    code = normalize_code(account_code)
    if not signature or not code:
        return False
    storage.upsert_learned_mapping(signature, code, client_id=client_id)
    storage.add_training_example(
        text=signature, label=code, confidence=float(confidence),
        engine_used=engine_used, approved_by=approved_by, client_id=client_id)
    try:
        RulesManager(storage).record_account_usage(
            code, sample_text=signature, client_id=client_id)
    except Exception as exc:  # noqa: BLE001 - best-effort reinforcement
        logger.warning("approval_profile_reinforce_failed", error=str(exc))
    return True


def near_miss_suggestions(
    results: List[RuleRunResult],
    df: pd.DataFrame,
    rules: List[KeywordRule],
    text_columns: Optional[List[str]] = None,
    min_overlap: float = 0.5,
    max_suggestions: int = 10,
) -> List[dict]:
    """Surface "near misses" after a strict rule run — never auto-fills.

    For every blank row where no rule fired, find the enabled rule whose
    phrases have the highest token overlap with the row's text. When at least
    ``min_overlap`` of a phrase's tokens are present (but the rule did not
    match), the row is evidence the rule is *almost* right — e.g. a spelling
    variant or a missing phrase. Returns one suggestion per rule, with the
    affected row indices and the most common row tokens the rule is missing,
    so the user can decide to widen the rule. Pure and deterministic.
    """
    active = [r for r in rules if r.enabled]
    if not active:
        return []

    # Pre-compute each rule's phrase token sets.
    rule_phrases: List[tuple] = []
    for rule in active:
        for phrase in rule.match_phrases():
            toks = set(_norm_space(phrase).split())
            toks = {t for t in toks if t and not t.isdigit()}
            if toks:
                rule_phrases.append((rule, phrase, toks))
    if not rule_phrases:
        return []

    best_per_row: dict = {}  # row_index -> (overlap, rule, missing_tokens)
    for r in results:
        if r.status != "no_rule_match":
            continue
        try:
            row = df.iloc[r.row_index]
        except (IndexError, KeyError):
            continue
        text = RulesManager.build_match_text(row, text_columns)
        row_toks = [t for t in _norm_space(text).split()
                    if t and not t.isdigit() and t not in _KEYWORD_STOPWORDS]
        if not row_toks:
            continue
        row_tok_set = set(row_toks)
        for rule, _phrase, ptoks in rule_phrases:
            overlap = len(ptoks & row_tok_set) / len(ptoks)
            if overlap < min_overlap:
                continue
            missing = [t for t in row_toks if t not in ptoks]
            cur = best_per_row.get(r.row_index)
            if cur is None or overlap > cur[0]:
                best_per_row[r.row_index] = (overlap, rule, missing)

    # Aggregate by rule: which rows nearly matched it, and which tokens recur.
    by_rule: dict = {}
    for row_idx, (overlap, rule, missing) in best_per_row.items():
        entry = by_rule.setdefault(
            rule.id,
            {"rule_id": rule.id, "rule_keyword": rule.keyword,
             "account_code": rule.account_code, "rows": [],
             "overlap": 0.0, "token_counts": {}})
        entry["rows"].append(row_idx)
        entry["overlap"] = max(entry["overlap"], overlap)
        for tok in missing[:6]:
            entry["token_counts"][tok] = entry["token_counts"].get(tok, 0) + 1

    out: List[dict] = []
    for entry in by_rule.values():
        common = sorted(entry.pop("token_counts").items(),
                        key=lambda kv: (-kv[1], kv[0]))
        entry["suggested_tokens"] = [t for t, _ in common[:4]]
        entry["rows"] = sorted(entry["rows"])
        entry["overlap"] = round(float(entry["overlap"]), 3)
        out.append(entry)
    out.sort(key=lambda e: (-len(e["rows"]), -e["overlap"], e["rule_keyword"]))
    return out[:max_suggestions]


def get_rule_application_summary(results: List[RuleRunResult]) -> dict:
    """Summarise a rule run: protected vs filled vs left-blank + success rate.

    ``success_rate`` is measured over **blank rows only** (filled / blanks),
    because rules never touch already-coded rows — that is the metric the client
    cares about for real-time coding.
    """
    protected = sum(1 for r in results if r.status == "protected_existing")
    filled = sum(1 for r in results if r.status == "keyword_rule")
    collisions = sum(1 for r in results if r.status == "collision")
    # Collision rows are blank rows a rule was *not allowed* to fill.
    left_blank = sum(1 for r in results if r.status == "no_rule_match") \
        + collisions
    # Protected rows that a rule *also* matched — proof the rules fire even when
    # nothing is filled because the cell was already coded.
    protected_but_matched = sum(
        1 for r in results
        if r.status == "protected_existing" and r.matched_rule_name)
    blanks = filled + left_blank
    return {
        "total": len(results),
        "protected_existing": protected,
        "protected_but_matched": protected_but_matched,
        "filled_by_rules": filled,
        "left_blank": left_blank,
        "collisions": collisions,
        "blanks_total": blanks,
        "success_rate": (filled / blanks) if blanks else 0.0,
    }


class RulesManager:
    """CRUD + matching for keyword rules, backed by persistent storage."""

    def __init__(self, storage: Storage):
        self.storage = storage

    # ----------------------------------------------------------------- CRUD
    def add_rule(self, keyword: str, account_code: str, **kwargs) -> KeywordRule:
        rule = KeywordRule(
            keyword=keyword,
            account_code=normalize_code(account_code),
            **kwargs,
        )
        saved = self.storage.add_rule(rule)
        logger.info("rule_added", keyword=saved.keyword, code=saved.account_code)
        # Reinforce the account knowledge base with this rule's keywords.
        self._learn_from_rule(saved)
        return saved

    def create_rule(self, keyword: str, account_code: str, **kwargs) -> KeywordRule:
        """Explicit, client-facing alias for :meth:`add_rule`.

        Accepts the full rule spec: ``match_type`` (contains | exact |
        starts_with | ends_with | fuzzy | regex), ``fields`` (columns to search),
        ``case_sensitive``, ``priority``, ``client_id`` and ``notes``. The
        ``keyword`` may contain several comma-separated phrases (OR logic).
        """
        return self.add_rule(keyword, account_code, **kwargs)

    def get_rule(self, rule_id: int) -> Optional[KeywordRule]:
        """Return a single rule by id, or ``None`` if it doesn't exist."""
        for r in self.list_rules():
            if r.id == rule_id:
                return r
        return None

    def update_rule(self, rule: KeywordRule) -> None:
        rule.account_code = normalize_code(rule.account_code)
        self.storage.update_rule(rule)
        logger.info("rule_updated", id=rule.id)
        # Keep the knowledge base in sync with edited keywords.
        self._learn_from_rule(rule)

    def delete_rule(self, rule_id: int) -> None:
        self.storage.delete_rule(rule_id)
        logger.info("rule_deleted", id=rule_id)

    def delete_rules(self, rule_ids: List[int]) -> int:
        """Bulk-delete rules by id. Returns how many were removed."""
        removed = self.storage.delete_rules(rule_ids)
        logger.info("rules_deleted", count=removed)
        return removed

    def clear_rules(self) -> int:
        """Delete every keyword rule. Returns how many were removed."""
        removed = self.storage.clear_rules()
        logger.info("rules_cleared", count=removed)
        return removed

    def list_rules(self, enabled_only: bool = False,
                   client_id: Optional[str] = None) -> List[KeywordRule]:
        """List rules (ordered by priority then id).

        ``client_id=None`` returns every rule; passing a client id narrows to
        that client's rules plus global ones.
        """
        return self.storage.list_rules(enabled_only=enabled_only,
                                       client_id=client_id)

    # ------------------------------------------------------------- matching
    def match_row(
        self,
        combined_text: str,
        row: Optional[pd.Series] = None,
        rules: Optional[List[KeywordRule]] = None,
    ) -> Optional[RuleMatch]:
        """Return the first enabled rule that matches this row, else ``None``.

        ``combined_text`` is the pre-built lowercased similarity text. ``row``
        (optional) lets field-scoped rules inspect individual columns, and
        enables Name-first matching (see :meth:`_first_matching_rule`).
        """
        rules = rules if rules is not None else self.list_rules(enabled_only=True)
        active = [r for r in rules if r.enabled]
        rule = self._first_matching_rule(active, combined_text, row)
        if rule is None:
            return None
        return RuleMatch(account_code=rule.account_code, rule=rule)

    @staticmethod
    def resolve_name_memo_columns(
        columns: Iterable[object],
        preferred: Optional[Iterable[object]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """(name column, memo column) among ``columns``.

        ``preferred`` (e.g. the configured keyword-source order) decides which
        header wins when several name-like or memo-like columns exist. Note-like
        headers, internal ``_`` columns and the answer column never qualify.
        """
        cols = [str(c) for c in columns
                if not str(c).startswith("_")
                and str(c) not in (NEW_ACCOUNT_COL, RULE_NOTES_COL)
                and not _is_note_header(c)]
        order: List[str] = []
        for c in list(preferred or []) + cols:
            c = str(c)
            if c in cols and c not in order:
                order.append(c)
        name_col = next((c for c in order if is_name_like(c)), None)
        memo_col = next((c for c in order if is_memo_like(c)), None)
        return name_col, memo_col

    def collision_lookup(
        self,
        df: Optional[pd.DataFrame] = None,
        client_id: Optional[str] = None,
        name_col: Optional[str] = None,
        memo_col: Optional[str] = None,
        na_col: str = NEW_ACCOUNT_COL,
    ) -> Dict[str, Set[str]]:
        """Stored collisions for the client, unioned with the book's own.

        A vendor coded two ways *inside the current file* collides just as
        much as one remembered from an earlier ingest.
        """
        stored = self.storage.get_collision_lookup(client_id=client_id)
        if df is None:
            return {k: set(v) for k, v in stored.items()}
        if name_col is None and memo_col is None:
            name_col, memo_col = self.resolve_name_memo_columns(df.columns)
        return detect_collisions(df, name_col, memo_col, na_col=na_col,
                                 seed=stored)

    @staticmethod
    def _split_name_fields(rule: KeywordRule,
                           row: pd.Series) -> Tuple[List[str], List[str]]:
        """A field-scoped rule's resolved columns split into (name, other)."""
        resolved = RulesManager._resolve_fields(rule.fields, row)
        names = [c for c in resolved if is_name_like(c)]
        others = [c for c in resolved if c not in names]
        return names, others

    @staticmethod
    def _matches_in_columns(rule: KeywordRule, row: pd.Series,
                            columns: List[str]) -> bool:
        """Does any phrase of ``rule`` hit any of these columns on ``row``?"""
        phrases = rule.match_phrases()
        for col in columns:
            val = row[col]
            if not _has_value(val):
                continue
            hay = str(val)
            for phrase in phrases:
                if RulesManager._phrase_matches(rule, phrase, hay):
                    return True
        return False

    @staticmethod
    def _haystacks(
        rule: KeywordRule,
        combined_text: str,
        row: Optional[pd.Series],
    ) -> List[str]:
        """The text fragments a rule should be tested against.

        **Default (row-wide):** the whole transaction record is concatenated into
        one string (all columns) and searched as a single haystack — so users
        never have to pick a column for a keyword to work. Individual columns are
        also exposed so anchored match types (``exact`` / ``starts_with`` /
        ``ends_with``) still behave sensibly per column.

        **Advanced (field-scoped):** when ``rule.fields`` is set, only those
        named columns are searched (tolerant to header spelling differences).
        """
        haystacks: List[str] = []
        if rule.fields:
            if row is None:
                return []
            for fname in RulesManager._resolve_fields(rule.fields, row):
                val = row[fname]
                if pd.notna(val):
                    haystacks.append(str(val))
            return haystacks

        # Row-wide default: the full concatenated record is the primary haystack.
        if row is not None:
            full = RulesManager.build_match_text(row)
            if full:
                haystacks.append(full)
            for col, val in row.items():
                name = str(col)
                # Skip internal/meta columns, the answer column, and Rule Notes
                # (its text is already folded into combined_text).
                if name.startswith("_") or name == NEW_ACCOUNT_COL \
                        or name == RULE_NOTES_COL:
                    continue
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                sval = str(val).strip()
                if sval:
                    haystacks.append(sval)
        if combined_text:
            haystacks.append(str(combined_text))
        if not haystacks:
            haystacks.append(str(combined_text or ""))
        return haystacks

    @staticmethod
    def _resolve_fields(fields: List[str], row: pd.Series) -> List[str]:
        """Resolve a rule's field names to the row's *actual* columns.

        Matching is case- and whitespace-insensitive so rules keep working when a
        later export headers a column slightly differently (e.g. ``"Memo /
        Description"`` vs ``"Memo/Description"``). Unknown fields are skipped.
        """
        norm_to_actual: dict = {}
        for col in row.index:
            norm_to_actual.setdefault(_norm_field(str(col)), col)
        out: List[str] = []
        for f in fields:
            actual = norm_to_actual.get(_norm_field(str(f)))
            if actual is not None and actual not in out:
                out.append(actual)
        return out

    @staticmethod
    def _fuzzy_contains(cand_space: str, key_space: str,
                        cutoff: float = _FUZZY_CUTOFF) -> bool:
        """Token-aware, typo-tolerant match of the keyword against a fragment.

        Compares the keyword against **whole tokens** (or same-width token
        windows) of the candidate — never substrings — so a vendor rule like
        "Shaw Media" fires on the misspelling "Shaw Medai" but does *not* fire
        on an address token that merely contains the same letters ("shaw" in
        "123 Shawnee Dr"). Multi-word phrases slide a window of the same token
        width across the candidate.
        """
        if not key_space or not cand_space:
            return False
        ktokens = key_space.split()
        ctokens = cand_space.split()
        if not ktokens or not ctokens:
            return False

        if len(ktokens) == 1:
            key = ktokens[0]
            return any(
                difflib.SequenceMatcher(None, key, tok).ratio() >= cutoff
                for tok in ctokens
            )

        width = len(ktokens)
        if len(ctokens) < width:
            # Candidate shorter than the phrase: compare the whole fragment.
            return difflib.SequenceMatcher(None, key_space, cand_space).ratio() \
                >= cutoff
        for start in range(len(ctokens) - width + 1):
            window = " ".join(ctokens[start:start + width])
            if difflib.SequenceMatcher(None, key_space, window).ratio() >= cutoff:
                return True
        return False

    def rule_matches(self, rule: KeywordRule, row_text: str) -> bool:
        """Public: does ``rule`` match the given text? (No per-column scoping.)

        Robust to punctuation/spacing/case and honours comma-separated phrases.
        Never raises — a malformed regex simply doesn't match.
        """
        return self._rule_matches(rule, row_text or "", None)

    @staticmethod
    def _phrase_matches(
        rule: KeywordRule,
        phrase: str,
        hay: str,
    ) -> bool:
        """Does a single phrase match one text fragment under the rule's mode?"""
        phrase = (phrase or "").strip()
        if not phrase or hay is None:
            return False
        hay = str(hay)

        # ---- regex: applied to the raw fragment ----------------------------
        if rule.match_type == "regex":
            flags = 0 if rule.case_sensitive else re.IGNORECASE
            try:
                return bool(re.search(phrase, hay, flags))
            except re.error:
                logger.warning("invalid_regex_rule", keyword=phrase)
                return False

        # ---- case-sensitive: literal comparison (no normalisation) ---------
        if rule.case_sensitive:
            if rule.match_type == "exact":
                return hay.strip() == phrase.strip()
            if rule.match_type == "starts_with":
                return hay.strip().startswith(phrase)
            if rule.match_type == "ends_with":
                return hay.strip().endswith(phrase)
            return phrase in hay  # contains / fuzzy fall back to literal here

        # ---- default case-insensitive, punctuation/space tolerant ----------
        cand_space = _norm_space(hay)
        key_space = _norm_space(phrase)
        if not key_space:
            return False

        if rule.match_type == "exact":
            return cand_space == key_space
        if rule.match_type == "starts_with":
            return (cand_space.startswith(key_space)
                    or _norm_tight(hay).startswith(_norm_tight(phrase)))
        if rule.match_type == "ends_with":
            return (cand_space.endswith(key_space)
                    or _norm_tight(hay).endswith(_norm_tight(phrase)))
        if rule.match_type == "fuzzy":
            return RulesManager._fuzzy_contains(cand_space, key_space)

        # "contains" (default): spaced substring, plus a tight (de-spaced) pass
        # so multi-word phrases match concatenated source ("Cloud Hosting" ->
        # "CloudHosting").
        if key_space in cand_space:
            return True
        if " " in key_space and _norm_tight(phrase) in _norm_tight(hay):
            return True
        return False

    @staticmethod
    def _rule_matches(
        rule: KeywordRule,
        combined_text: str,
        row: Optional[pd.Series],
    ) -> bool:
        phrases = rule.match_phrases()
        if not phrases:
            return False
        haystacks = RulesManager._haystacks(rule, combined_text, row)
        if rule.fields and not haystacks:
            return False
        # A rule matches if ANY of its phrases matches ANY haystack fragment.
        for phrase in phrases:
            for hay in haystacks:
                if RulesManager._phrase_matches(rule, phrase, hay):
                    return True
        return False

    # ---------------------------------------------------- dataframe application
    @staticmethod
    def _sorted_for_apply(rules: List[KeywordRule]) -> List[KeywordRule]:
        """Enabled rules in application order: priority asc, then id asc."""
        active = [r for r in rules if r.enabled]
        return sorted(active, key=lambda r: (getattr(r, "priority", 100),
                                             r.id if r.id is not None else 1 << 30))

    @staticmethod
    def build_match_text(row: pd.Series,
                         text_columns: Optional[List[str]] = None) -> str:
        """Concatenate a row's text into one lower-cased haystack.

        Uses ``text_columns`` when given, otherwise every string-like column
        except internal (``_`` prefixed), the answer column and Rule Notes.
        Handles object / string / StringDtype and NaN safely.
        """
        parts: List[str] = []
        if text_columns:
            cols = [c for c in text_columns if c in row.index]
        else:
            cols = [c for c in row.index
                    if not str(c).startswith("_")
                    and c != NEW_ACCOUNT_COL and c != RULE_NOTES_COL]
        for c in cols:
            val = row.get(c)
            if not _has_value(val):
                continue
            parts.append(str(val).strip())
        return " | ".join(parts).lower()

    def apply_rules_to_dataframe(
        self,
        df: pd.DataFrame,
        rules: Optional[List[KeywordRule]] = None,
        text_columns: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        collisions: Optional[Dict[str, Set[str]]] = None,
    ) -> tuple[pd.DataFrame, List[RuleRunResult]]:
        """Fill blank ``New Account`` cells from keyword rules — deterministically.

        Guarantees, in priority order:
          * A row whose ``New Account`` is already non-blank is **never** touched
            (status ``protected_existing``).
          * A blank row whose vendor (Name, else Memo) codes to more than one
            account — in this book or in the client's stored collisions — is
            **never** filled, whatever the rules say (status ``collision``).
          * Blank rows are matched against rules ordered by priority; the first
            matching rule wins and its normalised code is written
            (status ``keyword_rule``).
          * Blank rows with no match stay blank (status ``no_rule_match``).

        Mutates and returns ``df`` plus a complete, auditable per-row result list.
        """
        if rules is None:
            rules = self.list_rules(enabled_only=True)
        ordered = self._sorted_for_apply(rules)

        if NEW_ACCOUNT_COL not in df.columns:
            df[NEW_ACCOUNT_COL] = ""

        name_col, memo_col = self.resolve_name_memo_columns(df.columns)
        if collisions is None:
            collisions = self.collision_lookup(df, client_id=client_id,
                                               name_col=name_col,
                                               memo_col=memo_col)

        results: List[RuleRunResult] = []
        for pos, idx in enumerate(df.index):
            try:
                ri = int(idx)
            except (TypeError, ValueError):
                ri = pos

            row = df.loc[idx]
            match_text = self.build_match_text(row, text_columns)
            matched = self._first_matching_rule(ordered, match_text, row)

            existing = df.at[idx, NEW_ACCOUNT_COL]
            existing_str = "" if _is_blank(existing) else str(existing).strip()
            if not existing_str and collisions:
                key = collision_key_for_row(row, name_col, memo_col)
                if key and key in collisions:
                    results.append(RuleRunResult(
                        row_index=ri, original_value="", proposed_value="",
                        engine_used=COLLISION_ENGINE, confidence=0.0,
                        status="collision",
                        matched_rule_name=(
                            (matched.notes.strip() or matched.keyword)
                            if matched is not None else ""),
                        rationale=collision_rationale(collisions[key])))
                    continue
            if existing_str:
                # Never overwrite. If a rule *would* have matched, record it so
                # the audit can prove the rule fires (it's just already coded).
                rr = RuleRunResult(
                    row_index=ri, original_value=existing_str,
                    proposed_value=existing_str, engine_used="protected",
                    confidence=1.0, status="protected_existing",
                    rationale="New Account already set — preserved.")
                if matched is not None:
                    rr.matched_rule_name = matched.notes.strip() or matched.keyword
                    rr.matched_pattern = f"{matched.match_type}: {matched.keyword}"
                    rr.rationale = (
                        f"Rule '{matched.keyword}' matches this row, but its "
                        f"New Account ({existing_str}) is preserved.")
                results.append(rr)
                continue

            if matched is not None:
                code = normalize_code(matched.account_code)
                df.at[idx, NEW_ACCOUNT_COL] = code
                results.append(RuleRunResult(
                    row_index=ri, original_value="", proposed_value=code,
                    engine_used="rules", confidence=0.99, status="keyword_rule",
                    matched_rule_name=(matched.notes.strip() or matched.keyword),
                    matched_pattern=f"{matched.match_type}: {matched.keyword}",
                    rationale=f"Matched rule '{matched.keyword}' "
                              f"({matched.match_type}) -> {code}."))
            else:
                results.append(RuleRunResult(
                    row_index=ri, original_value="", proposed_value="",
                    engine_used="none", confidence=0.0, status="no_rule_match",
                    rationale="No keyword rule matched this row."))
        return df, results

    def _first_matching_rule(
        self,
        ordered_rules: List[KeywordRule],
        match_text: str,
        row: Optional[pd.Series],
    ) -> Optional[KeywordRule]:
        """First rule (in the given order) that matches — never raises.

        **Name-first.** For rules scoped to a vendor-name column (Name / Payee
        / Vendor …) plus other columns — the shape ingest produces — the Name
        column is searched first and a Name hit wins outright. A hit on the
        rule's other columns (Memo) is only *provisional*: it is kept as the
        answer unless a later rule hits the row's Name. So a noisy Memo can
        never steal a row whose vendor Name belongs to a different account,
        and Memo still codes rows whose Name is empty or unknown.

        Row-wide rules (no ``fields``) and rules scoped to non-name columns
        keep their plain priority order; a provisional Memo hit that came
        earlier in priority still beats them.
        """
        pending: Optional[KeywordRule] = None
        for rule in ordered_rules:
            try:
                if row is not None and rule.fields:
                    name_cols, other_cols = self._split_name_fields(rule, row)
                    if name_cols:
                        if self._matches_in_columns(rule, row, name_cols):
                            return rule
                        if pending is None and other_cols and \
                                self._matches_in_columns(rule, row, other_cols):
                            pending = rule
                        continue
                if self._rule_matches(rule, match_text, row):
                    return pending or rule
            except Exception as exc:  # noqa: BLE001 - never fail a whole run
                logger.warning("rule_match_error", rule=rule.keyword,
                               error=str(exc))
                continue
        return pending

    def prefilled_ingest_candidates(
        self,
        df: pd.DataFrame,
        text_cols: Optional[List[str]] = None,
        max_codes: int = 60,
        max_samples: int = 4,
    ) -> List[dict]:
        """Summarise already-coded rows for **human-reviewed** rule creation.

        Rather than silently inventing keywords, this returns, per distinct
        ``New Account`` code, the row count and a few sample transaction texts so
        the UI can show the accountant the actual data and let them type the
        keywords/phrases they want. Nothing is created here.

        Each item: ``{"code", "count", "samples": [str, ...]}`` sorted by count
        (most common codes first). Codes that already have a rule are still shown
        (the user may want additional phrases), but flagged via ``has_rule``.
        """
        if NEW_ACCOUNT_COL not in df.columns:
            return []

        if not text_cols:
            text_cols = [c for c in df.columns
                         if not str(c).startswith("_")
                         and c not in (NEW_ACCOUNT_COL, RULE_NOTES_COL)]
        else:
            text_cols = [c for c in text_cols if c in df.columns]

        existing_codes = {normalize_code(r.account_code)
                          for r in self.list_rules()}
        groups: dict = {}
        for idx in df.index:
            code_raw = df.at[idx, NEW_ACCOUNT_COL]
            if _is_blank(code_raw):
                continue
            code = normalize_code(str(code_raw).strip())
            row = df.loc[idx]
            sample = " | ".join(
                str(row.get(c)).strip() for c in text_cols[:3]
                if _has_value(row.get(c)))
            g = groups.setdefault(
                code, {"code": code, "count": 0, "samples": [],
                       "has_rule": code in existing_codes})
            g["count"] += 1
            if sample and sample not in g["samples"] \
                    and len(g["samples"]) < max_samples:
                g["samples"].append(sample)

        out = sorted(groups.values(), key=lambda g: -g["count"])
        return out[:max_codes]

    # Priority for ingested rules: low number so they run before broad manual
    # ``contains`` rules (default priority 100).
    INGEST_PRIORITY = 10

    def _ingest_columns(
        self,
        df: pd.DataFrame,
        source_text_cols: Optional[List[str]],
        source_text_col: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """(name column, memo column) ingest should read from.

        ``source_text_cols`` (the caller's mining columns, Name -> Memo ->
        Description …, note-like headers already excluded unless the user
        opted them in) decides preference. Without it, the legacy
        ``source_text_col`` leads, then the file's own headers. When the file
        has neither a name-like nor a memo-like header, the first text column
        with values stands in as the memo column so ingest still works.
        """
        if source_text_cols:
            pool = [c for c in source_text_cols if c in df.columns]
        else:
            pool = [c for c in [source_text_col] + list(df.columns)
                    if c in df.columns and not str(c).startswith("_")
                    and c not in (NEW_ACCOUNT_COL, RULE_NOTES_COL)
                    and not _is_note_header(c)]
        # Stable de-dupe.
        ordered: List[str] = []
        for c in pool:
            if c not in ordered:
                ordered.append(c)
        name_col, memo_col = self.resolve_name_memo_columns(ordered, ordered)
        if name_col is None and memo_col is None:
            for c in ordered:
                if df[c].apply(_has_value).any():
                    memo_col = c
                    break
        return name_col, memo_col

    def ingest_rules_per_account(
        self,
        df: pd.DataFrame,
        client_id: Optional[str] = None,
        source_text_col: str = "Description",
        source_text_cols: Optional[List[str]] = None,
        match_type: str = "contains",
    ) -> IngestReport:
        """Turn coded rows into **one rule per GL account** (merge on re-ingest).

        For each canonical ``New Account`` code on the coded rows, one rule is
        built whose phrase list is every distinct vendor **Name** on that
        account, then every distinct **Memo** term. A row with no Name
        contributes its Memo only. Phrases are persisted in full (the UI may
        abbreviate the display). When a rule for that account + client already
        exists it is **merged** (phrase union, same rule id) — never a second
        rule for the same destination code.

        Rules are ``contains`` (so "lease fee" hits "August lease fee — unit
        12"), high priority, and field-scoped to the Name + Memo columns so
        Notes can never participate. Matching is Name-first (see
        :meth:`_first_matching_rule`).

        **Collisions** — a Name (or, when Name is blank, a Memo) coded to two or
        more accounts in this book or in the client's existing rules — are
        persisted to storage and left *out* of every rule: rows carrying that
        vendor escalate to Review until the accountant decides.
        """
        report = IngestReport()
        if NEW_ACCOUNT_COL not in df.columns:
            return report
        name_col, memo_col = self._ingest_columns(df, source_text_cols,
                                                  source_text_col)
        report.name_col, report.memo_col = name_col, memo_col
        if name_col is None and memo_col is None:
            return report

        existing_rules = self.list_rules(client_id=client_id)
        # Only rules for THIS client (or global when no client) may be merged
        # into; a global rule is never rewritten from a client's book.
        own_rules = [r for r in existing_rules
                     if r.client_id == client_id and r.match_type != "regex"]
        # Merge target per account: an earlier ingested rule first, else the
        # oldest rule for that code (a regex rule is a single pattern and is
        # never spliced into).
        by_code: Dict[str, KeywordRule] = {}
        for r in sorted(own_rules,
                        key=lambda r: (not r.notes.startswith("Ingested"),
                                       r.id or 0)):
            by_code.setdefault(normalize_code(r.account_code), r)

        # ---- gather phrases per account (Names first, then Memos) --------
        names_by_code: Dict[str, List[str]] = {}
        memos_by_code: Dict[str, List[str]] = {}
        seen_by_code: Dict[str, Set[str]] = {}
        # Only codes the accountant owns (seeds / manual) become rules — an
        # engine's own guess must never be promoted into a rule.
        for idx in owned_code_rows(df):
            code_raw = df.at[idx, NEW_ACCOUNT_COL]
            if _is_blank(code_raw):
                continue
            code = normalize_code(str(code_raw).strip())
            if not code:
                continue
            row = df.loc[idx]
            seen = seen_by_code.setdefault(code, set())
            names_by_code.setdefault(code, [])
            memos_by_code.setdefault(code, [])
            if name_col is not None and _has_value(row.get(name_col)):
                disp = phrase_display(row.get(name_col))
                key = _norm_space(disp)
                if key and key not in seen:
                    seen.add(key)
                    names_by_code[code].append(disp)
            if memo_col is not None and _has_value(row.get(memo_col)):
                disp = phrase_display(row.get(memo_col))
                key = _norm_space(disp)
                if key and len(re.sub(r"[^a-z]", "", key)) >= 3 \
                        and key not in seen:
                    seen.add(key)
                    memos_by_code[code].append(disp)

        # ---- collisions: book rows + phrases of the client's existing rules
        seed: Dict[str, Set[str]] = {}
        for r in existing_rules:
            rcode = normalize_code(r.account_code)
            for phrase in r.match_phrases():
                k = phrase_key(phrase)
                if k:
                    seed.setdefault(k, set()).add(rcode)
        collisions = detect_collisions(df, name_col, memo_col, seed=seed)
        # A Memo term can also collide across accounts on its own (two rules
        # would both claim it); treat those as collisions too.
        memo_owners: Dict[str, Set[str]] = {}
        for code, memos in memos_by_code.items():
            for m in memos:
                memo_owners.setdefault(_norm_space(m), set()).add(code)
        for k, owners in memo_owners.items():
            owners = owners | seed.get(k, set())
            if len(owners) > 1:
                collisions[k] = collisions.get(k, set()) | owners
        for key, codes in collisions.items():
            self.storage.upsert_collision(key, codes, client_id=client_id,
                                          source="ingest")
        report.collisions = {k: sorted(v) for k, v in collisions.items()}

        # ---- build / merge one rule per account ---------------------------
        fields = [c for c in (name_col, memo_col) if c]
        for code in names_by_code:
            phrases = names_by_code[code] + memos_by_code[code]
            usable: List[str] = []
            for p in phrases:
                if _norm_space(p) in collisions:
                    report.skipped_collision_phrases += 1
                    continue
                usable.append(p)
            if not usable:
                continue

            rule = by_code.get(code)
            if rule is None:
                saved = self.add_rule(
                    ", ".join(usable), code, match_type=match_type,
                    priority=self.INGEST_PRIORITY, fields=list(fields),
                    client_id=client_id,
                    notes=f"Ingested from coded rows ({code}): one rule per "
                          "account, vendor names first, then memo terms.")
                by_code[code] = saved
                report.rules_created += 1
                report.phrases_added += len(usable)
                report.accounts.append(code)
                continue

            have = {_norm_space(p) for p in rule.match_phrases()}
            new = [p for p in usable if _norm_space(p) not in have]
            if not new:
                continue
            rule.keyword = ", ".join(rule.match_phrases() + new)
            # Widen the scope so the new Name/Memo phrases can be seen.
            for f in fields:
                if rule.fields and f not in rule.fields:
                    rule.fields.append(f)
            self.update_rule(rule)
            report.rules_updated += 1
            report.phrases_added += len(new)
            report.accounts.append(code)

        logger.info("rules_ingested_per_account",
                    created=report.rules_created, updated=report.rules_updated,
                    phrases=report.phrases_added,
                    collisions=len(report.collisions))
        return report

    def ingest_existing_new_account_as_rules(
        self,
        df: pd.DataFrame,
        client_id: Optional[str] = None,
        source_text_col: str = "Description",
        source_text_cols: Optional[List[str]] = None,
    ) -> int:
        """Ingest coded rows as one rule per account (see
        :meth:`ingest_rules_per_account`).

        Returns the number of rules created **or extended** — ``0`` when every
        vendor on every account is already covered (idempotent).
        """
        report = self.ingest_rules_per_account(
            df, client_id=client_id, source_text_col=source_text_col,
            source_text_cols=source_text_cols)
        return report.rules_touched

    def create_sample_rules(self) -> List[KeywordRule]:
        """Create a couple of sample rules (for demos / tests)."""
        return [
            self.add_rule("Acme Screening", "6610S", match_type="contains",
                          notes="Sample: resident screening."),
            self.add_rule("Staples", "6200", match_type="contains",
                          notes="Sample: office supplies."),
        ]

    def seed_default_rules(self) -> None:
        """Add a couple of illustrative rules on first run (idempotent)."""
        if self.list_rules():
            return
        self.add_rule("Cloud Hosting", "6100", notes="Example rule (software/IT).")
        self.add_rule("SaaS Tools", "6100", notes="Software subscriptions.")
        logger.info("default_rules_seeded")

    def get_rule_application_summary(self, results: List[RuleRunResult]) -> dict:
        """Clean metrics for a rule run (success rate is over blanks only)."""
        return get_rule_application_summary(results)

    # ============================================================= #
    # Account-Level Knowledge Base
    #
    # Every rule and every successful match teaches the tool what a given GL
    # account "looks like": its keywords/phrases, sample transactions and how
    # often it's used. This reinforced memory improves matching accuracy and
    # explainability over time — the ProfitCoach flywheel: better coding ->
    # deeper client insight -> more data shared -> smarter automation.
    # ============================================================= #
    def upsert_account_profile(
        self,
        account_code: str,
        new_keywords: Optional[List[str]] = None,
        sample_text: Optional[str] = None,
        client_id: Optional[str] = None,
        notes: Optional[str] = None,
        usage_increment: int = 0,
    ) -> dict:
        """Create or reinforce the knowledge profile for one account.

        Merges ``new_keywords`` and ``sample_text`` into the existing profile
        (de-duplicated and bounded), optionally bumps ``usage_count`` and updates
        ``notes``. Account codes are canonicalised (so ``6618 s`` and ``6618S``
        share one profile). Never raises — learning must never break a rule save.
        """
        try:
            code = canonicalize_account(account_code)
            if not code:
                return {}
            base = base_account(code)
            existing = self.storage.get_account_profile(code, client_id) or {}

            keywords = list(existing.get("keywords", []))
            seen = set(keywords)
            for kw in (new_keywords or []):
                kw = (kw or "").strip()
                if kw and kw not in seen:
                    seen.add(kw)
                    keywords.append(kw)
            keywords = keywords[:_MAX_PROFILE_KEYWORDS]

            samples = list(existing.get("sample_texts", []))
            if sample_text:
                s = str(sample_text).strip()[:200]
                if s and s not in samples:
                    samples.append(s)
            samples = samples[-_MAX_PROFILE_SAMPLES:]

            usage = int(existing.get("usage_count", 0)) + int(usage_increment or 0)
            final_notes = notes if notes is not None else existing.get("notes", "")

            self.storage.save_account_profile(
                account_code=code, base_account=base, client_id=client_id,
                keywords=keywords, sample_texts=samples, usage_count=usage,
                notes=final_notes or "")
            return self.storage.get_account_profile(code, client_id) or {}
        except Exception as exc:  # noqa: BLE001 - learning is best-effort
            logger.warning("account_profile_upsert_failed",
                           code=account_code, error=str(exc))
            return {}

    def get_account_profile(self, account_code: str,
                            client_id: Optional[str] = None) -> dict:
        """Return the knowledge profile for an account (``{}`` if none yet)."""
        return self.storage.get_account_profile(
            canonicalize_account(account_code), client_id) or {}

    def list_account_profiles(self,
                              client_id: Optional[str] = None) -> List[dict]:
        """All learned account profiles (most-used first)."""
        return self.storage.list_account_profiles(client_id=client_id)

    def clear_account_profiles(self) -> int:
        """Forget the entire account knowledge base. Returns rows removed."""
        removed = self.storage.clear_account_profiles()
        logger.info("account_profiles_cleared", count=removed)
        return removed

    def _learn_from_rule(self, rule: KeywordRule) -> None:
        """Fold a rule's keywords into its account's knowledge profile."""
        kws = extract_keywords(rule.keyword)
        note = (rule.notes or "").strip() or None
        self.upsert_account_profile(
            rule.account_code, new_keywords=kws, notes=note,
            client_id=rule.client_id)

    def record_account_usage(
        self,
        account_code: str,
        sample_text: Optional[str] = None,
        client_id: Optional[str] = None,
        increment: int = 1,
    ) -> None:
        """Reinforce a profile after a real match (bumps usage + adds a sample)."""
        self.upsert_account_profile(
            account_code, sample_text=sample_text, client_id=client_id,
            usage_increment=increment)

    def refresh_learning_from_rules(self,
                                    client_id: Optional[str] = None) -> int:
        """Rebuild account profiles from *all* current rules.

        A one-click backfill: replays every rule through the learner so the
        knowledge base reflects the full rule set (useful after importing rules
        or upgrading). Returns the number of distinct accounts touched.
        """
        touched: set = set()
        for rule in self.list_rules(client_id=client_id):
            self._learn_from_rule(rule)
            touched.add((canonicalize_account(rule.account_code), rule.client_id))
        logger.info("account_learning_refreshed", accounts=len(touched))
        return len(touched)

    def account_insight(self, account_code: str,
                        client_id: Optional[str] = None,
                        max_keywords: int = 6) -> str:
        """Short, human phrase describing what's learned for an account.

        e.g. ``"learned patterns for 6618S: cred hub, screening, maintenance"``.
        Returns ``""`` when nothing is known yet.
        """
        prof = self.get_account_profile(account_code, client_id)
        if not prof:
            return ""
        kws = [k for k in prof.get("keywords", []) if k][:max_keywords]
        if not kws:
            return ""
        return (f"learned patterns for {prof['account_code']}: "
                + ", ".join(kws))

    def enrich_results(self, results: List[RuleRunResult],
                       client_id: Optional[str] = None) -> None:
        """Append account-knowledge context to each matched result's rationale.

        Mutates ``results`` in place. Cached per code so a run does at most one
        profile lookup per distinct account. Safe: no-op for accounts with no
        learned profile, and never raises.
        """
        cache: dict = {}
        for r in results:
            if r.status not in ("keyword_rule", "protected_existing"):
                continue
            code = (r.proposed_value or r.original_value or "").strip()
            if not code:
                continue
            canon = canonicalize_account(code)
            if canon not in cache:
                cache[canon] = self.account_insight(canon, client_id)
            insight = cache[canon]
            if insight and insight not in r.rationale:
                r.rationale = f"{r.rationale} Matches {insight}.".strip()

    def reinforce_from_rule_run(
        self,
        results: List[RuleRunResult],
        df: Optional[pd.DataFrame] = None,
        text_columns: Optional[List[str]] = None,
        client_id: Optional[str] = None,
    ) -> int:
        """Reinforce account profiles from a completed rule run.

        For every row a rule *filled*, bump that account's ``usage_count`` and
        capture a sample of the real transaction text — so accounts that fire
        often accumulate the richest, most trustworthy knowledge. Returns the
        number of distinct accounts reinforced.
        """
        agg: dict = {}
        for r in results:
            if r.status != "keyword_rule":
                continue
            code = canonicalize_account(r.proposed_value)
            if not code:
                continue
            entry = agg.setdefault(code, {"count": 0, "samples": []})
            entry["count"] += 1
            if df is not None and len(entry["samples"]) < 3:
                try:
                    row = df.iloc[r.row_index]
                    sample = self.build_match_text(row, text_columns)
                    if sample:
                        entry["samples"].append(sample[:200])
                except Exception:  # noqa: BLE001
                    pass
        for code, entry in agg.items():
            samples = entry["samples"] or [None]
            for i, sample in enumerate(samples):
                self.record_account_usage(
                    code, sample_text=sample, client_id=client_id,
                    increment=entry["count"] if i == 0 else 0)
        return len(agg)
