"""Keyword-rule management and matching.

Rules give users a fast, deterministic override for the patterns they
already know (e.g. "Cloud Hosting" -> "6100"). They are persisted in SQLite via the
:class:`~utils.storage.Storage` layer and always take precedence over the
fuzzy similarity engine.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class RuleMatch:
    account_code: str
    rule: KeywordRule


def get_rule_application_summary(results: List[RuleRunResult]) -> dict:
    """Summarise a rule run: protected vs filled vs left-blank + success rate.

    ``success_rate`` is measured over **blank rows only** (filled / blanks),
    because rules never touch already-coded rows — that is the metric the client
    cares about for real-time coding.
    """
    protected = sum(1 for r in results if r.status == "protected_existing")
    filled = sum(1 for r in results if r.status == "keyword_rule")
    left_blank = sum(1 for r in results if r.status == "no_rule_match")
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
        (optional) lets field-scoped rules inspect individual columns.
        """
        rules = rules if rules is not None else self.list_rules(enabled_only=True)
        for rule in rules:
            if not rule.enabled:
                continue
            if self._rule_matches(rule, combined_text, row):
                return RuleMatch(account_code=rule.account_code, rule=rule)
        return None

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
        """True if a window of the candidate ~matches the keyword (typo-tolerant)."""
        if not key_space or not cand_space:
            return False
        if key_space in cand_space:
            return True
        ktokens = key_space.split()
        ctokens = cand_space.split()
        if not ktokens or not ctokens:
            return False
        width = len(ktokens)
        last = max(1, len(ctokens) - width + 1)
        for start in range(last):
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
    ) -> tuple[pd.DataFrame, List[RuleRunResult]]:
        """Fill blank ``New Account`` cells from keyword rules — deterministically.

        Guarantees, in priority order:
          * A row whose ``New Account`` is already non-blank is **never** touched
            (status ``protected_existing``).
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
        """First rule (in the given order) that matches — never raises."""
        for rule in ordered_rules:
            try:
                if self._rule_matches(rule, match_text, row):
                    return rule
            except Exception as exc:  # noqa: BLE001 - never fail a whole run
                logger.warning("rule_match_error", rule=rule.keyword,
                               error=str(exc))
                continue
        return None

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

    def ingest_existing_new_account_as_rules(
        self,
        df: pd.DataFrame,
        client_id: Optional[str] = None,
        source_text_col: str = "Description",
    ) -> int:
        """Turn pre-filled ``New Account`` rows into high-priority exact rules.

        Each coded row becomes a reusable exact-match rule (``priority=10`` so it
        wins over broad ``contains`` rules). The rule keyword is taken from
        ``source_text_col`` when present, else the first meaningful text column on
        the row. Duplicates (same phrase + code, or an already-existing rule) are
        skipped. Returns the number of new rules created.
        """
        if NEW_ACCOUNT_COL not in df.columns:
            return 0

        existing = {(r.keyword.strip().lower(), normalize_code(r.account_code))
                    for r in self.list_rules()}
        created = 0
        for idx in df.index:
            code_raw = df.at[idx, NEW_ACCOUNT_COL]
            code_str = "" if _is_blank(code_raw) else str(code_raw).strip()
            if not code_str:
                continue

            row = df.loc[idx]
            text = ""
            if source_text_col in df.columns and _has_value(row.get(source_text_col)):
                text = str(row.get(source_text_col)).strip()
            else:
                for c in df.columns:
                    if str(c).startswith("_") or c in (NEW_ACCOUNT_COL, RULE_NOTES_COL):
                        continue
                    if _has_value(row.get(c)):
                        text = str(row.get(c)).strip()
                        break
            if not text:
                continue

            norm_code = normalize_code(code_str)
            key = (text.lower(), norm_code)
            if key in existing:
                continue
            self.add_rule(
                text, norm_code, match_type="exact", priority=10,
                client_id=client_id,
                notes=f"Ingested from pre-filled New Account ({norm_code}).")
            existing.add(key)
            created += 1

        logger.info("rules_ingested_from_new_account", count=created)
        return created

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
