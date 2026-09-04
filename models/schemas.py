"""Pydantic models that describe the data flowing through the pipeline.

Keeping these in one place gives us a single source of truth for the shape of
rules, fill actions, training data and run summaries — which is what gets
persisted, shown in the UI and exported.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Timezone-aware UTC now (replaces deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)


class FillSource(str, Enum):
    """Why a particular ``New Account`` value was assigned to a row."""

    SEED = "seed"                # Value was already present in the upload.
    RULE = "rule"                # Matched a user-defined keyword rule.
    LEARNED = "learned"          # Matched a previously approved mapping.
    ML = "ml"                    # Predicted by a trained ML model.
    SIMILARITY = "similarity"    # Propagated from a similar transaction group.
    MANUAL = "manual"            # Edited by the user in the review queue.
    NONE = "none"                # No value could be determined.


class FillAction(str, Enum):
    """What the engine decided to do with a row."""

    KEPT_SEED = "kept_seed"          # Existing value preserved.
    AUTO_FILLED = "auto_filled"      # Confidence >= auto cutoff, applied.
    FILLED_REVIEW = "filled_review"  # Applied but flagged for review.
    NEEDS_REVIEW = "needs_review"    # Left blank, sent to review queue.
    NO_MATCH = "no_match"            # Nothing similar found, left blank.


class FillResult(BaseModel):
    """Per-row outcome of a fill-down run."""

    row_index: int = Field(..., description="0-based index into the dataframe.")
    original_value: Optional[str] = None
    proposed_value: Optional[str] = None
    confidence: float = 0.0
    source: FillSource = FillSource.NONE
    # Human-readable engine label, e.g. "rules", "ml:logreg", "similarity",
    # "ml+similarity". Always populated so every output row is auditable.
    engine_used: str = "none"
    action: FillAction = FillAction.NO_MATCH
    group_id: Optional[int] = None
    rationale: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class KeywordRule(BaseModel):
    """A user-defined rule: when a keyword/phrase is found, assign ``account_code``.

    ``keyword`` may hold **several comma-separated phrases** (OR logic): a rule
    like ``"cred hub, credit hub, cred-hub"`` matches a row containing *any* of
    those phrases. Commas are not split for ``regex`` rules (a pattern may
    legitimately contain them), so a regex rule is always a single pattern.

    ``fields`` restricts matching to specific columns (empty = search all text).
    ``priority`` orders rule application (lower runs first); exact rules ingested
    from pre-filled ``New Account`` values use a low number so they win.
    ``client_id`` scopes a rule to one client (``None`` = global, applies to all).
    """

    id: Optional[int] = None
    keyword: str = Field(..., min_length=1)
    account_code: str = Field(..., min_length=1)
    # Which logical fields to search. Empty -> search the combined text.
    fields: List[str] = Field(default_factory=list)
    match_type: str = Field(
        "contains",
        description="contains | exact | starts_with | ends_with | fuzzy | regex")
    case_sensitive: bool = False
    enabled: bool = True
    notes: str = ""
    priority: int = 100
    client_id: Optional[str] = None
    created_at: Optional[datetime] = None

    @field_validator("keyword", "account_code")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def match_phrases(self) -> List[str]:
        """The individual phrases this rule matches (OR logic).

        Regex rules are a single pattern; every other type splits on commas so
        users can pack multiple keywords/phrases for one account into one rule.
        """
        if self.match_type == "regex":
            kw = (self.keyword or "").strip()
            return [kw] if kw else []
        return [p.strip() for p in (self.keyword or "").split(",") if p.strip()]


class RuleRunResult(BaseModel):
    """Per-row audit record for a rule-driven run.

    ``status`` is one of ``protected_existing`` (row already had a New Account
    value and was left untouched), ``keyword_rule`` (a rule filled the blank),
    or ``no_rule_match`` (blank row, no rule matched).
    """

    row_index: int
    original_value: str = ""
    proposed_value: str = ""
    engine_used: str = "rules"
    confidence: float = 0.0
    rationale: str = ""
    matched_rule_name: str = ""
    matched_pattern: str = ""
    status: str = "no_rule_match"

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class LearnedMapping(BaseModel):
    """A text signature that has been confirmed to map to an account code.

    These accumulate over time as the user approves fill-downs, letting the
    tool get more accurate (and more confident) on each subsequent file.
    """

    id: Optional[int] = None
    signature: str
    account_code: str
    hits: int = 1
    last_seen: Optional[datetime] = None


class TrainingExample(BaseModel):
    """A labelled example captured from a human approval, used to train models."""

    id: Optional[int] = None
    text: str
    label: str
    confidence: float = 1.0
    engine_used: str = "manual"
    timestamp: Optional[datetime] = None
    approved_by: str = "user"


class TransactionGroup(BaseModel):
    """A cluster of similar transactions discovered by the similarity engine."""

    group_id: int
    row_indices: List[int]
    seed_account: Optional[str] = None
    seed_count: int = 0
    avg_similarity: float = 0.0
    representative_text: str = ""


class ModelStatus(BaseModel):
    """Status / metrics for one trained model type (shown in the Models page)."""

    name: str
    available: bool = False
    trained: bool = False
    version: int = 0
    n_examples: int = 0
    n_labels: int = 0
    accuracy: Optional[float] = None
    trained_at: Optional[str] = None
    note: str = ""


class RunSummary(BaseModel):
    """High-level statistics for a single automation run (persisted to history)."""

    id: Optional[int] = None
    run_at: datetime = Field(default_factory=_utcnow)
    file_name: str = ""
    total_rows: int = 0
    seeds: int = 0
    auto_filled: int = 0
    filled_review: int = 0
    needs_review: int = 0
    no_match: int = 0
    groups_found: int = 0
    embedding_backend: str = ""
    similarity_threshold: float = 0.0
    auto_apply_cutoff: float = 0.0
    notes: str = ""

    @property
    def total_filled(self) -> int:
        return self.auto_filled + self.filled_review

    @property
    def remaining_blank(self) -> int:
        return self.needs_review + self.no_match
