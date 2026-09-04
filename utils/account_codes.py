"""Helpers for account / category codes, including an optional letter suffix.

Codes look like ``6100`` or ``6100A`` — a numeric base account (``6100``) plus
an optional single-letter sub-account suffix (``A``). Real data is messy:
``6100 a``, ``6100-A``, ``6100.0`` (Excel turned it into a float) all show up.

These helpers give us one canonical form so seeds, rules and learned mappings
all line up, and so we can reason about the *base* account when grouping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# base digits, optional separator, optional single trailing letter
_CODE_RE = re.compile(r"^\s*(\d{2,})(?:\.0+)?\s*[-_ ]?\s*([A-Za-z])?\s*$")


@dataclass(frozen=True)
class AccountCode:
    """A parsed account code."""

    raw: str
    base: str            # numeric part, e.g. "6100"
    suffix: str          # single uppercase letter or "" e.g. "S"

    @property
    def normalized(self) -> str:
        """Canonical form, e.g. ``6100A`` (no spaces, uppercase suffix)."""
        return f"{self.base}{self.suffix}" if self.base else self.raw.strip()

    @property
    def is_sub_account(self) -> bool:
        return bool(self.suffix)


def parse_account_code(code: object) -> AccountCode:
    """Parse ``code`` into base + suffix. Non-conforming codes pass through.

    Examples
    --------
    ``"6100A"``  -> base="6100", suffix="S"
    ``"6100 s"`` -> base="6100", suffix="S"
    ``"6100.0"`` -> base="6100", suffix=""
    ``"Misc"``   -> base="",     suffix="" (raw preserved)
    """
    raw = "" if code is None else str(code).strip()
    if not raw:
        return AccountCode(raw="", base="", suffix="")

    m = _CODE_RE.match(raw)
    if not m:
        return AccountCode(raw=raw, base="", suffix="")
    base = m.group(1)
    suffix = (m.group(2) or "").upper()
    return AccountCode(raw=raw, base=base, suffix=suffix)


def normalize_code(code: object) -> str:
    """Return the canonical string form of an account code."""
    return parse_account_code(code).normalized


def base_account(code: object) -> str:
    """Return just the numeric base (``6100`` for ``6100A``), or "" if none."""
    return parse_account_code(code).base


def is_sub_account(code: object) -> bool:
    """True when the code carries a sub-account suffix (e.g. ``6100A``)."""
    return parse_account_code(code).is_sub_account


def same_base(a: object, b: object) -> bool:
    """True when two codes share the same numeric base account."""
    ba, bb = base_account(a), base_account(b)
    return bool(ba) and ba == bb


def describe(code: object) -> Optional[str]:
    """Human-readable description for tooltips, or ``None`` for plain codes."""
    parsed = parse_account_code(code)
    if not parsed.base:
        return None
    if parsed.is_sub_account:
        return f"Sub-account {parsed.suffix} of account {parsed.base}"
    return f"Account {parsed.base}"
