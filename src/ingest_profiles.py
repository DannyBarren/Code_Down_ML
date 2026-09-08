"""Source profiles for known accounting-export shapes (AppFolio / QuickBooks).

Accountants download transaction exports and drop them in — they should not
have to rename columns first. A profile maps the *logical* roles the engine
cares about (name / memo / amount / date / account / payee) to the header
spellings a given vendor export actually uses.

Detection is a simple header-overlap score: the profile whose aliases resolve
the most roles against the file's actual headers wins, provided it resolves
the three core roles (name, memo, amount) and beats every other profile
outright. If nothing wins, the loader's generic resolver is used unchanged.
Detection never fails a load — a missed profile just means "generic export".

No network access, no vendor APIs: this is purely header recognition on files
the accountant already downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SourceProfile:
    """A named export shape: logical role -> candidate headers.

    ``distinctive`` headers are spellings only this vendor's export uses; they
    break ties between profiles that resolve the same roles.
    """

    key: str
    label: str                       # user-facing, e.g. "AppFolio transaction detail"
    aliases: Dict[str, Tuple[str, ...]]
    distinctive: Tuple[str, ...] = ()

    def resolve(self, role: str, available: List[str]) -> Optional[str]:
        """The actual header for ``role`` among ``available`` (tolerant)."""
        norm = {str(c).strip().lower(): c for c in available}
        for cand in self.aliases.get(role, ()):
            hit = norm.get(str(cand).strip().lower())
            if hit is not None:
                return hit
        return None


# --------------------------------------------------------------------------- #
# Known export shapes
# --------------------------------------------------------------------------- #
# AppFolio's "Transaction Detail" report: Name + a combined Memo/Description
# column, Amount, Date, and an Account column for the existing coding.
APPFOLIO_TRANSACTION_DETAIL = SourceProfile(
    key="appfolio_transaction_detail",
    label="AppFolio transaction detail",
    aliases={
        "name": ("Name", "Payee", "Payee Name", "Vendor", "Payee/Payer"),
        "memo": ("Memo/Description", "Memo / Description", "Memo",
                 "Description"),
        "amount": ("Amount", "Amount (USD)", "Debit", "Credit", "Total"),
        "date": ("Date", "Transaction Date", "Posting Date"),
        "account": ("Account", "GL Account", "Category"),
        "payee": ("Payee", "Name", "Vendor"),
    },
    # The combined "Memo/Description" header is the AppFolio signature.
    distinctive=("Memo/Description", "Memo / Description"),
)

# QuickBooks "Transaction List" exports: Payee/Vendor rather than Name, Memo,
# and often a "Txn Date" spelling.
QUICKBOOKS_TRANSACTION_LIST = SourceProfile(
    key="quickbooks_transaction_list",
    label="QuickBooks transaction list",
    aliases={
        "name": ("Payee", "Vendor", "Name", "Supplier", "Payee Name"),
        "memo": ("Memo", "Memo/Description", "Description"),
        "amount": ("Amount", "Amount (USD)", "Debit", "Credit", "Total"),
        "date": ("Txn Date", "Date", "Transaction Date", "Posting Date"),
        "account": ("Account", "Category", "GL Account"),
        "payee": ("Payee", "Vendor", "Supplier", "Name"),
    },
    # QuickBooks signatures: a Payee/Vendor column and the "Txn Date" spelling.
    distinctive=("Payee", "Vendor", "Supplier", "Txn Date"),
)

PROFILES: Tuple[SourceProfile, ...] = (
    APPFOLIO_TRANSACTION_DETAIL,
    QUICKBOOKS_TRANSACTION_LIST,
)

# A profile must resolve at least these roles to be considered a match.
_REQUIRED_ROLES = ("name", "memo", "amount")


def score_profile(profile: SourceProfile, headers: List[str]) -> int:
    """Role-overlap score, weighted by the profile's distinctive headers."""
    roles = sum(1 for role in profile.aliases
                if profile.resolve(role, headers) is not None)
    norm = {str(h).strip().lower() for h in headers}
    bonus = 2 * sum(1 for d in profile.distinctive
                    if d.strip().lower() in norm)
    return roles + bonus


def detect_profile(headers: List[str]) -> Optional[SourceProfile]:
    """Pick the winning source profile for these headers, or ``None``.

    The winner must resolve every required role (name, memo, amount) and have
    a strictly higher role-overlap score than any other profile — a tie means
    the shape is ambiguous and the loader stays on its generic resolver.
    """
    best: Optional[SourceProfile] = None
    best_score = 0
    tied = False
    for profile in PROFILES:
        if any(profile.resolve(role, headers) is None
               for role in _REQUIRED_ROLES):
            continue
        score = score_profile(profile, headers)
        if score > best_score:
            best, best_score, tied = profile, score, False
        elif score == best_score:
            tied = True
    if best is None or tied:
        return None
    return best


def profile_by_key(key: str) -> Optional[SourceProfile]:
    """Look up a profile by its stable key (``""``/unknown -> ``None``)."""
    for profile in PROFILES:
        if profile.key == key:
            return profile
    return None


def profile_chip(key: str) -> str:
    """One-line chip text for the detected shape, e.g. 'Read as AppFolio …'."""
    profile = profile_by_key(key)
    return f"Read as {profile.label}" if profile else "Generic export"


def profile_similarity_candidates(profile: SourceProfile) -> List[str]:
    """Ordered candidate headers for similarity text when a profile wins.

    Vendor-first, mirroring the configured keyword-source order: the name/payee
    roles lead, then memo, then account.
    """
    ordered: List[str] = []
    for role in ("name", "payee", "memo", "account"):
        for cand in profile.aliases.get(role, ()):
            if cand not in ordered:
                ordered.append(cand)
    return ordered
