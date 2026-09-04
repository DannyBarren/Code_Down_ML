"""Generate a neutral, realistic sample transaction file for demos/tests.

Run:  python -m utils.sample_data   (from the project root)

Produces ``data/sample_transactions.xlsx`` mimicking a generic accounting
transaction export, with a few seed values pre-filled in the ``New Account``
column. Vendors span multiple industries (software, supplies, utilities,
insurance, payroll, marketing, logistics) so the demo stays industry-neutral.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (Name/Payee, Description, Memo, account/category code) templates.
# Neutral, fictional vendors across common SMB spend categories.
VENDORS = [
    ("Cloud Hosting Co", "Monthly cloud hosting", "Production servers", "6100"),
    ("SaaS Tools Inc", "Software subscription", "Team licenses", "6100"),
    ("Acme Supplies", "General supplies purchase", "Inventory restock", "6200"),
    ("Metro Hardware", "Hardware and tools", "Maintenance parts", "6200"),
    ("BuildRight Materials", "Building materials", "Project materials", "6210"),
    ("Prime Office Supply", "Office supplies", "Paper and toner", "6210"),
    ("City Utilities", "Utility bill", "Electric and water", "6300"),
    ("PowerGrid Energy", "Electricity service", "Facility power", "6300"),
    ("FastNet Internet", "Internet service", "Office connectivity", "6310"),
    ("SecureShield Insurance", "Insurance premium", "Liability coverage", "6400"),
    ("Payroll Partners", "Payroll processing fee", "Bi-weekly payroll", "6500"),
    ("AdBoost Marketing", "Online advertising", "Lead generation", "6600"),
    ("Swift Logistics", "Freight and shipping", "Inbound delivery", "6700"),
    ("Greenscape Services", "Landscaping service", "Grounds maintenance", "6730"),
    ("Bank Service Charge", "Monthly bank fee", "Account maintenance", ""),
    ("Misc Vendor LLC", "One-off consulting", "Special project", ""),
]


def generate(rows: int = 220, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    start = date(2025, 1, 1)
    records = []
    for _ in range(rows):
        name, desc, memo, code = random.choice(VENDORS)
        d = start + timedelta(days=random.randint(0, 120))
        amount = round(random.uniform(25, 2500), 2)
        records.append({
            "Date": d.isoformat(),
            "Description": desc,
            "Name": name,
            "Memo": memo,
            "Account": "Operating Expenses",
            "Amount": amount,
            "New Account": "",  # filled selectively below
        })

    df = pd.DataFrame(records)

    # Seed one representative row per known code (like a kickoff call would).
    for code in {c for *_rest, c in VENDORS if c}:
        idxs = df.index[df["New Account"] == ""].tolist()
        matching = [
            i for i in idxs
            if any(
                key.lower() in (str(df.at[i, "Name"]) + str(df.at[i, "Description"])).lower()
                for name, desc, memo, c in VENDORS if c == code
                for key in [name]
            )
        ]
        for i in matching[:1]:  # seed only the first example of each
            df.at[i, "New Account"] = code

    return df


def main() -> None:
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate()
    xlsx_path = out_dir / "sample_transactions.xlsx"
    csv_path = out_dir / "sample_transactions.csv"
    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)
    seeds = (df["New Account"].astype(str).str.strip() != "").sum()
    print(f"Wrote {len(df)} rows ({seeds} seeds) to:\n  {xlsx_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
