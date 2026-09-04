"""Core package for ProfitCoach — Fill Down Automation.

Modules are deliberately decoupled so each stage of the pipeline can be tested
and extended in isolation:

    data_loader      -> read & normalise transaction files
    similarity       -> embed transactions and group similar ones
    rules_manager    -> persistent keyword -> account-code rules
    fill_down_engine -> orchestrate propagation + confidence scoring
    review_queue     -> surface uncertain rows for human review
    exporter         -> write results back to Excel/CSV
"""

__all__ = [
    "data_loader",
    "similarity",
    "rules_manager",
    "fill_down_engine",
    "review_queue",
    "exporter",
    "config",
]
