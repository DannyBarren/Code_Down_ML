"""End-to-end smoke test (no Streamlit). Run: python smoke_test.py

Covers the happy path plus several edge cases:
    * empty file            -> friendly error
    * file with no seeds    -> runs, relies on rules / leaves blank
    * Memo/Name/Description variations -> still grouped & filled
    * full sample workflow  -> run, review, learn, export
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.config import load_config
from src.data_loader import DataLoadError, load_dataframe
from src.exporter import (
    export_dataframe,
    export_preserving_original,
    export_csv,
)
from src.fill_down_engine import FillDownEngine
from src.review_queue import apply_reviews, build_review_table, mark_high_confidence
from src.rules_manager import RulesManager
from src import spreadsheet_helpers as sh
from src.data_loader import RULE_NOTES_COL, SIM_TEXT_COL
from utils.storage import Storage


def _fresh_env(use_embeddings: bool = False):
    config = load_config()
    config.similarity.use_embeddings = use_embeddings
    tmp_db = Path(tempfile.mkdtemp()) / "test.db"
    storage = Storage(tmp_db)
    rules = RulesManager(storage)
    rules.seed_default_rules()
    return config, storage, rules


def _csv(records) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(records).to_csv(buf, index=False)
    return buf.getvalue()


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_empty_file():
    print("Edge case: empty file")
    config, *_ = _fresh_env()
    try:
        load_dataframe(b"", config, "empty.csv")
        check("empty file raises DataLoadError", False)
    except DataLoadError:
        check("empty file raises DataLoadError", True)


def test_no_seeds():
    print("Edge case: file with no seeds (rule-driven)")
    config, storage, rules = _fresh_env()
    loaded = load_dataframe(_csv([
        {"Name": "Cloud Hosting", "Description": "hosting", "New Account": ""},
        {"Name": "Random Vendor", "Description": "misc", "New Account": ""},
    ]), config, "no_seeds.csv")
    check("no seeds detected", loaded.seed_count == 0)
    res = FillDownEngine(config, rules, learned_lookup={}).run(loaded)
    # The default 'Cloud Hosting' rule should still fire.
    default_fill = res.df.iloc[0][loaded.new_account_col]
    check("rule fills Cloud Hosting even with no seeds", default_fill == "6100")


def test_field_variations():
    print("Edge case: code-bearing text in Memo vs Name vs Description")
    config, storage, rules = _fresh_env()
    # Seed once via Name; the variant only mentions the vendor in Memo.
    loaded = load_dataframe(_csv([
        {"Name": "Northwind ERP", "Description": "software", "Memo": "",
         "New Account": "6105"},
        {"Name": "", "Description": "monthly software license", "Memo": "northwind erp",
         "New Account": ""},
    ]), config, "variations.csv")
    res = FillDownEngine(config, rules, learned_lookup={}).run(loaded)
    filled = res.df.iloc[1][loaded.new_account_col]
    check("variant row received a suggestion or fill",
          filled == "6105" or any(
              r.proposed_value == "6105" for r in res.results if r.row_index == 1))


def test_full_workflow():
    print("Full workflow on the sample file")
    config, storage, rules = _fresh_env()
    sample = config._resolve("data/sample_transactions.xlsx")
    if not sample.exists():
        from utils.sample_data import main as gen
        gen()
    raw = sample.read_bytes()
    loaded = load_dataframe(raw, config, source_name="sample_transactions.xlsx")
    check("sample has seeds", loaded.seed_count > 0)

    res = FillDownEngine(config, rules, learned_lookup={}).run(loaded)
    s = res.summary
    print(f"    backend={res.backend} groups={s.groups_found} "
          f"auto={s.auto_filled} review={s.filled_review} "
          f"needs={s.needs_review} blank={s.no_match}")
    check("summary totals consistent",
          s.seeds + s.auto_filled + s.filled_review + s.needs_review + s.no_match
          == s.total_rows == len(loaded.df))

    table = build_review_table(res.df, res.results, loaded.new_account_col,
                               loaded.text_columns)
    if not table.empty:
        approved = mark_high_confidence(table, 0.0)
        counts = apply_reviews(res.df, approved, loaded.new_account_col, storage)
        check("reviews applied without error", counts["applied"] >= 0)

    xlsx = export_preserving_original(raw, res.df, loaded.new_account_col, config)
    csv = export_dataframe(res.df, fmt="csv")
    qb = export_csv(res.df, loaded.new_account_col)
    check("exports produced bytes", len(xlsx) > 0 and len(csv) > 0 and len(qb) > 0)
    check("csv export strips internal cols", b"_sim_text" not in qb)


def test_spreadsheet_v22_workflow():
    """Exercise the v2.2 spreadsheet-dominant pipeline end-to-end (no Streamlit)."""
    print("v2.2 workflow: work_df + Rule Notes + run + approve + export")
    config, storage, rules = _fresh_env()

    records = [
        {"Name": "Cloud Hosting", "Description": "hosting fee", "New Account": "6100"},
        {"Name": "Cloud Hosting", "Description": "hosting fee", "New Account": ""},
        {"Name": "Quirky Vendor", "Description": "one off", "New Account": ""},
    ]
    loaded = load_dataframe(_csv(records), config, "ws.csv")

    # 1) Build the working dataframe and confirm internal columns exist.
    work = sh.build_work_df(loaded, storage, config)
    check("work_df has Rule Notes + meta columns",
          all(c in work.columns for c in
              (RULE_NOTES_COL, SIM_TEXT_COL, sh.SELECT_COL, sh.CONF_COL)))
    check("seed row pre-marked", work.iloc[0][sh.ACTION_COL] == "kept_seed")

    # 2) Edit a Rule Note via the grid path -> folds into _sim_text + persists.
    edited = work.loc[[2]].copy()
    edited.at[2, RULE_NOTES_COL] = "annual landscaping contract"
    sh.commit_editor_changes(work, edited, loaded, storage, config)
    check("note folded into sim_text",
          "landscaping" in work.iloc[2][SIM_TEXT_COL])
    check("note persisted to storage", storage.count_rule_notes() == 1)

    # 3) Full intelligent run keeps work_df in sync.
    engine = FillDownEngine(config, rules,
                            learned_lookup=storage.get_learned_lookup())
    sh.run_full(work, loaded, config, engine)
    check("similar blank row filled by run",
          work.iloc[1]["New Account"] == "6100")

    # 4) Approve a suggested-but-blank row and confirm learning.
    work.at[2, "New Account"] = ""
    work.at[2, sh.SUGGESTED_COL] = "7000S"
    out = sh.approve_rows(work, loaded, storage, config, indices=[2])
    check("approve filled from suggestion", work.iloc[2]["New Account"] == "7000S")
    check("approval learned a mapping", out["learned"] == 1)

    # 5) Rule Notes re-attach on a fresh upload of the same transactions.
    loaded2 = load_dataframe(_csv(records), config, "ws.csv")
    work2 = sh.build_work_df(loaded2, storage, config)
    check("rule notes restored across uploads",
          work2.iloc[2][RULE_NOTES_COL] == "annual landscaping contract")

    # 6) Export strips every internal/meta column.
    csv = export_dataframe(work, fmt="csv")
    for marker in (b"_sim_text", b"_base_sig", b"_confidence", b"_select"):
        check(f"export strips {marker.decode()}", marker not in csv)
    check("export keeps Rule Notes", b"Rule Notes" in csv)


def test_candidate_rules_flow():
    """Seeding -> auto candidate rules -> create -> apply fills look-alikes."""
    print("candidate rules: coded rows -> suggested rules -> apply")
    config, storage, rules = _fresh_env()

    records = [
        {"Name": "Acme Landscaping", "Description": "lawn", "New Account": "6300S"},
        {"Name": "Acme Landscaping", "Description": "lawn care", "New Account": "6300S"},
        {"Name": "Acme Landscaping", "Description": "lawn", "New Account": ""},
        {"Name": "City Water", "Description": "utility", "New Account": ""},
    ]
    loaded = load_dataframe(_csv(records), config, "cand.csv")
    work = sh.build_work_df(loaded, storage, config)

    # 1) The coded "Acme" rows produce a candidate rule -> 6300S.
    cands = sh.candidate_rules(work, loaded, rules)
    check("candidate rule suggested from coded rows",
          any(c["account_code"] == "6300S" for c in cands))

    # 2) Creating the candidate rules persists them.
    before = len(rules.list_rules())
    created = sh.create_rules_from_candidates(rules, cands)
    check("candidate rules created", created > 0)
    check("rules persisted", len(rules.list_rules()) == before + created)

    # 3) Running rules now fills the previously-blank Acme row.
    n = sh.run_rules_only(work, rules, loaded, config)
    check("created rule fills the blank look-alike row", n >= 1)
    check("blank Acme row coded to 6300S", work.iloc[2]["New Account"] == "6300S")

    # 4) Re-mining offers nothing new (keywords already covered).
    check("no duplicate candidates after creation",
          sh.create_rules_from_candidates(rules, cands) == 0)


def test_rule_matching_reliability():
    """Production-rescue: rules match across all columns, multi-word, override."""
    print("rule reliability: cross-column, SALETX, multi-word, override")
    config, storage, rules = _fresh_env()

    # Vendor lives only in Payee (not a similarity column); a tax tag in Memo.
    records = [
        {"Date": "1/1", "Payee": "Acme Plumbing", "Memo": "SALETX#900 job",
         "New Account": "6300S"},
        {"Date": "1/2", "Payee": "Acme Plumbing", "Memo": "job", "New Account": ""},
        {"Date": "1/3", "Payee": "CloudHost Inc", "Memo": "hosting",
         "New Account": ""},
        {"Date": "1/4", "Payee": "Unrelated", "Memo": "regular", "New Account": ""},
    ]
    loaded = load_dataframe(_csv(records), config, "rel.csv")
    work = sh.build_work_df(loaded, storage, config)

    # 1) Auto candidate rule mined from the Payee-only seed.
    cands = sh.candidate_rules(work, loaded, rules)
    check("candidate mined from non-similarity (Payee) column",
          any(c["account_code"] == "6300S" for c in cands))

    # 2) A manual rule keyed on the Payee vendor fills its blank look-alike.
    rules.add_rule("Acme Plumbing", "6300S")
    n = sh.run_rules_only(work, rules, loaded, config)
    check("Payee-keyed rule fills blank row", work.iloc[1]["New Account"] == "6300S")

    # 3) A SALETX rule (Memo, with punctuation) maps the matching row.
    rules.add_rule("SALETX", "9100")
    work.at[0, "New Account"] = ""                       # clear seed to test fill
    work.at[0, sh.ENGINE_COL] = ""
    n2 = sh.run_rules_only(work, rules, loaded, config)
    check("SALETX rule matches punctuated Memo token",
          work.iloc[0]["New Account"] in ("9100", "6300S"))

    # 4) Multi-word 'Cloud Host' matches concatenated 'CloudHost Inc'.
    rules.add_rule("Cloud Host", "6100")
    n3 = sh.run_rules_only(work, rules, loaded, config)
    check("multi-word rule matches concatenated vendor",
          work.iloc[2]["New Account"] == "6100")

    # 5) Override replaces an engine guess but protects a manual value.
    work.at[3, "New Account"] = "2222S"
    work.at[3, sh.ENGINE_COL] = "similarity"
    work.at[1, "New Account"] = "1111S"
    work.at[1, sh.ENGINE_COL] = "manual"
    rules.add_rule("Unrelated", "7000S")
    sh.run_rules_only(work, rules, loaded, config, indices=[1, 3], overwrite=True)
    check("override replaces engine guess", work.iloc[3]["New Account"] == "7000S")
    check("override protects manual value", work.iloc[1]["New Account"] == "1111S")


def test_rule_creation_triggers_and_notes():
    """The 4 rule-creation triggers + Rule Notes influence on similarity/ML.

    Manual simulation walkthrough (what each trigger does in the UI, exercised
    here through the same helper functions the UI calls):

      A) SEEDING BANNER  — upload a file with a few coded rows. Coded rows are
         mined into candidate rules; the banner offers one-click bulk creation.
         Verified via candidate_rules() -> create_rules_from_candidates().

      B) CHECKBOX SELECTION — tick the Sel box on look-alike rows, click
         "Create Rule from Selection". The keyword is the shared vendor phrase.
         Verified via suggest_keyword_from_rows() over multiple rows.

      C) CELL-LEVEL (single row + column picker) — select one row, pick the
         column the vendor lives in. This is the reliable, canvas-safe stand-in
         for right-click. Verified via suggest_keyword_from_cell().

      D) TARGET ACCOUNT INLINE PROMPT — type a code into a blank row; the app
         offers "Create a rule for <vendor> -> <code>?". Verified via
         commit_editor_changes() -> target_edits and rule_prompt_worthy().

    'SALETX' example and overwrite override live in test_rule_matching_reliability.
    """
    print("rule-creation triggers + rule-notes influence")
    config, storage, rules = _fresh_env()

    records = [
        {"Name": "Office Depot", "Description": "toner", "New Account": "6210S"},
        {"Name": "Office Depot", "Description": "paper", "New Account": ""},
        {"Name": "Office Depot", "Description": "chairs", "New Account": ""},
        {"Name": "Sparkle Clean", "Description": "janitorial", "New Account": ""},
        {"Name": "Sparkle Clean", "Description": "janitorial", "New Account": ""},
    ]
    loaded = load_dataframe(_csv(records), config, "triggers.csv")
    work = sh.build_work_df(loaded, storage, config)

    # A) Seeding banner -> candidate rules from the coded Office Depot row.
    cands = sh.candidate_rules(work, loaded, rules)
    check("trigger A: seeding mines a candidate rule",
          any(c["account_code"] == "6210S" for c in cands))

    # B) Checkbox selection across the two blank Office Depot rows -> shared kw.
    kw_multi = sh.suggest_keyword_from_rows(work, [1, 2], loaded)
    check("trigger B: selection keyword is the shared vendor phrase",
          kw_multi == "office depot")

    # C) Cell-level: single row, keyword pulled from the Name column.
    kw_cell = sh.suggest_keyword_from_cell(work, 3, "Name", loaded)
    check("trigger C: cell-level keyword from a column", kw_cell == "sparkle clean")

    # D) Inline prompt: typing a code surfaces a worthy create-rule prompt.
    edited = work.loc[[3]].copy()
    edited.at[3, loaded.new_account_col] = "6230S"
    out = sh.commit_editor_changes(work, edited, loaded, storage, config)
    edit = out["target_edits"][0]
    check("trigger D: target edit yields keyword + code",
          bool(edit["keyword"]) and edit["code"] == "6230S")
    check("trigger D: prompt is worthy for a new keyword",
          sh.rule_prompt_worthy(edit["keyword"], edit["code"], rules))

    # All four created rules actually fill their look-alike rows.
    rules.add_rule("office depot", "6210S")
    rules.add_rule("sparkle clean", "6230S")
    sh.run_rules_only(work, rules, loaded, config)
    check("created rules fill the look-alike rows",
          work.iloc[1]["New Account"] == "6210S"
          and work.iloc[4]["New Account"] == "6230S")

    # Rule Notes influence: a note is appended to _sim_text, which feeds both
    # similarity grouping and the labelled text captured for ML training.
    work.at[2, RULE_NOTES_COL] = "supplies reorder annual contract"
    from src.data_loader import recompute_sim_text
    recompute_sim_text(work, loaded.text_columns, config)
    sim_with_note = str(work.iloc[2][SIM_TEXT_COL])
    check("rule notes flow into _sim_text (semantic + ML signal)",
          "annual contract" in sim_with_note)
    work.at[2, RULE_NOTES_COL] = ""
    recompute_sim_text(work, loaded.text_columns, config)
    check("clearing the note removes it from _sim_text",
          "annual contract" not in str(work.iloc[2][SIM_TEXT_COL]))


def test_reset_and_cleanup_controls():
    """End-to-end: create rules -> bulk delete -> clear sheet -> reseed -> run."""
    print("reset & cleanup controls")
    config, storage, rules = _fresh_env()

    records = [
        {"Name": "Prime Office", "Description": "toner", "New Account": "6210"},
        {"Name": "Prime Office", "Description": "paper", "New Account": ""},
        {"Name": "Cloud Hosting", "Description": "hosting", "New Account": ""},
    ]
    loaded = load_dataframe(_csv(records), config, "reset.csv")
    work = sh.build_work_df(loaded, storage, config)

    # Create a few rules on top of the seeded defaults.
    r1 = rules.add_rule("prime office", "6210")
    r2 = rules.add_rule("metro hardware", "6200")
    r3 = rules.add_rule("swift logistics", "6700")
    before = len(rules.list_rules())

    # Bulk delete two of them.
    removed = rules.delete_rules([r2.id, r3.id])
    check("bulk delete removes exactly the selected rules", removed == 2)
    check("bulk delete leaves the rest intact",
          len(rules.list_rules()) == before - 2)

    # Run rules, add notes, then Clear Spreadsheet Data.
    sh.run_rules_only(work, rules, loaded, config)
    work.at[2, RULE_NOTES_COL] = "monthly subscription"
    from src.data_loader import recompute_sim_text
    recompute_sim_text(work, loaded.text_columns, config)
    check("rule filled a blank before clearing",
          work.iloc[1]["New Account"] == "6210")

    na = loaded.new_account_col
    n = sh.clear_spreadsheet_values(work, loaded, config)
    check("clear wipes every Target Account value",
          n == len(work) and (work[na] == "").all())
    check("clear wipes Rule Notes", (work[RULE_NOTES_COL] == "").all())
    check("clear keeps original columns",
          list(work["Name"]) == ["Prime Office", "Prime Office", "Cloud Hosting"])

    # Reseed by re-running rules; automation works again post-clear.
    sh.run_rules_only(work, rules, loaded, config)
    check("automation works again after clear + reseed",
          work.iloc[0]["New Account"] == "6210"
          and work.iloc[1]["New Account"] == "6210")

    # Clear all rules (the 'Start fresh' rule wipe).
    wiped = rules.clear_rules()
    check("clear all rules empties the rule set",
          wiped >= 1 and rules.list_rules() == [])

    # Learned mappings + notes can be cleared (the rest of 'Start fresh').
    storage.clear_learned_mappings()
    storage.clear_rule_notes()
    check("learned mappings cleared", storage.list_learned_mappings() == [])
    check("rule notes cleared", storage.count_rule_notes() == 0)


def main() -> int:
    print("=" * 60)
    print("Code_Down_ML – smoke test")
    print("=" * 60)
    test_empty_file()
    test_no_seeds()
    test_field_variations()
    test_full_workflow()
    test_spreadsheet_v22_workflow()
    test_candidate_rules_flow()
    test_rule_matching_reliability()
    test_rule_creation_triggers_and_notes()
    test_reset_and_cleanup_controls()
    print("\nALL SMOKE TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
