"""Production-hardening tests — the locks for the sprint's new behaviour.

Covers:
  * keyword-suggestion source order (Name -> Memo -> Description, never Notes)
  * honest rule previews (blank fills vs protected matches)
  * token-aware fuzzy matching (no address false positives)
  * seed disagreement -> review, never silent auto-fill
  * review approvals write memory + training data; rejects never learn
  * bulk group approve / recode never overwrite existing codes
  * per-client isolation of rules, mappings and training data
  * the deterministic learned-memory run mode
  * regression locks: seeds sacred in every mode, strict never touches
    similarity/ML, reset keeps manual codes, export strips internal columns
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from models.schemas import FillAction
from src import spreadsheet_helpers as sh
from src.data_loader import NEW_ACCOUNT_COL, SIM_TEXT_COL
from src.fill_down_engine import FillDownEngine
from src.review_queue import apply_reviews


def _engine(config, rules, storage, **kwargs):
    return FillDownEngine(config, rules,
                          learned_lookup=storage.get_learned_lookup(),
                          **kwargs)


# --------------------------------------------------------------------------- #
# Mapping: keyword source order + honest previews + fuzzy + disagreement
# --------------------------------------------------------------------------- #
def test_keyword_source_order_name_then_memo_never_notes(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Description": "monthly service", "Name": "Cunningham Communications",
         "Memo": "call center", "Notes": "zzz coachnote qqq",
         "New Account": "6322"},
        {"Description": "monthly service", "Name": "Cunningham Communications",
         "Memo": "call center", "Notes": "zzz coachnote qqq",
         "New Account": "6322"},
        {"Description": "monthly service", "Name": "Cunningham Communications",
         "Memo": "call center", "Notes": "zzz coachnote qqq",
         "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)

    # Mining columns follow the configured order and exclude Notes entirely.
    cols = sh.mining_columns(work, loaded, config)
    assert "Notes" not in cols
    assert cols.index("Name") < cols.index("Memo") < cols.index("Description")

    # Suggestions come from Name/Memo/Description — never from Notes.
    cands = sh.candidate_rules(work, loaded, rules, config=config)
    assert any(c["account_code"] == "6322" for c in cands)
    for c in cands:
        assert "coachnote" not in c["keyword"]
        assert "zzz" not in c["keyword"]
        assert "qqq" not in c["keyword"]


def test_ingest_prefers_name_then_memo(make_loaded, rules, config):
    """Ingested exact rules key on Name first, then Memo — never Notes."""
    loaded = make_loaded([
        {"Name": "Cunningham Communications", "Memo": "call center",
         "Notes": "coach note", "New Account": "6322"},
        {"Name": "", "Memo": "Answering Service LLC",
         "Notes": "another note", "New Account": "6322"},
    ])
    created = rules.ingest_existing_new_account_as_rules(
        loaded.df,
        source_text_cols=sh.mining_columns(loaded.df, loaded, config))
    assert created == 2
    keywords = {r.keyword for r in rules.list_rules()}
    assert "Cunningham Communications" in keywords      # from Name
    assert "Answering Service LLC" in keywords          # from Memo (Name blank)
    assert all("note" not in k.lower() for k in keywords)


def test_rule_preview_counts_blanks_only_for_fill_but_reports_protected_hits(
        make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "monthly", "New Account": ""},
        {"Name": "Other", "Description": "misc", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    detail = sh.rule_preview_detail(work, "cred hub", "contains", False, [],
                                    loaded, config)
    assert detail["total"] == 2
    assert detail["blank_matches"] == 1       # only this row would be filled
    assert detail["protected_matches"] == 1   # matched but never overwritten
    assert len(detail["samples"]) == 2
    # The legacy two-tuple preview still works.
    count, sample = sh.rule_preview(work, "cred hub", "contains", False, [],
                                    loaded, config)
    assert count == 2 and len(sample) == 2


def test_fuzzy_does_not_match_token_inside_address(rules):
    # The classic false positive: 'shaw' inside an address token.
    rules.add_rule("shaw media", "6322", match_type="fuzzy")
    assert rules.match_row("123 shawnee dr apt 5", row=None) is None
    assert rules.match_row("payment 410 shaw street", row=None) is None
    # ...while genuine misspellings of the vendor still hit.
    assert rules.match_row("shaw medai invoice", row=None) is not None
    assert rules.match_row("shaw media llc", row=None) is not None

    # Single-token fuzzy is token-aware too: 'shaw' != 'shawnee'.
    rules.add_rule("shaw", "6322", match_type="fuzzy")
    assert rules.match_row("123 shawnee dr", row=None) is None
    assert rules.match_row("shaw communications", row=None) is not None


def test_near_miss_suggestions_surface_after_strict_run(
        make_loaded, rules, storage, config):
    from src.rules_manager import near_miss_suggestions

    loaded = make_loaded([
        {"Name": "Cunningham Communications", "New Account": ""},
        {"Name": "Cunningham Comm", "New Account": ""},
        {"Name": "Totally Unrelated", "New Account": ""},
    ])
    rules.create_rule("Cunningham Communications", "6322",
                      match_type="exact", fields=["Name"])
    engine = _engine(config, rules, storage)
    _, results = engine.run_selected_rules(loaded.df,
                                           text_columns=loaded.text_columns)
    near = near_miss_suggestions(results, loaded.df, rules.list_rules(),
                                 list(loaded.text_columns))
    # The "Cunningham Comm" row nearly matched the exact rule.
    assert near
    top = near[0]
    assert top["rule_keyword"] == "Cunningham Communications"
    assert top["account_code"] == "6322"
    assert top["rows"]                        # row indices listed
    # Nothing was auto-filled by the suggestion path.
    assert all(r.status != "keyword_rule" or r.row_index == 0
               for r in results)


def test_disagreement_among_seeds_goes_to_review_not_autofill(
        make_loaded, rules, storage, config):
    loaded = make_loaded([
        {"Name": "Acme Landscaping", "Description": "weekly mow service",
         "New Account": "6300"},
        {"Name": "Acme Landscaping", "Description": "weekly edge service",
         "New Account": "6310"},
        {"Name": "Acme Landscaping", "Description": "weekly trim service",
         "New Account": "6310"},
        # Blank row: nearest text is the 6300 seed, but the majority is 6310.
        {"Name": "Acme Landscaping", "Description": "weekly mow service",
         "New Account": ""},
    ])
    res = _engine(config, rules, storage).run(loaded)
    r = res.results[3]
    # Never silently auto-filled when the nearest seeds disagree.
    assert r.action in (FillAction.NEEDS_REVIEW, FillAction.FILLED_REVIEW)
    assert "disagree" in r.rationale.lower() or "majority" in r.rationale.lower()
    assert "6300" in r.rationale and "6310" in r.rationale
    if r.action == FillAction.NEEDS_REVIEW:
        assert res.df.iloc[3][loaded.new_account_col] == ""


# --------------------------------------------------------------------------- #
# Review: approvals learn, rejects don't, bulk actions never overwrite
# --------------------------------------------------------------------------- #
def test_apply_reviews_writes_learned_mapping_and_training_example(
        make_loaded, rules, storage, config):
    loaded = make_loaded([
        {"Name": "New Vendor", "Description": "thing", "New Account": ""},
    ])
    res = _engine(config, rules, storage).run(loaded)
    table = pd.DataFrame([{"row": 0, "New Account": "7000S", "Approve": True,
                           "Confidence": 0.9, "Engine": "similarity"}])
    counts = apply_reviews(res.df, table, loaded.new_account_col, storage)
    assert counts["applied"] == 1
    assert counts["learned"] == 1
    assert counts["trained"] == 1

    sig = str(res.df.iloc[0][SIM_TEXT_COL]).strip()
    assert storage.get_learned_lookup()[sig] == "7000S"
    assert "7000S" in storage.distinct_labels()
    # Account knowledge was reinforced too.
    profile = rules.get_account_profile("7000S")
    assert profile and int(profile["usage_count"]) >= 1


def test_reject_does_not_write_learned_mapping(make_loaded, storage, rules,
                                               config):
    loaded = make_loaded([
        {"Name": "Odd Vendor", "Description": "misc", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, sh.SUGGESTED_COL] = "6618S"
    work.at[0, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[0, sh.ENGINE_COL] = "similarity"

    out = sh.reject_rows(work, loaded, storage, [0])
    assert out["rejected"] == 1
    assert out["blocked"] == 1
    # No learned mapping, no training example for the rejected pairing.
    assert storage.list_learned_mappings() == []
    assert storage.count_training_data() == 0
    assert work.at[0, NEW_ACCOUNT_COL] == ""

    # The blocked pairing must not fire on a future run of the same text.
    sig = str(work.at[0, SIM_TEXT_COL]).strip()
    loaded2 = make_loaded([
        {"Name": "Odd Vendor", "Description": "misc", "New Account": ""},
    ])
    engine = FillDownEngine(
        config, rules, learned_lookup={sig: "6618S"},
        blocked_lookup=storage.get_blocked_lookup())
    res = engine.run(loaded2)
    assert res.df.iloc[0][loaded2.new_account_col] == ""
    assert res.results[0].action == FillAction.NEEDS_REVIEW


def test_bulk_approve_group_only_fills_blank_rows(make_loaded, storage, rules,
                                                  config):
    loaded = make_loaded([
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
        {"Name": "Acme Landscaping", "Description": "weekly", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Simulate a reviewed similarity group #5 with a consensus suggestion.
    for i in range(3):
        work.at[i, sh.GROUP_COL] = 5
        work.at[i, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
        work.at[i, sh.SUGGESTED_COL] = "6300S"
        work.at[i, sh.CONF_COL] = 0.8
    # Row 2 is already coded by the user — sacred.
    work.at[2, NEW_ACCOUNT_COL] = "1111"
    work.at[2, sh.ENGINE_COL] = "manual"

    out = sh.approve_similarity_group(work, loaded, storage, config, 5)
    assert out["approved"] == 2                 # only the two blank rows
    assert out["learned"] == 2
    assert work.at[0, NEW_ACCOUNT_COL] == "6300S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6300S"
    assert work.at[2, NEW_ACCOUNT_COL] == "1111"   # never overwritten

    # A split group is never one-click approved.
    work2 = sh.build_work_df(loaded, storage, config)
    for i, code in enumerate(("6300S", "6699")):
        work2.at[i, sh.GROUP_COL] = 9
        work2.at[i, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
        work2.at[i, sh.SUGGESTED_COL] = code
    out2 = sh.approve_similarity_group(work2, loaded, storage, config, 9)
    assert out2["approved"] == 0
    assert out2["split"] == {"6300S": 1, "6699": 1}
    assert (work2[NEW_ACCOUNT_COL] == "").all()


def test_never_overwrite_on_bulk_recode(make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "A", "Description": "x", "New Account": "1111"},  # upload seed
        {"Name": "B", "Description": "y", "New Account": ""},      # blank
        {"Name": "C", "Description": "z", "New Account": ""},      # engine fill
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[2, NEW_ACCOUNT_COL] = "2222"
    work.at[2, sh.ENGINE_COL] = "similarity"

    out = sh.recode_rows(work, loaded, storage, [0, 1, 2], "5555")
    assert work.at[0, NEW_ACCOUNT_COL] == "1111"   # seed protected
    assert work.at[2, NEW_ACCOUNT_COL] == "2222"   # existing code protected
    assert work.at[1, NEW_ACCOUNT_COL] == "5555"   # blank coded
    assert out["skipped_protected"] == 2

    # Explicit override may re-code automation fills — never seeds/manual.
    out2 = sh.recode_rows(work, loaded, storage, [0, 2], "6666",
                          overwrite_engine=True)
    assert work.at[2, NEW_ACCOUNT_COL] == "6666"
    assert work.at[0, NEW_ACCOUNT_COL] == "1111"
    assert out2["recoded"] == 1


def test_review_workspace_table_sorted_and_apply_learns(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "V1", "Description": "a", "New Account": ""},
        {"Name": "V2", "Description": "b", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    work.at[0, sh.ACTION_COL] = FillAction.NEEDS_REVIEW.value
    work.at[0, sh.SUGGESTED_COL] = "6300S"
    work.at[0, sh.CONF_COL] = 0.9
    work.at[1, sh.ACTION_COL] = FillAction.FILLED_REVIEW.value
    work.at[1, NEW_ACCOUNT_COL] = "6400S"
    work.at[1, sh.SUGGESTED_COL] = "6400S"
    work.at[1, sh.CONF_COL] = 0.6

    table = sh.review_rows_df(work, loaded)
    assert len(table) == 2
    # Least confident first.
    assert list(table["Confidence"]) == [0.6, 0.9]
    # FILLED_REVIEW rows default to Approve=True; NEEDS_REVIEW to False.
    assert bool(table.iloc[0]["Approve"]) is True
    assert bool(table.iloc[1]["Approve"]) is False

    out = sh.apply_review_table(work, loaded, storage, table)
    assert out["applied"] == 0          # 6400S already written by the engine
    assert out["learned"] == 1          # but the approval was learned
    assert storage.count_training_data() == 1


# --------------------------------------------------------------------------- #
# Memory: manual edits, recode last-write-wins, client isolation, memory mode
# --------------------------------------------------------------------------- #
def test_manual_grid_edit_upserts_mapping(make_loaded, storage, config):
    loaded = make_loaded([
        {"Name": "Acme Co", "Description": "service", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    edited = work.loc[[0]].copy()
    edited.at[0, NEW_ACCOUNT_COL] = "6300 s"
    sh.commit_editor_changes(work, edited, loaded, storage, config)
    sig = str(work.at[0, SIM_TEXT_COL]).strip()
    assert storage.get_learned_lookup()[sig] == "6300S"
    assert "6300S" in storage.distinct_labels()


def test_recode_overwrites_previous_mapping(make_loaded, storage, config):
    """Last human write wins: recoding a learned row replaces the mapping."""
    loaded = make_loaded([
        {"Name": "Acme Co", "Description": "service", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    sig = str(work.at[0, SIM_TEXT_COL]).strip()

    edited = work.loc[[0]].copy()
    edited.at[0, NEW_ACCOUNT_COL] = "6300S"
    sh.commit_editor_changes(work, edited, loaded, storage, config)
    assert storage.get_learned_lookup()[sig] == "6300S"

    edited2 = work.loc[[0]].copy()
    edited2.at[0, NEW_ACCOUNT_COL] = "6400S"
    sh.commit_editor_changes(work, edited2, loaded, storage, config)
    assert storage.get_learned_lookup()[sig] == "6400S"


def test_client_isolation_rules_and_mappings(rules, storage):
    rules.add_rule("acme", "6300S", client_id="client_a")
    rules.add_rule("globex", "6400S", client_id="client_b")
    rules.add_rule("shared", "1000")  # global rule
    assert {r.keyword for r in rules.list_rules(client_id="client_a")} == \
        {"acme", "shared"}
    assert {r.keyword for r in rules.list_rules(client_id="client_b")} == \
        {"globex", "shared"}

    storage.upsert_learned_mapping("sig a", "6300S", client_id="client_a")
    storage.upsert_learned_mapping("sig b", "6400S", client_id="client_b")
    storage.upsert_learned_mapping("sig shared", "1000")

    lookup_a = storage.get_learned_lookup(client_id="client_a")
    assert lookup_a == {"sig a": "6300S", "sig shared": "1000"}
    lookup_b = storage.get_learned_lookup(client_id="client_b")
    assert lookup_b == {"sig b": "6400S", "sig shared": "1000"}

    # The client's own mapping wins over the shared one on conflict.
    storage.upsert_learned_mapping("sig shared", "2000", client_id="client_a")
    assert storage.get_learned_lookup(client_id="client_a")["sig shared"] \
        == "2000"
    assert storage.get_learned_lookup(client_id="client_b")["sig shared"] \
        == "1000"

    # Training data is scoped the same way.
    storage.add_training_example("text a", "6300S", client_id="client_a")
    assert storage.count_training_data(client_id="client_a") == 1
    assert storage.count_training_data(client_id="client_b") == 0
    assert storage.count_training_data() == 1
    assert storage.distinct_labels(client_id="client_b") == []


def test_existing_db_migrates_to_client_schema(tmp_path):
    """A pre-client_id database keeps every mapping as the default client."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE learned_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT NOT NULL UNIQUE,
            account_code TEXT NOT NULL,
            hits INTEGER NOT NULL DEFAULT 1,
            last_seen TEXT NOT NULL
        );
        INSERT INTO learned_mappings (signature, account_code, hits, last_seen)
            VALUES ('old sig', '6300S', 3, '2026-01-01T00:00:00+00:00');
        CREATE TABLE training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, label TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            engine_used TEXT NOT NULL DEFAULT 'manual',
            timestamp TEXT NOT NULL,
            approved_by TEXT NOT NULL DEFAULT 'user',
            UNIQUE(text, label)
        );
        INSERT INTO training_data (text, label, timestamp)
            VALUES ('old text', '6300S', '2026-01-01T00:00:00+00:00');
        """)
    conn.close()

    from utils.storage import Storage
    storage = Storage(db)  # triggers the migration
    assert storage.get_learned_lookup()["old sig"] == "6300S"
    assert storage.get_learned_lookup(client_id="anyone")["old sig"] == "6300S"
    assert storage.count_training_data() == 1
    # New client-scoped writes coexist with the migrated rows.
    storage.upsert_learned_mapping("old sig", "6400S", client_id="client_a")
    assert storage.get_learned_lookup(client_id="client_a")["old sig"] == "6400S"
    assert storage.get_learned_lookup(client_id="client_b")["old sig"] == "6300S"
    storage.close()


def test_learned_memory_only_mode_fills_exact_signatures(
        make_loaded, storage, rules, config):
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
        {"Name": "Cred Hub", "Description": "monthly fee", "New Account": ""},
    ])
    work = sh.build_work_df(loaded, storage, config)
    # Learn the exact signature of row 1 (same text as the seed row).
    storage.upsert_learned_mapping(str(work.at[1, SIM_TEXT_COL]), "6618S")

    n = sh.run_learned_memory(work, loaded, storage, config)
    assert n == 1
    assert work.at[1, NEW_ACCOUNT_COL] == "6618S"     # exact match filled
    assert work.at[2, NEW_ACCOUNT_COL] == ""          # different text: no
    assert work.at[0, NEW_ACCOUNT_COL] == "6618S"     # seed preserved
    assert work.at[1, sh.ENGINE_COL] == "learned"

    # Rules + memory: each layer fills only its own rows.
    work2 = sh.build_work_df(loaded, storage, config)
    storage.upsert_learned_mapping(str(work2.at[1, SIM_TEXT_COL]), "6618S")
    rules.create_rule("monthly fee", "6200")
    total, audit = sh.run_rules_plus_memory(work2, rules, loaded, config)
    assert total == 2
    assert audit["keyword_filled"] == 1
    assert audit["memory_filled"] == 1
    assert work2.at[1, NEW_ACCOUNT_COL] == "6618S"
    assert work2.at[2, NEW_ACCOUNT_COL] == "6200"


# --------------------------------------------------------------------------- #
# Regression locks
# --------------------------------------------------------------------------- #
def _mixed_records():
    return [
        {"Name": "Cred Hub", "Description": "screening", "New Account": "6618S"},
        {"Name": "Staples", "Description": "paper", "New Account": "6200"},
        {"Name": "Cred Hub", "Description": "monthly screening",
         "New Account": ""},
        {"Name": "Staples", "Description": "office paper", "New Account": ""},
        {"Name": "Mystery", "Description": "zzz qqq", "New Account": ""},
    ]


def test_existing_values_never_change_across_all_three_run_modes(
        make_loaded, storage, rules, config):
    # A trap rule that would love to clobber the Staples seed.
    rules.create_rule("Cred Hub", "6618S")
    rules.create_rule("Staples", "9999")

    # 1) Strict rules.
    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)
    sh.run_selected_rules_audited(work, rules, loaded, config)
    assert work.at[0, NEW_ACCOUNT_COL] == "6618S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6200"

    # 2) Rules + memory.
    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)
    sh.run_rules_plus_memory(work, rules, loaded, config)
    assert work.at[0, NEW_ACCOUNT_COL] == "6618S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6200"

    # 3) Rules + similarity (hybrid).
    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)
    engine = FillDownEngine(config, rules,
                            learned_lookup=storage.get_learned_lookup(),
                            model_manager=None, mode="similarity_only")
    sh.run_rules_hybrid(work, rules, loaded, config, engine)
    assert work.at[0, NEW_ACCOUNT_COL] == "6618S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6200"

    # 4) Full intelligent run.
    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)
    sh.run_full(work, loaded, config, _engine(config, rules, storage))
    assert work.at[0, NEW_ACCOUNT_COL] == "6618S"
    assert work.at[1, NEW_ACCOUNT_COL] == "6200"


def test_strict_mode_never_calls_similarity_or_ml(
        make_loaded, storage, rules, config, monkeypatch):
    import src.similarity as similarity_mod

    def _boom(*args, **kwargs):
        raise AssertionError("strict mode touched the similarity/ML layer")

    monkeypatch.setattr(similarity_mod, "group_transactions", _boom)
    monkeypatch.setattr(similarity_mod.Embedder, "encode", _boom)
    monkeypatch.setattr(FillDownEngine, "run", _boom)

    rules.create_rule("Cred Hub", "6618S")
    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)

    # The UI strict path: fills only the blank Cred Hub row, never touches
    # the similarity/ML layer (patched to explode on contact).
    n, results = sh.run_selected_rules_audited(work, rules, loaded, config)
    assert n == 1
    assert work.at[2, NEW_ACCOUNT_COL] == "6618S"
    assert all(r.status in ("protected_existing", "keyword_rule",
                            "no_rule_match") for r in results)

    # The engine strict path.
    loaded2 = make_loaded(_mixed_records())
    engine = FillDownEngine(config, rules)
    df, _ = engine.run_selected_rules(loaded2.df,
                                      text_columns=loaded2.text_columns)
    assert df[NEW_ACCOUNT_COL].tolist()[2] == "6618S"


def test_reset_run_restores_last_automation_without_dropping_manual_codes(
        make_loaded, storage, rules, config):
    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)
    rules.create_rule("Cred Hub", "6618S")

    sh.run_selected_rules_audited(work, rules, loaded, config)
    assert work.at[2, NEW_ACCOUNT_COL] == "6618S"

    # A manual code typed after the run.
    edited = work.loc[[4]].copy()
    edited.at[4, NEW_ACCOUNT_COL] = "5000"
    sh.commit_editor_changes(work, edited, loaded, storage, config)

    cleared = sh.reset_automation(work, loaded, config)
    assert cleared >= 1
    assert work.at[0, NEW_ACCOUNT_COL] == "6618S"   # upload seed kept
    assert work.at[1, NEW_ACCOUNT_COL] == "6200"    # upload seed kept
    assert work.at[2, NEW_ACCOUNT_COL] == ""        # automation fill undone
    assert work.at[4, NEW_ACCOUNT_COL] == "5000"    # manual code kept
    assert len(rules.list_rules()) == 1             # rules kept


def test_export_still_strips_internal_columns_from_csv(
        make_loaded, storage, rules, config):
    from src.exporter import export_csv

    loaded = make_loaded(_mixed_records())
    work = sh.build_work_df(loaded, storage, config)
    rules.create_rule("Cred Hub", "6618S")
    sh.run_rules_plus_memory(work, rules, loaded, config)

    raw = export_csv(work, loaded.new_account_col)
    out = pd.read_csv(io.BytesIO(raw))
    assert not any(str(c).startswith("_") for c in out.columns)
    for marker in (b"_sim_text", b"_base_sig", b"_confidence", b"_group"):
        assert marker not in raw
    assert f"{loaded.new_account_col} (Base)" in out.columns
