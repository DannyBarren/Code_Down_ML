"""Fresh Demo Mode: data reset + demo/HF detection."""

from __future__ import annotations

from src import config as cfg_mod
from utils import demo_utils


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_is_demo_false_by_default(monkeypatch):
    for var in ("FILLDOWN_DEMO", "SPACE_ID", "HF_SPACE_ID", "FILLDOWN_DEMO_RESET"):
        monkeypatch.delenv(var, raising=False)
    assert cfg_mod.is_demo() is False
    assert cfg_mod.demo_reset_enabled() is False


def test_is_demo_via_flag(monkeypatch):
    monkeypatch.setenv("FILLDOWN_DEMO", "1")
    assert cfg_mod.is_demo() is True
    assert cfg_mod.demo_reset_enabled() is True


def test_is_demo_via_hf_space(monkeypatch):
    monkeypatch.delenv("FILLDOWN_DEMO", raising=False)
    monkeypatch.setenv("SPACE_ID", "acme/fill-down-demo")
    assert cfg_mod.is_hf_space() is True
    assert cfg_mod.is_demo() is True


def test_demo_reset_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FILLDOWN_DEMO", "1")
    monkeypatch.setenv("FILLDOWN_DEMO_RESET", "0")
    assert cfg_mod.is_demo() is True
    assert cfg_mod.demo_reset_enabled() is False


# --------------------------------------------------------------------------- #
# Storage reset
# --------------------------------------------------------------------------- #
def test_reset_all_empties_every_table_and_resets_ids(storage, rules):
    rules.add_rule("cred hub", "6618S")
    storage.upsert_learned_mapping("sig-1", "6618S")
    storage.add_training_example("cred hub fee", "6618S")
    assert rules.list_rules()
    assert storage.list_learned_mappings()
    assert storage.count_training_data() == 1

    storage.reset_all()

    assert rules.list_rules() == []
    assert storage.list_learned_mappings() == []
    assert storage.count_training_data() == 0
    assert storage.list_runs() == []

    # Schema intact + id counter reset to 1 (looks brand new).
    new_rule = rules.add_rule("appfolio", "6520S")
    assert new_rule.id == 1


# --------------------------------------------------------------------------- #
# reset_for_demo (DB + model store + runtime files)
# --------------------------------------------------------------------------- #
def test_reset_for_demo_clears_models_and_db(config, storage, rules, tmp_path):
    config.data_dir = str(tmp_path)   # isolate work dir + log from the real project
    rules.add_rule("cred hub", "6618S")
    model_dir = config.abs_model_store_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "registry.json").write_text('{"active": "logreg"}')
    (model_dir / "logreg").mkdir(exist_ok=True)
    (model_dir / "logreg" / "v1.joblib").write_text("blob")

    summary = demo_utils.reset_for_demo(config, storage)

    assert rules.list_rules() == []
    assert summary["models_removed"] >= 1
    assert list(model_dir.iterdir()) == []   # model store emptied, dir preserved
    assert model_dir.exists()


def test_maybe_auto_reset_only_runs_in_demo(config, storage, rules, monkeypatch,
                                            tmp_path):
    config.data_dir = str(tmp_path)   # isolate work dir + log from the real project
    demo_utils._RESET_DONE = False
    monkeypatch.delenv("FILLDOWN_DEMO", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("HF_SPACE_ID", raising=False)
    rules.add_rule("keep me", "6618S")

    # Off-demo: never wipes.
    assert demo_utils.maybe_auto_reset(config, storage) is False
    assert rules.list_rules()

    # In demo: wipes once, then is a no-op for the rest of the process.
    monkeypatch.setenv("FILLDOWN_DEMO", "1")
    demo_utils._RESET_DONE = False
    assert demo_utils.maybe_auto_reset(config, storage) is True
    assert rules.list_rules() == []
    assert demo_utils.maybe_auto_reset(config, storage) is False

    demo_utils._RESET_DONE = False  # reset module state for other tests
