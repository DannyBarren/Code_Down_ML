"""Tests for the trainable ML layer and its integration with the engine."""

import pandas as pd

from models.schemas import FillAction
from src.fill_down_engine import FillDownEngine
from src.ml_classifier import (
    InsufficientTrainingData,
    LogRegModel,
    ModelManager,
    setfit_available,
)


# --------------------------------------------------------------------------- #
# Availability / detection
# --------------------------------------------------------------------------- #

def test_setfit_available_returns_tuple():
    ok, detail = setfit_available()
    assert isinstance(ok, bool)
    assert isinstance(detail, str)


def test_logreg_always_available():
    assert LogRegModel.is_available() is True


# --------------------------------------------------------------------------- #
# LogReg model
# --------------------------------------------------------------------------- #

def _training_set(per_class=8):
    texts, labels = [], []
    for i in range(per_class):
        texts.append(f"cred hub resident screening fee {i}")
        labels.append("6618S")
        texts.append(f"appfolio monthly software license {i}")
        labels.append("6520S")
    return texts, labels


def test_logreg_train_and_predict():
    texts, labels = _training_set()
    model = LogRegModel()
    metrics = model.train(texts, labels, test_size=0.2, min_examples=6)
    assert metrics["n_labels"] == 2
    (label, conf), = model.predict(["cred hub screening charge"])
    assert label == "6618S"
    assert 0.0 < conf <= 1.0


def test_logreg_insufficient_data():
    model = LogRegModel()
    try:
        model.train(["only one"], ["A"], min_examples=6)
        assert False, "should have raised"
    except InsufficientTrainingData:
        pass


def test_logreg_save_load(tmp_path):
    texts, labels = _training_set()
    model = LogRegModel()
    model.train(texts, labels, min_examples=6)
    path = tmp_path / "logreg.joblib"
    model.save(path)
    loaded = LogRegModel.load(path)
    (label, _), = loaded.predict(["appfolio license renewal"])
    assert label == "6520S"


# --------------------------------------------------------------------------- #
# ModelManager
# --------------------------------------------------------------------------- #

def test_manager_can_train_guard(model_manager):
    ok, reason = model_manager.can_train()
    assert ok is False and reason  # no data yet


def test_manager_train_and_predict(model_manager, seed_training):
    seed_training(8)
    ok, _ = model_manager.can_train()
    assert ok
    results = model_manager.train_all()
    assert "logreg" in results and "error" not in results["logreg"]
    assert model_manager.has_model()
    assert model_manager.active_model_name() == "logreg"

    preds = model_manager.predict(["cred hub screening"])
    assert preds[0].label == "6618S"
    assert preds[0].model_name == "logreg"


def test_manager_versioning(model_manager, seed_training):
    seed_training(8)
    model_manager.train_all()
    v1 = model_manager._load_registry()["models"]["logreg"]["version"]
    model_manager.train_all()
    v2 = model_manager._load_registry()["models"]["logreg"]["version"]
    assert v2 == v1 + 1


def test_progressive_mode(model_manager, config, storage):
    # 0 examples -> assist
    assert model_manager.progressive_mode() == "assist"
    config.ml.hybrid_min = 2
    config.ml.ml_primary_min = 4
    storage.add_training_example("a", "X")
    storage.add_training_example("b", "Y")
    assert model_manager.progressive_mode() == "hybrid"
    storage.add_training_example("c", "X")
    storage.add_training_example("d", "Y")
    assert model_manager.progressive_mode() == "prefer_ml"


def test_status_list(model_manager, seed_training):
    seed_training(8)
    model_manager.train_all()
    statuses = {s.name: s for s in model_manager.status_list()}
    assert statuses["logreg"].trained is True
    assert statuses["logreg"].available is True
    assert "setfit" in statuses


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #

def test_engine_uses_ml_in_prefer_mode(config, rules, model_manager,
                                       seed_training, make_loaded):
    seed_training(8)
    model_manager.train_all()
    config.ml.ml_confidence_cutoff = 0.5  # deterministic for the test

    # A blank row with NO seeds -> similarity has nothing; ML should fill it.
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "screening", "New Account": ""},
    ])
    engine = FillDownEngine(config, rules, model_manager=model_manager,
                            mode="prefer_ml")
    res = engine.run(loaded)
    r0 = res.results[0]
    assert res.df.iloc[0][loaded.new_account_col] == "6618S"
    assert r0.engine_used.startswith("ml")
    assert r0.action in (FillAction.AUTO_FILLED, FillAction.FILLED_REVIEW)


def test_engine_similarity_only_ignores_ml(config, rules, model_manager,
                                           seed_training, make_loaded):
    seed_training(8)
    model_manager.train_all()
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "screening", "New Account": ""},
    ])
    engine = FillDownEngine(config, rules, model_manager=model_manager,
                            mode="similarity_only")
    res = engine.run(loaded)
    # No seeds + ML ignored -> nothing to fill.
    assert res.mode == "similarity_only"
    assert res.df.iloc[0][loaded.new_account_col] == ""
    assert res.results[0].engine_used in ("none", "similarity")


def test_engine_falls_back_when_no_model(config, rules, model_manager, make_loaded):
    # No training, no model -> engine must behave as pure similarity.
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    engine = FillDownEngine(config, rules, model_manager=model_manager,
                            mode="prefer_ml")
    res = engine.run(loaded)
    assert res.mode == "similarity_only"  # downgraded because no model
    assert res.df.iloc[1][loaded.new_account_col] == "6618S"


# --------------------------------------------------------------------------- #
# Training-data capture on approval
# --------------------------------------------------------------------------- #

def test_approval_captures_training_data(config, rules, storage, make_loaded):
    from src.review_queue import apply_reviews

    loaded = make_loaded([
        {"Name": "New Vendor", "Description": "thing", "New Account": ""},
    ])
    engine = FillDownEngine(config, rules)
    res = engine.run(loaded)
    table = pd.DataFrame([{"row": 0, "New Account": "7000S", "Approve": True,
                           "Confidence": 0.9, "Engine": "similarity"}])
    before = storage.count_training_data()
    counts = apply_reviews(res.df, table, loaded.new_account_col, storage)
    assert counts["trained"] >= 1
    assert storage.count_training_data() == before + 1
    assert "7000S" in storage.distinct_labels()
