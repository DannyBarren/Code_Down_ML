"""Per-client model store with global fallback (v1.1).

Locks the remaining isolation boundary: trained models. A client-scoped
ModelManager trains on that client's examples (plus the shared pool) and writes
under ``data/models/clients/<safe_id>/``; prediction tries the client model
first, then the global model, then similarity — and never raises.
"""

from __future__ import annotations

import pytest

from models.schemas import FillAction
from src.fill_down_engine import FillDownEngine
from src.ml_classifier import ModelManager, safe_client_id


def _seed_client(storage, client_id, per_class=8):
    """Two clearly separable classes of examples for one client."""
    for i in range(per_class):
        storage.add_training_example(
            f"cred hub resident screening fee variant {i}", "6618S",
            client_id=client_id)
        storage.add_training_example(
            f"appfolio monthly software license variant {i}", "6520S",
            client_id=client_id)


def _client_mm(config, storage, client_id):
    return ModelManager(config, storage, client_id=client_id)


# --------------------------------------------------------------------------- #
# Scoping + storage isolation
# --------------------------------------------------------------------------- #
def test_safe_client_id_slugifies():
    assert safe_client_id("Acme Co") == "acme_co"
    assert safe_client_id("Weird/Client: Name!") == "weird_client_name"
    assert safe_client_id("") == "client"
    assert safe_client_id(None) == "client"
    assert "/" not in safe_client_id("a/b/c") and "\\" not in safe_client_id("a\\b")


def test_client_training_data_isolation(storage):
    _seed_client(storage, "client_a")
    storage.add_training_example("shared pool example one", "1000")
    storage.add_training_example("shared pool example two", "2000")

    # Client A sees its own + the shared pool; client B sees only shared.
    assert storage.count_training_data(client_id="client_a") == 18
    assert storage.count_training_data(client_id="client_b") == 2
    texts_b, _ = storage.get_training_xy(client_id="client_b")
    assert all("cred hub" not in t and "appfolio" not in t for t in texts_b)
    # The global (unscoped) view still trains on everything.
    assert storage.count_training_data() == 18


# --------------------------------------------------------------------------- #
# Artifact placement
# --------------------------------------------------------------------------- #
def test_client_artifacts_land_under_clients_dir(config, storage):
    _seed_client(storage, "Acme Co")
    mm_a = _client_mm(config, storage, "Acme Co")
    results = mm_a.train_all(model_types=["logreg"])
    assert "error" not in results["logreg"]

    store = config.abs_model_store_dir()
    assert mm_a.dir == store / "clients" / "acme_co"
    assert (mm_a.dir / "registry.json").exists()
    assert (mm_a.dir / "logreg" / "v1.joblib").exists()
    # The global store root is untouched by the client train.
    assert not (store / "registry.json").exists()


def test_global_train_writes_store_root(config, storage, seed_training):
    seed_training(8)
    mm = ModelManager(config, storage)  # global (no client)
    results = mm.train_all(model_types=["logreg"])
    assert "error" not in results["logreg"]
    store = config.abs_model_store_dir()
    assert mm.dir == store
    assert (store / "registry.json").exists()
    assert (store / "logreg" / "v1.joblib").exists()


def test_client_b_does_not_load_client_a_model(config, storage):
    _seed_client(storage, "client_a")
    _client_mm(config, storage, "client_a").train_all(model_types=["logreg"])

    mm_b = _client_mm(config, storage, "client_b")
    assert mm_b.has_model() is False
    preds = mm_b.predict(["cred hub screening"])
    assert preds and preds[0].label is None      # empty prediction, no raise


# --------------------------------------------------------------------------- #
# Predict order: client -> global -> similarity
# --------------------------------------------------------------------------- #
def test_client_b_falls_back_to_global_model(config, rules, storage,
                                             seed_training, make_loaded):
    seed_training(8)  # shared pool
    global_mm = ModelManager(config, storage)
    global_mm.train_all(model_types=["logreg"])
    config.ml.ml_confidence_cutoff = 0.5  # deterministic for the test

    # Client B has no trained model; the global model must serve the row.
    mm_b = _client_mm(config, storage, "client_b")
    assert not mm_b.has_model()
    engine = FillDownEngine(config, rules, model_manager=mm_b,
                            fallback_model_manager=global_mm,
                            mode="prefer_ml")
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "screening", "New Account": ""},
    ])
    res = engine.run(loaded)
    assert res.df.iloc[0][loaded.new_account_col] == "6618S"
    assert res.results[0].engine_used.startswith("ml")
    assert res.results[0].action in (FillAction.AUTO_FILLED,
                                     FillAction.FILLED_REVIEW)


def test_client_model_wins_over_global_when_confident(config, rules, storage,
                                                      seed_training,
                                                      make_loaded):
    seed_training(8)                                   # shared pool -> global
    global_mm = ModelManager(config, storage)
    global_mm.train_all(model_types=["logreg"])

    # Client A's own model, trained on its own relabelled examples: the same
    # vendor text maps to a *different* code for this client.
    for i in range(8):
        storage.add_training_example(
            f"cred hub resident screening fee variant {i}", "7000S",
            client_id="client_a")
        storage.add_training_example(
            f"appfolio monthly software license variant {i}", "7100S",
            client_id="client_a")
    mm_a = _client_mm(config, storage, "client_a")
    mm_a.train_all(model_types=["logreg"])
    config.ml.ml_confidence_cutoff = 0.5

    engine = FillDownEngine(config, rules, model_manager=mm_a,
                            fallback_model_manager=global_mm,
                            mode="prefer_ml")
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "screening", "New Account": ""},
    ])
    res = engine.run(loaded)
    # The client model's answer (7000S) wins over the global model's (6618S).
    assert res.df.iloc[0][loaded.new_account_col] == "7000S"


def test_no_model_anywhere_falls_back_to_similarity(config, rules, storage,
                                                    make_loaded):
    mm_b = _client_mm(config, storage, "client_b")     # nothing trained
    global_mm = ModelManager(config, storage)          # nothing trained
    engine = FillDownEngine(config, rules, model_manager=mm_b,
                            fallback_model_manager=global_mm, mode="prefer_ml")
    loaded = make_loaded([
        {"Name": "Cred Hub", "Description": "fee", "New Account": "6618S"},
        {"Name": "Cred Hub", "Description": "fee", "New Account": ""},
    ])
    res = engine.run(loaded)
    assert res.mode == "similarity_only"
    assert res.df.iloc[1][loaded.new_account_col] == "6618S"   # similarity fill


# --------------------------------------------------------------------------- #
# Demo reset clears client model stores too
# --------------------------------------------------------------------------- #
def test_demo_reset_clears_client_model_stores(config, storage):
    _seed_client(storage, "client_a")
    mm_a = _client_mm(config, storage, "client_a")
    mm_a.train_all(model_types=["logreg"])
    clients_dir = config.abs_model_store_dir() / "clients"
    assert (clients_dir / "client_a" / "registry.json").exists()

    from utils import demo_utils
    demo_utils.reset_for_demo(config, storage, mm_a)
    # Everything under the model store — including clients/ — is gone.
    assert list(config.abs_model_store_dir().iterdir()) == []
