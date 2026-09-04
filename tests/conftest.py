"""Shared pytest fixtures.

Tests force the deterministic TF-IDF backend so they run fast and don't depend
on the optional semantic model being installed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data_loader import load_dataframe  # noqa: E402
from src.ml_classifier import ModelManager  # noqa: E402
from src.rules_manager import RulesManager  # noqa: E402
from utils.storage import Storage  # noqa: E402


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.similarity.use_embeddings = False           # deterministic TF-IDF
    cfg.ml.model_store_dir = str(tmp_path / "models")  # isolate model artifacts
    cfg.ml.min_examples_to_train = 6                # smaller for fast tests
    return cfg


@pytest.fixture
def storage(tmp_path):
    db = tmp_path / "test.db"
    s = Storage(db)
    yield s
    s.close()


@pytest.fixture
def rules(storage):
    return RulesManager(storage)


@pytest.fixture
def model_manager(config, storage):
    return ModelManager(config, storage)


@pytest.fixture
def make_loaded(config):
    """Factory: build LoadedData from a list-of-dicts."""
    def _make(records, name="test.csv"):
        df = pd.DataFrame(records)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return load_dataframe(buf.getvalue(), config, source_name=name)
    return _make


@pytest.fixture
def seed_training(storage):
    """Populate the training table with clearly separable examples."""
    def _seed(per_class=8):
        for i in range(per_class):
            storage.add_training_example(
                f"cred hub resident screening fee variant {i}", "6618S")
            storage.add_training_example(
                f"appfolio monthly software license variant {i}", "6520S")
        return storage.count_training_data()
    return _seed
