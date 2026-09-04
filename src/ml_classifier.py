"""Trainable ML layer that sits *on top* of the similarity engine.

This module is deliberately self-contained and optional. The similarity engine
never depends on it; the fill-down engine only consults it when a trained model
exists and the chosen mode allows it. If anything here fails — missing optional
dependency, no training data, corrupt model file — callers fall back to pure
similarity and the app keeps working.

Two model types are supported:

* **LogReg** (``logreg``) – a self-contained scikit-learn
  ``TfidfVectorizer + LogisticRegression`` pipeline. Always available (no torch
  needed), fast to train, and a solid baseline.
* **SetFit** (``setfit``) – an optional, more powerful few-shot transformer
  model. Used only when the ``setfit`` package (and its torch stack) is
  installed.

Trained models are versioned on disk under ``config.ml.model_store_dir`` and
tracked in a small ``registry.json``.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force the PyTorch-only Hugging Face backend; a broken/incompatible TensorFlow
# in the environment must never break SetFit / sentence-transformers imports.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from src.config import Config
from utils.logging_setup import get_logger
from utils.storage import Storage

logger = get_logger(__name__)

SETFIT_PIP_HINT = "pip install setfit"


class InsufficientTrainingData(Exception):
    """Raised when there are too few labelled examples / classes to train."""


@dataclass
class Prediction:
    """A single ML prediction."""

    label: Optional[str]
    confidence: float
    model_name: str = ""


@functools.lru_cache(maxsize=1)
def setfit_available() -> Tuple[bool, str]:
    """Return ``(available, detail)`` for the optional SetFit backend."""
    try:
        import setfit  # noqa: F401
        import torch  # noqa: F401

        return True, "setfit is installed."
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Individual model wrappers
# --------------------------------------------------------------------------- #


class LogRegModel:
    """TF-IDF + Logistic Regression. Self-contained and always available."""

    name = "logreg"

    def __init__(self, pipeline=None, labels: Optional[List[str]] = None):
        self.pipeline = pipeline
        self.labels = labels or []

    @staticmethod
    def is_available() -> bool:
        return True

    def train(self, texts: List[str], labels: List[str],
              test_size: float = 0.2, min_examples: int = 10) -> Dict:
        from collections import Counter

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        classes = sorted(set(labels))
        n = len(texts)
        if n < min_examples or len(classes) < 2:
            raise InsufficientTrainingData(
                f"Need at least {min_examples} examples across 2+ account codes "
                f"(have {n} examples, {len(classes)} codes)."
            )

        def _build() -> Pipeline:
            return Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char_wb",
                                          ngram_range=(3, 5), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])

        # Held-out accuracy when every class has at least 2 samples.
        accuracy: Optional[float] = None
        counts = Counter(labels)
        if all(c >= 2 for c in counts.values()) and n >= 2 * len(classes):
            try:
                x_tr, x_te, y_tr, y_te = train_test_split(
                    texts, labels, test_size=test_size,
                    stratify=labels, random_state=42)
                evalpipe = _build()
                evalpipe.fit(x_tr, y_tr)
                accuracy = float(accuracy_score(y_te, evalpipe.predict(x_te)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("logreg_eval_failed", error=str(exc))

        # Final model trained on all data.
        self.pipeline = _build()
        self.pipeline.fit(texts, labels)
        self.labels = classes
        return {"accuracy": accuracy, "n_examples": n, "n_labels": len(classes)}

    def predict(self, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        if not self.pipeline or not texts:
            return [(None, 0.0) for _ in texts]
        proba = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        out: List[Tuple[Optional[str], float]] = []
        for row in proba:
            idx = int(row.argmax())
            out.append((str(classes[idx]), float(row[idx])))
        return out

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: Path) -> "LogRegModel":
        import joblib

        data = joblib.load(path)
        return cls(pipeline=data["pipeline"], labels=data.get("labels", []))


class SetFitModel:
    """Optional few-shot transformer classifier (wraps the ``setfit`` package)."""

    name = "setfit"

    def __init__(self, model=None, labels: Optional[List[str]] = None):
        self.model = model
        self.labels = labels or []

    @staticmethod
    def is_available() -> bool:
        return setfit_available()[0]

    def train(self, texts: List[str], labels: List[str], base_model: str,
              num_epochs: int = 1, batch_size: int = 16,
              test_size: float = 0.2, min_examples: int = 10) -> Dict:
        from collections import Counter

        from datasets import Dataset
        from setfit import SetFitModel as _SetFit
        from setfit import Trainer, TrainingArguments
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        classes = sorted(set(labels))
        n = len(texts)
        if n < min_examples or len(classes) < 2:
            raise InsufficientTrainingData(
                f"Need at least {min_examples} examples across 2+ account codes "
                f"(have {n} examples, {len(classes)} codes)."
            )

        counts = Counter(labels)
        can_split = all(c >= 2 for c in counts.values()) and n >= 2 * len(classes)
        if can_split:
            x_tr, x_te, y_tr, y_te = train_test_split(
                texts, labels, test_size=test_size,
                stratify=labels, random_state=42)
        else:
            x_tr, y_tr, x_te, y_te = texts, labels, [], []

        model = _SetFit.from_pretrained(base_model, labels=classes)
        train_ds = Dataset.from_dict({"text": x_tr, "label": y_tr})
        args = TrainingArguments(
            batch_size=batch_size, num_epochs=num_epochs,
            num_iterations=20, report_to=[])
        trainer = Trainer(model=model, args=args, train_dataset=train_ds)
        trainer.train()

        accuracy: Optional[float] = None
        if x_te:
            try:
                preds = model.predict(x_te)
                accuracy = float(accuracy_score(y_te, [str(p) for p in preds]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("setfit_eval_failed", error=str(exc))

        self.model = model
        self.labels = classes
        return {"accuracy": accuracy, "n_examples": n, "n_labels": len(classes)}

    def predict(self, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        if self.model is None or not texts:
            return [(None, 0.0) for _ in texts]
        try:
            proba = self.model.predict_proba(texts)
            import numpy as np

            proba = np.asarray(proba)
            labels = list(self.model.labels) if getattr(self.model, "labels", None) \
                else self.labels
            out: List[Tuple[Optional[str], float]] = []
            for row in proba:
                idx = int(np.argmax(row))
                label = labels[idx] if idx < len(labels) else None
                out.append((str(label) if label is not None else None,
                            float(row[idx])))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("setfit_predict_failed", error=str(exc))
            return [(None, 0.0) for _ in texts]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        (path / "labels.json").write_text(json.dumps(self.labels))

    @classmethod
    def load(cls, path: Path) -> "SetFitModel":
        from setfit import SetFitModel as _SetFit

        model = _SetFit.from_pretrained(str(path))
        labels = []
        labels_file = path / "labels.json"
        if labels_file.exists():
            labels = json.loads(labels_file.read_text())
        return cls(model=model, labels=labels)


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class ModelManager:
    """Trains, versions, selects and serves the trainable models.

    All public methods are safe: prediction returns "no prediction" rather than
    raising when no model is trained, so the engine can always fall back to
    similarity.
    """

    MODEL_TYPES = ("logreg", "setfit")

    def __init__(self, config: Config, storage: Storage):
        self.config = config
        self.storage = storage
        self.dir = config.abs_model_store_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / "registry.json"
        self._cache: Dict[str, Tuple[int, object]] = {}

    # ----------------------------------------------------------- registry
    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except Exception:  # noqa: BLE001
                logger.warning("registry_unreadable", path=str(self.registry_path))
        return {"active": None, "models": {}}

    def _save_registry(self, reg: Dict) -> None:
        self.registry_path.write_text(json.dumps(reg, indent=2))

    # ----------------------------------------------------------- info
    def training_count(self) -> int:
        return self.storage.count_training_data()

    def available_model_types(self) -> Dict[str, bool]:
        return {"logreg": LogRegModel.is_available(),
                "setfit": SetFitModel.is_available()}

    def setfit_detail(self) -> str:
        return setfit_available()[1]

    def can_train(self) -> Tuple[bool, str]:
        n = self.training_count()
        n_labels = len(self.storage.distinct_labels())
        if n < self.config.ml.min_examples_to_train:
            return False, (f"Need at least {self.config.ml.min_examples_to_train} "
                           f"approved examples (have {n}).")
        if n_labels < 2:
            return False, (f"Need at least 2 distinct account codes "
                           f"(have {n_labels}).")
        return True, ""

    def has_model(self) -> bool:
        reg = self._load_registry()
        for name, meta in reg.get("models", {}).items():
            if self._path_for(meta).exists():
                return True
        return False

    def _path_for(self, meta: Dict) -> Path:
        return self.dir / meta.get("path", "")

    def best_model_name(self) -> Optional[str]:
        """Trained model with the highest held-out accuracy (logreg breaks ties)."""
        reg = self._load_registry()
        best: Optional[str] = None
        best_acc = -1.0
        for name, meta in reg.get("models", {}).items():
            if not self._path_for(meta).exists():
                continue
            acc = meta.get("accuracy")
            acc = -1.0 if acc is None else float(acc)
            if acc > best_acc or (acc == best_acc and name == "logreg"):
                best_acc = acc
                best = name
        return best

    def active_model_name(self) -> Optional[str]:
        """Which model prediction uses. Honors an explicit registry 'active'."""
        reg = self._load_registry()
        active = reg.get("active")
        if active and self._path_for(reg["models"].get(active, {})).exists():
            return active
        return self.best_model_name()

    # ----------------------------------------------------------- training
    def _next_version(self, name: str) -> int:
        reg = self._load_registry()
        return int(reg.get("models", {}).get(name, {}).get("version", 0)) + 1

    def train_all(self, progress_cb=None) -> Dict[str, Dict]:
        """Train every available model on the current labelled data.

        Returns a dict ``{model_name: metrics_or_error}``. Never raises for an
        individual model — failures are reported per-model.
        """
        def report(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        texts, labels = self.storage.get_training_xy()
        ml = self.config.ml
        results: Dict[str, Dict] = {}
        reg = self._load_registry()
        reg.setdefault("models", {})

        # --- LogReg (always available) ---------------------------------
        report("Training Logistic Regression baseline…")
        try:
            model = LogRegModel()
            metrics = model.train(texts, labels, ml.test_size,
                                  ml.min_examples_to_train)
            version = self._next_version("logreg")
            rel = f"logreg/v{version}.joblib"
            model.save(self.dir / rel)
            reg["models"]["logreg"] = {
                "version": version, "path": rel, "trained_at": _now(), **metrics}
            results["logreg"] = metrics
            self._cache.pop("logreg", None)
            logger.info("logreg_trained", version=version, **metrics)
        except InsufficientTrainingData as exc:
            results["logreg"] = {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.error("logreg_train_failed", error=str(exc))
            results["logreg"] = {"error": str(exc)}

        # --- SetFit (optional) -----------------------------------------
        if SetFitModel.is_available():
            report("Training SetFit model (this can take a while)…")
            try:
                model = SetFitModel()
                metrics = model.train(
                    texts, labels, ml.setfit_model_name,
                    ml.setfit_num_epochs, ml.setfit_batch_size,
                    ml.test_size, ml.min_examples_to_train)
                version = self._next_version("setfit")
                rel = f"setfit/v{version}"
                model.save(self.dir / rel)
                reg["models"]["setfit"] = {
                    "version": version, "path": rel, "trained_at": _now(), **metrics}
                results["setfit"] = metrics
                self._cache.pop("setfit", None)
                logger.info("setfit_trained", version=version, **metrics)
            except InsufficientTrainingData as exc:
                results["setfit"] = {"error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                logger.error("setfit_train_failed", error=str(exc))
                results["setfit"] = {"error": str(exc)}
        else:
            results["setfit"] = {"error": f"Not installed. {SETFIT_PIP_HINT}"}

        # Pick the active model = best trained one.
        self._save_registry(reg)
        reg["active"] = self.best_model_name()
        self._save_registry(reg)
        report("Done.")
        return results

    # ----------------------------------------------------------- predict
    def _get_model(self, name: str):
        reg = self._load_registry()
        meta = reg.get("models", {}).get(name)
        if not meta:
            return None
        path = self._path_for(meta)
        if not path.exists():
            return None
        version = int(meta.get("version", 0))
        cached = self._cache.get(name)
        if cached and cached[0] == version:
            return cached[1]
        try:
            model = (LogRegModel.load(path) if name == "logreg"
                     else SetFitModel.load(path))
            self._cache[name] = (version, model)
            return model
        except Exception as exc:  # noqa: BLE001
            logger.error("model_load_failed", name=name, error=str(exc))
            return None

    def predict(self, texts: List[str]) -> List[Prediction]:
        """Predict labels for ``texts`` using the active model.

        Returns empty/none predictions (never raises) when no model is ready.
        """
        empty = [Prediction(None, 0.0, "") for _ in texts]
        if not texts or not self.config.ml.enabled:
            return empty
        name = self.active_model_name()
        if not name:
            return empty
        model = self._get_model(name)
        if model is None:
            return empty
        try:
            preds = model.predict(texts)
            return [Prediction(lbl, conf, name) for lbl, conf in preds]
        except Exception as exc:  # noqa: BLE001
            logger.error("predict_failed", name=name, error=str(exc))
            return empty

    # ----------------------------------------------------------- status
    def status_list(self):
        """Per-model status objects for the Models page."""
        from models.schemas import ModelStatus

        reg = self._load_registry()
        avail = self.available_model_types()
        active = self.active_model_name()
        out = []
        for name in self.MODEL_TYPES:
            meta = reg.get("models", {}).get(name, {})
            trained = bool(meta) and self._path_for(meta).exists()
            note = ""
            if name == "setfit" and not avail.get("setfit"):
                note = f"Optional — not installed. {SETFIT_PIP_HINT}"
            if name == active and trained:
                note = (note + " " if note else "") + "★ active"
            out.append(ModelStatus(
                name=name,
                available=avail.get(name, False),
                trained=trained,
                version=int(meta.get("version", 0)),
                n_examples=int(meta.get("n_examples", 0)),
                n_labels=int(meta.get("n_labels", 0)),
                accuracy=meta.get("accuracy"),
                trained_at=meta.get("trained_at"),
                note=note.strip(),
            ))
        return out

    def progressive_mode(self) -> str:
        """Resolve 'auto' into a concrete internal mode from example count."""
        n = self.training_count()
        if n >= self.config.ml.ml_primary_min:
            return "prefer_ml"
        if n >= self.config.ml.hybrid_min:
            return "hybrid"
        return "assist"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
