"""Group similar transactions using semantic embeddings.

Design notes
------------
* Primary backend is ``sentence-transformers`` (all-MiniLM-L6-v2). It produces
  rich semantic embeddings so e.g. "Cloud Hosting fee" and "CloudHost monthly"
  land in the same group even though the strings differ.
* If sentence-transformers / torch is not installed (it is heavy), we fall back
  to a TF-IDF char/word vectoriser from scikit-learn. The rest of the pipeline
  is identical, so the app is fully functional either way.
* Grouping is done with DBSCAN on cosine distance, which does not require us to
  guess the number of clusters up front.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# The Hugging Face stack (used by sentence-transformers / setfit) will try to
# import TensorFlow when it is present, even though this app only needs PyTorch.
# In mixed environments a broken/incompatible TensorFlow then crashes the import
# of sentence-transformers. Forcing the PyTorch-only backend keeps us robust.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np

from src.config import Config
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Shown to users when the optional semantic stack is missing.
PIP_INSTALL_HINT = "pip install sentence-transformers torch"

BACKEND_SEMANTIC = "sentence-transformers"
BACKEND_TFIDF = "tfidf"


@functools.lru_cache(maxsize=1)
def semantic_backend_available() -> Tuple[bool, str]:
    """Return ``(available, detail)`` for the optional semantic backend.

    Only checks that the libraries can be imported — it does not download or
    load the model. Cached so the UI can call it cheaply on every rerun.
    """
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401

        return True, "sentence-transformers + torch are installed."
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@dataclass
class GroupingResult:
    """Output of the grouping step."""

    labels: np.ndarray            # cluster id per row (-1 = noise / singleton)
    embeddings: np.ndarray        # row embeddings (L2-normalised)
    backend: str                  # which embedding backend was used
    n_groups: int


class Embedder:
    """Encapsulates the embedding backend with graceful fallback."""

    def __init__(self, config: Config, progress_cb=None):
        self.config = config
        self.progress_cb = progress_cb
        self._st_model = None
        self.backend = BACKEND_TFIDF
        self.fallback_reason = ""
        if config.similarity.use_embeddings:
            self._try_load_sentence_transformer()
        else:
            self.fallback_reason = "Semantic backend disabled in settings."

    def _try_load_sentence_transformer(self) -> None:
        available, detail = semantic_backend_available()
        if not available:
            self.fallback_reason = (
                f"sentence-transformers/torch not installed ({detail})."
            )
            logger.warning("sentence_transformers_unavailable",
                           error=detail, fallback=BACKEND_TFIDF)
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._report(
                f"Loading embedding model '{self.config.similarity.model_name}'…")
            self._st_model = SentenceTransformer(self.config.similarity.model_name)
            self.backend = BACKEND_SEMANTIC
            logger.info("embedder_ready", backend=self.backend,
                        model=self.config.similarity.model_name)
        except Exception as exc:  # noqa: BLE001 - model download/load can fail
            self.fallback_reason = (
                f"Could not load model '{self.config.similarity.model_name}' "
                f"({exc}). Using TF-IDF instead."
            )
            logger.warning("model_load_failed", error=str(exc),
                           fallback=BACKEND_TFIDF)
            self._st_model = None
            self.backend = BACKEND_TFIDF

    def _report(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings for ``texts``."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self._st_model is not None:
            self._report("Computing semantic embeddings…")
            emb = self._st_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=64,
            )
            return np.asarray(emb, dtype=np.float32)

        # ---- TF-IDF fallback -------------------------------------------
        self._report("Computing TF-IDF vectors (fast, no model needed)…")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        # Word + character n-grams capture both tokens and fuzzy spellings.
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
        )
        safe_texts = [t if t and t.strip() else " " for t in texts]
        matrix = vectorizer.fit_transform(safe_texts)
        return normalize(matrix).astype(np.float32).toarray()


def group_transactions(
    texts: List[str],
    config: Config,
    embedder: Optional[Embedder] = None,
    progress_cb=None,
) -> GroupingResult:
    """Embed ``texts`` and cluster them into groups of similar transactions."""
    embedder = embedder or Embedder(config, progress_cb=progress_cb)
    embeddings = embedder.encode(texts)

    n = len(texts)
    if n == 0:
        return GroupingResult(np.array([]), embeddings, embedder.backend, 0)
    if n == 1:
        return GroupingResult(np.array([-1]), embeddings, embedder.backend, 0)

    if progress_cb:
        progress_cb("Clustering similar transactions…")

    labels = _cluster(embeddings, config)
    n_groups = len({l for l in labels if l != -1})
    logger.info("grouping_done", backend=embedder.backend, rows=n,
                groups=n_groups)
    return GroupingResult(labels, embeddings, embedder.backend, n_groups)


def _cluster(embeddings: np.ndarray, config: Config) -> np.ndarray:
    """Cluster L2-normalised embeddings with DBSCAN on cosine distance."""
    from sklearn.cluster import DBSCAN

    eps = config.similarity.eps
    min_samples = max(1, config.similarity.min_cluster_size)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(embeddings)


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for already L2-normalised embeddings."""
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0))
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)
