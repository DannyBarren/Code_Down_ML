"""Tests for the similarity/grouping layer (TF-IDF backend)."""

import numpy as np

from src.similarity import (
    BACKEND_TFIDF,
    Embedder,
    cosine_sim_matrix,
    group_transactions,
    semantic_backend_available,
)


def test_backend_available_returns_tuple():
    available, detail = semantic_backend_available()
    assert isinstance(available, bool)
    assert isinstance(detail, str)


def test_tfidf_embeddings_normalised(config):
    emb = Embedder(config)  # use_embeddings False -> tfidf
    assert emb.backend == BACKEND_TFIDF
    vecs = emb.encode(["cred hub fee", "cred hub monthly", "office depot"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_grouping_clusters_similar(config):
    texts = [
        "cred hub resident screening",
        "cred hub resident screening fee",
        "cred hub screening",
        "office depot paper",
        "office depot toner",
    ]
    result = group_transactions(texts, config)
    assert result.backend == BACKEND_TFIDF
    # The three cred-hub rows should share a label distinct from office depot.
    labels = result.labels
    credhub_labels = {labels[0], labels[1], labels[2]}
    assert len(credhub_labels) == 1 and -1 not in credhub_labels


def test_single_row_no_crash(config):
    result = group_transactions(["only one"], config)
    assert list(result.labels) == [-1]


def test_empty_input(config):
    result = group_transactions([], config)
    assert result.n_groups == 0


def test_cosine_matrix():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = cosine_sim_matrix(v)
    assert np.isclose(m[0, 1], 1.0)
    assert np.isclose(m[0, 2], 0.0)
