"""
Unit tests for the SlackEmbeddingEngine — TF-IDF vectorization, indexing, and search.

These tests do not require a Slack connection. They test the core engine
with synthetic data.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from embeddings import SlackEmbeddingEngine


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Create a fresh engine instance (no Slack client)."""
    return SlackEmbeddingEngine()


@pytest.fixture
def sample_messages():
    """Sample messages simulating Slack data for testing."""
    return [
        {
            "text": "Try restarting the Docker container, that usually fixes the deployment issue.",
            "channel_id": "C001",
            "channel_name": "devops",
            "user": "U001",
            "ts": "1712700000.000001",
        },
        {
            "text": "The database migration failed because of a missing column in the users table.",
            "channel_id": "C002",
            "channel_name": "backend",
            "user": "U002",
            "ts": "1712700100.000002",
        },
        {
            "text": "We should update the CI/CD pipeline to include linting checks.",
            "channel_id": "C001",
            "channel_name": "devops",
            "user": "U003",
            "ts": "1712700200.000003",
        },
        {
            "text": "The React component is re-rendering too many times, use useMemo to fix it.",
            "channel_id": "C003",
            "channel_name": "frontend",
            "user": "U004",
            "ts": "1712700300.000004",
        },
        {
            "text": "Has anyone tried the new Python 3.14 release? Any compatibility issues?",
            "channel_id": "C004",
            "channel_name": "general",
            "user": "U005",
            "ts": "1712700400.000005",
        },
        {
            "text": "For the Kubernetes pod crash, check the memory limits in the deployment yaml.",
            "channel_id": "C001",
            "channel_name": "devops",
            "user": "U001",
            "ts": "1712700500.000006",
        },
    ]


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporarily redirect the data directory to a temp folder."""
    import embeddings
    original_data_dir = embeddings.DATA_DIR
    original_index_path = embeddings.FAISS_INDEX_PATH
    original_meta_path = embeddings.METADATA_PATH
    original_last_path = embeddings.LAST_INDEXED_PATH
    original_vec_path = embeddings.VECTORIZER_PATH

    embeddings.DATA_DIR = tmp_path / "data"
    embeddings.FAISS_INDEX_PATH = embeddings.DATA_DIR / "faiss_index.bin"
    embeddings.METADATA_PATH = embeddings.DATA_DIR / "metadata.json"
    embeddings.LAST_INDEXED_PATH = embeddings.DATA_DIR / "last_indexed.json"
    embeddings.VECTORIZER_PATH = embeddings.DATA_DIR / "tfidf_vectorizer.pkl"
    embeddings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    yield tmp_path / "data"

    # Restore originals
    embeddings.DATA_DIR = original_data_dir
    embeddings.FAISS_INDEX_PATH = original_index_path
    embeddings.METADATA_PATH = original_meta_path
    embeddings.LAST_INDEXED_PATH = original_last_path
    embeddings.VECTORIZER_PATH = original_vec_path


# ── Test 1: Vectorizer builds ───────────────────────────────────────────────

def test_vectorizer_builds(engine):
    """TF-IDF vectorizer builds from text inputs without error."""
    texts = ["hello world", "test sentence"]
    matrix = engine.build_vectorizer(texts)

    assert engine.vectorizer is not None
    assert matrix is not None
    assert matrix.shape[0] == 2


# ── Test 2: Vector shape and normalization ───────────────────────────────────

def test_vector_shape_and_normalization(engine):
    """TF-IDF vectors are L2 normalized (for cosine similarity via inner product)."""
    texts = ["hello world foo bar", "test sentence another example"]
    matrix = engine.build_vectorizer(texts)

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32

    # Check L2 normalization (each row should have norm ~1.0)
    for i in range(matrix.shape[0]):
        norm = np.linalg.norm(matrix[i])
        np.testing.assert_almost_equal(norm, 1.0, decimal=3)


# ── Test 3: Query transform consistency ──────────────────────────────────────

def test_query_transform_consistency(engine):
    """Same query produces the same vector."""
    texts = ["The quick brown fox", "jumps over the lazy dog"]
    engine.build_vectorizer(texts)

    vec1 = engine.transform_query("quick fox")
    vec2 = engine.transform_query("quick fox")

    np.testing.assert_array_equal(vec1, vec2)


# ── Test 4: Index build and search ───────────────────────────────────────────

def test_index_build_and_search(engine, sample_messages):
    """Build index from sample messages, search returns results."""
    count = engine.build_index(sample_messages)

    assert count == len(sample_messages)
    assert engine.index is not None
    assert engine.index.ntotal == len(sample_messages)

    results = engine.search("docker deployment problem")
    assert len(results) > 0
    assert "text" in results[0]
    assert "channel_name" in results[0]
    assert "score" in results[0]


# ── Test 5: Semantic relevance ───────────────────────────────────────────────

def test_semantic_relevance(engine, sample_messages):
    """'docker container restart' should match Docker-related messages."""
    engine.build_index(sample_messages)

    results = engine.search("docker container restart fix deployment", top_k=1)
    assert len(results) == 1

    # The top result should mention docker or container or deployment
    top_text = results[0]["text"].lower()
    assert any(kw in top_text for kw in ["docker", "container", "deployment"])


# ── Test 6: Save and load index ──────────────────────────────────────────────

def test_save_and_load_index(engine, sample_messages, tmp_data_dir):
    """Index persists to disk and reloads correctly."""
    engine.build_index(sample_messages)
    engine.save_index()

    # Create a new engine and load
    engine2 = SlackEmbeddingEngine()
    loaded = engine2.load_index()

    assert loaded is True
    assert engine2.index.ntotal == engine.index.ntotal
    assert len(engine2.metadata) == len(engine.metadata)
    assert engine2.vectorizer is not None

    # Verify search still works after reload
    results = engine2.search("database migration")
    assert len(results) > 0


# ── Test 7: Incremental update ───────────────────────────────────────────────

def test_incremental_update(engine, sample_messages):
    """New messages added to existing index via rebuild."""
    # Build initial index with first 3 messages
    engine.build_index(sample_messages[:3])
    initial_count = engine.index.ntotal

    # Add remaining messages
    new_messages = sample_messages[3:]
    added = engine.add_to_index(new_messages)

    assert added == len(new_messages)
    # After rebuild, total should be all messages
    assert engine.index.ntotal == initial_count + len(new_messages)
    assert len(engine.metadata) == initial_count + len(new_messages)


# ── Test 8: Empty query ─────────────────────────────────────────────────────

def test_empty_query(engine, sample_messages):
    """Empty/whitespace query handled gracefully."""
    engine.build_index(sample_messages)

    assert engine.search("") == []
    assert engine.search("   ") == []


# ── Test 9: Metadata integrity ──────────────────────────────────────────────

def test_metadata_integrity(engine, sample_messages, tmp_data_dir):
    """Metadata (channel, user, timestamp) preserved through save/load cycle."""
    engine.build_index(sample_messages)
    engine.save_index()

    engine2 = SlackEmbeddingEngine()
    engine2.load_index()

    for meta in engine2.metadata:
        assert "channel_id" in meta
        assert "channel_name" in meta
        assert "user" in meta
        assert "text" in meta
        assert "ts" in meta
        assert "datetime" in meta
        assert meta["channel_id"].startswith("C")
        assert meta["user"].startswith("U")


# ── Test 10: top_k parameter ────────────────────────────────────────────────

def test_top_k_parameter(engine, sample_messages):
    """top_k=3 returns exactly 3 results."""
    engine.build_index(sample_messages)

    results = engine.search("deployment issue", top_k=3)
    assert len(results) == 3

    results = engine.search("deployment issue", top_k=1)
    assert len(results) == 1

    # top_k larger than index should return all
    results = engine.search("deployment issue", top_k=100)
    assert len(results) == len(sample_messages)
