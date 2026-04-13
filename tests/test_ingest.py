"""
Tests for the Slack ingestion pipeline.

Tests cover channel fetching, pagination, thread replies,
rate-limit handling, and timestamp tracking.
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import SlackEmbeddingEngine


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporarily redirect the data directory to a temp folder."""
    import embeddings
    original_data_dir = embeddings.DATA_DIR
    original_index_path = embeddings.FAISS_INDEX_PATH
    original_meta_path = embeddings.METADATA_PATH
    original_last_path = embeddings.LAST_INDEXED_PATH

    embeddings.DATA_DIR = tmp_path / "data"
    embeddings.FAISS_INDEX_PATH = embeddings.DATA_DIR / "faiss_index.bin"
    embeddings.METADATA_PATH = embeddings.DATA_DIR / "metadata.json"
    embeddings.LAST_INDEXED_PATH = embeddings.DATA_DIR / "last_indexed.json"
    embeddings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    yield tmp_path / "data"

    embeddings.DATA_DIR = original_data_dir
    embeddings.FAISS_INDEX_PATH = original_index_path
    embeddings.METADATA_PATH = original_meta_path
    embeddings.LAST_INDEXED_PATH = original_last_path


@pytest.fixture
def mock_slack_client():
    """Create a mock Slack WebClient."""
    client = MagicMock()
    return client


@pytest.fixture
def engine_with_mock(mock_slack_client):
    """Engine with a mocked Slack client."""
    engine = SlackEmbeddingEngine(slack_token="xoxb-fake-token")
    engine.slack_client = mock_slack_client
    return engine


# ── Test 1: Fetch channels ──────────────────────────────────────────────────

def test_fetch_channels(engine_with_mock, mock_slack_client):
    """Bot can list accessible channels from Slack API."""
    mock_slack_client.conversations_list.return_value = {
        "channels": [
            {"id": "C001", "name": "general"},
            {"id": "C002", "name": "random"},
        ],
        "response_metadata": {"next_cursor": ""},
    }

    channels = engine_with_mock.fetch_all_channels()
    assert len(channels) == 2
    assert channels[0]["id"] == "C001"
    assert channels[1]["name"] == "random"
    mock_slack_client.conversations_list.assert_called_once()


# ── Test 2: Fetch messages with pagination ───────────────────────────────────

def test_fetch_messages_pagination(engine_with_mock, mock_slack_client):
    """Messages beyond the first page are fetched correctly."""
    # Page 1
    page1 = {
        "messages": [
            {"text": "message 1", "ts": "1712700000.000001", "user": "U001"},
            {"text": "message 2", "ts": "1712700001.000002", "user": "U002"},
        ],
        "response_metadata": {"next_cursor": "page2_cursor"},
    }

    # Page 2
    page2 = {
        "messages": [
            {"text": "message 3", "ts": "1712700002.000003", "user": "U003"},
        ],
        "response_metadata": {"next_cursor": ""},
    }

    mock_slack_client.conversations_history.side_effect = [page1, page2]

    messages = engine_with_mock.fetch_channel_messages("C001")
    assert len(messages) == 3
    assert messages[0]["text"] == "message 1"
    assert messages[2]["text"] == "message 3"
    assert mock_slack_client.conversations_history.call_count == 2


# ── Test 3: Thread replies fetched ───────────────────────────────────────────

def test_thread_replies_fetched(engine_with_mock, mock_slack_client):
    """Thread replies are included in ingestion."""
    # Main message with replies
    mock_slack_client.conversations_history.return_value = {
        "messages": [
            {
                "text": "parent message",
                "ts": "1712700000.000001",
                "user": "U001",
                "reply_count": 2,
            },
        ],
        "response_metadata": {"next_cursor": ""},
    }

    # Thread replies
    mock_slack_client.conversations_replies.return_value = {
        "messages": [
            # Parent is repeated in replies (Slack behavior)
            {"text": "parent message", "ts": "1712700000.000001", "user": "U001"},
            {"text": "reply 1", "ts": "1712700001.000002", "user": "U002"},
            {"text": "reply 2", "ts": "1712700002.000003", "user": "U003"},
        ],
        "response_metadata": {"next_cursor": ""},
    }

    messages = engine_with_mock.fetch_channel_messages("C001")

    # Should have parent + 2 replies (parent not duplicated)
    assert len(messages) == 3
    texts = [m["text"] for m in messages]
    assert "parent message" in texts
    assert "reply 1" in texts
    assert "reply 2" in texts


# ── Test 4: Rate limit handling ──────────────────────────────────────────────

def test_rate_limit_handling(engine_with_mock, mock_slack_client):
    """Graceful retry on Slack API rate-limit response."""
    from slack_sdk.errors import SlackApiError

    # Create a rate-limit error
    rate_limit_response = MagicMock()
    rate_limit_response.__getitem__ = MagicMock(return_value="ratelimited")
    rate_limit_response.headers = {"Retry-After": "1"}

    rate_limit_error = SlackApiError(
        message="ratelimited",
        response=rate_limit_response,
    )

    # First call fails with rate limit, second succeeds
    success_response = {
        "channels": [{"id": "C001", "name": "general"}],
        "response_metadata": {"next_cursor": ""},
    }

    mock_slack_client.conversations_list.side_effect = [
        rate_limit_error,
        success_response,
    ]

    with patch("time.sleep"):  # Don't actually sleep in tests
        channels = engine_with_mock.fetch_all_channels()

    assert len(channels) == 1
    assert mock_slack_client.conversations_list.call_count == 2


# ── Test 5: Incremental timestamp tracking ──────────────────────────────────

def test_incremental_timestamp_tracking(engine_with_mock, tmp_data_dir):
    """last_indexed.json is written and read correctly."""
    timestamps = {
        "C001": "1712700500.000006",
        "C002": "1712700100.000002",
    }

    engine_with_mock.save_last_indexed(timestamps)

    loaded = engine_with_mock.load_last_indexed()
    assert loaded == timestamps
    assert loaded["C001"] == "1712700500.000006"


# ── Test 6: Full pipeline (mock) ────────────────────────────────────────────

def test_full_pipeline(engine_with_mock, mock_slack_client, tmp_data_dir):
    """End-to-end: fetch → embed → index → search works (with mocked Slack)."""
    # Mock channel list
    mock_slack_client.conversations_list.return_value = {
        "channels": [
            {"id": "C001", "name": "devops"},
        ],
        "response_metadata": {"next_cursor": ""},
    }

    # Mock messages
    mock_slack_client.conversations_history.return_value = {
        "messages": [
            {"text": "Restart Docker to fix deployment errors", "ts": "1712700000.000001", "user": "U001"},
            {"text": "The database backup runs every night at 2am", "ts": "1712700001.000002", "user": "U002"},
            {"text": "Use kubectl rollout restart for Kubernetes pods", "ts": "1712700002.000003", "user": "U003"},
        ],
        "response_metadata": {"next_cursor": ""},
    }

    # Step 1: Fetch
    channels = engine_with_mock.fetch_all_channels()
    all_messages = []
    for ch in channels:
        msgs = engine_with_mock.fetch_channel_messages(ch["id"])
        for msg in msgs:
            all_messages.append({
                "text": msg["text"],
                "channel_id": ch["id"],
                "channel_name": ch["name"],
                "user": msg["user"],
                "ts": msg["ts"],
            })

    # Step 2: Build index
    count = engine_with_mock.build_index(all_messages)
    assert count == 3

    # Step 3: Save
    engine_with_mock.save_index()

    # Step 4: Search
    results = engine_with_mock.search("docker deployment fix")
    assert len(results) > 0
    # Top result should be the Docker message
    assert "docker" in results[0]["text"].lower() or "deployment" in results[0]["text"].lower()
