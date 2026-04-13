"""
Slack Embedding Engine — FAISS + scikit-learn TF-IDF

Handles TF-IDF vectorization, FAISS index management, and semantic search
over Slack messages. Uses scikit-learn TF-IDF (no PyTorch dependency needed,
fully compatible with Python 3.14 on Windows).
"""

import json
import os
import pickle
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "metadata.json"
LAST_INDEXED_PATH = DATA_DIR / "last_indexed.json"
VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.npy"


class SlackEmbeddingEngine:
    """Manages the full lifecycle: ingest → vectorize → index → search."""

    def __init__(self, slack_token: str | None = None):
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self.slack_client: WebClient | None = None

        if slack_token:
            self.slack_client = WebClient(token=slack_token)
            self.token = slack_token

        self.ts_to_metadata: dict[str, dict] = {}

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Vectorization ────────────────────────────────────────────────────

    def build_vectorizer(self, texts: list[str]) -> np.ndarray:
        """Build a TF-IDF vectorizer from texts and return the TF-IDF matrix.

        Uses sublinear TF, L2 normalization, and ngram range (1,2) for better
        semantic matching compared to plain TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            sublinear_tf=True,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
            dtype=np.float32,
        )

        tfidf_sparse = self.vectorizer.fit_transform(texts)
        # Convert to dense for FAISS (FAISS requires dense vectors)
        self.tfidf_matrix = tfidf_sparse.toarray().astype(np.float32)
        return self.tfidf_matrix

    def transform_query(self, query: str) -> np.ndarray:
        """Transform a single query string into a TF-IDF vector."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not initialized. Build or load the index first.")

        query_sparse = self.vectorizer.transform([query])
        return query_sparse.toarray().astype(np.float32)

    # ── Slack Data Fetching ──────────────────────────────────────────────

    def fetch_all_channels(self) -> list[dict]:
        """Fetch all channels the bot has access to."""
        if not self.slack_client:
            raise ValueError("Slack client not initialized. Provide a slack_token.")

        channels = []
        cursor = None

        while True:
            try:
                response = self.slack_client.conversations_list(
                    limit=200,
                    cursor=cursor,
                    types="public_channel"
                )
                channels.extend(response["channels"])
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    raise

        return channels

    def fetch_channel_messages(
        self,
        channel_id: str,
        oldest: str | None = None
    ) -> list[dict]:
        """Fetch all messages from a channel, handling pagination. Optionally only fetch messages newer than `oldest` timestamp."""
        if not self.slack_client:
            raise ValueError("Slack client not initialized. Provide a slack_token.")

        messages = []
        cursor = None

        while True:
            try:
                kwargs = {
                    "channel": channel_id,
                    "limit": 200,
                    "cursor": cursor,
                }
                if oldest:
                    kwargs["oldest"] = oldest

                response = self.slack_client.conversations_history(**kwargs)

                for msg in response.get("messages", []):
                    # Skip certain system messages but KEEP file sharing
                    subtype = msg.get("subtype")
                    if subtype and subtype not in ["file_share"]:
                        continue
                    messages.append(msg)

                    # Fetch thread replies if this message has a thread
                    if msg.get("reply_count", 0) > 0:
                        thread_msgs = self._fetch_thread_replies(
                            channel_id, msg["ts"], oldest
                        )
                        messages.extend(thread_msgs)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                elif e.response["error"] == "not_in_channel":
                    print(f"  ⚠️ Bot not in channel {channel_id}, skipping.")
                    break
                elif e.response["error"] == "channel_not_found":
                    print(f"  ⚠️ Channel {channel_id} not found, skipping.")
                    break
                else:
                    raise

        return messages

    def _fetch_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
        oldest: str | None = None
    ) -> list[dict]:
        """Fetch replies in a thread, excluding the parent message."""
        replies = []
        cursor = None

        while True:
            try:
                kwargs = {
                    "channel": channel_id,
                    "ts": thread_ts,
                    "limit": 200,
                    "cursor": cursor,
                }
                if oldest:
                    kwargs["oldest"] = oldest

                response = self.slack_client.conversations_replies(**kwargs)

                for msg in response.get("messages", []):
                    # Skip the parent message (it's already captured)
                    if msg["ts"] == thread_ts:
                        continue
                    subtype = msg.get("subtype")
                    if subtype and subtype not in ["file_share"]:
                        continue
                    msg["thread_ts"] = thread_ts
                    replies.append(msg)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limited on thread. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    break

        return replies

    def download_file(self, url: str) -> bytes:
        """Download a private Slack file using the bot token."""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content

    # ── Index Building ───────────────────────────────────────────────────

    def build_index(self, messages_with_meta: list[dict]) -> int:
        """Build a FAISS index from processed message dicts.

        Each dict must have at least: text, channel_id, channel_name, user, ts.
        Returns the number of messages indexed.
        """
        if not messages_with_meta:
            print("⚠️ No messages to index.")
            return 0

        # Filter out empty texts
        messages_with_meta = [
            m for m in messages_with_meta if m.get("text", "").strip()
        ]

        if not messages_with_meta:
            print("⚠️ All messages were empty after filtering.")
            return 0

        texts = [m["text"] for m in messages_with_meta]

        print(f"🔄 Building TF-IDF vectors for {len(texts)} messages...")
        embeddings = self.build_vectorizer(texts)

        embedding_dim = embeddings.shape[1]

        # Build FAISS index (using Inner Product for cosine similarity
        # since TF-IDF vectors are L2 normalized)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings)

        # Build metadata (assign sequential IDs)
        self.metadata = []
        for i, m in enumerate(messages_with_meta):
            ts_float = float(m["ts"])
            dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)

            self.metadata.append({
                "id": i,
                "channel_id": m["channel_id"],
                "channel_name": m["channel_name"],
                "user": m.get("user", "unknown"),
                "text": m["text"],
                "ts": m["ts"],
                "thread_ts": m.get("thread_ts"),
                "datetime": dt.isoformat(),
            })

        # Build lookup cache
        self.ts_to_metadata = {m["ts"]: m for m in self.metadata}

        print(f"✅ FAISS index built: {self.index.ntotal} vectors indexed (dim={embedding_dim}).")
        return self.index.ntotal

    def add_to_index(self, messages_with_meta: list[dict]) -> int:
        """Add new messages to an existing index by rebuilding.

        Since TF-IDF vocabulary may change with new documents, we rebuild
        the entire index with all messages (old + new).
        Returns the number of new messages added.
        """
        if not messages_with_meta:
            return 0

        messages_with_meta = [
            m for m in messages_with_meta if m.get("text", "").strip()
        ]

        if not messages_with_meta:
            return 0

        if self.index is None or not self.metadata:
            return self.build_index(messages_with_meta)

        # Combine existing metadata with new messages
        all_messages = list(self.metadata) + [
            {
                "channel_id": m["channel_id"],
                "channel_name": m["channel_name"],
                "user": m.get("user", "unknown"),
                "text": m["text"],
                "ts": m["ts"],
                "thread_ts": m.get("thread_ts"),
            }
            for m in messages_with_meta
        ]

        new_count = len(messages_with_meta)

        # Rebuild entire index with updated vocabulary
        self.build_index(all_messages)

        return new_count

    # ── Index Persistence ────────────────────────────────────────────────

    def save_index(self):
        """Save FAISS index, TF-IDF vectorizer, and metadata to disk."""
        if self.index is None or not self.metadata:
            print("⚠️ Nothing to save — index is empty.")
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        # Save the vectorizer so we can transform queries later
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"💾 Index saved: {self.index.ntotal} vectors → {FAISS_INDEX_PATH}")
        print(f"💾 Metadata saved: {len(self.metadata)} entries → {METADATA_PATH}")
        print(f"💾 Vectorizer saved → {VECTORIZER_PATH}")

    def load_index(self) -> bool:
        """Load FAISS index, vectorizer, and metadata from disk. Returns True if loaded successfully."""
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists() or not VECTORIZER_PATH.exists():
            print("ℹ️ No existing index found on disk.")
            return False

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Build lookup cache
        self.ts_to_metadata = {m["ts"]: m for m in self.metadata}

        print(f"📂 Index loaded: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries.")
        return True

    # ── Last Indexed Tracking ────────────────────────────────────────────

    def save_last_indexed(self, channel_timestamps: dict[str, str]):
        """Save the last indexed timestamp per channel."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LAST_INDEXED_PATH, "w", encoding="utf-8") as f:
            json.dump(channel_timestamps, f, indent=2)

    def load_last_indexed(self) -> dict[str, str]:
        """Load the last indexed timestamp per channel."""
        if not LAST_INDEXED_PATH.exists():
            return {}
        with open(LAST_INDEXED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Semantic Search ──────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the FAISS index for messages most similar to the query.

        Uses TF-IDF cosine similarity (inner product on L2-normalized vectors).
        Returns a list of dicts with: text, channel_name, user, datetime, score.
        """
        if not query or not query.strip():
            return []

        if self.index is None or self.index.ntotal == 0:
            return []

        if self.vectorizer is None:
            return []

        # Clamp top_k to available vectors
        top_k = min(top_k, self.index.ntotal)

        query_vector = self.transform_query(query)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "channel_name": meta["channel_name"],
                "channel_id": meta["channel_id"],
                "user": meta["user"],
                "datetime": meta["datetime"],
                "thread_ts": meta.get("thread_ts"),
                "score": float(score),
            })

        return results
