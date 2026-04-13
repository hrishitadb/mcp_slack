"""
Slack Ingestion Script — Fetch messages and build/update the FAISS index.

Usage:
    python ingest.py              # Full re-index from scratch
    python ingest.py --update     # Incremental update (new messages only)
"""

import os
import sys
import time
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from embeddings import SlackEmbeddingEngine
import vision

load_dotenv()



def process_images_in_message(engine: SlackEmbeddingEngine, msg: dict) -> str:
    """Detect images, download them, and get descriptions from Groq Vision."""
    descriptions = []
    
    if "files" in msg:
        for f in msg["files"]:
            mimetype = f.get("mimetype", "")
            if mimetype.startswith("image/"):
                url = f.get("url_private")
                if not url:
                    continue
                
                print(f"   🖼️  Processing image: {f.get('name')}...")
                try:
                    content = engine.download_file(url)
                    desc = vision.get_image_description(content)
                    if desc:
                        descriptions.append(f"[Image Description: {desc}]")
                except Exception as e:
                    print(f"   ⚠️  Image download failed: {e}")
                    
    return "\n".join(descriptions)

def run_full_ingest(engine: SlackEmbeddingEngine):
    """Fetch ALL messages from all channels and build a fresh index."""

    print("\n" + "=" * 60)
    print("   FULL INGESTION — Fetching all Slack messages")
    print("=" * 60 + "\n")

    # Step 1: Get all channels
    print("📋 Fetching channel list...")
    channels = engine.fetch_all_channels()
    print(f"   Found {len(channels)} channels.\n")

    # Step 2: Fetch messages from each channel
    all_messages = []
    channel_timestamps = {}

    for i, channel in enumerate(channels):
        ch_id = channel["id"]
        ch_name = channel.get("name", "unknown")
        print(f"[{i+1}/{len(channels)}] 📥 Fetching: #{ch_name} ({ch_id})")

        messages = engine.fetch_channel_messages(ch_id)
        print(f"   → {len(messages)} messages fetched.")

        # Track the latest timestamp for incremental updates
        if messages:
            latest_ts = max(m["ts"] for m in messages)
            channel_timestamps[ch_id] = latest_ts

        # Enrich messages with channel info and process images
        for msg in messages:
            text = msg.get("text", "")
            
            # Add image descriptions to the text
            img_desc = process_images_in_message(engine, msg)
            if img_desc:
                text = f"{text}\n\n{img_desc}".strip()
                
            all_messages.append({
                "text": text,
                "channel_id": ch_id,
                "channel_name": ch_name,
                "user": msg.get("user", "unknown"),
                "ts": msg["ts"],
                "thread_ts": msg.get("thread_ts"),
            })

    print(f"\n📊 Total messages collected: {len(all_messages)}")

    # Step 3: Build index
    count = engine.build_index(all_messages)

    # Step 4: Save
    engine.save_index()
    engine.save_last_indexed(channel_timestamps)

    print(f"\n✅ Full ingestion complete! {count} messages indexed.\n")
    return count


def run_incremental_update(engine: SlackEmbeddingEngine):
    """Fetch only NEW messages since the last ingestion and add to the existing index."""

    print("\n" + "=" * 60)
    print("   INCREMENTAL UPDATE — Fetching new messages only")
    print("=" * 60 + "\n")

    # Load existing index
    loaded = engine.load_index()
    if not loaded:
        print("⚠️ No existing index found. Running full ingestion instead.\n")
        return run_full_ingest(engine)

    existing_count = engine.index.ntotal
    print(f"📂 Existing index has {existing_count} vectors.\n")

    # Load last indexed timestamps
    last_indexed = engine.load_last_indexed()

    # Get all channels
    print("📋 Fetching channel list...")
    channels = engine.fetch_all_channels()
    print(f"   Found {len(channels)} channels.\n")

    # Fetch only new messages
    new_messages = []
    channel_timestamps = dict(last_indexed)  # start with existing

    for i, channel in enumerate(channels):
        ch_id = channel["id"]
        ch_name = channel.get("name", "unknown")
        oldest = last_indexed.get(ch_id)

        label = f"(since {oldest})" if oldest else "(first time)"
        print(f"[{i+1}/{len(channels)}] 📥 #{ch_name} {label}")

        messages = engine.fetch_channel_messages(ch_id, oldest=oldest)

        if messages:
            # Filter out messages we already have (oldest is inclusive)
            if oldest:
                messages = [m for m in messages if m["ts"] > oldest]

            print(f"   → {len(messages)} new messages.")

            latest_ts = max(m["ts"] for m in messages) if messages else oldest
            if latest_ts:
                channel_timestamps[ch_id] = latest_ts

            for msg in messages:
                text = msg.get("text", "")
                
                img_desc = process_images_in_message(engine, msg)
                if img_desc:
                    text = f"{text}\n\n{img_desc}".strip()
                    
                new_messages.append({
                    "text": text,
                    "channel_id": ch_id,
                    "channel_name": ch_name,
                    "user": msg.get("user", "unknown"),
                    "ts": msg["ts"],
                    "thread_ts": msg.get("thread_ts"),
                })
        else:
            print(f"   → No new messages.")

    if new_messages:
        print(f"\n📊 New messages to add: {len(new_messages)}")
        added = engine.add_to_index(new_messages)
        engine.save_index()
        engine.save_last_indexed(channel_timestamps)
        print(f"\n✅ Incremental update complete! Added {added} messages.")
        print(f"   Total index size: {engine.index.ntotal} vectors.\n")
        return added
    else:
        print("\nℹ️ No new messages found. Index is up to date.\n")
        engine.save_last_indexed(channel_timestamps)
        return 0


def main():
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in .env file.")
        sys.exit(1)

    engine = SlackEmbeddingEngine(slack_token=token)

    if "--update" in sys.argv:
        run_incremental_update(engine)
    else:
        run_full_ingest(engine)


if __name__ == "__main__":
    main()
