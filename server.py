import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from mcp.server.fastmcp import FastMCP
from embeddings import SlackEmbeddingEngine

# load environment variables
load_dotenv()

# create MCP server
mcp = FastMCP("Slack Server")

# slack client
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

import sys
import contextlib

# embedding engine — load index on startup
with contextlib.redirect_stdout(sys.stderr):
    engine = SlackEmbeddingEngine()
    _index_loaded = engine.load_index()
    if _index_loaded:
        print(f"✅ Semantic search ready — {engine.index.ntotal} vectors loaded.")
    else:
        print("⚠️ No FAISS index found. Run 'python ingest.py' first to enable semantic search.")


# TOOL 1 — READ MESSAGES
@mcp.tool()
async def read_messages(channel_id: str, limit: int = 5):

    response = client.conversations_history(
        channel=channel_id,
        limit=limit
    )

    messages = []
    for i, msg in enumerate(response.get("messages", [])):
        ts = msg.get("ts")
        text = msg.get("text", "")
        
        # Cross-reference with our local knowledge base for vision descriptions
        if ts in engine.ts_to_metadata:
            enriched_text = engine.ts_to_metadata[ts].get("text")
            if enriched_text:
                text = enriched_text

        messages.append(f"[{i+1}] {text}")

    if not messages:
        return "No messages found in this channel."
    
    return "\n".join(messages)


# TOOL 2 — SEND MESSAGE
@mcp.tool()
async def send_message(channel_id: str, text: str):
    """Send a message to a specific Slack channel."""
    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=text
        )
        return {"status": "success", "message": response["message"]["text"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# TOOL 3 — SEMANTIC SEARCH across all Slack channels
@mcp.tool()
async def semantic_search(query: str, top_k: int = 10):
    """Search all indexed Slack messages for content semantically similar to the query.
    Use this tool when the user asks about a problem, wants to find past discussions,
    or needs solutions based on team knowledge shared in Slack channels.
    Returns the most relevant messages with channel name, author, and timestamp."""

    if engine.index is None or engine.index.ntotal == 0:
        return {"error": "No index available. Run 'python ingest.py' to build the index first."}

    results = engine.search(query, top_k=top_k)
    return results


if __name__ == "__main__":
    print("Slack MCP Server running...", file=sys.stderr)
    mcp.run()
