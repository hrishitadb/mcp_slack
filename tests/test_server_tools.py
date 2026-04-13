"""
Integration tests for the MCP semantic_search tool in server.py.

These tests verify that the semantic_search tool is properly registered
and returns results in the expected format.
"""

import asyncio
import json
import os
import sys

import pytest
import pytest_asyncio

# Add parent directory to path so we can import server modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ── Helpers ──────────────────────────────────────────────────────────────────

PYTHON_EXE = sys.executable
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.py")


@pytest_asyncio.fixture
async def mcp_session():
    """Start server.py as a subprocess and create an MCP session."""
    server = StdioServerParameters(
        command=PYTHON_EXE,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# ── Test 1: semantic_search tool exists ──────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search_tool_exists(mcp_session):
    """Tool is registered and discoverable via MCP."""
    tools_response = await mcp_session.list_tools()
    tool_names = [t.name for t in tools_response.tools]

    assert "semantic_search" in tool_names, (
        f"semantic_search not found in tools: {tool_names}"
    )


# ── Test 2: semantic_search returns results ──────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search_returns_results(mcp_session):
    """Tool returns results (or an appropriate error if no index exists)."""
    result = await mcp_session.call_tool("semantic_search", {"query": "deployment issue"})

    # Extract text content from MCP response
    texts = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            texts.append(block.text)

    response_text = "\n".join(texts)
    # Should return either results or an error message about missing index
    assert response_text, "Expected non-empty response from semantic_search"


# ── Test 3: semantic_search result format ────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search_result_format(mcp_session):
    """Each result has required fields if index is loaded."""
    result = await mcp_session.call_tool("semantic_search", {"query": "test query"})

    texts = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            texts.append(block.text)

    response_text = "\n".join(texts)
    parsed = json.loads(response_text)

    if isinstance(parsed, list) and len(parsed) > 0:
        # If we got results, verify the format
        item = parsed[0]
        assert "text" in item, "Missing 'text' field in result"
        assert "channel_name" in item, "Missing 'channel_name' field in result"
        assert "user" in item, "Missing 'user' field in result"
        assert "datetime" in item, "Missing 'datetime' field in result"
        assert "score" in item, "Missing 'score' field in result"
    elif isinstance(parsed, dict) and "error" in parsed:
        # No index available — that's OK for this test
        assert "index" in parsed["error"].lower()


# ── Test 4: semantic_search with custom top_k ────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search_with_custom_top_k(mcp_session):
    """Custom top_k parameter is accepted by the tool."""
    result = await mcp_session.call_tool(
        "semantic_search",
        {"query": "test query", "top_k": 2}
    )

    texts = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            texts.append(block.text)

    response_text = "\n".join(texts)
    parsed = json.loads(response_text)

    if isinstance(parsed, list):
        assert len(parsed) <= 2, f"Expected at most 2 results, got {len(parsed)}"
