import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from mcp.server.fastmcp import FastMCP

# load environment variables
load_dotenv()

# create MCP server
mcp = FastMCP("Slack Server")

# slack client
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))


# TOOL 1 — READ MESSAGES
@mcp.tool()
async def read_messages(channel_id: str, limit: int = 5):

    response = client.conversations_history(
        channel=channel_id,
        limit=limit
    )

    messages = []

    for msg in response["messages"]:
        text = msg.get("text", "")
        messages.append(text)

    return messages


# TOOL 2 — SEND MESSAGE
# @mcp.tool()
# async def send_message(channel_id: str, text: str):

#     response = client.chat_postMessage(
#         channel=channel_id,
#         text=text
#     )

#     return response["message"]["text"]

@mcp.tool()
async def send_message_to_channels(channel_ids: list[str], text: str):
    """Send the same message to multiple Slack channels at once."""
    results = []
    for channel_id in channel_ids:
        try:
            response = client.chat_postMessage(
                channel=channel_id,
                text=text
            )
            results.append({
                "channel_id": channel_id,
                "status": "success",
                "message": response["message"]["text"]
            })
        except Exception as e:
            results.append({
                "channel_id": channel_id,
                "status": "error",
                "error": str(e)
            })
    return results


if __name__ == "__main__":
    print("Slack MCP Server running...")
    mcp.run()

