import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── helpers ──────────────────────────────────────────────────────────────────

def mcp_tools_to_groq_tools(mcp_tools) -> list[dict]:
    tools = []
    for tool in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return tools


async def call_mcp_tool(session: ClientSession, tool_name: str, tool_input: dict):
    result = await session.call_tool(tool_name, tool_input)

    texts = []
    for block in result.content:
        if hasattr(block, "text") and block.text:
            texts.append(block.text)

    return "\n".join(texts) if texts else "No response"


# ── AI LOOP ──────────────────────────────────────────────────────────────────

async def run_agent_turn(session, groq_tools, messages):
    retries = 0
    while True:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages, #history
                tools=groq_tools,  #list of tools
                tool_choice="auto", # checks whether tool is needed or not
                max_tokens=1024,
            )
        except Exception as e:
            if "tool_use_failed" in str(e) and retries < 3:
                retries += 1
                messages.append({
                    "role": "user",
                    "content": "You generated an invalid tool call format. Please generate your response again using native valid JSON tool arguments without any raw <function> tags."
                })
                # Silent retry in the background
                continue
            else:
                print(f"\n❌ AI API Formatting Error: {e}")
                print("🤖 (The AI hallucinated a function call and halted. Please try your prompt again.)\n")
                break

        msg = response.choices[0].message  #extract ai response
        tool_calls = msg.tool_calls or []  # decision is stored

        if tool_calls:
            # We silently process tool calls now without dumping the raw JSON to the screen
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name, 
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            for tool_call in tool_calls:
                tool_name = tool_call.function.name  #extract tool name
                tool_input = json.loads(tool_call.function.arguments)

                # Fetching tool result silently
                output = await call_mcp_tool(session, tool_name, tool_input)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })

        else:
            # ✅ FINAL RESPONSE ONLY ONCE
            reply = msg.content or ""

            print("\n💬 AI Final Reply:")
            print(f"🤖 {reply}\n")

            messages.append({
                "role": "assistant",
                "content": reply
            })

            break


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("   Slack AI Assistant  (Groq + LLaMA 3.3)")
    print("=" * 55)
    print("Examples:  read last 5 messages from C08XXXXXX")
    print("           send 'Hello!' to C08XXXXXX")
    print("           summarize messages from C08XXXXXX")
    print("Type 'exit' to quit.\n")

    try:
        with open("d:/slack-mcp/channels.json", "r", encoding="utf-8") as f:
            channels = json.load(f)
            channel_context_str = ", ".join([f"Name: {k} -> ID: {v}" for k, v in channels.items()])
    except FileNotFoundError:
        channel_context_str = "No channel mapping found."

    server = StdioServerParameters(
        command="d:/slack-mcp/venv/Scripts/python.exe",
        args=["d:/slack-mcp/server.py"],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools_response = await session.list_tools()
            groq_tools = mcp_tools_to_groq_tools(mcp_tools_response.tools)

            print(f"✅ Ready — tools: {[t['function']['name'] for t in groq_tools]}\n")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Slack knowledge assistant. "
                        f"AVAILABLE CHANNELS: {channel_context_str}. "
                        "When the user mentions a channel by its Name, you MUST use its corresponding ID for tool calls.\n"
                        "When the user asks about a problem or needs help finding information, "
                        "Use the semantic_search tool to find relevant past discussions from Slack channels. "
                        "Synthesize the search results into a clear, comprehensive, actionable answer. "
                        "IMPORTANT: If you find multiple different solutions or answers discussed across different channels, you MUST mention all of them. Do not just summarize the first one you read. "
                        "IMPORTANT: You must ONLY answer the user's specific question. Discard and ignore any search results or context that are unrelated to the specific topic requested. "
                        "Always cite the channel name and relevant context from the results. "
                        "Use read_messages and send_message tools for direct channel operations. "
                        "IMPORTANT: If the user asks you to read, list, or give messages from a channel, you MUST display the actual messages verbatim. Do not suppress or summarize them into a single sentence. "
                        "Extract channel IDs directly from the user message or the AVAILABLE CHANNELS list when provided. "
                        "NOTE: Messages may contain '[Image Description: ...]' tags; these are generated by vision AI to describe image content. "
                        "Keep replies short and clear."
                    ),
                }
            ]

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("👋 Goodbye!")
                    break

                messages.append({
                    "role": "user",
                    "content": user_input
                })

                await run_agent_turn(session, groq_tools, messages)


if __name__ == "__main__":
    asyncio.run(main())




