import asyncio
import json
import os
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

    while True:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages, #history
            tools=groq_tools,  #list of tools
            tool_choice="auto", # checks whether tool is needed or not
            max_tokens=1024,
        )

        msg = response.choices[0].message  #extract ai response
        tool_calls = msg.tool_calls or []  # decision is stored

        # ✅ ONLY print raw when tool is called
        if tool_calls:
            print("\n🧠 AI RAW RESPONSE:")
            print(msg)

            print("\n🛠️ AI decided to call tool(s):")

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

                print(f"\n➡️ Tool Name: {tool_name}")
                print(f"📥 Input: {tool_input}")

                output = await call_mcp_tool(session, tool_name, tool_input)

                print(f"📤 Output:\n{output}")

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
                        "You are a concise Slack assistant. "
                        "Use tools when needed. "
                        "Keep replies short and clear. "
                        "Extract channel IDs directly from the user message."
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




