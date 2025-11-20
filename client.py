import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Load environment variables from project root .env if present
load_dotenv()

def to_serializable(x):
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_serializable(i) for i in x]
    # OpenAI message objects → convert to dict
    if hasattr(x, "model_dump"):
        return x.model_dump()  
    if hasattr(x, "__dict__"):
        return to_serializable(x.__dict__)
    return x

class MCPOpenAIClient:
    """High-level client orchestrating OpenAI + MCP tool calls."""

    def __init__(
        self,
        model: str = "gpt-4o",
        server_script_path: str = "server.py",
        python_executable: str = "python",
    ) -> None:
        """
        Args:
            model: OpenAI model name.
            server_script_path: Path to the MCP server script.
            python_executable: Python command used to start the MCP server.
        """
        self.model = model
        self.server_script_path = server_script_path
        self.python_executable = python_executable

        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._stdio_reader: Optional[Any] = None
        self._stdio_writer: Optional[Any] = None

        self._openai = AsyncOpenAI()

    # -------------------------------------------------------------------------
    # Lifecycle / connection management
    # -------------------------------------------------------------------------
    async def connect(self) -> None:
        """Start the MCP server and establish a client session."""
        if self._session is not None:
            return  # already connected

        self._exit_stack = AsyncExitStack()

        server_params = StdioServerParameters(
            command=self.python_executable,
            args=[self.server_script_path],
        )

        stdio_reader, stdio_writer = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self._stdio_reader = stdio_reader
        self._stdio_writer = stdio_writer

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(self._stdio_reader, self._stdio_writer)
        )

        await self._session.initialize()

    async def close(self) -> None:
        """Clean up all async resources."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._stdio_reader = None
            self._stdio_writer = None

    async def __aenter__(self) -> "MCPOpenAIClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    # MCP tool discovery
    # -------------------------------------------------------------------------
    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tools converted into OpenAI tool format."""
        self._ensure_session()
        tools_result = await self._session.list_tools()

        openai_tools: List[Dict[str, Any]] = []
        for tool in tools_result.tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
        return openai_tools

    # -------------------------------------------------------------------------
    # Main query pipeline
    # -------------------------------------------------------------------------
    async def process_query(self, query: str) -> str:
        """
        Full tool loop:
        - Keep calling the model with tool_choice="auto"
        - Execute any tool calls it requests
        - Stop only when there are no more tool calls
        """
        self._ensure_session()
        tools = await self.get_mcp_tools()

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "When tool output is present, do NOT re-run tools for the same problem_id."
                    "Use the tool response as authoritative structured data."
                    "After all tool calls are completed, you MUST produce a final answer directly."
                    "If not, then state what is the error."
                ),
            },
            {"role": "user", "content": query},
        ]

        while True:
            print("\n🧠 === MODEL INPUT MESSAGES ===")
            for i, m in enumerate(messages):
                print(f"\n--- message {i} ---")
                print(json.dumps(to_serializable(m), indent=2))
            print("================================\n")
            # Let the model decide: answer or call one/more tools
            response = await self._openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)

            # print("\n🔍 === RAW MODEL OUTPUT ===")
            # print("role:", assistant_msg.role)
            # print("content:", assistant_msg.content)
            # print("tool_calls:", assistant_msg.tool_calls)
            # print("full object:", assistant_msg)
            # print("================================\n")
 
            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")

                    print(f"[MCP] Tool requested:", tool_name, "Args:", args)

                    result = await self._session.call_tool(tool_name, arguments=args)

                    output = (
                        result.content[0].text
                        if result.content and hasattr(result.content[0], "text")
                        else ""
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    })

                # continue the loop → give LLM the tool results
                continue

            # CASE 2: no tools requested
            # we only end when there is ACTUAL content
            if assistant_msg.content and assistant_msg.content.strip():
                return assistant_msg.content

            # CASE 3: assistant returned nothing useful → ask it again explicitly
            messages.append({
                "role": "system",
                "content": "You must provide a final answer now, not call tools.",
            })


    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _ensure_session(self) -> None:
        if self._session is None:
            raise RuntimeError("MCP session is not initialized. Call connect() first.")

    async def _handle_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        assert self._session is not None

        tool_messages: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            # 🔥 **Add logging here**
            print(f"[MCP] Tool requested:", tool_name, "Args:", args)

            # Execute the tool
            result = await self._session.call_tool(tool_name, arguments=args)

            content_text = ""
            if result.content and hasattr(result.content[0], "text"):
                content_text = result.content[0].text

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": content_text,
                }
            )

        return tool_messages



async def main() -> None:
    client = MCPOpenAIClient(
        model="gpt-4o",
        server_script_path="server.py",
    )

    async with client:
        # query = "List all problems in data bank."
        # print(f"\nQuery: {query}\n")

        # response = await client.process_query(query)
        # print("Response:\n")
        # print(response)
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Shutting down.")
                break

            if not user_input:
                continue

            try:
                response = await client.process_query(user_input)
                print("\nAssistant:\n")
                print(response)
                print("\n" + "-" * 40 + "\n")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
