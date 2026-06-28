import os
import sys
import json
import asyncio
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from agent.runtime_config import PROJECT_ROOT, load_runtime_config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_runtime_config()


class MCPManager:
    def __init__(self):
        self.stack = AsyncExitStack()
        self.sessions = {}
        self.tools = {}

    async def connect(self):
        data_dir = os.getenv("DATA_DIR", "./data")
        db_path = os.getenv("DB_PATH", "./database/finance.db")
        finance_server_path = PROJECT_ROOT / "servers" / "finance_server.py"

        servers = {
            "filesystem": StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    data_dir,
                ],
            ),
            "sqlite": StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "mcp-sqlite",
                    db_path,
                ],
            ),
            "finance": StdioServerParameters(
                command=sys.executable,
                args=[
                    str(finance_server_path),
                ],
            ),
        }

        for server_name, params in servers.items():
            read, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            self.sessions[server_name] = session

            tools_result = await session.list_tools()

            for tool in tools_result.tools:
                full_name = f"{server_name}.{tool.name}"
                self.tools[full_name] = {
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }

        return self.tools

    async def call_tool(self, full_tool_name: str, args: dict):
        if full_tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {full_tool_name}")

        meta = self.tools[full_tool_name]
        session = self.sessions[meta["server"]]

        result = await session.call_tool(meta["name"], arguments=args)

        output_parts = []

        for item in result.content:
            if hasattr(item, "text"):
                output_parts.append(item.text)
            else:
                output_parts.append(str(item))

        return "\n".join(output_parts)

    async def close(self):
        await self.stack.aclose()


def short_json(obj, limit=300):
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text[:limit]
