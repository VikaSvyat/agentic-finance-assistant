### developer utility


import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from agent.mcp_manager import MCPManager


async def main():
    manager = MCPManager()

    try:
        tools = await manager.connect()

        print("\nSQLite MCP tools:\n")

        for name in sorted(tools.keys()):
            if name.startswith("sqlite."):
                print("=" * 80)
                print(name)
                print("Description:")
                print(tools[name]["description"])
                print("Input schema:")
                print(json.dumps(tools[name]["input_schema"], indent=2, ensure_ascii=False))

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())