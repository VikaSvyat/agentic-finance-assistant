import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


CSV_PATH = "data/bank_statement_may.csv"


async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=["servers/finance_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            print("Available tools:")
            for tool in tools.tools:
                print("-", tool.name)

            print("\n0. categorize_transactions")
            categorized = await session.call_tool(
                "categorize_transactions",
                arguments={"csv_path": CSV_PATH},
            )
            print(categorized.content[0].text[:2000])

            print("\n1. analyze_statement")
            analysis = await session.call_tool(
                "analyze_statement",
                arguments={"csv_path": CSV_PATH},
            )
            analysis_text = analysis.content[0].text
            print(analysis_text)

            print("\n1.1 get_category_breakdown")
            categories = await session.call_tool(
                "get_category_breakdown",
                arguments={"csv_path": CSV_PATH},
            )
            print(categories.content[0].text)

            print("\n1.2 get_top_merchants")
            merchants = await session.call_tool(
                "get_top_merchants",
                arguments={"csv_path": CSV_PATH, "limit": 10},
            )
            print(merchants.content[0].text)

            print("\n2. find_unusual_expenses")
            unusual = await session.call_tool(
                "find_unusual_expenses",
                arguments={"csv_path": CSV_PATH},
            )
            unusual_text = unusual.content[0].text
            print(unusual_text)

            print("\n3. generate_savings_advice")
            advice = await session.call_tool(
                "generate_savings_advice",
                arguments={
                    "analysis_json": analysis_text,
                    "unusual_json": unusual_text,
                },
            )
            print(advice.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())