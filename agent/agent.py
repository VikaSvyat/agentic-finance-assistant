# Streamlit app
# ↓
# run_agent(user_goal, csv_path)
# ↓
# MCPManager connects MCP servers
# ↓
# LLM gets a list of available tools
# ↓
# LLM selects the next tool (orchestration)
# ↓
# Python calls this tool via MCP
# ↓
# result is saved in memory
# ↓
# LLM receives a short observation
# ↓
# loop repeats
# ↓
# final_answer returns monthly_report

import os
import json
import logging
from dotenv import load_dotenv

from agent.llm import call_llm
from agent.mcp_manager import MCPManager, short_json

load_dotenv()

### Observability: log each tool call
logging.basicConfig(
    filename=os.getenv("LOG_FILE", "./logs/agent.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

### LLM must choose exactly one next action. This is needed for agentic orchestration.
SYSTEM_PROMPT = """
You are an agentic financial assistant.

You must choose exactly ONE next action at a time.

CRITICAL OUTPUT RULES:
- Return ONLY ONE JSON object.
- Do NOT return a list.
- Do NOT return multiple JSON objects.
- Do NOT explain outside JSON.
- Do NOT write markdown outside JSON.
- Do NOT plan all steps at once.

WORKFLOW RULES:
- Use Finance MCP tools to analyze the statement.
- Use finance.analyze_statement for the main summary.
- Use finance.find_unusual_expenses for large expenses to review.
- Use finance.generate_savings_advice for advice.
- Before final_answer, call finance.generate_monthly_report.
- Never invent financial numbers yourself.
- Never create a final report from memory.
- Always base the final answer on tool outputs.

IMPORTANT DATA PASSING RULE:
- If you need to call finance.generate_monthly_report, you may pass empty strings as args.
- The system will automatically inject the latest outputs from:
  - finance.analyze_statement
  - finance.find_unusual_expenses
  - finance.generate_savings_advice

Valid response format:

{
  "tool": "server.tool_name",
  "args": {},
  "reason": "short reason"
}

When finished, return exactly:

{
  "tool": "final_answer",
  "args": {
    "answer": "final user-facing report"
  },
  "reason": "task completed"
}
"""

### This function turns the list of MCP tools into short text for LLM. 
# LLM doesn't know in advance what tools exist. We have to show them to her.
# The schema is shortened correctly. Sending the full schema via tokens is expensive, so the code only takes
def tools_to_text(tools: dict) -> str:
    """
    Create a compact tool description for the LLM.
    We do NOT send the full schema because it wastes tokens.
    """

    lines = []

    for name, meta in tools.items():
        description = meta.get("description", "") or ""
        schema = meta.get("input_schema", {}) or {}

        properties = schema.get("properties", {})
        arg_names = list(properties.keys())

        lines.append(
            f"Tool: {name}\n"
            f"Description: {description[:250]}\n"
            f"Args: {arg_names}\n"
        )

    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    This protects us when a small model returns extra text or several JSON objects.
    """

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)

    if not isinstance(obj, dict):
        raise ValueError("LLM response must be one JSON object")

    return obj

### It saves the full result of tool into Python's internal memory.
# Because in LLM we only send a preview 
# # and the full result remains inside Python and is then passed to the next tool.
def remember_tool_output(tool_name: str, result: str, memory: dict) -> None:
    """
    Store full tool outputs internally.
    These are NOT fully sent back to the LLM, but we can reuse them safely.
    """

    if tool_name == "finance.analyze_statement":
        memory["analysis_json"] = result

    elif tool_name == "finance.find_unusual_expenses":
        memory["unusual_json"] = result

    elif tool_name == "finance.generate_savings_advice":
        memory["advice_text"] = result

    elif tool_name == "finance.generate_monthly_report":
        memory["monthly_report"] = result

### This function fixes the arguments before calling tool.
# LLM selects a tool, but may make a mistake in its arguments.
def build_args_for_tool(tool_name: str, args: dict, csv_path: str, memory: dict) -> dict:
    """
    Fix or enrich tool args before execution.

    The LLM chooses the tool, but this function prevents common mistakes:
    - passing file paths instead of JSON outputs
    - forgetting csv_path
    - trying to generate a report without real previous tool outputs
    """

    args = dict(args or {})

    finance_csv_tools = {
        "finance.analyze_statement",
        "finance.get_category_breakdown",
        "finance.get_top_merchants",
        "finance.find_unusual_expenses",
        "finance.categorize_transactions",
    }

    if tool_name in finance_csv_tools:
        args["csv_path"] = csv_path

    if tool_name == "finance.generate_savings_advice":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")


    if tool_name == "finance.generate_monthly_report":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")
        args["advice_text"] = memory.get("advice_text", "")

    return args

def build_sqlite_create_table_args() -> dict:
    return {
        "sql": """
CREATE TABLE IF NOT EXISTS monthly_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    total_spent REAL,
    transactions_count INTEGER,
    average_transaction REAL,
    analysis_json TEXT NOT NULL,
    unusual_json TEXT NOT NULL,
    advice_text TEXT NOT NULL,
    monthly_report TEXT NOT NULL
)
"""
    }


def build_sqlite_insert_args(memory: dict) -> dict:
    analysis = json.loads(memory.get("analysis_json", "{}"))

    return {
        "table": "monthly_reports",
        "data": {
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "total_spent": analysis.get("total_spent"),
            "transactions_count": analysis.get("transactions_count"),
            "average_transaction": analysis.get("average_transaction"),
            "analysis_json": memory.get("analysis_json", ""),
            "unusual_json": memory.get("unusual_json", ""),
            "advice_text": memory.get("advice_text", ""),
            "monthly_report": memory.get("monthly_report", ""),
        },
    }

def build_sqlite_create_categories_table_args() -> dict:
    return {
        "sql": """
CREATE TABLE IF NOT EXISTS monthly_category_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    category_en TEXT NOT NULL,
    total_amount REAL,
    transactions_count INTEGER,
    average_transaction REAL
)
"""
    }

def build_sqlite_category_insert_args(memory: dict, category: dict, report_id=None) -> dict:
    return {
        "table": "monthly_category_summaries",
        "data": {
            "report_id": report_id,
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "category": category.get("category"),
            "category_en": category.get("category_en"),
            "total_amount": category.get("total_amount"),
            "transactions_count": category.get("transactions_count"),
            "average_transaction": category.get("average_transaction"),
        },
    }

### This function checks whether tool can already be called.
# LLM plans. Python controls correctness
def missing_required_memory_for_tool(tool_name: str, memory: dict) -> list:
    """
    Check whether we already have the tool outputs needed by derived tools.
    """

    missing = []

    if tool_name == "finance.generate_savings_advice":
        if not memory.get("analysis_json"):
            missing.append("finance.analyze_statement")
        if not memory.get("unusual_json"):
            missing.append("finance.find_unusual_expenses")

    if tool_name == "finance.generate_monthly_report":
        if not memory.get("analysis_json"):
            missing.append("finance.analyze_statement")
        if not memory.get("unusual_json"):
            missing.append("finance.find_unusual_expenses")
        if not memory.get("advice_text"):
            missing.append("finance.generate_savings_advice")

    return missing

### This function makes a short message to LLM after calling tool
# Why a preview and not a full result? 
# To avoid spending too many Groq tokens.
# The full result is stored in memory.
def make_observation(tool_name: str, result: str, ok: bool = True, error: str | None = None) -> dict:
    """
    Create a compact observation for the LLM.
    Full results are stored separately in memory.
    """

    if not ok:
        return {
            "tool": tool_name,
            "ok": False,
            "error": error,
        }

    return {
        "tool": tool_name,
        "ok": True,
        "output_preview": str(result)[:1200],
    }


async def run_agent(user_goal: str, csv_path: str):
    manager = MCPManager()
    steps = []
    memory = {}

    try:
        tools = await manager.connect()
        tools_text = tools_to_text(tools)

        user_goal_full = f"""
User goal:
{user_goal}

CSV path to analyze:
{csv_path}

Required final result:
- analyze the bank statement
- find large expenses to review
- generate savings advice
- generate a clean monthly report
- save the report to a markdown file if filesystem tools are available
- save monthly summary to SQLite if SQLite tools are available

Important:
The final answer must be the report returned by finance.generate_monthly_report.
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Available tools:
{tools_text}

{user_goal_full}
""",
            },
        ]

        for step_number in range(1, 11):
            llm_text = call_llm(messages)

            try:
                action = extract_json(llm_text)
            except Exception as e:
                error_answer = (
                    f"LLM returned invalid JSON.\n\n"
                    f"Error: {e}\n\n"
                    f"Raw response:\n{llm_text}"
                )

                steps.append(
                    {
                        "step": step_number,
                        "tool": "invalid_llm_response",
                        "reason": "invalid json",
                        "output": error_answer,
                    }
                )

                return {"answer": error_answer, "steps": steps}

            tool_name = action.get("tool")
            raw_args = action.get("args", {})
            reason = action.get("reason", "")

            if tool_name == "final_answer":
                if memory.get("monthly_report"):
                    answer = memory["monthly_report"]
                else:
                    answer = raw_args.get(
                        "answer",
                        "The agent finished, but no monthly report was generated.",
                    )

                steps.append(
                    {
                        "step": step_number,
                        "tool": "final_answer",
                        "reason": reason,
                        "output": answer,
                    }
                )

                return {"answer": answer, "steps": steps}

            if tool_name not in tools:
                observation = {
                    "tool": tool_name,
                    "ok": False,
                    "error": f"Unknown tool: {tool_name}",
                }

                steps.append(
                    {
                        "step": step_number,
                        "tool": tool_name,
                        "args": raw_args,
                        "reason": reason,
                        "observation": observation,
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"""
Tool observation:
{json.dumps(observation, ensure_ascii=False)}

Choose another available tool.
""",
                    }
                )

                continue

            missing = missing_required_memory_for_tool(tool_name, memory)

            if missing:
                observation = {
                    "tool": tool_name,
                    "ok": False,
                    "error": f"Missing required previous outputs. First call: {missing}",
                }

                steps.append(
                    {
                        "step": step_number,
                        "tool": tool_name,
                        "args": raw_args,
                        "reason": reason,
                        "observation": observation,
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"""
Tool observation:
{json.dumps(observation, ensure_ascii=False)}

Continue. Choose ONE missing tool first.
""",
                    }
                )

                continue

            args = build_args_for_tool(tool_name, raw_args, csv_path, memory)

            step_record = {
                "step": step_number,
                "tool": tool_name,
                "args": args,
                "reason": reason,
            }

            logging.info(
                "CALL tool=%s args=%s reason=%s",
                tool_name,
                short_json(args),
                reason,
            )

            result = None
            last_error = None

            for attempt in range(1, 4):
                try:
                    result = await manager.call_tool(tool_name, args)
                    if isinstance(result, str) and result.startswith("Error executing tool"):
                        raise RuntimeError(result)
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    logging.error(
                        "ERROR tool=%s attempt=%s error=%s",
                        tool_name,
                        attempt,
                        last_error,
                    )

            if last_error:
                observation = make_observation(
                    tool_name=tool_name,
                    result="",
                    ok=False,
                    error=last_error,
                )
            else:
                remember_tool_output(tool_name, str(result), memory)
                observation = make_observation(
                    tool_name=tool_name,
                    result=str(result),
                    ok=True,
                )

                if tool_name == "finance.generate_monthly_report":
                    steps.append(
                        {
                            "step": step_number,
                            "tool": tool_name,
                            "args": args,
                            "reason": reason,
                            "observation": observation,
                        }
                    )

                    if "sqlite.query" in tools and "sqlite.create_record" in tools:
                        try:
                            # 1. Create monthly_reports table
                            create_table_args = build_sqlite_create_table_args()
                            create_table_result = await manager.call_tool(
                                "sqlite.query",
                                create_table_args,
                            )

                            steps.append(
                                {
                                    "step": step_number + 0.1,
                                    "tool": "sqlite.query",
                                    "args": create_table_args,
                                    "reason": "Create monthly_reports table if it does not exist",
                                    "observation": make_observation(
                                        "sqlite.query",
                                        str(create_table_result),
                                        ok=True,
                                    ),
                                }
                            )

                            # 2. Save main monthly report
                            insert_args = build_sqlite_insert_args(memory)
                            insert_result = await manager.call_tool(
                                "sqlite.create_record",
                                insert_args,
                            )

                            steps.append(
                                {
                                    "step": step_number + 0.2,
                                    "tool": "sqlite.create_record",
                                    "args": insert_args,
                                    "reason": "Save monthly finance report to SQLite",
                                    "observation": make_observation(
                                        "sqlite.create_record",
                                        str(insert_result),
                                        ok=True,
                                    ),
                                }
                            )

                            # 3. Create monthly_category_summaries table
                            create_categories_table_args = build_sqlite_create_categories_table_args()
                            create_categories_table_result = await manager.call_tool(
                                "sqlite.query",
                                create_categories_table_args,
                            )

                            steps.append(
                                {
                                    "step": step_number + 0.3,
                                    "tool": "sqlite.query",
                                    "args": create_categories_table_args,
                                    "reason": "Create monthly_category_summaries table if it does not exist",
                                    "observation": make_observation(
                                        "sqlite.query",
                                        str(create_categories_table_result),
                                        ok=True,
                                    ),
                                }
                            )

                            # 4. Save category summaries
                            analysis = json.loads(memory.get("analysis_json", "{}"))
                            categories = analysis.get("top_categories", [])

                            saved_categories_count = 0

                            for category in categories:
                                category_insert_args = build_sqlite_category_insert_args(
                                    memory=memory,
                                    category=category,
                                    report_id=None,
                                )

                                await manager.call_tool(
                                    "sqlite.create_record",
                                    category_insert_args,
                                )

                                saved_categories_count += 1

                            steps.append(
                                {
                                    "step": f"{step_number}.DB",
                                    "tool": "sqlite.save_category_summaries",
                                    "reason": "Save monthly category summaries to SQLite",
                                    "observation": {
                                        "tool": "sqlite.create_record",
                                        "ok": True,
                                        "saved_categories_count": saved_categories_count,
                                    },
                                }
                            )

                        except Exception as e:
                            steps.append(
                                {
                                    "step": step_number + 0.9,
                                    "tool": "sqlite.save_failed",
                                    "reason": "SQLite persistence failed",
                                    "observation": make_observation(
                                        "sqlite",
                                        "",
                                        ok=False,
                                        error=str(e),
                                    ),
                                }
                            )

                    return {
                        "answer": memory["monthly_report"],
                        "steps": steps,
                    }

                    return {
                        "answer": memory["monthly_report"],
                        "steps": steps,
                    }

            logging.info(
                "RESULT tool=%s output=%s",
                tool_name,
                short_json(observation),
            )

            step_record["observation"] = observation
            steps.append(step_record)

            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action, ensure_ascii=False),
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": f"""
Tool observation:
{json.dumps(observation, ensure_ascii=False)}

Internal state:
- analysis_json available: {bool(memory.get("analysis_json"))}
- unusual_json available: {bool(memory.get("unusual_json"))}
- advice_text available: {bool(memory.get("advice_text"))}
- monthly_report available: {bool(memory.get("monthly_report"))}

Continue. Choose exactly ONE next tool or final_answer.
""",
                }
            )

        if memory.get("monthly_report"):
            return {
                "answer": memory["monthly_report"],
                "steps": steps,
            }

        return {
            "answer": "Agent stopped after max steps. Check logs/agent.log.",
            "steps": steps,
        }

    finally:
        await manager.close() #Closing MCP connections