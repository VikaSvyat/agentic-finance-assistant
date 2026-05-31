import os
import json
import logging
from dotenv import load_dotenv

from agent.llm import call_llm
from agent.mcp_manager import MCPManager, short_json

load_dotenv()

logging.basicConfig(
    filename=os.getenv("LOG_FILE", "./logs/agent.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

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
        await manager.close()