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
import time
import re
from pathlib import Path
from dotenv import load_dotenv

from agent.config import (
    FAST_DEV_SKIP_SQLITE,
    FAST_DEV_MODE,
    MAX_AGENT_STEPS,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_OUTPUT_LENGTH,
)
from agent.llm import call_llm
from agent.mcp_manager import MCPManager, short_json

load_dotenv()

### Observability: log each tool call
logging.basicConfig(
    filename=os.getenv("LOG_FILE", "./logs/agent.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

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
- If the user provides a statements directory, first use filesystem tools to inspect the directory.
- Use filesystem.list_directory or a similar filesystem tool to discover CSV files.
- Then call finance.merge_statements to merge the discovered CSV files into one combined CSV.
- After finance.merge_statements, use the existing single-file Finance MCP tools on the merged CSV.
- Use finance.analyze_statement for the main summary.
- Use finance.find_unusual_expenses for large expenses to review.
- Use finance.generate_savings_advice for advice.
- Before final_answer, call finance.generate_monthly_report.
- After finance.generate_monthly_report, if SQLite tools are available, call finance.prepare_monthly_report_record.
- Then call sqlite.create_record to save the prepared monthly report record.
- Only after saving, return final_answer.
- Never invent financial numbers yourself.
- Never create a final report from memory.
- Always base the final answer on tool outputs.

IMPORTANT DATA PASSING RULE:
- If you need to call finance.merge_statements, you may pass an empty list as csv_files.
- The system will automatically inject CSV files discovered from filesystem tools.
- If you need to call finance.analyze_statement or finance.find_unusual_expenses after merging, you may pass an empty csv_path.
- The system will automatically inject the merged CSV path returned by finance.merge_statements.
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
  "args": {},
  "reason": "task completed"
}

IMPORTANT FINAL ANSWER RULE:
- Do NOT put the monthly report inside the final_answer JSON.
- Do NOT include markdown report text in args.answer.
- The system already has the full monthly report in memory.
- final_answer is only a signal that the workflow is complete.
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
            f"Description: {description[:80]}\n"
            f"Args: {arg_names}\n"
        )

    return "\n".join(lines)


def tools_for_current_mode(tools: dict, statements_dir: str) -> dict:
    """
    Keep directory-only tools out of the LLM prompt in single-file mode.
    The MCP servers still exist; this only reduces confusion and prompt size.
    """

    filtered = {
        name: meta
        for name, meta in tools.items()
        if name != "sqlite.db_info"
    }

    if FAST_DEV_SKIP_SQLITE:
        filtered = {
            name: meta
            for name, meta in filtered.items()
            if not name.startswith("sqlite.")
            and name != "finance.prepare_monthly_report_record"
        }

    if statements_dir:
        return filtered

    return {
        name: meta
        for name, meta in filtered.items()
        if not name.startswith("filesystem.")
        and name != "finance.merge_statements"
    }


def disallowed_tool_for_mode(tool_name: str, statements_dir: str) -> str:
    if tool_name == "sqlite.db_info":
        return (
            "Do not call sqlite.db_info. If saving is required, call "
            "finance.prepare_monthly_report_record and then sqlite.create_record."
        )

    if FAST_DEV_SKIP_SQLITE and (
        tool_name.startswith("sqlite.")
        or tool_name == "finance.prepare_monthly_report_record"
    ):
        return (
            "FAST_DEV_SKIP_SQLITE is active. Do not save to SQLite. "
            "Return final_answer after finance.generate_monthly_report."
        )

    if statements_dir:
        return ""

    if tool_name.startswith("filesystem."):
        return (
            "Single CSV mode is active. Do not inspect directories. "
            "Use finance.analyze_statement on the provided csv_path."
        )

    if tool_name == "finance.merge_statements":
        return (
            "Single CSV mode is active. Do not merge statements. "
            "Use finance.analyze_statement on the provided csv_path."
        )

    return ""


def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    This protects us when a small model returns extra text or several JSON objects.
    """

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
            if not tool_match:
                raise
            return {
                "tool": tool_match.group(1),
                "args": {},
                "reason": "recovered malformed JSON tool call",
            }

    if not isinstance(obj, dict):
        raise ValueError("LLM response must be one JSON object")

    return obj


def action_for_history(action: dict) -> dict:
    """
    Keep conversation history small.

    Python injects stored tool outputs into dependent tools, so the LLM should
    not carry large JSON strings forward in assistant messages.
    """

    tool_name = action.get("tool")
    args = action.get("args", {})

    memory_injected_tools = {
        "finance.generate_savings_advice",
        "finance.generate_monthly_report",
        "finance.prepare_monthly_report_record",
        "sqlite.create_record",
    }

    if tool_name in memory_injected_tools:
        args = {}

    return {
        "tool": tool_name,
        "args": args if isinstance(args, dict) else {},
        "reason": action.get("reason", ""),
    }


def truncate_text(value: str, limit: int = MAX_TOOL_OUTPUT_LENGTH) -> str:
    """
    Keep observations compact for LLM context and logs.
    Full tool outputs are still kept in internal memory when needed.
    """

    text = str(value)

    if len(text) <= limit:
        return text

    omitted = len(text) - limit
    return f"{text[:limit]}\n... [truncated {omitted} chars]"


def prune_messages(messages: list[dict]) -> list[dict]:
    """
    Keep the initial system/user context and only the latest runtime messages.
    This prevents prompt growth during long or error-prone agent loops.
    """

    if len(messages) <= 2 + MAX_HISTORY_MESSAGES:
        return messages

    return messages[:2] + messages[-MAX_HISTORY_MESSAGES:]


def memory_state_text(memory: dict, csv_path: str) -> str:
    return f"""
Internal state:
- analysis_json available: {bool(memory.get("analysis_json"))}
- unusual_json available: {bool(memory.get("unusual_json"))}
- advice_text available: {bool(memory.get("advice_text"))}
- monthly_report available: {bool(memory.get("monthly_report"))}
- monthly_report_record available: {bool(memory.get("monthly_report_record"))}
- discovered csv_files count: {len(memory.get("csv_files", []))}
- effective csv_path available: {bool(memory.get("csv_path") or csv_path)}
"""

### It saves the full result of tool into Python's internal memory.
# Because in LLM we only send a preview 
# # and the full result remains inside Python and is then passed to the next tool.

def extract_csv_paths_from_filesystem_output(result: str, statements_dir: str = "") -> list[str]:
    """
    Extract CSV file paths from common filesystem MCP outputs.

    Supports:
    - JSON lists
    - JSON objects with entries/files/items
    - dict items with path/name/file/filename
    - plain text directory listings
    """

    csv_files = []

    def normalize_path(value: str) -> str:
        value = str(value).strip().strip('"').strip("'")

        if not value.lower().endswith(".csv"):
            return ""

        # already absolute path
        if value.startswith("/"):
            return value

        # already contains directory
        if "/" in value:
            return value

        # discovered file name only
        if statements_dir:
            return str(Path(statements_dir) / value)

        # fallback for current project structure
        return str(Path("data") / value)

    try:
        data = json.loads(result)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("entries") or data.get("files") or data.get("items") or []
        else:
            items = []

        for item in items:
            if isinstance(item, str):
                path = normalize_path(item)
                if path:
                    csv_files.append(path)
            elif isinstance(item, dict):
                candidate = (
                    item.get("path")
                    or item.get("name")
                    or item.get("file")
                    or item.get("filename")
                )
                if candidate:
                    path = normalize_path(candidate)
                    if path:
                        csv_files.append(path)

    except Exception:
        for line in str(result).splitlines():
            line = line.strip()

            # Filesystem MCP often returns lines like:
            # [FILE] bank_statement_may.csv
            # [DIR] statements
            if line.startswith("[DIR]"):
                continue

            if line.startswith("[FILE]"):
                line = line.replace("[FILE]", "", 1).strip()

            line = line.strip("-").strip()

            path = normalize_path(line)

            if path:
                csv_files.append(path)

    return list(dict.fromkeys(csv_files))


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

    elif tool_name == "finance.prepare_monthly_report_record":
        memory["monthly_report_record"] = result

    elif tool_name == "finance.merge_statements":
        try:
            data = json.loads(result)
            merged_path = data.get("merged_csv_path") or data.get("csv_path")
            if merged_path:
                memory["csv_path"] = merged_path
                memory["merged_csv_path"] = merged_path
        except Exception:
            pass

    elif tool_name.startswith("filesystem."):
        memory.setdefault("filesystem_outputs", []).append(result)
        discovered = extract_csv_paths_from_filesystem_output(
            result,
            memory.get("statements_dir", ""),
        )
        if discovered:
            existing = memory.get("csv_files", [])
            memory["csv_files"] = list(dict.fromkeys(existing + discovered))

def normalize_args(args) -> dict:
    """
    Ensure tool args are always a dictionary.
    The LLM may accidentally return a list or a string.
    """
    if args is None:
        return {}

    if isinstance(args, dict):
        return args

    if isinstance(args, list):
        return {"csv_files": args}

    if isinstance(args, str):
        return {"path": args}

    return {}

### This function fixes the arguments before calling tool.
# LLM selects a tool, but may make a mistake in its arguments.

def build_args_for_tool(tool_name: str, args: dict, csv_path: str, memory: dict) -> dict:
    """
    Fix or enrich tool args before execution.
    """

    args = normalize_args(args)

    effective_csv_path = memory.get("csv_path") or csv_path

    finance_csv_tools = {
        "finance.analyze_statement",
        "finance.get_category_breakdown",
        "finance.get_top_merchants",
        "finance.find_unusual_expenses",
        "finance.categorize_transactions",
    }

    if tool_name in finance_csv_tools:
        args["csv_path"] = effective_csv_path

    if tool_name in {
        "filesystem.list_directory",
        "filesystem.list_directory_with_sizes",
    }:
        path = memory.get("statements_dir") or args.get("path")

        if not path:
            if args.get("csv_files") and isinstance(args["csv_files"], list):
                path = args["csv_files"][0]
            elif csv_path:
                path = str(Path(csv_path).parent)
            else:
                path = "data"

        args = {"path": path}

    if tool_name == "finance.merge_statements":
        if not args.get("csv_files"):
            args["csv_files"] = memory.get("csv_files", [])

        args["output_csv_path"] = str(Path("data") / "merged_statement.csv")

    if tool_name == "finance.generate_savings_advice":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")

    if tool_name == "finance.generate_monthly_report":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")
        args["advice_text"] = memory.get("advice_text", "")

    if tool_name == "finance.prepare_monthly_report_record":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")
        args["advice_text"] = memory.get("advice_text", "")
        args["monthly_report"] = memory.get("monthly_report", "")

    if tool_name == "sqlite.create_record" and memory.get("monthly_report_record"):
        record = json.loads(memory["monthly_report_record"])
        args["table"] = record["table"]
        args["data"] = record["data"]

    return args

### This function checks whether tool can already be called.
# LLM plans. Python controls correctness
def missing_required_memory_for_tool(tool_name: str, memory: dict) -> list:
    """
    Check whether the selected tool already has the outputs it needs.
    The LLM still decides the next step; Python only validates dependencies.
    """

    missing = []

    finance_csv_tools = {
        "finance.analyze_statement",
        "finance.get_category_breakdown",
        "finance.get_top_merchants",
        "finance.find_unusual_expenses",
        "finance.categorize_transactions",
    }

    if tool_name == "finance.merge_statements":
        if not memory.get("csv_files"):
            missing.append("filesystem.list_directory")

    if tool_name in finance_csv_tools:
        if memory.get("statements_dir") and not memory.get("csv_path"):
            missing.append("finance.merge_statements")

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

    if tool_name == "finance.prepare_monthly_report_record":
        if not memory.get("analysis_json"):
            missing.append("finance.analyze_statement")
        if not memory.get("unusual_json"):
            missing.append("finance.find_unusual_expenses")
        if not memory.get("advice_text"):
            missing.append("finance.generate_savings_advice")
        if not memory.get("monthly_report"):
            missing.append("finance.generate_monthly_report")

    if tool_name == "sqlite.create_record":
        if not memory.get("monthly_report_record"):
            missing.append("finance.prepare_monthly_report_record")

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
        "output_preview": truncate_text(result),
        "full_output": str(result),
    }


async def run_agent(user_goal: str, csv_path: str = "", statements_dir: str = ""):
    manager = MCPManager()
    steps = []
    memory = {}

    logger.info(
        "AGENT_START mode=%s csv_path=%s statements_dir=%s fast_dev_mode=%s max_steps=%s max_tool_output_length=%s max_history_messages=%s",
        "directory" if statements_dir else "single_csv",
        csv_path,
        statements_dir,
        FAST_DEV_MODE,
        MAX_AGENT_STEPS,
        MAX_TOOL_OUTPUT_LENGTH,
        MAX_HISTORY_MESSAGES,
    )

    if csv_path:
        memory["csv_path"] = csv_path

    if statements_dir:
        memory["statements_dir"] = statements_dir

    try:
        tools = await manager.connect()
        visible_tools = tools_for_current_mode(tools, statements_dir)
        tools_text = tools_to_text(visible_tools)
        persistence_instruction = (
            "- FAST_DEV_SKIP_SQLITE is active: do not save to SQLite. Return final_answer after finance.generate_monthly_report."
            if FAST_DEV_SKIP_SQLITE
            else "- Save monthly summary to SQLite by calling finance.prepare_monthly_report_record and then sqlite.create_record. Do not call sqlite.db_info."
        )

        if statements_dir:
            required_workflow = f"""
Required final result:
- First discover CSV files using filesystem tools.
- Then call finance.merge_statements to create one merged CSV.
- Analyze the merged CSV using the existing finance.analyze_statement tool.
- Find large expenses to review.
- Generate savings advice.
- Generate a clean monthly report.
{persistence_instruction}
"""
        else:
            required_workflow = f"""
Required final result:
- Analyze only the provided CSV path using finance.analyze_statement.
- Do not inspect directories.
- Do not merge statements.
- Find large expenses to review.
- Generate savings advice.
- Generate a clean monthly report.
{persistence_instruction}
"""

        user_goal_full = f"""
User goal:
{user_goal}

CSV path to analyze:
{csv_path}

Statements directory to analyze:
{statements_dir}

{required_workflow}

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

        for step_number in range(1, MAX_AGENT_STEPS + 1):
            step_start = time.perf_counter()
            llm_start = time.perf_counter()
            llm_text = call_llm(messages)
            llm_duration = time.perf_counter() - llm_start

            try:
                action = extract_json(llm_text)
            except Exception as e:
                total_duration = time.perf_counter() - step_start
                error_answer = (
                    f"LLM returned invalid JSON.\n\n"
                    f"Error: {e}\n\n"
                    f"Raw response:\n{truncate_text(llm_text)}"
                )

                steps.append(
                    {
                        "step": step_number,
                        "tool": "invalid_llm_response",
                        "reason": "invalid json",
                        "output": error_answer,
                        "timing": {
                            "llm_response_seconds": round(llm_duration, 3),
                            "tool_execution_seconds": 0.0,
                            "total_step_seconds": round(total_duration, 3),
                        },
                    }
                )

                logger.error(
                    "INVALID_LLM_JSON step=%s error=%s raw=%s",
                    step_number,
                    str(e),
                    truncate_text(llm_text),
                )

                logger.info(
                    "TIMING step=%s tool=invalid_llm_response llm=%.3fs tool=0.000s total=%.3fs",
                    step_number,
                    llm_duration,
                    total_duration,
                )

                if step_number >= MAX_AGENT_STEPS:
                    return {"answer": error_answer, "steps": steps}

                messages.append(
                    {
                        "role": "user",
                        "content": f"""
Your previous response was invalid JSON.

Error:
{str(e)}

Return exactly ONE JSON object in this format:
{{"tool": "server.tool_name", "args": {{}}, "reason": "short reason"}}

{memory_state_text(memory, csv_path)}

Continue. Choose exactly ONE next tool or final_answer.
""",
                    }
                )
                messages = prune_messages(messages)

                continue

            tool_name = action.get("tool")
            raw_args = action.get("args", {})
            reason = action.get("reason", "")

            if tool_name == "final_answer":
                total_duration = time.perf_counter() - step_start
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
                        "timing": {
                            "llm_response_seconds": round(llm_duration, 3),
                            "tool_execution_seconds": 0.0,
                            "total_step_seconds": round(total_duration, 3),
                        },
                    }
                )

                logger.info(
                    "TIMING step=%s tool=final_answer llm=%.3fs tool=0.000s total=%.3fs",
                    step_number,
                    llm_duration,
                    total_duration,
                )

                return {"answer": answer, "steps": steps}

            if tool_name not in tools:
                total_duration = time.perf_counter() - step_start
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
                        "timing": {
                            "llm_response_seconds": round(llm_duration, 3),
                            "tool_execution_seconds": 0.0,
                            "total_step_seconds": round(total_duration, 3),
                        },
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action_for_history(action), ensure_ascii=False),
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
                messages = prune_messages(messages)

                logger.info(
                    "TIMING step=%s tool=%s llm=%.3fs tool=0.000s total=%.3fs",
                    step_number,
                    tool_name,
                    llm_duration,
                    total_duration,
                )

                continue

            mode_error = disallowed_tool_for_mode(tool_name, statements_dir)

            if mode_error:
                total_duration = time.perf_counter() - step_start
                observation = {
                    "tool": tool_name,
                    "ok": False,
                    "error": mode_error,
                }

                steps.append(
                    {
                        "step": step_number,
                        "tool": tool_name,
                        "args": raw_args,
                        "reason": reason,
                        "observation": observation,
                        "timing": {
                            "llm_response_seconds": round(llm_duration, 3),
                            "tool_execution_seconds": 0.0,
                            "total_step_seconds": round(total_duration, 3),
                        },
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action_for_history(action), ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"""
Tool observation:
{json.dumps(observation, ensure_ascii=False)}

Continue in Single CSV mode. Choose finance.analyze_statement for the provided csv_path.
""",
                    }
                )
                messages = prune_messages(messages)

                logger.info(
                    "TIMING step=%s tool=%s llm=%.3fs tool=0.000s total=%.3fs",
                    step_number,
                    tool_name,
                    llm_duration,
                    total_duration,
                )

                continue

            missing = missing_required_memory_for_tool(tool_name, memory)

            if missing:
                total_duration = time.perf_counter() - step_start
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
                        "timing": {
                            "llm_response_seconds": round(llm_duration, 3),
                            "tool_execution_seconds": 0.0,
                            "total_step_seconds": round(total_duration, 3),
                        },
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action_for_history(action), ensure_ascii=False),
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
                messages = prune_messages(messages)

                logger.info(
                    "TIMING step=%s tool=%s llm=%.3fs tool=0.000s total=%.3fs",
                    step_number,
                    tool_name,
                    llm_duration,
                    total_duration,
                )

                continue

            args = build_args_for_tool(tool_name, raw_args, csv_path, memory)

            step_record = {
                "step": step_number,
                "tool": tool_name,
                "args": args,
                "reason": reason,
            }

            logger.info(
                "CALL tool=%s args=%s reason=%s",
                tool_name,
                short_json(args),
                reason,
            )

            result = None
            last_error = None
            tool_start = time.perf_counter()

            for attempt in range(1, 4):
                try:
                    result = await manager.call_tool(tool_name, args)
                    if isinstance(result, str) and result.startswith("Error executing tool"):
                        raise RuntimeError(result)
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.error(
                        "ERROR tool=%s attempt=%s error=%s",
                        tool_name,
                        attempt,
                        last_error,
                    )

            tool_duration = time.perf_counter() - tool_start

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

            llm_observation = {
                "tool": observation.get("tool"),
                "ok": observation.get("ok"),
                "output_preview": observation.get("output_preview", ""),
                "error": observation.get("error", ""),
            }

            total_duration = time.perf_counter() - step_start

            logger.info(
                "RESULT tool=%s output=%s",
                tool_name,
                short_json(observation),
            )

            logger.info(
                "TIMING step=%s tool=%s llm=%.3fs tool=%.3fs total=%.3fs",
                step_number,
                tool_name,
                llm_duration,
                tool_duration,
                total_duration,
            )

            step_record["timing"] = {
                "llm_response_seconds": round(llm_duration, 3),
                "tool_execution_seconds": round(tool_duration, 3),
                "total_step_seconds": round(total_duration, 3),
            }
            step_record["observation"] = observation
            steps.append(step_record)

            if (
                FAST_DEV_SKIP_SQLITE
                and tool_name == "finance.generate_monthly_report"
                and memory.get("monthly_report")
            ):
                steps.append(
                    {
                        "step": step_number + 1,
                        "tool": "final_answer",
                        "reason": "FAST_DEV_SKIP_SQLITE completed after report generation",
                        "output": memory["monthly_report"],
                        "timing": {
                            "llm_response_seconds": 0.0,
                            "tool_execution_seconds": 0.0,
                            "total_step_seconds": 0.0,
                        },
                    }
                )

                logger.info(
                    "FAST_DEV_SKIP_SQLITE returning final_answer after finance.generate_monthly_report"
                )

                return {"answer": memory["monthly_report"], "steps": steps}

            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action_for_history(action), ensure_ascii=False),
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": f"""
Tool observation:
{json.dumps(llm_observation, ensure_ascii=False)}

{memory_state_text(memory, csv_path)}

Continue. Choose exactly ONE next tool or final_answer.
""",
                }
            )
            messages = prune_messages(messages)

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
