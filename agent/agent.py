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
from typing import Any, TypeAlias
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

Message: TypeAlias = dict[str, str]
Memory: TypeAlias = dict[str, Any]
Step: TypeAlias = dict[str, Any]
ToolResults: TypeAlias = dict[str, str]
AgentResult: TypeAlias = dict[str, Any]

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
- Use finance.prepare_financial_insight_context to prepare structured insight context.
- Use finance.generate_ai_financial_insights to generate personalized observations and recommendations.
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
- If you need to call finance.prepare_financial_insight_context, you may pass empty strings as args.
- If you need to call finance.generate_ai_financial_insights, you may pass empty strings as args.
- If you need to call finance.generate_monthly_report, you may pass empty strings as args.
- The system will automatically inject the latest outputs from:
  - finance.analyze_statement
  - finance.find_unusual_expenses
  - finance.prepare_financial_insight_context
  - finance.generate_ai_financial_insights

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
def tools_to_text(tools: dict[str, Any]) -> str:
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


def add_virtual_agent_tools(tools: dict[str, Any]) -> dict[str, Any]:
    tools = dict(tools)
    tools["finance.generate_ai_financial_insights"] = {
        "description": (
            "Generate AI-powered financial observations and recommendations "
            "from prepared deterministic insight context."
        ),
        "input_schema": {
            "properties": {
                "financial_context_json": {"type": "string"},
                "analysis_json": {"type": "string"},
                "unusual_json": {"type": "string"},
            }
        },
    }
    return tools


def tools_for_current_mode(tools: dict[str, Any], statements_dir: str) -> dict[str, Any]:
    """
    Keep directory-only tools out of the LLM prompt in single-file mode.
    The MCP servers still exist; this only reduces confusion and prompt size.
    """

    filtered = {
        name: meta
        for name, meta in tools.items()
        if name != "sqlite.db_info"
        and name != "finance.generate_savings_advice"
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


def extract_json(text: str) -> dict[str, Any]:
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


def action_for_history(action: dict[str, Any]) -> dict[str, Any]:
    """
    Keep conversation history small.

    Python injects stored tool outputs into dependent tools, so the LLM should
    not carry large JSON strings forward in assistant messages.
    """

    tool_name = action.get("tool")
    args = action.get("args", {})

    memory_injected_tools = {
        "finance.generate_savings_advice",
        "finance.prepare_financial_insight_context",
        "finance.generate_ai_financial_insights",
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


def prune_messages(messages: list[Message]) -> list[Message]:
    """
    Keep the initial system/user context and only the latest runtime messages.
    This prevents prompt growth during long or error-prone agent loops.
    """

    if len(messages) <= 2 + MAX_HISTORY_MESSAGES:
        return messages

    return messages[:2] + messages[-MAX_HISTORY_MESSAGES:]


def memory_state_text(memory: Memory, csv_path: str) -> str:
    return f"""
Internal state:
- analysis_json available: {bool(memory.get("analysis_json"))}
- unusual_json available: {bool(memory.get("unusual_json"))}
- advice_text available: {bool(memory.get("advice_text"))}
- financial_insight_context available: {bool(memory.get("financial_insight_context"))}
- ai_insights_text available: {bool(memory.get("ai_insights_text"))}
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


def remember_tool_output(tool_name: str, result: str, memory: Memory) -> None:
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

    elif tool_name == "finance.prepare_financial_insight_context":
        memory["financial_insight_context"] = result

    elif tool_name == "finance.generate_ai_financial_insights":
        memory["ai_insights_text"] = result
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

def normalize_args(args: Any) -> dict[str, Any]:
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

def build_args_for_tool(
    tool_name: str,
    args: dict[str, Any],
    csv_path: str,
    memory: Memory,
) -> dict[str, Any]:
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
        discovered_csv_files = memory.get("csv_files", [])

        if discovered_csv_files:
            args["csv_files"] = discovered_csv_files
        elif not args.get("csv_files"):
            args["csv_files"] = []

        args["output_csv_path"] = str(Path("data") / "merged_statement.csv")

    if tool_name == "finance.generate_savings_advice":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")

    if tool_name == "finance.prepare_financial_insight_context":
        args["analysis_json"] = memory.get("analysis_json", "")
        args["unusual_json"] = memory.get("unusual_json", "")

    if tool_name == "finance.generate_ai_financial_insights":
        args["financial_context_json"] = memory.get("financial_insight_context", "")
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
def missing_required_memory_for_tool(tool_name: str, memory: Memory) -> list[str]:
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

    if tool_name == "finance.prepare_financial_insight_context":
        if not memory.get("analysis_json"):
            missing.append("finance.analyze_statement")
        if not memory.get("unusual_json"):
            missing.append("finance.find_unusual_expenses")

    if tool_name == "finance.generate_ai_financial_insights":
        if not memory.get("financial_insight_context"):
            missing.append("finance.prepare_financial_insight_context")

    if tool_name == "finance.generate_monthly_report":
        if not memory.get("analysis_json"):
            missing.append("finance.analyze_statement")
        if not memory.get("unusual_json"):
            missing.append("finance.find_unusual_expenses")
        if not memory.get("advice_text"):
            missing.append("finance.generate_ai_financial_insights")

    if tool_name == "finance.prepare_monthly_report_record":
        if not memory.get("analysis_json"):
            missing.append("finance.analyze_statement")
        if not memory.get("unusual_json"):
            missing.append("finance.find_unusual_expenses")
        if not memory.get("advice_text"):
            missing.append("finance.generate_ai_financial_insights")
        if not memory.get("monthly_report"):
            missing.append("finance.generate_monthly_report")

    if tool_name == "sqlite.create_record":
        if not memory.get("monthly_report_record"):
            missing.append("finance.prepare_monthly_report_record")

    return missing


def next_required_tool(memory: Memory) -> str:
    """
    Return the next missing workflow tool from internal state.

    This is used only to avoid repeating already-completed prerequisite tools
    when a small local model loses track of progress.
    """

    if memory.get("statements_dir") and not memory.get("csv_path"):
        if not memory.get("csv_files"):
            return "filesystem.list_directory"
        return "finance.merge_statements"

    if not memory.get("analysis_json"):
        return "finance.analyze_statement"

    if not memory.get("unusual_json"):
        return "finance.find_unusual_expenses"

    if not memory.get("financial_insight_context"):
        return "finance.prepare_financial_insight_context"

    if not memory.get("ai_insights_text"):
        return "finance.generate_ai_financial_insights"

    if not memory.get("monthly_report"):
        return "finance.generate_monthly_report"

    if FAST_DEV_SKIP_SQLITE:
        return "final_answer"

    if not memory.get("monthly_report_record"):
        return "finance.prepare_monthly_report_record"

    return "sqlite.create_record"


def completed_tool_replacement(tool_name: str, memory: Memory) -> str:
    completed_outputs = {
        "finance.analyze_statement": "analysis_json",
        "finance.find_unusual_expenses": "unusual_json",
        "finance.generate_savings_advice": "advice_text",
        "finance.prepare_financial_insight_context": "financial_insight_context",
        "finance.generate_ai_financial_insights": "ai_insights_text",
        "finance.generate_monthly_report": "monthly_report",
        "finance.prepare_monthly_report_record": "monthly_report_record",
        "finance.merge_statements": "csv_path",
    }

    memory_key = completed_outputs.get(tool_name)

    if not memory_key or not memory.get(memory_key):
        return ""

    replacement = next_required_tool(memory)

    if replacement == tool_name or replacement == "final_answer":
        return ""

    return replacement

### This function makes a short message to LLM after calling tool
# Why a preview and not a full result? 
# To avoid spending too many Groq tokens.
# The full result is stored in memory.
def make_observation(
    tool_name: str,
    result: str,
    ok: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
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


def enforce_currency_text(text: str, currency: str = "NIS") -> str:
    cleaned = str(text)

    cleaned = re.sub(r"\$\s*([\d,.]+)", rf"\1 {currency}", cleaned)
    cleaned = re.sub(
        r"([\d,.]+)\s*(?:USD|usd|US dollars|U\.S\. dollars|dollars|Dollars)",
        rf"\1 {currency}",
        cleaned,
    )
    cleaned = re.sub(r"\b(?:USD|usd|US dollars|U\.S\. dollars|dollars|Dollars)\b", currency, cleaned)

    return cleaned


class Agent:
    """
    Agent runtime for one finance-analysis request.

    The public orchestration stays the same; this class only owns runtime state
    that used to live as local variables in run_agent.
    """

    def __init__(self, user_goal: str, csv_path: str = "", statements_dir: str = "") -> None:
        self.user_goal = user_goal
        self.csv_path = csv_path
        self.statements_dir = statements_dir
        self.manager = MCPManager()

        self.messages: list[Message] = []
        self.memory: Memory = {}
        self.steps: list[Step] = []
        self.tool_results: ToolResults = {}
        self.tools: dict[str, Any] = {}

        if csv_path:
            self.memory["csv_path"] = csv_path

        if statements_dir:
            self.memory["statements_dir"] = statements_dir

    async def run(self) -> AgentResult:
        logger.info(
            "AGENT_START mode=%s csv_path=%s statements_dir=%s fast_dev_mode=%s max_steps=%s max_tool_output_length=%s max_history_messages=%s",
            "directory" if self.statements_dir else "single_csv",
            self.csv_path,
            self.statements_dir,
            FAST_DEV_MODE,
            MAX_AGENT_STEPS,
            MAX_TOOL_OUTPUT_LENGTH,
            MAX_HISTORY_MESSAGES,
        )

        try:
            self.tools = add_virtual_agent_tools(await self.manager.connect())
            self.messages = self.build_initial_messages()

            for step_number in range(1, MAX_AGENT_STEPS + 1):
                result = await self.run_step(step_number)
                if result is not None:
                    return result

            return self.result_after_max_steps()

        finally:
            await self.manager.close()

    def build_initial_messages(self) -> list[Message]:
        visible_tools = tools_for_current_mode(self.tools, self.statements_dir)
        tools_text = tools_to_text(visible_tools)
        persistence_instruction = (
            "- FAST_DEV_SKIP_SQLITE is active: do not save to SQLite. Return final_answer after finance.generate_monthly_report."
            if FAST_DEV_SKIP_SQLITE
            else "- Save monthly summary to SQLite by calling finance.prepare_monthly_report_record and then sqlite.create_record. Do not call sqlite.db_info."
        )

        if self.statements_dir:
            required_workflow = f"""
Required final result:
- First discover CSV files using filesystem tools.
- Then call finance.merge_statements to create one merged CSV.
- Analyze the merged CSV using the existing finance.analyze_statement tool.
- Find large expenses to review.
- Prepare financial insight context.
- Generate AI financial insights from the prepared context.
- Generate a clean monthly report.
{persistence_instruction}
"""
        else:
            required_workflow = f"""
Required final result:
- Analyze the persisted transactions table using finance.analyze_statement.
- If a CSV path is provided, it is only a compatibility fallback.
- Do not inspect directories.
- Do not merge statements.
- Find large expenses to review.
- Prepare financial insight context.
- Generate AI financial insights from the prepared context.
- Generate a clean monthly report.
{persistence_instruction}
"""

        user_goal_full = f"""
User goal:
{self.user_goal}

CSV path to analyze:
{self.csv_path}

Statements directory to analyze:
{self.statements_dir}

{required_workflow}

Important:
The final answer must be the report returned by finance.generate_monthly_report.
"""

        return [
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

    async def run_step(self, step_number: int) -> AgentResult | None:
        step_start = time.perf_counter()
        llm_start = time.perf_counter()
        llm_text = call_llm(self.messages)
        llm_duration = time.perf_counter() - llm_start

        try:
            action = extract_json(llm_text)
        except Exception as e:
            return self.handle_invalid_json(
                step_number,
                step_start,
                llm_duration,
                llm_text,
                e,
            )

        tool_name = action.get("tool")
        raw_args = action.get("args", {})
        reason = action.get("reason", "")

        if tool_name == "final_answer":
            return self.finish_with_final_answer(
                step_number,
                step_start,
                llm_duration,
                raw_args,
                reason,
            )

        if tool_name not in self.tools:
            observation = {
                "tool": tool_name,
                "ok": False,
                "error": f"Unknown tool: {tool_name}",
            }
            self.record_non_executed_tool(
                step_number,
                step_start,
                llm_duration,
                action,
                raw_args,
                reason,
                observation,
                "Choose another available tool.",
            )
            return None

        mode_error = disallowed_tool_for_mode(tool_name, self.statements_dir)

        if mode_error:
            observation = {
                "tool": tool_name,
                "ok": False,
                "error": mode_error,
            }
            self.record_non_executed_tool(
                step_number,
                step_start,
                llm_duration,
                action,
                raw_args,
                reason,
                observation,
                "Continue in Single CSV mode. Choose finance.analyze_statement for the provided csv_path.",
            )
            return None

        replacement_tool = completed_tool_replacement(tool_name, self.memory)

        if replacement_tool:
            logger.info(
                "REPLACE repeated_tool=%s next_required_tool=%s",
                tool_name,
                replacement_tool,
            )
            tool_name = replacement_tool
            raw_args = {}
            action = {
                "tool": tool_name,
                "args": raw_args,
                "reason": f"runtime advanced from repeated completed tool to {tool_name}",
            }
            reason = action["reason"]

        missing = missing_required_memory_for_tool(tool_name, self.memory)

        if missing:
            replacement_tool = next_required_tool(self.memory)

            if replacement_tool and replacement_tool != tool_name and replacement_tool in self.tools:
                logger.info(
                    "REPLACE missing_dependency_tool=%s next_required_tool=%s missing=%s",
                    tool_name,
                    replacement_tool,
                    missing,
                )
                tool_name = replacement_tool
                raw_args = {}
                action = {
                    "tool": tool_name,
                    "args": raw_args,
                    "reason": f"runtime selected missing prerequisite tool {tool_name}",
                }
                reason = action["reason"]
                missing = missing_required_memory_for_tool(tool_name, self.memory)

            if missing:
                observation = {
                    "tool": tool_name,
                    "ok": False,
                    "error": f"Missing required previous outputs. First call: {missing}",
                }
                self.record_non_executed_tool(
                    step_number,
                    step_start,
                    llm_duration,
                    action,
                    raw_args,
                    reason,
                    observation,
                    "Continue. Choose ONE missing tool first.",
                )
                return None

        if tool_name == "finance.generate_ai_financial_insights":
            return self.execute_ai_insights_step(
                step_number,
                step_start,
                llm_duration,
                action,
                raw_args,
                reason,
            )

        return await self.execute_tool_step(
            step_number,
            step_start,
            llm_duration,
            action,
            tool_name,
            raw_args,
            reason,
        )

    def handle_invalid_json(
        self,
        step_number: int,
        step_start: float,
        llm_duration: float,
        llm_text: str,
        error: Exception,
    ) -> AgentResult | None:
        total_duration = time.perf_counter() - step_start
        error_answer = (
            f"LLM returned invalid JSON.\n\n"
            f"Error: {error}\n\n"
            f"Raw response:\n{truncate_text(llm_text)}"
        )

        self.steps.append(
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
            str(error),
            truncate_text(llm_text),
        )

        logger.info(
            "TIMING step=%s tool=invalid_llm_response llm=%.3fs tool=0.000s total=%.3fs",
            step_number,
            llm_duration,
            total_duration,
        )

        if step_number >= MAX_AGENT_STEPS:
            return {"answer": error_answer, "steps": self.steps}

        self.messages.append(
            {
                "role": "user",
                "content": f"""
Your previous response was invalid JSON.

Error:
{str(error)}

Return exactly ONE JSON object in this format:
{{"tool": "server.tool_name", "args": {{}}, "reason": "short reason"}}

{memory_state_text(self.memory, self.csv_path)}

Continue. Choose exactly ONE next tool or final_answer.
""",
            }
        )
        self.prune_messages()

        return None

    def finish_with_final_answer(
        self,
        step_number: int,
        step_start: float,
        llm_duration: float,
        raw_args: Any,
        reason: str,
    ) -> AgentResult:
        total_duration = time.perf_counter() - step_start
        normalized_args = normalize_args(raw_args)
        if self.memory.get("monthly_report"):
            answer = self.memory["monthly_report"]
        else:
            answer = normalized_args.get(
                "answer",
                "The agent finished, but no monthly report was generated.",
            )

        self.steps.append(
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

        return {"answer": answer, "steps": self.steps}

    def record_non_executed_tool(
        self,
        step_number: int,
        step_start: float,
        llm_duration: float,
        action: dict[str, Any],
        raw_args: Any,
        reason: str,
        observation: dict[str, Any],
        next_instruction: str,
    ) -> None:
        total_duration = time.perf_counter() - step_start
        tool_name = str(observation.get("tool", ""))

        self.steps.append(
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

        self.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(action_for_history(action), ensure_ascii=False),
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": f"""
Tool observation:
{json.dumps(observation, ensure_ascii=False)}

{next_instruction}
""",
            }
        )
        self.prune_messages()

        logger.info(
            "TIMING step=%s tool=%s llm=%.3fs tool=0.000s total=%.3fs",
            step_number,
            tool_name,
            llm_duration,
            total_duration,
        )

    def execute_ai_insights_step(
        self,
        step_number: int,
        step_start: float,
        llm_duration: float,
        action: dict[str, Any],
        raw_args: Any,
        reason: str,
    ) -> AgentResult | None:
        args = build_args_for_tool(
            "finance.generate_ai_financial_insights",
            raw_args,
            self.csv_path,
            self.memory,
        )
        currency = "NIS"
        try:
            financial_context = json.loads(args.get("financial_context_json", "{}"))
            currency = financial_context.get("currency") or currency
        except Exception:
            pass

        insight_start = time.perf_counter()
        insight_text = call_llm(
            [
                {
                    "role": "system",
                    "content": """
You are a financial insight writer.
Use only the provided deterministic financial context.
Do not calculate new totals.
Do not invent merchants, categories, amounts, or percentages.
The only valid currency is the currency field from the context.
For this run, write every monetary amount in NIS/shekel terms only.
Never use $, USD, dollars, or any other foreign currency label.
Avoid generic advice such as "reduce spending by 10-15%".
Write 3-5 concise personalized observations and practical recommendations.
Ground every recommendation in the provided data.
""",
                },
                {
                    "role": "user",
                    "content": f"""
Prepared financial context JSON:
{args.get("financial_context_json", "")}

Spending summary JSON:
{args.get("analysis_json", "")}

Unusual expenses JSON:
{args.get("unusual_json", "")}

Return a concise markdown section with:
- 3-5 personalized observations
- practical recommendations
- explanation of spending patterns
- potential areas for review
""",
                },
            ]
        )
        insight_text = enforce_currency_text(insight_text, currency)
        insight_duration = time.perf_counter() - insight_start
        total_duration = time.perf_counter() - step_start

        remember_tool_output(
            "finance.generate_ai_financial_insights",
            insight_text,
            self.memory,
        )
        self.tool_results["finance.generate_ai_financial_insights"] = insight_text

        observation = make_observation(
            tool_name="finance.generate_ai_financial_insights",
            result=insight_text,
            ok=True,
        )

        step_record: Step = {
            "step": step_number,
            "tool": "finance.generate_ai_financial_insights",
            "args": args,
            "reason": reason,
            "observation": observation,
            "timing": {
                "llm_response_seconds": round(llm_duration, 3),
                "tool_execution_seconds": round(insight_duration, 3),
                "total_step_seconds": round(total_duration, 3),
            },
        }
        self.steps.append(step_record)

        logger.info(
            "RESULT tool=%s output=%s",
            "finance.generate_ai_financial_insights",
            short_json(observation),
        )
        logger.info(
            "TIMING step=%s tool=%s llm=%.3fs tool=%.3fs total=%.3fs",
            step_number,
            "finance.generate_ai_financial_insights",
            llm_duration,
            insight_duration,
            total_duration,
        )

        llm_observation = {
            "tool": observation.get("tool"),
            "ok": observation.get("ok"),
            "output_preview": observation.get("output_preview", ""),
            "error": observation.get("error", ""),
        }

        self.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(action_for_history(action), ensure_ascii=False),
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": f"""
Tool observation:
{json.dumps(llm_observation, ensure_ascii=False)}

{memory_state_text(self.memory, self.csv_path)}

Continue. Choose exactly ONE next tool or final_answer.
""",
            }
        )
        self.prune_messages()

        return None

    async def execute_tool_step(
        self,
        step_number: int,
        step_start: float,
        llm_duration: float,
        action: dict[str, Any],
        tool_name: str,
        raw_args: Any,
        reason: str,
    ) -> AgentResult | None:
        args = build_args_for_tool(tool_name, raw_args, self.csv_path, self.memory)

        step_record: Step = {
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

        result, last_error, tool_duration = await self.call_tool_with_retries(
            tool_name,
            args,
        )

        if last_error:
            observation = make_observation(
                tool_name=tool_name,
                result="",
                ok=False,
                error=last_error,
            )
        else:
            result_text = str(result)
            self.tool_results[tool_name] = result_text
            remember_tool_output(tool_name, result_text, self.memory)
            observation = make_observation(
                tool_name=tool_name,
                result=result_text,
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
        self.steps.append(step_record)

        if (
            FAST_DEV_SKIP_SQLITE
            and tool_name == "finance.generate_monthly_report"
            and self.memory.get("monthly_report")
        ):
            self.steps.append(
                {
                    "step": step_number + 1,
                    "tool": "final_answer",
                    "reason": "FAST_DEV_SKIP_SQLITE completed after report generation",
                    "output": self.memory["monthly_report"],
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

            return {"answer": self.memory["monthly_report"], "steps": self.steps}

        self.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(action_for_history(action), ensure_ascii=False),
            }
        )

        self.messages.append(
            {
                "role": "user",
                "content": f"""
Tool observation:
{json.dumps(llm_observation, ensure_ascii=False)}

{memory_state_text(self.memory, self.csv_path)}

Continue. Choose exactly ONE next tool or final_answer.
""",
            }
        )
        self.prune_messages()

        return None

    async def call_tool_with_retries(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[Any, str | None, float]:
        result: Any = None
        last_error: str | None = None
        tool_start = time.perf_counter()

        for attempt in range(1, 4):
            try:
                result = await self.manager.call_tool(tool_name, args)
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

        return result, last_error, tool_duration

    def prune_messages(self) -> None:
        self.messages = prune_messages(self.messages)

    def result_after_max_steps(self) -> AgentResult:
        if self.memory.get("monthly_report"):
            return {
                "answer": self.memory["monthly_report"],
                "steps": self.steps,
            }

        return {
            "answer": "Agent stopped after max steps. Check logs/agent.log.",
            "steps": self.steps,
        }


async def run_agent(
    user_goal: str,
    csv_path: str = "",
    statements_dir: str = "",
) -> AgentResult:
    return await Agent(user_goal, csv_path, statements_dir).run()
