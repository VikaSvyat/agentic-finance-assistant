"""Tool output memory, message-history, and observation helpers."""

import json
from pathlib import Path
from typing import Any

from agent.config import MAX_HISTORY_MESSAGES, MAX_TOOL_OUTPUT_LENGTH

Message = dict[str, str]
Memory = dict[str, Any]


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

        if value.startswith("/"):
            return value

        if "/" in value:
            return value

        if statements_dir:
            return str(Path(statements_dir) / value)

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
