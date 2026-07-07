"""Tool registry helpers for the agent prompt and mode-specific filtering."""

from typing import Any

from agent.config import FAST_DEV_SKIP_SQLITE


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
