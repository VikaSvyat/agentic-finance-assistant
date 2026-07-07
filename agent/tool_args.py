"""Tool argument normalization and workflow dependency helpers."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config import FAST_DEV_SKIP_SQLITE

Memory = dict[str, Any]


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


def coerce_positive_int(value: Any, default: int, maximum: int = 60) -> int:
    """
    Convert LLM-provided numeric arguments to bounded positive integers.
    Natural phrases such as "every month" fall back to the tool default.
    """

    if value is None or value == "":
        return default

    if isinstance(value, bool):
        return default

    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            number = int(text)
        else:
            match = re.search(r"\d+", text)
            if not match:
                return default
            number = int(match.group(0))
    else:
        return default

    if number < 1:
        return default

    return min(number, maximum)


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def limit_from_question(question: str) -> int | None:
    """Extract simple ranking limits such as "top 5" or "top five" from the question."""

    text = question.casefold()
    match = re.search(r"\btop\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\btop\s+{word}\b", text):
            return number
    return None


def normalize_query_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Fix common LLM argument mistakes before calling strict MCP query tools.
    """

    normalized = normalize_args(args).copy()

    if "analysis_month" in normalized and not normalized.get("month"):
        normalized["month"] = normalized["analysis_month"]

    if tool_name == "finance.get_unusual_transactions":
        month = normalized.get("month")
        if isinstance(month, str) and month.casefold().strip() in {
            "all months",
            "for all months",
            "all period",
            "entire period",
        }:
            normalized["month"] = "all"

    if tool_name in {"finance.get_recurring_merchants", "finance.get_category_trend"}:
        normalized["months"] = coerce_positive_int(normalized.get("months"), default=6)

    if tool_name == "finance.prepare_financial_review_context":
        normalized["months"] = coerce_positive_int(normalized.get("months"), default=6)

    if tool_name in {"finance.get_top_merchants", "finance.search_transactions"}:
        default_limit = 10 if tool_name == "finance.get_top_merchants" else 50
        normalized["limit"] = coerce_positive_int(normalized.get("limit"), default=default_limit)

    if tool_name == "finance.get_largest_transactions":
        normalized["limit"] = coerce_positive_int(normalized.get("limit"), default=10)

    if tool_name == "finance.get_category_comparison":
        months = normalized.get("months")
        if isinstance(months, list) and len(months) >= 2:
            normalized.setdefault("month_a", months[0])
            normalized.setdefault("month_b", months[1])

    allowed_args = {
        "finance.get_spending_summary": {"month"},
        "finance.get_category_breakdown": {"month"},
        "finance.get_top_merchants": {"month", "limit"},
        "finance.get_largest_transactions": {"month", "limit"},
        "finance.get_unusual_transactions": {"month"},
        "finance.compare_months": {"month_a", "month_b"},
        "finance.get_category_comparison": {"category", "month_a", "month_b"},
        "finance.get_category_trend": {"category", "months"},
        "finance.get_recurring_merchants": {"months"},
        "finance.prepare_financial_review_context": {"months"},
        "finance.search_transactions": {"keyword", "limit"},
    }

    allowed = allowed_args.get(tool_name)
    if allowed is None:
        return normalized

    return {
        key: value
        for key, value in normalized.items()
        if key in allowed
    }


QUESTION_MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def month_mentions_from_question(question: str) -> list[str]:
    """Resolve explicit English month names in the user question to YYYY-MM values."""

    months = []
    current_year = datetime.now().year
    words = re.findall(r"\w+", question.casefold())

    for word in words:
        month_num = QUESTION_MONTH_NAMES.get(word)
        if not month_num:
            continue
        month = f"{current_year:04d}-{month_num:02d}"
        if month not in months:
            months.append(month)

    return months


def category_from_question(question: str, args: dict[str, Any]) -> str:
    """Prefer an explicit tool category argument, then recover simple category mentions from text."""

    if args.get("category"):
        return str(args["category"])

    lowered = question.casefold()
    if "other" in lowered:
        return "Other"

    return ""


def asks_all_months(question: str) -> bool:
    """Detect phrases that intentionally request the full available period."""

    text = question.casefold()
    return any(
        marker in text
        for marker in [
            "all months",
            "for all months",
            "all period",
            "entire period",
        ]
    )


def maybe_redirect_query_tool(
    question: str,
    tool_name: str,
    args: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Correct high-risk router choices before calling MCP.

    This protects strict tools from vague LLM arguments and preserves user intent
    for common cases such as all-month unusual transactions and month ranges.
    """

    if tool_name == "finance.get_unusual_transactions" and asks_all_months(question):
        return tool_name, {"month": "all"}

    category = category_from_question(question, args)
    months = month_mentions_from_question(question)
    requested_limit = limit_from_question(question)

    if requested_limit and tool_name in {"finance.get_top_merchants", "finance.get_largest_transactions", "finance.search_transactions"}:
        args = {**args, "limit": requested_limit}

    if tool_name == "finance.get_unusual_transactions" and len(months) >= 2:
        return tool_name, {"month": f"{months[0]}:{months[1]}"}

    if tool_name == "finance.get_category_comparison" and not category and len(months) >= 2:
        return "finance.compare_months", {"month_a": months[0], "month_b": months[1]}

    if tool_name == "finance.compare_months" and len(months) >= 2:
        return tool_name, {"month_a": months[0], "month_b": months[1]}

    if category and len(months) >= 2 and tool_name != "finance.get_category_comparison":
        return (
            "finance.get_category_comparison",
            {
                "category": category,
                "month_a": months[0],
                "month_b": months[1],
            },
        )

    return tool_name, args


def missing_required_query_args(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Return required MCP arguments that are still missing after normalization."""

    required = {
        "finance.compare_months": ["month_a", "month_b"],
        "finance.get_category_comparison": ["category", "month_a", "month_b"],
        "finance.get_category_trend": ["category"],
        "finance.search_transactions": ["keyword"],
    }
    return [
        key
        for key in required.get(tool_name, [])
        if not args.get(key)
    ]


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
