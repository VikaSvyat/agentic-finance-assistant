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
from datetime import datetime
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
from agent.prompts import SYSTEM_PROMPT, insight_prompt_instructions
from agent.tool_registry import (
    add_virtual_agent_tools,
    disallowed_tool_for_mode,
    tools_for_current_mode,
    tools_to_text,
)
from finance.display import format_merchant_display

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


INSIGHT_INTENT_MARKERS = {
    "mortgage_review": [
        "approving my mortgage",
        "approve my mortgage",
        "mortgage approval",
        "mortgage review",
        "lender",
        "what would concern a lender",
        "concern a lender",
        "require explanation",
        "transactions would require explanation",
        "loan officer",
        "underwriter",
    ],
    "merchant_impact": [
        "merchants affect my budget",
        "which merchants affect",
        "merchant impact",
        "reduced spending with three merchants",
        "reduce spending with three merchants",
        "reduced spending at three merchants",
        "reduce spending at three merchants",
        "three merchants by 20%",
        "by 20%",
        "potential monthly impact",
        "merchant savings",
    ],
    "financial_coach": [
        "financial coach",
        "if you were my financial coach",
        "coach",
        "questions you would ask",
        "before giving recommendations",
        "where would you cut",
        "how could i save money",
        "what should i reduce first",
        "which three habits would have the biggest financial impact",
        "biggest financial impact",
        "stop spending money",
        "stop spending",
        "what should i stop",
    ],
    "spending_growth": [
        "biggest drivers",
        "spending growth",
        "growth drivers",
        "drivers of spending growth",
        "lowest-spending month",
        "lowest spending month",
        "highest-spending month",
        "highest spending month",
        "what drove",
        "drivers",
    ],
    "spending_changes": [
        "compare may and june",
        "compare",
        "month-to-month",
        "month to month",
        "what changed most",
        "changed most",
        "what changed",
        "spending changes",
        "compare and explain",
    ],
    "savings_opportunities": [
        "where to save",
        "optional spending",
        "optional categories",
        "largest optional",
        "biggest optional",
        "flexible spending",
        "controllable spending",
        "spending i can control",
        "areas where i can save",
        "discretionary spending",
        "reviewable spending",
        "largest discretionary categories",
        "biggest discretionary",
        "discretionary categories",
        "largest reviewable categories",
        "reviewable spending areas",
        "non-essential spending",
        "non essential spending",
        "reduce spending",
        "reduce my spending",
        "reduce expenses",
        "cut spending",
        "cut expenses",
        "10%",
        "without touching",
        "savings opportunities",
        "saving opportunities",
        "where should i start",
    ],
    "recommendations": [
        "recommendations",
        "recommendation",
        "recommend",
        "what recommendations",
        "advice",
        "what should i review",
        "review spending",
        "save money",
        "where should i start",
    ],
    "lifestyle": [
        "describe my lifestyle",
        "my lifestyle",
        "what do these transactions say about me",
        "what do my transactions say about me",
        "what kind of person do these transactions describe",
        "what do my spending habits say about me",
        "imagine you know nothing about me except these transactions",
        "nothing about me except these transactions",
        "what habits do you notice",
        "financial habits",
        "habits do you notice",
        "what patterns describe my spending behavior",
        "patterns describe my spending behavior",
        "spending behavior",
        "spending behaviour",
    ],
    "habits": [
        "financial habits",
        "habits do you notice",
        "habit",
        "habits",
        "behavior",
        "behaviour",
        "notice",
    ],
    "financial_review": [
        "summarize my spending behavior",
        "spending behavior",
        "financial review",
        "review my finances",
        "improve finances",
        "optimize spending",
        "unusual patterns",
        "patterns stand out",
    ],
    "commitments": [
        "which recurring payments are actual obligations",
        "which are payment methods",
        "just payment methods",
        "classify recurring payments",
        "obligations vs payment methods",
        "explain recurring financial commitments",
        "which recurring payments should i keep",
        "which recurring items are contractual obligations",
        "contractual obligations",
        "payment methods",
        "transaction type labels",
        "standing order",
        "digital transfer",
        "actual obligations",
        "stable recurring obligations",
        "recurring financial commitments",
        "recurring obligations",
        "fixed commitments",
        "monthly obligations",
        "long-term commitments",
        "long term commitments",
        "stable recurring payments",
        "recurring commitments",
        "financial commitments",
    ],
}


FINANCIAL_REVIEW_SUBTYPES = {
    "general": "General reasoning over the financial review context when no specialized mode fits.",
    "financial_coach": (
        "Practical personal finance coaching: savings ideas, what to reduce first, "
        "financial habits with the biggest impact, and where to stop spending."
    ),
    "lifestyle": (
        "Behavior/lifestyle observations from transactions, such as what spending habits "
        "or patterns say about the user, grounded in evidence."
    ),
    "commitments": (
        "Classify or explain recurring payments, obligations, payment methods, fixed "
        "commitments, and recurring financial responsibilities."
    ),
    "mortgage_review": (
        "Mortgage/lender perspective: what a lender might question, affordability concerns, "
        "or transactions that may require explanation."
    ),
    "merchant_impact": (
        "Merchant-level budget impact and estimated savings from reducing spending with "
        "specific recurring or high-impact merchants."
    ),
}


LLM_SUBTYPE_CONFIDENCE_THRESHOLD = 0.65


def asks_for_reasoning_or_coaching(question: str) -> bool:
    text = question.casefold()
    markers = [
        "what should i",
        "where should i",
        "how could i",
        "how can i",
        "why",
        "explain",
        "what changed",
        "what would i save",
        "safe to ignore",
        "safely ignore",
        "actually normal",
        "looks expensive",
        "expensive at first glance",
        "small recurring expenses",
        "over a year",
        "hidden opportunity",
        "intentional",
        "impulsive",
        "never recommend cutting",
        "never recommend",
        "single most important insight",
        "overlooking",
        "support your answer",
        "what would you recommend",
        "what do you recommend",
        "recommend",
        "recommendations",
        "advice",
        "strategy",
        "prioritize",
        "priorities",
        "priority",
        "biggest opportunity",
        "biggest opportunities",
        "best opportunity",
        "opportunities",
        "biggest impact",
        "financial impact",
        "long-term",
        "long term",
        "trade-off",
        "tradeoff",
        "trade-offs",
        "reasoning",
        "coach",
        "coaching",
        "habits",
        "what habits",
        "stop spending",
        "reduced spending",
        "reduce spending",
        "reduce first",
        "cut",
        "save money",
    ]
    return any(marker in text for marker in markers)


def asks_for_explicit_reporting(question: str) -> bool:
    text = question.casefold()
    reporting_markers = [
        "compare months",
        "compare may and june",
        "compare",
        "list unusual transactions",
        "show unusual transactions",
        "show categories",
        "list categories",
        "category breakdown",
        "breakdown",
        "recurring merchants",
        "merchants appear every month",
        "top merchants",
        "top categories",
        "total",
        "totals",
        "how much did i spend",
        "spending summary",
        "trend over time",
        "show transactions",
        "list transactions",
    ]
    return any(marker in text for marker in reporting_markers)


def classify_financial_intent(question: str) -> str:
    """
    Classify the user's broad financial intent before selecting a tool.

    The first checks deliberately prioritize reasoning/coaching language over
    simple keywords, so advice questions are not accidentally routed to factual
    reporting tools.
    """

    text = question.casefold()
    if asks_for_reasoning_or_coaching(question):
        # Semantic advice/coaching questions must reach Financial Review even when
        # they mention reporting nouns like categories, merchants, trends, or months.
        return "financial_review"
    if asks_for_explicit_reporting(question):
        return "deterministic_reporting"

    scores = {
        intent: sum(1 for marker in markers if marker in text)
        for intent, markers in INSIGHT_INTENT_MARKERS.items()
    }
    if "₪" in text and any(marker in text for marker in ["cut", "save", "reduce"]):
        scores["financial_coach"] = scores.get("financial_coach", 0) + 2
    if any(marker in text for marker in ["merchant", "merchants"]) and any(
        marker in text for marker in ["budget", "20%", "reduce", "impact", "savings"]
    ):
        scores["merchant_impact"] = scores.get("merchant_impact", 0) + 2
    if any(marker in text for marker in ["mortgage", "lender", "underwriter"]) and any(
        marker in text for marker in ["concern", "approve", "approval", "explanation", "risky"]
    ):
        scores["mortgage_review"] = scores.get("mortgage_review", 0) + 2
    if any(
        marker in text
        for marker in [
            "person",
            "lifestyle",
            "transactions say about me",
            "habits do you notice",
            "what habits do you notice",
            "financial habits do you notice",
            "spending habits say about me",
        ]
    ):
        scores["lifestyle"] = scores.get("lifestyle", 0) + 3
    top_intent = max(scores, key=lambda intent: scores[intent])
    if scores[top_intent] > 0:
        return top_intent
    return "deterministic_reporting"


def classify_response_mode(question: str) -> str:
    """
    Choose the answer architecture for the question.

    deterministic: factual MCP JSON is formatted by Python.
    insight: review context is prepared first, then interpreted by subtype.
    interpretive: legacy explanatory path over a single deterministic tool.
    """

    text = question.casefold()
    if asks_for_reasoning_or_coaching(question):
        return "insight"

    intent = classify_financial_intent(question)
    if intent != "deterministic_reporting":
        return "insight"

    interpretive_markers = [
        "why",
        "explain",
        "reason",
        "pattern",
        "patterns",
        "behavior",
        "behaviour",
        "notice",
        "review to save",
    ]
    return "interpretive" if any(marker in text for marker in interpretive_markers) else "deterministic"


def deterministic_insight_subtype(question: str) -> str | None:
    """
    Return a high-confidence insight subtype without calling the LLM.

    These shortcuts keep latency low for obvious ranking, commitments, and
    month-comparison questions while leaving ambiguous coaching questions to the
    lightweight subtype classifier.
    """

    text = question.casefold()
    if asks_for_reasoning_or_coaching(question):
        return None

    intent = classify_financial_intent(question)
    if intent in {"spending_changes", "spending_growth"} or any(
        marker in text
        for marker in [
            "compare",
            "month-to-month",
            "month to month",
            "between may and june",
            "may and june",
            "changed most",
            "spending growth",
        ]
    ):
        return "compare_months"

    ranking_markers = [
        "optional spending",
        "optional categories",
        "largest discretionary spending categories",
        "discretionary spending categories",
        "reviewable spending areas",
        "reviewable categories",
        "non-essential spending",
        "non essential spending",
        "flexible spending",
        "controllable spending",
        "spending i can control",
        "areas where i can save",
        "largest optional",
        "biggest optional",
        "largest discretionary",
        "biggest discretionary",
        "largest reviewable categories",
        "largest discretionary categories",
    ]
    if any(marker in text for marker in ranking_markers):
        return "ranking"

    commitment_markers = [
        "which recurring payments are actual obligations",
        "which are payment methods",
        "just payment methods",
        "classify recurring payments",
        "obligations vs payment methods",
        "explain recurring financial commitments",
        "which recurring payments should i keep",
        "which recurring items are contractual obligations",
        "contractual obligations",
        "payment methods",
        "transaction type labels",
        "standing order",
        "digital transfer",
        "actual obligations",
        "stable recurring obligations",
        "recurring financial commitments",
        "recurring obligations",
        "fixed commitments",
        "monthly obligations",
        "long-term commitments",
        "long term commitments",
        "stable recurring payments",
        "recurring commitments",
        "financial commitments",
    ]
    if any(marker in text for marker in commitment_markers):
        return "commitments"

    return None


def financial_review_subtypes_text() -> str:
    """Render supported Financial Review subtypes for the LLM classifier prompt."""

    return "\n".join(
        f"- {name}: {description}"
        for name, description in FINANCIAL_REVIEW_SUBTYPES.items()
    )


def classify_insight_subtype_with_llm(question: str) -> dict[str, Any]:
    """
    Use the LLM only to classify the subtype, not to answer the financial question.

    The classifier is constrained to structured JSON and falls back to "general"
    when confidence is low or the response is malformed.
    """

    subtype_names = set(FINANCIAL_REVIEW_SUBTYPES)
    try:
        classifier_text = call_llm(
            [
                {
                    "role": "system",
                    "content": f"""
You are a lightweight intent classifier for Financial Review questions.
Return ONLY one JSON object.
Do not answer the user's financial question.
Select exactly one subtype from the supported list.
If no specialized subtype clearly fits, choose "general".

Supported subtypes:
{financial_review_subtypes_text()}

Valid response format:
{{
  "subtype": "financial_coach",
  "confidence": 0.94,
  "reason": "The user asks for recommendations to improve spending habits."
}}
""",
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}",
                },
            ]
        )
        parsed = extract_json(classifier_text)
        subtype = str(parsed.get("subtype") or "general")
        try:
            confidence = float(parsed.get("confidence", 0))
        except Exception:
            confidence = 0
        reason = str(parsed.get("reason") or "")
        if subtype not in subtype_names:
            subtype = "general"
            confidence = 0
        if confidence < LLM_SUBTYPE_CONFIDENCE_THRESHOLD:
            subtype = "general"
        logger.info(
            "NLP_SUBTYPE_CLASSIFIER_RAW=%s parsed=%s",
            classifier_text,
            json.dumps(
                {"subtype": subtype, "confidence": confidence, "reason": reason},
                ensure_ascii=False,
            ),
        )
        return {
            "subtype": subtype,
            "confidence": confidence,
            "reason": reason,
        }
    except Exception as exc:
        logger.warning(
            "NLP_SUBTYPE_CLASSIFIER_FAILED question=%s error=%s",
            question,
            exc,
        )
        return {
            "subtype": "general",
            "confidence": 0,
            "reason": f"classifier failed: {exc}",
        }


def classify_insight_subtype(question: str, use_llm: bool = False) -> str:
    """Resolve the insight subtype using deterministic shortcuts before optional LLM classification."""

    deterministic_subtype = deterministic_insight_subtype(question)
    if deterministic_subtype:
        return deterministic_subtype
    if use_llm:
        return classify_insight_subtype_with_llm(question)["subtype"]
    intent = classify_financial_intent(question)
    if intent in FINANCIAL_REVIEW_SUBTYPES:
        return intent
    return "general"


def parse_tool_json(raw_result: str) -> dict[str, Any]:
    """Parse MCP tool output defensively so formatters can handle string or dict results."""

    if isinstance(raw_result, dict):
        return raw_result
    try:
        parsed = json.loads(raw_result)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


MERCHANT_ENTITY_KEYS = {"merchant", "label"}
HEBREW_ENTITY_RE = re.compile(r"[\u0590-\u05FF][\u0590-\u05FF\s\"'׳״./\\-]{1,}")
BRACKET_LABEL_RE = re.compile(r"\[([^\]]+)\]")
QUOTED_LATIN_LABEL_RE = re.compile(r"[\"“”']([A-Za-z][A-Za-z0-9&/ .\\-]{2,})[\"“”']")
EXPLICIT_MERCHANT_LABEL_RE = re.compile(
    r"(?im)(?:^|\b)(?:[-*\d.() ]*)?(?:\*\*)?\s*merchant(?:\s+display)?\s*:\s*(?:\*\*)?\s*([^:\n\r]{2,80})"
)


def normalize_entity_text(value: Any) -> str:
    """Normalize entity text for comparison without changing the original value."""

    return " ".join(str(value or "").strip().split())


def collect_allowed_merchant_entities(value: Any) -> set[str]:
    """
    Collect merchant names that are allowed to appear in an answer.

    Both the original stored name and the deterministic display label are allowed;
    this prevents the LLM from inventing new merchant names while still supporting
    user-friendly display names.
    """

    allowed = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key in MERCHANT_ENTITY_KEYS:
                    text = normalize_entity_text(item)
                    if text:
                        allowed.add(text)
                        display = format_merchant_display(text)
                        allowed.add(display)
                        match = BRACKET_LABEL_RE.search(display)
                        if match:
                            allowed.add(match.group(1))
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return allowed


def explicit_answer_merchant_labels(answer: str) -> set[str]:
    """Extract explicit LLM-created labels such as "Merchant: Coffee Shop"."""

    labels = set()
    for match in EXPLICIT_MERCHANT_LABEL_RE.finditer(str(answer)):
        candidate = normalize_entity_text(match.group(1).strip(" -*_`"))
        if candidate:
            labels.add(candidate)
    return labels


def answer_unknown_merchant_entities(answer: str, raw_result: str) -> list[str]:
    """
    Find merchant-like entities in an answer that are not present in tool JSON.

    This is intentionally conservative for Hebrew names and explicit merchant
    labels because hallucinated merchants were a recurring product-quality issue.
    """

    data = parse_tool_json(raw_result)
    if not data:
        return []

    allowed = collect_allowed_merchant_entities(data)
    if not allowed:
        return sorted(explicit_answer_merchant_labels(answer))

    allowed_normalized = {normalize_entity_text(item) for item in allowed if normalize_entity_text(item)}
    allowed_hebrew = {item for item in allowed_normalized if re.search(r"[\u0590-\u05FF]", item)}
    allowed_bracket_labels = {
        item
        for item in allowed_normalized
        if item and not re.search(r"[\u0590-\u05FF]", item)
    }

    unknown = set()

    for match in HEBREW_ENTITY_RE.finditer(str(answer)):
        candidate = normalize_entity_text(match.group(0).strip(" .,:;()[]"))
        if not candidate:
            continue
        if candidate in allowed_hebrew:
            continue
        if any(candidate in allowed_item or allowed_item in candidate for allowed_item in allowed_hebrew):
            continue
        unknown.add(candidate)

    for match in BRACKET_LABEL_RE.finditer(str(answer)):
        candidate = normalize_entity_text(match.group(1))
        if not candidate:
            continue
        if candidate in allowed_bracket_labels:
            continue
        unknown.add(candidate)

    for match in QUOTED_LATIN_LABEL_RE.finditer(str(answer)):
        candidate = normalize_entity_text(match.group(1))
        if not candidate:
            continue
        if candidate in allowed_bracket_labels or candidate in allowed_normalized:
            continue
        if any(candidate in allowed_item or allowed_item in candidate for allowed_item in allowed_bracket_labels):
            continue
        unknown.add(candidate)

    for candidate in explicit_answer_merchant_labels(answer):
        if candidate in allowed_bracket_labels or candidate in allowed_normalized:
            continue
        if any(candidate in allowed_item or allowed_item in candidate for allowed_item in allowed_bracket_labels):
            continue
        unknown.add(candidate)

    return sorted(unknown)


def enforce_answer_merchant_grounding(
    answer: str,
    raw_result: str,
    response_mode: str,
    question: str,
) -> str:
    """Apply strict merchant grounding for legacy/non-subtyped answer paths."""

    unknown = answer_unknown_merchant_entities(answer, raw_result)
    if not unknown:
        return answer

    logger.error(
        "NLP_UNGROUNDED_MERCHANTS response_mode=%s unknown=%s question=%s",
        response_mode,
        json.dumps(unknown, ensure_ascii=False),
        question,
    )
    return SAFE_GROUNDING_FALLBACK_PREFIX


SAFE_GROUNDING_FALLBACK_PREFIX = "__verified_financial_summary_required__"
STRICT_MERCHANT_GROUNDING_SUBTYPES = {"merchant_impact", "commitments"}
LIGHT_MERCHANT_GROUNDING_SUBTYPES = {"financial_coach", "lifestyle", "mortgage_review", "general"}


def merchant_grounding_rewrite_instruction(subtype: str) -> str:
    """Return subtype-specific rewrite instructions when merchant names are unsafe."""

    if subtype == "financial_coach":
        return (
            "Rewrite using only categories, verified amounts, reviewable areas, and mandatory-vs-discretionary distinctions. "
            "Do not mention merchant names. Keep the financial coach recommendation structure."
        )
    if subtype == "lifestyle":
        return (
            "Rewrite lifestyle observations using categories and evidence only. "
            "Do not mention merchant names. Provide 3 to 5 observations with Evidence lines."
        )
    if subtype == "mortgage_review":
        return (
            "Rewrite using categories, amounts, recurring obligations, month coverage, and lender questions only. "
            "Do not mention merchant names."
        )
    return (
        "Rewrite without merchant names. Use categories, verified amounts, and deterministic context only."
    )


def apply_subtype_aware_merchant_guard(
    answer: str,
    raw_result: str,
    response_mode: str,
    question: str,
    insight_subtype: str,
    insight_messages: list[dict[str, str]],
    result_currency: str,
) -> str:
    """
    Protect merchant-centric answers while keeping broader reasoning answers usable.

    merchant_impact and commitments require strict merchant grounding. Other
    insight subtypes first get a rewrite without merchant names, because they can
    usually answer with categories, amounts, and spending patterns instead.
    """

    unknown = answer_unknown_merchant_entities(answer, raw_result)
    if not unknown:
        return answer

    logger.warning(
        "NLP_UNGROUNDED_MERCHANTS subtype=%s response_mode=%s unknown=%s question=%s",
        insight_subtype,
        response_mode,
        json.dumps(unknown, ensure_ascii=False),
        question,
    )

    if insight_subtype in STRICT_MERCHANT_GROUNDING_SUBTYPES:
        return SAFE_GROUNDING_FALLBACK_PREFIX

    if insight_subtype not in LIGHT_MERCHANT_GROUNDING_SUBTYPES:
        return SAFE_GROUNDING_FALLBACK_PREFIX

    rewrite_messages = insight_messages + [
        {
            "role": "assistant",
            "content": answer,
        },
        {
            "role": "user",
            "content": (
                "The previous answer used merchant names that are not safe to rely on for this subtype. "
                f"{merchant_grounding_rewrite_instruction(insight_subtype)} "
                "Answer the user's original question directly. Do not mention validation, grounding, or fallback."
            ),
        },
    ]
    rewritten = call_llm(rewrite_messages)
    rewritten = postprocess_insight_currency(rewritten, result_currency)
    rewrite_errors = insight_answer_validation_errors(insight_subtype, rewritten)
    rewrite_fatal = fatal_insight_validation_errors(insight_subtype, rewrite_errors)
    rewritten_unknown = answer_unknown_merchant_entities(rewritten, raw_result)
    if rewrite_fatal or rewritten_unknown:
        logger.warning(
            "NLP_MERCHANT_SAFE_REWRITE_FAILED subtype=%s validation_errors=%s unknown=%s question=%s",
            insight_subtype,
            json.dumps(rewrite_fatal, ensure_ascii=False),
            json.dumps(rewritten_unknown, ensure_ascii=False),
            question,
        )
        return SAFE_GROUNDING_FALLBACK_PREFIX
    if rewrite_errors:
        logger.info(
            "NLP_MERCHANT_SAFE_REWRITE_ACCEPTED_WITH_NONFATAL_ERRORS subtype=%s validation_errors=%s question=%s",
            insight_subtype,
            json.dumps(rewrite_errors, ensure_ascii=False),
            question,
        )
    return rewritten


def validation_rewrite_instruction(subtype: str) -> str:
    """Return subtype-specific rewrite instructions for answers that fail validation."""

    if subtype == "financial_coach":
        return (
            "Rewrite as practical financial-coach advice using only reviewable categories and verified amounts. "
            "Do not recommend cutting or reviewing mandatory obligations, recurring obligations, rent, pension, insurance, "
            "education, taxes, utilities, standing orders, digital transfers, card fees, or payment mechanisms as first priorities. "
            "Use categories such as shopping, leisure, restaurants, food delivery, flights/travel, or subscriptions only when present in the context."
        )
    if subtype == "lifestyle":
        return (
            "Rewrite as 3 to 5 lifestyle observations with Evidence lines. Use categories, patterns, recurring activity types, and verified amounts only."
        )
    if subtype == "mortgage_review":
        return (
            "Rewrite as lender-style questions and concerns using categories, recurring obligations, large expenses, month coverage, and verified amounts only. "
            "Do not use fraud, investigation, or unsupported risk language."
        )
    return "Rewrite the answer using only verified categories, amounts, and deterministic context."


def apply_subtype_aware_validation_guard(
    answer: str,
    insight_subtype: str,
    insight_messages: list[dict[str, str]],
    question: str,
    result_currency: str,
) -> str:
    """
    Rewrite answers that violate subtype rules before falling back.

    This keeps the product from showing a verified fallback for every imperfect
    answer, while still blocking fatal issues such as wrong currency or mandatory
    obligations being recommended as first savings targets.
    """

    errors = insight_answer_validation_errors(insight_subtype, answer)
    fatal_errors = fatal_insight_validation_errors(insight_subtype, errors)
    if not errors:
        return answer

    logger.warning(
        "INSIGHT_VALIDATION_REWRITE subtype=%s errors=%s fatal_errors=%s question=%s",
        insight_subtype,
        json.dumps(errors, ensure_ascii=False),
        json.dumps(fatal_errors, ensure_ascii=False),
        question,
    )

    if fatal_errors and insight_subtype not in LIGHT_MERCHANT_GROUNDING_SUBTYPES:
        return SAFE_GROUNDING_FALLBACK_PREFIX

    rewrite_messages = insight_messages + [
        {
            "role": "assistant",
            "content": answer,
        },
        {
            "role": "user",
            "content": (
                f"{validation_rewrite_instruction(insight_subtype)} "
                "Answer the user's original question directly. Use ₪ for NIS amounts. "
                "Do not mention validation, grounding, or fallback."
            ),
        },
    ]
    rewritten = call_llm(rewrite_messages)
    rewritten = postprocess_insight_currency(rewritten, result_currency)
    rewritten_errors = insight_answer_validation_errors(insight_subtype, rewritten)
    rewritten_fatal = fatal_insight_validation_errors(
        insight_subtype,
        rewritten_errors,
    )
    if rewritten_fatal:
        logger.warning(
            "INSIGHT_FATAL_VALIDATION_REWRITE_FAILED subtype=%s errors=%s question=%s",
            insight_subtype,
            json.dumps(rewritten_fatal, ensure_ascii=False),
            question,
        )
        return SAFE_GROUNDING_FALLBACK_PREFIX
    if rewritten_errors:
        logger.info(
            "INSIGHT_VALIDATION_REWRITE_ACCEPTED_WITH_NONFATAL_ERRORS subtype=%s errors=%s question=%s",
            insight_subtype,
            json.dumps(rewritten_errors, ensure_ascii=False),
            question,
        )
    return rewritten


def verified_answer_intro() -> list[str]:
    """Standard user-facing intro for verified deterministic fallback answers."""

    return [
        "Verified Answer",
        "I prefer verified facts over uncertain interpretations for this question.",
        "Here are the most relevant facts from the deterministic financial analysis:",
    ]


def add_reviewable_area_lines(lines: list[str], areas: list[dict[str, Any]], currency: str, limit: int = 5) -> None:
    """Append concise reviewable spending facts to a fallback answer."""

    for item in areas[:limit]:
        lines.append(
            f"- {display_spending_area_type(item.get('type'))}: "
            f"{money_text(item.get('amount', 0), currency)}"
        )


def add_merchant_lines(lines: list[str], merchants: list[dict[str, Any]], currency: str, limit: int = 5) -> None:
    """Append merchant facts using deterministic display names only."""

    for item in merchants[:limit]:
        label = item.get("merchant_display") or format_merchant_display(item.get("merchant", ""))
        amount = item.get("amount", item.get("total_amount", 0))
        details = []
        if item.get("months_present"):
            details.append(f"{item.get('months_present')} months")
        if item.get("estimated_monthly_average"):
            details.append(f"estimated monthly impact {money_text(item.get('estimated_monthly_average'), currency)}")
        if item.get("reviewable_area"):
            details.append(f"area: {display_spending_area_type(item.get('reviewable_area'))}")
        detail_text = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {label}: {money_text(amount, currency)}{detail_text}")


def add_category_lines(lines: list[str], categories: list[dict[str, Any]], currency: str, limit: int = 5) -> None:
    """Append top category facts to a verified answer."""

    for item in categories[:limit]:
        label = item.get("category_en") or item.get("category")
        lines.append(f"- {label}: {money_text(item.get('total_amount', 0), currency)}")


def add_obligation_lines(lines: list[str], obligations: list[dict[str, Any]], currency: str, limit: int = 5) -> None:
    """Append recurring obligation facts without treating payment mechanisms as merchants."""

    for item in obligations[:limit]:
        label = item.get("merchant_display") or item.get("label") or item.get("merchant") or item.get("payment_type") or "Recurring obligation"
        amount = item.get("total_amount", item.get("amount", 0))
        details = []
        if item.get("months_present"):
            details.append(f"{item.get('months_present')} months")
        if item.get("transactions"):
            details.append(f"{item.get('transactions')} transactions")
        detail_text = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {label}: {money_text(amount, currency)}{detail_text}")


def add_unusual_or_large_lines(
    lines: list[str],
    title: str,
    items: list[dict[str, Any]],
    currency: str,
    limit: int = 3,
) -> None:
    """Append large/unusual transaction facts for mortgage-review fallback answers."""

    if not items:
        return
    lines.append(title)
    for item in items[:limit]:
        label = item.get("merchant_display") or item.get("merchant") or item.get("description") or item.get("category")
        amount = item.get("amount", item.get("amount_abs", item.get("total_amount", 0)))
        month = item.get("transaction_month") or item.get("month")
        month_text = f", {month}" if month else ""
        lines.append(f"- {label}: {money_text(amount, currency)}{month_text}")


def deterministic_insight_fallback(subtype: str, selected_context: dict[str, Any]) -> str:
    """
    Produce a verified deterministic answer when LLM wording is unsafe.

    This fallback is reserved for cases where a rewrite still cannot satisfy
    grounding rules, especially merchant-centric subtypes.
    """

    currency = selected_context.get("currency", "")
    lines = verified_answer_intro()

    if subtype == "merchant_impact":
        candidates = selected_context.get("reviewable_merchant_candidates") or selected_context.get("merchant_candidates") or []
        if not candidates:
            return ""
        add_merchant_lines(lines, candidates, currency)
        lines.append("These statements come directly from the deterministic financial analysis.")
        return "\n".join(lines)

    if subtype in {"financial_coach", "general"}:
        areas = selected_context.get("reviewable_spending_areas", [])
        if not areas:
            return ""
        add_reviewable_area_lines(lines, areas, currency)
        possible = selected_context.get("possible_review_areas", [])
        if possible:
            lines.append("Possible review areas:")
            add_reviewable_area_lines(lines, possible, currency, limit=3)
        lines.append("These statements come directly from the deterministic financial analysis.")
        return "\n".join(lines)

    if subtype == "lifestyle":
        categories = selected_context.get("top_categories", [])
        recurring_merchants = selected_context.get("recurring_merchants", [])
        reviewable = selected_context.get("reviewable_spending_areas", [])
        if not categories and not recurring_merchants and not reviewable:
            return ""
        if categories:
            lines.append("Top spending signals:")
            add_category_lines(lines, categories, currency)
        if recurring_merchants:
            lines.append("Recurring activities or merchants:")
            add_merchant_lines(lines, recurring_merchants, currency, limit=5)
        if reviewable:
            lines.append("Reviewable lifestyle-related areas:")
            add_reviewable_area_lines(lines, reviewable, currency, limit=5)
        lines.append("These statements come directly from the deterministic financial analysis.")
        return "\n".join(lines)

    if subtype == "mortgage_review":
        obligations = selected_context.get("stable_recurring_obligations", [])
        mandatory = selected_context.get("mandatory_spending_areas", [])
        unusual = selected_context.get("unusual_transactions", [])
        largest = selected_context.get("largest_expenses", [])
        coverage = selected_context.get("month_coverage", [])
        if not obligations and not mandatory and not unusual and not largest:
            return ""
        incomplete = [
            item for item in coverage
            if item and item.get("is_complete_month") is False
        ]
        if incomplete:
            month_notes = ", ".join(
                f"{item.get('month')} ({item.get('coverage_days')}/{item.get('expected_days')} days)"
                for item in incomplete
            )
            lines.append(f"- Incomplete month coverage: {month_notes}.")
        if obligations:
            lines.append("Recurring obligations:")
            add_obligation_lines(lines, obligations, currency)
        if mandatory:
            lines.append("Fixed or mandatory spending areas:")
            add_reviewable_area_lines(lines, mandatory, currency, limit=5)
        add_unusual_or_large_lines(lines, "Large or unusual expenses:", unusual or largest, currency)
        lines.append("These statements come directly from the deterministic financial analysis.")
        return "\n".join(lines)

    return ""


def deterministic_light_insight_answer(subtype: str, selected_context: dict[str, Any]) -> str:
    """
    Produce a safe subtype-shaped answer without the "Verified Answer" label.

    Light answers keep broad reasoning subtypes useful even when the local model
    fails to produce a reliable response.
    """

    currency = selected_context.get("currency", "")
    if subtype == "financial_coach":
        areas = selected_context.get("reviewable_spending_areas", [])
        if not areas:
            return ""
        lines = ["Recommended first review areas:"]
        for item in areas[:3]:
            lines.append(
                f"- {display_spending_area_type(item.get('type'))}: "
                f"{money_text(item.get('amount', 0), currency)}. "
                "This is reviewable spending, so it is a better first target than mandatory obligations."
            )
        lines.append("Estimated realistic monthly savings: not enough evidence to estimate precisely from the available context.")
        lines.append("Target achievable?: unclear without cutting mandatory obligations.")
        return "\n".join(lines)

    if subtype == "lifestyle":
        categories = selected_context.get("top_categories", [])
        reviewable = selected_context.get("reviewable_spending_areas", [])
        if not categories and not reviewable:
            return ""
        lines = []
        if categories:
            for index, item in enumerate(categories[:3], start=1):
                label = item.get("category_en") or item.get("category")
                lines.append(
                    f"{index}. Observation: A large part of the spending pattern is tied to {label}.\n"
                    f"   Evidence: {label} totals {money_text(item.get('total_amount', 0), currency)} in the deterministic context."
                )
        start = len(lines) + 1
        for offset, item in enumerate(reviewable[: max(0, 5 - len(lines))], start=start):
            label = display_spending_area_type(item.get("type"))
            lines.append(
                f"{offset}. Observation: There is also visible flexible lifestyle spending in {label}.\n"
                f"   Evidence: {label} totals {money_text(item.get('amount', 0), currency)} as a reviewable spending area."
            )
        return "\n".join(lines[:5])

    if subtype == "mortgage_review":
        lines = ["Lender-style concerns to review:"]
        coverage = selected_context.get("month_coverage", [])
        incomplete = [
            item for item in coverage
            if item and item.get("is_complete_month") is False
        ]
        for item in incomplete[:2]:
            lines.append(
                f"- Incomplete month coverage: {item.get('month')} has "
                f"{item.get('coverage_days')}/{item.get('expected_days')} days, so a lender should avoid treating it as a full-month trend."
            )
        for item in selected_context.get("mandatory_spending_areas", [])[:3]:
            lines.append(
                f"- Fixed obligations: {display_spending_area_type(item.get('type'))} totals "
                f"{money_text(item.get('amount', 0), currency)} and would matter for affordability."
            )
        for item in selected_context.get("reviewable_spending_areas", [])[:3]:
            lines.append(
                f"- Reviewable spending: {display_spending_area_type(item.get('type'))} totals "
                f"{money_text(item.get('amount', 0), currency)} and could prompt questions about flexibility."
            )
        return "\n".join(lines)

    return ""


def money_text(value: Any, currency: str = "") -> str:
    try:
        amount = f"{float(value):,.2f}"
    except Exception:
        amount = str(value)
    if currency == "NIS":
        return f"₪{amount}"
    if currency == "USD":
        return f"${amount}"
    return f"{amount} {currency}".strip()


SPENDING_AREA_DISPLAY_LABELS = {
    "shopping": "Shopping",
    "leisure_entertainment_sports": "Leisure, entertainment and sports",
    "restaurants": "Restaurants",
    "food_delivery": "Food delivery",
    "flights_and_travel": "Flights and travel",
    "subscriptions": "Subscriptions",
    "mandatory_fixed": "Mandatory fixed expenses",
}


def display_spending_area_type(value: Any) -> str:
    text = str(value or "")
    return SPENDING_AREA_DISPLAY_LABELS.get(text, text.replace("_", " ").title())


def pluralize(count: Any, singular: str, plural: str | None = None) -> str:
    try:
        number = int(count)
    except Exception:
        number = 0
    return singular if number == 1 else (plural or f"{singular}s")


def item_lines(items: list[dict[str, Any]], formatter, empty_text: str) -> str:
    if not items:
        return empty_text
    return "\n".join(formatter(item, index) for index, item in enumerate(items, start=1))


def requested_rank_limit(question: str, default: int = 3) -> int:
    text = question.casefold()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for word, number in word_numbers.items():
        if word in text:
            return number
    match = re.search(r"\b(\d{1,2})\b", text)
    if match:
        return max(1, min(int(match.group(1)), 10))
    return default


def format_reviewable_spending_ranking(question: str, raw_result: str) -> str:
    data = parse_tool_json(raw_result)
    currency = data.get("currency", "")
    limit = requested_rank_limit(question)
    areas = [
        item
        for item in data.get("reviewable_spending_areas", [])
        if float(item.get("amount", 0) or 0) > 0
    ]
    areas.sort(key=lambda item: float(item.get("amount", 0) or 0), reverse=True)

    if not areas:
        return "No reviewable spending areas were returned by the financial review context."

    lines = []
    for index, item in enumerate(areas[:limit], start=1):
        lines.append(
            f"{index}. {display_spending_area_type(item.get('type'))} — "
            f"{money_text(item.get('amount', 0), currency)}"
        )
    return "\n".join(lines)


def commitment_label(item: dict[str, Any]) -> str:
    label = item.get("merchant") or item.get("label") or item.get("type") or item.get("payment_type") or ""
    if item.get("type") and not item.get("merchant") and not item.get("label"):
        return display_spending_area_type(label)
    return format_merchant_display(label)


def commitment_amount(item: dict[str, Any]) -> float:
    return float(item.get("total_amount", item.get("amount", 0)) or 0)


TRANSACTION_TYPE_COMMITMENT_LABELS = {
    "העברה דיגיטל",
    "הוראת קבע",
    "דמי כרטיס",
}

CONTRACTUAL_PAYMENT_TYPES = {
    "pension",
    "insurance",
    "education_and_courses",
}

REGULAR_BILL_PAYMENT_TYPES = {
    "communication_services",
    "utilities",
    "electricity",
    "gas",
    "subscriptions",
}

REGULAR_BILL_CATEGORIES = {
    "communication_services",
    "utilities",
    "electricity",
    "gas",
}

RECURRING_MERCHANT_CATEGORIES = {
    "food_and_groceries",
    "health_and_pharmacies",
    "restaurants_cafes_bars",
    "fuel",
    "transport_and_vehicles",
    "home_and_furniture",
    "leisure_entertainment_sports",
}

TAX_OR_GOVERNMENT_OBLIGATION_KEYWORDS = [
    "tax",
    "taxes",
    "municipality",
    "עיריית",
    "עירית",
    "ארנונה",
    "מס",
    "ביטוח לאומי",
    "המוסד לביטוח לאומי",
]


def is_payment_mechanism_commitment_item(item: dict[str, Any]) -> bool:
    label = str(item.get("label") or item.get("merchant") or "").strip()
    if item.get("source") == "transaction_type_label":
        return True
    if item.get("merchant_is_transaction_type"):
        return True
    return label in TRANSACTION_TYPE_COMMITMENT_LABELS


def recurring_commitment_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    commitments = []
    seen = set()

    for section_name in ["stable_recurring_obligations", "recurring_payments"]:
        for item in data.get(section_name, []):
            label = item.get("merchant") or item.get("label")
            if not label:
                continue
            candidate = {**item, "source": item.get("source") or section_name}
            if is_payment_mechanism_commitment_item(candidate):
                continue
            key = normalize_args({"label": label}).get("label", str(label))
            if key in seen:
                continue
            seen.add(key)
            commitments.append(candidate)

    if not commitments:
        for item in data.get("mandatory_spending_areas", []):
            commitments.append({**item, "source": "mandatory_spending_areas"})

    commitments = [
        item
        for item in commitments
        if commitment_amount(item) > 0
    ]
    commitments.sort(key=commitment_amount, reverse=True)
    return commitments


def recurring_merchant_items(data: dict[str, Any], excluded_labels: set[str] | None = None) -> list[dict[str, Any]]:
    excluded_labels = excluded_labels or set()
    merchants = []
    for item in data.get("recurring_merchants", []):
        label = normalize_entity_text(item.get("merchant"))
        if not label or label in excluded_labels:
            continue
        merchants.append({**item, "source": "recurring_merchants"})
    merchants = [item for item in merchants if commitment_amount(item) > 0]
    merchants.sort(key=commitment_amount, reverse=True)
    return merchants


def recurring_payment_method_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    methods = []
    seen = set()
    for section_name in ["transaction_type_labels", "stable_recurring_obligations"]:
        for item in data.get(section_name, []):
            candidate = {**item, "source": item.get("source") or section_name}
            if not is_payment_mechanism_commitment_item(candidate):
                continue
            label = candidate.get("label") or candidate.get("merchant")
            key = normalize_args({"label": label}).get("label", str(label))
            if not key or key in seen:
                continue
            seen.add(key)
            methods.append(candidate)
    methods.sort(key=commitment_amount, reverse=True)
    return methods


def recurring_item_group(item: dict[str, Any]) -> str:
    if is_payment_mechanism_commitment_item(item):
        return "payment_methods"

    payment_type = str(item.get("payment_type") or "")
    category = str(item.get("category") or item.get("type") or "")
    category_en = str(item.get("category_en") or "")
    label_text = " ".join(
        str(item.get(key) or "")
        for key in ["merchant", "label"]
    ).casefold()

    if payment_type in CONTRACTUAL_PAYMENT_TYPES:
        return "contractual_obligations"

    if any(keyword.casefold() in label_text for keyword in TAX_OR_GOVERNMENT_OBLIGATION_KEYWORDS):
        return "contractual_obligations"

    if category in RECURRING_MERCHANT_CATEGORIES or category_en in {
        "Food and groceries",
        "Health and pharmacies",
        "Restaurants, cafes and bars",
        "Fuel",
        "Transport and vehicles",
        "Home and furniture",
        "Leisure, entertainment and sports",
    }:
        return "recurring_merchants"

    if payment_type in REGULAR_BILL_PAYMENT_TYPES or category in REGULAR_BILL_CATEGORIES:
        return "regular_bills"

    if item.get("source") == "recurring_merchants":
        return "recurring_merchants"

    return "regular_bills"


def recurring_item_explanation(item: dict[str, Any], group: str) -> str:
    payment_type = item.get("payment_type")
    category = item.get("category_en") or item.get("category") or item.get("type")
    source = item.get("source")
    months = item.get("months_present")

    recurrence = (
        f" and appears in {months} months"
        if months
        else ""
    )

    if group == "contractual_obligations":
        if payment_type in CONTRACTUAL_PAYMENT_TYPES:
            return f"Classified as a contractual obligation because payment_type is '{payment_type}'{recurrence}."
        if any(
            keyword.casefold() in " ".join(str(item.get(key) or "") for key in ["merchant", "label"]).casefold()
            for keyword in TAX_OR_GOVERNMENT_OBLIGATION_KEYWORDS
        ):
            return f"Classified as a contractual obligation because the merchant or label contains a tax/government obligation marker{recurrence}."
        return f"Classified as a contractual obligation because category is '{category}'{recurrence}."

    if group == "regular_bills":
        if payment_type:
            return f"Classified as a regular recurring bill because payment_type is '{payment_type}'{recurrence}."
        return f"Classified as a regular recurring bill because category is '{category}'{recurrence}."

    if group == "payment_methods":
        usage_hint = item.get("usage_hint")
        if usage_hint:
            return (
                "This is a payment mechanism, not the recipient of the money; "
                f"the structured transaction pattern indicates it is {usage_hint}{recurrence}."
            )
        return (
            "This is a payment mechanism, not the recipient of the money; "
            f"it is classified from source '{source}' or a transaction-mechanism label{recurrence}."
        )

    return f"Classified as a recurring merchant because source is '{source}'{recurrence}."


def asks_to_classify_payment_methods(question: str) -> bool:
    text = question.casefold()
    markers = [
        "payment methods",
        "actual obligations",
        "classify recurring payments",
        "obligations vs payment methods",
        "which are payment methods",
        "just payment methods",
        "transaction type labels",
        "standing order",
        "digital transfer",
    ]
    return any(marker in text for marker in markers)


def format_commitment_line(index: int, item: dict[str, Any], currency: str) -> str:
    details = []
    if item.get("months_present"):
        details.append(f"{item.get('months_present')} months")
    if item.get("transactions"):
        details.append(f"{item.get('transactions')} {pluralize(item.get('transactions'), 'transaction')}")
    detail_text = f" ({', '.join(details)})" if details else ""
    return (
        f"{index}. {commitment_label(item)} — "
        f"{money_text(commitment_amount(item), currency)}{detail_text}"
    )


def format_payment_mechanism_line(index: int, item: dict[str, Any]) -> str:
    details = ["payment mechanism, not a recipient"]
    if item.get("transactions"):
        details.append(f"{item.get('transactions')} {pluralize(item.get('transactions'), 'transaction')}")
    if item.get("months_present"):
        details.append(f"used in {item.get('months_present')} months")
    if item.get("usage_hint"):
        details.append(str(item.get("usage_hint")))
    return f"{index}. {commitment_label(item)} — {', '.join(details)}"


def format_commitment_line_with_explanation(
    index: int,
    item: dict[str, Any],
    currency: str,
    group: str,
) -> str:
    if group == "payment_methods":
        return (
            f"{format_payment_mechanism_line(index, item)}\n"
            f"   {recurring_item_explanation(item, group)}"
        )
    return (
        f"{format_commitment_line(index, item, currency)}\n"
        f"   {recurring_item_explanation(item, group)}"
    )


def format_recurring_payment_classification(question: str, data: dict[str, Any]) -> str:
    currency = data.get("currency", "")
    limit = requested_rank_limit(question, default=5)
    candidates = recurring_commitment_items(data) + recurring_payment_method_items(data)
    excluded = {
        normalize_entity_text(item.get("merchant") or item.get("label"))
        for item in candidates
    }
    candidates += recurring_merchant_items(data, excluded)

    grouped = {
        "contractual_obligations": [],
        "regular_bills": [],
        "payment_methods": [],
        "recurring_merchants": [],
    }
    for item in candidates:
        grouped[recurring_item_group(item)].append(item)

    section_titles = [
        ("contractual_obligations", "Contractual financial obligations:"),
        ("regular_bills", "Regular recurring bills:"),
        ("payment_methods", "Payment methods / transaction mechanisms:"),
        ("recurring_merchants", "Recurring merchants:"),
    ]

    lines = []
    for group, title in section_titles:
        lines.append(title)
        items = grouped[group]
        if items:
            lines.extend(
                format_commitment_line_with_explanation(index, item, currency, group)
                for index, item in enumerate(items[:limit], start=1)
            )
        else:
            lines.append("No items in this group were returned by the financial review context.")
        lines.append("")

    return "\n".join(lines).strip()


def format_recurring_commitments_ranking(question: str, raw_result: str) -> str:
    data = parse_tool_json(raw_result)
    currency = data.get("currency", "")
    limit = requested_rank_limit(question)

    if asks_to_classify_payment_methods(question):
        return format_recurring_payment_classification(question, data)

    commitments = [
        item
        for item in recurring_commitment_items(data)
        if recurring_item_group(item) == "contractual_obligations"
    ]

    if not commitments:
        return "No recurring obligations or financial commitments were returned by the financial review context."

    return "\n".join(
        format_commitment_line_with_explanation(index, item, currency, "contractual_obligations")
        for index, item in enumerate(commitments[:limit], start=1)
    )


def format_insight_subtype_answer(question: str, subtype: str, raw_result: str) -> str | None:
    """
    Return pure-Python answers for insight subtypes that must stay deterministic.

    Ranking and commitments are factual enough that they should not rely on
    free-form generation.
    """

    if subtype == "ranking":
        return format_reviewable_spending_ranking(question, raw_result)
    if subtype == "commitments":
        return format_recurring_commitments_ranking(question, raw_result)
    return None


COMMON_INSIGHT_CONTEXT_FIELDS = [
    "currency",
    "currency_symbol",
    "months_covered",
    "month_coverage",
    "amount_calculation_basis",
]


def common_insight_context(full_context: dict[str, Any], subtype: str) -> dict[str, Any]:
    """Start every focused context with shared metadata needed by prompts and guards."""

    context = {
        key: full_context.get(key)
        for key in COMMON_INSIGHT_CONTEXT_FIELDS
        if key in full_context
    }
    context["context_selection"] = {
        "subtype": subtype,
        "source": "prepare_financial_review_context",
        "note": "Reduced subtype-specific context selected deterministically before LLM reasoning.",
    }
    return context


def sorted_by_amount(items: list[dict[str, Any]], amount_key: str = "amount") -> list[dict[str, Any]]:
    """Sort analytics items by an amount-like field for ranked context slices."""

    return sorted(
        items,
        key=lambda item: float(item.get(amount_key, item.get("total_amount", 0)) or 0),
        reverse=True,
    )


def with_merchant_display(item: dict[str, Any]) -> dict[str, Any]:
    """Add a deterministic display label while preserving the original merchant value."""

    enriched = dict(item)
    merchant = enriched.get("merchant") or enriched.get("label")
    if merchant:
        enriched["merchant_display"] = format_merchant_display(str(merchant))
    return enriched


def add_merchant_display_fields(value: Any) -> Any:
    """Recursively enrich context with display-only merchant labels."""

    if isinstance(value, dict):
        enriched = {
            key: add_merchant_display_fields(item)
            for key, item in value.items()
        }
        merchant = enriched.get("merchant") or enriched.get("label")
        if merchant:
            enriched["merchant_display"] = format_merchant_display(str(merchant))
        return enriched
    if isinstance(value, list):
        return [add_merchant_display_fields(item) for item in value]
    return value


def reviewable_top_categories(full_context: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter top categories down to areas that can be discussed as reviewable."""

    return [
        item
        for item in full_context.get("top_categories", [])
        if str(item.get("spending_classification") or "").casefold() != "mandatory_fixed"
    ]


def is_reviewable_context_item(item: dict[str, Any]) -> bool:
    """Heuristically identify context items suitable for savings/review recommendations."""

    item_type = str(item.get("type") or item.get("category") or "").casefold()
    classification = str(item.get("spending_classification") or item.get("classification") or "").casefold()
    evidence = str(item.get("evidence") or item.get("reason") or "").casefold()
    return (
        "reviewable" in classification
        or "discretionary" in classification
        or "reviewable" in evidence
        or "discretionary" in evidence
        or item_type in {
            "shopping",
            "restaurants",
            "food_delivery",
            "leisure_entertainment_sports",
            "flights_and_travel",
            "subscriptions",
            "home_and_furniture",
            "restaurants_cafes_bars",
        }
    )


def payment_mechanism_labels(full_context: dict[str, Any]) -> set[str]:
    """Collect transaction-type labels that should not be ranked as merchants."""

    labels = set()
    for item in full_context.get("transaction_type_labels", []):
        label = normalize_entity_text(item.get("label") or item.get("merchant"))
        if label:
            labels.add(label)
    return labels


def build_merchant_impact_context(context: dict[str, Any], full_context: dict[str, Any]) -> None:
    """
    Build merchant-impact candidates while excluding payment mechanisms.

    This prevents labels such as Digital Transfer or Standing Order from being
    presented as merchants that affect the user's budget.
    """

    excluded = payment_mechanism_labels(full_context)
    context["excluded_payment_mechanisms"] = [
        item
        for item in full_context.get("transaction_type_labels", [])
        if normalize_entity_text(item.get("label") or item.get("merchant")) in excluded
    ]

    merchant_candidates = []
    for item in full_context.get("recurring_merchants", []):
        merchant = normalize_entity_text(item.get("merchant"))
        if merchant and merchant not in excluded:
            candidate = with_merchant_display(item)
            amount = float(candidate.get("total_amount", candidate.get("amount", 0)) or 0)
            months_present = int(candidate.get("months_present", 0) or 0)
            if months_present:
                candidate["estimated_monthly_average"] = round(amount / months_present, 2)
            candidate["source"] = "recurring_merchants"
            merchant_candidates.append(candidate)
    context["merchant_candidates"] = sorted_by_amount(merchant_candidates, "total_amount")[:15]

    reviewable_candidates = []
    for area in full_context.get("reviewable_spending_areas", []):
        for merchant in area.get("top_merchants", []):
            merchant_name = normalize_entity_text(merchant.get("merchant"))
            if not merchant_name or merchant_name in excluded:
                continue
            reviewable_candidates.append(
                with_merchant_display({
                    **merchant,
                    "reviewable_area": area.get("type"),
                    "reviewable_area_amount": area.get("amount"),
                    "calculation_basis": area.get("calculation_basis"),
                    "source": "reviewable_spending_area",
                })
            )
    context["reviewable_merchant_candidates"] = sorted_by_amount(reviewable_candidates, "amount")[:15]


def build_insight_context(subtype: str, full_context: dict[str, Any], question: str) -> dict[str, Any]:
    """
    Reduce the full financial review JSON to the fields needed by one subtype.

    Smaller focused contexts reduce hallucination risk and make the NLP debug
    output easier to explain during demos and reviews.
    """

    if subtype not in FINANCIAL_REVIEW_SUBTYPES and subtype not in {"ranking", "commitments", "compare_months"}:
        subtype = "general"

    context = common_insight_context(full_context, subtype)

    if subtype == "financial_coach":
        context["reviewable_spending_areas"] = full_context.get("reviewable_spending_areas", [])
        context["possible_review_areas"] = [
            item
            for item in full_context.get("possible_review_areas", [])
            if is_reviewable_context_item(item)
        ]
        context["top_categories"] = reviewable_top_categories(full_context)
        context["mandatory_spending_areas"] = full_context.get("mandatory_spending_areas", [])
        context["stable_recurring_obligations"] = full_context.get("stable_recurring_obligations", [])
        return add_merchant_display_fields(context)

    if subtype == "lifestyle":
        context["top_categories"] = full_context.get("top_categories", [])
        context["reviewable_spending_areas"] = full_context.get("reviewable_spending_areas", [])
        context["recurring_merchants"] = sorted_by_amount(
            full_context.get("recurring_merchants", []),
            "total_amount",
        )[:10]
        context["recurring_payments"] = [
            {
                "payment_type": item.get("payment_type"),
                "category": item.get("category"),
                "category_en": item.get("category_en"),
                "months_present": item.get("months_present"),
                "transactions": item.get("transactions"),
                "total_amount": item.get("total_amount"),
            }
            for item in full_context.get("recurring_payments", [])
        ]
        context["stable_recurring_obligations"] = full_context.get("stable_recurring_obligations", [])
        return add_merchant_display_fields(context)

    if subtype == "commitments":
        context["recurring_payments"] = full_context.get("recurring_payments", [])
        context["stable_recurring_obligations"] = full_context.get("stable_recurring_obligations", [])
        context["mandatory_spending_areas"] = full_context.get("mandatory_spending_areas", [])
        context["transaction_type_labels"] = full_context.get("transaction_type_labels", [])
        context["recurring_merchants_reference"] = full_context.get("recurring_merchants", [])[:10]
        return add_merchant_display_fields(context)

    if subtype == "mortgage_review":
        context["monthly_spending_totals"] = full_context.get("monthly_spending_totals", [])
        context["top_categories"] = full_context.get("top_categories", [])
        context["mandatory_spending_areas"] = full_context.get("mandatory_spending_areas", [])
        context["reviewable_spending_areas"] = full_context.get("reviewable_spending_areas", [])
        context["stable_recurring_obligations"] = full_context.get("stable_recurring_obligations", [])
        context["unusual_transactions"] = sorted_by_amount(
            full_context.get("unusual_transactions", []),
            "amount",
        )[:5]
        context["largest_expenses"] = sorted_by_amount(
            full_context.get("largest_expenses", []),
            "amount",
        )[:5]
        return add_merchant_display_fields(context)

    if subtype == "merchant_impact":
        build_merchant_impact_context(context, full_context)
        return add_merchant_display_fields(context)

    context["top_categories"] = full_context.get("top_categories", [])
    context["category_concentration"] = full_context.get("category_concentration", {})
    context["reviewable_spending_areas"] = full_context.get("reviewable_spending_areas", [])
    context["mandatory_spending_areas"] = full_context.get("mandatory_spending_areas", [])
    context["stable_recurring_obligations"] = full_context.get("stable_recurring_obligations", [])
    return add_merchant_display_fields(context)


def format_insight_key_facts(raw_result: str) -> str:
    data = parse_tool_json(raw_result)
    if not data:
        return ""

    currency = data.get("currency", "")
    months = data.get("months_covered", [])
    calculation_basis = data.get("amount_calculation_basis", "")
    lines = ["**Deterministic Key Facts**"]

    if months:
        lines.append(f"- Period: {', '.join(str(month) for month in months)}")
    if data.get("total_spent") is not None:
        lines.append(f"- Total reviewed spending: {money_text(data.get('total_spent'), currency)}")
    if calculation_basis:
        lines.append(f"- Amount basis: `{calculation_basis}`")

    reviewable = data.get("reviewable_spending_areas", [])
    if reviewable:
        lines.append("- Reviewable spending areas:")
        for item in reviewable[:6]:
            type_label = str(item.get("type", "")).replace("_", " ")
            lines.append(
                f"  - {type_label}: {money_text(item.get('amount', 0), currency)} "
                f"({item.get('transactions', 0)} {pluralize(item.get('transactions', 0), 'transaction')}, "
                f"{item.get('debit_transactions', 0)} debit / {item.get('credit_transactions', 0)} credit)"
            )
            top_merchants = item.get("top_merchants", [])
            if top_merchants:
                merchant_text = ", ".join(
                    f"{format_merchant_display(merchant.get('merchant'))}: "
                    f"{money_text(merchant.get('amount', 0), currency)}"
                    for merchant in top_merchants[:3]
                )
                lines.append(f"    Top merchants: {merchant_text}")

    categories = data.get("top_categories", [])
    if categories:
        lines.append("- Top categories:")
        for item in categories[:5]:
            lines.append(
                f"  - {item.get('category_en') or item.get('category')}: "
                f"{money_text(item.get('total_amount', 0), currency)} "
                f"({item.get('percent_of_total', 0)}%)"
            )

    stable = data.get("stable_recurring_obligations", [])
    if stable:
        lines.append(f"- Stable recurring obligations detected: {len(stable)}")

    return "\n".join(lines)


def format_deterministic_answer(question: str, tool_name: str, raw_result: str) -> str:
    data = parse_tool_json(raw_result)
    if not data:
        return "The tool did not return valid JSON, so I cannot format a deterministic answer."

    currency = data.get("currency", "")

    if tool_name == "finance.get_spending_summary":
        month = data.get("month", "")
        total = money_text(data.get("total_spent", 0), currency)
        count = data.get("transaction_count", 0)
        average = money_text(data.get("average_transaction", 0), currency)
        return f"For {month}, total spending was {total} across {count} transactions. Average transaction: {average}."

    if tool_name == "finance.get_top_merchants":
        month = data.get("month", "")
        merchants = data.get("merchants", [])
        header = f"Top merchants for {month}:" if month else "Top merchants:"
        return header + "\n" + item_lines(
            merchants,
            lambda item, index: (
                f"{index}. {format_merchant_display(item.get('merchant', ''))}: "
                f"{money_text(item.get('amount', 0), currency)} "
                f"({item.get('transactions', 0)} {pluralize(item.get('transactions', 0), 'transaction')})"
            ),
            "No merchants were returned by the tool.",
        )

    if tool_name == "finance.get_category_breakdown":
        month = data.get("month", "")
        categories = data.get("categories", [])
        header = f"Categories for {month}:" if month else "Categories:"
        return header + "\n" + item_lines(
            categories,
            lambda item, index: (
                f"{index}. {item.get('category_en') or item.get('category', '')}: "
                f"{money_text(item.get('amount', 0), currency)} "
                f"({item.get('transactions', 0)} transactions)"
            ),
            "No categories were returned by the tool.",
        )

    if tool_name == "finance.get_recurring_merchants":
        months = data.get("months", [])
        merchants = data.get("merchants_every_month", [])
        month_text = ", ".join(months)
        header = f"Merchants appearing in every selected month ({month_text}):"
        return header + "\n" + item_lines(
            merchants,
            lambda item, index: (
                f"{index}. {format_merchant_display(item.get('merchant', ''))}\n"
                f"   Months: {item.get('months_present', 0)} ({', '.join(item.get('months', []))})\n"
                f"   Average amount: {money_text(item.get('average_amount', 0), currency)}\n"
                f"   Total amount: {money_text(item.get('total_amount', 0), currency)}\n"
                f"   Transactions: {item.get('transactions', 0)}"
            ),
            "No merchants appeared in every selected month.",
        )

    if tool_name == "finance.get_unusual_transactions":
        month = data.get("month", "")
        if "unusual_by_month" in data:
            sections = ["Unusual transactions by month:"]
            has_transactions = False
            for group in data.get("unusual_by_month", []):
                group_month = group.get("month", "")
                transactions = group.get("large_expenses_to_review", [])
                sections.append(f"\n{group_month}:")
                if not transactions:
                    sections.append("No unusual transactions returned by the tool.")
                    continue
                has_transactions = True
                sections.append(
                    item_lines(
                        transactions,
                        lambda item, index: (
                            f"{index}. {item.get('transaction_date', '')} - "
                            f"{format_merchant_display(item.get('merchant', ''))}: "
                            f"{money_text(item.get('amount', 0), currency)} "
                            f"({item.get('normalized_category_en') or item.get('category_en') or item.get('normalized_category') or ''})"
                        ),
                        "No unusual transactions returned by the tool.",
                    )
                )
            if not has_transactions:
                return "No unusual transactions were returned by the tool for any month."
            return "\n".join(sections)

        transactions = data.get("large_expenses_to_review", [])
        header = f"Unusual transactions for {month}:" if month else "Unusual transactions:"
        return header + "\n" + item_lines(
            transactions,
            lambda item, index: (
                f"{index}. {item.get('transaction_date', '')} - "
                f"{format_merchant_display(item.get('merchant', ''))}: "
                f"{money_text(item.get('amount', 0), currency)} "
                f"({item.get('normalized_category_en') or item.get('category_en') or item.get('normalized_category') or ''})"
            ),
            "No unusual transactions were returned by the tool.",
        )

    if tool_name == "finance.search_transactions":
        keyword = data.get("keyword", "")
        matches = data.get("matches", [])
        header = f"Transactions matching {keyword}:"
        return header + "\n" + item_lines(
            matches,
            lambda item, index: (
                f"{index}. {item.get('transaction_date', '')} - "
                f"{format_merchant_display(item.get('merchant', ''))}: "
                f"{money_text(item.get('amount_abs', item.get('amount', 0)), item.get('currency', currency))} "
                f"({item.get('category_en') or item.get('category') or ''})"
            ),
            "No matching transactions were returned by the tool.",
        )

    if tool_name == "finance.get_category_trend":
        category = data.get("category_en") or data.get("category", "")
        totals = data.get("monthly_totals", [])
        header = f"{category} by month:"
        return header + "\n" + item_lines(
            totals,
            lambda item, index: (
                f"{index}. {item.get('month', '')}: "
                f"{money_text(item.get('amount', 0), currency)} "
                f"({item.get('transactions', 0)} {pluralize(item.get('transactions', 0), 'transaction')})"
            ),
            "No monthly totals were returned by the tool.",
        )

    if tool_name == "finance.compare_months":
        month_a = data.get("month_a", "")
        month_b = data.get("month_b", "")
        lines = []
        for coverage_key in ["month_a_coverage", "month_b_coverage"]:
            coverage = data.get(coverage_key) or {}
            if coverage and not coverage.get("is_complete_month", True):
                lines.append(
                    "Warning: "
                    f"{coverage.get('month', '')} appears incomplete "
                    f"({coverage.get('coverage_days', '?')}/{coverage.get('expected_days', '?')} days), "
                    "so differences may reflect partial data rather than a full month-to-month trend."
                )
        lines.extend([
            f"{month_a}: {money_text(data.get('month_a_total', 0), currency)}",
            f"{month_b}: {money_text(data.get('month_b_total', 0), currency)}",
            f"Difference: {money_text(data.get('spending_difference', 0), currency)}",
        ])
        if data.get("percentage_change") is not None:
            lines.append(f"Percentage change: {data.get('percentage_change')}%")
        changes = data.get("largest_category_changes", [])
        if changes:
            lines.append("Category changes returned by the tool:")
            lines.extend(
                f"{index}. {item.get('category_en') or item.get('category', '')}: "
                f"{money_text(item.get('month_a_amount', 0), currency)} -> "
                f"{money_text(item.get('month_b_amount', 0), currency)} "
                f"(difference {money_text(item.get('difference', 0), currency)})"
                for index, item in enumerate(changes, start=1)
            )
        return "\n".join(lines)

    if tool_name == "finance.get_category_comparison":
        category = data.get("category_en") or data.get("category", "")
        months = data.get("months", [])
        lines = [f"{category} comparison:"]
        lines.extend(
            f"{item.get('month', '')}: {money_text(item.get('amount', 0), currency)} "
            f"({item.get('transactions', 0)} {pluralize(item.get('transactions', 0), 'transaction')})"
            for item in months
        )
        lines.append(f"Difference: {money_text(data.get('difference', 0), currency)}")
        return "\n".join(lines)

    if tool_name == "finance.get_largest_transactions":
        month = data.get("month", "")
        transactions = data.get("transactions", [])
        header = f"Largest transactions for {month}:" if month else "Largest transactions:"
        return header + "\n" + item_lines(
            transactions,
            lambda item, index: (
                f"{index}. {item.get('transaction_date', '')} - "
                f"{format_merchant_display(item.get('merchant', ''))}: "
                f"{money_text(item.get('amount', 0), currency)} "
                f"({item.get('category_en') or item.get('category') or ''})"
            ),
            "No transactions were returned by the tool.",
        )

    return "Tool JSON result:\n```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

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
    """Normalize obvious currency wording mistakes in interpretive answers."""

    cleaned = str(text)
    symbol = "₪" if currency == "NIS" else currency

    cleaned = re.sub(r"\$\s*([\d,.]+)", lambda match: f"{symbol}{match.group(1)}", cleaned)
    cleaned = re.sub(
        r"([\d,.]+)\s*(?:USD|usd|US dollars|U\.S\. dollars|dollars|Dollars)",
        lambda match: f"{symbol}{match.group(1)}",
        cleaned,
    )
    cleaned = re.sub(r"\b(?:USD|usd|US dollars|U\.S\. dollars|dollars|Dollars)\b", symbol, cleaned)

    return cleaned


def postprocess_insight_currency(text: str, currency: str = "NIS") -> str:
    """
    Enforce the currency selected by deterministic financial context.

    Local models sometimes default to USD; this post-processing keeps NIS answers
    visually consistent by using the shekel symbol.
    """

    cleaned = str(text)
    if currency != "NIS":
        return cleaned

    if "$" in cleaned:
        logger.warning(
            "INSIGHT_CURRENCY_SYMBOL_MISMATCH currency=NIS contains_dollar=true action=replace_symbol"
        )
        cleaned = cleaned.replace("$", "₪")

    if re.search(r"\b(?:USD|usd|US dollars|U\.S\. dollars|dollars|Dollars)\b", cleaned):
        logger.warning(
            "INSIGHT_CURRENCY_LABEL_MISMATCH currency=NIS contains_usd_label=true action=replace_label"
        )
        cleaned = re.sub(
            r"\b(?:USD|usd|US dollars|U\.S\. dollars|dollars|Dollars)\b",
            "₪",
            cleaned,
        )

    cleaned = re.sub(r"\bNIS\b", "₪", cleaned)

    return cleaned


GENERIC_SUMMARY_OPENING_RE = re.compile(
    r"^\s*(?:the\s+)?(?:report|analysis|data|dataset|transactions?)\s+"
    r"(?:shows?|indicates?|reveals?|highlights?|presents?)\b",
    re.IGNORECASE,
)


def insight_answer_validation_errors(subtype: str, answer: str) -> list[str]:
    """
    Validate whether an insight answer follows the subtype response contract.

    These checks catch product-facing issues that prompts alone did not reliably
    prevent, such as generic summaries, wrong currency labels, or recommendations
    to cut mandatory obligations.
    """

    text = str(answer or "")
    lowered = text.casefold()
    errors = []

    if GENERIC_SUMMARY_OPENING_RE.search(text):
        errors.append("starts with a generic dataset/report summary")

    if re.search(r"\b(?:HKD|EUR|GBP|USD)\b", text):
        errors.append("contains unsupported currency label")

    if subtype == "financial_coach":
        required = ["recommendation", "current spending", "estimated monthly savings", "evidence"]
        for marker in required:
            if marker not in lowered:
                errors.append(f"missing {marker}")
        mandatory_terms = [
            "rent",
            "pension",
            "insurance",
            "education",
            "tax",
            "taxes",
            "utilities",
            "utility",
            "standing order",
            "standing orders",
            "digital transfer",
            "stable_recurring_obligations",
            "recurring obligation",
            "recurring obligations",
            "debit card",
            "card fee",
            "mor pension fund",
            "digital transfer insurance",
        ]
        for line in text.splitlines():
            line_lowered = line.casefold()
            if "recommendation" in line_lowered and any(term in line_lowered for term in mandatory_terms):
                errors.append("recommends mandatory obligations as savings targets")
                break

    if subtype == "lifestyle":
        if "evidence:" not in lowered:
            errors.append("missing evidence lines")
        if len(re.findall(r"(?m)^\s*\d+\.", text)) > 5:
            errors.append("more than five observations")

    if subtype == "mortgage_review":
        for marker in ["why", "evidence"]:
            if marker not in lowered:
                errors.append(f"missing {marker}")

    if subtype == "merchant_impact":
        if "estimated" not in lowered and "monthly impact" not in lowered:
            errors.append("missing estimated monthly impact")
        if "evidence" not in lowered:
            errors.append("missing evidence")
        if not re.search(r"(?m)^\s*(?:\d+\.|-)\s+", text):
            errors.append("missing ranked merchant list")

    return errors


def fatal_insight_validation_errors(subtype: str, errors: list[str]) -> list[str]:
    """
    Separate fatal safety issues from softer formatting issues.

    Nonfatal issues can be accepted after a rewrite so the UI does not overuse
    verified fallback for imperfect but usable answers.
    """

    fatal = []
    for error in errors:
        if error == "contains unsupported currency label":
            fatal.append(error)
        elif error == "recommends mandatory obligations as savings targets":
            fatal.append(error)
        elif subtype == "merchant_impact":
            fatal.append(error)
    return fatal


def call_llm_with_insight_retry(
    messages: list[dict[str, str]],
    subtype: str,
    question: str,
) -> str:
    """
    Ask the LLM for an insight answer and retry once if it violates the template.

    The retry prompt is corrective only; the financial facts still come from the
    deterministic context already included in the original messages.
    """

    answer = call_llm(messages)
    errors = insight_answer_validation_errors(subtype, answer)
    if not errors:
        return answer

    logger.warning(
        "INSIGHT_ANSWER_VALIDATION_FAILED subtype=%s errors=%s question=%s",
        subtype,
        json.dumps(errors, ensure_ascii=False),
        question,
    )

    retry_messages = messages + [
        {
            "role": "assistant",
            "content": answer,
        },
        {
            "role": "user",
            "content": (
                "Your previous answer did not follow the required response template. "
                f"Problems: {', '.join(errors)}. Rewrite the answer now. "
                "Do not summarize the dataset. Start by directly answering the question. "
                "Use only evidence from the provided JSON context and do not invent merchant names."
            ),
        },
    ]
    retry_answer = call_llm(retry_messages)
    retry_errors = insight_answer_validation_errors(subtype, retry_answer)
    fatal_errors = fatal_insight_validation_errors(subtype, retry_errors)
    if fatal_errors and subtype == "merchant_impact":
        logger.warning(
            "INSIGHT_ANSWER_RETRY_VALIDATION_FAILED subtype=%s errors=%s question=%s",
            subtype,
            json.dumps(fatal_errors, ensure_ascii=False),
            question,
        )
        return SAFE_GROUNDING_FALLBACK_PREFIX
    if fatal_errors:
        logger.warning(
            "INSIGHT_ANSWER_RETRY_RETURNED_FOR_SUBTYPE_GUARD subtype=%s errors=%s question=%s",
            subtype,
            json.dumps(fatal_errors, ensure_ascii=False),
            question,
        )
    if retry_errors:
        logger.info(
            "INSIGHT_ANSWER_RETRY_ACCEPTED_WITH_NONFATAL_ERRORS subtype=%s errors=%s question=%s",
            subtype,
            json.dumps(retry_errors, ensure_ascii=False),
            question,
        )
    return retry_answer


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


QUERY_TOOL_NAMES = {
    "finance.get_spending_summary",
    "finance.get_category_breakdown",
    "finance.get_top_merchants",
    "finance.get_largest_transactions",
    "finance.get_unusual_transactions",
    "finance.compare_months",
    "finance.get_category_comparison",
    "finance.get_category_trend",
    "finance.get_recurring_merchants",
    "finance.prepare_financial_review_context",
    "finance.search_transactions",
}

QUERY_TOOL_ALIASES = {
    "finance.get_top_categories": "finance.get_category_breakdown",
}

INSIGHT_WORKFLOW_TOOLS = [
    "finance.prepare_financial_review_context",
]


def invalid_financial_tool_result(
    question: str,
    tool_name: str,
    args: dict[str, Any],
    response_mode: str,
    insight_subtype: str,
    reason: str,
    error: str,
    nlp_debug: dict[str, Any] | None = None,
) -> AgentResult:
    raw_result = {
        "ok": False,
        "error": error,
        "selected_tool": tool_name,
        "args": args,
    }
    return {
        "question": question,
        "tool": tool_name,
        "args": args,
        "reason": reason,
        "raw_result": json.dumps(raw_result, ensure_ascii=False),
        "answer": "I could not select a valid financial query tool for this question.",
        "key_facts": "",
        "response_mode": response_mode,
        "insight_subtype": insight_subtype,
        "nlp_debug": nlp_debug or {},
        "timing": {
            "tool_execution_seconds": 0,
        },
    }


async def run_financial_question(question: str) -> AgentResult:
    """
    Answer a natural-language financial question using one read-only MCP query tool.

    The LLM chooses the tool and arguments, but the tool performs deterministic
    SQLite analytics. No SQL is generated by the LLM.
    """

    manager = MCPManager()
    try:
        # Stage 1: classify the question before selecting any MCP tool.
        # This keeps advice/coaching questions away from factual reporting tools.
        financial_intent = classify_financial_intent(question)
        response_mode = classify_response_mode(question)
        deterministic_subtype = deterministic_insight_subtype(question) if response_mode == "insight" else None
        llm_classifier_used = response_mode == "insight" and deterministic_subtype is None
        insight_subtype = classify_insight_subtype(question, use_llm=True) if response_mode == "insight" else ""
        nlp_debug = {
            "deterministic_router": "matched" if response_mode == "deterministic" else "skipped",
            "llm_classifier_used": "yes" if llm_classifier_used else "no",
            "intent": financial_intent,
            "response_mode": response_mode,
            "subtype": insight_subtype or "n/a",
            "focused_context_keys": [],
            "guard": "not_applicable",
        }
        logger.info(
            "NLP_INTENT=%s NLP_RESPONSE_MODE=%s NLP_INSIGHT_SUBTYPE=%s question=%s",
            financial_intent,
            response_mode,
            insight_subtype,
            question,
        )

        tools = await manager.connect()
        query_tools = {
            name: meta
            for name, meta in tools.items()
            if name in QUERY_TOOL_NAMES
        }
        tools_text = tools_to_text(query_tools)
        today = datetime.now().date().isoformat()

        if response_mode == "insight":
            # Insight workflows always begin with deterministic review context;
            # the subtype later decides which context fields are exposed to the LLM.
            tool_name = INSIGHT_WORKFLOW_TOOLS[0]
            args = normalize_query_tool_args(tool_name, {})
            reason = (
                "Insight questions begin with deterministic financial review context. "
                f"intent={financial_intent} subtype={insight_subtype}"
            )

            if tool_name not in query_tools:
                logger.error(
                    "NLP_INVALID_TOOL selected=%s response_mode=%s reason=%s",
                    tool_name,
                    response_mode,
                    "required insight tool unavailable",
                )
                return invalid_financial_tool_result(
                    question=question,
                    tool_name=tool_name,
                    args=args,
                    response_mode=response_mode,
                    insight_subtype=insight_subtype,
                    reason=reason,
                    error=f"Required insight tool is unavailable: {tool_name}",
                    nlp_debug=nlp_debug,
                )
        else:
            # For factual questions, the LLM only chooses a predefined query tool.
            # The tool itself performs deterministic SQLite analytics.
            selection_text = call_llm(
                [
                    {
                        "role": "system",
                        "content": f"""
You are a read-only financial query router.
Today is {today}.
Return ONLY one JSON object.
Do NOT generate SQL.
Do NOT call finance.analyze_statement.
Choose exactly one available Finance MCP query tool.

Month resolution rules:
- Use YYYY-MM when the user gives a specific month.
- English month names are supported. If the user gives a month name without a year, use the current year.
- Examples for the current year: May -> 2026-05, April -> 2026-04, June -> 2026-06.
- current month and last month may be passed as text; the tool can resolve them.
- For tools with a months argument, pass an integer only. If the user says "every month" without a number, omit months.
- For tools with a limit argument, pass an integer only.
- For one category across exactly two named months, use finance.get_category_comparison with category, month_a, and month_b.
- Example: "Show category Other for May and June" -> finance.get_category_comparison with category "Other", month_a "2026-05", month_b "2026-06".
- For largest transaction questions, use finance.get_largest_transactions.
- For unusual transaction questions that say "all months", "for all months", "all period", or "entire period", use finance.get_unusual_transactions with month "all".
- For unusual transaction questions with a month range, pass month as "YYYY-MM:YYYY-MM".
- Example: "Show unusual transactions from January to June" -> finance.get_unusual_transactions with month "2026-01:2026-06".
- For finance.search_transactions, copy the search keyword exactly from the user question.
- Do not translate, transliterate, correct spelling, normalize, or rewrite mixed-language search text.

Valid response format:
{{
  "tool": "finance.tool_name",
  "args": {{}},
  "reason": "short reason"
}}
""",
                    },
                    {
                        "role": "user",
                        "content": f"""
Available read-only tools:
{tools_text}

Question:
{question}
""",
                    },
                ]
            )
            action = extract_json(selection_text)
            logger.info("NLP_ROUTER_RAW=%s", selection_text)
            logger.info("NLP_ROUTER_ACTION=%s", json.dumps(action, ensure_ascii=False, default=str))
            tool_name = action.get("tool")
            if tool_name in QUERY_TOOL_ALIASES:
                aliased_tool = QUERY_TOOL_ALIASES[tool_name]
                logger.warning(
                    "NLP_TOOL_ALIAS selected=%s aliased_to=%s question=%s",
                    tool_name,
                    aliased_tool,
                    question,
                )
                tool_name = aliased_tool
            args = normalize_query_tool_args(tool_name, action.get("args", {}))
            tool_name, args = maybe_redirect_query_tool(question, tool_name, args)
            if tool_name in QUERY_TOOL_ALIASES:
                aliased_tool = QUERY_TOOL_ALIASES[tool_name]
                logger.warning(
                    "NLP_TOOL_ALIAS selected=%s aliased_to=%s question=%s",
                    tool_name,
                    aliased_tool,
                    question,
                )
                tool_name = aliased_tool
            args = normalize_query_tool_args(tool_name, args)
            reason = action.get("reason", "")
            logger.info(
                "NLP_ROUTER_FINAL tool=%s args=%s",
                tool_name,
                json.dumps(args, ensure_ascii=False, default=str),
            )
            if tool_name == "finance.search_transactions":
                keyword = str(args.get("keyword", ""))
                if keyword and keyword not in question:
                    logger.warning(
                        "NLP_SEARCH_KEYWORD_NOT_IN_ORIGINAL keyword=%s question=%s",
                        keyword,
                        question,
                    )

        # Stage 2: validate the selected tool before execution so UI errors stay graceful.
        if tool_name not in query_tools:
            error = f"LLM selected non-query or unavailable tool: {tool_name}"
            logger.error(
                "NLP_INVALID_TOOL selected=%s response_mode=%s args=%s question=%s",
                tool_name,
                response_mode,
                json.dumps(args, ensure_ascii=False, default=str),
                question,
            )
            return invalid_financial_tool_result(
                question=question,
                tool_name=tool_name,
                args=args,
                response_mode=response_mode,
                insight_subtype=insight_subtype,
                reason=reason,
                error=error,
                nlp_debug=nlp_debug,
            )

        missing_args = missing_required_query_args(tool_name, args)
        if missing_args:
            error = f"Missing required tool arguments for {tool_name}: {', '.join(missing_args)}"
            logger.error(
                "NLP_MISSING_TOOL_ARGS tool=%s missing=%s args=%s question=%s",
                tool_name,
                missing_args,
                json.dumps(args, ensure_ascii=False, default=str),
                question,
            )
            return invalid_financial_tool_result(
                question=question,
                tool_name=tool_name,
                args=args,
                response_mode=response_mode,
                insight_subtype=insight_subtype,
                reason=reason,
                error=error,
                nlp_debug=nlp_debug,
            )

        started = time.perf_counter()
        raw_result = await manager.call_tool(tool_name, args)
        tool_duration = time.perf_counter() - started

        if response_mode == "deterministic":
            # Deterministic answers are pure Python formatting over MCP JSON.
            answer = format_deterministic_answer(question, tool_name, raw_result)
            key_facts = ""
        elif response_mode == "insight":
            # Stage 3: prepare a focused context slice for this insight subtype.
            result_currency = "NIS"
            try:
                result_data = raw_result if isinstance(raw_result, dict) else json.loads(str(raw_result))
                result_currency = result_data.get("currency") or result_currency
            except Exception:
                result_data = {}
                pass
            selected_insight_context = build_insight_context(
                insight_subtype,
                result_data if isinstance(result_data, dict) else {},
                question,
            )
            nlp_debug["focused_context_keys"] = list(selected_insight_context.keys())
            selected_insight_context_json = json.dumps(
                selected_insight_context,
                ensure_ascii=False,
                default=str,
            )
            llm_specialized_subtypes = {
                "lifestyle",
                "financial_coach",
                "mortgage_review",
                "merchant_impact",
                "compare_months",
            }
            key_facts = (
                ""
                if insight_subtype in llm_specialized_subtypes
                else format_insight_key_facts(raw_result)
            )
            subtype_answer = format_insight_subtype_answer(question, insight_subtype, raw_result)
            if subtype_answer is not None:
                # Some insight subtypes still use deterministic formatters because
                # their expected answers are factual rankings/classifications.
                key_facts = ""
                answer = subtype_answer
                nlp_debug["guard"] = "passed"
            else:
                insight_messages = [
                        {
                            "role": "system",
                            "content": f"""
You are producing a financial review from deterministic JSON context.
The review context is the primary source of truth.
You may identify habits, patterns, concentration, recurring obligations, and review areas, but every statement must be grounded in the JSON.
The deterministic key facts are displayed to the user separately before your commentary.
Do not restate overview totals or key monetary facts unless copied directly from the provided deterministic key facts.
Do not create a separate overview section with new totals.
Do not open with generic dataset-summary phrases such as "This dataset reveals", "The analysis shows", "The data indicates", "The transactions indicate", "The dataset presents", or "The data quality log".
Start by directly answering the user's question.
Do not discuss anomaly detection, transaction categorization, or data quality unless the user explicitly asks about them.
Focus on patterns, habits, and practical review suggestions.
If insight_subtype is ranking, answer only the exact ranking question using reviewable_spending_areas.
If insight_subtype is commitments, answer only with recurring obligations or commitments using stable_recurring_obligations, recurring_payments, or mandatory_spending_areas. Do not treat recurring_merchants as commitments.
{insight_prompt_instructions(insight_subtype)}
Use category_concentration when discussing dominant spending categories.
Use reviewable_spending_areas as the primary source when answering savings, reduction, or recommendation questions.
Mandatory or fixed obligations must not become savings recommendations solely because they are large.
Use amount_calculation_basis to understand whether amounts are gross or net.
If amount_calculation_basis says net_debits_minus_credits, treat credits/refunds as reducing spending.
Distinguish recurring_merchants from recurring_payments:
- recurring_merchants are merchants appearing across multiple months.
- recurring_payments are likely recurring obligations or services identified deterministically in the JSON.
Use the currency field from the JSON context for every amount.
Use currency_symbol from the JSON context when present.
Use the currency exactly as provided.
If currency == "NIS" or currency_symbol == "₪", display amounts with ₪. Never write NIS as "$".
Never use "$" unless currency == "USD".
Never assume USD.
Never invent merchant names, categories, payments, recipients, transactions, months, amounts, causes, or subscriptions.
Never invent, rename, merge, or translate merchants beyond the provided deterministic display names.
Every merchant mentioned in the answer must appear verbatim in the input JSON or as an exact deterministic display name shown in the key facts.
When merchant_display is present, use merchant_display exactly as provided.
Do not translate, transliterate, romanize, rename, or correct merchant names.
If an item has no merchant_display, avoid mentioning it by merchant name.
If a merchant is not present in the JSON, do not mention it.
Never guess unknown entities.
Never attempt typo correction.
Never infer recipient names.
If an entity is not present in the JSON context, do not mention it.
Use merchant, category, payment, and transaction labels exactly as they appear in JSON.
Hebrew merchant names are valid entities.
Do not create your own translations, transliterations, or romanizations of merchant names.
If a deterministic key fact includes a display label in brackets, use that exact display text.
Descriptions may contain free text. Do not infer a counterparty or recipient from descriptions.
Do not discuss fraud, suspicious activity, compliance, financial crime, or regulatory concerns unless such indicators explicitly exist in the JSON context.
Do not describe stable_recurring_obligations as suspicious, anomalous, or unexpected.
If a transaction is marked review_classification = "stable_recurring_obligation_not_suspicious", describe it as a recurring obligation only.
Do not introduce review suggestions that are not grounded in possible_review_areas or other JSON fields.
When giving recommendations, phrase them as review suggestions based only on possible_review_areas.
Use only the provided JSON context.
Be concise and practical.
""",
                        },
                        {
                            "role": "user",
                            "content": f"""
Question:
{question}

Insight workflow tool:
{tool_name}

Insight subtype:
{insight_subtype}

Tool args:
{json.dumps(args, ensure_ascii=False)}

Deterministic key facts already shown to the user:
{key_facts}

Financial review JSON context:
{selected_insight_context_json}

Write only the LLM commentary. Do not repeat the deterministic key facts as a list.
""",
                        },
                    ]
                answer = call_llm_with_insight_retry(
                    insight_messages,
                    insight_subtype,
                    question,
                )
                answer = postprocess_insight_currency(answer, result_currency)
                # Stage 4: repair or replace unsafe wording before returning it.
                guard_status = "passed"
                answer_before_validation_guard = answer
                answer = apply_subtype_aware_validation_guard(
                    answer,
                    insight_subtype,
                    insight_messages,
                    question,
                    result_currency,
                )
                if answer.startswith(SAFE_GROUNDING_FALLBACK_PREFIX):
                    guard_status = "fallback"
                elif answer != answer_before_validation_guard:
                    guard_status = "regenerated"
                answer_before_merchant_guard = answer
                answer = apply_subtype_aware_merchant_guard(
                    answer,
                    raw_result,
                    response_mode,
                    question,
                    insight_subtype,
                    insight_messages,
                    result_currency,
                )
                if answer.startswith(SAFE_GROUNDING_FALLBACK_PREFIX):
                    guard_status = "fallback"
                elif answer != answer_before_merchant_guard and guard_status != "fallback":
                    guard_status = "regenerated"
                if answer.startswith(SAFE_GROUNDING_FALLBACK_PREFIX):
                    # Stage 5: final safety net. If an answer cannot be repaired,
                    # return verified deterministic facts instead of risky text.
                    fallback = deterministic_light_insight_answer(insight_subtype, selected_insight_context)
                    if not fallback:
                        fallback = deterministic_insight_fallback(insight_subtype, selected_insight_context)
                    if fallback:
                        answer = fallback
                    guard_status = "fallback"
                nlp_debug["guard"] = guard_status
        else:
            # Legacy interpretive mode still receives currency and merchant safeguards.
            key_facts = ""
            result_currency = "NIS"
            try:
                result_data = raw_result if isinstance(raw_result, dict) else json.loads(str(raw_result))
                result_currency = result_data.get("currency") or result_currency
            except Exception:
                pass
            answer = call_llm(
                [
                    {
                        "role": "system",
                        "content": """
You answer interpretive financial questions from deterministic JSON tool output.
Every statement must be grounded in the JSON tool output.
Do not invent merchants, categories, transactions, months, or amounts.
Never invent, rename, merge, or translate merchants beyond the provided deterministic display names.
Every merchant mentioned in the answer must appear verbatim in the input JSON.
If a merchant is not present in the JSON, do not mention it.
Do not mention SQL.
Be concise and practical.
Use the currency field from the JSON context for every amount.
If currency == "NIS", display amounts with ₪. Never write NIS as "$".
Never use "$" unless currency == "USD".
Never assume USD.
""",
                    },
                    {
                        "role": "user",
                        "content": f"""
Question:
{question}

Tool selected:
{tool_name}

Tool args:
{json.dumps(args, ensure_ascii=False)}

Tool JSON result:
{raw_result}

Write the final answer.
""",
                    },
                ]
            )
            answer = enforce_currency_text(answer, result_currency)
            answer = enforce_answer_merchant_grounding(
                answer,
                raw_result,
                response_mode,
                question,
            )
            nlp_debug["guard"] = (
                "fallback"
                if str(answer).startswith(SAFE_GROUNDING_FALLBACK_PREFIX)
                else "passed"
            )

        return {
            "question": question,
            "tool": tool_name,
            "args": args,
            "reason": reason,
            "raw_result": raw_result,
            "answer": answer,
            "key_facts": key_facts,
            "response_mode": response_mode,
            "insight_subtype": insight_subtype,
            "nlp_debug": nlp_debug,
            "timing": {
                "tool_execution_seconds": round(tool_duration, 3),
            },
        }
    finally:
        await manager.close()
