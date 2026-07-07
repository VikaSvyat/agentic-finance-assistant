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
from agent.answer_formatters import (
    format_deterministic_answer,
    format_insight_key_facts,
    format_insight_subtype_answer,
    money_text,
    normalize_entity_text,
    parse_tool_json,
)
from agent.guards import (
    SAFE_GROUNDING_FALLBACK_PREFIX,
    apply_subtype_aware_merchant_guard,
    apply_subtype_aware_validation_guard,
    call_llm_with_insight_retry,
    deterministic_insight_fallback,
    deterministic_light_insight_answer,
    enforce_answer_merchant_grounding,
    enforce_currency_text,
    postprocess_insight_currency,
)
from agent.llm import call_llm
from agent.mcp_manager import MCPManager, short_json
from agent.prompts import SYSTEM_PROMPT, insight_prompt_instructions
from agent.tool_args import (
    build_args_for_tool,
    completed_tool_replacement,
    maybe_redirect_query_tool,
    missing_required_memory_for_tool,
    missing_required_query_args,
    normalize_args,
    normalize_query_tool_args,
)
from agent.tool_registry import (
    add_virtual_agent_tools,
    disallowed_tool_for_mode,
    tools_for_current_mode,
    tools_to_text,
)
from agent.tool_results import (
    action_for_history,
    make_observation,
    memory_state_text,
    prune_messages,
    remember_tool_output,
    truncate_text,
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
