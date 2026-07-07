"""Financial question routing and insight subtype classification."""

import json
import logging
import re
from typing import Any

from agent.llm import call_llm

logger = logging.getLogger(__name__)

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
