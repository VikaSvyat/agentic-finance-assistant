"""Answer validation, merchant grounding, and safe fallback helpers."""

import json
import logging
import re
from typing import Any

from agent.answer_formatters import (
    BRACKET_LABEL_RE,
    EXPLICIT_MERCHANT_LABEL_RE,
    HEBREW_ENTITY_RE,
    MERCHANT_ENTITY_KEYS,
    QUOTED_LATIN_LABEL_RE,
    display_spending_area_type,
    format_merchant_display,
    money_text,
    normalize_entity_text,
    parse_tool_json,
)
from agent.llm import call_llm

logger = logging.getLogger(__name__)

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
