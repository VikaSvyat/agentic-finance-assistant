"""Deterministic answer and insight subtype formatters."""

import json
import re
from typing import Any

from agent.tool_args import normalize_args
from finance.display import format_merchant_display

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
