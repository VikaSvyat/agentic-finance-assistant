"""Subtype-specific Financial Review context selection."""

from typing import Any

from agent.answer_formatters import COMMON_INSIGHT_CONTEXT_FIELDS, normalize_entity_text
from agent.routing import FINANCIAL_REVIEW_SUBTYPES
from finance.display import format_merchant_display

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
