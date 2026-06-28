import unittest
import json
from unittest.mock import patch

from agent.agent import (
    FINANCIAL_REVIEW_SUBTYPES,
    INSIGHT_WORKFLOW_TOOLS,
    QUERY_TOOL_ALIASES,
    classify_financial_intent,
    classify_insight_subtype,
    classify_insight_subtype_with_llm,
    classify_response_mode,
    build_insight_context,
    deterministic_insight_subtype,
    deterministic_insight_fallback,
    apply_subtype_aware_merchant_guard,
    enforce_answer_merchant_grounding,
    format_insight_subtype_answer,
    format_deterministic_answer,
    insight_answer_validation_errors,
    insight_prompt_instructions,
    invalid_financial_tool_result,
    limit_from_question,
    maybe_redirect_query_tool,
    missing_required_query_args,
    SAFE_GROUNDING_FALLBACK_PREFIX,
)


INSIGHT_ROUTING_CASES = {
    "What financial habits do you notice?": "financial_review",
    "Describe my lifestyle": "lifestyle",
    "What do these transactions say about me?": "lifestyle",
    "Imagine you know nothing about me except these transactions.": "lifestyle",
    "What habits do you notice?": "financial_review",
    "What patterns describe my spending behavior?": "lifestyle",
    "What kind of person do these transactions describe?": "lifestyle",
    "What do my spending habits say about me?": "financial_review",
    "Which three habits would have the biggest financial impact?": "financial_review",
    "What should I stop spending money on?": "financial_review",
    "Which recurring expenses could I safely ignore when reviewing my budget?": "financial_review",
    "Which spending looks expensive at first glance but is actually normal?": "financial_review",
    "Which small recurring expenses are most likely to become expensive over a year?": "financial_review",
    "Based on everything you know from my transactions, what is the single most important insight about my finances that I might be overlooking? Support your answer with evidence from my spending.": "financial_review",
    "What recommendations do you have?": "financial_review",
    "What should I review to save money?": "financial_review",
    "Where would you cut ₪2,000 per month?": "financial_review",
    "How could I save money?": "financial_review",
    "What should I reduce first?": "financial_review",
    "Where should I start?": "financial_review",
    "If you were approving my mortgage, what would concern you?": "mortgage_review",
    "What would concern a lender?": "mortgage_review",
    "Which transactions would require explanation?": "mortgage_review",
    "Which merchants affect my budget the most?": "merchant_impact",
    "If I reduced spending with three merchants by 20%, what would I save?": "financial_review",
    "Compare May and June and explain the difference.": "financial_review",
    (
        "If you were my financial coach, what are the three questions "
        "you would ask me before giving recommendations?"
    ): "financial_review",
    (
        "What were the three biggest drivers of spending growth between "
        "my lowest-spending month and my highest-spending month?"
    ): "spending_growth",
    (
        "If I wanted to reduce spending by 10% without touching housing, "
        "pension, insurance, education, taxes, or utilities, where should I start?"
    ): "financial_review",
    "What are the three largest discretionary spending categories in my data?": "savings_opportunities",
    "What are the three largest optional spending categories in my data?": "savings_opportunities",
    "What are the three largest recurring financial commitments in my data?": "commitments",
    "Which recurring payments are actual obligations, and which are payment methods?": "commitments",
    "Which recurring payments are actual obligations, and which are just payment methods?": "commitments",
    "Classify recurring payments.": "commitments",
    "Obligations vs payment methods.": "commitments",
    "Which recurring items are contractual obligations?": "commitments",
    "Show stable recurring obligations and transaction type labels.": "commitments",
}

INSIGHT_SUBTYPE_CASES = {
    "What financial habits do you notice?": "general",
    "Describe my lifestyle": "lifestyle",
    "What do these transactions say about me?": "lifestyle",
    "Imagine you know nothing about me except these transactions.": "lifestyle",
    "What habits do you notice?": "general",
    "What patterns describe my spending behavior?": "lifestyle",
    "What kind of person do these transactions describe?": "lifestyle",
    "What do my spending habits say about me?": "general",
    "Which three habits would have the biggest financial impact?": "general",
    "What should I stop spending money on?": "general",
    "Which recurring expenses could I safely ignore when reviewing my budget?": "general",
    "Which spending looks expensive at first glance but is actually normal?": "general",
    "Which small recurring expenses are most likely to become expensive over a year?": "general",
    "Based on everything you know from my transactions, what is the single most important insight about my finances that I might be overlooking? Support your answer with evidence from my spending.": "general",
    "Where would you cut ₪2,000 per month?": "general",
    "How could I save money?": "general",
    "What should I reduce first?": "general",
    "If you were approving my mortgage, what would concern you?": "mortgage_review",
    "What would concern a lender?": "mortgage_review",
    "Which transactions would require explanation?": "mortgage_review",
    "Which merchants affect my budget the most?": "merchant_impact",
    "If I reduced spending with three merchants by 20%, what would I save?": "general",
    "Compare May and June and explain the difference.": "general",
    "What are the three largest discretionary spending categories in my data?": "ranking",
    "What are the three largest optional spending categories in my data?": "ranking",
    "What are the three largest recurring financial commitments in my data?": "commitments",
    "Which recurring payments are actual obligations, and which are payment methods?": "commitments",
    "Which recurring payments are actual obligations, and which are just payment methods?": "commitments",
    "Classify recurring payments.": "commitments",
    (
        "What were the three biggest drivers of spending growth between "
        "my lowest-spending month and my highest-spending month?"
    ): "compare_months",
}


class FinancialRoutingTests(unittest.TestCase):
    def sample_review_context(self):
        return {
            "currency": "NIS",
            "currency_symbol": "₪",
            "months_covered": ["2026-05", "2026-06"],
            "month_coverage": [{"month": "2026-06", "is_complete_month": False}],
            "amount_calculation_basis": "net_debits_minus_credits_for_financial_review",
            "data_quality_log": {"debug": True},
            "top_categories": [{"category": "shopping", "total_amount": 1000}],
            "category_concentration": {"top_category": "Shopping"},
            "reviewable_spending_areas": [
                {
                    "type": "shopping",
                    "amount": 1000,
                    "top_merchants": [
                        {"merchant": "WOLT", "amount": 300, "months_present": 2},
                        {"merchant": "העברה דיגיטל", "amount": 600, "months_present": 2},
                    ],
                }
            ],
            "mandatory_spending_areas": [{"type": "rent_and_utilities", "amount": 5000}],
            "stable_recurring_obligations": [{"merchant": "Rent", "total_amount": 10000}],
            "possible_review_areas": [
                {"type": "shopping", "amount": 1000, "evidence": "Discretionary or non-essential spending category"},
                {"type": "large_expense", "amount": 5000, "evidence": "Mandatory fixed spending category"},
            ],
            "recurring_merchants": [
                {"merchant": "WOLT", "total_amount": 1200, "months_present": 2},
                {"merchant": "העברה דיגיטל", "total_amount": 12000, "months_present": 2},
            ],
            "recurring_payments": [{"merchant": "Bezeq", "total_amount": 600}],
            "transaction_type_labels": [{"label": "העברה דיגיטל", "total_amount": 12000}],
            "unusual_transactions": [{"merchant": "A", "amount": 900}, {"merchant": "B", "amount": 100}],
            "largest_expenses": [{"merchant": "C", "amount": 800}, {"merchant": "D", "amount": 200}],
        }

    def test_financial_coach_context_is_reduced(self):
        context = build_insight_context(
            "financial_coach",
            self.sample_review_context(),
            "How could I save money?",
        )
        self.assertIn("reviewable_spending_areas", context)
        self.assertIn("mandatory_spending_areas", context)
        self.assertIn("stable_recurring_obligations", context)
        self.assertNotIn("data_quality_log", context)
        self.assertNotIn("transaction_type_labels", context)
        self.assertNotIn("unusual_transactions", context)
        self.assertEqual(len(context["possible_review_areas"]), 1)

    def test_lifestyle_context_excludes_unrelated_debug_and_anomalies(self):
        context = build_insight_context(
            "lifestyle",
            self.sample_review_context(),
            "Describe my lifestyle based only on these transactions.",
        )
        self.assertIn("recurring_merchants", context)
        self.assertIn("recurring_payments", context)
        self.assertNotIn("data_quality_log", context)
        self.assertNotIn("unusual_transactions", context)
        self.assertNotIn("largest_expenses", context)

    def test_merchant_impact_context_builds_candidates_and_exclusions(self):
        context = build_insight_context(
            "merchant_impact",
            self.sample_review_context(),
            "Which merchants affect my budget the most?",
        )
        self.assertIn("merchant_candidates", context)
        self.assertIn("reviewable_merchant_candidates", context)
        self.assertIn("excluded_payment_mechanisms", context)
        self.assertEqual(context["merchant_candidates"][0]["merchant"], "WOLT")
        self.assertEqual(context["merchant_candidates"][0]["merchant_display"], "WOLT")
        self.assertEqual(context["reviewable_merchant_candidates"][0]["merchant"], "WOLT")
        self.assertEqual(context["reviewable_merchant_candidates"][0]["merchant_display"], "WOLT")
        self.assertNotIn("data_quality_log", context)

    def test_general_context_does_not_pass_full_context(self):
        context = build_insight_context("unknown_subtype", self.sample_review_context(), "Review this.")
        self.assertIn("top_categories", context)
        self.assertIn("category_concentration", context)
        self.assertNotIn("data_quality_log", context)
        self.assertNotIn("recurring_merchants", context)

    def test_lifestyle_context_limits_recurring_merchants_and_adds_display(self):
        full_context = self.sample_review_context()
        full_context["recurring_merchants"] = [
            {"merchant": f"Merchant {index}", "total_amount": index}
            for index in range(20)
        ]
        context = build_insight_context("lifestyle", full_context, "Describe my lifestyle.")
        self.assertEqual(len(context["recurring_merchants"]), 10)
        self.assertIn("merchant_display", context["recurring_merchants"][0])

    def test_deterministic_fallback_for_merchant_impact(self):
        context = build_insight_context(
            "merchant_impact",
            self.sample_review_context(),
            "Which merchants affect my budget the most?",
        )
        fallback = deterministic_insight_fallback("merchant_impact", context)
        self.assertIn("Verified Answer", fallback)
        self.assertIn("verified facts over uncertain interpretations", fallback)
        self.assertIn("WOLT", fallback)
        self.assertNotIn("error", fallback.casefold())
        self.assertNotIn("validation", fallback.casefold())
        self.assertNotIn("AI-generated", fallback)

    def test_verified_fallback_for_financial_coach_is_user_friendly(self):
        context = build_insight_context(
            "financial_coach",
            self.sample_review_context(),
            "How could I save money?",
        )
        fallback = deterministic_insight_fallback("financial_coach", context)
        self.assertIn("Verified Answer", fallback)
        self.assertIn("Shopping", fallback)
        self.assertIn("deterministic financial analysis", fallback)
        self.assertNotIn("merchant grounding", fallback.casefold())
        self.assertNotIn("could not validate", fallback.casefold())
        self.assertNotIn("AI failure", fallback)

    def test_verified_fallback_for_mortgage_review_uses_relevant_context(self):
        context = build_insight_context(
            "mortgage_review",
            self.sample_review_context(),
            "If you were reviewing my finances before approving a mortgage, what would concern you?",
        )
        fallback = deterministic_insight_fallback("mortgage_review", context)
        self.assertIn("Verified Answer", fallback)
        self.assertIn("Recurring obligations", fallback)
        self.assertIn("Fixed or mandatory spending areas", fallback)
        self.assertIn("Large or unusual expenses", fallback)
        self.assertNotIn("validation", fallback.casefold())

    def test_insight_questions_route_to_insight_mode(self):
        for question in INSIGHT_ROUTING_CASES:
            with self.subTest(question=question):
                self.assertEqual(classify_response_mode(question), "insight")

    def test_insight_questions_have_review_intents(self):
        for question, expected_intent in INSIGHT_ROUTING_CASES.items():
            with self.subTest(question=question):
                self.assertEqual(classify_financial_intent(question), expected_intent)

    def test_insight_workflow_starts_with_review_context(self):
        self.assertEqual(
            INSIGHT_WORKFLOW_TOOLS[0],
            "finance.prepare_financial_review_context",
        )

    def test_insight_subtypes_are_classified(self):
        for question, expected_subtype in INSIGHT_SUBTYPE_CASES.items():
            with self.subTest(question=question):
                self.assertEqual(classify_response_mode(question), "insight")
                self.assertEqual(classify_insight_subtype(question), expected_subtype)

    def test_llm_subtype_classifier_uses_structured_json(self):
        with patch(
            "agent.agent.call_llm",
            return_value=json.dumps(
                {
                    "subtype": "financial_coach",
                    "confidence": 0.94,
                    "reason": "The user asks where to reduce spending.",
                }
            ),
        ):
            result = classify_insight_subtype_with_llm("What should I stop spending money on?")

        self.assertEqual(result["subtype"], "financial_coach")
        self.assertEqual(result["confidence"], 0.94)

    def test_llm_subtype_classifier_low_confidence_falls_back_to_general(self):
        with patch(
            "agent.agent.call_llm",
            return_value=json.dumps(
                {
                    "subtype": "merchant_impact",
                    "confidence": 0.42,
                    "reason": "Weak signal.",
                }
            ),
        ):
            result = classify_insight_subtype_with_llm("Tell me something useful.")

        self.assertEqual(result["subtype"], "general")

    def test_deterministic_subtype_shortcuts_do_not_need_llm(self):
        self.assertEqual(
            deterministic_insight_subtype(
                "What are the three largest discretionary spending categories in my data?"
            ),
            "ranking",
        )
        with patch("agent.agent.call_llm") as mocked_call:
            subtype = classify_insight_subtype(
                "What are the three largest discretionary spending categories in my data?",
                use_llm=True,
            )
        self.assertEqual(subtype, "ranking")
        mocked_call.assert_not_called()

    def test_coaching_questions_bypass_deterministic_shortcuts(self):
        coaching_questions = [
            "What should I stop spending money on?",
            "Which three habits would have the biggest financial impact?",
            "What should I do about my top merchant categories?",
            "Where should I start if I want to reduce spending by category?",
            "What are my biggest long-term savings opportunities by merchant?",
        ]
        for question in coaching_questions:
            with self.subTest(question=question):
                self.assertEqual(classify_response_mode(question), "insight")
                self.assertIsNone(deterministic_insight_subtype(question))

        with patch(
            "agent.agent.call_llm",
            return_value=json.dumps(
                {
                    "subtype": "financial_coach",
                    "confidence": 0.91,
                    "reason": "The user asks for coaching and prioritization.",
                }
            ),
        ):
            subtype = classify_insight_subtype(
                "Where should I start if I want to reduce spending by category?",
                use_llm=True,
            )
        self.assertEqual(subtype, "financial_coach")

    def test_explicit_reporting_requests_stay_deterministic(self):
        reporting_questions = [
            "Compare May and June spending.",
            "Show categories for May.",
            "What are my top merchants?",
            "Show unusual transactions.",
            "How much did I spend in May?",
        ]
        for question in reporting_questions:
            with self.subTest(question=question):
                self.assertEqual(classify_response_mode(question), "deterministic")

    def test_reasoning_requests_with_reporting_nouns_go_to_review(self):
        questions = [
            "Compare May and June and explain the difference.",
            "What should I do about my top merchants?",
            "What are the biggest opportunities in my categories?",
            "Which merchant trends should I prioritize long-term?",
            "Which spending looks expensive at first glance but is actually normal?",
            "Which small recurring expenses are most likely to become expensive over a year?",
            "Which recurring expenses could I safely ignore when reviewing my budget?",
            "Based on everything you know from my transactions, what is the single most important insight about my finances that I might be overlooking? Support your answer with evidence from my spending.",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(classify_response_mode(question), "insight")
                self.assertIsNone(deterministic_insight_subtype(question))

    def test_tool_arg_safety_redirects_and_limits(self):
        self.assertEqual(limit_from_question("Show my top five merchants."), 5)
        tool, args = maybe_redirect_query_tool(
            "Which categories changed the most between May and June?",
            "finance.get_category_comparison",
            {},
        )
        self.assertEqual(tool, "finance.compare_months")
        self.assertEqual(args, {"month_a": "2026-05", "month_b": "2026-06"})
        tool, args = maybe_redirect_query_tool(
            "Show my top five merchants.",
            "finance.get_top_merchants",
            {},
        )
        self.assertEqual(args["limit"], 5)
        self.assertEqual(
            missing_required_query_args("finance.get_category_trend", {"months": 6}),
            ["category"],
        )

    def test_insight_answer_validator_rejects_summary_style(self):
        errors = insight_answer_validation_errors(
            "financial_coach",
            "The report indicates stable obligations.",
        )
        self.assertIn("starts with a generic dataset/report summary", errors)
        self.assertIn("missing recommendation", errors)
        self.assertIn(
            "contains unsupported currency label",
            insight_answer_validation_errors("mortgage_review", "Evidence: HKD 2000."),
        )

    def test_compare_months_warns_on_incomplete_month(self):
        answer = format_deterministic_answer(
            "Compare May and June.",
            "finance.compare_months",
            json.dumps(
                {
                    "month_a": "2026-05",
                    "month_b": "2026-06",
                    "currency": "NIS",
                    "month_a_coverage": {"month": "2026-05", "is_complete_month": True, "coverage_days": 31, "expected_days": 31},
                    "month_b_coverage": {"month": "2026-06", "is_complete_month": False, "coverage_days": 26, "expected_days": 30},
                    "month_a_total": 100,
                    "month_b_total": 50,
                    "spending_difference": -50,
                    "percentage_change": -50,
                    "largest_category_changes": [],
                }
            ),
        )
        self.assertIn("Warning: 2026-06 appears incomplete (26/30 days)", answer)

    def test_supported_llm_subtypes_are_centralized(self):
        self.assertEqual(
            set(FINANCIAL_REVIEW_SUBTYPES),
            {
                "general",
                "financial_coach",
                "lifestyle",
                "commitments",
                "mortgage_review",
                "merchant_impact",
            },
        )

    def test_lifestyle_subtype_uses_llm_interpretation_path(self):
        self.assertIsNone(
            format_insight_subtype_answer(
                "Describe my lifestyle",
                "lifestyle",
                json.dumps({"currency": "NIS"}),
            )
        )

    def test_specialized_reasoning_subtypes_use_llm_interpretation_path(self):
        for subtype in ["financial_coach", "mortgage_review", "merchant_impact", "compare_months"]:
            with self.subTest(subtype=subtype):
                self.assertIsNone(
                    format_insight_subtype_answer(
                        "specialized reasoning question",
                        subtype,
                        json.dumps({"currency": "NIS"}),
                    )
                )

    def test_financial_coach_prompt_is_advice_not_dataset_summary(self):
        prompt = insight_prompt_instructions("financial_coach")
        self.assertIn("personal financial coach", prompt)
        self.assertIn("Start immediately with recommendations", prompt)
        self.assertIn("Current spending", prompt)
        self.assertIn("Estimated monthly savings", prompt)
        self.assertIn("Target achievable?", prompt)
        self.assertIn("Never recommend rent, pension, insurance, education, taxes, utilities", prompt)
        self.assertIn("Do not summarize the dataset", prompt)

    def test_common_wrong_tool_aliases_are_known(self):
        self.assertEqual(
            QUERY_TOOL_ALIASES["finance.get_top_categories"],
            "finance.get_category_breakdown",
        )

    def test_invalid_tool_result_is_safe_for_ui(self):
        result = invalid_financial_tool_result(
            question="bad question",
            tool_name="finance.not_a_tool",
            args={},
            response_mode="deterministic",
            insight_subtype="",
            reason="test",
            error="not available",
        )
        self.assertEqual(
            result["answer"],
            "I could not select a valid financial query tool for this question.",
        )
        self.assertEqual(result["tool"], "finance.not_a_tool")
        self.assertIn("not available", result["raw_result"])

    def test_ranking_subtype_formats_reviewable_spending_areas(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "reviewable_spending_areas": [
                    {"type": "restaurants", "amount": 2292.22},
                    {"type": "shopping", "amount": 4143.21},
                    {"type": "leisure_entertainment_sports", "amount": 3365.83},
                ],
            }
        )
        answer = format_insight_subtype_answer(
            "What are the three largest discretionary spending categories in my data?",
            "ranking",
            raw_result,
        )
        self.assertIn("1. Shopping — ₪4,143.21", answer)
        self.assertIn("2. Leisure, entertainment and sports — ₪3,365.83", answer)
        self.assertIn("3. Restaurants — ₪2,292.22", answer)

    def test_commitments_subtype_uses_obligations_not_recurring_merchants(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "stable_recurring_obligations": [
                    {
                        "label": "העברה דיגיטל",
                        "total_amount": 33600,
                        "months_present": 6,
                        "transactions": 6,
                        "source": "transaction_type_label",
                    },
                    {
                        "label": "הוראת קבע",
                        "total_amount": 2500,
                        "months_present": 5,
                        "transactions": 5,
                        "source": "transaction_type_label",
                    },
                    {
                        "label": "דמי כרטיס",
                        "total_amount": 120,
                        "months_present": 6,
                        "transactions": 6,
                        "source": "transaction_type_label",
                    }
                ],
                "recurring_payments": [
                    {
                        "merchant": "מור גמל ופנס-י",
                        "total_amount": 12000,
                        "months_present": 6,
                        "transactions": 6,
                        "payment_type": "pension",
                        "category": "rent_and_utilities",
                        "category_en": "Rent and utilities",
                    }
                ],
                "recurring_merchants": [
                    {
                        "merchant": "אושר עד חיפה",
                        "total_amount": 14000,
                    }
                ],
            },
            ensure_ascii=False,
        )
        answer = format_insight_subtype_answer(
            "What are the three largest recurring financial commitments in my data?",
            "commitments",
            raw_result,
        )
        self.assertIn("מור גמל ופנס-י", answer)
        self.assertNotIn("העברה דיגיטל", answer)
        self.assertNotIn("הוראת קבע", answer)
        self.assertNotIn("דמי כרטיס", answer)
        self.assertNotIn("אושר עד חיפה", answer)

    def test_commitments_subtype_classifies_payment_methods_separately(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "transaction_type_labels": [
                    {
                        "label": "העברה דיגיטל",
                        "total_amount": 33600,
                        "months_present": 6,
                        "transactions": 6,
                        "usage_hint": "used for paying rent and utilities",
                    },
                    {
                        "label": "הוראת קבע",
                        "total_amount": 2500,
                        "months_present": 5,
                        "transactions": 5,
                    },
                ],
                "recurring_payments": [
                    {
                        "merchant": "מור גמל ופנס-י",
                        "total_amount": 12000,
                        "months_present": 6,
                        "transactions": 6,
                        "payment_type": "pension",
                        "category": "rent_and_utilities",
                        "category_en": "Rent and utilities",
                    }
                ],
                "recurring_merchants": [
                    {
                        "merchant": "אושר עד חיפה",
                        "total_amount": 14000,
                        "months_present": 6,
                        "transactions": 23,
                    }
                ],
            },
            ensure_ascii=False,
        )
        answer = format_insight_subtype_answer(
            "Which recurring payments are actual obligations, and which are payment methods?",
            "commitments",
            raw_result,
        )
        self.assertIn("Contractual financial obligations:", answer)
        self.assertIn("מור גמל ופנס-י", answer)
        self.assertIn("Payment methods / transaction mechanisms:", answer)
        self.assertIn("העברה דיגיטל", answer)
        self.assertIn("הוראת קבע", answer)
        self.assertIn("payment mechanism, not a recipient", answer)
        self.assertIn("6 transactions", answer)
        self.assertIn("used in 6 months", answer)
        self.assertIn("used for paying rent and utilities", answer)
        self.assertNotIn("₪33,600.00", answer)
        self.assertIn("Recurring merchants:", answer)
        self.assertIn("אושר עד חיפה", answer)
        self.assertIn("Classified as a contractual obligation because payment_type is 'pension'", answer)
        self.assertIn("This is a payment mechanism, not the recipient of the money", answer)

    def test_simple_recurring_merchant_listing_stays_deterministic(self):
        self.assertEqual(
            classify_response_mode("Which merchants appear every month?"),
            "deterministic",
        )
        self.assertEqual(
            classify_response_mode("Show recurring merchants."),
            "deterministic",
        )

    def test_answer_merchant_grounding_replaces_ungrounded_hebrew_names(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "recurring_payments": [
                    {"merchant": "מור גמל ופנס-י", "total_amount": 12000}
                ],
            },
            ensure_ascii=False,
        )
        answer = enforce_answer_merchant_grounding(
            "You should review בנק דמיוני.",
            raw_result,
            "insight",
            "test question",
        )
        self.assertEqual(answer, "__verified_financial_summary_required__")

    def test_answer_merchant_grounding_detects_explicit_merchant_labels(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "recurring_merchants": [
                    {"merchant": "WOLT", "total_amount": 1200}
                ],
            },
            ensure_ascii=False,
        )
        answer = enforce_answer_merchant_grounding(
            "1. Merchant Display: Coffee Shop\nEvidence: recurring spend.",
            raw_result,
            "insight",
            "Which merchants affect my budget?",
        )
        self.assertEqual(answer, SAFE_GROUNDING_FALLBACK_PREFIX)

    def test_financial_coach_validator_rejects_mandatory_savings_targets(self):
        errors = insight_answer_validation_errors(
            "financial_coach",
            "\n".join(
                [
                    "Recommendation: Review rent payments.",
                    "Current spending: ₪5,600",
                    "Estimated monthly savings: not enough evidence to estimate",
                    "Evidence: stable recurring obligation",
                ]
            ),
        )
        self.assertIn("recommends mandatory obligations as savings targets", errors)

    def test_merchant_grounding_is_strict_for_merchant_impact(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "recurring_merchants": [
                    {"merchant": "WOLT", "total_amount": 1200}
                ],
            },
            ensure_ascii=False,
        )
        guarded = apply_subtype_aware_merchant_guard(
            "1. Merchant: Coffee Shop\nEvidence: recurring spend.",
            raw_result,
            "insight",
            "Which merchants affect my budget?",
            "merchant_impact",
            [{"role": "user", "content": "test"}],
            "NIS",
        )
        self.assertEqual(guarded, SAFE_GROUNDING_FALLBACK_PREFIX)

    def test_merchant_grounding_rewrites_financial_coach_without_merchants(self):
        raw_result = json.dumps(
            {
                "currency": "NIS",
                "reviewable_spending_areas": [
                    {"type": "shopping", "amount": 1000}
                ],
            },
            ensure_ascii=False,
        )
        rewritten_answer = "\n".join(
            [
                "Recommendation: Review shopping.",
                "Current spending: ₪1,000",
                "Why realistic: It is a reviewable category.",
                "Estimated monthly savings: not enough evidence to estimate",
                "Evidence: reviewable spending area.",
            ]
        )
        with patch("agent.agent.call_llm", return_value=rewritten_answer) as mocked_call:
            guarded = apply_subtype_aware_merchant_guard(
                "Recommendation: Review Coffee Shop.\nCurrent spending: ₪500\nEstimated monthly savings: ₪50\nEvidence: Merchant: Coffee Shop",
                raw_result,
                "insight",
                "How could I save money?",
                "financial_coach",
                [{"role": "user", "content": "test"}],
                "NIS",
            )
        self.assertEqual(guarded, rewritten_answer)
        mocked_call.assert_called_once()
        self.assertNotIn("Verified Answer", guarded)

    def test_simple_numeric_category_question_stays_deterministic(self):
        self.assertEqual(
            classify_response_mode("How much did I spend on fuel?"),
            "deterministic",
        )


if __name__ == "__main__":
    unittest.main()
