"""Report and AI-insight execution helpers used by the agent runner."""

import json
import logging
import time
from typing import Any

from agent.guards import enforce_currency_text
from agent.llm import call_llm
from agent.mcp_manager import short_json
from agent.tool_args import build_args_for_tool
from agent.tool_results import (
    action_for_history,
    make_observation,
    memory_state_text,
    remember_tool_output,
)

logger = logging.getLogger(__name__)


def execute_ai_insights_step_impl(
    agent: Any,
    step_number: int,
    step_start: float,
    llm_duration: float,
    action: dict[str, Any],
    raw_args: Any,
    reason: str,
) -> dict[str, Any] | None:
    """Execute the virtual AI-insights tool while preserving Agent state updates."""

    args = build_args_for_tool(
        "finance.generate_ai_financial_insights",
        raw_args,
        agent.csv_path,
        agent.memory,
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
        agent.memory,
    )
    agent.tool_results["finance.generate_ai_financial_insights"] = insight_text

    observation = make_observation(
        tool_name="finance.generate_ai_financial_insights",
        result=insight_text,
        ok=True,
    )

    step_record = {
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
    agent.steps.append(step_record)

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

    agent.messages.append(
        {
            "role": "assistant",
            "content": json.dumps(action_for_history(action), ensure_ascii=False),
        }
    )
    agent.messages.append(
        {
            "role": "user",
            "content": f"""
Tool observation:
{json.dumps(llm_observation, ensure_ascii=False)}

{memory_state_text(agent.memory, agent.csv_path)}

Continue. Choose exactly ONE next tool or final_answer.
""",
        }
    )
    agent.prune_messages()

    return None
