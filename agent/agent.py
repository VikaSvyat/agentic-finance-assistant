"""Compatibility wrapper for the finance agent public API.

The implementation lives in agent.runner after the structural refactor. Keeping
this module preserves existing imports such as `from agent.agent import run_agent`.
"""

from agent.runner import Agent, run_agent, run_financial_question

__all__ = ["Agent", "run_agent", "run_financial_question"]
