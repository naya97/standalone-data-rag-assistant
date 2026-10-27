"""Sales summary and insights over the cleaned dataset (sales_clean), written by the model from facts we compute.

    SQL -> FACTS (numbers) -> model words them as JSON -> guardrails (every number must be in FACTS) -> result
    anything fails -> a fixed template built from the same FACTS (no model), flagged mode="fallback"
"""

from __future__ import annotations

import json
from pathlib import Path

from src.llm.week9.client import LLMClient
from src.llm.week9.data_facts import get_sales_facts
from src.llm.week9.guardrails import check_sales_insights
from src.llm.week9.prompts import SALES_PROMPT_VERSION, build_sales_messages
from src.llm.week9.runner import InsightResult, ask_json
from src.llm.week9.schemas import SalesInsights


def fallback_sales(facts: dict) -> dict:
    """The same JSON shape, from a fixed template: every number is copied from FACTS."""
    o, scope = facts["overall"], facts["scope"]
    top = facts["by_category"][0]
    pay = facts["by_payment_method"][0]
    summary = (f"From {scope['first_month']} to {scope['last_month']}: {o['transactions']} transactions, "
               f"total revenue {o['total_revenue']}, average order value {o['average_order_value']}.")
    insights = [f"{top['category']} has the highest revenue: {top['revenue']} ({top['share_pct']}% of the total).",
                f"{pay['payment_method']} is the payment method with the highest revenue: {pay['revenue']} ({pay['share_pct']}%)."]
    if facts["best_month"]:
        insights.append(f"The best month is {facts['best_month']['month']} (revenue {facts['best_month']['revenue']}); "
                        f"the weakest is {facts['worst_month']['month']} (revenue {facts['worst_month']['revenue']}).")
    with_target = [c for c in facts["by_category"] if c.get("target_achievement_pct") is not None]
    behind = [c for c in with_target if c["target_achievement_pct"] < 100]
    if behind:
        worst = min(behind, key=lambda c: c["target_achievement_pct"])
        insights.append(f"{len(behind)} categories are below their revenue target; the furthest is {worst['category']} "
                        f"at {worst['target_achievement_pct']}% of its target.")
    elif with_target:
        low = min(with_target, key=lambda c: c["target_achievement_pct"])
        insights.append(f"No category is below its revenue target; the lowest is {low['category']} "
                        f"at {low['target_achievement_pct']}% of its target.")
    return {"summary": summary, "insights": insights,
            "caveats": ["Written from a fixed template, not by the language model; English only.", *facts["notes"]]}


def sales_insights(client: LLMClient | None, db_path: str | Path, *, category: str | None = None,
                   start: str | None = None, end: str | None = None, focus: str | None = None, language: str = "en",
                   unavailable_reason: str | None = None) -> InsightResult:
    """Raises DataUnavailableError (no database/table) or ValueError (bad scope or language). Model problems never raise."""
    if language not in ("en", "ar"):
        raise ValueError(f"language must be 'en' or 'ar', got {language!r}")
    facts = get_sales_facts(db_path, category=category, start=start, end=end)
    facts_json = json.dumps(facts, ensure_ascii=False)
    system, user = build_sales_messages(facts_json, focus, language)
    obj, info, reason, sent = ask_json(client, system, user, SalesInsights, lambda o: check_sales_insights(o, facts_json),
                                       prompt_version=SALES_PROMPT_VERSION, unavailable_reason=unavailable_reason)
    if obj is not None:
        return InsightResult("llm", obj.model_dump(), None, facts=facts, notes=facts["notes"], generation=info, prompt_sent=sent)
    return InsightResult("fallback", fallback_sales(facts), reason, facts=facts, notes=facts["notes"], generation=info,
                         prompt_sent=sent)
