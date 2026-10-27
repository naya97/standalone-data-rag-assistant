"""Sales insights with a scripted model: success, repair, every fallback, and that the fallback text never invents a number."""

import json

import pytest

from src.llm.week9.client import FakeLLMClient, LLMQuotaExhaustedError, LLMResult, LLMTimeoutError
from src.llm.week9.data_facts import DataUnavailableError, get_sales_facts
from src.llm.week9.guardrails import check_sales_insights, ungrounded_numbers
from src.llm.week9.insights import fallback_sales, sales_insights
from src.llm.week9.runner import ask_json
from src.llm.week9.schemas import SalesInsights
from tests.week9.test_data_facts import make_db


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "retail.db")


def reply(summary="There were 5 transactions with total revenue 800.",
          insights=("Furniture leads with revenue 500 (62.5% of the total).", "The best month is 2024-02 with revenue 450."),
          caveats=()):
    return json.dumps({"summary": summary, "insights": list(insights), "caveats": list(caveats)})


def run(db, script, **kw):
    client = FakeLLMClient(script)
    return sales_insights(client, db, **kw), client


# ----------------------------------------------------------------------------------------------- the model path
def test_a_grounded_answer_is_used_and_the_facts_travel_with_it(db):
    res, client = run(db, [reply()], focus="Which category leads?")
    assert res.mode == "llm" and res.fallback_reason is None and res.result["insights"][0].startswith("Furniture leads")
    assert res.facts["overall"]["total_revenue"] == 800.0 and res.generation["repairs"] == 0
    user = client.calls[0]["user"]
    assert "Which category leads?" in user and '"total_revenue": 800.0' in user and "Language: en" in user
    assert res.prompt_sent.startswith("### SYSTEM")


def test_the_model_never_gets_to_compute(db):
    res, client = run(db, [reply(summary="Revenue is 800 across 5 transactions, about 160.5 per order."), reply()])
    assert res.mode == "llm" and res.generation["repairs"] == 1  # 160.5 is not in FACTS (it is 160.0): rejected, repaired
    assert "160.5" in client.calls[1]["user"] and "do not appear in FACTS" in client.calls[1]["user"]


def test_two_rejected_answers_give_the_template_with_the_reason(db):
    res, client = run(db, [reply(summary="Revenue is 999."), "not json"])
    assert res.mode == "fallback" and res.fallback_reason.startswith("guardrails:") and len(client.calls) == 2
    assert "800.0" in res.result["summary"] and "Written from a fixed template" in res.result["caveats"][0]


@pytest.mark.parametrize("error", [LLMTimeoutError("slow"), LLMQuotaExhaustedError("quota")])
def test_a_model_failure_still_returns_the_numbers(db, error):
    res, _ = run(db, [error])
    assert res.mode == "fallback" and type(error).__name__ in res.fallback_reason and res.facts["overall"]["transactions"] == 5


def test_no_client_means_the_template_and_says_why(db):
    res = sales_insights(None, db, unavailable_reason="LLMConfigError: GEMINI_API_KEY is not set")
    assert res.mode == "fallback" and "GEMINI_API_KEY" in res.fallback_reason and res.generation["attempts"] == 0


def test_thinking_tokens_and_usage_are_summed(db):
    result = LLMResult(text=reply(), model="m", input_tokens=900, output_tokens=60, thinking_tokens=300)
    res = sales_insights(FakeLLMClient([result]), db)
    assert (res.generation["input_tokens"], res.generation["output_tokens"], res.generation["thinking_tokens"]) == (900, 60, 300)


def test_arabic_is_requested_and_arabic_digits_are_accepted(db):
    text = reply(summary="بلغ إجمالي الإيرادات ٨٠٠ من ٥ عمليات.", insights=["الأثاث الأعلى بإيرادات 500"])
    res, client = run(db, [text], language="ar")
    assert res.mode == "llm" and "Language: ar" in client.calls[0]["user"]


# ----------------------------------------------------------------------------------------------- bad requests are errors
def test_bad_requests_raise_instead_of_falling_back(db, tmp_path):
    with pytest.raises(ValueError, match="language"):
        sales_insights(None, db, language="fr")
    with pytest.raises(ValueError, match="Unknown category"):
        sales_insights(None, db, category="Nope")
    with pytest.raises(DataUnavailableError):
        sales_insights(None, tmp_path / "missing.db")


# ----------------------------------------------------------------------------------------------- the template
def test_the_template_only_contains_numbers_from_the_facts(db):
    facts = get_sales_facts(db)
    out = fallback_sales(facts)
    text = " ".join([out["summary"], *out["insights"], *out["caveats"]])
    assert ungrounded_numbers(text, [json.dumps(facts)]) == []
    assert any("below their revenue target" in i and "Furniture" in i and "50.0%" in i for i in out["insights"])


def test_the_template_has_no_target_line_without_targets(tmp_path):
    out = fallback_sales(get_sales_facts(make_db(tmp_path / "r.db", targets=False)))
    assert not any("target" in i for i in out["insights"])


# ----------------------------------------------------------------------------------------------- the shared runner
def test_ask_json_reports_what_it_did_and_never_raises_for_the_model():
    client = FakeLLMClient([LLMTimeoutError("slow")])
    obj, info, reason, sent = ask_json(client, "sys", "user", SalesInsights, lambda o: [], prompt_version="v")
    assert obj is None and reason.startswith("LLMTimeoutError") and info["prompt_version"] == "v" and sent.startswith("### SYSTEM")


def test_ask_json_stops_after_one_repair():
    client = FakeLLMClient(["x", "y", reply()])
    obj, info, reason, _ = ask_json(client, "s", "u", SalesInsights, lambda o: [], prompt_version="v")
    assert obj is None and info["repairs"] == 1 and len(client.calls) == 2 and "not valid JSON" in reason


def test_the_check_requires_a_summary_and_an_insight():
    empty = SalesInsights(summary=" ", insights=[], caveats=[])
    assert check_sales_insights(empty, "{}") == ["the summary is empty", "at least one insight is required"]


# ----------------------------------------------------------------------------------------------- found by the real run
def test_the_100_percent_reference_is_grounded_so_the_model_can_explain_a_target(db):
    text = reply(summary="Furniture reached 50.0% of its target, below the 100 mark; total revenue was 800.",
                 insights=["Food reached exactly 100.0% of its target."])
    res, _ = run(db, [text])
    assert res.mode == "llm" and res.generation["repairs"] == 0  # "100" is in FACTS (the definition of target_achievement_pct)


def test_the_template_says_so_when_no_category_is_behind_its_target(tmp_path):
    rows = [("T1", "Food", 1, 500.0, "Cash", "Online", "2024-01-01 00:00:00"), ("T2", "Food", 1, 500.0, "Cash", "Online", "2024-01-31 00:00:00")]
    out = fallback_sales(get_sales_facts(make_db(tmp_path / "r.db", rows)))
    assert any(i.startswith("No category is below its revenue target; the lowest is Food at 333.33%") for i in out["insights"])


def test_the_template_carries_the_data_notes_into_its_caveats(tmp_path):
    rows = [("T1", "Food", 1, 100.0, "Cash", "Online", "2024-01-01 00:00:00"), ("T2", "Food", 1, 100.0, "Cash", "Online", "2024-01-31 00:00:00"),
            ("T3", "Food", 1, 20.0, "Cash", "Online", "2024-02-09 00:00:00")]
    facts = get_sales_facts(make_db(tmp_path / "r.db", rows))
    out = fallback_sales(facts)
    assert any("2024-02 is an incomplete month" in c for c in out["caveats"])
    text = " ".join([out["summary"], *out["insights"], *out["caveats"]])
    assert ungrounded_numbers(text, [json.dumps(facts)]) == []
    assert "weakest is 2024-01" in text  # not the half-finished February


def test_model_latency_is_the_models_own_time_even_when_the_reply_is_cached(db):
    result = LLMResult(text=reply(), model="m", latency_ms=900, total_latency_ms=1400, input_tokens=10, output_tokens=5)
    assert sales_insights(FakeLLMClient([result]), db).generation["model_latency_ms"] == 1400
