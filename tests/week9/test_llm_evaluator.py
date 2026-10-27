"""LLM evaluator: facts check, reply cache + budget, row scoring, abort on quota, variants. No network, no model."""

import json
from types import SimpleNamespace

import pytest

from src.llm.week9.client import FakeLLMClient, LLMAuthError, LLMQuotaExhaustedError, LLMResult
from src.rag.week8.answer_generator import ExtractiveGenerator
from src.rag.week8.answerability import AnswerabilityChecker
from src.rag.week8.rag_service import RagResponse
from src.rag.week9 import evaluator as ev


# ----------------------------------------------------------------------------------------------- facts
@pytest.mark.parametrize("answer, facts, expected", [
    ("The total amount due is $4,500 USD.", ["4,500"], (True, "exact")),
    ("Total: 1,250.00 USD", ["1250"], (True, "exact")),
    ("المبلغ ٤٥٠٠ دولار", ["4500"], (True, "exact")),
    ("تجرى المقابلات يوم الأحد بتاريخ 6/7.", ["6/7", "الأحد"], (True, "exact")),
    ("يجب الدفع خلال 30 يوماً", ["30 days"], (True, "numbers")),        # translated word, same number
    ("يجب الدفع خلال 60 يوماً", ["30 days"], (False, None)),
    ("The report is due soon", ["read-only"], (False, None)),           # no number: only a literal match counts
    ("Total 1250, due in 30 days", ["1,250", "30 days"], (True, "exact")),
])
def test_facts(answer, facts, expected):
    assert ev.check_facts(answer, facts) == expected


# ----------------------------------------------------------------------------------------------- cache + budget
def ok(text="x", attempts=1):
    return LLMResult(text=text, model="m", attempts=attempts, input_tokens=10, output_tokens=2, thinking_tokens=5)


def cache(tmp_path, script, max_requests=10, fingerprint="m|v1"):
    inner = FakeLLMClient(script)
    return ev.CachingClient(inner, tmp_path / "cache.json", fingerprint, max_requests), inner


def test_a_second_identical_request_is_served_from_the_cache(tmp_path):
    client, inner = cache(tmp_path, [ok("first"), ok("second")])
    assert client.generate("s", "u").text == "first"
    assert client.generate("s", "u").text == "first"
    assert len(inner.calls) == 1 and client.cache_hits == 1 and client.requests_used == 1


def test_the_cache_survives_a_restart_and_keeps_the_token_counts(tmp_path):
    cache(tmp_path, [ok("kept")])[0].generate("s", "u")
    again, inner = cache(tmp_path, [ok("new")])
    result = again.generate("s", "u")
    assert result.text == "kept" and result.thinking_tokens == 5 and inner.calls == [] and again.requests_used == 0


@pytest.mark.parametrize("system, user, fingerprint", [("s2", "u", "m|v1"), ("s", "u2", "m|v1"), ("s", "u", "m|v2")])
def test_any_change_to_the_request_or_the_setup_is_a_new_request(tmp_path, system, user, fingerprint):
    cache(tmp_path, [ok("old")])[0].generate("s", "u")
    other, inner = cache(tmp_path, [ok("new")], fingerprint=fingerprint)
    assert other.generate(system, user).text == "new" and len(inner.calls) == 1


def test_the_budget_stops_real_requests_but_not_cache_hits(tmp_path):
    client, inner = cache(tmp_path, [ok("a"), ok("b"), ok("c")], max_requests=2)
    client.generate("s", "1"); client.generate("s", "2")
    with pytest.raises(LLMQuotaExhaustedError, match="budget"):
        client.generate("s", "3")
    assert client.generate("s", "1").text == "a"  # still free


def test_retries_count_against_the_budget_because_they_use_quota_too(tmp_path):
    client, _ = cache(tmp_path, [ok("a", attempts=3)], max_requests=3)
    client.generate("s", "1")
    with pytest.raises(LLMQuotaExhaustedError):
        client.generate("s", "2")


def test_failures_are_not_cached_and_still_cost_budget(tmp_path):
    error = LLMAuthError("bad key")
    client, inner = cache(tmp_path, [error, ok("fine")], max_requests=5)
    with pytest.raises(LLMAuthError):
        client.generate("s", "u")
    assert client.requests_used == 1 and client.generate("s", "u").text == "fine"


def test_a_corrupt_cache_file_is_ignored(tmp_path):
    (tmp_path / "cache.json").write_text("{not json", encoding="utf-8")
    client, _ = cache(tmp_path, [ok("x")])
    assert client.generate("s", "u").text == "x"


# ----------------------------------------------------------------------------------------------- scoring a row
def q(**kw):
    base = {"id": 1, "question": "What is the total amount due?", "should_reject": False, "kind": "answerable",
            "split": "dev", "expected_sources": ["invoice_*"], "answer_contains": ["1250"]}
    return {**base, **kw}


CITED = {"citation_id": 1, "original_filename": "invoice_2024.txt", "text": "Total 1,250.00", "chunk_id": "c"}


def response(status="answered", answer="Total 1250 USD", citations=(CITED,), reasons=(), generation=None, retrieved=(CITED,)):
    return RagResponse(status, answer, list(citations), answerability={"answerable": status == "answered", "reasons": list(reasons), "signals": {}},
                       retrieved=list(retrieved), generation=generation, latency_ms=100)


class FakeService:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def ask(self, question, top_k=None, filters=None, include_prompt=False):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


LLM_OK = {"mode": "llm", "fallback_reason": None, "attempts": 1, "repairs": 0, "input_tokens": 800, "output_tokens": 40, "thinking_tokens": 100}


def test_a_correct_llm_answer_is_scored_with_its_cost():
    row = ev.evaluate_one(FakeService(response(generation=LLM_OK)), q(), 3)
    assert row["outcome"] == "correct" and row["facts_rule"] == "exact" and row["input_tokens"] == 800 and row["thinking_tokens"] == 100


@pytest.mark.parametrize("kw, outcome", [
    ({"answer": "Total 999 USD"}, "right_source_missing_fact"),
    ({"citations": ({**CITED, "original_filename": "report.txt"},)}, "wrong_source"),
    ({"status": "no_relevant_context"}, "false_refusal"),
])
def test_answerable_outcomes(kw, outcome):
    assert ev.evaluate_one(FakeService(response(generation=LLM_OK, **kw)), q(), 3)["outcome"] == outcome


def test_unanswerable_outcomes_and_who_rejected():
    leak = ev.evaluate_one(FakeService(response(generation=LLM_OK)), q(should_reject=True, kind="near_topic_missing"), 3)
    assert leak["outcome"] == "leak"
    by_llm = response("no_relevant_context", reasons=["llm_not_found: the model judged..."], generation=LLM_OK)
    by_gate = response("no_relevant_context", reasons=["only 33% of the question's key words..."])
    by_strict = response("no_relevant_context", reasons=["stricter gate for a verbatim answer (mode=extractive_fallback): ..."], generation={"mode": "extractive_fallback", "fallback_reason": "guardrails: x"})
    by_filter = response("no_documents_match_filter")
    got = [ev.evaluate_one(FakeService(r), q(should_reject=True), 3) for r in (by_llm, by_gate, by_strict, by_filter)]
    assert [g["outcome"] for g in got] == ["correct_rejection"] * 4
    assert [g["rejected_by"] for g in got] == ["llm", "gate", "strict_gate", "filter"]


# ----------------------------------------------------------------------------------------------- infrastructure vs quality
def fallback(reason):
    return response(generation={"mode": "extractive_fallback", "fallback_reason": reason, "attempts": 1})


@pytest.mark.parametrize("reason, kind, abort", [
    ("LLMQuotaExhaustedError: Quota exhausted (429)", "LLMQuotaExhaustedError", True),
    ("LLMAuthError: bad key", "LLMAuthError", True),
    ("LLMConfigError: GEMINI_API_KEY is not set", "LLMConfigError", True),
    ("LLMTimeoutError: Request timed out", "LLMTimeoutError", False),
    ("LLMRateLimitError: 429", "LLMRateLimitError", False),
])
def test_infrastructure_failures_are_not_scored_as_answers(reason, kind, abort):
    row = ev.evaluate_one(FakeService(fallback(reason)), q(), 3)
    assert row["outcome"] == "not_evaluated" and row["fallback_kind"] == kind and row["abort"] is abort


@pytest.mark.parametrize("reason", ["guardrails: the output is not valid JSON", "LLMBlockedError: the model returned no text"])
def test_the_models_own_failures_are_scored(reason):
    row = ev.evaluate_one(FakeService(fallback(reason)), q(), 3)
    assert row["outcome"] == "correct" and row["fallback_kind"] in ("guardrails", "LLMBlockedError")  # the extractive answer is what the system returned


def test_a_crashing_question_is_an_error_row_not_a_crash():
    assert ev.evaluate_one(FakeService(RuntimeError("boom")), q(), 3)["outcome"] == "error"


def test_the_run_stops_at_the_first_quota_failure_and_keeps_the_partial_rows():
    service = FakeService(response(generation=LLM_OK), fallback("LLMQuotaExhaustedError: Quota exhausted"), response(generation=LLM_OK))
    rows, aborted = ev.evaluate_questions(service, [q(id=1), q(id=2), q(id=3)], 3, "llm@0.33", progress=False)
    assert [r["outcome"] for r in rows] == ["correct", "not_evaluated"] and aborted.startswith("LLMQuotaExhaustedError") and service.calls == 2


def test_transient_failures_do_not_stop_the_run():
    service = FakeService(fallback("LLMTimeoutError: slow"), response(generation=LLM_OK))
    rows, aborted = ev.evaluate_questions(service, [q(id=1), q(id=2)], 3, "llm@0.33", progress=False)
    assert aborted is None and [r["outcome"] for r in rows] == ["not_evaluated", "correct"]


# ----------------------------------------------------------------------------------------------- variants
def test_parse_variant():
    assert ev.parse_variant("llm@0.33") == ("llm@0.33", "llm", 0.33)
    assert ev.parse_variant(" extractive@0.5 ") == ("extractive@0.50", "extractive", 0.5)
    for bad in ("llm", "gpt@0.3", "llm@abc", "llm@1.5"):
        with pytest.raises(ValueError):
            ev.parse_variant(bad)


def test_apply_variant_uses_the_same_two_tier_rule_as_the_live_service():
    base = AnswerabilityChecker(set(), min_lexical_coverage=0.33)
    service = SimpleNamespace(checker=None, strict_checker=None, generator=None)
    generators = {"llm": "LLM", "extractive": "EXT"}
    ev.apply_variant(service, "llm", 0.33, base, 0.5, generators)
    assert (service.checker.min_lexical_coverage, service.strict_checker.min_lexical_coverage, service.generator) == (0.33, 0.5, "LLM")
    ev.apply_variant(service, "extractive", 0.5, base, 0.5, generators)
    assert service.checker.min_lexical_coverage == 0.5 and service.strict_checker is None and service.generator == "EXT"
    ev.apply_variant(service, "llm", 0.0, base, 0.5, generators)
    assert service.checker.min_lexical_coverage == 0.0 and service.strict_checker.min_lexical_coverage == 0.5


# ----------------------------------------------------------------------------------------------- summary + report
def rows_for_summary():
    svc = lambda *r: FakeService(*r)
    llm_cat = {**LLM_OK, "attempts": 2}
    out = [ev.evaluate_one(svc(response(generation=LLM_OK)), q(id=1), 3),
           ev.evaluate_one(svc(response(generation=llm_cat)), q(id=2, answer_contains=[]), 3),
           ev.evaluate_one(svc(response("no_relevant_context", reasons=["llm_not_found: x"], generation=LLM_OK)), q(id=3, should_reject=True, kind="near_topic_missing"), 3),
           ev.evaluate_one(svc(response(generation=LLM_OK)), q(id=4, should_reject=True, kind="near_topic_missing"), 3),
           ev.evaluate_one(svc(response("no_relevant_context", reasons=["only 20% ..."])), q(id=5, should_reject=True, kind="off_topic"), 3),
           ev.evaluate_one(svc(fallback("LLMQuotaExhaustedError: q")), q(id=6), 3)]
    return out


def test_summary_counts_and_cost():
    s = ev.summarize_variant(rows_for_summary())
    assert (s["questions"], s["evaluated"], s["not_evaluated"]) == (6, 5, 1)
    assert s["answerable"]["correct"] == 2 and s["unanswerable"]["leak"] == 1 and s["unanswerable"]["correct_rejection"] == 2
    assert s["correct_rejection_by"] == {"llm": 1, "gate": 1} and s["leaks_by_kind"] == {"near_topic_missing": 1}
    assert s["llm_requests"] == 4 and s["tokens_mean"] == {"input": 800, "output": 40, "thinking": 100}
    assert s["tokens_total"]["thinking"] == 400 and s["retried_requests"] == 1


def test_summary_of_a_run_with_nothing_evaluated_does_not_crash():
    s = ev.summarize_variant([ev.evaluate_one(FakeService(fallback("LLMQuotaExhaustedError: q")), q(), 3)])
    assert s["evaluated"] == 0 and s["not_evaluated"] == 1 and s["llm_requests"] == 0


def test_the_report_shows_the_stop_the_decisions_and_the_cost():
    a = rows_for_summary()
    b = [dict(r, outcome="correct") if r["id"] == 4 else r for r in a]
    results = {"llm@0.50": {"rows": a, "summary": ev.summarize_variant(a)}, "llm@0.33": {"rows": b, "summary": ev.summarize_variant(b)}}
    meta = {"model": "m", "prompt_version": "v1", "top_k": 3, "requests_used": 5, "max_requests": 300, "cache_hits": 2,
            "aborted": "LLMQuotaExhaustedError: q"}
    text = ev.format_comparison(results, meta, ("llm@0.50", "llm@0.33"))
    for expected in ("RUN STOPPED", "not_evaluated", "llm@0.33", "DECISIONS THAT DIFFER", "leak -> correct", "think/req", "not evidence"):
        assert expected in text


# ----------------------------------------------------------------------------------------------- picking a small set
def test_select_questions_keeps_the_given_order_and_drops_duplicates():
    pool = [q(id=i) for i in (1, 7, 27, 38, 48)]
    assert [x["id"] for x in ev.select_questions(pool, "38, 7,38,1")] == [38, 7, 1]


@pytest.mark.parametrize("text, message", [("7,999", "unknown question ids"), ("a,b", "comma list"), (" , ", "empty")])
def test_select_questions_refuses_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        ev.select_questions([q(id=1), q(id=7)], text)


# ----------------------------------------------------------------------------------------------- latency of cached replies
def test_latency_is_the_models_time_not_the_time_a_cached_reply_took():
    fast_cache = {**LLM_OK, "model_latency_ms": 7000}
    rows = [ev.evaluate_one(FakeService(response(generation=fast_cache)), q(id=i), 3) for i in (1, 2, 3)]
    assert all(r["latency_ms"] == 100 for r in rows)  # what ask() measured (a cache hit)
    assert ev.summarize_variant(rows)["latency_ms"] == {"p50": 7000, "p95": 7000}


def test_rows_without_a_model_time_do_not_pull_the_latency_down():
    rows = [ev.evaluate_one(FakeService(response(generation=LLM_OK)), q(id=1), 3)]
    assert ev.summarize_variant(rows)["latency_ms"] == {"p50": None, "p95": None}
