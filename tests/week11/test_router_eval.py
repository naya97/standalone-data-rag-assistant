
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.llm.week10.evaluation import Case
from src.llm.week10.summarize import REJECTED_PREFIX, Summary
from src.llm.week11.evaluation import (SELECTED_IDS, RouterCase, expectations, format_report, load_designed_cases,
                                       load_existing_cases, router_row, run_router_cases)
from src.llm.week11.validation import MAX_QUESTION_CHARS

ROOT = Path(__file__).resolve().parents[2]
PASSAGES = [{"id": 1, "source": "invoice.txt", "score": 0.9, "text": "Total amount due: $4,500"}]


def doc_case(**kw):
    base = dict(id=1, question="q", split="dev", kind="answerable", domain="documents", expected_tool="search_documents",
                should_reject=False, expected_sources=("invoice.txt",), answer_contains=("4,500",))
    return Case(**{**base, **kw})


def sales_case(**kw):
    base = dict(id=1001, question="q", split="dev", kind="sales_answerable", domain="sales", expected_tool="get_sales_summary",
                expected_args={}, facts=("overall.total_revenue",), expect="answer")
    return Case(**{**base, **kw})


def existing(case):
    routes, final, reason = expectations(case)
    return RouterCase(f"Q{case.id}", case.question, case.kind, routes, final, reason, "existing", case)


def designed(label, kind, routes, final, reason=None):
    return RouterCase(label, "q", kind, tuple(routes), final, reason, "designed")


def state(**kw):
    return {"question": "q", "route": "fallback", "trace": [], "choose_calls": 1, "choose_tokens": {"input": 5, "output": 1}, **kw}


def rag_state(text="The total is $4,500 [1].", **kw):
    return state(tool="search_documents", tool_status="ok", args={"query": "x"}, result={"passages": PASSAGES}, route="rag",
                 outcome="answer", answer=text, summary=Summary(text, "model", "answered", [1], requests=1), **kw)


#  expectations
def test_expectations_of_existing_questions():
    assert expectations(doc_case()) == (("rag",), "answer", None)
    assert expectations(doc_case(should_reject=True)) == (("rag",), "not_found", None)
    assert expectations(doc_case(kind="off_topic", expected_tool=None, should_reject=True)) == (("fallback",), "fallback", "no_tool")
    assert expectations(sales_case()) == (("sql",), "answer", None)
    assert expectations(sales_case(expect="not_found")) == (("sql",), "not_found", None)
    assert expectations(sales_case(expect="tool_error")) == (("sql",), "fallback", "tool_error")


def test_designed_cases():
    cases = load_designed_cases(ROOT / "data" / "eval" / "week11" / "router_cases.json")
    assert [c.label for c in cases] == [f"R{i}" for i in range(1, 11)]
    by = {c.label: c for c in cases}
    assert len(by["R7"].question) > MAX_QUESTION_CHARS and by["R6"].question.strip() == ""
    assert by["R8"].expected_route == ("rag",) and by["R8"].expected_reason == "rejected_call"
    assert by["R9"].expected_final == "not_found" and by["R10"].expected_reason == "tool_error"
    assert set(by["R4"].expected_route) == {"sql", "rag"}


#  verdicts
def test_existing_answerable_question_passes_and_is_scored_by_week10():
    row = router_row(existing(doc_case()), rag_state(), 1.0, 2)
    assert row["verdict"] == "pass" and row["week10_outcome"] == "correct" and row["content_ok"] is True


def test_wrong_content_fails():
    row = router_row(existing(doc_case()), rag_state("The total is $9,999 [1]."), 1.0, 2)
    assert row["verdict"] == "fail" and row["content_ok"] is False and row["route_ok"] and row["final_ok"]


def test_not_found_expected():
    case = existing(doc_case(should_reject=True, expected_sources=(), answer_contains=()))
    ok = state(tool="search_documents", tool_status="ok", result={"passages": []}, route="rag", outcome="not_found",
               answer="no", summary=Summary("no", "none", "not_found"))
    assert router_row(case, ok, 1.0, 1)["verdict"] == "pass"
    leak = router_row(case, rag_state(), 1.0, 2)
    assert leak["verdict"] == "fail" and leak["week10_outcome"] == "leak" and leak["final_ok"] is False


def test_off_topic_must_not_use_a_tool():
    case = existing(doc_case(kind="off_topic", expected_tool=None, should_reject=True, expected_sources=(), answer_contains=()))
    assert router_row(case, state(outcome="fallback", reason="no_tool", answer="x"), 1.0, 1)["verdict"] == "pass"
    used = router_row(case, rag_state(), 1.0, 2)
    assert used["verdict"] == "fail" and used["route_ok"] is False


def test_a_rejected_call_still_counts_as_classified():
    case = designed("R8", "no_valid_tool_call", ["rag"], "fallback", "rejected_call")
    s = state(tool="search_documents", tool_status="rejected", outcome="fallback", reason="rejected_call", answer="x")
    row = router_row(case, s, 1.0, 1)
    assert row["classified"] == "rag" and row["verdict"] == "pass"


def test_wrong_reason_fails():
    case = designed("R10", "sql_no_data", ["sql"], "fallback", "tool_error")
    s = state(tool="get_sales_summary", tool_status="ok", outcome="fallback", reason="no_data", answer="x")
    row = router_row(case, s, 1.0, 1)
    assert row["verdict"] == "fail" and row["reason_ok"] is False and row["route_ok"] is True


def test_empty_input_case():
    case = designed("R6", "invalid_input", ["fallback"], "fallback", "empty_question")
    row = router_row(case, state(outcome="fallback", reason="empty_question", answer="x", choose_calls=0), 0.0, 0)
    assert row["verdict"] == "pass" and row["model_calls"] == 0


def test_model_failure_is_not_scored_as_wrong():
    case = designed("R3", "out_of_scope", ["fallback"], "fallback", "no_tool")
    quota = router_row(case, state(outcome="fallback", reason="model_failed", error="LLMQuotaExhaustedError: budget"), 1.0, 0)
    assert quota["verdict"] == "not_evaluated" and quota["abort"] is True
    timeout = router_row(case, state(outcome="fallback", reason="model_failed", error="LLMTimeoutError: slow"), 1.0, 1)
    assert timeout["verdict"] == "not_evaluated" and timeout["abort"] is False
    broken = rag_state()
    broken["summary"] = Summary("x", "fallback", "answered", fallback_reason="RuntimeError: boom", requests=1)
    assert router_row(existing(doc_case()), broken, 1.0, 2)["verdict"] == "not_evaluated"
    rejected = rag_state()
    rejected["summary"] = Summary("x", "fallback", "answered", fallback_reason=REJECTED_PREFIX + "bad number", requests=2)
    assert router_row(existing(doc_case()), rejected, 1.0, 3)["verdict"] != "not_evaluated"   # a guardrail rejection IS scored


#  run loop and report
class FakeApp:
    def __init__(self, states, layer):
        self.states, self.layer = list(states), layer

    def invoke(self, value):
        self.layer.requests += 2
        return self.states.pop(0)


def test_run_loop_counts_requests_per_case_and_stops_on_abort():
    layer = NS(requests=10)
    cases = [designed("R1", "out_of_scope", ["fallback"], "fallback", "no_tool"),
             designed("R2", "out_of_scope", ["fallback"], "fallback", "no_tool"),
             designed("R3", "out_of_scope", ["fallback"], "fallback", "no_tool")]
    app = FakeApp([state(outcome="fallback", reason="no_tool", answer="x"),
                   state(outcome="fallback", reason="model_failed", error="LLMQuotaExhaustedError: budget"),
                   state(outcome="fallback", reason="no_tool", answer="x")], layer)
    rows, aborted = run_router_cases(app, NS(layer=layer), cases, progress=None, clock=iter(range(100)).__next__)
    assert [r["attempts"] for r in rows] == [2, 2] and len(rows) == 2          # stopped before R3
    assert aborted.startswith("LLMQuotaExhaustedError") and rows[0]["seconds"] == 1


def test_report_has_its_sections():
    rows = [router_row(existing(doc_case()), rag_state(), 1.0, 2),
            router_row(existing(doc_case(id=2)), rag_state("The total is $9,999 [1]."), 1.0, 2),
            router_row(designed("R4", "ambiguous", ["sql", "rag"], "answer"), rag_state(), 1.0, 2),
            router_row(designed("R3", "out_of_scope", ["fallback"], "fallback", "no_tool"),
                       state(outcome="fallback", reason="no_tool", answer="x"), 1.0, 1),
            router_row(designed("R5", "out_of_scope", ["fallback"], "fallback", "no_tool"),
                       state(outcome="fallback", reason="model_failed", error="LLMTimeoutError: slow"), 1.0, 1)]
    meta = {"model": "m", "requests_sent": 3, "max_requests": 60, "cache_hits": 4, "cache_misses": 3, "aborted": None}
    text = format_report(rows, meta)
    for part in ("ROUTE AND ENDING", "FAILED (1)", "READ THESE BY A PERSON", "COST", "fallbacks by reason", "cases=5 (evaluated 4)",
                 "no threshold is set"):
        assert part in text


#  the project's real files
@pytest.mark.skipif(not (ROOT / "data" / "eval" / "week8").exists(), reason="the project's evaluation files are not here")
def test_the_15_selected_questions_exist_and_have_the_expected_shape():
    cases = load_existing_cases(ROOT / "data" / "eval")
    assert [c.label for c in cases] == [f"Q{i}" for i in SELECTED_IDS] and len(cases) == 15
    assert Counter(c.expected_final for c in cases) == {"answer": 9, "not_found": 4, "fallback": 2}
