"""The real nodes against a FAKE assistant (no model, no tools, no network). The real Assistant is checked by the live run."""
from types import SimpleNamespace as NS

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.llm.week11.graph import build_graph
from src.llm.week11.nodes import make_nodes
from src.llm.week11.state import model_requests

SALES = {"scope": {}, "overall": {"transactions": 3, "total_revenue": 120.5, "average_order_value": 40.17}}
PASSAGES = {"passages": [{"id": 1, "source": "policy.txt", "score": 0.9, "text": "Refunds within 30 days."}]}


class FakeLog:
    def __init__(self):
        self.steps = []

    def step(self, stage, text):
        self.steps.append((stage, text))

    def start(self, question):
        self.steps.append(("QUESTION", question))


class FakeAssistant:
    def __init__(self, choice, ran=None, summarizer="default"):
        self.log, self._choice, self._ran, self.run_calls, self.choose_calls = FakeLog(), choice, ran, [], 0
        self.summarizer = says("The answer is in the data [1].") if summarizer == "default" else summarizer

    def choose(self, question):
        self.choose_calls += 1
        return self._choice

    def _run(self, question, name, args, summarize):
        self.run_calls.append((question, name, args, summarize))
        return self._ran


def says(text):
    return RunnableLambda(lambda _: AIMessage(text))


def broken(_):
    raise RuntimeError("the model must not be called")


def choice(**kw):
    base = dict(tool=None, args={}, text=None, error=None, calls=1, tokens={"input": 10, "output": 2})
    return NS(**{**base, **kw})


def ran(result, status="ok"):
    return NS(result=result, tool_status=status, run_args={}, clean_notes=[])


def run(assistant, question="q"):
    return build_graph(make_nodes(assistant)).invoke({"question": question, "trace": []})


def test_sql_route_runs_the_chosen_tool_without_summary():
    a = FakeAssistant(choice(tool="get_sales_summary", args={"category": "Furniture"}), ran(SALES))
    out = run(a, "revenue of furniture?")
    assert out["trace"] == ["check_input", "classify", "sql_route", "validate_result", "answer"]
    assert out["route"] == "sql" and out["tool"] == "get_sales_summary"
    assert a.run_calls == [("revenue of furniture?", "get_sales_summary", {"category": "Furniture"}, False)]
    assert out["choose_calls"] == 1 and out["choose_tokens"] == {"input": 10, "output": 2}


def test_rag_route():
    out = run(FakeAssistant(choice(tool="search_documents", args={"query": "refunds"}), ran(PASSAGES)))
    assert out["trace"] == ["check_input", "classify", "rag_route", "validate_result", "answer"]
    assert out["route"] == "rag" and out["outcome"] == "answer" and out["answer"] == "The answer is in the data [1]."


def test_no_tool_goes_to_fallback_and_runs_nothing():
    a = FakeAssistant(choice(text="It is sunny."))
    out = run(a)
    assert out["trace"] == ["check_input", "classify", "fallback"] and out["reason"] == "no_tool" and a.run_calls == []
    assert out["outcome"] == "fallback"


def test_model_not_available_and_model_failed():
    out = run(FakeAssistant(choice(error="no key", calls=0)))
    assert out["reason"] == "model_unavailable" and out["choose_calls"] == 0 and out["error"] == "no key"
    assert run(FakeAssistant(choice(error="timeout", calls=1)))["reason"] == "model_failed"


def test_unknown_tool_goes_to_fallback():
    assert run(FakeAssistant(choice(tool="drop_tables")))["reason"] == "unknown_tool"


def test_rejected_call_and_tool_error_end_in_fallback():
    rejected = run(FakeAssistant(choice(tool="search_documents"), ran({"error": "top_k must be 1..10", "rejected": True}, "rejected")))
    assert rejected["trace"] == ["check_input", "classify", "rag_route", "validate_result", "fallback"]
    assert rejected["reason"] == "rejected_call" and rejected["error"] == "top_k must be 1..10"
    failed = run(FakeAssistant(choice(tool="get_sales_summary"), ran({"error": "unknown category Toys"}, "error")))
    assert failed["trace"][-1] == "fallback" and failed["reason"] == "tool_error"


def test_empty_and_too_long_input_never_reach_the_model():
    for question, reason in (("", "empty_question"), ("   ", "empty_question"), ("x" * 1001, "question_too_long")):
        a = FakeAssistant(choice(tool="get_sales_summary"), ran(SALES))
        out = run(a, question)
        assert out["trace"] == ["check_input", "fallback"] and out["reason"] == reason
        assert a.choose_calls == 0 and a.run_calls == [] and out["choose_calls"] == 0
        assert [s[0] for s in a.log.steps] == ["QUESTION", "INPUT", "ANSWER"]   # one log entry, with its own id


def test_input_at_the_limit_is_accepted():
    a = FakeAssistant(choice(tool="get_sales_summary"), ran(SALES))
    assert run(a, "x" * 1000)["trace"][1] == "classify"


def test_no_data_goes_to_fallback():
    empty = {"scope": {}, "overall": {"transactions": 0}}
    out = run(FakeAssistant(choice(tool="get_sales_summary"), ran(empty)))
    assert out["reason"] == "no_data" and out["trace"][-1] == "fallback"


def test_rag_without_passages_is_NOT_a_fallback():
    out = run(FakeAssistant(choice(tool="search_documents"), ran({"passages": [], "note": "no_passages"})))
    assert out["trace"][-1] == "answer" and out["route"] == "rag"


def test_fallback_message_follows_the_language_of_the_question():
    assert "tools" not in run(FakeAssistant(choice(text="x")), "ما هو الطقس اليوم؟")["answer"]   # the Arabic text, not the English one
    assert run(FakeAssistant(choice(text="x")), "weather?")["answer"].startswith("I can only answer")


def test_log_has_one_route_line():
    a = FakeAssistant(choice(tool="get_sales_summary"), ran(SALES))
    run(a)
    assert [s for s in a.log.steps if s[0] == "ROUTE"] == [("ROUTE", "sql")]


def test_answer_is_the_models_summary_and_costs_two_requests():
    out = run(FakeAssistant(choice(tool="get_sales_summary"), ran(SALES), says("The total revenue is 120.5.")))
    assert out["answer"] == "The total revenue is 120.5." and out["outcome"] == "answer"
    assert out["summary"].source == "model" and model_requests(out) == 2


def test_rag_without_passages_is_not_found_with_no_summary_request():
    out = run(FakeAssistant(choice(tool="search_documents"), ran({"passages": []}), RunnableLambda(broken)), "What is in the Gamma invoice?")
    assert out["outcome"] == "not_found" and out["route"] == "rag" and out["summary"].source == "none"
    assert "does not contain" in out["answer"] and model_requests(out) == 1


def test_model_says_not_found():
    out = run(FakeAssistant(choice(tool="search_documents"), ran(PASSAGES), says("NOT_FOUND")))
    assert out["outcome"] == "not_found" and model_requests(out) == 2


def test_a_failing_summary_model_still_gives_an_answer_built_from_the_data():
    out = run(FakeAssistant(choice(tool="get_sales_summary"), ran(SALES), RunnableLambda(broken)))
    assert out["outcome"] == "answer" and out["summary"].source == "fallback"
    assert "120.5" in out["answer"] and out["summary"].fallback_reason


def test_model_requests_of_a_fallback():
    assert model_requests(run(FakeAssistant(choice(text="x")))) == 1
    assert model_requests(run(FakeAssistant(choice(text="x")), "")) == 0
