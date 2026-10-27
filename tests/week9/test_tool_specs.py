"""Function-calling concept: declarations, validation of the model's arguments, running tools, choosing with retries."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from src.llm.week9.client import FakeLLMClient, LLMResult, LLMTimeoutError
from src.llm.week9.retry import ResilientClient
from src.llm.week9.tool_specs import (
    TOOL_SYSTEM, TOOLS, ToolRuntime, decide_tool, run_tool, validate_call,
)
from tests.week9.test_client_retry import FakeClock, cfg  # noqa: F401  (cfg is a fixture)
from tests.week9.test_data_facts import make_db


def chose(tool, **args):
    return LLMResult(text="", model="m", tool_name=tool, tool_args=args, input_tokens=300, output_tokens=12, latency_ms=900)


# ----------------------------------------------------------------------------------------------- the declarations
def test_two_tools_with_clear_names_and_descriptions():
    assert [t.name for t in TOOLS] == ["search_documents", "get_sales_summary"]
    assert all(len(t.description) > 60 for t in TOOLS) and "query" in TOOLS[0].parameters["required"] and TOOLS[1].parameters["required"] == []


def test_only_simple_keywords_are_sent_to_the_api():
    """Ranges and formats are checked by us; the API schema keeps to what every version understands."""
    sent = json.dumps([t.parameters for t in TOOLS])
    for keyword in ("pattern", "minimum", "maximum", "format", "$ref", "anyOf"):
        assert f'"{keyword}"' not in sent


def test_the_system_message_says_when_not_to_call_a_tool():
    assert "do not call a tool" in TOOL_SYSTEM and "never invent values" in TOOL_SYSTEM


def test_the_sdk_accepts_the_declarations():
    pytest.importorskip("google.genai")
    from google.genai import types
    from src.llm.week9.client import GeminiClient
    tool = GeminiClient._build_tool(types, list(TOOLS))
    assert [d.name for d in tool.function_declarations] == ["search_documents", "get_sales_summary"]
    assert tool.function_declarations[0].parameters.required == ["query"]


# ----------------------------------------------------------------------------------------------- validating arguments
@pytest.mark.parametrize("name, args, expected", [
    ("search_documents", {"query": "  total due  "}, {"query": "total due"}),
    ("search_documents", {"query": "x", "top_k": 3.0}, {"query": "x", "top_k": 3}),          # JSON numbers arrive as 3.0
    ("search_documents", {"query": "x", "document_type": "invoice", "month": "2024-02", "project": "projectBeta"},
     {"query": "x", "document_type": "invoice", "month": "2024-02", "project": "projectBeta"}),
    ("get_sales_summary", {}, {}),
    ("get_sales_summary", {"category": "Furniture", "start": "2024-01", "end": "2024-06"}, {"category": "Furniture", "start": "2024-01", "end": "2024-06"}),
    ("get_sales_summary", None, {}),
])
def test_valid_calls_are_cleaned(name, args, expected):
    assert validate_call(name, args) == (expected, None)


@pytest.mark.parametrize("name, args, message", [
    ("drop_tables", {}, "unknown tool"),
    (None, {}, "unknown tool"),
    ("search_documents", {"query": "x", "colour": "red"}, "unknown arguments"),
    ("search_documents", {}, "missing required"),
    ("search_documents", {"query": "  "}, "non-empty string"),
    ("search_documents", {"query": 5}, "non-empty string"),
    ("search_documents", {"query": "x", "top_k": 0}, "between 1 and 10"),
    ("search_documents", {"query": "x", "top_k": 99}, "between 1 and 10"),
    ("search_documents", {"query": "x", "top_k": 2.5}, "must be an integer"),
    ("search_documents", {"query": "x", "top_k": True}, "must be an integer"),     # True is not a number of passages
    ("search_documents", {"query": "x", "document_type": "memo"}, "must be one of"),
    ("search_documents", {"query": "x", "month": "February"}, "YYYY|2024-03"),
    ("get_sales_summary", {"start": "2024-13"}, "2024-03"),
])
def test_bad_calls_are_refused_with_a_reason(name, args, message):
    clean, error = validate_call(name, args)
    assert clean is None and error is not None
    assert any(part in error for part in message.split("|"))


# ----------------------------------------------------------------------------------------------- running tools
class FakeRetriever:
    def __init__(self, status="retrieved", raises=None):
        self.status, self.raises, self.seen = status, raises, None

    def retrieve(self, question, top_k=None, filters=None):
        self.seen = (question, top_k, filters)
        if self.raises:
            raise self.raises
        chunks = [SimpleNamespace(original_filename="invoice_a.txt", score=0.71234, text="Total 1,200 USD")] if self.status == "retrieved" else []
        return SimpleNamespace(status=self.status, chunks=chunks)


def test_search_passes_the_validated_arguments_to_the_retriever():
    retriever = FakeRetriever()
    out = run_tool("search_documents", {"query": "total due", "project": "projectBeta", "top_k": 2}, ToolRuntime(retriever=retriever))
    assert retriever.seen == ("total due", 2, {"project": "projectBeta"})
    assert out == {"passages": [{"id": 1, "source": "invoice_a.txt", "score": 0.7123, "text": "Total 1,200 USD"}]}
    run_tool("search_documents", {"query": "q"}, ToolRuntime(retriever=retriever))
    assert retriever.seen == ("q", 3, None)  # default top_k, no filters


def test_search_with_nothing_found_says_why():
    assert run_tool("search_documents", {"query": "q"}, ToolRuntime(retriever=FakeRetriever("no_documents_match_filter"))) == {"passages": [], "note": "no_documents_match_filter"}


def test_sales_summary_returns_the_computed_facts(tmp_path):
    out = run_tool("get_sales_summary", {"category": "Food"}, ToolRuntime(retail_db_path=make_db(tmp_path / "r.db")))
    assert out["overall"]["transactions"] == 3 and out["scope"]["category"] == "Food"


@pytest.mark.parametrize("name, args, runtime, fragment", [
    ("search_documents", {"query": "q"}, ToolRuntime(), "no retriever"),
    ("get_sales_summary", {}, ToolRuntime(), "no retail database"),
    ("search_documents", {"query": "q"}, ToolRuntime(retriever=FakeRetriever(raises=ValueError("Unknown filter"))), "ValueError: Unknown filter"),
    ("search_documents", {"query": "q"}, ToolRuntime(retriever=FakeRetriever(raises=RuntimeError("index exploded"))), "RuntimeError: index exploded"),
    ("get_sales_summary", {"category": "Nope"}, "db", "Unknown category"),
    ("get_sales_summary", {}, ToolRuntime(retail_db_path="/no/such/file.db"), "run_pipeline"),
    ("launch_rocket", {}, ToolRuntime(), "unknown tool"),
])
def test_a_tool_never_raises_it_returns_an_error(tmp_path, name, args, runtime, fragment):
    if runtime == "db":
        runtime = ToolRuntime(retail_db_path=make_db(tmp_path / "r.db"))
    out = run_tool(name, args, runtime)
    assert set(out) == {"error"} and fragment in out["error"]


# ----------------------------------------------------------------------------------------------- the model's choice
def test_a_valid_choice_is_returned_with_its_cost():
    d = decide_tool(FakeLLMClient([chose("get_sales_summary", category="Furniture", start="2024-01")]), "Furniture sales since January 2024?")
    assert (d.tool, d.args, d.error, d.answer_text) == ("get_sales_summary", {"category": "Furniture", "start": "2024-01"}, None, None)
    assert d.generation["input_tokens"] == 300 and d.generation["model_latency_ms"] == 900


def test_the_question_and_the_tools_reach_the_model():
    client = FakeLLMClient([chose("search_documents", query="x")])
    decide_tool(client, "  What is due?  ")
    assert client.calls[0]["user"] == "What is due?" and client.calls[0]["system"] == TOOL_SYSTEM


def test_arguments_the_model_invented_are_caught_not_run():
    d = decide_tool(FakeLLMClient([chose("search_documents", query="x", top_k=500)]), "q")
    assert d.tool == "search_documents" and d.args == {} and "between 1 and 10" in d.error


def test_a_tool_that_does_not_exist_is_caught():
    d = decide_tool(FakeLLMClient([chose("delete_everything", confirm="yes")]), "q")
    assert d.tool == "delete_everything" and "unknown tool" in d.error


def test_no_tool_needed_means_the_model_answers_in_words():
    d = decide_tool(FakeLLMClient([LLMResult(text="  Paris.  ", model="m")]), "What is the capital of France?")
    assert d.tool is None and d.answer_text == "Paris." and d.error is None


def test_a_model_failure_or_no_client_is_a_result_not_a_crash():
    d = decide_tool(FakeLLMClient([LLMTimeoutError("slow")]), "q")
    assert d.tool is None and d.error.startswith("LLMTimeoutError")
    assert decide_tool(None, "q", "LLMConfigError: no key").error == "LLMConfigError: no key"
    with pytest.raises(ValueError):
        decide_tool(FakeLLMClient(["x"]), "   ")


# ----------------------------------------------------------------------------------------------- retries apply to the choice too
def test_choosing_a_tool_is_retried_like_any_other_request(cfg):  # noqa: F811
    clock = FakeClock()
    inner = FakeLLMClient([LLMTimeoutError("slow"), chose("get_sales_summary")])
    client = ResilientClient(inner, dataclasses.replace(cfg, max_rpm=600), sleep=clock.sleep, clock=clock.clock, jitter=lambda: 0.5)
    d = decide_tool(client, "revenue?")
    assert d.tool == "get_sales_summary" and d.generation["attempts"] == 2 and len(inner.calls) == 2
