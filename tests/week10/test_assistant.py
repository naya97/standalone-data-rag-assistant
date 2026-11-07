"""Stage 4: the assistant flow and the readable tool log, with scripted models (no network, no key)."""
import logging
import sqlite3
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.llm.week9.client import LLMConnectionError
from src.llm.week9.tool_specs import ToolRuntime
from src.llm.week10.assistant import Assistant, AssistantAnswer, _LazyRetriever, brief_result
from src.llm.week10.tool_logging import LOGGER_NAME, ToolLog, setup_tool_log
from src.llm.week10.tools import build_tools


def _chunk(n, text):
    return SimpleNamespace(chunk_id=f"c{n}", score=0.9, text=text, section_header=None, original_filename=f"file{n}.md",
                           document_id="d", chunk_index=n, document_type="invoice", month="2024-01", project="projectAlpha", service=None)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def retrieve(self, question, top_k=None, filters=None):
        self.calls.append({"question": question, "top_k": top_k, "filters": filters})
        return SimpleNamespace(status="retrieved", chunks=self.chunks, applied_filters=filters or {}, stale_chunks=0)


class Model:
    """A scripted model runnable: replies in order (AIMessage, or an exception to raise); counts its requests."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), 0

    def __call__(self, _prompt_value):
        self.requests += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def runnable(self):
        return RunnableLambda(self)


def tool_call(name, args, tokens=(500, 12)):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "1", "type": "tool_call"}],
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)})


def words(text, tokens=(100, 20)):
    return AIMessage(content=text, usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)})


@pytest.fixture
def retail_db(tmp_path):
    path = tmp_path / "retail.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sales_clean (transaction_date TEXT, category TEXT, payment_method TEXT, location TEXT, total_spent REAL, quantity INTEGER)")
    con.executemany("INSERT INTO sales_clean VALUES (?,?,?,?,?,?)", [
        ("2024-01-01", "Furniture", "Cash", "Online", 100.0, 1), ("2024-01-31", "Furniture", "Card", "In-store", 300.0, 3),
        ("2024-02-01", "Food", "Card", "Online", 50.0, 5), ("2024-02-29", "Food", "Cash", "In-store", 150.0, 2)])
    con.commit()
    con.close()
    return path


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "logs" / "tool_calls.log"
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    setup_tool_log(path)
    yield path
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def make(chooser, summarizer, retail_db, retriever=None):
    tools = build_tools(ToolRuntime(retriever=retriever, retail_db_path=retail_db))
    return Assistant(chooser.runnable() if chooser else None, summarizer.runnable() if summarizer else None, tools, ToolLog())


# ----------------------------------------------------------------------------------------------- the flow
def test_a_sales_question_uses_two_requests_and_the_sales_tool(retail_db):
    chooser = Model(tool_call("get_sales_summary", {}))
    summarizer = Model(words("Furniture has the highest revenue with 400.0 out of 600.0 in total."))
    answer = make(chooser, summarizer, retail_db).ask("Which category has the highest revenue?")
    assert (answer.route, answer.tool, answer.tool_status) == ("tool", "get_sales_summary", "ok")
    assert answer.text.startswith("Furniture has the highest revenue") and answer.summary.source == "model"
    assert answer.model_requests == 2 and (chooser.requests, summarizer.requests) == (1, 1)
    assert answer.choose_tokens == {"input": 500, "output": 12}
    assert answer.result["overall"]["total_revenue"] == 600.0


def test_a_document_question_passes_the_filters_and_cites_the_passage(retail_db):
    retriever = FakeRetriever([_chunk(1, "Total amount due: 1,250.00 EUR.")])
    chooser = Model(tool_call("search_documents", {"query": "total due", "project": "projectAlpha", "month": "2024-01", "top_k": 2.0}))
    summarizer = Model(words("The total due is 1,250.00 EUR [1]."))
    answer = make(chooser, summarizer, retail_db, retriever).ask("What is the total due for Project Alpha in January 2024?")
    assert answer.tool == "search_documents" and answer.summary.citations == [1]
    assert retriever.calls == [{"question": "total due", "top_k": 2, "filters": {"project": "projectAlpha", "month": "2024-01"}}]


def test_when_no_tool_fits_the_models_words_are_the_answer(retail_db):
    chooser, summarizer = Model(words("The capital of France is Paris.")), Model()
    answer = make(chooser, summarizer, retail_db).ask("What is the capital of France?")
    assert (answer.route, answer.tool, answer.summary) == ("no_tool", None, None)
    assert answer.text == "The capital of France is Paris." and answer.model_requests == 1 and summarizer.requests == 0


def test_an_invalid_call_is_rejected_and_nothing_runs(retail_db):
    retriever = FakeRetriever([_chunk(1, "x")])
    chooser, summarizer = Model(tool_call("search_documents", {"query": "q", "top_k": 99})), Model()
    answer = make(chooser, summarizer, retail_db, retriever).ask("anything")
    assert answer.tool_status == "rejected" and "between 1 and 10" in answer.text
    assert retriever.calls == [] and summarizer.requests == 0 and answer.model_requests == 1


def test_a_failing_tool_is_explained_without_a_summary_request(retail_db):
    chooser, summarizer = Model(tool_call("get_sales_summary", {"category": "Toys"})), Model()
    answer = make(chooser, summarizer, retail_db).ask("Sales of Toys?")
    assert answer.tool_status == "error" and "Unknown category" in answer.text and summarizer.requests == 0


def test_no_summary_returns_the_tool_result_as_it_is(retail_db):
    retriever = FakeRetriever([_chunk(1, "Total amount due: 1,250.00 EUR."), _chunk(2, "Late fee 5%.")])
    chooser, summarizer = Model(tool_call("search_documents", {"query": "q"})), Model()
    answer = make(chooser, summarizer, retail_db, retriever).ask("q", summarize=False)
    assert answer.text.startswith("[1] (file1.md) Total amount due") and answer.summary is None
    assert summarizer.requests == 0 and answer.model_requests == 1


def test_a_failed_model_request_gives_an_unavailable_answer_with_the_reason(retail_db):
    answer = make(Model(LLMConnectionError("Network error: down")), Model(), retail_db).ask("Which category leads?")
    assert answer.route == "unavailable" and "LLMConnectionError" in answer.error and "--tool" in answer.text
    assert answer.model_requests == 1 and answer.tool is None     # the choice request was made, and failed
    assert answer.attempts is None                                  # no request layer in this test: nothing to measure


def test_without_a_model_the_assistant_still_starts_and_says_why(retail_db):
    assistant = Assistant(None, None, build_tools(ToolRuntime(retail_db_path=retail_db)), ToolLog(), unavailable_reason="GEMINI_API_KEY is not set")
    answer = assistant.ask("Which category leads?")
    assert answer.route == "unavailable" and "GEMINI_API_KEY is not set" in answer.text


def test_run_tool_needs_no_model_at_all(retail_db):
    assistant = Assistant(None, None, build_tools(ToolRuntime(retail_db_path=retail_db)), ToolLog())
    answer = assistant.run_tool("get_sales_summary", {"category": "Food"})
    assert answer.tool_status == "ok" and answer.result["overall"]["total_revenue"] == 200.0 and answer.model_requests == 0
    assert assistant.run_tool("get_sales_summary", {"category": "Toys"}).tool_status == "error"
    assert assistant.run_tool("nope", {}).tool_status == "rejected"


def test_run_tool_can_summarize_when_a_model_exists(retail_db):
    summarizer = Model(words("Food revenue is 200.0."))
    assistant = make(Model(), summarizer, retail_db)
    answer = assistant.run_tool("get_sales_summary", {"category": "Food"}, summarize=True, question="Food revenue?")
    assert answer.text == "Food revenue is 200.0." and answer.model_requests == 1


def test_only_the_first_of_several_tool_calls_is_used(retail_db, log_file):
    two = AIMessage(content="", tool_calls=[{"name": "get_sales_summary", "args": {"category": "Food"}, "id": "1", "type": "tool_call"},
                                           {"name": "get_sales_summary", "args": {"category": "Furniture"}, "id": "2", "type": "tool_call"}])
    answer = make(Model(two), Model(words("Food is 200.0.")), retail_db).ask("q")
    assert answer.args == {"category": "Food"}
    assert "1 more tool call(s) ignored" in log_file.read_text(encoding="utf-8")


def test_an_empty_question_is_an_error(retail_db):
    with pytest.raises(ValueError):
        make(Model(), Model(), retail_db).ask("   ")


def test_the_assistant_is_a_runnable_chain(retail_db):
    chain = make(Model(tool_call("get_sales_summary", {})), Model(words("Total 600.0.")), retail_db).as_chain()
    out = chain.invoke("Total revenue?")
    assert isinstance(out, AssistantAnswer) and out.text == "Total 600.0."


def test_a_sales_question_never_loads_the_documents_retriever(retail_db):
    built = []
    lazy = _LazyRetriever(lambda: built.append(1) or FakeRetriever([_chunk(1, "x")]))
    tools = build_tools(ToolRuntime(retriever=lazy, retail_db_path=retail_db))
    Assistant(Model(tool_call("get_sales_summary", {})).runnable(), Model(words("Total 600.0.")).runnable(), tools, ToolLog()).ask("q")
    assert built == []
    lazy.retrieve("a"); lazy.retrieve("b")
    assert built == [1]                       # built once, on first use


# ----------------------------------------------------------------------------------------------- the readable log
def test_the_log_follows_one_question_step_by_step(retail_db, log_file):
    retriever = FakeRetriever([_chunk(1, "Total amount due: 1,250.00 EUR.")])
    chooser = Model(tool_call("search_documents", {"query": "total due", "project": "projectAlpha"}))
    answer = make(chooser, Model(words("The total due is 1,250.00 EUR [1].")), retail_db, retriever).ask("What is due for Alpha?")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and all(f"| {answer.question_id} |" in line for line in lines)
    stages = [line.split("|")[2].split()[0] for line in lines]
    assert stages == ["QUESTION", "MODEL", "TOOL", "RESULT", "SUMMARY", "ANSWER"]
    text = "\n".join(lines)
    assert "chose search_documents" in text and "1 passages from ['file1.md']" in text and "tokens 500/12" in text
    assert "source=model" in text and "citations=[1]" in text
    assert "Total amount due: 1,250.00" not in text       # passages are described, never copied into the log


def test_failures_and_rejections_are_visible_in_the_log(retail_db, log_file):
    make(Model(tool_call("get_sales_summary", {"category": "Toys"})), Model(), retail_db).ask("Toys?")
    make(Model(tool_call("search_documents", {"query": "q", "top_k": 99})), Model(), retail_db).ask("bad call")
    make(Model(LLMConnectionError("Network error: down")), Model(), retail_db).ask("down?")
    text = log_file.read_text(encoding="utf-8")
    assert "get_sales_summary FAILED" in text and "Unknown category" in text
    assert "REJECTED" in text and "between 1 and 10" in text
    assert "MODEL     FAILED LLMConnectionError" in text


def test_setup_is_idempotent_and_the_log_stays_out_of_the_root_logger(tmp_path):
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    path = tmp_path / "a" / "tool_calls.log"
    setup_tool_log(path); setup_tool_log(path); setup_tool_log(path, echo=True); setup_tool_log(path, echo=True)
    assert len(logger.handlers) == 2 and logger.propagate is False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_brief_result_shortens_long_passages():
    long = {"passages": [{"id": 1, "source": "a.md", "score": 0.5, "text": "x" * 500}]}
    assert len(brief_result(long)["passages"][0]["text"]) == 160
    assert brief_result({"error": "boom"}) == {"error": "boom"}
