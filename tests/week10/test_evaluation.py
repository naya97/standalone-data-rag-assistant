"""Stage 6: the evaluation (cases, argument checks, scoring, run loop, reply cache, report). Scripted models: no network, no key."""
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.llm.week9.client import LLMConnectionError, LLMQuotaExhaustedError
from src.llm.week9.config import LLMConfig
from src.llm.week9.tool_specs import ToolRuntime
from src.llm.week10 import evaluation as ev
from src.llm.week10.assistant import Assistant, Choice
from src.llm.week10.evaluation import (Case, ReplyCache, answer_row, argument_issues, choice_row, expected_facts, fingerprint,
                                       format_report, load_cases, resolve_path, run_cases)
from src.llm.week10.request_layer import LangChainRequestLayer, RequestCounter
from src.llm.week10.tool_logging import ToolLog
from src.llm.week10.tools import build_tools

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------------------------- helpers
def _chunk(n, text, name):
    return SimpleNamespace(chunk_id=f"c{n}", score=0.9, text=text, section_header=None, original_filename=name, document_id="d",
                           chunk_index=n, document_type="invoice", month="2024-01", project="projectAlpha", service=None)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k=None, filters=None):
        return SimpleNamespace(status="retrieved", chunks=self.chunks, applied_filters=filters or {}, stale_chunks=0)


class Model:
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), 0

    def __call__(self, _value):
        self.requests += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def runnable(self):
        return RunnableLambda(self)


def calls(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "1", "type": "tool_call"}],
                     usage_metadata={"input_tokens": 500, "output_tokens": 12, "total_tokens": 512})


def says(text):
    return AIMessage(content=text, usage_metadata={"input_tokens": 300, "output_tokens": 20, "total_tokens": 320})


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "retail.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sales_clean (transaction_date TEXT, category TEXT, payment_method TEXT, location TEXT, total_spent REAL, quantity INTEGER)")
    con.executemany("INSERT INTO sales_clean VALUES (?,?,?,?,?,?)", [
        ("2024-01-01", "Furniture", "Cash", "Online", 100.0, 1), ("2024-01-31", "Furniture", "Card", "In-store", 300.0, 3),
        ("2024-02-01", "Food", "Card", "Online", 50.0, 5), ("2024-02-29", "Food", "Cash", "In-store", 150.0, 2)])
    con.commit(); con.close()
    return path


def assistant(chooser, summarizer=None, retriever=None, db_path=None):
    tools = build_tools(ToolRuntime(retriever=retriever, retail_db_path=db_path))
    return Assistant(chooser.runnable(), summarizer.runnable() if summarizer else None, tools, ToolLog())


DOC = dict(domain="documents", split="dev", expected_tool=ev.DOC_TOOL)
INVOICE = Case(id=1, question="What is the total amount due for Project Alpha in January 2024?", kind="answerable",
               expected_sources=("invoice.pdf", "invoice.txt"), answer_contains=("4,500",), **DOC)
CHUNKS = [_chunk(1, "Total amount due: $4,500 USD. Payment within 30 days.", "invoice.pdf"), _chunk(2, "Unit price $8.50.", "other.pdf")]


def ask_doc(case, chooser, summarizer, chunks=CHUNKS):
    a = assistant(chooser, summarizer, FakeRetriever(chunks))
    return answer_row(case, a.ask(case.question))


# ----------------------------------------------------------------------------------------------- arguments
def test_a_well_supported_call_has_no_issues():
    q = "What is the total amount due for Project Alpha in January 2024?"
    assert argument_issues(q, {"query": "total due", "project": "projectAlpha", "month": "2024-01"}) == []


@pytest.mark.parametrize("question, args, expected", [
    ("Project Alpha password length?", {"project": "Alpha"}, "format"),
    ("Project Alpha password length?", {"project": "projectBeta"}, "invented"),
    ("What is the password length?", {"document_type": "invoice"}, "invented"),
    ("What was the total in January?", {"month": "2024-01"}, "assumed"),
    ("What was the total due?", {"month": "2024-03"}, "invented"),
    ("What is the password length for Project Alpha?", {}, "missed"),
])
def test_arguments_the_question_does_not_support_are_named(question, args, expected):
    issues = argument_issues(question, args)
    assert len(issues) == 1 and issues[0].startswith(expected + ":")


def test_arabic_wording_supports_arguments_too():
    assert argument_issues("ما إجمالي فاتورة مشروع Alpha في يناير 2024؟", {"project": "projectAlpha", "document_type": "invoice", "month": "2024-01"}) == []
    assert argument_issues("ما سياسة مشروع Alpha؟", {"project": "projectAlpha", "document_type": "policy"}) == []


# ----------------------------------------------------------------------------------------------- the choice alone
@pytest.mark.parametrize("expected_tool, choice, outcome", [
    (ev.DOC_TOOL, Choice(tool=ev.DOC_TOOL, args={"query": "q"}, calls=1), "tool_ok"),
    (ev.DOC_TOOL, Choice(tool=ev.SALES_TOOL, calls=1), "wrong_tool"),
    (ev.DOC_TOOL, Choice(text="I think...", calls=1), "no_tool"),
    (None, Choice(text="Paris.", calls=1), "tool_ok"),
    (None, Choice(tool=ev.DOC_TOOL, args={"query": "q"}, calls=1), "unneeded_tool"),
])
def test_choice_outcomes(expected_tool, choice, outcome):
    case = Case(id=1, question="q", kind="answerable", domain="documents", split="dev", expected_tool=expected_tool)
    assert choice_row(case, choice)["outcome"] == outcome


def test_sales_arguments_must_match_exactly():
    case = Case(id=1005, question="Furniture revenue?", kind="sales_answerable", domain="sales", split="dev",
                expected_tool=ev.SALES_TOOL, expected_args={"category": "Furniture"})
    assert choice_row(case, Choice(tool=ev.SALES_TOOL, args={"category": "Furniture"}, calls=1))["outcome"] == "tool_ok"
    assert choice_row(case, Choice(tool=ev.SALES_TOOL, args={}, calls=1))["outcome"] == "wrong_args"


def test_a_failed_choice_is_not_evaluated_and_a_quota_stop_aborts():
    case = Case(id=1, question="q", kind="answerable", domain="documents", split="dev", expected_tool=ev.DOC_TOOL)
    network = choice_row(case, Choice(error="LLMConnectionError: down", calls=1))
    quota = choice_row(case, Choice(error="LLMQuotaExhaustedError: local request budget reached", calls=1))
    assert network["outcome"] == quota["outcome"] == "not_evaluated" and not network["abort"] and quota["abort"]


# ----------------------------------------------------------------------------------------------- the whole flow: documents
def test_documents_correct_when_source_and_fact_match():
    row = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "total due", "project": "projectAlpha", "month": "2024-01"})),
                  Model(says("The total due is $4,500 USD [1].")))
    assert row["outcome"] == "correct" and row["cited_files"] == ["invoice.pdf"] and row["rank"] == 1 and row["facts_ok"] is True
    assert row["issues"] == [] and row["model_calls"] == 2


def test_documents_the_right_source_without_the_fact_is_reported_apart():
    row = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(says("Payment is due within 30 days [1].")))
    assert row["outcome"] == "right_source_missing_fact" and row["facts_ok"] is False


def test_documents_a_wrong_cited_source():
    row = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(says("The unit price is $8.50 [2].")))
    assert row["outcome"] == "wrong_source" and row["cited_files"] == ["other.pdf"]


def test_documents_a_refusal_is_split_by_whether_the_source_was_retrieved():
    row = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(says("NOT_FOUND")))
    assert row["outcome"] == "false_refusal" and row["rank"] == 1       # it WAS retrieved: the model refused anyway
    missing = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(says("NOT_FOUND")), chunks=[CHUNKS[1]])
    assert missing["outcome"] == "retrieval_miss" and missing["rank"] is None   # it was never retrieved: not the model's reading


def test_documents_tool_mistakes():
    assert ask_doc(INVOICE, Model(says("Probably 4,500.")), None)["outcome"] == "missed_tool"
    assert ask_doc(INVOICE, Model(calls(ev.SALES_TOOL, {})), None, chunks=[])["outcome"] in ("wrong_tool", "tool_error")
    assert ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q", "top_k": 99})), None)["outcome"] == "tool_error"


REJECT = Case(id=39, question="What was the total amount due for Project Beta in March 2024?", kind="near_topic_missing",
              should_reject=True, **DOC)
OFF_TOPIC = Case(id=45, question="What is the capital of France?", kind="off_topic", should_reject=True, domain="documents",
                 split="dev", expected_tool=None)


def test_documents_rejections_leaks_and_words_without_a_tool():
    refused = ask_doc(REJECT, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(says("NOT_FOUND")))
    leaked = ask_doc(REJECT, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(says("The total due is $4,500 USD [1].")))
    words = ask_doc(REJECT, Model(says("I do not have that information.")), None)
    off = ask_doc(OFF_TOPIC, Model(says("The capital of France is Paris.")), None)
    off_tool = ask_doc(OFF_TOPIC, Model(calls(ev.DOC_TOOL, {"query": "capital"})), Model(says("NOT_FOUND")))
    assert (refused["outcome"], leaked["outcome"], words["outcome"], off["outcome"], off_tool["outcome"]) == \
           ("correct_rejection", "leak", "no_tool_answer", "no_tool_answer", "correct_rejection")


def test_a_filter_that_matches_nothing_is_a_rejection_without_a_summary_request():
    a = assistant(Model(calls(ev.DOC_TOOL, {"query": "q", "project": "projectDelta"})), Model(), FakeRetriever([]))
    row = answer_row(REJECT, a.ask(REJECT.question))
    assert row["outcome"] == "correct_rejection" and row["model_calls"] == 1


def test_a_guardrail_failure_is_scored_but_a_network_failure_is_not():
    rejected = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})),
                       Model(says("Total 9,999 USD [1]."), says("Total 8,888 USD [1].")))
    assert rejected["outcome"] == "correct" and rejected["summary_source"] == "fallback"      # the verbatim passage holds $4,500
    network = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(LLMConnectionError("down")))
    quota = ask_doc(INVOICE, Model(calls(ev.DOC_TOOL, {"query": "q"})), Model(LLMQuotaExhaustedError("budget")))
    assert network["outcome"] == "not_evaluated" and not network["abort"]
    assert quota["outcome"] == "not_evaluated" and quota["abort"]


def test_an_unavailable_model_is_not_evaluated():
    row = ask_doc(INVOICE, Model(LLMQuotaExhaustedError("local request budget reached")), None)
    assert row["outcome"] == "not_evaluated" and row["abort"] is True


# ----------------------------------------------------------------------------------------------- sales
def sales_case(**kw):
    base = dict(id=1001, question="Which category has the highest revenue?", kind="sales_answerable", domain="sales", split="dev",
                expected_tool=ev.SALES_TOOL, expected_args={}, facts=("by_category.0.category", "by_category.0.revenue"))
    return Case(**{**base, **kw})


def test_the_expected_figures_come_from_the_sql(db):
    assert expected_facts(sales_case(), db) == ["Furniture", "400.0"]
    assert expected_facts(sales_case(facts=("by_category.-1.category",)), db) == ["Food"]
    assert expected_facts(sales_case(expected_args={"category": "Food"}, facts=("overall.total_revenue",)), db) == ["200.0"]
    assert expected_facts(sales_case(facts=()), db) == []
    assert resolve_path({"a": [{"b": 5}]}, "a.0.b") == 5


def ask_sales(case, chooser, summarizer, db_path):
    a = assistant(chooser, summarizer, None, db_path)
    return answer_row(case, a.ask(case.question), ev.expected_facts(case, db_path) if case.expect == "answer" else None)


def test_sales_correct_wrong_args_and_missing_fact(db):
    ok = ask_sales(sales_case(), Model(calls(ev.SALES_TOOL, {})), Model(says("Furniture leads with 400.0 in revenue.")), db)
    wrong_args = ask_sales(sales_case(), Model(calls(ev.SALES_TOOL, {"category": "Food"})), Model(says("Food has 200.0.")), db)
    missing = ask_sales(sales_case(), Model(calls(ev.SALES_TOOL, {})), Model(says("Food leads with 200.0 in revenue.")), db)
    assert (ok["outcome"], wrong_args["outcome"], missing["outcome"]) == ("correct", "wrong_args", "right_source_missing_fact")
    assert ask_sales(sales_case(), Model(says("Furniture, I guess.")), None, db)["outcome"] == "missed_tool"
    assert ask_sales(sales_case(), Model(calls(ev.DOC_TOOL, {"query": "q"})), None, db)["outcome"] in ("wrong_tool", "tool_error")


def test_sales_a_missing_category_must_end_as_a_tool_error(db):
    case = sales_case(id=1015, question="Revenue of Toys?", expected_args={"category": "Toys"}, facts=(), expect="tool_error", kind="sales_tool_error")
    assert ask_sales(case, Model(calls(ev.SALES_TOOL, {"category": "Toys"})), None, db)["outcome"] == "correct"


def test_sales_a_figure_that_is_not_in_the_facts_must_be_refused(db):
    case = sales_case(id=1016, question="What is the profit margin?", expected_args={}, facts=(), expect="not_found", kind="sales_not_found")
    assert ask_sales(case, Model(calls(ev.SALES_TOOL, {})), Model(says("NOT_FOUND")), db)["outcome"] == "correct"
    assert ask_sales(case, Model(calls(ev.SALES_TOOL, {})), Model(says("The margin is 400.0.")), db)["outcome"] == "leak"
    assert ask_sales(case, Model(says("I have no profit data.")), None, db)["outcome"] == "no_tool_answer"


# ----------------------------------------------------------------------------------------------- the run loop
def test_choose_mode_asks_one_request_per_question_and_runs_no_tool():
    cases = [INVOICE, OFF_TOPIC]
    model = Model(calls(ev.DOC_TOOL, {"query": "q"}), says("Paris."))
    rows, aborted = run_cases(assistant(model, None, FakeRetriever(CHUNKS)), cases, "choose", progress=None)
    assert [r["outcome"] for r in rows] == ["tool_ok", "tool_ok"] and aborted is None and model.requests == 2


def test_a_quota_stop_ends_the_run_and_keeps_what_was_done():
    cases = [INVOICE, REJECT, OFF_TOPIC]
    model = Model(calls(ev.DOC_TOOL, {"query": "q"}), LLMQuotaExhaustedError("local request budget reached"), says("never reached"))
    rows, aborted = run_cases(assistant(model, None, FakeRetriever(CHUNKS)), cases, "choose", progress=None)
    assert [r["outcome"] for r in rows] == ["tool_ok", "not_evaluated"] and "LLMQuotaExhaustedError" in aborted and model.requests == 2


def test_full_mode_runs_the_whole_flow(db):
    cases = [INVOICE, sales_case()]
    chooser = Model(calls(ev.DOC_TOOL, {"query": "q"}), calls(ev.SALES_TOOL, {}))
    summarizer = Model(says("The total due is $4,500 USD [1]."), says("Furniture leads with 400.0."))
    rows, aborted = run_cases(assistant(chooser, summarizer, FakeRetriever(CHUNKS), db), cases, "full", db, progress=None)
    assert [r["outcome"] for r in rows] == ["correct", "correct"] and aborted is None


def test_an_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        run_cases(assistant(Model()), [], "everything")


# ----------------------------------------------------------------------------------------------- the reply cache
def _config(**changes) -> LLMConfig:
    values = dict(base_dir=Path("."), provider="gemini", api_key="k", model="gemini-test", temperature=0.1, max_output_tokens=256,
                  timeout_seconds=7.0, deadline_seconds=100.0, max_retries=2, backoff_base_seconds=0.0, max_rpm=60,
                  log_path=Path("llm.log"), thinking_level="")
    values.update(changes)
    return LLMConfig(**values)


def _prompted(runnable):
    return ChatPromptTemplate.from_messages([("human", "{question}")]) | runnable


def test_a_reply_is_asked_once_then_read_from_disk(tmp_path):
    inner = Model(calls(ev.DOC_TOOL, {"query": "q", "project": "projectAlpha"}), says("second reply"))
    cache = ReplyCache(tmp_path / "cache.json", "fp1")
    chain = _prompted(cache.wrap(inner.runnable(), "choose"))
    first, again = chain.invoke({"question": "hello"}), chain.invoke({"question": "hello"})
    assert inner.requests == 1 and (cache.hits, cache.misses) == (1, 1)
    assert again.tool_calls[0]["name"] == ev.DOC_TOOL and again.tool_calls[0]["args"] == {"query": "q", "project": "projectAlpha"}
    assert again.usage_metadata["input_tokens"] == 500 and first.tool_calls[0]["args"] == again.tool_calls[0]["args"]
    reborn = ReplyCache(tmp_path / "cache.json", "fp1")                      # a new process reads the same file
    assert _prompted(reborn.wrap(Model().runnable(), "choose")).invoke({"question": "hello"}).tool_calls[0]["name"] == ev.DOC_TOOL


def test_role_text_and_fingerprint_all_change_the_key(tmp_path):
    inner = Model(says("a"), says("b"), says("c"), says("d"))
    cache = ReplyCache(tmp_path / "c.json", "fp1")
    _prompted(cache.wrap(inner.runnable(), "choose")).invoke({"question": "hello"})
    _prompted(cache.wrap(inner.runnable(), "summarize")).invoke({"question": "hello"})          # another role
    _prompted(cache.wrap(inner.runnable(), "choose")).invoke({"question": "hello!"})            # another text
    other = ReplyCache(tmp_path / "c.json", "fp2")                                               # another model/prompt
    _prompted(other.wrap(inner.runnable(), "choose")).invoke({"question": "hello"})
    assert inner.requests == 4


def test_a_failed_request_is_never_cached(tmp_path):
    inner = Model(LLMConnectionError("down"), says("ok"))
    cache = ReplyCache(tmp_path / "c.json", "fp")
    chain = _prompted(cache.wrap(inner.runnable(), "choose"))
    with pytest.raises(LLMConnectionError):
        chain.invoke({"question": "q"})
    assert chain.invoke({"question": "q"}).text == "ok" and inner.requests == 2


def test_an_unreadable_cache_file_starts_empty(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text("{ not json", encoding="utf-8")
    assert ReplyCache(path, "fp")._data == {} and "unreadable" in capsys.readouterr().out


def test_the_fingerprint_follows_the_settings_and_every_prompt(monkeypatch):
    base = fingerprint(_config())
    assert base == fingerprint(_config()) and "gemini-test" in base
    assert fingerprint(_config(model="other")) != base and fingerprint(_config(temperature=0.5)) != base
    monkeypatch.setattr(ev, "SYSTEM_DOCUMENTS", ev.SYSTEM_DOCUMENTS + " Be brief.")
    assert fingerprint(_config()) != base


def test_a_cache_hit_costs_no_request_and_no_budget(tmp_path):
    config = _config()
    layer = LangChainRequestLayer(config, limiter=SimpleNamespace(wait=lambda: None), counter=RequestCounter(limit=1),
                                  sleep=lambda s: None, clock=lambda: 0.0, jitter=lambda: 0.5)
    inner = Model(calls(ev.DOC_TOOL, {"query": "q"}))
    cache = ReplyCache(tmp_path / "c.json", fingerprint(config))
    chooser = cache.wrap(layer.guard(inner.runnable()), "choose")
    a = Assistant(chooser, None, build_tools(ToolRuntime()), ToolLog(), layer=layer)
    first, second = a.choose("What is due?"), a.choose("What is due?")        # budget is 1: the second must not need it
    assert first.tool == second.tool == ev.DOC_TOOL and layer.requests == 1 and cache.hits == 1


# ----------------------------------------------------------------------------------------------- the report
def test_the_report_names_what_to_read(db):
    cases = [INVOICE, REJECT, OFF_TOPIC, sales_case()]
    chooser = Model(calls(ev.DOC_TOOL, {"query": "q", "project": "projectAlpha", "month": "2024-01"}), calls(ev.DOC_TOOL, {"query": "q"}),
                    says("Paris."), calls(ev.SALES_TOOL, {}))
    summarizer = Model(says("The total due is $4,500 USD [1]."), says("The total due is $4,500 USD [1]."), says("Furniture leads with 400.0."))
    rows, aborted = run_cases(assistant(chooser, summarizer, FakeRetriever(CHUNKS), db), cases, "full", db, progress=None)
    meta = {"mode": "full", "model": "gemini-test", "splits": ["dev"], "requests_sent": 7, "max_requests": 15, "cache_hits": 0,
            "cache_misses": 7, "aborted": None}
    text = format_report(rows, meta, excluded=[{"id": 8}, {"id": 36}])
    for needle in ("WEEK 10 EVALUATION", "TOOL CHOICE", "OUTCOMES (whole flow)", "READ THESE BY A PERSON", "documents/off_topic",
                   "sales/sales_answerable", "ids [8, 36]", "choice request:", "summary request:", "leak", "no_tool_answer"):
        assert needle in text, needle
    stopped = format_report(rows[:1], {**meta, "aborted": "LLMQuotaExhaustedError: budget"}, None)
    assert "RUN STOPPED" in stopped


# ----------------------------------------------------------------------------------------------- the real question files
@pytest.mark.skipif(not (ROOT / "data" / "eval" / "week8" / "eval_questions.json").exists(), reason="the saved question sets are not here")
def test_the_saved_sets_load_with_the_default_splits():
    cases, excluded = load_cases(ROOT / "data" / "eval", "all", {"dev", "test", "calib"})
    assert {c.domain for c in cases} == {"documents", "sales"} and all(q.get("filters") for q in excluded)
    assert all(c.split != "holdout" for c in cases) and all(c.id >= 1001 for c in cases if c.domain == "sales")
    assert all(c.expected_tool is None for c in cases if c.kind == "off_topic")
    holdout, _ = load_cases(ROOT / "data" / "eval", "all", {"holdout"}, domain="documents")
    assert holdout and all(c.split == "holdout" for c in holdout)
    with pytest.raises(ValueError, match="unknown question ids"):
        load_cases(ROOT / "data" / "eval", "all", {"dev"}, ids=[999999])
