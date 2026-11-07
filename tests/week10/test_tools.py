import sqlite3
from types import SimpleNamespace

import pytest

from src.llm.week9.tool_specs import TOOLS, ToolRuntime
from src.llm.week10.tools import build_tools, run_validated
from src.rag.week10.langchain_retriever import RagRetriever


def _chunk(n: int, text: str, score: float = 0.8123456) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=f"c{n}", score=score, text=text, section_header="Totals", original_filename=f"file{n}.md",
                           document_id="d", chunk_index=n, document_type="invoice", month="2024-01", project="projectAlpha",
                           service="hosting")


class FakeRetriever:
    """Same call as src.rag.week8.retriever.Retriever.retrieve; records what it was asked."""

    def __init__(self, chunks=None, status="retrieved", error: Exception | None = None):
        self.chunks, self.status, self.error, self.calls = chunks or [], status, error, []

    def retrieve(self, question, top_k=None, filters=None):
        self.calls.append({"question": question, "top_k": top_k, "filters": filters})
        if self.error:
            raise self.error
        return SimpleNamespace(status=self.status, chunks=self.chunks, applied_filters=filters or {}, stale_chunks=0)


@pytest.fixture
def retail_db(tmp_path):
    path = tmp_path / "retail.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sales_clean (transaction_date TEXT, category TEXT, payment_method TEXT, location TEXT, "
                "total_spent REAL, quantity INTEGER)")
    rows = [("2024-01-01", "Furniture", "Cash", "Online", 100.0, 1), ("2024-01-31", "Furniture", "Card", "In-store", 300.0, 3),
            ("2024-02-01", "Food", "Card", "Online", 50.0, 5), ("2024-02-29", "Food", "Cash", "In-store", 150.0, 2)]
    con.executemany("INSERT INTO sales_clean VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _by_name(tools):
    return {t.name: t for t in tools}


# ----------------------------------------------------------------------------------------------- the adapter
def test_the_adapter_returns_documents_and_passes_top_k_and_filters():
    fake = FakeRetriever([_chunk(1, "Total due: 1,250.00"), _chunk(2, "Late fee 5%")])
    docs = RagRetriever(retriever=fake).invoke("total due", top_k=2, filters={"project": "projectAlpha"})
    assert [d.page_content for d in docs] == ["Total due: 1,250.00", "Late fee 5%"]
    assert docs[0].metadata["source"] == "file1.md" and docs[0].metadata["score"] == 0.8123456
    assert docs[0].id == "c1"
    assert fake.calls == [{"question": "total due", "top_k": 2, "filters": {"project": "projectAlpha"}}]


def test_the_adapter_without_arguments_leaves_the_defaults_to_week8():
    fake = FakeRetriever([_chunk(1, "x")])
    RagRetriever(retriever=fake).invoke("question")
    assert fake.calls[0]["top_k"] is None and fake.calls[0]["filters"] is None


# ----------------------------------------------------------------------------------------------- the tools
def test_the_tools_carry_the_week9_names_descriptions_and_schemas():
    tools = build_tools(ToolRuntime())
    assert [t.name for t in tools] == [s.name for s in TOOLS]
    assert [t.description for t in tools] == [s.description for s in TOOLS]
    assert [set(t.args) for t in tools] == [set(s.parameters["properties"]) for s in TOOLS]


def test_search_documents_returns_the_week9_passage_shape():
    fake = FakeRetriever([_chunk(1, "Total due: 1,250.00"), _chunk(2, "Late fee 5%")])
    result = _by_name(build_tools(ToolRuntime(retriever=fake)))["search_documents"].invoke(
        {"query": "total due", "project": "projectAlpha", "month": "2024-01"})
    assert result == {"passages": [{"id": 1, "source": "file1.md", "score": 0.8123, "text": "Total due: 1,250.00"},
                                   {"id": 2, "source": "file2.md", "score": 0.8123, "text": "Late fee 5%"}]}
    assert fake.calls[0]["top_k"] == 3                                               # the documented default
    assert fake.calls[0]["filters"] == {"project": "projectAlpha", "month": "2024-01"}   # only what the model gave


def test_search_documents_without_filters_sends_none():
    fake = FakeRetriever([_chunk(1, "x")])
    _by_name(build_tools(ToolRuntime(retriever=fake)))["search_documents"].invoke({"query": "q", "top_k": 5})
    assert fake.calls[0]["filters"] is None and fake.calls[0]["top_k"] == 5


def test_search_documents_with_a_filter_that_matches_nothing():
    fake = FakeRetriever([])
    result = _by_name(build_tools(ToolRuntime(retriever=fake)))["search_documents"].invoke({"query": "q", "project": "projectZ"})
    assert result == {"passages": [], "note": "no_documents_match_filter"}


def test_search_documents_never_raises():
    boom = _by_name(build_tools(ToolRuntime(retriever=FakeRetriever(error=RuntimeError("index is empty")))))["search_documents"]
    assert "RuntimeError: index is empty" in boom.invoke({"query": "q"})["error"]
    none = _by_name(build_tools(ToolRuntime()))["search_documents"].invoke({"query": "q"})
    assert "no retriever" in none["error"]


def test_get_sales_summary_reads_the_cleaned_data(retail_db):
    tool = _by_name(build_tools(ToolRuntime(retail_db_path=retail_db)))["get_sales_summary"]
    everything = tool.invoke({})
    assert everything["overall"]["total_revenue"] == 600.0 and everything["overall"]["transactions"] == 4
    furniture = tool.invoke({"category": "Furniture", "start": "2024-01", "end": "2024-01"})
    assert furniture["overall"]["total_revenue"] == 400.0
    assert [r["category"] for r in everything["by_category"]] == ["Furniture", "Food"]


def test_get_sales_summary_failures_are_messages(retail_db, tmp_path):
    tool = _by_name(build_tools(ToolRuntime(retail_db_path=retail_db)))["get_sales_summary"]
    assert "Unknown category" in tool.invoke({"category": "Toys"})["error"]
    missing = _by_name(build_tools(ToolRuntime(retail_db_path=tmp_path / "nope.db")))["get_sales_summary"].invoke({})
    assert "Database not found" in missing["error"]
    assert "no retail database" in _by_name(build_tools(ToolRuntime()))["get_sales_summary"].invoke({})["error"]


# ----------------------------------------------------------------------------------------------- running the model's call
def test_run_validated_runs_a_valid_call(retail_db):
    tools = build_tools(ToolRuntime(retail_db_path=retail_db))
    assert run_validated(tools, "get_sales_summary", {"category": " Food "})["overall"]["total_revenue"] == 200.0  # cleaned


def test_run_validated_cleans_a_float_top_k():
    fake = FakeRetriever([_chunk(1, "x")])
    run_validated(build_tools(ToolRuntime(retriever=fake)), "search_documents", {"query": "q", "top_k": 4.0})
    assert fake.calls[0]["top_k"] == 4 and isinstance(fake.calls[0]["top_k"], int)


@pytest.mark.parametrize("name, args", [
    ("delete_everything", {}),                                         # unknown tool
    ("search_documents", {}),                                          # missing required argument
    ("search_documents", {"query": "q", "colour": "red"}),             # unknown argument
    ("search_documents", {"query": "q", "top_k": 99}),                 # out of range
    ("search_documents", {"query": "q", "document_type": "memo"}),     # not in the enum
    ("get_sales_summary", {"start": "January"}),                       # wrong month format
    (None, None),                                                      # nothing chosen
])
def test_run_validated_rejects_a_bad_call_without_running_anything(name, args):
    fake = FakeRetriever([_chunk(1, "x")])
    result = run_validated(build_tools(ToolRuntime(retriever=fake, retail_db_path="unused.db")), name, args)
    assert result["rejected"] is True and result["error"].startswith("invalid tool call")
    assert fake.calls == []


def test_run_validated_rejects_a_known_tool_that_is_not_in_the_list(retail_db):
    only_sales = [t for t in build_tools(ToolRuntime(retail_db_path=retail_db)) if t.name == "get_sales_summary"]
    result = run_validated(only_sales, "search_documents", {"query": "q"})
    assert result["rejected"] is True and "not available" in result["error"]
