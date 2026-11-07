import copy
import sqlite3
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.llm.week9.tool_specs import ToolRuntime
from src.llm.week10.arguments import MetadataCatalog, argument_issues, canonical_project, clean_search_arguments
from src.llm.week10.assistant import Assistant
from src.llm.week10.tool_logging import ToolLog
from src.llm.week10.tools import build_tools

# what the index of this project holds: two projects, three months of ONE year
CATALOG = MetadataCatalog(frozenset({"projectAlpha", "projectBeta"}), frozenset({"2024-01", "2024-02", "2024-03"}))
TWO_YEARS = MetadataCatalog(frozenset({"projectAlpha"}), frozenset({"2024-03", "2025-03", "2024-01"}))

# project
@pytest.mark.parametrize("given, expected", [
    ("Project Beta", "projectBeta"), ("project beta", "projectBeta"), ("Beta", "projectBeta"), ("project_alpha", "projectAlpha"),
    ("projectBeta", "projectBeta"), ("Project  Delta", "projectDelta"), ("PROJECT Gamma", "projectGamma"), ("project-alpha", "projectAlpha")])
def test_every_spelling_of_a_project_becomes_the_stored_form(given, expected):
    assert canonical_project(given) == expected


def test_a_project_without_a_name_has_no_canonical_form():
    assert canonical_project("Project") is None and canonical_project("  ") is None


def test_a_known_project_is_rewritten_and_the_change_is_reported():
    out, notes = clean_search_arguments("Who approves the Project Beta budget?", {"project": "Project Beta", "query": "approval"}, CATALOG)
    assert out == {"project": "projectBeta", "query": "approval"} and notes == ["project 'Project Beta' -> 'projectBeta'"]


def test_an_unknown_project_is_never_dropped_and_never_turned_into_another_one():
    out, notes = clean_search_arguments("What is the password policy for Project Delta?", {"project": "Project Delta"}, CATALOG)
    assert out == {"project": "projectDelta"}                      # stays: it matches nothing, so the answer is "not found"
    assert any("not in the index" in n for n in notes)
    # without a catalog the same filter is also kept (dropping it would search the OTHER projects)
    assert clean_search_arguments("q", {"project": "Project Delta"})[0] == {"project": "projectDelta"}


def test_a_correct_project_is_left_alone_silently():
    assert clean_search_arguments("q", {"project": "projectAlpha"}, CATALOG) == ({"project": "projectAlpha"}, [])


def test_a_project_with_no_name_is_dropped():
    out, notes = clean_search_arguments("q", {"project": "Project"}, CATALOG)
    assert "project" not in out and "dropped" in notes[0]


# month and year
MARCH = "Which vulnerabilities were found in Project Beta's March security scan?"


def test_a_month_without_a_year_is_never_sent_with_a_guessed_year_and_no_catalog_means_dropped():
    out, notes = clean_search_arguments(MARCH, {"project": "projectBeta", "month": "2025-03", "query": "x"})
    assert out == {"project": "projectBeta", "query": "x"} and "does not state the year" in notes[0]


def test_with_a_catalog_the_only_matching_month_is_used_and_the_year_comes_from_the_data():
    out, notes = clean_search_arguments(MARCH, {"project": "projectBeta", "month": "2025-03"}, CATALOG)
    assert out["month"] == "2024-03" and "only such month in the index" in notes[0]      # not the model's 2025


def test_with_several_candidate_years_the_month_is_dropped_and_the_question_is_called_ambiguous():
    out, notes = clean_search_arguments("What was the cost in March?", {"project": "projectAlpha", "month": "2024-03"}, TWO_YEARS)
    assert "month" not in out and "ambiguous" in notes[0] and "2024-03" in notes[0] and "2025-03" in notes[0]


def test_when_no_indexed_document_is_from_that_month_the_filter_is_kept_so_nothing_matches():
    out, notes = clean_search_arguments("What was the cost in July?", {"month": "2024-07"}, CATALOG)
    assert out == {"month": "2024-07"} and "nothing will match" in notes[0]


@pytest.mark.parametrize("question, model_month", [
    ("What was the total due in March 2024?", "2024-03"), ("ما إجمالي الفاتورة في مارس 2024؟", "2024-03"), ("the invoice of 2024-03", "2024-03")])
def test_an_explicit_year_is_never_removed(question, model_month):
    assert clean_search_arguments(question, {"month": model_month}, CATALOG) == ({"month": model_month}, [])
    assert clean_search_arguments(question, {"month": model_month}) == ({"month": model_month}, [])


def test_when_the_question_states_another_year_the_question_wins():
    out, notes = clean_search_arguments("What was the cost in March 2025?", {"month": "2024-03"}, CATALOG)
    assert out == {"month": "2025-03"} and "stated in the question" in notes[0]


def test_a_month_the_question_never_mentions_is_dropped_as_invented():
    out, notes = clean_search_arguments("What is the total amount due?", {"month": "2024-03", "query": "q"}, CATALOG)
    assert out == {"query": "q"} and "does not mention that month" in notes[0]


def test_arabic_month_names_count_as_named():
    out, _ = clean_search_arguments("كم كانت تكلفة المشروع في يناير 2024؟", {"month": "2024-01"}, CATALOG)
    assert out == {"month": "2024-01"}
    out, notes = clean_search_arguments("كم كانت تكلفة مشروع Alpha في يناير؟", {"month": "2025-01"}, CATALOG)
    assert out == {"month": "2024-01"} and notes                    # unique January in the index: its stored value


def test_the_arguments_that_were_not_the_cleanings_business_are_untouched_and_the_input_is_never_modified():
    args = {"query": "late fee", "document_type": "invoice", "top_k": 5, "project": "Project Alpha"}
    before = copy.deepcopy(args)
    out, _ = clean_search_arguments("Project Alpha invoice late fee?", args, CATALOG)
    assert args == before and out["document_type"] == "invoice" and out["top_k"] == 5 and out["query"] == "late fee"


def test_after_the_cleaning_without_a_catalog_the_argument_check_has_nothing_left_to_report():
    for question, args in [(MARCH, {"project": "Project Beta", "month": "2025-03"}), ("Project Beta pay days?", {"project": "Beta"})]:
        cleaned, _ = clean_search_arguments(question, args)
        assert [i for i in argument_issues(question, cleaned) if not i.startswith("missed:")] == []


def test_with_a_catalog_the_text_only_check_still_calls_the_data_derived_month_assumed_and_that_is_expected():
    # the check reads the QUESTION only, so it cannot know that the year came from the index and not from the model
    cleaned, _ = clean_search_arguments(MARCH, {"project": "Project Beta", "month": "2025-03"}, CATALOG)
    assert cleaned["month"] == "2024-03"
    assert [i.split(":")[0] for i in argument_issues(MARCH, cleaned)] == ["assumed"]


def test_a_catalog_is_built_from_stored_rows():
    rows = [{"project": "projectAlpha", "month": "2024-01"}, {"project": "projectBeta", "month": "2024-03"}, {"project": None, "month": None}]
    catalog = MetadataCatalog.from_rows(rows)
    assert catalog.projects == {"projectAlpha", "projectBeta"} and catalog.months == {"2024-01", "2024-03"}
    objects = [SimpleNamespace(project="projectAlpha", month="2024-02")]
    assert MetadataCatalog.from_rows(objects).months == {"2024-02"}


# in the assistant
class ExactRetriever:
    """Like the Week 8 retriever: an exact filter that matches nothing returns no passage."""

    def __init__(self):
        self.calls = []

    def retrieve(self, question, top_k=None, filters=None):
        self.calls.append(filters)
        ok = not filters or (filters.get("project") in (None, "projectBeta") and filters.get("month") in (None, "2024-03"))
        chunk = SimpleNamespace(chunk_id="c", score=0.9, text="Two low-level vulnerabilities were found. The certificates were rotated on March 6th.",
                                section_header=None, original_filename="report_2024-03_projectBeta_security.txt", document_id="d", chunk_index=0,
                                document_type="report", month="2024-03", project="projectBeta", service="Security")
        return SimpleNamespace(status="retrieved" if ok else "no_documents_match_filter", chunks=[chunk] if ok else [], applied_filters=filters or {}, stale_chunks=0)


def _assistant(retriever, cleaner, chosen_args):
    choose = AIMessage(content="", tool_calls=[{"name": "search_documents", "args": chosen_args, "id": "1", "type": "tool_call"}])
    summary = AIMessage(content="The certificates were rotated on March 6th [1].")
    return Assistant(RunnableLambda(lambda v: choose), RunnableLambda(lambda v: summary), build_tools(ToolRuntime(retriever=retriever)), ToolLog(),
                     cleaner=cleaner)


MODEL_ARGS = {"project": "Project Beta", "month": "2025-03", "query": "vulnerabilities security scan"}


def test_without_cleaning_the_models_filters_find_nothing_which_is_the_measured_failure():
    retriever = ExactRetriever()
    answer = _assistant(retriever, None, MODEL_ARGS).ask(MARCH)
    assert retriever.calls == [{"project": "Project Beta", "month": "2025-03"}] and answer.summary.status == "not_found"
    assert answer.model_requests == 1                                # the model was never asked to read anything


def test_with_cleaning_the_same_question_reaches_the_document_and_is_answered():
    retriever = ExactRetriever()
    answer = _assistant(retriever, lambda q, a: clean_search_arguments(q, a), MODEL_ARGS).ask(MARCH)
    assert retriever.calls == [{"project": "projectBeta"}]
    assert answer.summary.status == "answered" and answer.summary.citations == [1] and answer.model_requests == 2
    assert answer.args == MODEL_ARGS                                  # what the MODEL chose is kept as it was
    assert answer.run_args == {"project": "projectBeta", "query": "vulnerabilities security scan"}
    assert len(answer.clean_notes) == 2


def test_with_a_catalog_the_month_is_resolved_from_the_data_not_dropped():
    retriever = ExactRetriever()
    answer = _assistant(retriever, lambda q, a: clean_search_arguments(q, a, CATALOG), MODEL_ARGS).ask(MARCH)
    assert retriever.calls == [{"project": "projectBeta", "month": "2024-03"}] and answer.summary.status == "answered"


def test_every_change_is_written_to_the_log(tmp_path):
    import logging
    from src.llm.week10.tool_logging import LOGGER_NAME, setup_tool_log
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    path = tmp_path / "tool_calls.log"
    setup_tool_log(path)
    _assistant(ExactRetriever(), lambda q, a: clean_search_arguments(q, a), MODEL_ARGS).ask(MARCH)
    for handler in list(logger.handlers):
        handler.close(); logger.removeHandler(handler)
    text = path.read_text(encoding="utf-8")
    assert "CLEAN" in text and "project 'Project Beta' -> 'projectBeta'" in text and "does not state the year" in text


def test_a_direct_tool_run_and_the_sales_tool_are_never_cleaned(tmp_path):
    retriever = ExactRetriever()
    assistant = _assistant(retriever, lambda q, a: clean_search_arguments(q, a), MODEL_ARGS)
    answer = assistant.run_tool("search_documents", {"query": "q", "project": "Project Beta"})   # the user's own arguments
    assert retriever.calls == [{"project": "Project Beta"}] and answer.clean_notes == []
    db = tmp_path / "r.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE sales_clean (transaction_date TEXT, category TEXT, payment_method TEXT, location TEXT, total_spent REAL, quantity INTEGER)")
    con.execute("INSERT INTO sales_clean VALUES ('2024-01-05','Food','Cash','Online',200.0,2)")
    con.commit(); con.close()
    choose = AIMessage(content="", tool_calls=[{"name": "get_sales_summary", "args": {"category": "Food"}, "id": "1", "type": "tool_call"}])
    sales = Assistant(RunnableLambda(lambda v: choose), RunnableLambda(lambda v: AIMessage(content="Food made 200.0.")),
                      build_tools(ToolRuntime(retail_db_path=db)), ToolLog(), cleaner=lambda q, a: (_ for _ in ()).throw(AssertionError("must not run")))
    assert sales.ask("Food revenue?").run_args == {"category": "Food"}
