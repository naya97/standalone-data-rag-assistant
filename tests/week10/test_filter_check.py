from types import SimpleNamespace

import pytest

from src.llm.week10.evaluation import Case
from src.llm.week10.filter_check import check_rows, format_check, repair_args, select_rows


@pytest.mark.parametrize("given, expected", [
    ("Project Beta", "projectBeta"), ("project beta", "projectBeta"), ("Beta", "projectBeta"), ("project_alpha", "projectAlpha"),
    ("projectBeta", "projectBeta"), ("Project  Delta", "projectDelta"), ("PROJECT Gamma", "projectGamma")])
def test_the_project_is_written_like_the_stored_form(given, expected):
    assert repair_args("Project Beta budget?", {"project": given, "query": "q"})[0]["project"] == expected


def test_a_project_without_a_name_is_dropped_and_a_correct_one_is_left_alone():
    fixed, notes = repair_args("q", {"project": "Project"})
    assert "project" not in fixed and notes
    assert repair_args("q", {"project": "projectBeta"}) == ({"project": "projectBeta"}, [])


def test_a_month_is_dropped_only_when_the_question_does_not_support_its_year():
    q = "Which vulnerabilities were found in Project Beta's March security scan?"
    fixed, notes = repair_args(q, {"project": "projectBeta", "month": "2025-03", "query": "x"})
    assert "month" not in fixed and fixed["project"] == "projectBeta" and "does not state the year" in notes[0]
    kept, none = repair_args("What was the total in March 2024?", {"month": "2024-03"})
    assert kept == {"month": "2024-03"} and none == []


def _row(id, kind, outcome, args, issues=(), tool="search_documents", domain="documents"):
    return {"id": id, "kind": kind, "question": f"question {id}", "outcome": outcome, "args": args, "issues": list(issues), "tool": tool, "domain": domain}


def test_only_rows_worth_checking_are_selected():
    rows = [_row(1, "answerable", "correct", {"query": "q"}),
            _row(2, "answerable", "retrieval_miss", {"query": "q", "project": "projectBeta"}),
            _row(3, "near_topic_missing", "correct_rejection", {"query": "q", "project": "Project Beta"}, ["format: x"]),
            _row(4, "answerable", "false_refusal", {"query": "q"}, tool="search_documents"),
            _row(5, "off_topic", "no_tool_answer", {}, tool=None),
            _row(1001, "sales_answerable", "false_refusal", {}, tool="get_sales_summary", domain="sales")]
    assert [r["id"] for r in select_rows(rows)] == [2, 3, 4]


class ExactRetriever:
    """Filters must equal the stored values, like the Week 8 retriever; only projectBeta's report exists."""

    def retrieve(self, question, top_k=None, filters=None):
        chunk = SimpleNamespace(original_filename="report_2024-03_projectBeta_security.txt")
        ok = not filters or (filters.get("project") in (None, "projectBeta") and filters.get("month") in (None, "2024-03"))
        return SimpleNamespace(status="retrieved" if ok else "no_documents_match_filter", chunks=[chunk] if ok else [])


def test_the_table_separates_a_filter_that_kills_retrieval_from_everything_else():
    cases = {2: Case(id=2, question="question 2", split="dev", kind="answerable", domain="documents", expected_tool="search_documents",
                     expected_sources=("report_2024-03_projectBeta_security.txt",)),
             9: Case(id=9, question="question 9", split="dev", kind="near_topic_missing", domain="documents", expected_tool="search_documents",
                     should_reject=True)}
    rows = [_row(2, "answerable", "retrieval_miss", {"query": "q", "project": "Project Beta"}, ["format: project"]),
            _row(9, "near_topic_missing", "correct_rejection", {"query": "q", "project": "Project Beta"}, ["format: project"])]
    results = check_rows(rows, cases, ExactRetriever(), lambda name, expected: name in expected)
    by_id = {r["id"]: r for r in results}
    assert by_id[2]["as_chosen"]["n"] == 0 and by_id[2]["as_chosen"]["rank"] is None and by_id[2]["as_chosen"]["status"] == "no_documents_match_filter"
    assert by_id[2]["repaired"]["rank"] == 1 and by_id[2]["no_filters"]["rank"] == 1
    assert by_id[9]["should_reject"] and by_id[9]["as_chosen"]["n"] == 0 and by_id[9]["repaired"]["n"] == 1
    text = format_check(results)
    assert "as chosen 0   repaired 1   no filters 1" in text
    assert "filter matched nothing" in text and "[9]" in text and "project 'Project Beta' -> 'projectBeta'" in text


def test_a_retrieval_error_shows_in_the_table_instead_of_stopping_it():
    class Broken:
        def retrieve(self, *a, **k):
            raise RuntimeError("The vector index is empty")
    results = check_rows([_row(2, "answerable", "retrieval_miss", {"query": "q"})], {}, Broken(), lambda n, e: False)
    assert results[0]["as_chosen"]["error"].startswith("RuntimeError") and "error" in format_check(results)


# ----------------------------------------------------------------------------------------------- the fact, not only the file
class ChunkRetriever:
    """Returns the file of the expected source but only the chunk that does NOT hold the fact (like Q66: the chunk ends before 'Ribbon')."""

    def __init__(self, texts):
        self.texts = texts

    def retrieve(self, question, top_k=None, filters=None):
        chunks = [SimpleNamespace(original_filename="task.pdf", text=t) for t in self.texts[: top_k or 3]]
        return SimpleNamespace(status="retrieved", chunks=chunks)


def _contains(text, facts):
    return all(f.lower() in text.lower() for f in facts), "exact"


def test_a_file_can_be_rank_1_while_the_fact_is_not_in_the_retrieved_text():
    case = Case(id=66, question="question 66", split="dev", kind="answerable", domain="documents", expected_tool="search_documents",
                expected_sources=("task.pdf",), answer_contains=("Ribbon",))
    rows = [_row(66, "answerable", "false_refusal", {"query": "load balancing"})]
    heading_only = ["7. Load Balancing:", "unrelated", "unrelated"]
    results = check_rows(rows, {66: case}, ChunkRetriever(heading_only + ["Explain the integration of load balancing (such as Ribbon)"]),
                         lambda n, e: n in e, top_k=3, check_facts=_contains)
    assert results[0]["as_chosen"]["rank"] == 1 and results[0]["as_chosen"]["fact"] is False
    wider = check_rows(rows, {66: case}, ChunkRetriever(heading_only + ["Explain the integration of load balancing (such as Ribbon)"]),
                       lambda n, e: n in e, top_k=4, check_facts=_contains)
    assert wider[0]["repaired"]["fact"] is True                      # a larger top_k brings the fact: that is the evidence for it
    text = format_check(results)
    assert "fact=N" in text and "NOT in the text: [66]" in text


def test_a_wrong_answer_with_the_right_source_is_also_re_checked_and_all_answerable_takes_every_question():
    rows = [_row(1, "answerable", "right_source_missing_fact", {"query": "q"}), _row(2, "answerable", "correct", {"query": "q"})]
    assert [r["id"] for r in select_rows(rows)] == [1]
    assert [r["id"] for r in select_rows(rows, all_answerable=True)] == [1, 2]


# ----------------------------------------------------------------------------------------------- one retrieval, the fact at every k
def test_one_retrieval_gives_the_recall_of_the_fact_at_every_k_and_names_the_questions_a_larger_k_rescues():
    case = Case(id=66, question="question 66", split="dev", kind="answerable", domain="documents", expected_tool="search_documents",
                expected_sources=("task.pdf",), answer_contains=("Ribbon",))
    texts = ["7. Load Balancing:", "unrelated", "unrelated", "unrelated", "choose a mechanism (such as Ribbon)"]
    rows = [_row(66, "answerable", "false_refusal", {"query": "load balancing"})]
    results = check_rows(rows, {66: case}, ChunkRetriever(texts), lambda n, e: n in e, check_facts=_contains, ks=(3, 5, 8))
    by_k = results[0]["repaired"]["by_k"]
    assert [by_k[k]["fact"] for k in (3, 5, 8)] == [False, True, True]
    assert [by_k[k]["source"] for k in (3, 5, 8)] == [True, True, True]          # the FILE is there at every k: only the fact tells them apart
    text = format_check(results)
    assert "k=3: 0   k=5: 1   k=8: 1" in text and "found only from k=5 (repaired): [66]" in text and "not found even at k=8 (repaired): []" in text
    assert "expected source is in the top 8" in text and "retrieved text at k=8" in text      # the label follows the k that was measured


def test_the_fact_is_only_looked_for_in_the_expected_source_not_in_other_documents():
    class TwoFiles:
        def retrieve(self, question, top_k=None, filters=None):
            other = SimpleNamespace(original_filename="other.pdf", text="Build it in 15 hours")
            wanted = SimpleNamespace(original_filename="task.pdf", text="Total 70 hours")
            return SimpleNamespace(status="retrieved", chunks=[other, wanted])

    case = Case(id=1, question="q", split="dev", kind="answerable", domain="documents", expected_tool="search_documents",
                expected_sources=("task.pdf",), answer_contains=("15 hours",))
    results = check_rows([_row(1, "answerable", "right_source_missing_fact", {"query": "q"})], {1: case}, TwoFiles(),
                         lambda n, e: n in e, check_facts=_contains)
    assert results[0]["repaired"]["fact"] is False        # '15 hours' is in other.pdf only
