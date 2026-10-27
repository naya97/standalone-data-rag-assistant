"""Guardrails: JSON parsing, schema validation, number grounding, citation checks. No model involved."""

import pytest

from src.llm.week9.client import LLMInvalidOutputError
from src.llm.week9.guardrails import (
    check_rag_answer, extract_numbers, parse_json, ungrounded_numbers, validate_output,
)
from src.llm.week9.schemas import RagAnswer

GOOD = '{"status": "answered", "answer": "The total is 1250 USD.", "citations": [1], "language": "en"}'
TEXTS = ["Total amount due: 1,250.00 USD. Payment due within 30 days.", "Invoice date: 2024-02-10."]


# ----------------------------------------------------------------------------------------------- JSON
def test_parse_plain_and_fenced_json():
    assert parse_json(GOOD)["status"] == "answered"
    assert parse_json(f"```json\n{GOOD}\n```")["citations"] == [1]
    assert parse_json(f"```\n{GOOD}\n```")["language"] == "en"


@pytest.mark.parametrize("text", ["", "not json", '{"status": "answered"', "[1, 2]", '"just a string"'])
def test_invalid_json_is_rejected(text):
    with pytest.raises(LLMInvalidOutputError):
        parse_json(text)


def test_schema_mismatch_is_rejected_with_a_readable_reason():
    with pytest.raises(LLMInvalidOutputError, match="schema"):
        validate_output('{"status": "maybe", "answer": "x", "citations": [], "language": "en"}', RagAnswer)
    with pytest.raises(LLMInvalidOutputError, match="citations"):
        validate_output('{"status": "answered", "answer": "x", "language": "en"}', RagAnswer)


def test_valid_output_becomes_a_schema_object():
    answer = validate_output(GOOD, RagAnswer)
    assert answer.status == "answered" and answer.citations == [1]


# ----------------------------------------------------------------------------------------------- numbers
@pytest.mark.parametrize("a, b", [("1,250.00", "1250"), ("1250", "1,250"), ("12,5", "12.5"), ("007", "7"),
                                  ("٣٠", "30"), ("1250.50", "1,250.5")])
def test_numbers_are_compared_by_value_not_spelling(a, b):
    assert extract_numbers(a) == extract_numbers(b)


def test_dates_split_into_their_numbers():
    assert extract_numbers("2024-02-10") == {"2024", "2", "10"}


def test_grounded_and_ungrounded_numbers():
    assert ungrounded_numbers("The total is 1250 USD, due in 30 days.", TEXTS) == []
    assert ungrounded_numbers("The total is 1300 USD.", TEXTS) == ["1300"]


def test_known_weakness_small_numbers_can_collide_with_date_parts():
    # "2 invoices" is invented, but 2 also appears in "2024-02-10", so it is not flagged. Larger numbers are safe.
    assert ungrounded_numbers("Total: 1250 USD, 2 invoices.", TEXTS) == []


def test_a_computed_number_is_flagged():
    assert ungrounded_numbers("Two items of 600 and 650.", TEXTS) == ["600", "650"]


# ----------------------------------------------------------------------------------------------- RAG answer
def _answer(**kw):
    base = {"status": "answered", "answer": "The total is 1250 USD.", "citations": [1], "language": "en"}
    return RagAnswer(**{**base, **kw})


def test_a_supported_answer_has_no_problems():
    assert check_rag_answer(_answer(), TEXTS) == []


def test_citation_outside_the_passages_is_a_problem():
    problems = check_rag_answer(_answer(citations=[1, 5]), TEXTS)
    assert any("5" in p and "1..2" in p for p in problems)


def test_answered_without_citation_is_a_problem():
    assert any("no valid passage" in p for p in check_rag_answer(_answer(citations=[]), TEXTS))


def test_answered_with_empty_text_is_a_problem():
    assert any("empty" in p for p in check_rag_answer(_answer(answer="  "), TEXTS))


def test_number_missing_from_the_cited_passage_is_a_problem_even_if_another_passage_has_it():
    # 2024 appears in passage 2, but the answer cites passage 1 only
    problems = check_rag_answer(_answer(answer="The invoice is from 2024.", citations=[1]), TEXTS)
    assert any("2024" in p for p in problems)
    assert check_rag_answer(_answer(answer="The invoice is from 2024.", citations=[2]), TEXTS) == []


def test_not_found_is_accepted_without_citations():
    assert check_rag_answer(_answer(status="not_found", answer="I could not find this.", citations=[]), TEXTS) == []
