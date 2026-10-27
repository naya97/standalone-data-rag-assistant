"""Checks applied to LLM output BEFORE it is trusted. Nothing here calls a model.

  parse_json / validate_output : text -> schema object, or LLMInvalidOutputError with a short, model-readable reason
  ungrounded_numbers           : numbers the answer states that its sources never state (the cheapest hallucination
                                 signal that is also hard to fool: amounts, dates, counts)
  check_rag_answer             : schema is valid AND citations are real AND the cited passages support the numbers

Failures are returned as plain-language problems, so the caller can put them in a repair request or in a log line.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.week9.client import LLMInvalidOutputError
from src.llm.week9.schemas import DocumentSummary, RagAnswer, SalesInsights

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")


# ----------------------------------------------------------------------------------------------- JSON
def parse_json(text: str) -> dict:
    cleaned = _FENCE.sub("", (text or "").strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMInvalidOutputError(f"the output is not valid JSON ({exc.msg} at character {exc.pos})") from None
    if not isinstance(data, dict):
        raise LLMInvalidOutputError("the output must be a single JSON object")
    return data


def validate_output(text: str, schema: type[T]) -> T:
    data = parse_json(text)
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3])
        raise LLMInvalidOutputError(f"the JSON does not match the required schema ({problems})") from None


# ----------------------------------------------------------------------------------------------- numbers
def _canonical(token: str) -> str:
    """'1,250.00' -> '1250', '12,5' -> '12.5', '007' -> '7': one spelling per value."""
    if _THOUSANDS.match(token):
        token = token.replace(",", "")
    elif "," in token:
        token = token.replace(",", ".")
    if "." in token:
        head, _, tail = token.partition(".")
        tail = tail.replace(".", "").rstrip("0")
        head = head.lstrip("0") or "0"
        return f"{head}.{tail}" if tail else head
    return token.lstrip("0") or "0"


def extract_numbers(text: str) -> set[str]:
    """Canonical numbers in a text; Arabic-Indic digits count as ordinary digits."""
    return {_canonical(m.group()) for m in _NUMBER.finditer((text or "").translate(_ARABIC_DIGITS))}


def ungrounded_numbers(answer: str, sources: list[str]) -> list[str]:
    allowed: set[str] = set()
    for source in sources:
        allowed |= extract_numbers(source)
    return sorted(extract_numbers(answer) - allowed)


# ----------------------------------------------------------------------------------------------- RAG answer
def check_rag_answer(answer: RagAnswer, chunk_texts: list[str]) -> list[str]:
    """Problems found in an already schema-valid answer (empty list = accepted).
    `chunk_texts[i-1]` is the passage the model saw as number i."""
    problems: list[str] = []
    total = len(chunk_texts)
    cited = sorted(set(answer.citations))
    out_of_range = [c for c in cited if not 1 <= c <= total]
    if out_of_range:
        problems.append(f"citations {out_of_range} do not exist; valid passage numbers are 1..{total}")
    if answer.status == "answered":
        valid = [c for c in cited if 1 <= c <= total]
        if not answer.answer.strip():
            problems.append("status is 'answered' but the answer is empty")
        if not valid:
            problems.append("status is 'answered' but no valid passage is cited")
        else:
            extra = ungrounded_numbers(answer.answer, [chunk_texts[c - 1] for c in valid])
            if extra:
                problems.append(f"the numbers {extra} do not appear in the cited passages; "
                                "copy numbers exactly from the passages and never compute new ones")
    return problems


# ----------------------------------------------------------------------------------------------- sales insights
def check_sales_insights(result: SalesInsights, facts_json: str) -> list[str]:
    """Problems in a schema-valid SalesInsights (empty list = accepted)."""
    problems: list[str] = []
    if not result.summary.strip():
        problems.append("the summary is empty")
    if not [i for i in result.insights if i.strip()]:
        problems.append("at least one insight is required")
    text = " ".join([result.summary, *result.insights, *result.caveats])
    extra = ungrounded_numbers(text, [facts_json])
    if extra:
        problems.append(f"the numbers {extra} do not appear in FACTS; copy numbers exactly from FACTS and never compute new ones")
    return problems


# ----------------------------------------------------------------------------------------------- document summary
def check_document_summary(result: DocumentSummary, passage_texts: list[str]) -> list[str]:
    """Problems in a schema-valid DocumentSummary. `passage_texts[i-1]` is the passage the model saw as number i."""
    problems: list[str] = []
    total = len(passage_texts)
    if not result.summary.strip():
        problems.append("the summary is empty")
    for number, point in enumerate(result.key_points, 1):
        bad = sorted({s for s in point.sources if not 1 <= s <= total})
        if bad:
            problems.append(f"key point {number} cites passages {bad}, but the valid numbers are 1..{total}")
        elif not point.sources:
            problems.append(f"key point {number} has no source passage")
    text = " ".join([result.summary, *(p.text for p in result.key_points)])
    extra = ungrounded_numbers(text, passage_texts)  # a summary may use any passage, so all of them count
    if extra:
        problems.append(f"the numbers {extra} do not appear in the passages; copy numbers exactly and never compute new ones")
    return problems
