"""Function-calling CONCEPT (Week 9): the model chooses a tool and its arguments; we validate them and run the tool.

    question -> model sees the tool declarations -> picks ONE tool (or answers in words) -> we validate -> we execute

What belongs here: declaring tools, reading the model's choice, never trusting its arguments, running a tool without ever
raising. What does NOT belong here: sending the tool's result back to the model for a final answer, several calls in a row,
choosing between RAG / insights / tools automatically. That loop and the router are Weeks 10-11 (LangChain tools).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm.week9.client import MAX_ERROR_CHARS, LLMClient, LLMError
from src.llm.week9.data_facts import get_sales_facts


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict  # OpenAPI-style subset (type, properties, enum, required): all the API is sent. Ranges/patterns are below.


SEARCH_DOCUMENTS = ToolSpec(
    name="search_documents",
    description=("Search the company documents (invoices, billing and security policies, monthly reports, job briefs, the "
                 "email payload specification) and return the most relevant passages for a question."),
    parameters={"type": "object", "properties": {
        "query": {"type": "string", "description": "What to look for, as a short question or phrase."},
        "project": {"type": "string", "description": "Only this project, exact name such as projectAlpha. Only if the question names it."},
        "document_type": {"type": "string", "enum": ["invoice", "policy", "report"], "description": "Only this kind of document. Only if the question says so."},
        "month": {"type": "string", "description": "Only this month, written YYYY-MM. Only if the question names it."},
        "top_k": {"type": "integer", "description": "How many passages to return, 1 to 10. Default 3."}},
        "required": ["query"]})

GET_SALES_SUMMARY = ToolSpec(
    name="get_sales_summary",
    description=("Get computed figures from the cleaned retail sales data: revenue, number of orders, average order value, "
                 "months, categories, payment methods, locations, and category targets."),
    parameters={"type": "object", "properties": {
        "category": {"type": "string", "description": "Only this product category, exact name such as Furniture. Only if the question names it."},
        "start": {"type": "string", "description": "First month, written YYYY-MM. Only if the question gives a period."},
        "end": {"type": "string", "description": "Last month, written YYYY-MM. Only if the question gives a period."}},
        "required": []})

TOOLS: tuple[ToolSpec, ...] = (SEARCH_DOCUMENTS, GET_SALES_SUMMARY)
_BY_NAME = {t.name: t for t in TOOLS}

# What we check ourselves (not sent to the API, which may not understand every keyword): ranges and formats.
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_RANGES = {("search_documents", "top_k"): (1, 10)}
_FORMATS = {("search_documents", "month"): _MONTH, ("get_sales_summary", "start"): _MONTH, ("get_sales_summary", "end"): _MONTH}

TOOL_SYSTEM = """You decide which tool answers the user's question.
- search_documents: questions about what the company documents say (invoices, billing and security policies, reports, job briefs, the email payload specification).
- get_sales_summary: questions about retail sales figures (revenue, orders, months, categories, payment methods).
Call exactly one tool when a tool fits. Fill an argument only with a value the question states; never invent values. If no tool fits (general knowledge, small talk), answer in one short sentence and do not call a tool."""


# ----------------------------------------------------------------------------------------------- validating the model's choice
def validate_call(name: str | None, args: dict | None) -> tuple[dict | None, str | None]:
    """(clean arguments, None) or (None, why not). The model's arguments are untrusted input, like any user's."""
    spec = _BY_NAME.get(name or "")
    if spec is None:
        return None, f"unknown tool {name!r}; known tools: {sorted(_BY_NAME)}"
    props, required = spec.parameters["properties"], spec.parameters.get("required", [])
    args = dict(args or {})
    unknown = sorted(k for k in args if k not in props)
    if unknown:
        return None, f"{name}: unknown arguments {unknown}; allowed: {sorted(props)}"
    clean: dict[str, Any] = {}
    for key, value in args.items():
        kind, allowed = props[key]["type"], props[key].get("enum")
        if kind == "integer":
            if isinstance(value, float) and value.is_integer():
                value = int(value)  # JSON numbers often arrive as 3.0
            if isinstance(value, bool) or not isinstance(value, int):
                return None, f"{name}.{key} must be an integer, got {value!r}"
            low, high = _RANGES.get((name, key), (None, None))
            if (low is not None and value < low) or (high is not None and value > high):
                return None, f"{name}.{key} must be between {low} and {high}, got {value}"
        else:
            if not isinstance(value, str) or not value.strip():
                return None, f"{name}.{key} must be a non-empty string, got {value!r}"
            value = value.strip()
            if allowed and value not in allowed:
                return None, f"{name}.{key} must be one of {allowed}, got {value!r}"
            pattern = _FORMATS.get((name, key))
            if pattern and not pattern.match(value):
                return None, f"{name}.{key} must look like 2024-03, got {value!r}"
        clean[key] = value
    missing = [k for k in required if k not in clean]
    if missing:
        return None, f"{name}: missing required arguments {missing}"
    return clean, None


# ----------------------------------------------------------------------------------------------- the model's choice
@dataclass
class ToolDecision:
    tool: str | None                  # None: the model chose no tool (see answer_text) or the request failed (see error)
    args: dict = field(default_factory=dict)
    error: str | None = None          # failed request, or arguments that did not validate (tool is then still the model's choice)
    answer_text: str | None = None    # the model answered in words: no tool needed
    generation: dict = field(default_factory=dict)


def decide_tool(client: LLMClient | None, question: str, unavailable_reason: str | None = None) -> ToolDecision:
    """One model request: which tool, if any. Never raises for model or network problems."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if client is None:
        return ToolDecision(None, error=unavailable_reason or "LLM client not configured")
    try:
        result = client.choose_tool(TOOL_SYSTEM, question.strip(), TOOLS)
    except LLMError as exc:
        return ToolDecision(None, error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS], generation={"model": getattr(client, "model", None), "attempts": exc.attempts})
    info = {"model": result.model, "attempts": result.attempts, "input_tokens": result.input_tokens or 0, "output_tokens": result.output_tokens or 0,
            "thinking_tokens": result.thinking_tokens or 0, "model_latency_ms": result.total_latency_ms or result.latency_ms or 0}
    if result.tool_name is None:
        return ToolDecision(None, answer_text=result.text.strip(), generation=info)
    clean, error = validate_call(result.tool_name, result.tool_args)
    return ToolDecision(result.tool_name, clean or {}, error, generation=info)


# ----------------------------------------------------------------------------------------------- running a tool
@dataclass
class ToolRuntime:
    """What the tools need. Anything missing makes that tool answer {"error": ...} instead of crashing."""
    retriever: Any = None                 # needs .retrieve(question, top_k=, filters=) like src.rag.week8.retriever.Retriever
    retail_db_path: str | Path | None = None


def run_tool(name: str, args: dict, runtime: ToolRuntime) -> dict:
    """Execute a tool with ALREADY VALIDATED arguments. Always returns a dict; failures are {"error": "..."}."""
    try:
        if name == "search_documents":
            if runtime.retriever is None:
                return {"error": "search_documents is not available: no retriever was given"}
            filters = {k: args[k] for k in ("project", "document_type", "month") if k in args}
            found = runtime.retriever.retrieve(args["query"], top_k=args.get("top_k", 3), filters=filters or None)
            if found.status != "retrieved":
                return {"passages": [], "note": found.status}
            return {"passages": [{"id": i, "source": c.original_filename, "score": round(c.score, 4), "text": c.text}
                                 for i, c in enumerate(found.chunks, 1)]}
        if name == "get_sales_summary":
            if runtime.retail_db_path is None:
                return {"error": "get_sales_summary is not available: no retail database path was given"}
            return get_sales_facts(runtime.retail_db_path, category=args.get("category"), start=args.get("start"), end=args.get("end"))
        return {"error": f"unknown tool {name!r}"}
    except Exception as exc:  # noqa: BLE001 - the boundary of a tool: whatever goes wrong becomes a message, never a crash
        return {"error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]}
