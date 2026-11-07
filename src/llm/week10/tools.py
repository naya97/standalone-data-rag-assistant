from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from src.llm.week9.client import MAX_ERROR_CHARS
from src.llm.week9.tool_specs import (GET_SALES_SUMMARY, SEARCH_DOCUMENTS, TOOLS, ToolRuntime, ToolSpec, run_tool,
                                      validate_call)
from src.rag.week10.langchain_retriever import RagRetriever

DEFAULT_TOP_K = 3  # the same default the model is told in the tool description


def _search_documents_function(runtime: ToolRuntime):
    def search_documents(query: str, project: str | None = None, document_type: str | None = None,
                         month: str | None = None, top_k: int = DEFAULT_TOP_K) -> dict:
        try:
            if runtime.retriever is None:
                return {"error": "search_documents is not available: no retriever was given"}
            filters = {k: v for k, v in (("project", project), ("document_type", document_type), ("month", month)) if v}
            docs = RagRetriever(retriever=runtime.retriever).invoke(query, top_k=top_k, filters=filters or None)
            if not docs:
                return {"passages": [], "note": "no_documents_match_filter" if filters else "no_passages"}
            # the same keys and rounding as Week 9's run_tool, so the guardrails and the summary prompt see the same shape
            return {"passages": [{"id": i, "source": d.metadata["source"], "score": round(d.metadata["score"], 4),
                                  "text": d.page_content} for i, d in enumerate(docs, 1)]}
        except Exception as exc:  # noqa: BLE001 - the boundary of a tool: whatever goes wrong becomes a message
            return {"error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]}
    return search_documents


def _get_sales_summary_function(runtime: ToolRuntime):
    def get_sales_summary(category: str | None = None, start: str | None = None, end: str | None = None) -> dict:
        args = {k: v for k, v in (("category", category), ("start", start), ("end", end)) if v}
        return run_tool("get_sales_summary", args, runtime)  # never raises: failures are {"error": ...}
    return get_sales_summary


_FUNCTIONS = {SEARCH_DOCUMENTS.name: _search_documents_function, GET_SALES_SUMMARY.name: _get_sales_summary_function}


def _as_tool(spec: ToolSpec, runtime: ToolRuntime) -> StructuredTool:
    return StructuredTool.from_function(func=_FUNCTIONS[spec.name](runtime), name=spec.name, description=spec.description,
                                        args_schema=spec.parameters)


def build_tools(runtime: ToolRuntime) -> list[StructuredTool]:
    """The Week 9 tools as LangChain tools, ready for `model.bind_tools(...)`. A missing retriever or database path does not
    fail here: that tool answers {"error": ...} when it is called."""
    return [_as_tool(spec, runtime) for spec in TOOLS]


def run_validated(tools: list[StructuredTool], name: str | None, args: dict | None, config: RunnableConfig | None = None) -> dict:
    """Validate the model's call, then run the tool. ALWAYS returns a dict:
       - a call that fails validation -> {"error": reason, "rejected": True} and nothing is executed
       - a tool that fails -> {"error": ...}
       - otherwise the tool's result.
    `config` carries the callbacks (stage 4: the readable tool log)."""
    clean, problem = validate_call(name, args)
    if problem is not None:
        return {"error": f"invalid tool call: {problem}", "rejected": True}
    by_name: dict[str, Any] = {t.name: t for t in tools}
    tool = by_name.get(name)
    if tool is None:
        return {"error": f"tool {name!r} is not available in this assistant", "rejected": True}
    try:
        return tool.invoke(clean, config=config)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]}
