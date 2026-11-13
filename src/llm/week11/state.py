from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Route = Literal["rag", "sql", "fallback"]
Outcome = Literal["answer", "not_found", "fallback"]


class RouterState(TypedDict, total=False):
    """The one object that travels through the graph. A node returns only the keys it changes."""
    question: str
    route: Route                                # set by classify
    tool: str | None                            # search_documents | get_sales_summary
    args: dict                                  # the tool arguments as the MODEL chose them
    run_args: dict | None                       # the arguments the tool really ran with (after the backend's cleaning)
    clean_notes: list[str]                      # what the cleaning changed
    result: Any                                 # the tool's own result
    tool_status: Literal["ok", "error", "rejected"] | None
    valid: bool                                 # set by validate_result
    reason: str | None                          # why the question went to the fallback (a short code)
    error: str | None                           # the detail behind the reason (model error, tool error), for the log
    choose_calls: int                           # model requests made by classify (0 or 1)
    choose_tokens: dict
    answer: str
    outcome: Outcome                            # how the question ended: answered, 'not found' (a correct answer), or fallback
    summary: Any                                # the Week 10 Summary (source, status, citations, requests, tokens) when a summary ran
    trace: Annotated[list[str], operator.add]   # every node appends its name: the path is visible and testable


def model_requests(state: RouterState) -> int:
    """Model requests made for this question: the classification (0 or 1) plus the summary (0, 1 or 2 with its one repair)."""
    summary = state.get("summary")
    return state.get("choose_calls", 0) + (summary.requests if summary else 0)
