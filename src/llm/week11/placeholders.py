"""Stage 1 PLACEHOLDER nodes (no model, no tools): used by the graph tests and as defaults. The real nodes are in nodes.py.
"""
from __future__ import annotations

from src.llm.week11.state import RouterState


def check_input(state: RouterState) -> dict:
    return {"trace": ["check_input"]}


def classify(state: RouterState) -> dict:
    low = state["question"].lower()
    if any(w in low for w in ("revenue", "sales", "category")):
        route = "sql"
    elif any(w in low for w in ("policy", "invoice", "report", "document")):
        route = "rag"
    else:
        route = "fallback"
    return {"route": route, "trace": ["classify"]}


def rag_route(state: RouterState) -> dict:
    return {"tool": "search_documents", "result": {"passages": []}, "trace": ["rag_route"]}


def sql_route(state: RouterState) -> dict:
    return {"tool": "get_sales_summary", "result": {"overall": {}}, "trace": ["sql_route"]}


def validate_result(state: RouterState) -> dict:
    # placeholder rule: a result must exist
    ok = state.get("result") is not None
    return {"valid": ok, "reason": None if ok else "no result", "trace": ["validate_result"]}


def answer(state: RouterState) -> dict:
    return {"answer": f"(placeholder answer via {state['tool']})", "trace": ["answer"]}


def fallback(state: RouterState) -> dict:
    return {"route": "fallback", "answer": "(placeholder fallback message)",
            "reason": state.get("reason") or "not classified", "trace": ["fallback"]}
