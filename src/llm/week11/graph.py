from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from src.llm.week11 import placeholders as default_nodes
from src.llm.week11.state import RouterState

NODE_NAMES = ("check_input", "classify", "rag_route", "sql_route", "validate_result", "answer", "fallback")


def route_after_input(state: RouterState) -> str:
    """Edge 0: a rejected input goes straight to the fallback, before any model request."""
    return "fallback" if state.get("route") == "fallback" else "classify"


def route_after_classify(state: RouterState) -> str:
    """Edge 1: where does the question go? Anything that is not clearly rag or sql goes to the fallback."""
    return {"rag": "rag_route", "sql": "sql_route"}.get(state.get("route"), "fallback")


def route_after_validate(state: RouterState) -> str:
    """Edge 2: a valid result is answered, anything else goes to the fallback."""
    return "answer" if state.get("valid") else "fallback"


def build_graph(nodes: dict[str, Callable] | None = None):
    """Compile the router. `nodes` replaces any node by name (tests, and the real nodes of make_nodes)."""
    fns = {name: getattr(default_nodes, name) for name in NODE_NAMES}
    fns.update(nodes or {})
    graph = StateGraph(RouterState)
    for name in NODE_NAMES:
        graph.add_node(name, fns[name])
    graph.add_edge(START, "check_input")
    graph.add_conditional_edges("check_input", route_after_input, {"classify": "classify", "fallback": "fallback"})
    graph.add_conditional_edges("classify", route_after_classify,
                                {"rag_route": "rag_route", "sql_route": "sql_route", "fallback": "fallback"})
    graph.add_edge("rag_route", "validate_result")
    graph.add_edge("sql_route", "validate_result")
    graph.add_conditional_edges("validate_result", route_after_validate, {"answer": "answer", "fallback": "fallback"})
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)
    return graph.compile()
