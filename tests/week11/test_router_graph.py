from src.llm.week11.graph import build_graph, route_after_classify, route_after_input, route_after_validate


def run(question, nodes=None):
    return build_graph(nodes).invoke({"question": question, "trace": []})


def test_sql_path():
    assert run("What is the total revenue?")["trace"] == ["check_input", "classify", "sql_route", "validate_result", "answer"]


def test_rag_path():
    assert run("Explain the company policy.")["trace"] == ["check_input", "classify", "rag_route", "validate_result", "answer"]


def test_unclassified_goes_to_fallback_directly():
    out = run("Tell me something interesting.")
    assert out["trace"] == ["check_input", "classify", "fallback"] and out["route"] == "fallback"


def test_invalid_result_goes_to_fallback():
    out = run("What is the total revenue?", {"validate_result": lambda s: {"valid": False, "reason": "empty", "trace": ["validate_result"]}})
    assert out["trace"] == ["check_input", "classify", "sql_route", "validate_result", "fallback"]
    assert out["reason"] == "empty"


def test_edge_functions():
    assert route_after_classify({"route": "rag"}) == "rag_route"
    assert route_after_classify({"route": "sql"}) == "sql_route"
    assert route_after_classify({"route": "fallback"}) == "fallback"
    assert route_after_classify({}) == "fallback"
    assert route_after_validate({"valid": True}) == "answer"
    assert route_after_validate({"valid": False}) == "fallback"
    assert route_after_validate({}) == "fallback"


def test_rejected_input_skips_classify():
    out = run("   ", {"check_input": lambda s: {"route": "fallback", "reason": "empty_question", "trace": ["check_input"]}})
    assert out["trace"] == ["check_input", "fallback"]
    assert route_after_input({"route": "fallback"}) == "fallback" and route_after_input({}) == "classify"
