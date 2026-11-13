from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.llm.week9.tool_specs import ToolRuntime
from src.llm.week10.assistant import Assistant
from src.llm.week10.tool_logging import ToolLog
from src.llm.week10.tools import build_tools
from src.llm.week11.graph import build_graph
from src.llm.week11.nodes import make_nodes


def router(message: AIMessage):
    tools = build_tools(ToolRuntime(retriever=None, retail_db_path=None))
    assistant = Assistant(RunnableLambda(lambda _: message), None, tools, ToolLog())
    return build_graph(make_nodes(assistant))


def test_invalid_tool_call_is_rejected_and_ends_in_fallback():
    call = AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {}, "id": "c1"}])   # the required `query` is missing
    out = router(call).invoke({"question": "Find the documents.", "trace": []})
    assert out["trace"] == ["check_input", "classify", "rag_route", "validate_result", "fallback"]
    assert out["tool_status"] == "rejected" and out["reason"] == "rejected_call"


def test_answer_in_words_ends_in_fallback_with_no_tool():
    out = router(AIMessage(content="It is sunny.")).invoke({"question": "What is the weather today?", "trace": []})
    assert out["trace"] == ["check_input", "classify", "fallback"] and out["reason"] == "no_tool"
    assert out["choose_calls"] == 1
