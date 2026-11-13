"""The real nodes, built around a Week 10 `Assistant` (its chooser, tools, cleaner and log are reused as they are).

Stage 4: all six nodes are final. `answer` is the Week 10 summary step (the model words the tool result; its guardrails and its
one repair request are reused as they are)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from src.llm.week10.summarize import render_data, summarize_result
from src.llm.week11.state import RouterState, model_requests
from src.llm.week11.validation import check_question, check_result, fallback_message

if TYPE_CHECKING:  # only for the reader: the nodes need no import of the assistant to run
    from src.llm.week10.assistant import Assistant

TOOL_TO_ROUTE = {"search_documents": "rag", "get_sales_summary": "sql"}


def make_nodes(assistant: "Assistant") -> dict[str, Callable[[RouterState], dict]]:
    log = assistant.log

    def check_input(state: RouterState) -> dict:
        """Empty or too long: stopped here, with no model request."""
        question = state.get("question")
        reason = check_question(question)
        if reason is None:
            return {"trace": ["check_input"]}
        log.start(str(question or ""))                      # a valid question is logged by classify (choose); this one never gets there
        log.step("INPUT", f"rejected: {reason}")
        return {"route": "fallback", "reason": reason, "choose_calls": 0, "trace": ["check_input"]}

    def classify(state: RouterState) -> dict:
        """One model request: the model's tool choice IS the classification (no extra request)."""
        choice = assistant.choose(state["question"])      # also starts the log entry (QUESTION + MODEL lines)
        out: dict = {"choose_calls": choice.calls, "choose_tokens": choice.tokens, "trace": ["classify"]}
        if choice.error is not None:
            out.update(route="fallback", reason="model_unavailable" if choice.calls == 0 else "model_failed", error=choice.error)
        elif choice.tool is None:
            out.update(route="fallback", reason="no_tool")                        # the model answered in words
        elif choice.tool not in TOOL_TO_ROUTE:
            out.update(route="fallback", reason="unknown_tool", error=choice.tool)
        else:
            out.update(route=TOOL_TO_ROUTE[choice.tool], tool=choice.tool, args=choice.args)
        log.step("ROUTE", out["route"] + (f" (reason: {out['reason']})" if out.get("reason") else ""))
        return out

    def _run_selected_tool(state: RouterState) -> dict:
        # the Week 10 step: clean the arguments, validate the call, run the tool. No summary here (that is the answer node).
        ran = assistant._run(state["question"], state["tool"], state["args"], summarize=False)
        return {"result": ran.result, "tool_status": ran.tool_status, "run_args": ran.run_args, "clean_notes": ran.clean_notes}

    def rag_route(state: RouterState) -> dict:
        return {**_run_selected_tool(state), "trace": ["rag_route"]}

    def sql_route(state: RouterState) -> dict:
        return {**_run_selected_tool(state), "trace": ["sql_route"]}

    def validate_result(state: RouterState) -> dict:
        reason = check_result(state.get("tool"), state.get("result"), state.get("tool_status"))
        if reason is None:
            return {"valid": True, "trace": ["validate_result"]}
        result = state.get("result")
        error = result.get("error") if isinstance(result, dict) else None
        log.step("VALIDATE", f"failed: {reason}")
        return {"valid": False, "reason": reason, "error": error, "trace": ["validate_result"]}

    def answer(state: RouterState) -> dict:
        question = state["question"].strip()
        out: dict = {"trace": ["answer"]}
        if assistant.summarizer is None:     # cannot happen after a successful classify; kept for the tests with no model
            out.update(answer=render_data(state["tool"], state["result"]), outcome="answer")
            log.step("SUMMARY", "skipped: no summarizer")
        else:
            summary = summarize_result(question, state["tool"], state["result"], assistant.summarizer)
            log.step("SUMMARY", f"source={summary.source} status={summary.status} requests={summary.requests} "
                                f"tokens={summary.input_tokens}/{summary.output_tokens} citations={summary.citations}"
                                + (f" fallback_reason={summary.fallback_reason}" if summary.fallback_reason else ""))
            out.update(answer=summary.text, summary=summary, outcome="not_found" if summary.status == "not_found" else "answer")
        log.step("ANSWER", f"[{state['route']}/{out['outcome']}, {model_requests({**state, **out})} model calls] {out['answer']}")
        return out

    def fallback(state: RouterState) -> dict:
        text = fallback_message(state.get("reason"), state.get("question"))
        log.step("ANSWER", f"[fallback/{state.get('reason')}, {model_requests(state)} model calls] {text}")
        return {"route": "fallback", "answer": text, "outcome": "fallback", "trace": ["fallback"]}

    return {"check_input": check_input, "classify": classify, "rag_route": rag_route, "sql_route": sql_route,
            "validate_result": validate_result, "answer": answer, "fallback": fallback}
