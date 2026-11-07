from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.llm.week9.client import MAX_ERROR_CHARS, LLMConfigError
from src.llm.week9.config import LLMConfig, get_retail_db_path
from src.llm.week9.tool_specs import SEARCH_DOCUMENTS, TOOL_SYSTEM, ToolRuntime
from src.llm.week10.arguments import MetadataCatalog, clean_search_arguments
from src.llm.week10.model_factory import build_chat_model, to_llm_error
from src.llm.week10.request_layer import LangChainRequestLayer
from src.llm.week10.summarize import Summary, render_data, summarize_result
from src.llm.week10.tool_logging import ToolLog
from src.llm.week10.tools import build_tools, run_validated

Route = Literal["tool", "no_tool", "unavailable"]
ToolStatus = Literal["ok", "error", "rejected"]


@dataclass
class AssistantAnswer:
    question: str
    text: str
    route: Route                                  # tool: a tool was chosen | no_tool: the model answered in words | unavailable: no model
    tool: str | None = None
    args: dict = field(default_factory=dict)
    tool_status: ToolStatus | None = None
    summary: Summary | None = None                # None: not summarized (summarize=False, direct tool run, or no tool)
    error: str | None = None                      # why the route is "unavailable"
    choose_tokens: dict = field(default_factory=dict)   # {"input": .., "output": ..} of the choice request (only when it succeeded)
    choose_calls: int = 0                               # 1 when the choice request was made, even if it failed
    attempts: int | None = None                         # requests actually SENT for this answer, retries included (from the
                                                        # request layer); None when there is no layer (tests with scripted models)
    seconds: float = 0.0
    question_id: str = ""
    result: Any = field(default=None, repr=False)       # the tool's own result (None when no tool ran)
    run_args: dict | None = None                        # the arguments the tool really ran with (after the backend's cleaning)
    clean_notes: list[str] = field(default_factory=list)  # what the cleaning changed, and why (args stays what the MODEL chose)

    @property
    def model_requests(self) -> int:
        """The number of model requests that were made to produce this answer (1 for the choice, 1 for the summary, 0 for a direct tool run)."""
        return self.choose_calls + (self.summary.requests if self.summary else 0)

    def to_dict(self) -> dict:
        """Printable form: the tool result is shortened (see brief_result)."""
        return {**asdict(self), "result": brief_result(self.result), "model_requests": self.model_requests}


@dataclass
class Choice:
    """The model's answer to "which tool?": step 1 of the flow, usable on its own (evaluation: one request per question)."""
    tool: str | None = None                  # None: no tool chosen (see text) or the request failed (see error)
    args: dict = field(default_factory=dict)
    text: str | None = None                  # the model answered in words
    error: str | None = None
    calls: int = 0                           # 0: the model was never asked (not configured); 1: asked (even if it failed)
    tokens: dict = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)   # further tool calls in the same reply (only the first is used)


class _LazyRetriever:
    """The Week 8 retriever, built on first use: a sales question never pays for loading the embedding model."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory, self._real = factory, None

    def retrieve(self, question: str, top_k: int | None = None, filters: dict | None = None):
        if self._real is None:
            self._real = self._factory()
        return self._real.retrieve(question, top_k=top_k, filters=filters)


def _week8_retriever():
    from src.rag.week8.config import get_config
    from src.rag.week8.rag_service import RagService
    return RagService.from_config(get_config(), warm_up=True).retriever


def default_runtime() -> ToolRuntime:
    return ToolRuntime(retriever=_LazyRetriever(_week8_retriever), retail_db_path=get_retail_db_path())


class Assistant:
    def __init__(self, chooser: Runnable | None, summarizer: Runnable | None, tools: list, log: ToolLog | None = None,
                 unavailable_reason: str | None = None, clock: Callable[[], float] = time.perf_counter,
                 layer: LangChainRequestLayer | None = None,
                 cleaner: Callable[[str, dict], tuple[dict, list[str]]] | None = None):
        self.chooser, self.summarizer, self.tools = chooser, summarizer, tools
        self.cleaner = cleaner   # (question, model's search arguments) -> (arguments to run, notes); None = run what the model chose
        self.layer = layer   # the request layer behind chooser/summarizer (None in tests with scripted models)
        self.log = log or ToolLog()
        self.unavailable_reason = unavailable_reason or "no model was configured"
        self._clock = clock
        # SystemMessage object, not a template string: the tool rules are never parsed as {variables}
        self._choose_prompt = ChatPromptTemplate.from_messages([SystemMessage(TOOL_SYSTEM), ("human", "{question}")])

    # ------------------------------------------------------------------------------------------- the flow
    def ask(self, question: str, summarize: bool = True) -> AssistantAnswer:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        question = question.strip()
        started, sent_before = self._clock(), self._sent()
        self.log.start(question)

        choice = self._choose(question)
        if choice.error is not None:
            why = "the model is not available" if choice.calls == 0 else "the model request failed"
            return self._finish(AssistantAnswer(question, f"No answer: {why}. Reason: {choice.error}\n"
                                                 "The tools can still be run directly (--tool).", "unavailable", error=choice.error,
                                                 choose_calls=choice.calls), started, sent_before)
        if choice.tool is None:
            return self._finish(AssistantAnswer(question, choice.text or "", "no_tool", choose_tokens=choice.tokens,
                                                 choose_calls=1), started, sent_before)
        # 2 + 3 validate and run
        answer = self._run(question, choice.tool, choice.args, summarize)
        answer.choose_tokens, answer.choose_calls = choice.tokens, 1
        return self._finish(answer, started, sent_before)

    def choose(self, question: str) -> Choice:
        """Step 1 alone: one model request, no tool is run (evaluation of the tool choice)."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        self.log.start(question.strip())
        return self._choose(question.strip())

    def run_tool(self, name: str, args: dict | None = None, summarize: bool = False, question: str = "") -> AssistantAnswer:
        """A tool run with no choice request (command line --tool, tests). With summarize=True the summary needs `question`."""
        started, sent_before = self._clock(), self._sent()
        self.log.start(question or f"(direct) {name} {args or {}}")
        return self._finish(self._run(question, name, args or {}, summarize), started, sent_before)

    def as_chain(self) -> Runnable:
        """The whole assistant as one runnable: question string in, AssistantAnswer out (composable, e.g. in Week 11)."""
        return RunnableLambda(self.ask)

    def _choose(self, question: str) -> Choice:
        if self.chooser is None:
            return Choice(error=self.unavailable_reason)
        chose_started = self._clock()
        try:
            message = (self._choose_prompt | self.chooser).invoke({"question": question})
        except Exception as exc:  # noqa: BLE001 - any model/network failure ends here, with the reason
            mapped = to_llm_error(exc)
            reason = f"{type(mapped).__name__}: {mapped}"[:MAX_ERROR_CHARS]
            self.log.step("MODEL", f"FAILED {reason}")
            return Choice(error=reason, calls=1)
        usage = getattr(message, "usage_metadata", None) or {}
        tokens = {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)}
        calls = list(message.tool_calls or [])
        took = f"tokens {tokens['input']}/{tokens['output']}, {self._clock() - chose_started:.1f}s"
        if not calls:
            self.log.step("MODEL", f"no tool needed, answered in words ({took})")
            return Choice(text=message.text.strip(), calls=1, tokens=tokens)
        name, args = calls[0]["name"], calls[0].get("args") or {}
        self.log.step("MODEL", f"chose {name} args={args} ({took})")
        ignored = [c["name"] for c in calls[1:]]
        if ignored:
            self.log.step("MODEL", f"{len(ignored)} more tool call(s) ignored: {ignored}")
        return Choice(tool=name, args=args, calls=1, tokens=tokens, ignored=ignored)

    # steps
    def _run(self, question: str, name: str | None, args: dict, summarize: bool) -> AssistantAnswer:
        run_args, notes = args, []
        if name == SEARCH_DOCUMENTS.name and self.cleaner is not None and question:   # a direct --tool call has no question: arguments are the user's
            run_args, notes = self.cleaner(question, args)
            for note in notes:
                self.log.step("CLEAN", note)
        result = run_validated(self.tools, name, run_args, config={"callbacks": [self.log]})
        rejected = bool(result.get("rejected"))
        status: ToolStatus = "rejected" if rejected else ("error" if "error" in result else "ok")
        if rejected:
            self.log.step("REJECTED", f"{name} args={args}: {result['error']}")  # never executed, so no callback wrote it
        answer = AssistantAnswer(question, "", "tool", tool=name, args=args, tool_status=status, result=result,
                                 run_args=run_args, clean_notes=notes)
        if status != "ok":
            answer.text = f"The tool could not answer: {result['error']}"
            return answer
        if not summarize:
            answer.text = render_data(name, result)
            return answer
        if self.summarizer is None:
            answer.text = render_data(name, result)
            answer.error = f"not summarized: the model is not available ({self.unavailable_reason})"
            return answer
        summary = summarize_result(question, name, result, self.summarizer)
        self.log.step("SUMMARY", f"source={summary.source} status={summary.status} requests={summary.requests} "
                                 f"tokens={summary.input_tokens}/{summary.output_tokens} citations={summary.citations}"
                                 + (f" fallback_reason={summary.fallback_reason}" if summary.fallback_reason else ""))
        answer.summary, answer.text = summary, summary.text
        return answer

    def _sent(self) -> int:
        """Requests the layer has sent so far in this process (0 without a layer). It never resets, so an answer's own
        number is the difference between the end and the start of that answer."""
        return self.layer.requests if self.layer else 0

    def _finish(self, answer: AssistantAnswer, started: float, sent_before: int) -> AssistantAnswer:
        answer.seconds = round(self._clock() - started, 2)
        answer.question_id = self.log.question_id
        answer.attempts = self._sent() - sent_before if self.layer else None
        sent = f", {answer.attempts} requests sent" if answer.attempts is not None else ""
        self.log.step("ANSWER", f"[{answer.route}{'/' + answer.tool_status if answer.tool_status else ''}, "
                                f"{answer.model_requests} model calls{sent}, {answer.seconds}s] {answer.text}")
        return answer


def build_assistant(config: LLMConfig, runtime: ToolRuntime | None = None, log: ToolLog | None = None,
                    layer: LangChainRequestLayer | None = None,
                    outer: Callable[[Runnable, str], Runnable] | None = None,
                    catalog: MetadataCatalog | None = None, clean_arguments: bool = True) -> Assistant:
    """The real assistant. A missing key or model does not raise: the assistant starts and says why it cannot answer.
    Every model call goes through ONE request layer (shared throttle, retries, request counter): pass your own `layer` to
    share a counter/budget across assistants, or leave it None."""
    tools = build_tools(runtime or default_runtime())
    try:
        model = build_chat_model(config)
    except LLMConfigError as exc:
        return Assistant(None, None, tools, log, unavailable_reason=f"{type(exc).__name__}: {exc}")
    cleaner = (lambda question, args: clean_search_arguments(question, args, catalog)) if clean_arguments else None
    layer = layer or LangChainRequestLayer(config, log=log)
    chooser, summarizer = layer.guard(model.bind_tools(tools)), layer.guard(model)
    if outer is not None:   # e.g. the evaluation's reply cache: outside the layer, so a cache hit never reaches the network
        chooser, summarizer = outer(chooser, "choose"), outer(summarizer, "summarize")
    return Assistant(chooser, summarizer, tools, log, layer=layer, cleaner=cleaner)


def brief_result(result: Any) -> Any:
    """A short, printable view of a tool result for the command line."""
    if isinstance(result, dict) and "passages" in result:
        return {"passages": [{"id": p["id"], "source": p["source"], "score": p["score"], "text": p["text"][:160]} for p in result["passages"]]}
    if isinstance(result, dict) and "overall" in result:
        return {"scope": result["scope"], "overall": result["overall"], "best_month": result["best_month"],
                "top_category": result["by_category"][0] if result["by_category"] else None, "notes": result["notes"]}
    return result


__all__ = ["Assistant", "AssistantAnswer", "Choice", "MetadataCatalog", "build_assistant", "default_runtime", "brief_result"]
