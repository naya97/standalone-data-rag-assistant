from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import warnings
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm.week9.client import MAX_ERROR_CHARS, GeminiClient, LLMConfigError, LLMError, map_exception
from src.llm.week9.config import LLMConfig, get_llm_config, setup_llm_logging
from src.llm.week9.retry import RateLimiter
from src.llm.week9.tool_specs import TOOL_SYSTEM, TOOLS

# Some Gemini 3.x models ignore `temperature` and langchain-google-genai warns about it on EVERY request. The setting stays in
# .env (older models still honour it); only this one known, harmless message is hidden.
warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*", category=UserWarning)

DEMO_QUESTIONS = (
    "What is the total amount due for Project Alpha in January 2024?",   # expected: search_documents
    "Which category has the highest revenue?",                           # expected: get_sales_summary
    "What is the capital of France?",                                    # expected: no tool
)


def build_chat_model(config: LLMConfig) -> ChatGoogleGenerativeAI:
    """ChatGoogleGenerativeAI with the Week 9 settings. Raises LLMConfigError for a missing key/model/bad thinking level."""
    if not config.api_key:
        raise LLMConfigError("GEMINI_API_KEY is not set. Add it to .env (and never commit that file).")
    if not config.model:
        raise LLMConfigError("GEMINI_MODEL is not set. Copy the exact model id from Google AI Studio into .env.")
    settings: dict[str, Any] = {
        "model": config.model,
        "google_api_key": config.api_key,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "timeout": config.timeout_seconds,   # seconds (Week 9's GeminiClient sends milliseconds to the SDK; same limit)
        "max_retries": 0,                    # see the module docstring: ONE retry layer, ours
    }
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - langchain-google-genai depends on it
        raise LLMConfigError("The google-genai package is not installed: pip install google-genai") from exc
    thinking = GeminiClient._build_thinking(types, config.thinking_level)  # same validation and message as Week 9
    if thinking is not None:
        settings["thinking_config"] = thinking
    return ChatGoogleGenerativeAI(**settings)


def tool_declarations() -> list[dict]:
    """The Week 9 tool declarations in the plain dict form `bind_tools` accepts (stage 2 replaces them with StructuredTools)."""
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in TOOLS]


def unwrap_provider_error(exc: BaseException) -> BaseException:
    """The first error in the `__cause__` chain that carries an HTTP status code (the original google.genai error).
    Without it, Week 9's map_exception() sees only a wrapper without a code and treats every API failure as 'unexpected'."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(getattr(current, "code", None), int):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return exc


def to_llm_error(exc: BaseException) -> LLMError:
    """Any exception raised by a LangChain model call -> the Week 9 error types (retryable or not, retry_after, ...)."""
    return map_exception(unwrap_provider_error(exc))


# ----------------------------------------------------------------------------------------------- smoke test
def _describe(message: AIMessage, seconds: float) -> dict:
    meta = message.response_metadata or {}
    return {
        "tool_calls": [{"name": c.get("name"), "args": c.get("args")} for c in (message.tool_calls or [])],
        "text": str(message.text)[:300],
        "content_type": type(message.content).__name__,
        "usage_metadata": dict(message.usage_metadata or {}),
        "finish_reason": meta.get("finish_reason"),
        "seconds": round(seconds, 2),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--demo", action="store_true", help="3 fixed questions")
    parser.add_argument("--model", help="override GEMINI_MODEL for this run (e.g. a wrong id, to see the error mapping)")
    args = parser.parse_args(argv)
    if not args.question and not args.demo:
        parser.error("give a question, or --demo")

    config = get_llm_config()
    if args.model:
        config = dataclasses.replace(config, model=args.model)
    setup_llm_logging(config)
    try:
        model = build_chat_model(config)
    except LLMConfigError as exc:
        print(f"FAILED ({type(exc).__name__}): {exc}")
        return 1

    bound = model.bind_tools(tool_declarations())
    limiter = RateLimiter(config.max_rpm)   # the same spacing rule as Week 9
    questions = list(DEMO_QUESTIONS) if args.demo else [args.question]
    for question in questions:
        limiter.wait()
        started = time.perf_counter()
        try:
            message = bound.invoke([SystemMessage(TOOL_SYSTEM), HumanMessage(question)])
        except Exception as exc:  # noqa: BLE001 - the point of this test is to see what arrives
            mapped = to_llm_error(exc)
            print(json.dumps({"question": question, "error": {
                "raised": type(exc).__name__, "cause": type(exc.__cause__).__name__ if exc.__cause__ else None,
                "mapped_to": type(mapped).__name__, "retryable": mapped.retryable, "retry_after": mapped.retry_after,
                "message": str(mapped)[:MAX_ERROR_CHARS]}}, ensure_ascii=False, indent=2))
            continue
        print(json.dumps({"question": question, **_describe(message, time.perf_counter() - started)},
                         ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
