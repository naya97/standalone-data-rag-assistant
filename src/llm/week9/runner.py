"""One way to ask the model for a small JSON object: send, validate, check, repair ONCE, or give up with a reason.

Used by the sales insights and the document summary. (The RAG answer has its own generator because it also handles
'not found' and citations of retrieved chunks.) Never raises for model or network problems: the caller gets a reason
and builds its own fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, TypeVar

from pydantic import BaseModel

from src.llm.week9.client import MAX_ERROR_CHARS, LLMClient, LLMError, LLMInvalidOutputError
from src.llm.week9.guardrails import validate_output

T = TypeVar("T", bound=BaseModel)


@dataclass
class InsightResult:
    """What the API and the CLI return for an insight or a summary."""
    mode: str                          # "llm" | "fallback"
    result: dict                       # the small JSON object (same fields in both modes)
    fallback_reason: str | None = None  # why the model's answer was not used
    facts: dict | None = None          # sales: the numbers the text must come from
    sources: list[dict] | None = None  # documents: the numbered passages that were summarised
    notes: list[str] = field(default_factory=list)
    generation: dict = field(default_factory=dict)  # model, prompt_version, attempts, repairs, tokens
    prompt_sent: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def ask_json(client: LLMClient | None, system: str, user: str, schema: type[T], check: Callable[[T], list[str]], *,
             prompt_version: str, max_repairs: int = 1,
             unavailable_reason: str | None = None) -> tuple[T | None, dict, str | None, str | None]:
    """Returns (object or None, generation info, failure reason or None, the prompt that was last sent)."""
    info: dict = {"model": getattr(client, "model", None), "prompt_version": prompt_version, "attempts": 0,
                  "repairs": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "model_latency_ms": 0}
    if client is None:
        return None, info, unavailable_reason or "LLM client not configured", None

    message, sent, problems = user, None, []
    for round_no in range(max_repairs + 1):
        sent = f"### SYSTEM\n{system}\n\n### USER\n{message}"
        try:
            result = client.generate(system, message, schema)
        except LLMError as exc:  # the retries for transient errors already happened inside the client
            info["attempts"] += exc.attempts
            return None, info, f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS], sent
        info["attempts"] += result.attempts
        info["input_tokens"] += result.input_tokens or 0
        info["output_tokens"] += result.output_tokens or 0
        info["thinking_tokens"] += result.thinking_tokens or 0
        info["model_latency_ms"] += result.total_latency_ms or result.latency_ms or 0

        try:
            obj = validate_output(result.text, schema)
            problems = check(obj)
        except LLMInvalidOutputError as exc:
            obj = None
            cut = " (the output was cut off at the token limit)" if result.finish_reason == "MAX_TOKENS" else ""
            problems = [str(exc) + cut]
        if not problems:
            return obj, info, None, sent
        if round_no < max_repairs:
            info["repairs"] += 1
            listed = "\n".join(f"- {p}" for p in problems)
            message = (f"{user}\n\nYour previous reply was rejected:\n{listed}\n\nPrevious reply:\n{result.text[:800]}\n\n"
                       "Return a corrected JSON object only, following every rule.")
    return None, info, ("guardrails: " + "; ".join(problems))[:MAX_ERROR_CHARS], sent
