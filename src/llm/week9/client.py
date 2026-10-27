"""LLM client layer: error types, the LLMClient interface, GeminiClient, FakeLLMClient.

Everything else in Week 9+ talks to `LLMClient.generate(system, user, schema)`, so the provider can change
(or be faked in tests) without touching prompts, guardrails or the RAG generator.
`build_client()` returns the production client: Gemini wrapped with retries, timeout budget and throttling.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from pydantic import BaseModel

    from src.llm.week9.config import LLMConfig

# Error text is kept long enough to show the provider's quota details (which limit, which model, retry delay).
MAX_ERROR_CHARS = 1000


# ----------------------------------------------------------------------------------------------- errors
class LLMError(Exception):
    """Base class. `retryable` tells the retry layer whether another attempt can help."""

    retryable = False

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after  # seconds, when the provider says how long to wait
        self.attempts = 0               # filled in by the retry layer


class LLMConfigError(LLMError):
    """Missing key/model, unknown model (404), SDK not installed, invalid thinking level. Fix the configuration."""


class LLMAuthError(LLMError):
    """Invalid key or no permission (401/403)."""


class LLMRequestError(LLMError):
    """The request itself was rejected (400, bad schema/parameters)."""


class LLMBlockedError(LLMError):
    """The model returned no text (safety block, empty candidate)."""


class LLMInvalidOutputError(LLMError):
    """Text came back but is not valid JSON / does not fit the schema (raised by guardrails, not by the client)."""


class LLMQuotaExhaustedError(LLMError):
    """429 that waiting will not fix: the daily quota is used up, or this model has a quota of 0 on this key.
    Not retried: three more attempts would only burn 15-30 seconds before failing the same way."""


class LLMTimeoutError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):  # 429 per-minute limit: a short wait fixes it
    retryable = True


class LLMServerError(LLMError):  # 5xx
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


# ----------------------------------------------------------------------------------------------- result / interface
@dataclass
class LLMResult:
    text: str
    model: str = ""
    latency_ms: int = 0          # the successful request
    total_latency_ms: int = 0    # the whole call including retries and waits (set by the retry layer)
    attempts: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None    # the visible answer only
    thinking_tokens: int | None = None  # hidden reasoning tokens (not part of output_tokens)
    finish_reason: str | None = None  # e.g. "MAX_TOKENS" explains truncated JSON
    tool_name: str | None = None      # choose_tool(): the tool the model chose (None: it answered in words, see text)
    tool_args: dict | None = None


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, user: str, schema: "type[BaseModel] | None" = None) -> LLMResult: ...


# ----------------------------------------------------------------------------------------------- error mapping
_RETRY_IN_TEXT = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"^([0-9]+(?:\.[0-9]+)?)s$")


def _detail_items(exc: BaseException) -> list:
    """The provider's structured error details (a list of dicts), whatever shape the SDK gives us."""
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and isinstance(details.get("error"), dict):
        details = details["error"]
    items = details.get("details") if isinstance(details, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _retry_after(items: list, message: str) -> float | None:
    for item in items:
        if str(item.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY.match(str(item.get("retryDelay", "")).strip())
            if match:
                return float(match.group(1))
    match = _RETRY_IN_TEXT.search(message)
    return float(match.group(1)) if match else None


def _quota_is_exhausted(items: list, message: str) -> bool:
    """Daily quota used up, or a quota of 0 (the model is not available on this key's plan)."""
    if "limit: 0" in message.lower():
        return True
    for item in items:
        if str(item.get("@type", "")).endswith("QuotaFailure"):
            for violation in item.get("violations") or []:
                label = f"{violation.get('quotaId', '')} {violation.get('quotaMetric', '')}".lower()
                if "perday" in label or "per_day" in label or "daily" in label:
                    return True
    return False


def map_exception(exc: BaseException) -> LLMError:
    """Translate whatever the SDK/network raised into our error types (duck-typed: no SDK import needed)."""
    if isinstance(exc, LLMError):
        return exc
    code = getattr(exc, "code", None)  # google.genai.errors.APIError.code is the HTTP status
    message = str(getattr(exc, "message", None) or exc)[:MAX_ERROR_CHARS]
    if isinstance(code, int):
        lowered = message.lower()
        if code in (401, 403) or (code == 400 and "api key" in lowered):
            return LLMAuthError(f"Gemini rejected the API key or permissions ({code}): {message}")
        if code == 404:
            return LLMConfigError(f"Model not found (404). Check GEMINI_MODEL in .env: {message}")
        if code == 429:
            items = _detail_items(exc)
            if _quota_is_exhausted(items, message):
                return LLMQuotaExhaustedError(f"Quota exhausted (429), retrying will not help: {message}")
            return LLMRateLimitError(f"Rate limit exceeded (429): {message}", retry_after=_retry_after(items, message))
        if code >= 500:
            return LLMServerError(f"Gemini server error ({code}): {message}")
        return LLMRequestError(f"Request rejected ({code}): {message}")
    name = type(exc).__name__.lower()
    module = type(exc).__module__ or ""
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return LLMTimeoutError(f"Request timed out: {type(exc).__name__}")
    if isinstance(exc, (ConnectionError, OSError)) or module.startswith(("httpx", "httpcore")):
        return LLMConnectionError(f"Network error: {type(exc).__name__}: {message}")
    return LLMError(f"Unexpected {type(exc).__name__}: {message}")


# ----------------------------------------------------------------------------------------------- Gemini
class GeminiClient:
    """One request to Gemini. No retries here (see retry.ResilientClient)."""

    def __init__(self, config: "LLMConfig"):
        if not config.api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Add it to .env (and never commit that file).")
        if not config.model:
            raise LLMConfigError("GEMINI_MODEL is not set. Copy the exact model id from Google AI Studio into .env.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMConfigError("The google-genai package is not installed: pip install google-genai") from exc
        self._types = types
        self._config = config
        self.model = config.model
        self._thinking = self._build_thinking(types, config.thinking_level)
        self._client = genai.Client(
            api_key=config.api_key,
            http_options=types.HttpOptions(timeout=int(config.timeout_seconds * 1000)),  # milliseconds
        )

    @staticmethod
    def _build_thinking(types: Any, level: str):
        """Optional: limit hidden reasoning (it costs time and tokens). Empty = the provider's default."""
        if not level:
            return None
        valid = [m.name for m in types.ThinkingLevel if m.name != "THINKING_LEVEL_UNSPECIFIED"]
        if level.upper() not in valid:
            raise LLMConfigError(f"LLM_THINKING_LEVEL must be empty or one of {valid}, got {level!r}")
        return types.ThinkingConfig(thinking_level=level.upper())

    def _request(self, system: str, user: str, extra: dict[str, Any]):
        """One request. Returns (response, latency in ms); SDK/network errors become our error types."""
        settings: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            **extra,
        }
        if self._thinking is not None:
            settings["thinking_config"] = self._thinking
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=self._types.GenerateContentConfig(**settings))
        except Exception as exc:  # noqa: BLE001 - mapped to our own types
            raise map_exception(exc) from exc
        return response, int((time.perf_counter() - started) * 1000)

    def _result(self, response: Any, latency_ms: int, text: str = "", **extra) -> LLMResult:
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(text=text, model=self.model, latency_ms=latency_ms, finish_reason=_finish_reason(response),
                         input_tokens=getattr(usage, "prompt_token_count", None),
                         output_tokens=getattr(usage, "candidates_token_count", None),
                         thinking_tokens=getattr(usage, "thoughts_token_count", None), **extra)

    @staticmethod
    def _text(response: Any) -> str:
        try:
            return response.text or ""
        except Exception:  # noqa: BLE001 - the SDK can raise when there is no usable candidate
            return ""

    def generate(self, system: str, user: str, schema: "type[BaseModel] | None" = None) -> LLMResult:
        extra: dict[str, Any] = {}
        if schema is not None:  # mime type and schema must be used together
            extra["response_mime_type"] = "application/json"
            extra["response_schema"] = schema
        response, latency_ms = self._request(system, user, extra)
        text = self._text(response)
        if not text.strip():
            raise LLMBlockedError(f"The model returned no text (finish_reason={_finish_reason(response)}, "
                                  f"block_reason={_block_reason(response)}).")
        return self._result(response, latency_ms, text)

    @staticmethod
    def _build_tool(types: Any, tools: list) -> Any:
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t.name, description=t.description, parameters=t.parameters) for t in tools])

    def choose_tool(self, system: str, user: str, tools: list) -> LLMResult:
        """Function-calling concept: the model only CHOOSES a tool and its arguments (result.tool_name / tool_args). Nothing is
        executed here and the SDK's automatic execution is off. If no tool fits it answers in words (result.text)."""
        types = self._types
        extra = {"tools": [self._build_tool(types, tools)],
                 "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
                 "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))}
        response, latency_ms = self._request(system, user, extra)
        calls = list(getattr(response, "function_calls", None) or [])
        if calls:
            return self._result(response, latency_ms, tool_name=calls[0].name, tool_args=dict(calls[0].args or {}))
        text = self._text(response)
        if not text.strip():
            raise LLMBlockedError(f"The model returned neither a tool call nor text (finish_reason={_finish_reason(response)}).")
        return self._result(response, latency_ms, text)


def _finish_reason(response: Any) -> str | None:
    try:
        reason = response.candidates[0].finish_reason
        return getattr(reason, "name", None) or (str(reason) if reason is not None else None)
    except Exception:  # noqa: BLE001
        return None


def _block_reason(response: Any) -> str | None:
    feedback = getattr(response, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None)
    return getattr(reason, "name", None) or (str(reason) if reason is not None else None)


# ----------------------------------------------------------------------------------------------- Fake (tests/offline)
class FakeLLMClient:
    """Scripted client. Each call consumes the next item: a string (returned as text), an LLMResult (returned as is),
    an Exception (raised), or a callable(system, user) -> str. When the script is used up, the last item repeats."""

    def __init__(self, script: list, model: str = "fake-model"):
        if not script:
            raise ValueError("FakeLLMClient needs at least one scripted item")
        self.script = list(script)
        self.model = model
        self.calls: list[dict] = []

    def generate(self, system: str, user: str, schema: "type[BaseModel] | None" = None) -> LLMResult:
        self.calls.append({"system": system, "user": user, "schema": schema})
        item = self.script[min(len(self.calls), len(self.script)) - 1]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMResult):
            return item
        text = item(system, user) if callable(item) else item
        return LLMResult(text=text, model=self.model, latency_ms=1, finish_reason="STOP",
                         input_tokens=len(user) // 4, output_tokens=len(text) // 4)

    def choose_tool(self, system: str, user: str, tools: list) -> LLMResult:
        """Same script as generate(): an LLMResult with tool_name/tool_args is returned as is, a string means "no tool"."""
        return self.generate(system, user)


# ----------------------------------------------------------------------------------------------- factory
# One client per configuration: the RAG generator and the insight endpoints must share ONE rate limiter and one request count.
_CLIENT_FIELDS = ("provider", "model", "api_key", "temperature", "max_output_tokens", "timeout_seconds", "deadline_seconds",
                  "max_retries", "backoff_base_seconds", "max_rpm", "thinking_level")
_CLIENTS: dict[tuple, "LLMClient"] = {}
_CLIENTS_LOCK = threading.Lock()


def build_client(config: "LLMConfig") -> LLMClient:
    """Production client: Gemini + retries + timeout budget + throttle. The same configuration returns the SAME client.
    Raises LLMConfigError when the key/model/SDK is missing (callers fall back); failures are never cached."""
    from src.llm.week9.retry import ResilientClient  # local import: retry.py imports this module

    key = tuple(getattr(config, name, None) for name in _CLIENT_FIELDS)
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            _CLIENTS[key] = ResilientClient(GeminiClient(config), config)
        return _CLIENTS[key]


if __name__ == "__main__":  # smoke test with the real key:  python -m src.llm.week9.client "Say hi in one word"
    from src.llm.week9.config import get_llm_config, setup_llm_logging

    cfg = get_llm_config()
    setup_llm_logging(cfg)
    try:
        out = build_client(cfg).generate("You are a terse assistant.", " ".join(sys.argv[1:]) or "Say hi in one word.")
    except LLMError as exc:
        print(f"FAILED ({type(exc).__name__}, attempts={exc.attempts}): {exc}")
        sys.exit(1)
    print(f"{out.text.strip()}\n[model={out.model} attempts={out.attempts} latency_ms={out.total_latency_ms} "
          f"tokens in/out/thinking={out.input_tokens}/{out.output_tokens}/{out.thinking_tokens} finish={out.finish_reason}]")
