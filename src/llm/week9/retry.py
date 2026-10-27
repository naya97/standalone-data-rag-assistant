"""Retries, per-call time budget and client-side throttling around any LLMClient.

Rules (the ones the tests pin down):
  - only errors flagged `retryable` are retried (timeout, 429, 5xx, network); auth/config/request errors fail at once
  - at most `max_retries` retries after the first attempt, exponential backoff with jitter (provider's retry-after wins)
  - the whole call, including waits, stays inside `deadline_seconds`: we give up instead of sleeping past it
  - a RateLimiter spaces requests so the free-tier requests-per-minute limit is not hit by our own loops
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import TYPE_CHECKING, Callable

from src.llm.week9.client import MAX_ERROR_CHARS, LLMError, LLMResult

if TYPE_CHECKING:  # pragma: no cover
    from pydantic import BaseModel

    from src.llm.week9.client import LLMClient
    from src.llm.week9.config import LLMConfig

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30.0


class RateLimiter:
    """Minimum spacing of 60/max_rpm seconds between requests. Thread-safe: each caller reserves its own slot."""

    def __init__(self, max_rpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / max_rpm
        self._clock, self._sleep = clock, sleep
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.interval
        if slot > now:
            self._sleep(slot - now)


def call_with_retry(fn: Callable[[], LLMResult], *, max_retries: int, backoff_base: float, deadline: float,
                    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                    jitter: Callable[[], float] = random.random,
                    before_attempt: Callable[[], None] | None = None,
                    on_attempt: Callable[[int, LLMResult | None, LLMError | None], None] | None = None,
                    ) -> tuple[LLMResult, int]:
    """Returns (result, attempts). On failure raises the last LLMError with `.attempts` set."""
    started = clock()
    attempt = 0
    while True:
        attempt += 1
        if before_attempt:
            before_attempt()
        try:
            result = fn()
        except LLMError as exc:
            exc.attempts = attempt
            if on_attempt:
                on_attempt(attempt, None, exc)
            if not exc.retryable or attempt > max_retries:
                raise
            if exc.retry_after is not None:
                delay = exc.retry_after  # the provider knows its own window: honour it fully, the budget check decides
            else:
                delay = min(backoff_base * 2 ** (attempt - 1) * (0.5 + jitter()), MAX_BACKOFF_SECONDS)
            if clock() - started + delay >= deadline:
                logger.warning("llm giving up: waiting %.1fs would exceed the %.1fs budget", delay, deadline)
                raise
            sleep(delay)
            continue
        if on_attempt:
            on_attempt(attempt, result, None)
        return result, attempt


class ResilientClient:
    """LLMClient decorator: throttle -> request -> retry on transient errors -> log every attempt."""

    def __init__(self, inner: "LLMClient", config: "LLMConfig", limiter: RateLimiter | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 jitter: Callable[[], float] = random.random):
        self.inner = inner
        self.config = config
        self.model = inner.model
        self.limiter = limiter or RateLimiter(config.max_rpm, clock=clock, sleep=sleep)
        self._sleep, self._clock, self._jitter = sleep, clock, jitter

    def generate(self, system: str, user: str, schema: "type[BaseModel] | None" = None) -> LLMResult:
        return self._run(lambda: self.inner.generate(system, user, schema))

    def choose_tool(self, system: str, user: str, tools: list) -> LLMResult:
        return self._run(lambda: self.inner.choose_tool(system, user, tools))

    def _run(self, call) -> LLMResult:
        started = self._clock()
        try:
            result, attempts = call_with_retry(
                call,
                max_retries=self.config.max_retries, backoff_base=self.config.backoff_base_seconds,
                deadline=self.config.deadline_seconds, sleep=self._sleep, clock=self._clock, jitter=self._jitter,
                before_attempt=self.limiter.wait, on_attempt=self._log_attempt)
        except LLMError as exc:
            exc.args = (self._redact(str(exc)),)  # the message may reach an API response: no key in it
            raise
        result.attempts = attempts
        result.total_latency_ms = int((self._clock() - started) * 1000)
        return result

    def _redact(self, message: str) -> str:
        return message.replace(self.config.api_key, "***") if self.config.api_key else message

    def _log_attempt(self, attempt: int, result: LLMResult | None, error: LLMError | None) -> None:
        if error is None:
            logger.info("llm ok attempt=%d model=%s latency_ms=%d tokens in/out/thinking=%s/%s/%s finish=%s", attempt, self.model,
                        result.latency_ms, result.input_tokens, result.output_tokens, result.thinking_tokens,
                        result.finish_reason)
            return
        message = self._redact(str(error))  # a key must never reach the log
        logger.warning("llm error attempt=%d model=%s type=%s retryable=%s msg=%s", attempt, self.model,
                       type(error).__name__, error.retryable, message[:MAX_ERROR_CHARS])
