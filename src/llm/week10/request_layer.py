from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from src.llm.week9.client import LLMError, LLMQuotaExhaustedError, build_client
from src.llm.week9.config import LLMConfig
from src.llm.week9.retry import RateLimiter, call_with_retry
from src.llm.week10.model_factory import to_llm_error
from src.llm.week10.tool_logging import ToolLog

logger = logging.getLogger("src.llm.week9.langchain")  # under src.llm.week9: lands in logs/week9/llm.log with the other attempts


class RequestCounter:
    """Counts model requests (each attempt, retries included). With a `limit`, the request after the last allowed one is
    refused BEFORE it is sent."""

    def __init__(self, limit: int | None = None):
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit, self.count = limit, 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.limit is not None and self.count >= self.limit:
                raise LLMQuotaExhaustedError(f"local request budget reached: {self.limit} requests already sent in this run")
            self.count += 1


def shared_limiter(config: LLMConfig) -> RateLimiter:
    """The Week 9 client's limiter when a client can be built for this configuration, else a private one with the same rate."""
    try:
        limiter = getattr(build_client(config), "limiter", None)
    except LLMError:
        limiter = None
    return limiter or RateLimiter(config.max_rpm)


class LangChainRequestLayer:
    def __init__(self, config: LLMConfig, limiter: RateLimiter | None = None, counter: RequestCounter | None = None,
                 log: ToolLog | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, jitter: Callable[[], float] = random.random):
        self.config = config
        self.limiter = limiter or shared_limiter(config)
        self.counter = counter or RequestCounter()
        self.log = log
        self._sleep, self._clock, self._jitter = sleep, clock, jitter

    @property
    def requests(self) -> int:
        return self.counter.count

    def call(self, fn: Callable[[], AIMessage]) -> AIMessage:
        """Run one model call: budget -> throttle -> request -> retry transient errors. Raises an LLMError when it gives up
        (`.attempts` tells how many requests were sent)."""
        def attempt() -> AIMessage:
            try:
                return fn()
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - whatever the provider/network raised -> a Week 9 error type
                raise to_llm_error(exc) from exc

        def before() -> None:
            self.counter.take()
            self.limiter.wait()

        try:
            message, _ = call_with_retry(attempt, max_retries=self.config.max_retries, backoff_base=self.config.backoff_base_seconds,
                                         deadline=self.config.deadline_seconds, sleep=self._sleep, clock=self._clock,
                                         jitter=self._jitter, before_attempt=before, on_attempt=self._observe)
        except LLMError as exc:
            exc.args = (self._redact(str(exc)),)  # this message reaches logs and API responses: never the key
            raise
        return message

    def guard(self, runnable: Runnable) -> Runnable:
        """The same runnable, with every invocation going through `call`."""
        return RunnableLambda(lambda value: self.call(lambda: runnable.invoke(value)), name="guarded_model_call")

    # ------------------------------------------------------------------------------------------- logging
    def _redact(self, message: str) -> str:
        return message.replace(self.config.api_key, "***") if self.config.api_key else message

    def _observe(self, attempt: int, result: AIMessage | None, error: LLMError | None) -> None:
        if error is None:
            usage = getattr(result, "usage_metadata", None) or {}
            logger.info("langchain llm ok attempt=%d model=%s tokens in/out=%s/%s", attempt, self.config.model,
                        usage.get("input_tokens"), usage.get("output_tokens"))
            return
        message = self._redact(str(error))[:300]
        logger.warning("langchain llm error attempt=%d model=%s type=%s retryable=%s msg=%s", attempt, self.config.model,
                       type(error).__name__, error.retryable, message)
        if self.log is not None:
            wait = f", provider asks to wait {error.retry_after:g}s" if error.retry_after else ""
            self.log.step("REQUEST", f"attempt {attempt} failed: {type(error).__name__} retryable={error.retryable}{wait}")
