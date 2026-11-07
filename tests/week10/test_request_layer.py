"""Stage 5: one request layer for every model call: shared throttle, retries, time budget, request budget. No network, no key."""
from pathlib import Path

import pytest
from google.genai import errors
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from src.llm.week9.client import LLMConfigError, LLMError, LLMQuotaExhaustedError, LLMRateLimitError
from src.llm.week9.config import LLMConfig
from src.llm.week9.retry import RateLimiter
from src.llm.week9.tool_specs import ToolRuntime
from src.llm.week10 import request_layer
from src.llm.week10.assistant import Assistant
from src.llm.week10.request_layer import LangChainRequestLayer, RequestCounter, shared_limiter
from src.llm.week10.tool_logging import ToolLog
from src.llm.week10.tools import build_tools


def _config(**changes) -> LLMConfig:
    values = dict(base_dir=Path("."), provider="gemini", api_key="secret-key-123", model="gemini-test", temperature=0.1,
                  max_output_tokens=256, timeout_seconds=7.0, deadline_seconds=100.0, max_retries=2, backoff_base_seconds=1.0,
                  max_rpm=60, log_path=Path("llm.log"), thinking_level="")
    values.update(changes)
    return LLMConfig(**values)


class Clock:
    """Time that only moves when something sleeps."""

    def __init__(self):
        self.now, self.slept = 0.0, []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class CountingLimiter:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def wrapped(code: int, body: dict) -> Exception:
    """What langchain-google-genai raises: its own error, caused by the real provider error."""
    try:
        try:
            raise errors.ClientError(code, body)
        except errors.ClientError as original:
            raise ChatGoogleGenerativeAIError(f"Error calling model ({original.status}): {original}") from original
    except ChatGoogleGenerativeAIError as exc:
        return exc


PER_MINUTE = {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}
NOT_FOUND = {"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}}
OK = AIMessage(content="ok", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})


def make_layer(config=None, counter=None, limiter=None, log=None):
    clock = Clock()
    layer = LangChainRequestLayer(config or _config(), limiter=limiter or CountingLimiter(), counter=counter, log=log,
                                  sleep=clock.sleep, clock=clock.time, jitter=lambda: 0.5)   # jitter 0.5 -> backoff = base * 2^(n-1)
    return layer, clock


def script(*steps):
    """A function that raises/returns the next step each time it is called."""
    steps, calls = list(steps), []

    def fn():
        calls.append(1)
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
    fn.calls = calls
    return fn


# ----------------------------------------------------------------------------------------------- retries
def test_a_first_try_success_costs_one_request_and_one_throttle_wait():
    layer, clock = make_layer()
    assert layer.call(script(OK)) is OK
    assert layer.requests == 1 and layer.limiter.waits == 1 and clock.slept == []


def test_a_per_minute_limit_is_retried_after_the_providers_delay_and_every_attempt_is_throttled():
    layer, clock = make_layer()
    fn = script(wrapped(429, PER_MINUTE), OK)
    assert layer.call(fn) is OK
    assert clock.slept == [12.0]                    # the provider's own retry-after, not our backoff
    assert layer.requests == 2 and layer.limiter.waits == 2 and len(fn.calls) == 2


def test_a_network_error_is_retried_with_exponential_backoff():
    layer, clock = make_layer()
    assert layer.call(script(ConnectionError("down"), ConnectionError("down"), OK)) is OK
    assert clock.slept == [1.0, 2.0] and layer.requests == 3


def test_a_wrong_model_is_not_retried():
    layer, clock = make_layer()
    fn = script(wrapped(404, NOT_FOUND), OK)
    with pytest.raises(LLMConfigError) as caught:
        layer.call(fn)
    assert caught.value.attempts == 1 and len(fn.calls) == 1 and clock.slept == [] and layer.requests == 1


def test_when_retries_run_out_the_last_error_is_raised_with_the_attempt_count():
    layer, _ = make_layer(_config(max_retries=2))
    fn = script(wrapped(429, PER_MINUTE), wrapped(429, PER_MINUTE), wrapped(429, PER_MINUTE), OK)
    with pytest.raises(LLMRateLimitError) as caught:
        layer.call(fn)
    assert caught.value.attempts == 3 and len(fn.calls) == 3          # first try + 2 retries, never the fourth


def test_it_gives_up_instead_of_sleeping_past_the_time_budget():
    layer, clock = make_layer(_config(deadline_seconds=10.0))
    with pytest.raises(LLMRateLimitError):
        layer.call(script(wrapped(429, PER_MINUTE), OK))             # the provider asks 12 s, the budget is 10 s
    assert clock.slept == [] and layer.requests == 1


# ----------------------------------------------------------------------------------------------- budget, secrets
def test_the_request_budget_stops_a_run_before_the_limit_request_is_sent():
    layer, _ = make_layer(counter=RequestCounter(limit=2))
    fn = script(OK, OK, OK)
    layer.call(fn); layer.call(fn)
    with pytest.raises(LLMQuotaExhaustedError, match="local request budget"):
        layer.call(fn)
    assert len(fn.calls) == 2 and layer.requests == 2 and layer.limiter.waits == 2


def test_retries_count_against_the_budget():
    layer, _ = make_layer(counter=RequestCounter(limit=2))
    with pytest.raises(LLMQuotaExhaustedError):
        layer.call(script(wrapped(429, PER_MINUTE), wrapped(429, PER_MINUTE), OK))
    assert layer.requests == 2


def test_a_budget_must_be_positive():
    with pytest.raises(ValueError):
        RequestCounter(limit=0)


def test_the_api_key_never_survives_in_an_error_message():
    layer, _ = make_layer()
    with pytest.raises(LLMError) as caught:
        layer.call(script(wrapped(404, {"error": {"code": 404, "message": "bad url ?key=secret-key-123", "status": "NOT_FOUND"}})))
    assert "secret-key-123" not in str(caught.value) and "***" in str(caught.value)


def test_failed_attempts_appear_in_the_tool_log(tmp_path):
    import logging
    from src.llm.week10.tool_logging import LOGGER_NAME, setup_tool_log
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    path = tmp_path / "tool_calls.log"
    setup_tool_log(path)
    layer, _ = make_layer(log=ToolLog())
    layer.call(script(wrapped(429, PER_MINUTE), OK))
    for handler in list(logger.handlers):
        handler.close(); logger.removeHandler(handler)
    text = path.read_text(encoding="utf-8")
    assert "REQUEST" in text and "attempt 1 failed: LLMRateLimitError retryable=True, provider asks to wait 12s" in text
    assert "secret-key-123" not in text


# ----------------------------------------------------------------------------------------------- guard + sharing
def test_guard_wraps_any_runnable_inside_a_chain():
    layer, _ = make_layer()
    flaky = script(ConnectionError("down"), OK)
    guarded = layer.guard(RunnableLambda(lambda value: flaky()))
    chain = ChatPromptTemplate.from_messages([("human", "{question}")]) | guarded
    assert chain.invoke({"question": "hi"}) is OK
    assert layer.requests == 2


def test_guard_raises_week9_errors_for_provider_failures():
    layer, _ = make_layer()
    guarded = layer.guard(RunnableLambda(lambda value: script(wrapped(404, NOT_FOUND))()))
    with pytest.raises(LLMConfigError):
        guarded.invoke("hi")


def test_the_real_rate_limiter_spaces_attempts_by_the_configured_rate():
    clock = Clock()
    limiter = RateLimiter(max_rpm=30, clock=clock.time, sleep=clock.sleep)    # 2 s between requests
    layer = LangChainRequestLayer(_config(max_rpm=30), limiter=limiter, sleep=clock.sleep, clock=clock.time, jitter=lambda: 0.5)
    layer.call(script(OK)); layer.call(script(OK)); layer.call(script(OK))
    assert clock.slept == [2.0, 2.0] and layer.requests == 3


def test_the_week9_clients_limiter_is_shared_when_a_client_can_be_built(monkeypatch):
    mine = RateLimiter(10)
    monkeypatch.setattr(request_layer, "build_client", lambda config: type("Client", (), {"limiter": mine})())
    assert shared_limiter(_config()) is mine


def test_without_a_client_a_private_limiter_with_the_same_rate_is_used(monkeypatch):
    def refuse(config):
        raise LLMConfigError("GEMINI_API_KEY is not set")
    monkeypatch.setattr(request_layer, "build_client", refuse)
    limiter = shared_limiter(_config(max_rpm=20))
    assert isinstance(limiter, RateLimiter) and limiter.interval == 3.0


# ----------------------------------------------------------------------------------------------- the assistant behind the layer
def test_the_assistant_survives_a_transient_failure_and_counts_every_attempt(tmp_path):
    import sqlite3
    db = tmp_path / "retail.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE sales_clean (transaction_date TEXT, category TEXT, payment_method TEXT, location TEXT, total_spent REAL, quantity INTEGER)")
    con.execute("INSERT INTO sales_clean VALUES ('2024-01-05','Food','Cash','Online',200.0,2)")
    con.commit(); con.close()

    layer, clock = make_layer()
    choose_steps = script(wrapped(429, PER_MINUTE),
                          AIMessage(content="", tool_calls=[{"name": "get_sales_summary", "args": {}, "id": "1", "type": "tool_call"}],
                                    usage_metadata={"input_tokens": 500, "output_tokens": 12, "total_tokens": 512}))
    summary_steps = script(AIMessage(content="Total revenue is 200.0.", usage_metadata={"input_tokens": 900, "output_tokens": 9, "total_tokens": 909}))
    assistant = Assistant(layer.guard(RunnableLambda(lambda value: choose_steps())), layer.guard(RunnableLambda(lambda value: summary_steps())),
                          build_tools(ToolRuntime(retail_db_path=db)), ToolLog(), layer=layer)
    answer = assistant.ask("Total revenue?")
    assert answer.text == "Total revenue is 200.0." and answer.model_requests == 2      # two model CALLS...
    assert answer.attempts == 3 and layer.requests == 3 and clock.slept == [12.0]        # ...three requests SENT, one wait


def test_a_provider_that_keeps_failing_gives_an_unavailable_answer(tmp_path):
    layer, _ = make_layer(_config(max_retries=1))
    always = script(wrapped(429, PER_MINUTE), wrapped(429, PER_MINUTE))
    assistant = Assistant(layer.guard(RunnableLambda(lambda value: always())), None, [], ToolLog(), layer=layer)
    answer = assistant.ask("Which category leads?")
    assert answer.route == "unavailable" and "LLMRateLimitError" in answer.error
    assert answer.model_requests == 1 and answer.attempts == 2 == layer.requests   # one call, two requests sent


# ----------------------------------------------------------------------------------------------- counting: calls vs requests sent
RATE, NET = wrapped(429, PER_MINUTE), ConnectionError("down")
CHOICE = AIMessage(content="", tool_calls=[{"name": "get_sales_summary", "args": {}, "id": "1", "type": "tool_call"}],
                   usage_metadata={"input_tokens": 500, "output_tokens": 12, "total_tokens": 512})


def _says(text):
    return AIMessage(content=text, usage_metadata={"input_tokens": 900, "output_tokens": 9, "total_tokens": 909})


@pytest.fixture
def retail_db(tmp_path):
    import sqlite3
    path = tmp_path / "retail.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sales_clean (transaction_date TEXT, category TEXT, payment_method TEXT, location TEXT, total_spent REAL, quantity INTEGER)")
    con.execute("INSERT INTO sales_clean VALUES ('2024-01-05','Food','Cash','Online',200.0,2)")
    con.commit(); con.close()
    return path


def _assistant(layer, retail_db, choose, summarize):
    c, s = script(*choose), script(*summarize)
    return Assistant(layer.guard(RunnableLambda(lambda v: c())), layer.guard(RunnableLambda(lambda v: s())),
                     build_tools(ToolRuntime(retail_db_path=retail_db)), ToolLog(), layer=layer)


@pytest.mark.parametrize("name, choose, summarize, route, calls, sent", [
    ("everything first try",                          [CHOICE],      [_says("Total revenue is 200.0.")],                                   "tool",        2, 2),
    ("the choice needs one retry",                    [RATE, CHOICE], [_says("Total revenue is 200.0.")],                                  "tool",        2, 3),
    ("the summary fails on every attempt",            [CHOICE],      [NET, NET, NET],                                                      "tool",        2, 4),
    ("the choice fails on every attempt",             [NET, NET, NET], [],                                                                 "unavailable", 1, 3),
    ("a rejected answer, then a repair that retries", [CHOICE],      [_says("Revenue is 9999.0."), RATE, _says("Revenue is 200.0.")],      "tool",        3, 4),
])
def test_calls_and_requests_sent_are_counted_separately(retail_db, name, choose, summarize, route, calls, sent):
    layer, _ = make_layer()
    answer = _assistant(layer, retail_db, choose, summarize).ask("Total revenue?")
    assert answer.route == route
    assert (answer.model_requests, answer.attempts) == (calls, sent), name
    assert answer.attempts == layer.requests          # one question in this layer: the answer's number is the layer's number
    assert answer.attempts >= answer.model_requests   # a call is sent at least once; retries only add


def test_an_answers_attempts_exclude_the_questions_before_it(retail_db):
    layer, _ = make_layer()
    assistant = _assistant(layer, retail_db, [CHOICE, RATE, CHOICE], [_says("Total revenue is 200.0."), _says("Revenue is 200.0.")])
    first, second = assistant.ask("Total revenue?"), assistant.ask("Revenue again?")
    assert (first.attempts, second.attempts) == (2, 3)
    assert layer.requests == 5 == first.attempts + second.attempts    # the layer counter is cumulative; the answers are not
