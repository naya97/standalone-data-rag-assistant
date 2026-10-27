"""Week 9 / Phase 1: error mapping, retries, deadline, throttle, config. No network, no API key needed.
Run from the project root:  python -m pytest tests/week9 -v
"""

import dataclasses

import pytest

from src.llm.week9.client import (
    MAX_ERROR_CHARS, FakeLLMClient, LLMAuthError, LLMBlockedError, LLMConfigError, LLMConnectionError, LLMError,
    LLMQuotaExhaustedError, LLMRateLimitError, LLMRequestError, LLMResult, LLMServerError, LLMTimeoutError,
    map_exception,
)
from src.llm.week9.config import get_llm_config
from src.llm.week9.retry import RateLimiter, ResilientClient, call_with_retry

OK = '{"status": "answered"}'


class FakeClock:
    """sleep() advances the clock, so waits cost no real time and can be measured."""
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def cfg(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "LLM_TEMPERATURE", "LLM_MAX_OUTPUT_TOKENS",
                 "LLM_TIMEOUT_SECONDS", "LLM_DEADLINE_SECONDS", "LLM_MAX_RETRIES", "LLM_BACKOFF_BASE_SECONDS",
                 "LLM_MAX_RPM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("src.llm.week9.config.load_dotenv", lambda *a, **k: None)  # ignore a real .env
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    monkeypatch.setenv("LLM_MAX_RPM", "600")  # 0.1 s spacing: the throttle never dominates these tests
    return get_llm_config()


def resilient(cfg, script, **overrides):
    clock = FakeClock()
    config = dataclasses.replace(cfg, **overrides) if overrides else cfg
    inner = FakeLLMClient(script)
    client = ResilientClient(inner, config, sleep=clock.sleep, clock=clock.clock, jitter=lambda: 0.5)
    return client, inner, clock


# ----------------------------------------------------------------------------------------------- retries
def test_success_on_first_attempt(cfg):
    client, inner, clock = resilient(cfg, [OK])
    result = client.generate("sys", "user")
    assert result.text == OK and result.attempts == 1 and len(inner.calls) == 1


def test_timeout_then_success_is_retried(cfg):
    client, inner, clock = resilient(cfg, [LLMTimeoutError("slow"), OK])
    result = client.generate("sys", "user")
    assert result.attempts == 2 and len(inner.calls) == 2
    assert clock.sleeps[0] == pytest.approx(cfg.backoff_base_seconds)  # base * 2**0 * (0.5 + 0.5)


@pytest.mark.parametrize("error", [LLMAuthError("bad key"), LLMConfigError("no model"),
                                   LLMRequestError("bad request"), LLMBlockedError("blocked")])
def test_non_retryable_errors_fail_immediately(cfg, error):
    client, inner, clock = resilient(cfg, [error, OK])
    with pytest.raises(type(error)) as info:
        client.generate("sys", "user")
    assert len(inner.calls) == 1 and info.value.attempts == 1 and clock.sleeps == []


def test_retries_are_bounded(cfg):
    client, inner, _ = resilient(cfg, [LLMServerError("503")], max_retries=2)
    with pytest.raises(LLMServerError) as info:
        client.generate("sys", "user")
    assert len(inner.calls) == 3 and info.value.attempts == 3  # first attempt + 2 retries


def test_zero_retries_means_one_attempt(cfg):
    client, inner, _ = resilient(cfg, [LLMTimeoutError("slow")], max_retries=0)
    with pytest.raises(LLMTimeoutError):
        client.generate("sys", "user")
    assert len(inner.calls) == 1


def test_provider_retry_after_wins_over_backoff(cfg):
    client, _, clock = resilient(cfg, [LLMRateLimitError("429", retry_after=7.0), OK])
    client.generate("sys", "user")
    assert clock.sleeps[0] == 7.0


def test_backoff_doubles(cfg):
    client, _, clock = resilient(cfg, [LLMTimeoutError("a"), LLMTimeoutError("b"), OK], max_retries=2)
    client.generate("sys", "user")
    assert clock.sleeps[0] == pytest.approx(1.0) and clock.sleeps[1] == pytest.approx(2.0)


def test_deadline_stops_retrying_instead_of_sleeping_past_it(cfg):
    client, inner, clock = resilient(cfg, [LLMRateLimitError("429", retry_after=25.0)], max_retries=5,
                                     timeout_seconds=10.0, deadline_seconds=20.0)
    with pytest.raises(LLMRateLimitError):
        client.generate("sys", "user")
    assert len(inner.calls) == 1 and clock.sleeps == []


def test_retry_after_is_honoured_in_full_when_it_fits_the_budget(cfg):
    clock = FakeClock()
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise LLMRateLimitError("429", retry_after=37.0)
        return FakeLLMClient([OK]).generate("", "")

    call_with_retry(fn, max_retries=1, backoff_base=1.0, deadline=40.0, sleep=clock.sleep, clock=clock.clock)
    assert clock.sleeps == [37.0]  # the provider said 37 s: not cut to 30, which would fail again


def test_our_own_backoff_stays_capped():
    clock = FakeClock()
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 6:
            raise LLMTimeoutError("slow")
        return FakeLLMClient([OK]).generate("", "")

    call_with_retry(fn, max_retries=5, backoff_base=10.0, deadline=10_000, sleep=clock.sleep, clock=clock.clock,
                    jitter=lambda: 0.5)
    assert max(clock.sleeps) == 30.0


# ----------------------------------------------------------------------------------------------- throttle
def test_rate_limiter_spaces_requests():
    clock = FakeClock()
    limiter = RateLimiter(max_rpm=10, clock=clock.clock, sleep=clock.sleep)  # 6 s apart
    for _ in range(3):
        limiter.wait()
    assert clock.sleeps == pytest.approx([6.0, 6.0]) and clock.now == pytest.approx(12.0)


def test_rate_limiter_does_not_wait_when_calls_are_far_apart():
    clock = FakeClock()
    limiter = RateLimiter(max_rpm=10, clock=clock.clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 60
    limiter.wait()
    assert clock.sleeps == []


# ----------------------------------------------------------------------------------------------- error mapping
class _Api(Exception):  # shaped like google.genai.errors.APIError
    def __init__(self, code, message="boom"):
        super().__init__(message)
        self.code, self.message = code, message


class ReadTimeout(Exception):  # shaped like httpx.ReadTimeout
    pass


@pytest.mark.parametrize("exc, expected", [
    (_Api(401), LLMAuthError), (_Api(403), LLMAuthError),
    (_Api(400, "API key not valid. Please pass a valid API key."), LLMAuthError),
    (_Api(400, "Invalid JSON payload"), LLMRequestError),
    (_Api(404, "models/x is not found"), LLMConfigError),
    (_Api(429), LLMRateLimitError), (_Api(500), LLMServerError), (_Api(503), LLMServerError),
    (ReadTimeout("t"), LLMTimeoutError), (TimeoutError("t"), LLMTimeoutError),
    (ConnectionError("down"), LLMConnectionError),
])
def test_exception_mapping(exc, expected):
    assert type(map_exception(exc)) is expected


def test_unknown_exception_is_not_retryable():
    mapped = map_exception(RuntimeError("???"))
    assert type(mapped) is LLMError and not mapped.retryable


def test_retryable_flags():
    assert all(c.retryable for c in (LLMTimeoutError, LLMRateLimitError, LLMServerError, LLMConnectionError))
    assert not any(c.retryable for c in (LLMAuthError, LLMConfigError, LLMRequestError, LLMBlockedError))


# ----------------------------------------------------------------------------------------------- logging / config
def test_api_key_never_reaches_the_log(cfg, caplog):
    client, _, _ = resilient(cfg, [LLMAuthError("rejected key secret-key-123")])
    with caplog.at_level("INFO", logger="src.llm.week9"):
        with pytest.raises(LLMAuthError):
            client.generate("sys", "user")
    assert "secret-key-123" not in caplog.text and "***" in caplog.text


def test_config_repr_hides_the_key(cfg):
    assert "secret-key-123" not in repr(cfg)


def test_missing_key_is_not_a_config_error(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    config = get_llm_config()
    assert not config.has_api_key  # the app must still start


def test_gemini_client_refuses_missing_key_and_model(cfg, monkeypatch):
    from src.llm.week9.client import GeminiClient
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiClient(dataclasses.replace(cfg, api_key=""))
    with pytest.raises(LLMConfigError, match="GEMINI_MODEL"):
        GeminiClient(dataclasses.replace(cfg, model=""))


@pytest.mark.parametrize("name, value", [("LLM_TEMPERATURE", "3"), ("LLM_MAX_RETRIES", "9"), ("LLM_MAX_RPM", "0"),
                                         ("LLM_TIMEOUT_SECONDS", "abc"), ("LLM_PROVIDER", "openai")])
def test_invalid_config_values_are_rejected(cfg, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        get_llm_config()


def test_deadline_must_cover_one_timeout(cfg, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LLM_DEADLINE_SECONDS", "10")
    with pytest.raises(ValueError, match="LLM_DEADLINE_SECONDS"):
        get_llm_config()


def test_fake_client_repeats_last_item_and_records_calls():
    fake = FakeLLMClient(["a", "b"])
    assert [fake.generate("s", f"u{i}").text for i in range(3)] == ["a", "b", "b"]
    assert fake.calls[2]["user"] == "u2"


# ----------------------------------------------------------------------------------------------- quota (429)
def _429(message="Quota exceeded", details=None):
    body = {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED", "details": details or []}}
    pytest.importorskip("google.genai")
    from google.genai import errors
    return errors.ClientError(429, body)


RETRY_INFO = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}
DAILY = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaMetric": "x", "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}
PER_MINUTE = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
              "violations": [{"quotaMetric": "x", "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}


def test_per_minute_429_is_retryable_and_uses_the_providers_delay():
    mapped = map_exception(_429(details=[PER_MINUTE, RETRY_INFO]))
    assert type(mapped) is LLMRateLimitError and mapped.retryable and mapped.retry_after == 37.0


def test_retry_delay_can_come_from_the_message_text():
    mapped = map_exception(_429("You exceeded your current quota. Please retry in 12.5s."))
    assert type(mapped) is LLMRateLimitError and mapped.retry_after == 12.5


def test_plain_429_has_no_delay_and_falls_back_to_our_backoff():
    mapped = map_exception(_429("Too many requests"))
    assert type(mapped) is LLMRateLimitError and mapped.retry_after is None


@pytest.mark.parametrize("message, details", [
    ("You exceeded your current quota", [DAILY, RETRY_INFO]),   # daily limit, even though a retry delay is present
    ("Quota exceeded for metric x, limit: 0, model: m", []),    # this model has no quota on this key
])
def test_exhausted_quota_is_permanent_and_not_retried(cfg, message, details):
    mapped = map_exception(_429(message, details))
    assert type(mapped) is LLMQuotaExhaustedError and not mapped.retryable
    client, inner, clock = resilient(cfg, [mapped, OK])
    with pytest.raises(LLMQuotaExhaustedError) as info:
        client.generate("sys", "user")
    assert len(inner.calls) == 1 and info.value.attempts == 1 and clock.sleeps == []  # no 15-30 s wasted


def test_long_error_text_is_kept_up_to_the_limit():
    mapped = map_exception(_429("q" * 900 + " the end"))
    assert "the end" in str(mapped)
    assert len(str(map_exception(_429("z" * 5000)))) < MAX_ERROR_CHARS + 120


def test_the_key_is_removed_from_the_error_that_leaves_the_client(cfg):
    client, _, _ = resilient(cfg, [LLMAuthError("rejected key secret-key-123 for project")])
    with pytest.raises(LLMAuthError) as info:
        client.generate("sys", "user")
    assert "secret-key-123" not in str(info.value)


# ----------------------------------------------------------------------------------------------- thinking tokens / level
def test_thinking_tokens_travel_with_the_result():
    fake = FakeLLMClient([LLMResult(text=OK, input_tokens=800, output_tokens=40, thinking_tokens=500)])
    assert fake.generate("s", "u").thinking_tokens == 500


def test_thinking_level_is_validated_when_the_client_is_built(cfg):
    pytest.importorskip("google.genai")
    from src.llm.week9.client import GeminiClient
    assert GeminiClient(dataclasses.replace(cfg, thinking_level="low"))._thinking is not None
    assert GeminiClient(dataclasses.replace(cfg, thinking_level=""))._thinking is None
    with pytest.raises(LLMConfigError, match="LLM_THINKING_LEVEL"):
        GeminiClient(dataclasses.replace(cfg, thinking_level="turbo"))


def test_thinking_level_is_read_from_the_environment(cfg, monkeypatch):
    assert cfg.thinking_level == ""
    monkeypatch.setenv("LLM_THINKING_LEVEL", " LOW ")
    assert get_llm_config().thinking_level == "low"


# ----------------------------------------------------------------------------------------------- one shared client
def test_the_same_configuration_gives_the_same_client_so_the_quota_is_shared(cfg):
    pytest.importorskip("google.genai")
    from src.llm.week9.client import build_client
    first = build_client(cfg)
    assert build_client(cfg) is first
    assert build_client(dataclasses.replace(cfg, model="another-model")) is not first


def test_a_failed_build_is_not_cached(cfg):
    from src.llm.week9.client import build_client
    broken = dataclasses.replace(cfg, api_key="")
    for _ in range(2):
        with pytest.raises(LLMConfigError):
            build_client(broken)
