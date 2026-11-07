"""Stage 1: the LangChain model is built from the Week 9 settings, and provider errors map to the Week 9 error types."""
from pathlib import Path

import pytest
from google.genai import errors
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from src.llm.week9.client import (LLMConfigError, LLMConnectionError, LLMQuotaExhaustedError, LLMRateLimitError, LLMAuthError)
from src.llm.week9.config import LLMConfig
from src.llm.week10.model_factory import build_chat_model, to_llm_error, tool_declarations, unwrap_provider_error
from src.llm.week9.tool_specs import TOOLS


def _config(**changes) -> LLMConfig:
    values = dict(base_dir=Path("."), provider="gemini", api_key="test-key", model="gemini-test", temperature=0.1,
                  max_output_tokens=256, timeout_seconds=7.0, deadline_seconds=20.0, max_retries=2,
                  backoff_base_seconds=0.0, max_rpm=60, log_path=Path("llm.log"), thinking_level="")
    values.update(changes)
    return LLMConfig(**values)


def _wrapped(code: int, body: dict) -> Exception:
    """What langchain-google-genai raises: its own error, with the real google.genai error as the cause."""
    try:
        try:
            raise errors.ClientError(code, body)
        except errors.ClientError as original:
            raise ChatGoogleGenerativeAIError(f"Error calling model ({original.status}): {original}") from original
    except ChatGoogleGenerativeAIError as exc:
        return exc


def test_the_model_uses_the_week9_settings_and_never_retries_by_itself():
    model = build_chat_model(_config())
    assert model.model == "gemini-test"
    assert model.max_retries == 0          # the package default is 6: ONE retry layer, ours
    assert model.timeout == 7.0
    assert model.max_output_tokens == 256


@pytest.mark.parametrize("changes", [{"api_key": ""}, {"model": ""}, {"thinking_level": "not-a-level"}])
def test_configuration_problems_are_config_errors(changes):
    with pytest.raises(LLMConfigError):
        build_chat_model(_config(**changes))


def test_tool_declarations_are_the_week9_specs():
    declared = tool_declarations()
    assert [d["name"] for d in declared] == [t.name for t in TOOLS]
    assert all(d["parameters"] == t.parameters for d, t in zip(declared, TOOLS))
    model = build_chat_model(_config()).bind_tools(declared)
    assert len(model.kwargs["tools"]) == len(TOOLS)


def test_a_per_minute_limit_is_retryable_and_keeps_the_providers_delay():
    body = {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}
    mapped = to_llm_error(_wrapped(429, body))
    assert isinstance(mapped, LLMRateLimitError) and mapped.retryable and mapped.retry_after == 12.0


def test_a_daily_quota_is_not_retried():
    body = {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel", "quotaMetric": "m"}]}]}}
    mapped = to_llm_error(_wrapped(429, body))
    assert isinstance(mapped, LLMQuotaExhaustedError) and not mapped.retryable


def test_unknown_model_and_bad_key_are_not_retryable():
    not_found = to_llm_error(_wrapped(404, {"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}}))
    bad_key = to_llm_error(_wrapped(400, {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}))
    assert isinstance(not_found, LLMConfigError) and not not_found.retryable
    assert isinstance(bad_key, LLMAuthError) and not bad_key.retryable


def test_without_unwrapping_the_status_would_be_lost():
    exc = _wrapped(429, {"error": {"code": 429, "message": "x", "status": "RESOURCE_EXHAUSTED"}})
    assert not isinstance(getattr(exc, "code", None), int)       # the wrapper has no status code of its own...
    assert unwrap_provider_error(exc).code == 429                # ...the original error has


def test_network_errors_pass_through_and_are_retryable():
    mapped = to_llm_error(ConnectionError("no route"))
    assert isinstance(mapped, LLMConnectionError) and mapped.retryable
