"""Week 9 LLM settings, read from environment / .env (same pattern as src/rag/week8/config.py)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# src/llm/week9/config.py -> project root is three levels up
BASE_DIR = Path(__file__).resolve().parents[3]

VALID_PROVIDERS = {"gemini"}
LOGGER_NAME = "src.llm.week9"


@dataclass(frozen=True)
class LLMConfig:
    base_dir: Path
    provider: str
    api_key: str = field(repr=False)  # never printed or logged
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float   # one request
    deadline_seconds: float  # one call including every retry
    max_retries: int         # retries AFTER the first attempt
    backoff_base_seconds: float
    max_rpm: int
    log_path: Path
    thinking_level: str = ""  # "" = provider default; else MINIMAL | LOW | MEDIUM | HIGH

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def _read(name: str, default: str, cast, low=None, high=None):
    raw = os.getenv(name, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from None
    if low is not None and value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high}, got {value}")
    return value


def get_llm_config() -> LLMConfig:
    """A missing API key or model is NOT an error here: the application must still start (extractive mode).
    It becomes an LLMConfigError only when a client is actually built."""
    load_dotenv(BASE_DIR / ".env")  # never overrides variables already set in the environment

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got {provider!r}")

    timeout = _read("LLM_TIMEOUT_SECONDS", "45", float, low=0.1)
    deadline = _read("LLM_DEADLINE_SECONDS", "100", float, low=0.1)
    if deadline < timeout:
        raise ValueError(f"LLM_DEADLINE_SECONDS ({deadline}) must be >= LLM_TIMEOUT_SECONDS ({timeout})")

    return LLMConfig(
        base_dir=BASE_DIR,
        provider=provider,
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        model=os.getenv("GEMINI_MODEL", "").strip(),
        temperature=_read("LLM_TEMPERATURE", "0.1", float, low=0.0, high=2.0),
        max_output_tokens=_read("LLM_MAX_OUTPUT_TOKENS", "1024", int, low=1),
        timeout_seconds=timeout,
        deadline_seconds=deadline,
        max_retries=_read("LLM_MAX_RETRIES", "2", int, low=0, high=5),
        backoff_base_seconds=_read("LLM_BACKOFF_BASE_SECONDS", "1.0", float, low=0.0),
        max_rpm=_read("LLM_MAX_RPM", "10", int, low=1),
        log_path=BASE_DIR / "logs" / "week9" / "llm.log",
        thinking_level=os.getenv("LLM_THINKING_LEVEL", "").strip().lower(),
    )


def setup_llm_logging(config: LLMConfig) -> None:
    """Send every src.llm.week9.* log record to logs/week9/llm.log (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    target = str(config.log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


def get_retail_db_path() -> Path:
    """The cleaned retail data written by the Week 4 pipeline (read by the sales insights)."""
    load_dotenv(BASE_DIR / ".env")
    path = Path(os.getenv("RETAIL_DB_PATH", "data/db/retail_pipeline.db"))
    return path if path.is_absolute() else BASE_DIR / path
