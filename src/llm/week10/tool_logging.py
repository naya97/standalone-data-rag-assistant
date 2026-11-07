"""
  2026-10-31 14:02:11 | q-3fa9 | QUESTION  Which category has the highest revenue?
  2026-10-31 14:02:13 | q-3fa9 | MODEL     chose get_sales_summary args={} (tokens 499/12, 2.4s)
  2026-10-31 14:02:13 | q-3fa9 | TOOL      get_sales_summary args={} ... running
  2026-10-31 14:02:13 | q-3fa9 | RESULT    get_sales_summary ok in 41 ms, 8.1 KB: sales 2024-01..2024-06, revenue 512340.5
  2026-10-31 14:02:16 | q-3fa9 | SUMMARY   source=model requests=1 tokens=2600/80 citations=[]
  2026-10-31 14:02:16 | q-3fa9 | ANSWER    Furniture has the highest revenue ...

`ToolLog` is a LangChain callback handler: passed in `config={"callbacks": [log]}` it writes the TOOL / RESULT lines by itself
when a tool starts, ends or fails; the assistant adds the other steps. Long texts are cut. Nothing secret is ever passed here
(no key, no full passages): results are logged as a size and a one-line description.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

LOGGER_NAME = "src.llm.week10.tool_log"
MAX_TEXT_CHARS = 300


def default_log_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "logs" / "week10" / "tool_calls.log"


def setup_tool_log(log_path: str | Path, echo: bool = False) -> logging.Logger:
    """File handler (always) and a stderr echo (optional, `--verbose`). Idempotent: calling it again adds nothing twice."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # not mixed into the Week 9 llm.log or the root logger
    target = str(Path(log_path).resolve())
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in logger.handlers):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    if echo and not any(getattr(h, "_week10_echo", False) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(message)s"))
        stream._week10_echo = True  # type: ignore[attr-defined]
        logger.addHandler(stream)
    return logger


def _cut(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS] + "..."


def describe_result(result: Any) -> str:
    """One line about a tool result (never the passages themselves)."""
    if not isinstance(result, dict):
        return type(result).__name__
    if "error" in result:
        return f"error: {_cut(result['error'])}"
    if "passages" in result:
        sources = sorted({p["source"] for p in result["passages"]})
        return f"{len(result['passages'])} passages from {sources}" + (f" ({result['note']})" if result.get("note") else "")
    if "overall" in result:
        scope, overall = result["scope"], result["overall"]
        what = scope.get("category") or "all categories"
        return (f"sales {scope.get('first_month')}..{scope.get('last_month')}, {what}: "
                f"{overall['transactions']} transactions, revenue {overall['total_revenue']}")
    return _cut(result)


class ToolLog(BaseCallbackHandler):
    raise_error = False  # a logging problem must never break an answer

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.question_id = "q-none"
        self._started: dict[Any, tuple[str, float]] = {}

    # --- steps written by the assistant
    def start(self, question: str) -> str:
        self.question_id = f"q-{uuid.uuid4().hex[:4]}"
        self.step("QUESTION", question)
        return self.question_id

    def step(self, stage: str, text: Any) -> None:
        self.logger.info("%s | %-9s %s", self.question_id, stage, _cut(text))

    # --- LangChain callbacks: written by the tool run itself
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: Any = None, inputs: dict | None = None, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._started[run_id] = (name, time.perf_counter())
        self.step("TOOL", f"{name} args={_cut(inputs if inputs is not None else input_str)} ... running")

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        name, started = self._started.pop(run_id, ("tool", time.perf_counter()))
        ms = int((time.perf_counter() - started) * 1000)
        failed = isinstance(output, dict) and "error" in output
        size = len(json.dumps(output, ensure_ascii=False, default=str)) / 1024
        self.step("RESULT", f"{name} {'FAILED' if failed else 'ok'} in {ms} ms, {size:.1f} KB: {describe_result(output)}")

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        name, started = self._started.pop(run_id, ("tool", time.perf_counter()))
        self.step("RESULT", f"{name} FAILED in {int((time.perf_counter() - started) * 1000)} ms: {type(error).__name__}: {_cut(str(error))}")
