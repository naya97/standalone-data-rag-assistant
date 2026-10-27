"""LLM answer generator for the RAG service (implements the Week 8 AnswerGenerator interface).

    chunks -> build_messages (system + user) -> Gemini (JSON schema) -> parse -> guardrails
                                                   |                          |
                                             transient errors          invalid / unsupported answer
                                             retried by the client     -> ONE repair request with the reasons
    anything still wrong -> ExtractiveGenerator answer, flagged mode="extractive_fallback" + fallback_reason

Whatever happens, ask() still returns an answer; `generation` in the response says how it was produced.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from src.llm.week9.client import (
    MAX_ERROR_CHARS, LLMClient, LLMConfigError, LLMError, LLMInvalidOutputError, LLMResult, build_client,
)
from src.llm.week9.config import LLMConfig, get_llm_config, setup_llm_logging
from src.llm.week9.guardrails import check_rag_answer, validate_output
from src.llm.week9.schemas import RagAnswer
from src.rag.week8.answer_generator import AnswerGenerator, ExtractiveGenerator, GeneratedAnswer
from src.rag.week8.config import RagConfig
from src.rag.week8.prompt_builder import NO_ANSWER, RAG_PROMPT_VERSION, build_messages, detect_language

if TYPE_CHECKING:  # pragma: no cover
    from src.rag.week8.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(self, client: LLMClient | None, fallback: AnswerGenerator, *, max_repairs: int = 1,
                 unavailable_reason: str | None = None):
        """client=None means the LLM is unavailable (e.g. no API key): every answer goes to the fallback and
        carries `unavailable_reason`, so the API keeps working exactly like Week 8."""
        self.client = client
        self.fallback = fallback
        self.max_repairs = max_repairs
        self.unavailable_reason = unavailable_reason or "LLM client not configured"

    def generate(self, question: str, chunks: list["RetrievedChunk"], prompt: str) -> GeneratedAnswer:
        info: dict = {"model": getattr(self.client, "model", None), "prompt_version": RAG_PROMPT_VERSION,
                      "attempts": 0, "repairs": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
                      "model_latency_ms": 0}
        if self.client is None:
            return self._fall_back(question, chunks, prompt, info, self.unavailable_reason)

        system, user = build_messages(question, chunks)
        texts = [c.text for c in chunks]
        message = user
        problems: list[str] = []
        for round_no in range(self.max_repairs + 1):
            sent = f"### SYSTEM\n{system}\n\n### USER\n{message}"
            try:
                result = self.client.generate(system, message, RagAnswer)
            except LLMError as exc:  # retries for transient errors already happened inside the client
                info["attempts"] += exc.attempts
                return self._fall_back(question, chunks, prompt, info, f"{type(exc).__name__}: {exc}", sent)
            self._add_usage(info, result)

            answer, problems = self._review(result, texts)
            if not problems:
                return self._accept(question, answer, info, sent)
            if round_no < self.max_repairs:
                info["repairs"] += 1
                logger.info("llm answer rejected (%s); asking for one repair", "; ".join(problems))
                message = self._repair_message(user, result.text, problems)

        return self._fall_back(question, chunks, prompt, info, "guardrails: " + "; ".join(problems), sent)

    # ------------------------------------------------------------------------------------------- steps
    @staticmethod
    def _review(result: LLMResult, texts: list[str]) -> tuple[RagAnswer | None, list[str]]:
        try:
            answer = validate_output(result.text, RagAnswer)
        except LLMInvalidOutputError as exc:
            cut = " (the output was cut off at the token limit)" if result.finish_reason == "MAX_TOKENS" else ""
            return None, [str(exc) + cut]
        return answer, check_rag_answer(answer, texts)

    @staticmethod
    def _accept(question: str, answer: RagAnswer, info: dict, sent: str) -> GeneratedAnswer:
        if answer.status == "not_found":
            return GeneratedAnswer(text=NO_ANSWER[detect_language(question)], cited_ids=[], not_found=True,
                                   mode="llm", meta=info, prompt_sent=sent)
        return GeneratedAnswer(text=answer.answer.strip(), cited_ids=sorted(set(answer.citations)), mode="llm",
                               meta=info, prompt_sent=sent)

    def _fall_back(self, question, chunks, prompt, info: dict, reason: str, sent: str | None = None) -> GeneratedAnswer:
        logger.warning("llm fallback to extractive answer: %s", reason)
        base = self.fallback.generate(question, chunks, prompt)
        return replace(base, mode="extractive_fallback", fallback_reason=reason[:MAX_ERROR_CHARS], meta=info,
                       prompt_sent=sent)

    @staticmethod
    def _repair_message(user: str, previous: str, problems: list[str]) -> str:
        listed = "\n".join(f"- {p}" for p in problems)
        return (f"{user}\n\nYour previous reply was rejected:\n{listed}\n\nPrevious reply:\n{previous[:800]}\n\n"
                "Return a corrected JSON object only, following every rule.")

    @staticmethod
    def _add_usage(info: dict, result: LLMResult) -> None:
        info["attempts"] += result.attempts
        info["input_tokens"] += result.input_tokens or 0
        info["output_tokens"] += result.output_tokens or 0
        info["thinking_tokens"] += result.thinking_tokens or 0
        info["model_latency_ms"] += result.total_latency_ms or result.latency_ms or 0  # the model's own time, even if cached


def build_answer_generator(config: RagConfig, llm_config: LLMConfig | None = None,
                           client: LLMClient | None = None) -> AnswerGenerator:
    """The generator the service should use for `config.answer_mode`.
    A missing key/model/SDK never stops the application: it answers in extractive mode and says why."""
    extractive = ExtractiveGenerator(support_ratio=config.answer_support_ratio)
    if config.answer_mode == "extractive":
        return extractive
    llm_config = llm_config or get_llm_config()  # ValueError on an invalid .env: fail loudly
    setup_llm_logging(llm_config)
    if client is None:
        try:
            client = build_client(llm_config)
        except LLMConfigError as exc:
            logger.warning("LLM unavailable, answering in extractive mode: %s", exc)
            return LLMGenerator(None, extractive, unavailable_reason=f"{type(exc).__name__}: {exc}")
    return LLMGenerator(client, extractive)
