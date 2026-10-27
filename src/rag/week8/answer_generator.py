"""Turns retrieved chunks into an answer. The interface is what Week 9's LLM generator implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from src.rag.week8.retriever import RetrievedChunk


@dataclass
class GeneratedAnswer:
    text: str
    cited_ids: list[int]  # 1-based positions in the chunk list that the answer actually relies on
    # Week 9 (all optional, so ExtractiveGenerator is unchanged):
    not_found: bool = False                 # the LLM judged that the context does not contain the answer
    mode: str = "extractive"                # "extractive" | "llm" | "extractive_fallback"
    fallback_reason: str | None = None      # why the LLM answer was not used
    meta: dict = field(default_factory=dict)  # model, prompt_version, attempts, repairs, tokens
    prompt_sent: str | None = None          # the exact system+user text sent to the LLM (None: nothing was sent)


class AnswerGenerator(Protocol):
    def generate(self, question: str, chunks: list["RetrievedChunk"], prompt: str) -> GeneratedAnswer: ...


class ExtractiveGenerator:
    """No LLM: the answer is the best chunk quoted verbatim, cited as [1], plus at most one SUPPORTING chunk.

    Why a supporting chunk: in the evaluation, 4 answers came from the right document but from the neighbouring
    chunk, so the fact was missing (right_source_missing_fact). A second retrieved chunk of the SAME document is
    added when its score is at least `support_ratio` x the best score. Observed ratios: the needed neighbours scored
    0.77-0.99 of the best chunk, unneeded ones 0.19-0.55; 0.65 sits in that gap (only 4 observed cases: re-check it
    with run_rag_eval.py). It never changes the cited source ([1] is still the best chunk) and never rewrites text.
    Citation numbers are the chunk's position in the retrieved list, the same numbers used in the prompt.
    """

    def __init__(self, support_ratio: float = 0.65, max_supporting: int = 1):
        self.support_ratio = support_ratio
        self.max_supporting = max_supporting

    def generate(self, question: str, chunks: list["RetrievedChunk"], prompt: str) -> GeneratedAnswer:
        top = chunks[0]
        parts, cited = [f"{top.text.strip()} [1]"], [1]
        for position, chunk in enumerate(chunks[1:], start=2):
            if len(cited) > self.max_supporting:
                break
            if chunk.document_id == top.document_id and chunk.score >= self.support_ratio * top.score:
                parts.append(f"{chunk.text.strip()} [{position}]")
                cited.append(position)
        return GeneratedAnswer(text="\n\n".join(parts), cited_ids=cited)
