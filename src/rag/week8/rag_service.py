"""ask(): retrieve -> answerability gate -> grounded prompt -> answer + citations.

status values:
  answered                   the gate passed; answer + citations returned
  no_relevant_context        the gate rejected the best chunk (reasons + signals are returned),
                             or (Week 9) the LLM judged that the retrieved context does not contain the answer,
                             or (Week 9) the answer would be a verbatim chunk (LLM unavailable / extractive mode)
                             and the stricter extractive gate refuses it
  no_documents_match_filter  the metadata filters match no document

"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from src.rag.week7.db_store import RagStore
from src.rag.week8.answer_generator import AnswerGenerator, ExtractiveGenerator
from src.rag.week8.answerability import AnswerabilityChecker
from src.rag.week8.config import RagConfig
from src.rag.week8.embedder import Embedder
from src.rag.week8.prompt_builder import NO_ANSWER, NO_FILTER_MATCH, build_prompt, detect_language
from src.rag.week8.retriever import RetrievedChunk, Retriever
from src.rag.week8.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RagResponse:
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    applied_filters: dict = field(default_factory=dict)
    answerability: dict | None = None
    index_stale_chunks: int = 0
    prompt: str | None = None        # only with include_prompt=True (answered requests)
    retrieved: list[dict] | None = None  # only with include_prompt=True: every chunk retrieval returned
    latency_ms: int = 0
    generation: dict | None = None   # Week 9: how the answer was produced (mode, model, fallback_reason, tokens...)

    def to_dict(self) -> dict:
        return asdict(self)


def _citation(citation_id: int, chunk: RetrievedChunk) -> dict:
    return {
        "citation_id": citation_id, "chunk_id": chunk.chunk_id, "original_filename": chunk.original_filename,
        "section_header": chunk.section_header, "chunk_index": chunk.chunk_index,
        "document_type": chunk.document_type, "month": chunk.month, "project": chunk.project,
        "service": chunk.service, "score": round(chunk.score, 4), "text": chunk.text,
    }


def build_checkers(config: RagConfig, rag_store: RagStore) -> tuple[AnswerabilityChecker, AnswerabilityChecker | None]:
    """(gate in front of the LLM, stricter gate for verbatim answers or None when both would be the same).
    The strict gate is never looser than the main one."""
    checker = AnswerabilityChecker.from_config(config, rag_store)
    strict = checker.with_lexical_coverage(max(config.extractive_min_lexical_coverage, config.min_lexical_coverage))
    return checker, (strict if strict.min_lexical_coverage > checker.min_lexical_coverage else None)


class RagService:
    def __init__(self, retriever: Retriever, checker: AnswerabilityChecker, generator: AnswerGenerator | None = None,
                 strict_checker: AnswerabilityChecker | None = None):
        self.retriever = retriever
        self.checker = checker
        self.generator = generator or ExtractiveGenerator()
        # Week 9: the gate in front of the LLM may be lenient because the LLM reads the context and can say "not found".
        # A verbatim chunk (extractive mode, or the fallback when the LLM fails) has no such reader: it is judged again here.
        self.strict_checker = strict_checker

    @classmethod
    def from_config(cls, config: RagConfig, embedder: Embedder | None = None,
                    generator: AnswerGenerator | None = None, warm_up: bool = True) -> "RagService":
        """Raises IndexNotBuiltError / VectorStoreError (model mismatch) if the index is unusable.
        warm_up=True loads the embedding model now, so latency_ms of the first ask() is a real latency.
        Week 9: the caller passes the generator (see src.rag.week9.llm_generator.build_answer_generator)."""
        rag_store = RagStore(config.db_path)
        store = VectorStore(config.chroma_path, config.collection, config.model_name, config.embedding_text_variant)
        store.open(create=False)
        embedder = embedder or Embedder(config.model_name, config.embed_batch_size)
        if warm_up:
            embedder.load()
        retriever = Retriever(config, embedder, store, rag_store)
        generator = generator or ExtractiveGenerator(support_ratio=config.answer_support_ratio)
        checker, strict_checker = build_checkers(config, rag_store)
        return cls(retriever, checker, generator, strict_checker)

    def index_status(self) -> dict:
        return self.retriever.index_status()

    def ask(self, question: str, top_k: int | None = None, filters: dict | None = None,
            include_prompt: bool = False) -> RagResponse:
        started = time.perf_counter()
        lang = detect_language(question)
        retrieval = self.retriever.retrieve(question, top_k=top_k, filters=filters)

        base = dict(applied_filters=retrieval.applied_filters, index_stale_chunks=retrieval.stale_chunks)
        if retrieval.status == "no_documents_match_filter":
            response = RagResponse("no_documents_match_filter", NO_FILTER_MATCH[lang], **base)
        else:
            decision = self.checker.check(question, retrieval.chunks)
            retrieved = [_citation(i, c) for i, c in enumerate(retrieval.chunks, 1)] if include_prompt else None
            if not decision.answerable:
                response = RagResponse("no_relevant_context", NO_ANSWER[lang], answerability=decision.to_dict(),
                                       retrieved=retrieved, **base)
            else:
                prompt = build_prompt(question, retrieval.chunks)
                generated = self.generator.generate(question, retrieval.chunks, prompt)
                generation = {"mode": generated.mode, "fallback_reason": generated.fallback_reason, **generated.meta}
                answerability = decision.to_dict()
                # Week 9: show what was REALLY sent to the LLM (system + user); the Week 8 prompt only when nothing was sent
                shown = (generated.prompt_sent or prompt) if include_prompt else None
                strict = None
                if generated.mode != "llm" and not generated.not_found and self.strict_checker is not None:
                    strict = self.strict_checker.check(question, retrieval.chunks)
                if generated.not_found:
                    # Week 9: the gate let it through, the LLM read the context and says the fact is not there.
                    answerability["reasons"] = [*answerability.get("reasons", []),
                                                "llm_not_found: the model judged that the retrieved passages do not contain the answer"]
                    response = RagResponse("no_relevant_context", NO_ANSWER[lang], answerability=answerability,
                                           prompt=shown, retrieved=retrieved, generation=generation, **base)
                elif strict is not None and not strict.answerable:
                    # Week 9: no LLM reader, and the lenient gate is not enough to quote a chunk as the answer.
                    answerability = strict.to_dict()
                    answerability["reasons"] = [f"stricter gate for a verbatim answer (mode={generated.mode}): {r}"
                                                for r in strict.reasons]
                    response = RagResponse("no_relevant_context", NO_ANSWER[lang], answerability=answerability,
                                           prompt=shown, retrieved=retrieved, generation=generation, **base)
                else:
                    citations = [_citation(i, retrieval.chunks[i - 1]) for i in generated.cited_ids]
                    response = RagResponse("answered", generated.text, citations, answerability=answerability,
                                           prompt=shown, retrieved=retrieved, generation=generation, **base)

        response.latency_ms = int((time.perf_counter() - started) * 1000)
        top1 = response.answerability["signals"].get("top1_score") if response.answerability else None
        logger.info("ask status=%s mode=%s filters=%s top1=%s latency_ms=%d q=%r",
                    response.status, (response.generation or {}).get("mode"), response.applied_filters, top1,
                    response.latency_ms, question[:80])
        return response
