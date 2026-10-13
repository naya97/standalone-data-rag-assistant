"""ask(): retrieve -> answerability gate -> grounded prompt -> answer + citations.

status values:
  answered                   the gate passed; answer + citations returned
  no_relevant_context        the gate rejected the best chunk (reasons + signals are returned)
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

    def to_dict(self) -> dict:
        return asdict(self)


def _citation(citation_id: int, chunk: RetrievedChunk) -> dict:
    return {
        "citation_id": citation_id, "chunk_id": chunk.chunk_id, "original_filename": chunk.original_filename,
        "section_header": chunk.section_header, "chunk_index": chunk.chunk_index,
        "document_type": chunk.document_type, "month": chunk.month, "project": chunk.project,
        "service": chunk.service, "score": round(chunk.score, 4), "text": chunk.text,
    }


class RagService:
    def __init__(self, retriever: Retriever, checker: AnswerabilityChecker, generator: AnswerGenerator | None = None):
        self.retriever = retriever
        self.checker = checker
        self.generator = generator or ExtractiveGenerator()

    @classmethod
    def from_config(cls, config: RagConfig, embedder: Embedder | None = None,
                    generator: AnswerGenerator | None = None, warm_up: bool = True) -> "RagService":
        """Raises IndexNotBuiltError / VectorStoreError (model mismatch) if the index is unusable.
        warm_up=True loads the embedding model now, so latency_ms of the first ask() is a real latency."""
        rag_store = RagStore(config.db_path)
        store = VectorStore(config.chroma_path, config.collection, config.model_name, config.embedding_text_variant)
        store.open(create=False)
        embedder = embedder or Embedder(config.model_name, config.embed_batch_size)
        if warm_up:
            embedder.load()
        retriever = Retriever(config, embedder, store, rag_store)
        generator = generator or ExtractiveGenerator(support_ratio=config.answer_support_ratio)
        return cls(retriever, AnswerabilityChecker.from_config(config, rag_store), generator)

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
                citations = [_citation(i, retrieval.chunks[i - 1]) for i in generated.cited_ids]
                response = RagResponse("answered", generated.text, citations, answerability=decision.to_dict(),
                                       prompt=prompt if include_prompt else None, retrieved=retrieved, **base)

        response.latency_ms = int((time.perf_counter() - started) * 1000)
        top1 = response.answerability["signals"].get("top1_score") if response.answerability else None
        logger.info("ask status=%s filters=%s top1=%s latency_ms=%d q=%r",
                    response.status, response.applied_filters, top1, response.latency_ms, question[:80])
        return response
