from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from src.rag.week7.db_store import RagStore
from src.rag.week8.config import RagConfig
from src.rag.week8.embedder import Embedder
from src.rag.week8.vector_store import IndexNotBuiltError, VectorStore, build_where

logger = logging.getLogger(__name__)

# Filters are matched against the STORED values exactly (e.g. project="projectBeta", not "Beta").
ALLOWED_FILTERS = ("document_type", "month", "project", "service", "original_filename")
MAX_TOP_K = 10


class RetrievalError(Exception):
    pass


class InvalidFilterError(RetrievalError, ValueError):
    pass


class IndexStaleError(RetrievalError):
    """The vector index points to chunks that no longer exist in SQLite (ingestion re-ran without re-indexing)."""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    section_header: str | None
    original_filename: str
    document_id: str
    chunk_index: int
    document_type: str | None
    month: str | None
    project: str | None
    service: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    status: str  # "retrieved" | "no_documents_match_filter"
    chunks: list[RetrievedChunk]
    applied_filters: dict = field(default_factory=dict)
    stale_chunks: int = 0


def validate_filters(filters: dict | None) -> dict:
    if not filters:
        return {}
    clean = {}
    for key, value in filters.items():
        if key not in ALLOWED_FILTERS:
            raise InvalidFilterError(f"Unknown filter '{key}'. Allowed filters: {', '.join(ALLOWED_FILTERS)}.")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue  # an empty value means "no filter on this field"
        if not isinstance(value, str):
            raise InvalidFilterError(f"Filter '{key}' must be a string, got {type(value).__name__}.")
        clean[key] = value.strip()
    return clean


class Retriever:
    def __init__(self, config: RagConfig, embedder: Embedder, store: VectorStore, rag_store: RagStore):
        self.config = config
        self.embedder = embedder
        self.store = store
        self.rag_store = rag_store

    def index_status(self) -> dict:
        """Is the vector index in sync with SQLite? Compared by chunk_id, NOT by count: re-ingestion replaces
        every chunk_id while keeping the same number of chunks."""
        sqlite_ids = self.rag_store.get_all_chunk_ids()
        index_ids = self.store.existing_ids()
        return {
            "sqlite_chunks": len(sqlite_ids),
            "index_vectors": len(index_ids),
            "missing_from_index": len(sqlite_ids - index_ids),
            "stale_in_index": len(index_ids - sqlite_ids),
            "in_sync": sqlite_ids == index_ids,
        }

    def retrieve(self, question: str, top_k: int | None = None, filters: dict | None = None) -> RetrievalResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        k = self.config.top_k if top_k is None else top_k
        if not 1 <= k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {k}")

        applied = validate_filters(filters)
        where = build_where(applied)

        if self.store.count() == 0:
            raise IndexNotBuiltError("The vector index is empty. Run `python run_rag_indexing.py` first.")

        if where is not None and not self.store.has_match(where):
            logger.info("No chunk matches filters %s", applied)
            return RetrievalResult("no_documents_match_filter", [], applied)

        hits = self.store.query(self.embedder.encode_query(question), n_results=k, where=where)
        rows = self.rag_store.get_chunks_by_ids([h["chunk_id"] for h in hits])

        chunks: list[RetrievedChunk] = []
        stale = 0
        for hit in hits:
            row = rows.get(hit["chunk_id"])
            if row is None:
                stale += 1
                logger.warning("Index entry %s has no row in SQLite (stale index)", hit["chunk_id"])
                continue
            chunks.append(RetrievedChunk(
                chunk_id=row["chunk_id"], score=hit["score"], text=row["chunk_text"],
                section_header=row["section_header"], original_filename=row["original_filename"],
                document_id=row["document_id"], chunk_index=row["chunk_index"],
                document_type=row["document_type"], month=row["month"], project=row["project"],
                service=row["service"],
            ))

        if hits and not chunks:
            raise IndexStaleError(
                "Every retrieved index entry is missing from SQLite: the index is stale. "
                "Run `python run_rag_indexing.py`."
            )
        if stale:
            logger.warning("%d of %d retrieved entries were stale and skipped; re-run run_rag_indexing.py",
                           stale, len(hits))
        return RetrievalResult("retrieved", chunks, applied, stale)
