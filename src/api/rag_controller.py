from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.rag.week8.config import get_config
from src.rag.week8.embedder import EmbedderError
from src.rag.week8.rag_service import RagService
from src.rag.week8.retriever import IndexStaleError, InvalidFilterError
from src.rag.week8.vector_store import IndexNotBuiltError, VectorStoreError
from src.rag.week9.llm_generator import build_answer_generator  # Week 9

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])
_INIT_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------- schemas
class Filters(BaseModel):
    """Matched against the STORED metadata values exactly (e.g. project 'projectBeta', not 'Beta')."""
    model_config = ConfigDict(extra="forbid")  # an unknown filter name is a 422, never silently ignored

    document_type: Optional[str] = Field(None, examples=["invoice"])
    month: Optional[str] = Field(None, examples=["2024-02"])
    project: Optional[str] = Field(None, examples=["projectBeta"])
    service: Optional[str] = Field(None, examples=["cloudStorage"])
    original_filename: Optional[str] = None


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # catches typos such as "topk"
        # No filters in the prefilled example on purpose: Swagger sends the example as-is, and a leftover filter would
        # silently restrict every question you type to that subset of documents.
        json_schema_extra={"example": {"question": "What is the total amount due?", "top_k": 3, "include_prompt": False}},
    )
    question: str = Field(..., max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Defaults to RAG_TOP_K")
    filters: Optional[Filters] = None
    include_prompt: bool = Field(False, description="Debug switch: also return the grounded prompt and every retrieved chunk")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class Citation(BaseModel):
    citation_id: int
    chunk_id: str
    original_filename: str
    section_header: Optional[str] = None
    chunk_index: int
    document_type: Optional[str] = None
    month: Optional[str] = None
    project: Optional[str] = None
    service: Optional[str] = None
    score: float
    text: str


class Answerability(BaseModel):
    answerable: bool
    reasons: list[str]
    signals: dict[str, Any]


class Generation(BaseModel):
    """Week 9: how the answer was produced."""
    mode: str  # "llm" | "extractive" | "extractive_fallback"
    fallback_reason: Optional[str] = None  # why the LLM answer was not used (mode = extractive_fallback)
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    attempts: Optional[int] = None  # HTTP attempts, retries of transient errors included
    repairs: Optional[int] = None   # times the model was asked to correct a rejected answer
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None  # hidden reasoning tokens, not included in output_tokens
    model_latency_ms: Optional[int] = None  # time spent waiting for the model (retries included)


class AskResponse(BaseModel):
    status: Literal["answered", "no_relevant_context", "no_documents_match_filter"]
    answer: str
    citations: list[Citation]
    applied_filters: dict[str, str]
    answerability: Optional[Answerability] = None
    index_stale_chunks: int = 0
    prompt: Optional[str] = None
    retrieved: Optional[list[Citation]] = None
    latency_ms: int
    generation: Optional[Generation] = None  # Week 9


# ----------------------------------------------------------------------------------------------- lifecycle
def _init_service(app: FastAPI) -> None:
    try:
        config = get_config()
        generator = build_answer_generator(config)  # Week 9: LLM (extractive fallback) or extractive only
        app.state.rag_service = RagService.from_config(config, generator=generator, warm_up=True)
        app.state.rag_error = None
        logger.info("RAG service ready (embedding model loaded, index opened, answer_mode=%s)", config.answer_mode)
    except (VectorStoreError, EmbedderError, ValueError) as exc:  # ValueError: invalid configuration
        app.state.rag_service = None
        app.state.rag_error = f"{type(exc).__name__}: {exc}"
        logger.error("RAG service unavailable: %s", app.state.rag_error)


def _configure_logging() -> None:
    config = get_config()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(config.log_path, encoding="utf-8"), logging.StreamHandler()])


@asynccontextmanager
async def rag_lifespan(app: FastAPI):
    try:
        _configure_logging()
    except ValueError:
        pass  # invalid configuration is reported by _init_service; the app must still start
    app.state.rag_service = None
    app.state.rag_error = None
    _init_service(app)
    yield


def get_service(request: Request) -> RagService:
    app = request.app
    if getattr(app.state, "rag_service", None) is None:
        with _INIT_LOCK:  # several requests may arrive together: initialise once
            if getattr(app.state, "rag_service", None) is None:
                _init_service(app)
    service = getattr(app.state, "rag_service", None)
    if service is None:
        error = getattr(app.state, "rag_error", None) or "RAG service is not available."
        hint = " Run `python run_rag_indexing.py`." if "IndexNotBuilt" in error else ""
        raise HTTPException(status_code=503, detail=error + hint)
    return service


# ----------------------------------------------------------------------------------------------- endpoints
@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    service = get_service(request)
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None
    try:
        response = service.ask(body.question, top_k=body.top_k, filters=filters, include_prompt=body.include_prompt)
    except (InvalidFilterError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (IndexNotBuiltError, IndexStaleError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return response.to_dict()


@router.get("/status")
def status(request: Request):
    """Is the vector index in sync with SQLite? (compared by chunk_id; out of sync = run run_rag_indexing.py)"""
    service = get_service(request)
    try:
        return service.index_status()
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
