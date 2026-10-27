from __future__ import annotations

import logging
import threading
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.rag_controller import Filters, Generation, get_service
from src.llm.week9.client import LLMConfigError, build_client
from src.llm.week9.config import get_llm_config, get_retail_db_path, setup_llm_logging
from src.llm.week9.data_facts import DataUnavailableError
from src.llm.week9.insights import sales_insights
from src.llm.week9.runner import InsightResult
from src.rag.week7.db_store import RagStore
from src.rag.week8.config import get_config
from src.rag.week8.retriever import IndexStaleError, InvalidFilterError
from src.rag.week8.vector_store import IndexNotBuiltError, VectorStoreError
from src.rag.week9.doc_summary import chunks_by_filters, chunks_by_question, summarize_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])
_LLM_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------- schemas
class SalesInsightsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"focus": "Which categories are behind their revenue target?", "language": "en"}})
    category: Optional[str] = Field(None, max_length=100, description="Only this category (exact name)")
    start: Optional[str] = Field(None, examples=["2024-01"], description="First month, YYYY-MM")
    end: Optional[str] = Field(None, examples=["2024-06"], description="Last month, YYYY-MM")
    focus: Optional[str] = Field(None, max_length=300, description="What the insights should concentrate on")
    language: Literal["en", "ar"] = "en"
    include_prompt: bool = Field(False, description="Debug switch: also return the prompt that was sent")


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"filters": {"project": "projectBeta", "document_type": "invoice"}}})
    question: Optional[str] = Field(None, max_length=1000, description="Focus the summary on this (retrieval picks the passages)")
    filters: Optional[Filters] = Field(None, description="Exact stored metadata values; alone, they select every matching chunk")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Passages retrieved for a question (default 5)")
    language: Optional[Literal["en", "ar"]] = Field(None, description="Default: the language of the question, else English")
    include_prompt: bool = False

    @model_validator(mode="after")
    def _needs_a_selection(self):
        has_filter = bool(self.filters and self.filters.model_dump(exclude_none=True))
        if not (self.question and self.question.strip()) and not has_filter:
            raise ValueError("give a question, or at least one filter, to select what to summarise")
        return self


class InsightResponse(BaseModel):
    mode: Literal["llm", "fallback"]
    result: dict[str, Any]
    fallback_reason: Optional[str] = None
    facts: Optional[dict[str, Any]] = None             # sales: the numbers the text must come from
    sources: Optional[list[dict[str, Any]]] = None     # documents: the numbered passages
    notes: list[str] = []
    generation: Generation
    prompt: Optional[str] = None


# ----------------------------------------------------------------------------------------------- helpers
def get_llm(request: Request):
    """(client or None, reason). A missing key/model never fails a request: the fallback answers and says why."""
    app = request.app
    if not getattr(app.state, "llm_ready", False):
        with _LLM_LOCK:
            if not getattr(app.state, "llm_ready", False):
                try:
                    config = get_llm_config()
                    setup_llm_logging(config)
                    app.state.llm_client, app.state.llm_reason = build_client(config), None  # shared with /rag/ask
                except LLMConfigError as exc:
                    app.state.llm_client, app.state.llm_reason = None, f"{type(exc).__name__}: {exc}"
                except ValueError as exc:
                    raise HTTPException(status_code=503, detail=f"Invalid LLM configuration: {exc}") from exc
                app.state.llm_ready = True
    return app.state.llm_client, app.state.llm_reason


def _response(res: InsightResult, include_prompt: bool) -> dict:
    data = res.to_dict()
    generation = {"mode": res.mode, "fallback_reason": res.fallback_reason, **res.generation}
    return {"mode": res.mode, "result": res.result, "fallback_reason": res.fallback_reason, "facts": res.facts,
            "sources": res.sources, "notes": res.notes, "generation": generation,
            "prompt": data["prompt_sent"] if include_prompt else None}


# ----------------------------------------------------------------------------------------------- endpoints
@router.post("/sales-insights", response_model=InsightResponse)
def sales_insights_endpoint(body: SalesInsightsRequest, request: Request):
    client, reason = get_llm(request)
    try:
        res = sales_insights(client, get_retail_db_path(), category=body.category, start=body.start, end=body.end,
                             focus=body.focus, language=body.language, unavailable_reason=reason)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _response(res, body.include_prompt)


@router.post("/summarize-documents", response_model=InsightResponse)
def summarize_documents_endpoint(body: SummarizeRequest, request: Request):
    client, reason = get_llm(request)
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None
    question = body.question.strip() if body.question else None
    try:
        if question:  # retrieval needs the vector index
            chunks = chunks_by_question(get_service(request).retriever, question, body.top_k or 5, filters)
        else:         # filters alone only need SQLite
            chunks = chunks_by_filters(RagStore(get_config().db_path), filters)
        if not chunks:
            raise HTTPException(status_code=404, detail="No document matches the filters or the question.")
        res = summarize_documents(client, chunks, question=question, language=body.language, unavailable_reason=reason)
    except (InvalidFilterError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (IndexNotBuiltError, IndexStaleError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _response(res, body.include_prompt)
