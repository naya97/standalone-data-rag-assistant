"""Summary of retrieved documents: which passages (by filters or by a question), then the model words a short JSON summary.

    passages -> numbered -> model (summary + key points with source numbers) -> guardrails -> result
    anything fails -> the first lines of each document as key points (no model), flagged mode="fallback"
"""

from __future__ import annotations

from src.llm.week9.client import LLMClient
from src.llm.week9.guardrails import check_document_summary
from src.llm.week9.prompts import DOCS_PROMPT_VERSION, build_docs_messages, detect_language
from src.llm.week9.runner import InsightResult, ask_json
from src.llm.week9.schemas import DocumentSummary
from src.rag.week8.retriever import validate_filters

MAX_PASSAGES = 20


def chunks_by_filters(rag_store, filters: dict | None) -> list[dict]:
    """Every stored chunk whose metadata equals the filters (exact stored values, like /rag/ask), in reading order."""
    clean = validate_filters(filters)
    rows = [dict(c) for c in rag_store.get_all_chunks()]
    rows = [c for c in rows if all(c.get(key) == value for key, value in clean.items())]
    return sorted(rows, key=lambda c: (c["original_filename"], c["chunk_index"]))


def chunks_by_question(retriever, question: str, top_k: int, filters: dict | None) -> list[dict]:
    """The chunks retrieval returns for a question (the same search /rag/ask uses)."""
    result = retriever.retrieve(question, top_k=top_k, filters=filters)
    if result.status != "retrieved":
        return []
    return [{"chunk_text": c.text, "original_filename": c.original_filename, "section_header": c.section_header,
             "chunk_index": c.chunk_index} for c in result.chunks]


def _label(chunk: dict) -> str:
    label = chunk["original_filename"]
    if chunk.get("section_header"):
        label += f" | section: {chunk['section_header'].lstrip('# ').strip()}"
    return label


MAX_FALLBACK_POINTS = 5
FALLBACK_CLIP = 300  # chunks are at most 300 characters by design (Week 7), so a clipped passage is a whole passage


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= FALLBACK_CLIP else text[:FALLBACK_CLIP] + "..."


def fallback_summary(passages: list[dict]) -> dict:
    """No model, nothing invented. A small selection is shown passage by passage, as written (the fact you need may be in
    the second passage, not the first); a large one shows the opening passage of each document."""
    documents = len({p["file"] for p in passages})
    if len(passages) <= MAX_FALLBACK_POINTS:
        chosen, how = list(enumerate(passages, 1)), "the passages as written"
    else:
        chosen, seen = [], set()
        for number, p in enumerate(passages, 1):
            if p["file"] not in seen:
                seen.add(p["file"])
                chosen.append((number, p))
            if len(chosen) >= MAX_FALLBACK_POINTS:
                break
        how = "the opening passage of each document"
    return {"summary": f"Extract of {documents} document(s), {len(passages)} passage(s): {how}.",
            "key_points": [{"text": _clip(p["text"]), "sources": [number]} for number, p in chosen]}


def summarize_documents(client: LLMClient | None, chunks: list[dict], *, question: str | None = None,
                        language: str | None = None, unavailable_reason: str | None = None) -> InsightResult:
    """`chunks` are rows with chunk_text, original_filename, chunk_index, section_header (RagStore rows or
    chunks_by_question output). Raises ValueError when there is nothing to summarise or the language is invalid."""
    if not chunks:
        raise ValueError("No passages to summarise: the filters or the question matched no document.")
    language = language or (detect_language(question) if question else "en")
    if language not in ("en", "ar"):
        raise ValueError(f"language must be 'en' or 'ar', got {language!r}")
    notes = []
    if len(chunks) > MAX_PASSAGES:
        notes.append(f"Only the first {MAX_PASSAGES} of {len(chunks)} passages were used.")
        chunks = chunks[:MAX_PASSAGES]
    passages = [{"text": c["chunk_text"], "source": _label(c), "file": c["original_filename"],
                 "chunk_index": c["chunk_index"]} for c in chunks]
    sources = [{"id": i, "original_filename": p["file"], "chunk_index": p["chunk_index"], "text": p["text"]}
               for i, p in enumerate(passages, 1)]

    system, user = build_docs_messages(passages, question, language)
    texts = [p["text"] for p in passages]
    obj, info, reason, sent = ask_json(client, system, user, DocumentSummary, lambda o: check_document_summary(o, texts),
                                       prompt_version=DOCS_PROMPT_VERSION, unavailable_reason=unavailable_reason)
    if obj is not None:
        return InsightResult("llm", obj.model_dump(), None, sources=sources, notes=notes, generation=info, prompt_sent=sent)
    return InsightResult("fallback", fallback_summary(passages), reason, sources=sources, notes=notes, generation=info,
                         prompt_sent=sent)
