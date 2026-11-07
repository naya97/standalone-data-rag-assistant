from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RagRetriever(BaseRetriever):
    retriever: Any  # src.rag.week8.retriever.Retriever (duck-typed)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                top_k: int | None = None, filters: dict | None = None) -> list[Document]:
        found = self.retriever.retrieve(query, top_k=top_k, filters=filters)
        return [Document(
            id=chunk.chunk_id,
            page_content=chunk.text,
            metadata={"source": chunk.original_filename, "score": chunk.score, "section_header": chunk.section_header,
                      "document_type": chunk.document_type, "month": chunk.month, "project": chunk.project,
                      "service": chunk.service, "chunk_index": chunk.chunk_index})
            for chunk in found.chunks]
