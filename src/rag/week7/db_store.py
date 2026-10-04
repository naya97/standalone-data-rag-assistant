
from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    document_type TEXT,
    source_format TEXT,
    file_path TEXT NOT NULL,
    month TEXT,
    project TEXT,
    service TEXT,
    classification_status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES rag_documents(document_id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    section_header TEXT,
    char_count INTEGER NOT NULL,
    original_filename TEXT,
    document_type TEXT,
    month TEXT,
    project TEXT,
    service TEXT
);
"""


class RagStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_DOCUMENTS_TABLE)
            conn.execute(CREATE_CHUNKS_TABLE)
        logger.info("rag_documents / rag_chunks schema ready at %s", self.db_path)

    def insert_document(self, metadata: dict, file_path: str) -> str:
        """Insert one row into rag_documents and return its new document_id (UUID)."""
        document_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_documents
                    (document_id, original_filename, document_type, source_format,
                     file_path, month, project, service, classification_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    metadata["original_filename"],
                    metadata["document_type"],
                    metadata["source_format"],
                    str(file_path),
                    metadata["month"],
                    metadata["project"],
                    metadata["service"],
                    metadata["classification_status"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return document_id

    def insert_chunks(self, document_id: str, metadata: dict, chunks: list[dict]) -> int:
        """Insert all chunks for one document.

        `chunks` is the list of dicts returned by DocumentChunker.chunk():
        {"chunk_text": str, "section_header": str|None}.
        Returns number of rows inserted.
        """
        rows = [
            (
                str(uuid.uuid4()),
                document_id,
                idx,
                item["chunk_text"],
                item["section_header"],
                len(item["chunk_text"]),
                metadata["original_filename"],
                metadata["document_type"],
                metadata["month"],
                metadata["project"],
                metadata["service"],
            )
            for idx, item in enumerate(chunks)
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO rag_chunks
                    (chunk_id, document_id, chunk_index, chunk_text, section_header,
                     char_count, original_filename, document_type, month, project, service)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)
