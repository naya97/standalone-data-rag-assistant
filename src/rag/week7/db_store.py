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

    # ------------------------------------------------------------------
    # Row-level helpers. They take an already-open connection so that
    # several writes can share ONE transaction (needed by replace_document).
    # ------------------------------------------------------------------
    @staticmethod
    def _insert_document_row(conn: sqlite3.Connection, metadata: dict, file_path: str) -> str:
        document_id = str(uuid.uuid4())
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

    @staticmethod
    def _insert_chunk_rows(
        conn: sqlite3.Connection, document_id: str, metadata: dict, chunks: list[dict]
    ) -> int:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert_document(self, metadata: dict, file_path: str) -> str:
        """Insert one row into rag_documents and return its new document_id (UUID)."""
        with self._connect() as conn:
            return self._insert_document_row(conn, metadata, file_path)

    def insert_chunks(self, document_id: str, metadata: dict, chunks: list[dict]) -> int:
        """Insert all chunks for one document.

        `chunks` is the list of dicts returned by DocumentChunker.chunk():
        {"chunk_text": str, "section_header": str|None}.
        Returns number of rows inserted.
        """
        with self._connect() as conn:
            return self._insert_chunk_rows(conn, document_id, metadata, chunks)

    def replace_document(
        self, metadata: dict, file_path: str | Path, chunks: list[dict]
    ) -> tuple[str, int]:
        """Store a document, REPLACING any previous version of the same file.

        The match key is `file_path`. Old rows (document + its chunks) are deleted
        and the new ones inserted inside ONE transaction: if anything fails midway,
        everything rolls back and the previous version stays intact.

        Returns (new_document_id, number_of_old_versions_removed). The second value
        is normally 0 or 1; it can be >1 only if the DB already holds duplicates
        created by the old insert-only behaviour - they are all cleaned up here.
        """
        file_path = str(file_path)
        with self._connect() as conn:
            old_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT document_id FROM rag_documents WHERE file_path = ?", (file_path,)
                ).fetchall()
            ]
            for old_id in old_ids:
                # chunks first: SQLite does not enforce the FK by default
                conn.execute("DELETE FROM rag_chunks WHERE document_id = ?", (old_id,))
                conn.execute("DELETE FROM rag_documents WHERE document_id = ?", (old_id,))

            document_id = self._insert_document_row(conn, metadata, file_path)
            self._insert_chunk_rows(conn, document_id, metadata, chunks)
        return document_id, len(old_ids)

    def delete_documents_not_in(self, file_paths) -> list[str]:
        """Delete every stored document whose file_path is not in `file_paths`
        (i.e. the source file no longer exists). Returns the removed filenames.

        Refuses to run with an empty list, so a bad scan can never wipe the collection.
        """
        keep = {str(p) for p in file_paths}
        if not keep:
            logger.warning("delete_documents_not_in called with no paths - skipping cleanup")
            return []

        removed: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT document_id, file_path, original_filename FROM rag_documents"
            ).fetchall()
            for document_id, path, filename in rows:
                if path not in keep:
                    conn.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
                    conn.execute("DELETE FROM rag_documents WHERE document_id = ?", (document_id,))
                    removed.append(filename)
                    logger.info("Removed '%s' (source file no longer present)", filename)
        return removed

    # ------------------------------------------------------------------
    # Read API (Week 8: the vector index is derived from these rows)
    # ------------------------------------------------------------------
    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]

    def get_all_chunks(self) -> list[dict]:
        """Every stored chunk, in a deterministic order (filename, chunk_index)."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT chunk_id, document_id, chunk_index, chunk_text, section_header,
                       original_filename, document_type, month, project, service
                FROM rag_chunks
                ORDER BY original_filename, chunk_index
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_chunk_ids(self) -> set[str]:
        with self._connect() as conn:
            return {row[0] for row in conn.execute("SELECT chunk_id FROM rag_chunks")}

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """chunk_id -> full chunk row. Ids that no longer exist are simply absent from the result
        (the retriever treats that as a stale index entry)."""
        found: dict[str, dict] = {}
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            for start in range(0, len(chunk_ids), 500):  # stay far below SQLite's variable limit
                batch = chunk_ids[start:start + 500]
                marks = ",".join("?" * len(batch))
                for row in conn.execute(
                    f"""
                    SELECT chunk_id, document_id, chunk_index, chunk_text, section_header,
                           original_filename, document_type, month, project, service
                    FROM rag_chunks WHERE chunk_id IN ({marks})
                    """,
                    batch,
                ):
                    found[row["chunk_id"]] = dict(row)
        return found
