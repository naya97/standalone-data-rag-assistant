"""
Scan -> Load -> Metadata (best-effort) -> Chunk (one universal algorithm) -> Store
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.rag.week7.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DocumentChunker
from src.rag.week7.db_store import RagStore
from src.rag.week7.document_loader import (
    DocumentLoaderError,
    load_document,
    scan_documents,
)
from src.rag.week7.metadata_extractor import extract_metadata_from_filename
from src.rag.week7.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

# Header pre-segmentation is only trusted for formats we author ourselves.
HEADER_PRESPLIT_FORMATS = {"md", "txt"}
# The aggressive cleaning rules (de-fragmentation, forced paragraphs) are
# only safe for PDF-extracted text - see text_cleaner.py's module docstring
# for the concrete evidence (running them on our own .md files merges
# headers into body text).
AGGRESSIVE_CLEAN_FORMATS = {"pdf"}


class IngestionError(Exception):
    """Raised when the pipeline cannot proceed at all (e.g. no input files)."""


def run_ingestion(
    raw_dir: str | Path,
    db_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> dict:
    raw_dir = Path(raw_dir)

    logger.info("Scan phase: looking for documents under %s", raw_dir)
    files = scan_documents(raw_dir)
    if not files:
        msg = f"No supported documents (.txt/.md/.pdf) found in {raw_dir}"
        logger.critical("Ingestion Terminated: %s", msg)
        raise IngestionError(msg)
    logger.info("Scan phase: found %d document(s)", len(files))

    store = RagStore(db_path)
    store.initialize_schema()

    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    cleaner = TextCleaner()

    summary = {
        "total_files": len(files),
        "processed": 0,
        "failed": 0,
        "classified": 0,
        "unclassified": 0,
        "total_chunks": 0,
        "by_type": {},
        "by_format": {},
        "failures": [],
    }

    for path in files:
        try:
            logger.info("Load phase: reading %s", path.name)
            raw_text = load_document(path)

            logger.info("Metadata phase: best-effort classification for %s", path.name)
            metadata = extract_metadata_from_filename(path.name)
            if metadata["classification_status"] == "unclassified":
                logger.info("'%s' is unclassified - proceeding anyway (not fatal).", path.name)

            aggressive = metadata["source_format"] in AGGRESSIVE_CLEAN_FORMATS
            logger.info("Clean phase: cleaning %s (aggressive=%s)", path.name, aggressive)
            clean_text = cleaner.clean(raw_text, aggressive=aggressive)

            logger.info("Chunk phase: splitting %s (chunk_size=%d, overlap=%d)",
                        path.name, chunk_size, overlap)
            allow_presplit = metadata["source_format"] in HEADER_PRESPLIT_FORMATS
            chunks = chunker.chunk(clean_text, allow_header_presplit=allow_presplit)
            if not chunks:
                raise IngestionError(f"Chunking produced 0 chunks for {path.name}")

            logger.info("Store phase: writing %s (%d chunks) to %s",
                        path.name, len(chunks), db_path)
            document_id = store.insert_document(metadata, path)
            store.insert_chunks(document_id, metadata, chunks)

            summary["processed"] += 1
            summary["total_chunks"] += len(chunks)
            summary[metadata["classification_status"]] += 1
            doc_type = metadata["document_type"] or "unclassified"
            fmt = metadata["source_format"] or "unknown"
            summary["by_type"][doc_type] = summary["by_type"].get(doc_type, 0) + 1
            summary["by_format"][fmt] = summary["by_format"].get(fmt, 0) + 1

            logger.info("Finished %s: %d chunk(s) stored (document_id=%s, status=%s)",
                        path.name, len(chunks), document_id, metadata["classification_status"])

        except DocumentLoaderError as exc:
            logger.error("Load Phase FAILED for %s: %s", path.name, exc)
            summary["failed"] += 1
            summary["failures"].append({"file": path.name, "phase": "Load", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            logger.error("Ingestion FAILED for %s: %s", path.name, exc)
            summary["failed"] += 1
            summary["failures"].append({"file": path.name, "phase": "Unknown", "error": str(exc)})

    logger.info(
        "Ingestion finished: %d/%d files processed (%d classified, %d unclassified), "
        "%d chunks stored, %d failure(s)",
        summary["processed"], summary["total_files"],
        summary["classified"], summary["unclassified"],
        summary["total_chunks"], summary["failed"],
    )
    return summary
