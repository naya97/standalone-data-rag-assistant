from __future__ import annotations

import logging
import sqlite3
import time

from src.rag.week7.db_store import RagStore
from src.rag.week8.config import RagConfig
from src.rag.week8.embedder import Embedder, build_embedding_text
from src.rag.week8.vector_store import VectorStore, to_chroma_metadata

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Indexing cannot proceed (nothing to index, or the result failed verification)."""


def run_indexing(config: RagConfig, rebuild: bool = False, embedder: Embedder | None = None) -> dict:
    started = time.time()

    # ---- Read
    logger.info("Read phase: loading chunks from %s", config.db_path)
    if not config.db_path.exists():
        raise IndexingError(f"Database not found: {config.db_path}. Run `python run_rag_ingestion.py` first.")
    try:
        chunks = RagStore(config.db_path).get_all_chunks()
    except sqlite3.OperationalError as exc:
        raise IndexingError(
            f"Could not read rag_chunks from {config.db_path} ({exc}). Run `python run_rag_ingestion.py` first."
        ) from exc
    if not chunks:
        raise IndexingError("rag_chunks is empty: nothing to index. Run `python run_rag_ingestion.py` first.")
    current = {c["chunk_id"]: c for c in chunks}
    logger.info("Read phase: %d chunk(s) in SQLite", len(current))

    # ---- Open
    store = VectorStore(config.chroma_path, config.collection, config.model_name, config.embedding_text_variant)
    store.open(create=True, rebuild=rebuild)
    existing = store.existing_ids()
    logger.info("Open phase: collection '%s' holds %d vector(s)%s",
                config.collection, len(existing), " (rebuilt from scratch)" if rebuild else "")

    # ---- Diff
    to_add = [cid for cid in current if cid not in existing]
    to_delete = [cid for cid in existing if cid not in current]
    unchanged = len(current) - len(to_add)
    logger.info("Diff phase: add=%d delete=%d unchanged=%d", len(to_add), len(to_delete), unchanged)

    # ---- Embed (before touching the index, so a failure here leaves it untouched)
    vectors, truncated = [], 0
    if to_add:
        embedder = embedder or Embedder(config.model_name, config.embed_batch_size)
        texts = [build_embedding_text(current[cid], config.embedding_text_variant) for cid in to_add]
        truncated = embedder.count_truncated(texts)
        if truncated:
            logger.warning("Embed phase: %d chunk(s) exceed the model's %d-token limit; their tail is not "
                           "represented in the vector (known limitation)", truncated, embedder.max_seq_length)
        logger.info("Embed phase: encoding %d chunk(s) with '%s' (variant=%s)",
                    len(texts), config.model_name, config.embedding_text_variant)
        vectors = embedder.encode(texts)

    # ---- Sync (re-running repairs a partial failure: the operation is idempotent)
    if to_delete:
        store.delete(to_delete)
    if to_add:
        store.upsert(to_add, vectors, [to_chroma_metadata(current[cid]) for cid in to_add])
    logger.info("Sync phase: +%d / -%d vector(s)", len(to_add), len(to_delete))

    # ---- Verify
    final_ids = store.existing_ids()
    if final_ids != set(current):
        raise IndexingError(
            f"Verification failed: index has {len(final_ids)} vector(s), SQLite has {len(current)} chunk(s) "
            f"(missing={len(set(current) - final_ids)}, extra={len(final_ids - set(current))})."
        )
    summary = {
        "chunks_in_sqlite": len(current), "indexed_before": len(existing), "added": len(to_add),
        "removed": len(to_delete), "unchanged": unchanged, "truncated_chunks": truncated,
        "vectors_now": store.count(), "collection": config.collection, "model": config.model_name,
        "text_variant": config.embedding_text_variant, "rebuilt": rebuild,
        "seconds": round(time.time() - started, 2),
    }
    logger.info("Indexing finished: %s", summary)
    return summary
