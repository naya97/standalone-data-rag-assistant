from __future__ import annotations

import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    pass


class IndexNotBuiltError(VectorStoreError):
    """The collection does not exist yet: run run_rag_indexing.py."""


def to_chroma_metadata(row: dict) -> dict:
    meta = {
        "document_id": row["document_id"],
        "original_filename": row["original_filename"],
        "document_type": row["document_type"] or "unclassified",
    }
    for key in ("month", "project", "service"):
        if row.get(key):  # NULL / empty -> key is simply not stored
            meta[key] = row[key]
    return meta


def build_where(filters: dict | None) -> dict | None:
    """{"project": "projectBeta", "document_type": "invoice"} -> Chroma $and filter."""
    if not filters:
        return None
    items = [{key: {"$eq": value}} for key, value in filters.items()]
    return items[0] if len(items) == 1 else {"$and": items}


class VectorStore:
    BATCH = 256

    def __init__(self, path: str | Path, collection_name: str, model_name: str, text_variant: str):
        self.path = Path(path)
        self.collection_name = collection_name
        self.model_name = model_name
        self.text_variant = text_variant
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._col = None

    # ---------------------------------------------------------------- lifecycle
    def _expected_metadata(self) -> dict:
        return {
            "hnsw:space": "cosine",
            "embedding_model": self.model_name,
            "embedding_text_variant": self.text_variant,
        }

    def _check_compatible(self, col) -> None:
        stored = col.metadata or {}
        for key, expected in (("embedding_model", self.model_name), ("embedding_text_variant", self.text_variant)):
            if stored.get(key) != expected:
                raise VectorStoreError(
                    f"The index was built with {key}={stored.get(key)!r} but the current configuration "
                    f"uses {expected!r}. Vectors from different models/variants are not comparable: "
                    "re-run `python run_rag_indexing.py --rebuild`."
                )

    def drop(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Dropped collection '%s'", self.collection_name)
        except Exception:  # noqa: BLE001 - nothing to drop is fine
            pass
        self._col = None

    def open(self, create: bool = False, rebuild: bool = False) -> None:
        """create=True (indexer): create if missing. create=False (query time): must already exist."""
        if rebuild:
            self.drop()
        try:
            col = self._client.get_collection(name=self.collection_name, embedding_function=None)
        except Exception:  # noqa: BLE001 - not found
            col = None

        if col is None:
            if not create:
                raise IndexNotBuiltError(
                    f"Vector index '{self.collection_name}' not found at {self.path}. "
                    "Run `python run_rag_indexing.py` first."
                )
            try:
                col = self._client.create_collection(
                    name=self.collection_name, metadata=self._expected_metadata(), embedding_function=None
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not create collection '{self.collection_name}': {exc}") from exc
            logger.info("Created collection '%s' (cosine, model=%s, variant=%s)",
                        self.collection_name, self.model_name, self.text_variant)
        else:
            self._check_compatible(col)
        self._col = col

    def _require_open(self):
        if self._col is None:
            raise VectorStoreError("VectorStore.open() must be called first.")
        return self._col

    # ---------------------------------------------------------------- reads
    def count(self) -> int:
        return self._require_open().count()

    def existing_ids(self) -> set[str]:
        return set(self._require_open().get(include=[])["ids"])

    def has_match(self, where: dict | None) -> bool:
        """True if at least one chunk satisfies the filter (checked BEFORE querying)."""
        return bool(self._require_open().get(where=where, limit=1, include=[])["ids"])

    def query(self, vector, n_results: int, where: dict | None = None) -> list[dict]:
        """Nearest chunks as [{chunk_id, score, metadata}], best first. score = 1 - cosine distance."""
        col = self._require_open()
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        res = col.query(query_embeddings=[vector], n_results=n_results, where=where,
                        include=["metadatas", "distances"])
        return [
            {"chunk_id": cid, "score": 1.0 - dist, "metadata": meta}
            for cid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
        ]

    # ---------------------------------------------------------------- writes
    def upsert(self, ids: list[str], vectors, metadatas: list[dict]) -> None:
        col = self._require_open()
        for start in range(0, len(ids), self.BATCH):
            end = start + self.BATCH
            col.upsert(ids=ids[start:end], embeddings=[list(map(float, v)) for v in vectors[start:end]],
                       metadatas=metadatas[start:end])

    def delete(self, ids: list[str]) -> None:
        col = self._require_open()
        for start in range(0, len(ids), self.BATCH):
            col.delete(ids=ids[start:start + self.BATCH])
