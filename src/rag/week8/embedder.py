from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class EmbedderError(Exception):
    """The embedding model could not be loaded or used."""


def build_embedding_text(chunk: dict, variant: str) -> str:
    """The text that gets embedded. The stored / displayed / cited text is always the original chunk_text."""
    if variant == "header_plus_text" and chunk.get("section_header"):
        return f"{chunk['section_header']}\n{chunk['chunk_text']}"
    return chunk["chunk_text"]


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.RLock()  # one encode at a time: the API serves requests from a thread pool

    def load(self) -> "Embedder":
        """Load the model now (call at startup so the first request does not pay for it)."""
        self._load()
        return self

    def _load(self):
        with self._lock:
            return self._load_locked()

    def _load_locked(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt"
                ) from exc
            try:
                # Local cache first: no network round-trips (and no HuggingFace warnings) once the model is
                # downloaded. Only if it is not cached yet do we fall back to downloading it.
                logger.info("Loading embedding model '%s' from the local cache", self.model_name)
                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception as local_exc:  # noqa: BLE001
                    logger.info("Not available locally (%s); downloading '%s'", type(local_exc).__name__, self.model_name)
                    self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001 - surface any loader problem with a useful hint
                raise EmbedderError(
                    f"Could not load embedding model '{self.model_name}': {type(exc).__name__}: {exc}. "
                    "The first run needs internet access to download it. A 401 'Repository Not Found' "
                    "usually means a stale Hugging Face token (try `hf auth logout`)."
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        return int(self._load().max_seq_length)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Normalized float32 vectors, shape (n, dimension). Dot product == cosine similarity."""
        model = self._load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        with self._lock:
            vectors = model.encode(
                texts, normalize_embeddings=True, batch_size=self.batch_size, show_progress_bar=False
            )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def count_truncated(self, texts: list[str]) -> int:
        """How many texts exceed the model's input limit (the tail would be silently ignored)."""
        model = self._load()
        limit = self.max_seq_length
        return sum(
            len(model.tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"]) > limit
            for t in texts
        )