"""Week 8 settings, read from environment / .env (single place, no scattered constants)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/rag/week8/config.py -> project root is three levels up
BASE_DIR = Path(__file__).resolve().parents[3]

VALID_TEXT_VARIANTS = {"text_only", "header_plus_text"}


@dataclass(frozen=True)
class RagConfig:
    base_dir: Path
    db_path: Path
    chroma_path: Path
    collection: str
    model_name: str
    embedding_text_variant: str
    embed_batch_size: int
    # answerability gates (calibrated in notebooks/week8, used from Phase 3 on)
    top_k: int
    min_score: float
    min_lexical_coverage: float
    unknown_entity_check: bool
    min_score_cross_lingual: float
    answer_support_ratio: float
    log_path: Path


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_config() -> RagConfig:
    load_dotenv(BASE_DIR / ".env")  # never overrides variables already set in the environment

    variant = os.getenv("EMBEDDING_TEXT_VARIANT", "text_only")
    if variant not in VALID_TEXT_VARIANTS:
        raise ValueError(f"EMBEDDING_TEXT_VARIANT must be one of {sorted(VALID_TEXT_VARIANTS)}, got {variant!r}")

    top_k = int(os.getenv("RAG_TOP_K", "3"))
    if not 1 <= top_k <= 10:
        raise ValueError(f"RAG_TOP_K must be between 1 and 10, got {top_k}")

    return RagConfig(
        base_dir=BASE_DIR,
        db_path=_resolve(os.getenv("DB_PATH", "data/db/local_storage.sqlite")),
        chroma_path=_resolve(os.getenv("CHROMA_PATH", "data/db/chroma")),
        collection=os.getenv("RAG_COLLECTION", "rag_chunks_v1"),
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"),
        embedding_text_variant=variant,
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        top_k=top_k,
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.0")),
        min_lexical_coverage=float(os.getenv("RAG_MIN_LEXICAL_COVERAGE", "0.5")),
        unknown_entity_check=_as_bool(os.getenv("RAG_UNKNOWN_ENTITY_CHECK", "True")),
        min_score_cross_lingual=float(os.getenv("RAG_MIN_SCORE_CROSS_LINGUAL", "0.30")),
        answer_support_ratio=float(os.getenv("RAG_SUPPORT_RATIO", "0.65")),
        log_path=BASE_DIR / "logs" / "week8" / "rag.log",
    )
