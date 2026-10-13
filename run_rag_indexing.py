import argparse
import logging

from src.rag.week8.config import get_config
from src.rag.week8.embedder import EmbedderError
from src.rag.week8.indexer import IndexingError, run_indexing
from src.rag.week8.vector_store import VectorStoreError


def configure_logging(log_path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build / sync the Week 8 vector index from rag_chunks.")
    parser.add_argument("--rebuild", action="store_true",
                        help="drop the collection and re-embed everything (needed after changing the model or text variant)")
    args = parser.parse_args()

    try:
        config = get_config()
    except ValueError as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc

    configure_logging(config.log_path)
    logger = logging.getLogger(__name__)

    try:
        summary = run_indexing(config, rebuild=args.rebuild)
    except (IndexingError, VectorStoreError, EmbedderError) as exc:
        logger.critical("Indexing Terminated Due To Error: %s", exc)
        raise SystemExit(1) from exc

    print("\n--- Week 8 Indexing Summary ---")
    print(f"Chunks in SQLite: {summary['chunks_in_sqlite']}")
    print(f"Vectors in index: {summary['vectors_now']}  (collection '{summary['collection']}')")
    print(f"Added / removed / unchanged: {summary['added']} / {summary['removed']} / {summary['unchanged']}")
    print(f"Model: {summary['model']} (text variant: {summary['text_variant']})")
    if summary["truncated_chunks"]:
        print(f"WARNING: {summary['truncated_chunks']} chunk(s) exceed the model's token limit")
    print(f"Done in {summary['seconds']}s")


if __name__ == "__main__":
    main()
