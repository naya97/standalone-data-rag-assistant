
import argparse
import json
import logging

from src.rag.week7.db_store import RagStore
from src.rag.week8.answerability import AnswerabilityChecker
from src.rag.week8.config import get_config
from src.rag.week8.embedder import Embedder, EmbedderError
from src.rag.week8.rag_service import RagService
from src.rag.week8.retriever import RetrievalError, Retriever
from src.rag.week8.vector_store import VectorStore, VectorStoreError


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve + answerability check for one question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--ask", action="store_true",
                        help="print the full RagService response (status, answer, citations) as JSON")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    filters = {}
    for item in args.filter:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--filter expects KEY=VALUE, got {item!r}")
        filters[key] = value

    config = get_config()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(config.log_path, encoding="utf-8")])

    if args.ask:
        try:
            response = RagService.from_config(config).ask(args.question, top_k=args.top_k, filters=filters,
                                                          include_prompt=True)
        except (RetrievalError, VectorStoreError, EmbedderError, ValueError) as exc:
            raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        return

    try:
        rag_store = RagStore(config.db_path)
        store = VectorStore(config.chroma_path, config.collection, config.model_name, config.embedding_text_variant)
        store.open(create=False)
        retriever = Retriever(config, Embedder(config.model_name, config.embed_batch_size), store, rag_store)
        status = retriever.index_status()
        if not status["in_sync"]:
            print(f"WARNING: index out of sync with SQLite {status}. Run `python run_rag_indexing.py`.\n")
        result = retriever.retrieve(args.question, top_k=args.top_k, filters=filters)
    except (RetrievalError, VectorStoreError, EmbedderError, ValueError) as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc

    print(f"status: {result.status} | filters: {result.applied_filters}")
    if result.status != "retrieved":
        return
    decision = AnswerabilityChecker.from_config(config, rag_store).check(args.question, result.chunks)
    print(f"answerable: {decision.answerable}")
    for reason in decision.reasons:
        print(f"  reason: {reason}")
    print(f"signals: {decision.signals}\n")
    for i, c in enumerate(result.chunks, 1):
        print(f"[{i}] score={c.score:.3f}  {c.original_filename}  (section: {c.section_header})")
        print("    " + c.text.replace("\n", "\n    ")[:300])


if __name__ == "__main__":
    main()
