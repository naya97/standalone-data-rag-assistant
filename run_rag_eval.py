""" python run_rag_eval.py                       # all questions
    python run_rag_eval.py --set extended        # only the cross-lingual / hold-out questions
    python run_rag_eval.py --split test,holdout  # only some splits
"""

import argparse
import logging

from src.rag.week7.db_store import RagStore
from src.rag.week8.config import get_config
from src.rag.week8.embedder import EmbedderError
from src.rag.week8.evaluator import format_report, load_questions, run_evaluation, save_results, validate_questions
from src.rag.week8.rag_service import RagService
from src.rag.week8.vector_store import VectorStoreError


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval, answerability, sources and facts.")
    parser.add_argument("--set", choices=["original", "extended", "all"], default="all")
    parser.add_argument("--split", default="", help="comma-separated, e.g. dev,test,calib,holdout (default: all)")
    parser.add_argument("--eval-top-k", type=int, default=5, help="how many chunks to retrieve per question (for Hit@k)")
    parser.add_argument("--skip-validation", action="store_true", help="run even if the question file has errors")
    args = parser.parse_args()

    config = get_config()
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(config.log_path, encoding="utf-8")])

    eval_dir = config.base_dir / "data" / "eval" / "week8"
    questions = load_questions(eval_dir, args.set, set(filter(None, args.split.split(","))) or None)
    if not questions:
        raise SystemExit("No questions selected.")

    errors, warnings = validate_questions(questions, RagStore(config.db_path))
    for w in warnings:
        print("WARNING:", w)
    if errors and not args.skip_validation:
        print("\nThe QUESTION FILE has problems (fix them first, otherwise a failing question may be the file's fault):")
        for e in errors:
            print("  -", e)
        raise SystemExit(2)

    try:
        service = RagService.from_config(config)
    except (VectorStoreError, EmbedderError) as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc

    status = service.index_status()
    if not status["in_sync"]:
        raise SystemExit(f"Index out of sync with SQLite {status}. Run `python run_rag_indexing.py` first.")

    results = run_evaluation(service, questions, eval_top_k=args.eval_top_k)
    print("\n" + format_report(results))
    out = eval_dir / "eval_results.json"
    save_results(out, config, results)
    print(f"\nFull results (every question, every signal): {out}")


if __name__ == "__main__":
    main()