"""Summaries and insights from the command line (the same functions the API uses).

  python run_llm_insights.py sales --focus "Which categories are behind their target?"
  python run_llm_insights.py sales --category Furniture --start 2024-01 --end 2024-06 --language ar
  python run_llm_insights.py docs --filter project=projectBeta --filter document_type=invoice
  python run_llm_insights.py docs --question "What does the security report recommend?"
  add --no-llm to see the fallback (no request is sent to the model)
"""
import argparse
import json
import sys

from src.llm.week9.client import LLMConfigError, build_client
from src.llm.week9.config import get_llm_config, get_retail_db_path, setup_llm_logging
from src.llm.week9.data_facts import DataUnavailableError
from src.llm.week9.insights import sales_insights


def _client(no_llm: bool):
    if no_llm:
        return None, "--no-llm: the model was not used"
    config = get_llm_config()
    setup_llm_logging(config)
    try:
        return build_client(config), None
    except LLMConfigError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic in a Windows console
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-llm", action="store_true", help="skip the model and show the fallback")
    parser.add_argument("--show-prompt", action="store_true", help="also print the prompt that was sent")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sales", help="summary and insights over the cleaned retail data")
    s.add_argument("--category"); s.add_argument("--start", help="YYYY-MM"); s.add_argument("--end", help="YYYY-MM")
    s.add_argument("--focus"); s.add_argument("--language", choices=["en", "ar"], default="en")
    d = sub.add_parser("docs", help="summary of retrieved documents")
    d.add_argument("--question"); d.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    d.add_argument("--top-k", type=int, default=5); d.add_argument("--language", choices=["en", "ar"])
    args = parser.parse_args(argv)

    client, reason = _client(args.no_llm)
    try:
        if args.command == "sales":
            result = sales_insights(client, get_retail_db_path(), category=args.category, start=args.start, end=args.end,
                                    focus=args.focus, language=args.language, unavailable_reason=reason)
        else:
            from src.rag.week7.db_store import RagStore
            from src.rag.week8.config import get_config
            from src.rag.week9.doc_summary import chunks_by_filters, chunks_by_question, summarize_documents
            filters = dict(f.split("=", 1) for f in args.filter if "=" in f) or None
            if not args.question and not filters:
                print("Give --question or at least one --filter KEY=VALUE")
                return 1
            config = get_config()
            if args.question:
                from src.rag.week8.rag_service import RagService
                chunks = chunks_by_question(RagService.from_config(config, warm_up=True).retriever, args.question, args.top_k, filters)
            else:
                chunks = chunks_by_filters(RagStore(config.db_path), filters)
            result = summarize_documents(client, chunks, question=args.question, language=args.language, unavailable_reason=reason)
    except (DataUnavailableError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
    data = result.to_dict()
    if not args.show_prompt:
        data.pop("prompt_sent", None)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
