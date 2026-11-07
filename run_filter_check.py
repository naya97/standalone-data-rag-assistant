"""Do the model's filters explain the "not retrieved" refusals? Re-runs retrieval only: ZERO Gemini requests.

  python run_langchain_eval.py --mode full --max-requests 5 --out data/eval/week10/full_calib-dev-test_results.json   (all replies are cached: 0 requests)
  python run_filter_check.py data/eval/week10/full_calib-dev-test_results.json

  python run_filter_check.py <results file> --all-answerable --top-ks 3,5,8     (ONE run: does the FACT appear at a larger top_k?)

See src/llm/week10/filter_check.py for how to read the table. Needs the vector index (python run_rag_indexing.py), like the assistant.
"""
import argparse
import json
import sys
from pathlib import Path

from src.llm.week9.config import get_llm_config
from src.llm.week10.assistant import _week8_retriever
from src.llm.week10.evaluation import load_cases
from src.llm.week10.filter_check import check_rows, format_check


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="a results file written by run_langchain_eval.py (any mode)")
    parser.add_argument("--top-k", type=int, default=3, help="passages per retrieval (default 3, the assistant's default)")
    parser.add_argument("--top-ks", help="several values at once, e.g. 3,5,8: one retrieval, the recall of the FACT at each k (overrides --top-k)")
    parser.add_argument("--all-answerable", action="store_true", help="re-check EVERY answerable document question (the top_k evidence)")
    parser.add_argument("--out", help="also save the table as JSON")
    args = parser.parse_args(argv)

    from src.rag.week8.evaluator import matches_expected
    from src.rag.week9.evaluator import check_facts
    config = get_llm_config()
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))["rows"]
    cases = {c.id: c for c in load_cases(config.base_dir / "data" / "eval", "all", None, None, "documents")[0]}
    ks = tuple(int(x) for x in args.top_ks.split(",")) if args.top_ks else None
    results = check_rows(rows, cases, _week8_retriever(), matches_expected, args.top_k, check_facts, args.all_answerable, ks)
    print(format_check(results, args.top_k))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
