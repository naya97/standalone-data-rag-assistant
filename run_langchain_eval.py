"""Week 10 evaluation: does the assistant choose the right tool, and does the whole flow answer correctly?

  python run_langchain_eval.py --mode choose --max-requests 100        1 request per question: the tool choice (dev, test, calib + sales)
  python run_langchain_eval.py --mode full --max-requests 200          choose + run + summarize (about 2 requests per question)
  python run_langchain_eval.py --mode full --ids 1,2,5,45 --max-requests 12      a small hand-picked set
  python run_langchain_eval.py --mode choose --splits holdout           the final check: ONCE, at the end

The results file is named after the run (full_calib-dev-test_results.json, full_holdout_results.json, ...): runs never overwrite each other.

Replies are cached (data/eval/week10/llm_cache.json): a re-run costs nothing, and a run stopped by the quota can simply be
repeated later. --max-requests is the number of requests allowed in THIS run (retries included); keep it below your daily limit.
The default sets are dev, test and calib; holdout is only read when you ask for it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.llm.week9.config import get_llm_config, get_retail_db_path, setup_llm_logging
from src.llm.week9.tool_specs import ToolRuntime
from src.llm.week10.assistant import build_assistant, default_runtime
from src.llm.week10.evaluation import ReplyCache, fingerprint, format_report, load_cases, run_cases
from src.llm.week10.request_layer import LangChainRequestLayer, RequestCounter
from src.llm.week10.tool_logging import ToolLog, default_log_path, setup_tool_log


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["choose", "full"], default="choose")
    parser.add_argument("--domain", choices=["documents", "sales", "all"], default="all")
    parser.add_argument("--set", choices=["original", "extended", "all"], default="all", help="which document question files")
    parser.add_argument("--splits", default="dev,test,calib", help="comma list; holdout only for the final check (default dev,test,calib)")
    parser.add_argument("--ids", help="only these question ids, e.g. 1,7,45,1001 (sales ids start at 1001)")
    parser.add_argument("--max-requests", type=int, default=15, help="requests allowed in this run, retries included (default 15)")
    parser.add_argument("--tag", help="added to the results file name (e.g. --tag after_cleaning): a second run never overwrites the first")
    parser.add_argument("--no-cleaning", action="store_true", help="run exactly what the model chose (reproduces the results before the cleaning)")
    parser.add_argument("--cache", help="reply cache (default data/eval/week10/llm_cache.json)")
    parser.add_argument("--out", help="results file (default data/eval/week10/<mode>_results.json)")
    args = parser.parse_args(argv)

    config = get_llm_config()
    setup_llm_logging(config)
    eval_dir = config.base_dir / "data" / "eval"
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    try:
        ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
        cases, excluded = load_cases(eval_dir, args.set, splits, ids, args.domain)
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    if not cases:
        print("no questions selected")
        return 1
    if "holdout" in splits and len(splits) > 1:
        print("NOTE: holdout is meant to be read once, at the end, on its own (--splits holdout).")

    db_path = get_retail_db_path()
    if args.mode == "full" and any(c.domain == "sales" for c in cases) and not db_path.exists():
        print(f"The retail database is missing: {db_path}. Run `python run_pipeline.py` first, or use --domain documents.")
        return 1

    log = ToolLog()
    setup_tool_log(default_log_path(config.base_dir))
    layer = LangChainRequestLayer(config, counter=RequestCounter(args.max_requests), log=log)
    cache = ReplyCache(args.cache or eval_dir / "week10" / "llm_cache.json", fingerprint(config))
    assistant = build_assistant(config, default_runtime() if args.mode == "full" else ToolRuntime(), log, layer, outer=cache.wrap, clean_arguments=not args.no_cleaning)
    if assistant.chooser is None:
        print(f"Cannot evaluate: the model is not available. {assistant.unavailable_reason}")
        return 1

    print(f"{len(cases)} questions, mode={args.mode}, model={config.model}, request budget {args.max_requests}")
    started = time.time()
    rows, aborted = run_cases(assistant, cases, args.mode, db_path)
    meta = {"mode": args.mode, "model": config.model, "splits": sorted(splits), "domain": args.domain, "ids": ids,
            "requests_sent": layer.requests, "max_requests": args.max_requests, "cache_hits": cache.hits,
            "cache_misses": cache.misses, "aborted": aborted, "seconds": round(time.time() - started),
            "fingerprint": cache.fingerprint, "cleaning": not args.no_cleaning, "tag": args.tag}
    name = "_".join([args.mode, "-".join(sorted(splits))] + ([args.domain] if args.domain != "all" else []) + (["ids"] if ids else []) + ([args.tag] if args.tag else []))
    out = Path(args.out) if args.out else eval_dir / "week10" / f"{name}_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "excluded_ids": [q["id"] for q in excluded], "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + format_report(rows, meta, excluded, cache))
    print(f"\nSaved: {out}")
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
