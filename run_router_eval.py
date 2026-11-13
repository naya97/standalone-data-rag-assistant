import argparse
import json
import sys
import time
from pathlib import Path

from src.llm.week9.config import get_llm_config, get_retail_db_path, setup_llm_logging
from src.llm.week10.assistant import build_assistant, default_runtime
from src.llm.week10.evaluation import ReplyCache, fingerprint
from src.llm.week10.request_layer import LangChainRequestLayer, RequestCounter
from src.llm.week10.tool_logging import ToolLog, setup_tool_log
from src.llm.week11.evaluation import format_report, load_designed_cases, load_existing_cases, run_router_cases
from src.llm.week11.graph import build_graph
from src.llm.week11.nodes import make_nodes


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-requests", type=int, default=60, help="requests allowed in this run, retries included (default 60)")
    parser.add_argument("--tag", help="added to the results file name")
    parser.add_argument("--cache", help="reply cache (default data/eval/week10/llm_cache.json, shared with Week 10)")
    parser.add_argument("--out", help="results file (default data/eval/week11/router_results[_TAG].json)")
    args = parser.parse_args(argv)

    config = get_llm_config()
    setup_llm_logging(config)
    eval_dir = config.base_dir / "data" / "eval"
    try:
        cases = load_existing_cases(eval_dir) + load_designed_cases(eval_dir / "week11" / "router_cases.json")
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    db_path = get_retail_db_path()
    if not db_path.exists():
        print(f"The retail database is missing: {db_path}. Run `python run_pipeline.py` first.")
        return 1

    log = ToolLog()
    setup_tool_log(config.base_dir / "logs" / "week11" / "router_eval.log")
    layer = LangChainRequestLayer(config, counter=RequestCounter(args.max_requests), log=log)
    cache = ReplyCache(args.cache or eval_dir / "week10" / "llm_cache.json", fingerprint(config))
    assistant = build_assistant(config, default_runtime(), log, layer, outer=cache.wrap)
    if assistant.chooser is None:
        print(f"Cannot evaluate: the model is not available. {assistant.unavailable_reason}")
        return 1
    app = build_graph(make_nodes(assistant))

    print(f"{len(cases)} cases, model={config.model}, request budget {args.max_requests}")
    started = time.time()
    rows, aborted = run_router_cases(app, assistant, cases, db_path)
    meta = {"model": config.model, "requests_sent": layer.requests, "max_requests": args.max_requests, "cache_hits": cache.hits,
            "cache_misses": cache.misses, "aborted": aborted, "seconds": round(time.time() - started),
            "fingerprint": cache.fingerprint, "tag": args.tag, "cases": [c.label for c in cases]}
    out = Path(args.out) if args.out else eval_dir / "week11" / f"router_results{'_' + args.tag if args.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n" + format_report(rows, meta))
    print(f"\nSaved: {out}")
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
