"""
  python run_langgraph_router.py "Which category has the highest revenue?"
  python run_langgraph_router.py "What is the late payment fee?" --verbose
  python run_langgraph_router.py "What is the weather today?" --json
  python run_langgraph_router.py --show-graph                         (the graph as a Mermaid diagram, no model)
  python run_langgraph_router.py "What is the total revenue?" --placeholder     (placeholder nodes: no model, no tools)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.llm.week11.graph import build_graph
from src.llm.week11.state import model_requests


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--show-graph", action="store_true", help="print the graph as a Mermaid diagram")
    parser.add_argument("--placeholder", action="store_true", help="use the placeholder nodes: no model, no tools")
    parser.add_argument("--verbose", action="store_true", help="also print the log lines to the terminal (stderr)")
    parser.add_argument("--json", action="store_true", help="print the final state as JSON")
    args = parser.parse_args(argv)
    if args.question is None and not args.show_graph:
        parser.error("give a question, or --show-graph")

    assistant = log_path = None
    if args.placeholder or args.question is None:
        app = build_graph()
    else:
        # imported here so --show-graph and --placeholder work without the model configuration
        from src.llm.week9.config import get_llm_config, setup_llm_logging
        from src.llm.week10.assistant import build_assistant
        from src.llm.week10.tool_logging import ToolLog, setup_tool_log
        from src.llm.week11.nodes import make_nodes

        config = get_llm_config()
        setup_llm_logging(config)
        log_path = Path(config.base_dir) / "logs" / "week11" / "router.log"
        setup_tool_log(log_path, echo=args.verbose)
        assistant = build_assistant(config, log=ToolLog())
        app = build_graph(make_nodes(assistant))

    if args.show_graph:
        print(app.get_graph().draw_mermaid())
    if args.question is None:
        return 0

    out = app.invoke({"question": args.question, "trace": []})   # an empty or too long question is stopped by the graph itself
    if args.json:
        shown = {k: v for k, v in out.items() if k != "result"}
        if out.get("summary") is not None:
            shown["summary"] = asdict(out["summary"])
        shown["model_requests"] = model_requests(out)
        print(json.dumps(shown, ensure_ascii=False, indent=2, default=str))
        return 0
    print(out["answer"])
    sent = assistant.layer.requests if assistant is not None and assistant.layer else "-"
    notes = out.get("clean_notes") or []
    if notes:
        print("\n[arguments cleaned: " + "; ".join(notes) + "]", file=sys.stderr)
    summary = out.get("summary")
    tokens = out.get("choose_tokens") or {}
    print(f"\n[path: {' -> '.join(out['trace'])}; route={out.get('route')}"
          + (f" ({out['reason']})" if out.get("reason") else "")
          + f"; outcome={out.get('outcome')}; tool={out.get('tool')} ({out.get('tool_status')}); "
            f"model calls={model_requests(out)}, requests sent (retries included)={sent}; "
            f"tokens choose={tokens.get('input', '-')}/{tokens.get('output', '-')}"
          + (f" summary={summary.input_tokens}/{summary.output_tokens} ({summary.source})" if summary else "")
          + (f"; log: {log_path}" if log_path else "") + "]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
