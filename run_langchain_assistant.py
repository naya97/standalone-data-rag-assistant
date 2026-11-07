"""Week 10 data-aware assistant from the command line (LangChain tools + Gemini).

  python run_langchain_assistant.py "Which category has the highest revenue?"
  python run_langchain_assistant.py "What is the late payment fee?" --verbose
  python run_langchain_assistant.py "Which category has the highest revenue?" --no-summary     (1 model request: tool result as is)
  python run_langchain_assistant.py --tool get_sales_summary --arg category=Furniture              (no model request at all)
  python run_langchain_assistant.py --tool search_documents --arg query="late fee" --arg project=projectAlpha --arg top_k=5
  python run_langchain_assistant.py --tool get_sales_summary --arg start=2024-01 --arg end=2024-06

--arg KEY=VALUE can be repeated (numbers such as top_k become numbers). It avoids the quoting problems of --args, whose JSON
needs escaped double quotes in the Windows command line:  --args "{\"category\": \"Furniture\"}"

Normal question: 2 model requests (choose the tool, then word the result); 1 when no summary is needed. The readable log of every
step is logs/week10/tool_calls.log (--verbose also prints it to the terminal).
"""
import argparse
import json
import sys

from src.llm.week9.config import get_llm_config, setup_llm_logging
from src.llm.week10.assistant import brief_result, build_assistant
from src.llm.week10.tool_logging import ToolLog, default_log_path, setup_tool_log


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--no-summary", action="store_true", help="skip the second request: print the tool result as it is")
    parser.add_argument("--tool", help="run this tool directly, without any model request")
    parser.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE", help="one argument for --tool (repeatable)")
    parser.add_argument("--args", default="{}", help="all arguments for --tool as a JSON object (--arg values win)")
    parser.add_argument("--no-cleaning", action="store_true", help="run the search arguments exactly as the model chose them")
    parser.add_argument("--full", action="store_true", help="with --tool: print the complete result (whole passages), not the short view")
    parser.add_argument("--verbose", action="store_true", help="also print the log lines to the terminal (stderr)")
    parser.add_argument("--json", action="store_true", help="print the whole answer as JSON")
    args = parser.parse_args(argv)
    if not args.question and not args.tool:
        parser.error("give a question, or --tool NAME")
    try:
        tool_args = json.loads(args.args)
        if not isinstance(tool_args, dict):
            raise ValueError
    except ValueError:
        parser.error("--args must be a JSON object (on Windows: --args \"{\\\"category\\\": \\\"Furniture\\\"}\"); "
                     "simpler: --arg category=Furniture")
    for item in args.arg:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            parser.error(f"--arg must look like KEY=VALUE, got {item!r}")
        try:
            tool_args[key.strip()] = json.loads(raw)   # 5 -> 5, Furniture / 2024-01 -> text
        except ValueError:
            tool_args[key.strip()] = raw

    config = get_llm_config()
    setup_llm_logging(config)
    log_path = default_log_path(config.base_dir)
    setup_tool_log(log_path, echo=args.verbose)
    assistant = build_assistant(config, log=ToolLog(), clean_arguments=not args.no_cleaning)

    if args.tool:
        answer = assistant.run_tool(args.tool, tool_args)
        shown = (answer.result if args.full else brief_result(answer.result)) if answer.result is not None else answer.text
        print(json.dumps(shown, ensure_ascii=False, indent=2, default=str))
        return 0 if answer.tool_status == "ok" else 1

    answer = assistant.ask(args.question, summarize=not args.no_summary)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print(answer.text)
        if answer.clean_notes:
            print("\n[arguments cleaned: " + "; ".join(answer.clean_notes) + "]", file=sys.stderr)
        used = f"tool={answer.tool} args={answer.args} ({answer.tool_status})" if answer.tool else f"route={answer.route}"
        tokens = answer.choose_tokens, (answer.summary.input_tokens, answer.summary.output_tokens) if answer.summary else None
        sent = answer.attempts if answer.attempts is not None else "-"
        print(f"\n[{used}; model calls={answer.model_requests}, requests sent (retries included)={sent}; "
              f"tokens choose={tokens[0] or '-'} summary={tokens[1] or '-'}; {answer.seconds}s; "
              f"{answer.question_id}; log: {log_path}]", file=sys.stderr)
    return 0 if answer.route != "unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
