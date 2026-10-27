"""Function-calling concept from the command line: which tool does the model choose for a question?

  python run_llm_tools.py "What is the total amount due for Project Alpha in January 2024?"
  python run_llm_tools.py "Which category has the highest revenue?" --run
  python run_llm_tools.py --demo          (6 requests: how often the choice is the expected one)

The model only CHOOSES; its arguments are validated here before anything runs. --run executes the chosen tool and prints
a short result. Sending that result back to the model for a final answer is Week 10.
"""
import argparse
import json
import sys

from src.llm.week9.client import LLMConfigError, build_client
from src.llm.week9.config import get_llm_config, get_retail_db_path, setup_llm_logging
from src.llm.week9.tool_specs import ToolRuntime, decide_tool, run_tool

DEMO = [  # question, the tool a person would pick (None: no tool)
    ("What is the total amount due for Project Alpha in January 2024?", "search_documents"),
    ("ما الحد الأدنى لطول كلمة المرور في سياسة مشروع Alpha؟", "search_documents"),
    ("Which fields are required in the email ingestion payload?", "search_documents"),
    ("Which category has the highest revenue?", "get_sales_summary"),
    ("How did the Furniture category sell between 2024-01 and 2024-06?", "get_sales_summary"),
    ("What is the capital of France?", None),
]


def _client():
    config = get_llm_config()
    setup_llm_logging(config)
    try:
        return build_client(config), None
    except LLMConfigError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _brief(result: dict) -> dict:
    if "passages" in result:
        return {"passages": [{"id": p["id"], "source": p["source"], "score": p["score"], "text": p["text"][:120]} for p in result["passages"]],
                **({"note": result["note"]} if "note" in result else {})}
    if "overall" in result:
        return {"scope": result["scope"], "overall": result["overall"], "best_month": result["best_month"],
                "top_category": result["by_category"][0] if result["by_category"] else None, "notes": result["notes"]}
    return result


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--run", action="store_true", help="also execute the chosen tool (search needs the vector index)")
    parser.add_argument("--demo", action="store_true", help="6 fixed questions: expected tool vs chosen tool")
    args = parser.parse_args(argv)
    if not args.question and not args.demo:
        parser.error("give a question, or --demo")

    client, reason = _client()
    if args.demo:
        right = 0
        print(f"{'question':<66}{'expected':<19}{'chosen':<19}args")
        for question, expected in DEMO:
            d = decide_tool(client, question, reason)
            ok = d.tool == expected and d.error is None
            right += ok
            print(f"{question[:64]:<66}{str(expected):<19}{str(d.tool):<19}{d.args or ''}{'' if ok else '   <-- ' + str(d.error or 'different tool')[:80]}")
        print(f"\n{right}/{len(DEMO)} as expected. (6 questions: a sample, not a rate.)")
        return 0

    decision = decide_tool(client, args.question, reason)
    out = {"decision": {"tool": decision.tool, "arguments": decision.args, "error": decision.error,
                        "answered_in_words": decision.answer_text}, "generation": decision.generation}
    if args.run and decision.tool and not decision.error:
        runtime = ToolRuntime(retail_db_path=get_retail_db_path())
        if decision.tool == "search_documents":
            from src.rag.week8.config import get_config
            from src.rag.week8.rag_service import RagService
            runtime.retriever = RagService.from_config(get_config(), warm_up=True).retriever
        out["tool_result"] = _brief(run_tool(decision.tool, decision.args, runtime))
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
