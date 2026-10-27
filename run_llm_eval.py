"""LLM evaluation entry point.
   python run_llm_eval.py --splits dev,test,calib --diff llm@0.50,llm@0.33
Replies are cached (data/eval/week9/llm_cache.json); --max-requests protects the daily quota."""
import sys

from src.rag.week9.evaluator import main

if __name__ == "__main__":
    sys.exit(main())
