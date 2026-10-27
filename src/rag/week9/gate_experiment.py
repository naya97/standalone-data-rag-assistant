"""Gate experiment: which lexical-gate setting is best, decided from the evaluation questions, not guessed.

For every saved question, retrieval runs ONCE and every gate variant judges the same retrieved chunks
(no LLM, no API quota, takes seconds). Each variant is scored by what the gate alone does:

  answerable question    ok (passes, top-1 is the expected source) | wrong_source | false_refusal (the gate refuses it)
  unanswerable question  correct_rejection | leak (the gate lets it through)

Variants = lexical-coverage threshold x light Arabic mode (off/on). Read the result per language pair
(question -> best chunk): the Arabic same-language group is the one this experiment is about, and it is small.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Callable

from src.rag.week8.answerability import AnswerabilityChecker, script_of
from src.rag.week8.evaluator import load_questions, matches_expected, validate_questions

THRESHOLD_GRID = (0.25, 0.33, 0.40, 0.50, 0.60)
FOCUS_GROUP = "ar->ar"


def variant_name(threshold: float, light: bool) -> str:
    return f"light={'on' if light else 'off'} thr={threshold:.2f}"


def build_variants(config, vocabulary: set[str], grid=THRESHOLD_GRID) -> dict[str, AnswerabilityChecker]:
    thresholds = sorted({*grid, config.min_lexical_coverage})
    return {variant_name(t, light): AnswerabilityChecker(
                vocabulary, config.min_score, t, config.unknown_entity_check, config.min_score_cross_lingual, light)
            for light in (False, True) for t in thresholds}


def classify(question: dict, decision, top1_file: str) -> str:
    if question["should_reject"]:
        return "leak" if decision.answerable else "correct_rejection"
    if not decision.answerable:
        return "false_refusal"
    return "ok" if matches_expected(top1_file, question["expected_sources"]) else "wrong_source"


def compare_variants(questions: list[dict], retrieve: Callable[[dict], list | None],
                     variants: dict[str, AnswerabilityChecker]) -> dict:
    """retrieve(q) -> retrieved chunks, or None when the filters match no document (nothing to judge)."""
    rows, skipped = [], []
    for q in questions:
        chunks = retrieve(q)
        if not chunks:
            skipped.append(q["id"])
            continue
        top = chunks[0]
        rows.append({
            "id": q["id"], "question": q["question"], "split": q.get("split"), "kind": q.get("kind"),
            "should_reject": bool(q["should_reject"]), "top1_file": top.original_filename, "top1_score": round(top.score, 4),
            "group": f"{script_of(q['question'])}->{script_of(top.text)}",
            "outcomes": {name: classify(q, checker.check(q["question"], chunks), top.original_filename)
                         for name, checker in variants.items()},
        })
    return {"rows": rows, "skipped": skipped, "variants": list(variants)}


def tally(rows: list[dict], variant: str, group: str | None = None) -> dict:
    chosen = [r for r in rows if group is None or r["group"] == group]
    count = Counter(r["outcomes"][variant] for r in chosen)
    return {"answerable": sum(not r["should_reject"] for r in chosen),
            "unanswerable": sum(r["should_reject"] for r in chosen),
            "ok": count["ok"], "wrong_source": count["wrong_source"], "false_refusal": count["false_refusal"],
            "leak": count["leak"], "correct_rejection": count["correct_rejection"]}


def changed_vs(rows: list[dict], baseline: str, variant: str) -> list[dict]:
    return [{"id": r["id"], "question": r["question"], "group": r["group"],
             "from": r["outcomes"][baseline], "to": r["outcomes"][variant]}
            for r in rows if r["outcomes"][baseline] != r["outcomes"][variant]]


def format_report(result: dict, baseline: str, candidate: str) -> str:
    rows, names = result["rows"], result["variants"]
    groups = Counter(r["group"] for r in rows)
    lines = [f"GATE EXPERIMENT: {len(rows)} questions judged"
             + (f", {len(result['skipped'])} skipped (filters match no document)" if result["skipped"] else ""),
             "groups (question language -> best chunk language): " + ", ".join(f"{g}={n}" for g, n in sorted(groups.items())),
             "", f"{'variant':<24}{'ok':>4}{'refused':>9}{'wrong':>7}{'leaked':>8}   | {FOCUS_GROUP}: refused/answerable  leaked/unanswerable"]
    for name in names:
        t, f = tally(rows, name), tally(rows, name, FOCUS_GROUP)
        mark = "  <- current" if name == baseline else ""
        lines.append(f"{name:<24}{t['ok']:>4}{t['false_refusal']:>9}{t['wrong_source']:>7}{t['leak']:>8}   | "
                     f"{f['false_refusal']:>8}/{f['answerable']:<12}{f['leak']:>8}/{f['unanswerable']}{mark}")
    diff = changed_vs(rows, baseline, candidate)
    lines += ["", f"DECISIONS THAT CHANGE: '{baseline}' -> '{candidate}' ({len(diff)})"]
    lines += [f"  Q{d['id']:<4}{d['from']:>16} -> {d['to']:<16}[{d['group']}] {d['question'][:60]!r}" for d in diff]
    small = tally(rows, baseline, FOCUS_GROUP)
    lines += ["", f"Evidence size: {small['answerable']} answerable and {small['unanswerable']} unanswerable questions in {FOCUS_GROUP}. "
                  "A difference of one or two questions is not evidence; grow the set before changing a default."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from src.rag.week8.config import get_config
    from src.rag.week8.rag_service import RagService

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=["original", "extended", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5, help="same as the evaluation (the gate only reads the top-1 chunk)")
    parser.add_argument("--baseline", help="variant treated as the current setting (default: the threshold in .env, light off)")
    parser.add_argument("--candidate", help="variant to compare with the current setting, e.g. \"light=off thr=0.33\" "
                                            "(default: light=on at the current threshold)")
    args = parser.parse_args(argv)

    config = get_config()
    service = RagService.from_config(config, warm_up=True)
    questions = load_questions(config.base_dir / "data" / "eval" / "week8", args.set)
    errors, _ = validate_questions(questions, service.retriever.rag_store)
    if errors:
        print("Question set is invalid:\n  " + "\n  ".join(errors))
        return 1

    def retrieve(q: dict):
        result = service.retriever.retrieve(q["question"], top_k=args.top_k, filters=q.get("filters") or None)
        return result.chunks if result.status == "retrieved" else None

    variants = build_variants(config, service.checker.vocabulary)
    baseline = args.baseline or variant_name(config.min_lexical_coverage, False)
    candidate = args.candidate or variant_name(config.min_lexical_coverage, True)
    for name in (baseline, candidate):
        if name not in variants:
            print(f"Unknown variant {name!r}. Available:\n  " + "\n  ".join(variants))
            return 1
    if baseline == candidate:
        print(f"NOTE: baseline and candidate are the same variant ({baseline!r}), so no decision can change. "
              f"RAG_MIN_LEXICAL_COVERAGE in .env is {config.min_lexical_coverage}; pass --baseline to compare against another setting.\n")
    result = compare_variants(questions, retrieve, variants)
    print(format_report(result, baseline, candidate))

    out = config.base_dir / "data" / "eval" / "week9" / "gate_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())