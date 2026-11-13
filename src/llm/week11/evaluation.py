from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.llm.week10.summarize import REJECTED_PREFIX
from src.llm.week11.nodes import TOOL_TO_ROUTE
from src.llm.week11.state import model_requests
from src.llm.week11.validation import MAX_QUESTION_CHARS

# The 15 existing questions, picked to cover each behaviour (ids of data/eval/week8 and data/eval/week10/sales_questions.json):
SELECTED_IDS = (1, 3, 6, 7, 23,         # rag, answerable (invoice, policy, PDF table, Arabic, report)
                9, 10, 37,              # rag, the documents do not contain it (near topic, unknown project, near topic)
                45, 47,                 # out of scope (English, Arabic): fallback
                1001, 1005, 1012, 1013,  # sql, answerable (no args, category, a year as a month range, Arabic)
                1016)                   # sql, the facts do not contain it (profit margin)
SPLITS = {"dev", "test", "calib"}        # holdout is never read here


@dataclass(frozen=True)
class RouterCase:
    label: str                           # Q1 / Q1001 (existing) or R1 (designed)
    question: str
    kind: str
    expected_route: tuple[str, ...]      # any of these is correct
    expected_final: str                  # answer | not_found | fallback
    expected_reason: str | None = None   # for a fallback: the reason code
    source: str = "existing"             # existing | designed
    base: Any = None                     # the Week 10 Case of an existing question: its scoring rules are reused
    note: str = ""


# the cases
def expectations(case) -> tuple[tuple[str, ...], str, str | None]:
    """(routes, final, reason) of an existing Week 10 case, from its own labels."""
    if case.expected_tool is None:
        return ("fallback",), "fallback", "no_tool"
    if case.domain == "documents":
        return ("rag",), ("not_found" if case.should_reject else "answer"), None
    return {"answer": (("sql",), "answer", None), "not_found": (("sql",), "not_found", None),
            "tool_error": (("sql",), "fallback", "tool_error")}[case.expect]


def load_existing_cases(eval_dir: Path, ids=SELECTED_IDS) -> list[RouterCase]:
    from src.llm.week10.evaluation import load_cases
    cases, _ = load_cases(Path(eval_dir), "all", SPLITS, list(ids), "all")      # an unknown id is an error, never skipped
    out = []
    for c in cases:
        routes, final, reason = expectations(c)
        out.append(RouterCase(f"Q{c.id}", c.question, c.kind, routes, final, reason, "existing", c))
    return out


def load_designed_cases(path: Path) -> list[RouterCase]:
    out = []
    for c in json.loads(Path(path).read_text(encoding="utf-8"))["cases"]:
        question = c["question"]
        if question is None:                                  # built at run time: longer than the input limit
            unit = c["build"]["repeat"]
            question = unit * (MAX_QUESTION_CHARS // len(unit) + 2)
        out.append(RouterCase(c["id"], question, c["kind"], tuple(c["expected_route"]), c["expected_final"], c.get("reason"),
                              "designed", None, c.get("note", "")))
    return out


# scoring
def to_assistant_answer(state: dict, seconds: float, attempts: int | None):
    """The router's final state as a Week 10 AssistantAnswer, so that `answer_row` (facts, sources, citations) can score it."""
    from src.llm.week10.assistant import AssistantAnswer
    if state.get("reason") in ("model_unavailable", "model_failed"):
        route = "unavailable"
    elif state.get("tool") and state.get("tool_status"):
        route = "tool"
    else:
        route = "no_tool"
    return AssistantAnswer(state["question"], state.get("answer", ""), route, state.get("tool"), state.get("args") or {},
                           state.get("tool_status"), state.get("summary"), state.get("error") if route == "unavailable" else None,
                           state.get("choose_tokens") or {}, state.get("choose_calls", 0), attempts, seconds,
                           result=state.get("result"), run_args=state.get("run_args"), clean_notes=state.get("clean_notes") or [])


def _model_failure(state: dict) -> str | None:
    """The model/quota failed: the row says nothing about the router, so it is never scored as wrong (same rule as Week 10)."""
    if state.get("reason") in ("model_unavailable", "model_failed"):
        return state.get("error") or state["reason"]
    summary = state.get("summary")
    if summary is not None and summary.source == "fallback" and not (summary.fallback_reason or "").startswith(REJECTED_PREFIX):
        return summary.fallback_reason
    return None


def router_row(case: RouterCase, state: dict, seconds: float, attempts: int | None, sales_facts: list[str] | None = None) -> dict:
    from src.llm.week10.evaluation import ABORT_ERRORS, answer_row
    summary = state.get("summary")
    classified = TOOL_TO_ROUTE.get(state.get("tool"), "fallback")
    row = {"label": case.label, "source": case.source, "kind": case.kind, "question": case.question[:200],
           "expected_route": list(case.expected_route), "expected_final": case.expected_final, "expected_reason": case.expected_reason,
           "classified": classified, "outcome": state.get("outcome"), "reason": state.get("reason"), "error": state.get("error"),
           "tool": state.get("tool"), "tool_status": state.get("tool_status"), "trace": state.get("trace"),
           "answer": state.get("answer"), "model_calls": model_requests(state), "attempts": attempts, "seconds": seconds,
           "choose_tokens": state.get("choose_tokens") or {},
           "summary_source": summary.source if summary else None,
           "summary_tokens": {"input": summary.input_tokens, "output": summary.output_tokens} if summary else {},
           "week10_outcome": None, "note": case.note}
    failure = _model_failure(state)
    if failure is not None:
        return {**row, "verdict": "not_evaluated", "error": failure, "abort": failure.split(":", 1)[0] in ABORT_ERRORS}

    route_ok = classified in case.expected_route
    final_ok = row["outcome"] == case.expected_final
    reason_ok = (row["reason"] == case.expected_reason) if case.expected_final == "fallback" and case.expected_reason else None
    content_ok = None
    if case.base is not None:
        w10 = answer_row(case.base, to_assistant_answer(state, seconds, attempts), sales_facts)
        row["week10_outcome"] = w10["outcome"]
        if case.expected_final == "answer":
            content_ok = w10["outcome"] == "correct"
    checks = {"route_ok": route_ok, "final_ok": final_ok, "reason_ok": reason_ok, "content_ok": content_ok}
    return {**row, **checks, "verdict": "pass" if all(v is not False for v in checks.values()) else "fail"}


def run_router_cases(app, assistant, cases: list[RouterCase], db_path=None, progress: Callable[[str], None] | None = print,
                     clock: Callable[[], float] = time.perf_counter) -> tuple[list[dict], str | None]:
    """Returns (rows, abort_reason). A quota/auth failure stops the run: more rows would say nothing about the router."""
    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        facts = None
        if case.base is not None and case.base.domain == "sales" and case.base.expect == "answer":
            from src.llm.week10.evaluation import expected_facts
            facts = expected_facts(case.base, db_path)
        before = assistant.layer.requests if assistant.layer else 0
        started = clock()
        state = app.invoke({"question": case.question, "trace": []})
        seconds = round(clock() - started, 2)
        attempts = (assistant.layer.requests - before) if assistant.layer else None
        row = router_row(case, state, seconds, attempts, facts)
        rows.append(row)
        if progress:
            progress(f"  [{i}/{len(cases)}] {case.label} -> {row['verdict']}  (route {row['classified']}, ended {row['outcome']}"
                     + (f"/{row['reason']}" if row["reason"] else "") + ")")
        if row.get("abort"):
            return rows, row.get("error")
    return rows, None


# report
def _ratio(rows: list[dict], key: str) -> str:
    known = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(known)}/{len(known)}" if known else "-"


def format_report(rows: list[dict], meta: dict) -> str:
    done = [r for r in rows if r["verdict"] != "not_evaluated"]
    lines = [f"WEEK 11 ROUTER EVALUATION  model={meta.get('model')}  cases={len(rows)} (evaluated {len(done)})",
             f"real requests sent this run: {meta.get('requests_sent')} (budget {meta.get('max_requests')}), "
             f"cache hits: {meta.get('cache_hits')}, new replies cached: {meta.get('cache_misses')}", ""]
    if meta.get("aborted"):
        lines += [f"!! RUN STOPPED: {meta['aborted']}", "   Rows without a real answer are 'not_evaluated' (never scored as wrong). "
                  "Run again later: replies already received are cached.", ""]
    lines += ["ROUTE AND ENDING   (pass = right route + right ending [+ right reason] [+ right content, existing answerable questions])",
              f"{'kind':<22}{'n':>3}{'route':>8}{'final':>8}{'reason':>8}{'content':>9}{'pass':>7}"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in done:
        groups[r["kind"]].append(r)
    for kind, group in sorted(groups.items()):
        lines.append(f"{kind:<22}{len(group):>3}{_ratio(group, 'route_ok'):>8}{_ratio(group, 'final_ok'):>8}"
                     f"{_ratio(group, 'reason_ok'):>8}{_ratio(group, 'content_ok'):>9}{sum(r['verdict'] == 'pass' for r in group):>7}")
    lines.append(f"{'ALL':<22}{len(done):>3}{_ratio(done, 'route_ok'):>8}{_ratio(done, 'final_ok'):>8}"
                 f"{_ratio(done, 'reason_ok'):>8}{_ratio(done, 'content_ok'):>9}{sum(r['verdict'] == 'pass' for r in done):>7}")

    failed = [r for r in done if r["verdict"] == "fail"]
    lines += ["", f"FAILED ({len(failed)})"]
    for r in failed:
        lines.append(f"  {r['label']:<6}[{r['kind']}] expected route {r['expected_route']} / {r['expected_final']}"
                     + (f"({r['expected_reason']})" if r["expected_reason"] else "")
                     + f"  got {r['classified']} / {r['outcome']}" + (f"({r['reason']})" if r["reason"] else "")
                     + (f"  week10={r['week10_outcome']}" if r["week10_outcome"] else ""))
        lines.append(f"         {r['question'][:70]!r} -> {str(r.get('answer'))[:160]!r}")
    by_hand = [r for r in done if r["verdict"] == "pass" and r["kind"] == "ambiguous" and r["outcome"] == "answer"]
    if by_hand:
        lines += ["", "READ THESE BY A PERSON (an automatic rule cannot judge an answer to an ambiguous question)"]
        lines += [f"  {r['label']}: {r['question']!r} -> route {r['classified']}: {str(r['answer'])[:200]!r}" for r in by_hand]

    lines += ["", "COST"]
    by_end: dict[str, list[dict]] = defaultdict(list)
    for r in done:
        by_end[r["outcome"]].append(r)
    for end, group in sorted(by_end.items()):
        lines.append(f"  ended {end:<10} n={len(group):<3} model calls per question: "
                     f"{sum(r['model_calls'] for r in group) / len(group):.1f}")
    reasons = Counter(r["reason"] for r in done if r["outcome"] == "fallback")
    lines.append(f"  fallbacks by reason: {dict(reasons) or 0};  summaries rebuilt without the model: "
                 f"{sum(r['summary_source'] == 'fallback' for r in done)}")
    week10 = Counter(r["week10_outcome"] for r in done if r["week10_outcome"])
    if week10:
        lines.append(f"  the {sum(week10.values())} existing questions by the Week 10 rules: {dict(week10)}")
    lines += ["", f"{len(done)} cases: a difference of one or two cases is not evidence, and no threshold is set from a set this small. "
                  "If a rule needs a threshold, a bigger set is needed first."]
    return "\n".join(lines)
