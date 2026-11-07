
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from src.llm.week9.config import LLMConfig
from src.llm.week9.data_facts import get_sales_facts
from src.llm.week9.tool_specs import GET_SALES_SUMMARY, SEARCH_DOCUMENTS, TOOL_SYSTEM, TOOLS
from src.llm.week10.arguments import argument_issues, clean_search_arguments  # noqa: F401  (argument_issues is re-exported)
from src.llm.week10.assistant import Assistant, AssistantAnswer, Choice
from src.llm.week10.summarize import REJECTED_PREFIX, SYSTEM_DOCUMENTS, SYSTEM_SALES

DOC_TOOL, SALES_TOOL = SEARCH_DOCUMENTS.name, GET_SALES_SUMMARY.name
MAX_LISTED = 30   # rows printed in a list; the results file has them all
ABORT_ERRORS = {"LLMQuotaExhaustedError", "LLMAuthError", "LLMConfigError", "LLMRequestError"}  # retrying cannot help


# questions
@dataclass(frozen=True)
class Case:
    id: int
    question: str
    split: str
    kind: str
    domain: str                              # "documents" | "sales"
    expected_tool: str | None                # None: no tool should be used (off-topic)
    should_reject: bool = False              # documents: the documents do not contain the answer
    expected_sources: tuple[str, ...] = ()
    answer_contains: tuple[str, ...] = ()
    expected_args: dict | None = None        # sales: exact arguments
    facts: tuple[str, ...] = ()              # sales: paths into the SQL facts, e.g. "by_category.0.revenue"
    expect: str = "answer"                   # sales: "answer" | "tool_error" | "not_found"


def _doc_case(q: dict) -> Case:
    return Case(id=q["id"], question=q["question"], split=q.get("split", ""), kind=q["kind"], domain="documents",
                expected_tool=None if q["kind"] == "off_topic" else DOC_TOOL, should_reject=bool(q["should_reject"]),
                expected_sources=tuple(q.get("expected_sources") or ()), answer_contains=tuple(q.get("answer_contains") or ()))


def _sales_case(q: dict) -> Case:
    return Case(id=q["id"], question=q["question"], split=q.get("split", "dev"), kind=q["kind"], domain="sales",
                expected_tool=q["expected_tool"], expected_args=dict(q["expected_args"]), facts=tuple(q.get("facts") or ()),
                expect=q.get("expect", "answer"))


def load_cases(eval_dir: Path, set_name: str = "all", splits: set[str] | None = None, ids: list[int] | None = None,
               domain: str = "all") -> tuple[list[Case], list[dict]]:
    """(cases, excluded document questions). Unknown ids are an error, never skipped."""
    from src.rag.week8.evaluator import load_questions   # the project's own loader: same sets, same split names
    cases: list[Case] = []
    excluded: list[dict] = []
    if domain in ("documents", "all"):
        for q in load_questions(Path(eval_dir) / "week8", set_name, splits):
            if q.get("filters"):
                excluded.append(q)
            else:
                cases.append(_doc_case(q))
    if domain in ("sales", "all"):
        path = Path(eval_dir) / "week10" / "sales_questions.json"
        for q in json.loads(path.read_text(encoding="utf-8")):
            if not splits or q.get("split", "dev") in splits:
                cases.append(_sales_case(q))
    if ids:
        by_id = {c.id: c for c in cases}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown question ids {unknown} (excluded ids: {[q['id'] for q in excluded]})")
        cases = [by_id[i] for i in dict.fromkeys(ids)]
    return cases, excluded


# ----------------------------------------------------------------------------------------------- the expected figures (sales)
def resolve_path(data: Any, path: str) -> Any:
    """'by_category.0.revenue' / 'by_category.-1.category' -> the value inside the facts."""
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def expected_facts(case: Case, db_path: str | Path) -> list[str]:
    """What a correct sales answer must contain, from the SQL (the tool's own source), as text."""
    if not case.facts:
        return []
    facts = get_sales_facts(db_path, **(case.expected_args or {}))
    return [str(resolve_path(facts, p)) for p in case.facts]


# ----------------------------------------------------------------------------------------------- scoring: choice only
def choice_row(case: Case, choice: Choice) -> dict:
    row = {"id": case.id, "question": case.question, "split": case.split, "kind": case.kind, "domain": case.domain,
           "expected_tool": case.expected_tool, "tool": choice.tool, "args": choice.args, "answered_in_words": choice.text,
           "choose_tokens": choice.tokens, "issues": []}
    if choice.error is not None:
        return {**row, "outcome": "not_evaluated", "error": choice.error, "abort": _aborts(choice.error)}
    if case.expected_tool is None:
        outcome = "tool_ok" if choice.tool is None else "unneeded_tool"
    elif choice.tool is None:
        outcome = "no_tool"
    elif choice.tool != case.expected_tool:
        outcome = "wrong_tool"
    elif case.domain == "sales" and choice.args != case.expected_args:
        outcome = "wrong_args"
    else:
        outcome = "tool_ok"
    if choice.tool == DOC_TOOL:
        row["issues"] = argument_issues(case.question, choice.args)
        row["clean_notes"] = clean_search_arguments(case.question, choice.args)[1]   # what the backend would change
    return {**row, "outcome": outcome}


def _aborts(error: str | None) -> bool:
    return bool(error) and error.split(":", 1)[0] in ABORT_ERRORS


# ----------------------------------------------------------------------------------------------- scoring: the whole flow
def _check_facts(answer: str, facts: list[str]):
    from src.rag.week9.evaluator import check_facts   # the Week 9 rule: numbers normalised, Arabic digits, no guessing
    return check_facts(answer, facts)


def _matches_expected(filename: str, expected) -> bool:
    from src.rag.week8.evaluator import matches_expected
    return matches_expected(filename, list(expected))


def answer_row(case: Case, answer: AssistantAnswer, sales_facts: list[str] | None = None) -> dict:
    summary = answer.summary
    row = {"id": case.id, "question": case.question, "split": case.split, "kind": case.kind, "domain": case.domain,
           "expected_tool": case.expected_tool, "route": answer.route, "tool": answer.tool, "args": answer.args,
           "tool_status": answer.tool_status, "answer": answer.text, "model_calls": answer.model_requests,
           "attempts": answer.attempts, "choose_tokens": answer.choose_tokens, "seconds": answer.seconds,
           "summary_source": summary.source if summary else None, "summary_status": summary.status if summary else None,
           "summary_tokens": {"input": summary.input_tokens, "output": summary.output_tokens} if summary else {},
           "fallback_reason": summary.fallback_reason if summary else None, "issues": [], "rank": None,
           "cited_files": [], "facts_ok": None, "facts_rule": None, "fact_in_passages": None,
           "run_args": answer.run_args, "clean_notes": answer.clean_notes, "issues_after": []}
    if answer.tool == DOC_TOOL:
        row["issues"] = argument_issues(case.question, answer.args)                       # what the MODEL chose
        row["issues_after"] = argument_issues(case.question, answer.run_args or answer.args)  # what actually ran

    def done(outcome: str, **extra) -> dict:
        return {**row, "outcome": outcome, **extra}

    if answer.route == "unavailable":
        return done("not_evaluated", error=answer.error, abort=_aborts(answer.error))
    if summary and summary.source == "fallback" and not (summary.fallback_reason or "").startswith(REJECTED_PREFIX):
        return done("not_evaluated", error=summary.fallback_reason, abort=_aborts(summary.fallback_reason))

    passages = (answer.result or {}).get("passages", []) if isinstance(answer.result, dict) else []
    if answer.tool == DOC_TOOL:
        cited = [passages[c - 1]["source"] for c in (summary.citations if summary else []) if 1 <= c <= len(passages)]
        if summary and summary.source == "fallback" and passages:
            cited = [passages[0]["source"]]                      # the fallback quotes the first passage
        row["cited_files"] = cited
        if case.expected_sources:
            row["rank"] = next((i for i, p in enumerate(passages, 1) if _matches_expected(p["source"], case.expected_sources)), None)
        if case.answer_contains:   # is the FACT in the text the model was given? (rank only says the FILE was retrieved)
            row["fact_in_passages"] = bool(passages) and bool(_check_facts(" ".join(p["text"] for p in passages), list(case.answer_contains))[0])
    refused = summary is not None and summary.status == "not_found"

    # ---- documents
    if case.domain == "documents":
        if case.expected_tool is None or case.should_reject:
            if answer.route == "no_tool":
                return done("no_tool_answer")                      # words, no tool: to be read by a person
            if answer.tool != DOC_TOOL:
                return done("wrong_tool")
            if answer.tool_status != "ok":
                return done("tool_error")          # the call was rejected or failed: nothing leaked, nothing was answered
            return done("correct_rejection" if refused else "leak")
        if answer.route == "no_tool":
            return done("missed_tool")
        if answer.tool != DOC_TOOL:
            return done("wrong_tool")
        if answer.tool_status != "ok":
            return done("tool_error")
        if refused:
            # the model refused. Was the expected source even in front of it? If not, the cause is upstream (filters, query,
            # retrieval), not the model's reading: the two need different fixes, so they are different outcomes.
            return done("false_refusal" if row["rank"] is not None else "retrieval_miss")
        if not (row["cited_files"] and _matches_expected(row["cited_files"][0], case.expected_sources)):
            return done("wrong_source")
        if case.answer_contains:
            ok, rule = _check_facts(answer.text, list(case.answer_contains))
            row["facts_ok"], row["facts_rule"] = ok, rule
            if not ok:
                return done("right_source_missing_fact")
        return done("correct")

    # ---- sales
    if answer.route == "no_tool":
        return done("no_tool_answer" if case.expect == "not_found" else "missed_tool")
    if answer.tool != SALES_TOOL:
        return done("wrong_tool")
    if case.expect == "tool_error":
        return done("correct" if answer.tool_status == "error" else "leak")
    if answer.tool_status != "ok":
        return done("tool_error")
    if answer.args != case.expected_args:
        return done("wrong_args")
    if case.expect == "not_found":
        return done("correct" if refused else "leak")
    if refused:
        return done("false_refusal")
    ok, rule = _check_facts(answer.text, list(sales_facts or []))
    row["facts_ok"], row["facts_rule"] = ok, rule
    return done("correct" if ok else "right_source_missing_fact")


# ----------------------------------------------------------------------------------------------- the run
def run_cases(assistant: Assistant, cases: list[Case], mode: str, db_path: str | Path | None = None,
              progress: Callable[[str], None] | None = print) -> tuple[list[dict], str | None]:
    """Returns (rows, abort_reason). A quota/auth/config failure stops the run: more rows would say nothing about the model."""
    if mode not in ("choose", "full"):
        raise ValueError("mode must be 'choose' or 'full'")
    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        if mode == "choose":
            row = choice_row(case, assistant.choose(case.question))
        else:
            facts = None
            if case.domain == "sales" and case.expect == "answer":
                facts = expected_facts(case, db_path)
            row = answer_row(case, assistant.ask(case.question), facts)
        rows.append(row)
        if progress:
            progress(f"  [{i}/{len(cases)}] Q{case.id} -> {row['outcome']}" + (f"  (tool={row['tool']})" if row["outcome"] != "correct" and row.get("tool") else ""))
        if row.get("abort"):
            return rows, row.get("error")
    return rows, None


# ----------------------------------------------------------------------------------------------- reply cache
def fingerprint(config: LLMConfig) -> str:
    """What makes an old reply invalid: the model settings and every prompt/tool text the model reads. A guardrail is NOT
    here: changing a guardrail needs no new request."""
    texts = json.dumps([TOOL_SYSTEM, SYSTEM_DOCUMENTS, SYSTEM_SALES, [(t.name, t.description, t.parameters) for t in TOOLS]],
                       sort_keys=True, ensure_ascii=False)
    return "|".join([config.model, f"t={config.temperature}", f"max={config.max_output_tokens}", f"think={config.thinking_level}",
                     hashlib.sha256(texts.encode("utf-8")).hexdigest()[:12]])


class ReplyCache:
    """Model replies on disk. `wrap(runnable, role)` returns a runnable that answers from the cache when it can."""

    def __init__(self, path: str | Path, fingerprint_text: str):
        self.path, self.fingerprint = Path(path), fingerprint_text
        self.hits = self.misses = 0
        self.latencies: dict[str, list[int]] = defaultdict(list)
        self._data = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except (OSError, ValueError):
            print(f"WARNING: cache file {self.path} is unreadable; starting with an empty cache")
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)   # a crash or a quota stop never leaves a half-written cache

    def key(self, role: str, messages: list) -> str:
        raw = "\x1f".join([self.fingerprint, role, *(f"{m.type}:{m.text}" for m in messages)])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def wrap(self, runnable: Runnable, role: str) -> Runnable:
        def call(value) -> AIMessage:
            messages = value.to_messages() if hasattr(value, "to_messages") else list(value)
            key = self.key(role, messages)
            hit = self._data.get(key)
            if hit is not None:
                self.hits += 1
                self.latencies[role].append(hit.get("latency_ms", 0))
                return AIMessage(content=hit["text"], usage_metadata=hit.get("usage"),
                                 tool_calls=[{"name": c["name"], "args": c["args"], "id": f"cached-{n}", "type": "tool_call"}
                                             for n, c in enumerate(hit.get("tool_calls", []))])
            started = time.perf_counter()
            message = runnable.invoke(value)       # may raise: only successful replies are cached
            latency = int((time.perf_counter() - started) * 1000)
            self.misses += 1
            self.latencies[role].append(latency)
            self._data[key] = {"text": message.text, "usage": dict(message.usage_metadata or {}), "latency_ms": latency,
                               "tool_calls": [{"name": c["name"], "args": c.get("args") or {}} for c in (message.tool_calls or [])]}
            self._save()
            return message
        return RunnableLambda(call, name=f"cached_{role}")


# ----------------------------------------------------------------------------------------------- report
def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole}" if whole else "-"


def _percentile(values: list[int], p: float) -> int | None:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else None


def format_report(rows: list[dict], meta: dict, excluded: list[dict] | None = None, cache: ReplyCache | None = None) -> str:
    evaluated = [r for r in rows if r["outcome"] != "not_evaluated"]
    lines = [f"WEEK 10 EVALUATION  mode={meta['mode']}  model={meta.get('model')}  splits={meta.get('splits') or 'all'}  "
             f"questions={len(rows)} (evaluated {len(evaluated)})",
             f"real requests sent this run: {meta.get('requests_sent')} (budget {meta.get('max_requests')}), "
             f"cache hits: {meta.get('cache_hits')}, new replies cached: {meta.get('cache_misses')}", ""]
    if meta.get("aborted"):
        lines += [f"!! RUN STOPPED: {meta['aborted']}", "   Rows without a real answer are 'not_evaluated' (never scored as wrong). "
                  "Run again later: replies already received are cached.", ""]
    if excluded:
        lines += [f"left out (they carry API filters that this assistant never sees): ids {[q['id'] for q in excluded]}", ""]

    lines.append("TOOL CHOICE  (expected tool vs chosen tool)")
    lines.append(f"{'group':<32}{'n':>4}{'tool_ok':>9}{'wrong_tool':>11}{'no_tool':>9}{'unneeded':>9}{'wrong_args':>11}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in evaluated:
        groups[f"{r['domain']}/{r['kind']}"].append(r)
    for name, group in sorted(groups.items()):
        c = Counter(_choice_class(r) for r in group)
        lines.append(f"{name:<32}{len(group):>4}{c['tool_ok']:>9}{c['wrong_tool']:>11}{c['no_tool']:>9}{c['unneeded_tool']:>9}{c['wrong_args']:>11}")
    total = Counter(_choice_class(r) for r in evaluated)
    lines.append(f"{'ALL':<32}{len(evaluated):>4}{total['tool_ok']:>9}{total['wrong_tool']:>11}{total['no_tool']:>9}{total['unneeded_tool']:>9}{total['wrong_args']:>11}")

    issues = [(r, i) for r in evaluated for i in r.get("issues", [])]
    wrong = [(r, i) for r, i in issues if not i.startswith("missed:")]
    missed = [r["id"] for r, i in issues if i.startswith("missed:")]
    by_type = Counter(i.split(":", 1)[0] for _, i in issues)
    lines += ["", f"ARGUMENTS (heuristic, from the question text alone)  {dict(by_type) or 'no issues'}"]
    for r, i in wrong[:MAX_LISTED]:
        lines.append(f"  Q{r['id']:<5}{i}   | {r['question'][:60]!r}")
    if len(wrong) > MAX_LISTED:
        lines.append(f"  ... {len(wrong) - MAX_LISTED} more: see the results file")
    if meta["mode"] == "full" and any(r.get("run_args") is not None for r in evaluated):
        after = Counter(i.split(":", 1)[0] for r in evaluated for i in r.get("issues_after", []))
        changed = [r["id"] for r in evaluated if r.get("clean_notes")]
        lines.append(f"  after the backend's cleaning: {dict(after) or 'no issues'};  questions whose arguments were changed: {len(changed)} {changed[:30]}")
    if missed:
        lines.append(f"  missed = the question names a project but the model used no project filter (retrieval still works): ids {missed[:40]}"
                     + (" ..." if len(missed) > 40 else ""))

    if meta["mode"] == "full":
        lines += ["", "OUTCOMES (whole flow)"]
        lines.append(f"{'group':<32}{'n':>4}   outcomes (only the ones that happened)")
        for name, group in sorted(groups.items()):
            c = Counter(r["outcome"] for r in group)
            lines.append(f"{name:<32}{len(group):>4}   " + ", ".join(f"{o} {n}" for o, n in c.most_common()))
        review = [r for r in evaluated if r["outcome"] in ("no_tool_answer", "leak", "false_refusal", "retrieval_miss",
                                                           "right_source_missing_fact", "wrong_source", "tool_error", "wrong_args",
                                                           "missed_tool", "wrong_tool")]
        unseen = [r["id"] for r in evaluated if r["domain"] == "documents" and r.get("fact_in_passages") is False
                  and r["outcome"] in ("false_refusal", "retrieval_miss", "right_source_missing_fact")]
        if unseen:
            lines += ["", f"the expected FACT was not in the text the model received (so it could not have answered correctly): {len(unseen)} {unseen}"]
        lines += ["", f"READ THESE BY A PERSON ({len(review)}; an automatic rule cannot judge them)"]
        for r in review[:MAX_LISTED]:
            rank = ""
            if r["domain"] == "documents" and r["outcome"] in ("false_refusal", "retrieval_miss", "wrong_source", "right_source_missing_fact"):
                rank = f" rank={r['rank']}" if r.get("rank") is not None else " rank=not retrieved"
                if r.get("fact_in_passages") is not None:
                    rank += f" fact_in_passages={'yes' if r['fact_in_passages'] else 'NO'}"
            lines.append(f"  Q{r['id']:<5}[{r['kind']}] {r['outcome']}{rank}  tool={r['tool']} args={r['args']}")
            lines.append(f"         {r['question'][:70]!r} -> {str(r.get('answer'))[:160]!r}")
        if len(review) > MAX_LISTED:
            lines.append(f"  ... {len(review) - MAX_LISTED} more: see the results file")
        lines += ["", "COST"]
        calls = sum(r["model_calls"] for r in evaluated)
        sent = sum(r["attempts"] or 0 for r in evaluated)
        lines.append(f"  model calls {calls}, requests sent (retries included, cache hits excluded) {sent}, "
                     f"fallbacks {dict(Counter(r['summary_source'] for r in evaluated if r['summary_source'] == 'fallback')) or 0}")
    tokens_c = [r["choose_tokens"] for r in evaluated if r.get("choose_tokens")]
    if tokens_c:
        lines.append(f"  choice request: {round(sum(t['input'] for t in tokens_c) / len(tokens_c))} tokens in / "
                     f"{round(sum(t['output'] for t in tokens_c) / len(tokens_c))} out (mean of {len(tokens_c)})")
    if meta["mode"] == "full":
        tokens_s = [r["summary_tokens"] for r in evaluated if r.get("summary_tokens") and r["summary_tokens"]["input"]]
        if tokens_s:
            lines.append(f"  summary request: {round(sum(t['input'] for t in tokens_s) / len(tokens_s))} tokens in / "
                         f"{round(sum(t['output'] for t in tokens_s) / len(tokens_s))} out (mean of {len(tokens_s)})")
    if cache:
        for role, values in cache.latencies.items():
            lines.append(f"  model time, {role}: p50 {_percentile(values, 0.5)} ms, p95 {_percentile(values, 0.95)} ms ({len(values)} replies)")
    lines += ["", "Read it with the sizes in mind: a difference of one or two questions is not evidence. "
                  "Decide on dev/test/calib; look at holdout once, at the end."]
    return "\n".join(lines)


def _choice_class(row: dict) -> str:
    """The tool-choice verdict of a row of either mode."""
    if row["outcome"] in ("tool_ok", "wrong_tool", "no_tool", "unneeded_tool", "wrong_args"):
        return row["outcome"]
    expected, tool = row["expected_tool"], row.get("tool")
    if expected is None:
        return "tool_ok" if tool is None else "unneeded_tool"
    if tool is None:
        return "no_tool"
    if tool != expected:
        return "wrong_tool"
    if row["domain"] == "sales" and row["outcome"] == "wrong_args":
        return "wrong_args"
    return "tool_ok"
