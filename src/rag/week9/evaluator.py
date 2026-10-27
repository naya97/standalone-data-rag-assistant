"""End-to-end evaluation of the RAG answer: Gemini vs the extractive answer, for several gate settings.

What it answers (from the saved question sets, never by guessing):
  - does the LLM catch the questions the gate lets through (leaks), and does it answer the ones the gate lets in?
  - does a more lenient gate (0.33, or the lexical check off) pay off once an LLM reads the context?
  - what does it cost: requests, tokens (input / output / hidden thinking), latency, repairs, fallbacks

Rules that keep the numbers honest
  - A variant is "<mode>@<threshold>": mode = llm | extractive, threshold = lexical coverage in front of the answer.
    The extractive answer is always judged by the stricter 0.50 gate (same rule as the live service).
  - LLM replies are cached on disk, keyed by the exact request. Variants share replies, a re-run costs nothing, and
    changing a guardrail needs no new request.
  - A request budget (--max-requests) protects the daily quota. When the quota or the key fails, the run STOPS and
    the rows that never got a real LLM answer are "not_evaluated". They are NEVER scored as extractive answers: that
    would silently mix a different system into the LLM numbers. Run again later: answers already received are cached.
  - A fallback caused by the model's own output (invalid JSON, citation that does not exist, invented number, blocked)
    IS scored: it is how the real system behaves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.llm.week9.client import LLMClient, LLMConfigError, LLMError, LLMQuotaExhaustedError, LLMResult
from src.llm.week9.guardrails import extract_numbers
from src.rag.week8.answerability import script_of
from src.rag.week8.evaluator import load_questions, matches_expected, summarize, validate_questions

DEFAULT_VARIANTS = "extractive@0.50,llm@0.50,llm@0.33,llm@0.00"
QUALITY_FALLBACKS = {"guardrails", "LLMBlockedError", "LLMInvalidOutputError"}  # the model's own failures: scored
ABORT_ERRORS = {"LLMQuotaExhaustedError", "LLMAuthError", "LLMConfigError", "LLMRequestError"}  # retrying cannot help


# ----------------------------------------------------------------------------------------------- facts
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _norm(text: str) -> str:
    text = text.translate(_ARABIC_DIGITS).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 4,500 -> 4500
    text = re.sub(r"(?<=\d)\.0+\b", "", text)        # 1250.00 -> 1250
    return re.sub(r"\s+", " ", text.replace("$", "")).strip()


def check_facts(answer: str, facts: list[str]) -> tuple[bool, str | None]:
    """Does the answer contain every expected fact? (ok, rule). An LLM rewrites, so the Week 8 literal check is not enough:
      exact   : the fact is in the answer after normalising numbers ("$4,500" = "4500", "1250.00" = "1250", Arabic digits)
      numbers : the fact contains a number and every number of the fact is in the answer ("30 days" vs "30 يوماً").
    Facts without numbers must match literally, so a translated word is reported for human review, never guessed."""
    normalised, numbers, rule = _norm(answer), extract_numbers(answer), "exact"
    for fact in facts:
        if _norm(fact) in normalised:
            continue
        wanted = extract_numbers(fact)
        if wanted and wanted <= numbers:
            rule = "numbers"
            continue
        return False, None
    return True, rule


# ----------------------------------------------------------------------------------------------- cache + budget
class CachingClient:
    """LLMClient wrapper for evaluation: replies are cached on disk and real requests are budgeted.
    Wrap the production client (it already retries), so cache hits never touch the network or the rate limiter."""

    def __init__(self, inner: LLMClient, path: Path, fingerprint: str, max_requests: int):
        self.inner, self.path, self.fingerprint, self.max_requests = inner, Path(path), fingerprint, max_requests
        self.model = inner.model
        self.requests_used = 0  # HTTP attempts really spent in this run (retries count: they use quota too)
        self.cache_hits = 0
        self._cache = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except (OSError, ValueError):
            print(f"WARNING: cache file {self.path} is unreadable; starting with an empty cache")
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # a crash or a quota stop never leaves a half-written cache

    def _key(self, system: str, user: str, schema) -> str:
        raw = "\x1f".join([self.fingerprint, getattr(schema, "__name__", ""), system, user])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, system: str, user: str, schema=None) -> LLMResult:
        key = self._key(system, user, schema)
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return LLMResult(**{k: v for k, v in hit.items() if k in LLMResult.__dataclass_fields__})
        if self.requests_used >= self.max_requests:
            raise LLMQuotaExhaustedError(f"the evaluation request budget ({self.max_requests}) is used up; run again later, "
                                         "replies already received are cached")
        try:
            result = self.inner.generate(system, user, schema)
        except LLMError as exc:
            self.requests_used += max(exc.attempts, 1)
            raise
        self.requests_used += max(result.attempts, 1)
        self._cache[key] = asdict(result)  # only successful replies are cached
        self._save()
        return result


# ----------------------------------------------------------------------------------------------- one question
def fallback_kind(generation: dict | None) -> str | None:
    """None: the LLM answer (or no LLM) was used. 'guardrails': the model's answers were rejected. Else: the error type."""
    if not generation or generation.get("mode") != "extractive_fallback":
        return None
    reason = generation.get("fallback_reason") or ""
    return "guardrails" if reason.startswith("guardrails:") else (reason.split(":", 1)[0] or "unknown")


def rejected_by(response) -> str | None:
    """Who refused: the lexical/cosine gate, the model, the stricter gate for verbatim answers, or the filter."""
    if response.status == "answered":
        return None
    if response.status == "no_documents_match_filter":
        return "filter"
    reasons = (response.answerability or {}).get("reasons", [])
    if any(r.startswith("llm_not_found") for r in reasons):
        return "llm"
    if any(r.startswith("stricter gate") for r in reasons):
        return "strict_gate"
    return "gate"


def evaluate_one(service, q: dict, top_k: int) -> dict:
    row = {"id": q["id"], "question": q["question"], "split": q.get("split"), "kind": q.get("kind"),
           "should_reject": bool(q["should_reject"]), "expected_facts": q.get("answer_contains") or []}
    try:
        r = service.ask(q["question"], top_k=top_k, filters=q.get("filters") or None, include_prompt=True)
    except Exception as exc:  # noqa: BLE001 - one broken question must not stop the run
        return {**row, "status": "error", "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}

    generation = r.generation or {}
    kind = fallback_kind(r.generation)
    if kind and kind not in QUALITY_FALLBACKS:  # infrastructure failure: this row says nothing about the LLM
        return {**row, "status": r.status, "outcome": "not_evaluated", "mode": generation.get("mode"),
                "fallback_kind": kind, "fallback_reason": generation.get("fallback_reason"), "abort": kind in ABORT_ERRORS}

    retrieved = r.retrieved or []
    cited = [c["original_filename"] for c in r.citations]
    rank = None
    if not q["should_reject"]:
        rank = next((i for i, c in enumerate(retrieved, 1) if matches_expected(c["original_filename"], q["expected_sources"])), None)
    facts_ok, facts_rule = None, None
    if not q["should_reject"] and q.get("answer_contains"):
        facts_ok, facts_rule = check_facts(r.answer, q["answer_contains"])

    if q["should_reject"]:
        outcome = "leak" if r.status == "answered" else "correct_rejection"
    elif r.status != "answered":
        outcome = "false_refusal"
    elif not (cited and matches_expected(cited[0], q["expected_sources"])):
        outcome = "wrong_source"
    elif facts_ok is False:
        outcome = "right_source_missing_fact"
    else:
        outcome = "correct"

    context_lang = script_of(retrieved[0]["text"]) if retrieved else None
    question_lang = script_of(q["question"])
    return {**row, "status": r.status, "outcome": outcome, "answer": r.answer, "cited_files": cited, "rank": rank,
            "facts_ok": facts_ok, "facts_rule": facts_rule, "rejected_by": rejected_by(r),
            "reasons": (r.answerability or {}).get("reasons", []), "mode": generation.get("mode"),
            "fallback_kind": kind, "fallback_reason": generation.get("fallback_reason"),
            "repairs": generation.get("repairs", 0), "attempts": generation.get("attempts", 0),
            "input_tokens": generation.get("input_tokens", 0), "output_tokens": generation.get("output_tokens", 0),
            "thinking_tokens": generation.get("thinking_tokens", 0), "model": generation.get("model"),
            "model_latency_ms": generation.get("model_latency_ms", 0),
            "group": f"{question_lang}->{context_lang}", "cross_lingual": bool(context_lang and context_lang != question_lang),
            "latency_ms": r.latency_ms}


def evaluate_questions(service, questions: list[dict], top_k: int, label: str, progress: bool = True) -> tuple[list[dict], str | None]:
    """Returns (rows, abort_reason). A quota/auth/config failure stops the run: continuing would only produce rows that
    say nothing about the LLM."""
    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        row = evaluate_one(service, q, top_k)
        rows.append(row)
        if progress:
            print(f"  [{i}/{len(questions)}] {label} Q{q['id']} -> {row['outcome']}"
                  + (f" ({row['fallback_kind']})" if row.get("fallback_kind") else ""), flush=True)
        if row.get("abort"):
            return rows, row.get("fallback_reason") or row["fallback_kind"]  # the reason already starts with the error type
    return rows, None


def select_questions(questions: list[dict], ids_text: str) -> list[dict]:
    """Keep only the listed question ids, in the order given ('7,27,38'). Unknown ids are an error, never skipped."""
    try:
        wanted = [int(x) for x in ids_text.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"--ids must be a comma list of numbers, got {ids_text!r}") from None
    if not wanted:
        raise ValueError("--ids is empty")
    by_id = {q["id"]: q for q in questions}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise ValueError(f"unknown question ids {missing} (known ids: {min(by_id)}..{max(by_id)})")
    return [by_id[i] for i in dict.fromkeys(wanted)]


# ----------------------------------------------------------------------------------------------- variants
def parse_variant(text: str) -> tuple[str, str, float]:
    """'llm@0.33' -> ('llm@0.33', 'llm', 0.33)"""
    mode, _, value = text.strip().partition("@")
    if mode not in ("llm", "extractive") or not value:
        raise ValueError(f"variant must look like llm@0.33 or extractive@0.50, got {text!r}")
    try:
        threshold = float(value)
    except ValueError:
        raise ValueError(f"variant {text!r}: the threshold must be a number") from None
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"variant {text!r}: the threshold must be between 0 and 1")
    return f"{mode}@{threshold:.2f}", mode, threshold


def apply_variant(service, mode: str, threshold: float, base_checker, extractive_floor: float, generators: dict) -> None:
    """Same rule as the live service: the gate in front of the answer has `threshold`; a verbatim answer is judged again by
    the stricter gate whenever that is stricter."""
    service.checker = base_checker.with_lexical_coverage(threshold)
    strict = base_checker.with_lexical_coverage(max(extractive_floor, threshold))
    service.strict_checker = strict if strict.min_lexical_coverage > threshold else None
    service.generator = generators[mode]


# ----------------------------------------------------------------------------------------------- summary
def _percentile(values: list[int], p: float) -> int | None:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else None


def summarize_variant(rows: list[dict]) -> dict:
    evaluated = [r for r in rows if r["outcome"] not in ("not_evaluated", "error")]
    base = summarize(evaluated) if evaluated else {"answerable": {"n": 0}, "unanswerable": {"n": 0}}
    refused_ok = Counter(r["rejected_by"] for r in evaluated if r["outcome"] == "correct_rejection")
    refused_wrong = Counter(r["rejected_by"] for r in evaluated if r["outcome"] == "false_refusal")
    llm_rows = [r for r in evaluated if r.get("input_tokens")]
    n = len(llm_rows)
    return {
        "questions": len(rows), "evaluated": len(evaluated), "not_evaluated": sum(r["outcome"] == "not_evaluated" for r in rows),
        "errors": sum(r["outcome"] == "error" for r in rows),
        "answerable": base["answerable"], "unanswerable": base["unanswerable"],
        "correct_rejection_by": dict(refused_ok), "false_refusal_by": dict(refused_wrong),
        "leaks_by_kind": dict(Counter(r["kind"] for r in evaluated if r["outcome"] == "leak")),
        "fallbacks": dict(Counter(r["fallback_kind"] for r in evaluated if r.get("fallback_kind"))),
        "answers_repaired": sum(1 for r in evaluated if r.get("repairs")),
        "retried_requests": sum(1 for r in evaluated if (r.get("attempts") or 0) > (1 + (r.get("repairs") or 0))),
        "llm_requests": n,
        "tokens_mean": {k: round(sum(r.get(f"{k}_tokens") or 0 for r in llm_rows) / n) if n else 0
                        for k in ("input", "output", "thinking")},
        "tokens_total": {k: sum(r.get(f"{k}_tokens") or 0 for r in llm_rows) for k in ("input", "output", "thinking")},
        # the model's own time: end-to-end time would show a cached reply as 20 ms
        "latency_ms": {"p50": _percentile([r["model_latency_ms"] for r in llm_rows if r.get("model_latency_ms")], 0.5),
                       "p95": _percentile([r["model_latency_ms"] for r in llm_rows if r.get("model_latency_ms")], 0.95)},
    }


def diff_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in rows_b}
    return [{"id": a["id"], "question": a["question"], "split": a["split"], "kind": a["kind"],
             "from": a["outcome"], "to": by_id[a["id"]]["outcome"], "answer": by_id[a["id"]].get("answer")}
            for a in rows_a if a["id"] in by_id and a["outcome"] != by_id[a["id"]]["outcome"]]


def format_comparison(results: dict, meta: dict, diff: tuple[str, str] | None = None) -> str:
    lines = [f"LLM EVALUATION  model={meta.get('model')}  prompt={meta.get('prompt_version')}  top_k={meta.get('top_k')}  "
             f"splits={meta.get('splits') or 'all'}",
             f"requests used this run: {meta.get('requests_used')} (budget {meta.get('max_requests')}), cache hits: {meta.get('cache_hits')}",
             ""]
    if meta.get("aborted"):
        lines += [f"!! RUN STOPPED: {meta['aborted']}", "   Rows without a real LLM answer are 'not_evaluated' (never scored as extractive). "
                  "Run again later: replies already received are cached.", ""]
    head = f"{'variant':<15}{'ans':>4}{'correct':>8}{'miss_fact':>10}{'wrong_src':>10}{'false_ref':>10} |{'unans':>6}{'rejected':>9}{'leaked':>7} |{'not_eval':>9}"
    lines += ["OUTCOMES", head]
    for name, res in results.items():
        s = res["summary"]; a, u = s["answerable"], s["unanswerable"]
        lines.append(f"{name:<15}{a['n']:>4}{a.get('correct', 0):>8}{a.get('right_source_missing_fact', 0):>10}"
                     f"{a.get('wrong_source', 0):>10}{a.get('false_refusal', 0):>10} |{u['n']:>6}"
                     f"{u.get('correct_rejection', 0):>9}{u.get('leak', 0):>7} |{s['not_evaluated']:>9}")
    lines += ["", "WHO DECIDES (rejections by gate / llm / strict_gate; leaks by question kind; fallbacks; repairs)"]
    for name, res in results.items():
        s = res["summary"]
        lines.append(f"{name:<15} right rejections {s['correct_rejection_by']} | wrong refusals {s['false_refusal_by']} | "
                     f"leaks {s['leaks_by_kind']} | fallbacks {s['fallbacks']} | repaired {s['answers_repaired']} retried {s['retried_requests']}")
    lines += ["", "COST (requests that reached the model)",
              f"{'variant':<15}{'requests':>9}{'in/req':>8}{'out/req':>8}{'think/req':>10}{'total in':>10}{'total out':>10}{'total think':>12}{'model p50':>10}{'model p95':>10}"]
    for name, res in results.items():
        s = res["summary"]; m, t, l = s["tokens_mean"], s["tokens_total"], s["latency_ms"]
        lines.append(f"{name:<15}{s['llm_requests']:>9}{m['input']:>8}{m['output']:>8}{m['thinking']:>10}{t['input']:>10}{t['output']:>10}"
                     f"{t['thinking']:>12}{str(l['p50']):>10}{str(l['p95']):>10}")
    review = [(n, r) for n, res in results.items() if n.startswith("llm") for r in res["rows"]
              if r["outcome"] == "right_source_missing_fact" and r.get("cross_lingual")]
    if review:
        lines += ["", "FACT REVIEW (cross-language answers that miss an expected fact: a translated word is not an error; read them)"]
        for name, r in review[:12]:
            lines.append(f"  {name} Q{r['id']} expected {r['expected_facts']} | answer: {str(r.get('answer'))[:110]!r}")
    if diff and all(d in results for d in diff):
        changes = diff_rows(results[diff[0]]["rows"], results[diff[1]]["rows"])
        lines += ["", f"DECISIONS THAT DIFFER: {diff[0]} -> {diff[1]} ({len(changes)})"]
        lines += [f"  Q{c['id']:<4}[{c['split']}, {c['kind']}] {c['from']} -> {c['to']}  {c['question'][:55]!r}" for c in changes]
    lines += ["", "Read it with the sizes in mind: a difference of one or two questions is not evidence. "
                  "Decide on dev/test/calib; look at holdout once, at the end."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    from src.llm.week9.client import build_client
    from src.llm.week9.config import get_llm_config, setup_llm_logging
    from src.rag.week8.answer_generator import ExtractiveGenerator
    from src.rag.week8.config import get_config
    from src.rag.week8.prompt_builder import RAG_PROMPT_VERSION
    from src.rag.week8.rag_service import RagService
    from src.rag.week9.llm_generator import LLMGenerator

    parser = argparse.ArgumentParser(description="End-to-end evaluation: LLM vs extractive, for several gate settings")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help=f"comma list of mode@threshold (default {DEFAULT_VARIANTS})")
    parser.add_argument("--set", choices=["original", "extended", "all"], default="all")
    parser.add_argument("--splits", help="comma list, e.g. dev,test,calib (decide here); add holdout only for the final check")
    parser.add_argument("--top-k", type=int, help="passages given to the answer (default RAG_TOP_K, the production value)")
    parser.add_argument("--ids", help="only these question ids, e.g. 1,7,27,38 (a small hand-picked set instead of a whole split)")
    parser.add_argument("--max-requests", type=int, default=15, help="HTTP attempts allowed in this run, retries included. Keep it below your DAILY limit "
                        "(the free tier of some models allows only 20 a day; check AI Studio) and raise it only if yours is higher")
    parser.add_argument("--diff", help="two variants to compare question by question, e.g. llm@0.50,llm@0.33")
    parser.add_argument("--cache", help="reply cache (default data/eval/week9/llm_cache.json)")
    parser.add_argument("--out", help="results file (default data/eval/week9/llm_eval_results.json)")
    args = parser.parse_args(argv)

    try:
        variants = [parse_variant(v) for v in args.variants.split(",") if v.strip()]
    except ValueError as exc:
        print(exc)
        return 1
    config, llm_config = get_config(), get_llm_config()
    setup_llm_logging(llm_config)
    top_k = args.top_k or config.top_k
    eval_dir = config.base_dir / "data" / "eval"
    cache_path = Path(args.cache) if args.cache else eval_dir / "week9" / "llm_cache.json"
    out_path = Path(args.out) if args.out else eval_dir / "week9" / "llm_eval_results.json"

    service = RagService.from_config(config, warm_up=True)
    splits = {s.strip() for s in args.splits.split(",")} if args.splits else None
    questions = load_questions(eval_dir / "week8", args.set, splits)
    if args.ids:
        try:
            questions = select_questions(questions, args.ids)
        except ValueError as exc:
            print(exc)
            return 1
    errors, _ = validate_questions(questions, service.retriever.rag_store)
    if errors:
        print("Question set is invalid:\n  " + "\n  ".join(errors))
        return 1

    caching = None
    if any(mode == "llm" for _, mode, _ in variants):
        try:
            client = build_client(llm_config)
        except LLMConfigError as exc:
            print(f"Cannot run LLM variants: {exc}")
            return 1
        fingerprint = "|".join([llm_config.model, f"t={llm_config.temperature}", f"max={llm_config.max_output_tokens}",
                                f"think={llm_config.thinking_level}", RAG_PROMPT_VERSION])
        caching = CachingClient(client, cache_path, fingerprint, args.max_requests)
    extractive = ExtractiveGenerator(support_ratio=config.answer_support_ratio)
    generators = {"extractive": extractive, "llm": LLMGenerator(caching, extractive) if caching else extractive}
    base_checker = service.checker

    print(f"{len(questions)} questions, variants {[v[0] for v in variants]}, top_k={top_k}, model={llm_config.model}")
    results: dict[str, dict] = {}
    aborted = None
    started = time.time()
    for name, mode, threshold in variants:
        apply_variant(service, mode, threshold, base_checker, config.extractive_min_lexical_coverage, generators)
        rows, aborted = evaluate_questions(service, questions, top_k, name)
        results[name] = {"rows": rows, "summary": summarize_variant(rows)}
        if aborted:
            break

    meta = {"model": llm_config.model, "prompt_version": RAG_PROMPT_VERSION, "top_k": top_k, "splits": sorted(splits) if splits else None,
            "requests_used": caching.requests_used if caching else 0, "max_requests": args.max_requests,
            "cache_hits": caching.cache_hits if caching else 0, "aborted": aborted, "seconds": round(time.time() - started)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "variants": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    diff = tuple(d.strip() for d in args.diff.split(",")) if args.diff else None
    print("\n" + format_comparison(results, meta, diff if diff and len(diff) == 2 else None))
    print(f"\nSaved: {out_path}")
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
