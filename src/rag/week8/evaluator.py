"""
  1. Retrieval   : does a chunk of the expected document appear in the top-k? (Hit@k, MRR)       -> `rank`
  2. The gate    : was an answerable question answered, and an unanswerable one refused?         -> `outcome`
  3. The source  : is the cited document the expected one?                                         -> `outcome`
  4. The facts   : does the returned answer text actually contain the expected facts?             -> `outcome`

Outcomes
  answerable question   : correct | right_source_missing_fact | wrong_source | false_refusal | error
  unanswerable question : correct_rejection | leak | error
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import statistics
from collections import defaultdict
from pathlib import Path

from src.rag.week7.db_store import RagStore
from src.rag.week8.answerability import script_of
from src.rag.week8.retriever import ALLOWED_FILTERS

logger = logging.getLogger(__name__)

QUESTION_SETS = {
    "original": ["eval_questions.json"],
    "extended": ["eval_questions_extended.json"],
    "all": ["eval_questions.json", "eval_questions_extended.json"],
}
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def matches_expected(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(filename, p) for p in patterns)


# ----------------------------------------------------------------------------------------------- loading
def load_questions(eval_dir: Path, which: str = "all", splits: set[str] | None = None) -> list[dict]:
    questions: list[dict] = []
    for name in QUESTION_SETS[which]:
        path = Path(eval_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"Question file not found: {path}")
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    if splits:
        questions = [q for q in questions if q.get("split") in splits]
    return questions


def validate_questions(questions: list[dict], rag_store: RagStore) -> tuple[list[str], list[str]]:
    
    errors: list[str] = []
    warnings: list[str] = []
    chunks = rag_store.get_all_chunks()
    filenames = sorted({c["original_filename"] for c in chunks})
    seen_ids: set = set()

    for q in questions:
        qid = q.get("id")
        if qid in seen_ids:
            errors.append(f"Q{qid}: duplicate id")
        seen_ids.add(qid)
        if not str(q.get("question", "")).strip():
            errors.append(f"Q{qid}: empty question")
        for key in q.get("filters") or {}:
            if key not in ALLOWED_FILTERS:
                errors.append(f"Q{qid}: unknown filter '{key}'")
        if q.get("should_reject"):
            if q.get("expected_sources"):
                errors.append(f"Q{qid}: an unanswerable question must not list expected_sources")
            continue
        expected = q.get("expected_sources") or []
        if not expected:
            errors.append(f"Q{qid}: an answerable question needs expected_sources")
        matched_docs = set()
        for pattern in expected:
            hits = [f for f in filenames if fnmatch.fnmatchcase(f, pattern)]
            if not hits:
                errors.append(f"Q{qid}: no stored document matches '{pattern}'")
            matched_docs.update(hits)
        doc_chunks = [_norm(c["chunk_text"]) for c in chunks if c["original_filename"] in matched_docs]
        joined = " ".join(doc_chunks)
        for fact in q.get("answer_contains") or []:
            f = _norm(fact)
            if f not in joined:
                errors.append(f"Q{qid}: fact {fact!r} appears nowhere in {sorted(matched_docs)}")
            elif not any(f in t for t in doc_chunks):
                warnings.append(f"Q{qid}: fact {fact!r} spans several chunks (an extractive answer cannot contain it)")
    return errors, warnings


# ----------------------------------------------------------------------------------------------- running
def _evaluate_one(service, q: dict, eval_top_k: int) -> dict:
    result = {"id": q["id"], "question": q["question"], "split": q.get("split"), "kind": q.get("kind"),
              "should_reject": bool(q["should_reject"]), "filters": q.get("filters") or {}, "tests": q.get("tests")}
    try:
        r = service.ask(q["question"], top_k=eval_top_k, filters=q.get("filters") or None, include_prompt=True)
    except Exception as exc:  # noqa: BLE001 - one broken question must not stop the run
        logger.error("Q%s failed: %s: %s", q["id"], type(exc).__name__, exc)
        return {**result, "status": "error", "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}

    retrieved = r.retrieved or []
    cited = [c["original_filename"] for c in r.citations]
    signals = (r.answerability or {}).get("signals", {})
    context_lang = script_of(retrieved[0]["text"]) if retrieved else None
    question_lang = script_of(q["question"])

    rank = None
    if not q["should_reject"]:
        rank = next((i for i, c in enumerate(retrieved, 1) if matches_expected(c["original_filename"], q["expected_sources"])), None)

    facts_ok = None
    if not q["should_reject"] and q.get("answer_contains"):
        answer = _norm(r.answer)
        facts_ok = all(_norm(f) in answer for f in q["answer_contains"])

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

    return {**result, "status": r.status, "outcome": outcome, "answer": r.answer, "cited_files": cited,
            "retrieved_files": [c["original_filename"] for c in retrieved],
            "retrieved_scores": [c["score"] for c in retrieved], "rank": rank, "facts_ok": facts_ok,
            "question_lang": question_lang, "context_lang": context_lang,
            "cross_lingual": bool(context_lang and context_lang != question_lang),
            "signals": signals, "reasons": (r.answerability or {}).get("reasons", []), "latency_ms": r.latency_ms}


def run_evaluation(service, questions: list[dict], eval_top_k: int = 5) -> list[dict]:
    results = []
    for i, q in enumerate(questions, 1):
        res = _evaluate_one(service, q, eval_top_k)
        logger.info("eval %d/%d Q%s -> %s (%s)", i, len(questions), q["id"], res["outcome"], res["status"])
        results.append(res)
    return results


# ----------------------------------------------------------------------------------------------- summary
def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def summarize(results: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    pos = [r for r in results if not r["should_reject"] and r["outcome"] != "error"]
    neg = [r for r in results if r["should_reject"] and r["outcome"] != "error"]
    count = lambda rows, o: sum(r["outcome"] == o for r in rows)
    ranked = [r for r in pos if r["status"] != "no_documents_match_filter" or r["rank"] is not None]
    lat = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
    return {
        "questions": len(results), "errors": count(results, "error"),
        "answerable": {"n": len(pos), **{o: count(pos, o) for o in
                       ("correct", "right_source_missing_fact", "wrong_source", "false_refusal")},
                       "correct_rate": _rate(count(pos, "correct"), len(pos)),
                       "false_refusal_rate": _rate(count(pos, "false_refusal"), len(pos))},
        "unanswerable": {"n": len(neg), "correct_rejection": count(neg, "correct_rejection"), "leak": count(neg, "leak"),
                         "rejection_rate": _rate(count(neg, "correct_rejection"), len(neg))},
        "retrieval": {**{f"hit@{k}": _rate(sum(r["rank"] is not None and r["rank"] <= k for r in ranked), len(ranked)) for k in ks},
                      "mrr": round(sum(1 / r["rank"] for r in ranked if r["rank"]) / len(ranked), 3) if ranked else None},
        "latency_ms": {"p50": lat[len(lat) // 2] if lat else None,
                       "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None, "max": lat[-1] if lat else None},
    }


def summarize_groups(results: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[f"split={r.get('split')}"].append(r)
        groups[f"kind={r.get('kind')}"].append(r)
        if r.get("cross_lingual") is not None:
            groups["cross_lingual=" + ("yes" if r["cross_lingual"] else "no")].append(r)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def format_report(results: list[dict]) -> str:
    s = summarize(results)
    a, u, rt = s["answerable"], s["unanswerable"], s["retrieval"]
    lines = [
        f"QUESTIONS: {s['questions']}  (errors: {s['errors']})",
        f"ANSWERABLE   n={a['n']}: correct={a['correct']}  missing_fact={a['right_source_missing_fact']}  "
        f"wrong_source={a['wrong_source']}  false_refusal={a['false_refusal']}  -> correct_rate={a['correct_rate']}",
        f"UNANSWERABLE n={u['n']}: correctly_rejected={u['correct_rejection']}  leaked={u['leak']}  -> rejection_rate={u['rejection_rate']}",
        f"RETRIEVAL    hit@1={rt['hit@1']}  hit@3={rt['hit@3']}  hit@5={rt['hit@5']}  mrr={rt['mrr']}   (independent of the gate)",
        f"LATENCY ms   p50={s['latency_ms']['p50']}  p95={s['latency_ms']['p95']}  max={s['latency_ms']['max']}",
        "", "BY GROUP:  (ans = answerable: correct/false_refusal/n | unans = unanswerable: rejected/n)",
    ]
    for name, g in summarize_groups(results).items():
        ga, gu = g["answerable"], g["unanswerable"]
        lines.append(f"  {name:<34} ans {ga['correct']}/{ga['false_refusal']}/{ga['n']:<3}  unans {gu['correct_rejection']}/{gu['n']}")
    problems = [r for r in results if r["outcome"] not in ("correct", "correct_rejection")]
    lines += ["", f"NOT CORRECT ({len(problems)}):"]
    for r in problems:
        why = r.get("error") or "; ".join(r.get("reasons") or []) or f"cited={r.get('cited_files')}"
        lines.append(f"  Q{r['id']:<3} {r['outcome']:<26} {r['question'][:56]!r}  [{r.get('kind')}, "
                     f"{'cross' if r.get('cross_lingual') else 'same'}-lang]  {why[:90]}")
    return "\n".join(lines)


def save_results(path: Path, config, results: list[dict]) -> None:
    payload = {
        "config": {"model": config.model_name, "text_variant": config.embedding_text_variant, "top_k": config.top_k,
                   "min_score": config.min_score, "min_lexical_coverage": config.min_lexical_coverage,
                   "unknown_entity_check": config.unknown_entity_check,
                   "min_score_cross_lingual": config.min_score_cross_lingual,
                   "answer_support_ratio": config.answer_support_ratio},
        "summary": summarize(results), "groups": summarize_groups(results), "results": results,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
