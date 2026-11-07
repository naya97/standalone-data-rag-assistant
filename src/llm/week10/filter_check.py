from __future__ import annotations

import re
from typing import Any, Callable

from src.llm.week10.arguments import MetadataCatalog, clean_search_arguments
from src.llm.week10.evaluation import Case

REFUSALS = ("false_refusal", "retrieval_miss", "right_source_missing_fact")


def repair_args(question: str, args: dict, catalog: MetadataCatalog | None = None) -> tuple[dict, list[str]]:
    """(arguments after the backend's cleaning, what was changed): the production rules, see arguments.py."""
    return clean_search_arguments(question, args, catalog)


def select_rows(rows: list[dict], all_answerable: bool = False) -> list[dict]:
    """The document rows worth re-checking: a search was made, and its arguments looked wrong, or the answer failed.
    all_answerable=True also takes every answerable question (the whole table: how often is the fact retrieved at this top_k?)."""
    picked = []
    for row in rows:
        if row.get("domain") != "documents" or row.get("tool") != "search_documents":
            continue
        flagged = any(i.startswith(("format:", "assumed:", "invented:")) for i in row.get("issues", []))
        if flagged or row.get("outcome") in REFUSALS or (all_answerable and row.get("kind") == "answerable"):
            picked.append(row)
    return picked


def _retrieve(retriever: Any, query: str, filters: dict | None, ks: tuple[int, ...], expected: tuple[str, ...],
              matches: Callable[[str, Any], bool], facts: tuple[str, ...] = (), check_facts: Callable | None = None) -> dict:
    """One retrieval at the largest k. The ranking of k passages is the prefix of the ranking of a larger k, so the recall at every
    smaller k is read from the same list (no extra request, no extra retrieval)."""
    top_k = max(ks)
    try:
        found = retriever.retrieve(query, top_k=top_k, filters=filters or None)
    except Exception as exc:  # noqa: BLE001 - an index problem must show up in the table, not stop it
        return {"n": None, "rank": None, "error": f"{type(exc).__name__}: {exc}"[:100]}
    chunks = list(found.chunks)
    rank = next((i for i, c in enumerate(chunks, 1) if expected and matches(c.original_filename, list(expected))), None)

    def fact_in(prefix: list) -> bool | None:
        if not (facts and check_facts is not None):
            return None
        # only the passages of the expected source count: the same words in another document prove nothing
        relevant = [c for c in prefix if not expected or matches(c.original_filename, list(expected))]
        return bool(relevant) and bool(check_facts(" ".join(c.text for c in relevant), list(facts))[0])

    by_k = {k: {"source": rank is not None and rank <= k, "fact": fact_in(chunks[:k])} for k in sorted(ks)}
    return {"n": len(chunks), "rank": rank, "status": found.status, "fact": by_k[top_k]["fact"], "by_k": by_k}


def check_rows(rows: list[dict], cases: dict[int, Case], retriever: Any, matches: Callable[[str, Any], bool],
               top_k: int = 3, check_facts: Callable | None = None, all_answerable: bool = False,
               ks: tuple[int, ...] | None = None) -> list[dict]:
    ks = tuple(sorted(set(ks))) if ks else (top_k,)
    out = []
    for row in select_rows(rows, all_answerable):
        case = cases.get(row["id"])
        args = row.get("args") or {}
        query = args.get("query") or row["question"]
        expected = case.expected_sources if case else ()
        facts = case.answer_contains if case else ()
        repaired, notes = repair_args(row["question"], args)
        filters = {k: args[k] for k in ("project", "document_type", "month") if args.get(k)}
        fixed = {k: repaired[k] for k in ("project", "document_type", "month") if repaired.get(k)}
        out.append({"id": row["id"], "kind": row["kind"], "question": row["question"], "outcome": row["outcome"],
                    "should_reject": bool(case and case.should_reject), "has_expected": bool(expected),
                    "args": {k: v for k, v in args.items() if k != "query"}, "notes": notes,
                    "as_chosen": _retrieve(retriever, query, filters, ks, expected, matches, facts, check_facts),
                    "repaired": _retrieve(retriever, query, fixed, ks, expected, matches, facts, check_facts),
                    "no_filters": _retrieve(retriever, query, None, ks, expected, matches, facts, check_facts)})
    return out


def _cell(result: dict) -> str:
    if result.get("error"):
        return "error"
    fact = "" if result.get("fact") is None else (" fact=Y" if result["fact"] else " fact=N")
    return f"n={result['n']} rank={result['rank'] if result['rank'] else '-'}{fact}"


def format_check(results: list[dict], top_k: int = 3) -> str:
    lines = [f"{'Q':<6}{'kind':<20}{'as chosen':<23}{'repaired':<23}{'no filters':<23}what the cleaning changed"]
    for r in results:
        lines.append(f"Q{r['id']:<5}{r['kind'][:19]:<20}{_cell(r['as_chosen']):<23}{_cell(r['repaired']):<23}{_cell(r['no_filters']):<23}{'; '.join(r['notes']) or '-'}")
    answerable = [r for r in results if r["has_expected"]]
    top_k = max([k for r in results for k in (r["repaired"].get("by_k") or {})] or [top_k])   # the numbers below are measured at the largest k asked
    found = lambda variant: sum(1 for r in answerable if r[variant].get("rank"))   # noqa: E731
    with_fact = [r for r in answerable if r["repaired"].get("fact") is not None]
    mechanical = [r for r in results if r["should_reject"] and r["as_chosen"].get("n") == 0 and (r["repaired"].get("n") or 0) > 0]
    lines += ["", f"ANSWERABLE questions re-checked: {len(answerable)}; the expected source is in the top {top_k}:",
              f"  as chosen {found('as_chosen')}   repaired {found('repaired')}   no filters {found('no_filters')}"]
    if with_fact:
        fact_in = lambda variant: sum(1 for r in with_fact if r[variant].get("fact"))   # noqa: E731
        lines += [f"the expected FACT is in the retrieved text at k={top_k} (of {len(with_fact)} questions with a checkable fact):",
                  f"  as chosen {fact_in('as_chosen')}   repaired {fact_in('repaired')}   no filters {fact_in('no_filters')}",
                  f"  repaired but the fact is NOT in the text: {[r['id'] for r in with_fact if not r['repaired'].get('fact')]}"]
    lines += [
              f"questions that should be REFUSED but were refused only because the filter matched nothing "
              f"(the model never read a passage): {len(mechanical)}  {[r['id'] for r in mechanical]}"]
    ks = sorted({k for r in answerable for k in (r["repaired"].get("by_k") or {})})
    if len(ks) > 1 and with_fact:
        lines += ["", f"the expected FACT is in the retrieved text, by k (of {len(with_fact)} questions with a checkable fact):"]
        for variant, label in (("as_chosen", "as chosen"), ("repaired", "repaired"), ("no_filters", "no filters")):
            counts = [sum(1 for r in with_fact if (r[variant].get("by_k") or {}).get(k, {}).get("fact")) for k in ks]
            lines.append(f"  {label:<11}" + "   ".join(f"k={k}: {c}" for k, c in zip(ks, counts)))
        first = {r["id"]: next((k for k in ks if r["repaired"]["by_k"][k]["fact"]), None) for r in with_fact}
        for k in ks[1:]:
            gained = [i for i, v in first.items() if v == k]
            lines.append(f"  found only from k={k} (repaired): {gained}")
        lines.append(f"  not found even at k={ks[-1]} (repaired): {[i for i, v in first.items() if v is None]}")
    if any(r["as_chosen"].get("error") for r in results):
        lines.append("some retrievals raised an error: see the 'error' cells in the JSON of this check")
    return "\n".join(lines)
