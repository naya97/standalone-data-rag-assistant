from __future__ import annotations

import re
from dataclasses import dataclass

#  checking: what the question does not support
_MONTHS = {1: ("january", "يناير"), 2: ("february", "فبراير"), 3: ("march", "مارس"), 4: ("april", "أبريل", "ابريل"), 5: ("may", "مايو"),
           6: ("june", "يونيو"), 7: ("july", "يوليو"), 8: ("august", "أغسطس", "اغسطس"), 9: ("september", "سبتمبر"),
           10: ("october", "أكتوبر", "اكتوبر"), 11: ("november", "نوفمبر"), 12: ("december", "ديسمبر")}
_DOC_TYPE_WORDS = {"invoice": ("invoice", "فاتور", "فواتير"), "policy": ("polic", "سياس"), "report": ("report", "تقرير", "تقارير")}
_PROJECT_STORED = re.compile(r"^project[A-Z][A-Za-z0-9]*$")
_PROJECT_IN_TEXT = re.compile(r"(?:project|مشروع)\s+([A-Z][A-Za-z0-9]*)", re.IGNORECASE)


def argument_issues(question: str, args: dict) -> list[str]:
   
    low, issues = question.lower(), []
    project = args.get("project")
    if project:
        if not _PROJECT_STORED.match(project):
            issues.append(f"format: project {project!r} is not in the stored form (like projectAlpha); an exact filter matches nothing")
        elif project[len("project"):].lower() not in low:
            issues.append(f"invented: project {project!r} is not named in the question")
    elif _PROJECT_IN_TEXT.search(question):
        issues.append("missed: the question names a project but no project filter was used")
    kind = args.get("document_type")
    if kind and not any(word in low for word in _DOC_TYPE_WORDS.get(kind, ())):
        issues.append(f"invented: document_type {kind!r} is not mentioned in the question")
    month = args.get("month")
    if month and re.match(r"^\d{4}-(0[1-9]|1[0-2])$", month):
        year, number = month.split("-")[0], int(month.split("-")[1])
        named = month in question or any(name in low for name in _MONTHS[number])
        if not named:
            issues.append(f"invented: month {month!r} is not named in the question")
        elif year not in question:
            issues.append(f"assumed: month {month!r}: the question names the month but not the year")
    return issues


# cleaning
_MONTH_VALUE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_YEAR_IN_TEXT = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


@dataclass(frozen=True)
class MetadataCatalog:

    projects: frozenset[str] = frozenset()
    months: frozenset[str] = frozenset()

    @classmethod
    def from_rows(cls, rows) -> "MetadataCatalog":
        get = (lambda r, k: r.get(k)) if hasattr(next(iter(rows), {}), "get") else (lambda r, k: getattr(r, k, None))
        rows = list(rows)
        return cls(frozenset(v for r in rows if (v := get(r, "project"))), frozenset(v for r in rows if (v := get(r, "month"))))


def canonical_project(value: str) -> str | None:
    name = re.sub(r"[\s_\-]+", "", re.sub(r"(?i)^\s*project[\s_\-]*", "", value or ""))
    return "project" + name[0].upper() + name[1:] if name else None


def _clean_month(question: str, month: str, catalog: MetadataCatalog | None, notes: list[str]) -> str | None:
    match = _MONTH_VALUE.match(month)
    if not match:
        return month                                   # not our business: the argument validation rejects a bad format
    year, number = match.group(1), int(match.group(2))
    low = question.lower()
    if not (month in question or any(name in low for name in _MONTHS[number])):
        notes.append(f"month {month!r} dropped (the question does not mention that month)")
        return None
    stated = set(_YEAR_IN_TEXT.findall(question))
    if stated:
        if year in stated:
            return month
        if len(stated) == 1:                           # the question gives a different year: the question wins
            fixed = f"{next(iter(stated))}-{match.group(2)}"
            notes.append(f"month {month!r} -> {fixed!r} (the year stated in the question)")
            return fixed
        return month
    if catalog is None or not catalog.months:
        notes.append(f"month {month!r} dropped (the question does not state the year)")
        return None
    candidates = sorted(m for m in catalog.months if m.endswith("-" + match.group(2)))
    if len(candidates) == 1:
        notes.append(f"month {month!r} -> {candidates[0]!r} (the question does not state the year; this is the only such month in the index)")
        return candidates[0]
    if not candidates:
        notes.append(f"month {month!r} kept: no indexed document is from that month, so nothing will match")
        return month
    notes.append(f"month {month!r} dropped: the year is not stated and the index has {candidates}, so the question is ambiguous")
    return None


def clean_search_arguments(question: str, args: dict, catalog: MetadataCatalog | None = None) -> tuple[dict, list[str]]:
    out, notes = dict(args), []
    project = out.get("project")
    if isinstance(project, str):
        canonical = canonical_project(project)
        if canonical is None:
            out.pop("project")
            notes.append(f"project {project!r} dropped (no name in it)")
        else:
            if canonical != project:
                out["project"] = canonical
                notes.append(f"project {project!r} -> {canonical!r}")
            if catalog and catalog.projects and canonical not in catalog.projects:
                notes.append(f"project {canonical!r} is not in the index: nothing will match, the answer will be 'not found'")
    month = out.get("month")
    if isinstance(month, str):
        cleaned = _clean_month(question or "", month, catalog, notes)
        if cleaned is None:
            out.pop("month")
        else:
            out["month"] = cleaned
    return out, notes
