"""The numbers behind the sales insights, computed by SQL and Python, never by the model.

The model only words what is here; guardrails then check that every number it writes appears in these facts.
Reads `sales_clean` (Week 4 pipeline output) read-only; `category_targets` (Week 6) is used when it exists.
"""

from __future__ import annotations

import calendar
import re
import sqlite3
from datetime import date
from pathlib import Path

MAX_MONTHS_SHOWN = 24
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class DataUnavailableError(Exception):
    """The retail database or the sales_clean table is missing: run `python run_pipeline.py` first."""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise DataUnavailableError(f"Database not found: {path}. Run `python run_pipeline.py` first.")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)  # read-only: this module never writes


def _where(category: str | None, start: str | None, end: str | None) -> tuple[str, dict]:
    clauses, params = [], {}
    if category is not None:
        clauses.append("category = :category")
        params["category"] = category
    if start is not None:
        clauses.append("strftime('%Y-%m', transaction_date) >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("strftime('%Y-%m', transaction_date) <= :end")
        params["end"] = end
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def _r(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def _incomplete_months(con: sqlite3.Connection) -> tuple[dict[str, str], date | None, date | None]:
    """The months of the WHOLE dataset that do not start on day 1 / end on the last day, with the reason.
    Judged on the whole table, not on the scope: a category that sold nothing on the 31st is not an incomplete month."""
    first, last = con.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM sales_clean").fetchone()
    if first is None or last is None:
        return {}, None, None
    first, last = date.fromisoformat(str(first)[:10]), date.fromisoformat(str(last)[:10])
    found = {}
    if first.day != 1:
        found[f"{first:%Y-%m}"] = f"the data starts on {first.isoformat()}"
    if last.day != calendar.monthrange(last.year, last.month)[1]:
        found[f"{last:%Y-%m}"] = f"the data ends on {last.isoformat()}"
    return found, first, last


def _breakdown(con: sqlite3.Connection, column: str, where: str, params: dict, total: float) -> list[dict]:
    """Revenue per value of a fixed, trusted column name (never user input)."""
    rows = con.execute(
        f"SELECT {column}, COUNT(*), COALESCE(SUM(total_spent), 0) FROM sales_clean {where} "
        f"GROUP BY {column} ORDER BY SUM(total_spent) DESC", params).fetchall()
    return [{column: name, "transactions": n, "revenue": _r(rev), "share_pct": _r(rev * 100 / total) if total else None}
            for name, n, rev in rows]


def get_sales_facts(db_path: str | Path, category: str | None = None, start: str | None = None,
                    end: str | None = None) -> dict:
    """Facts for the whole dataset or a slice of it (a category and/or a month range 'YYYY-MM').
    Raises DataUnavailableError (database/table missing) or ValueError (bad scope, or nothing matches it)."""
    for name, value in (("start", start), ("end", end)):
        if value is not None and not _MONTH.match(value):
            raise ValueError(f"{name} must look like 2024-03, got {value!r}")
    if start and end and start > end:
        raise ValueError(f"start ({start}) must not be after end ({end})")

    con = _connect(db_path)
    try:
        try:
            categories = [r[0] for r in con.execute("SELECT DISTINCT category FROM sales_clean ORDER BY category")]
        except sqlite3.OperationalError as exc:
            raise DataUnavailableError(f"Table sales_clean cannot be read ({exc}). Run `python run_pipeline.py` first.") from exc
        if category is not None and category not in categories:
            raise ValueError(f"Unknown category {category!r}. Available: {categories}")

        where, params = _where(category, start, end)
        n, revenue, average, units = con.execute(
            f"SELECT COUNT(*), COALESCE(SUM(total_spent), 0), COALESCE(AVG(total_spent), 0), COALESCE(SUM(quantity), 0) "
            f"FROM sales_clean {where}", params).fetchone()
        if n == 0:
            raise ValueError("No sales match this scope (category / months).")

        monthly = [(m, c, rev) for m, c, rev in con.execute(
            f"SELECT strftime('%Y-%m', transaction_date) AS m, COUNT(*), COALESCE(SUM(total_spent), 0) "
            f"FROM sales_clean {where} GROUP BY m ORDER BY m", params) if m is not None]
        incomplete, data_first, data_last = _incomplete_months(con)
        by_month, previous, previous_incomplete = [], None, False
        for m, c, rev in monthly:
            is_incomplete = m in incomplete
            # growth against or from an incomplete month is meaningless (it would read as a collapse): left empty
            growth = _r((rev - previous) * 100 / previous) if previous and not is_incomplete and not previous_incomplete else None
            entry = {"month": m, "transactions": c, "revenue": _r(rev), "mom_growth_pct": growth}
            if is_incomplete:
                entry["incomplete"] = True
            by_month.append(entry)
            previous, previous_incomplete = rev, is_incomplete
        full_months = [r for r in monthly if r[0] not in incomplete] or monthly  # best/worst only among complete months
        best = max(full_months, key=lambda r: r[2]) if full_months else None
        worst = min(full_months, key=lambda r: r[2]) if full_months else None

        by_category = _breakdown(con, "category", where, params, revenue)
        notes = [f"{m} is an incomplete month ({incomplete[m]}): it is left out of best_month, worst_month and growth."
                 for m, _, _ in monthly if m in incomplete]
        try:
            targets = dict(con.execute("SELECT category, revenue_target FROM category_targets").fetchall())
        except sqlite3.OperationalError:
            targets = {}
        if targets and (start or end):
            notes.append("Revenue targets are omitted: they cover the whole dataset period, not a slice of it.")
            targets = {}
        for row in by_category:
            target = targets.get(row["category"])
            row["revenue_target"] = _r(target)
            row["target_achievement_pct"] = _r(row["revenue"] * 100 / target) if target else None
        definitions = {"share_pct": "percentage of the total revenue of this scope",
                       "mom_growth_pct": "percentage change in revenue compared with the previous month (null when not meaningful)"}
        if any(r["target_achievement_pct"] is not None for r in by_category):
            definitions["target_achievement_pct"] = ("revenue as a percentage of the category's revenue target: "
                                                     "100 means exactly on target, above 100 means ahead of target. "
                                                     "The targets are hypothetical figures added for this project, not source data")

        shown = by_month[-MAX_MONTHS_SHOWN:]
        if len(shown) < len(by_month):
            notes.append(f"by_month lists only the latest {MAX_MONTHS_SHOWN} of {len(by_month)} months; "
                         "best_month and worst_month use all of them.")
        return {
            "scope": {"category": category, "first_month": monthly[0][0] if monthly else None,
                      "last_month": monthly[-1][0] if monthly else None, "months_total": len(by_month),
                      "months_shown": len(shown)},
            "data_covers": {"from": data_first.isoformat() if data_first else None, "to": data_last.isoformat() if data_last else None},
            "overall": {"transactions": n, "total_revenue": _r(revenue), "average_order_value": _r(average),
                        "units_sold": int(units)},
            "best_month": {"month": best[0], "revenue": _r(best[2])} if best else None,
            "worst_month": {"month": worst[0], "revenue": _r(worst[2])} if worst else None,
            "by_month": shown,
            "by_category": by_category,
            "by_payment_method": _breakdown(con, "payment_method", where, params, revenue),
            "by_location": _breakdown(con, "location", where, params, revenue),
            "definitions": definitions,
            "notes": notes,
        }
    finally:
        con.close()
