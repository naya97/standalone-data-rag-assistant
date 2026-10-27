"""Sales facts: numbers checked against hand calculation on a tiny database. No network, no model."""

import sqlite3

import pytest

from src.llm.week9.data_facts import DataUnavailableError, get_sales_facts

ROWS = [  # id, category, qty, total_spent, payment, location, date
    ("T1", "Food", 2, 100.0, "Cash", "Online", "2024-01-01 00:00:00"),
    ("T2", "Food", 1, 50.0, "Card", "In-store", "2024-01-20 00:00:00"),
    ("T3", "Furniture", 1, 300.0, "Card", "Online", "2024-02-10 00:00:00"),
    ("T4", "Food", 3, 150.0, "Cash", "In-store", "2024-02-15 00:00:00"),
    ("T5", "Furniture", 1, 200.0, "Card", "Online", "2024-03-31 00:00:00"),
]


def make_db(path, rows=ROWS, targets=True):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE sales_clean (transaction_id TEXT, customer_id TEXT, category TEXT, item TEXT,
                   price_per_unit REAL, quantity INTEGER, total_spent REAL, payment_method TEXT, location TEXT,
                   transaction_date TIMESTAMP, discount_applied INTEGER)""")
    con.executemany("INSERT INTO sales_clean VALUES (?, 'C', ?, 'i', 1.0, ?, ?, ?, ?, ?, 0)", rows)
    if targets:
        con.execute("CREATE TABLE category_targets (category TEXT PRIMARY KEY, revenue_target REAL NOT NULL)")
        con.executemany("INSERT INTO category_targets VALUES (?, ?)", [("Furniture", 1000.0), ("Food", 300.0)])
    con.commit(); con.close()
    return path


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "retail.db")


def test_overall_numbers(db):
    o = get_sales_facts(db)["overall"]
    assert o == {"transactions": 5, "total_revenue": 800.0, "average_order_value": 160.0, "units_sold": 8}


def test_months_growth_best_and_worst(db):
    f = get_sales_facts(db)
    assert [(m["month"], m["revenue"], m["transactions"]) for m in f["by_month"]] == [("2024-01", 150.0, 2), ("2024-02", 450.0, 2), ("2024-03", 200.0, 1)]
    assert [m["mom_growth_pct"] for m in f["by_month"]] == [None, 200.0, -55.56]
    assert f["best_month"] == {"month": "2024-02", "revenue": 450.0} and f["worst_month"] == {"month": "2024-01", "revenue": 150.0}
    assert f["scope"] == {"category": None, "first_month": "2024-01", "last_month": "2024-03", "months_total": 3, "months_shown": 3}


def test_shares_and_targets_for_the_whole_period(db):
    f = get_sales_facts(db)
    by_cat = {c["category"]: c for c in f["by_category"]}
    assert [c["category"] for c in f["by_category"]] == ["Furniture", "Food"]  # highest revenue first
    assert by_cat["Furniture"]["revenue"] == 500.0 and by_cat["Furniture"]["share_pct"] == 62.5
    assert by_cat["Furniture"]["target_achievement_pct"] == 50.0 and by_cat["Food"]["target_achievement_pct"] == 100.0
    assert {p["payment_method"]: p["share_pct"] for p in f["by_payment_method"]} == {"Card": 68.75, "Cash": 31.25}
    assert {p["location"]: p["revenue"] for p in f["by_location"]} == {"Online": 600.0, "In-store": 200.0}


def test_a_category_scope(db):
    f = get_sales_facts(db, category="Food")
    assert f["overall"]["transactions"] == 3 and f["overall"]["total_revenue"] == 300.0
    assert [c["category"] for c in f["by_category"]] == ["Food"] and f["by_category"][0]["share_pct"] == 100.0


def test_a_month_range_drops_the_targets_because_they_cover_the_whole_period(db):
    f = get_sales_facts(db, start="2024-02", end="2024-03")
    assert f["overall"]["transactions"] == 3 and f["scope"]["first_month"] == "2024-02"
    assert all(c["revenue_target"] is None and c["target_achievement_pct"] is None for c in f["by_category"])
    assert any("targets are omitted" in n for n in f["notes"])


def test_works_without_the_targets_table(tmp_path):
    f = get_sales_facts(make_db(tmp_path / "r.db", targets=False))
    assert all(c["revenue_target"] is None for c in f["by_category"]) and f["notes"] == []


def test_only_the_latest_24_months_are_listed_but_best_and_worst_use_all(tmp_path):
    day = lambda i: 1 if i == 0 else (30 if i == 29 else 15)  # first and last months complete (2024-06 ends on the 30th)
    rows = [(f"T{i}", "Food", 1, float(100 + i), "Cash", "Online", f"{2022 + i // 12}-{i % 12 + 1:02d}-{day(i):02d} 00:00:00") for i in range(30)]
    f = get_sales_facts(make_db(tmp_path / "long.db", rows))
    assert len(f["by_month"]) == 24 and f["scope"]["months_total"] == 30 and f["scope"]["months_shown"] == 24
    assert f["by_month"][0]["month"] == "2022-07" and f["worst_month"]["month"] == "2022-01"
    assert any("latest 24 of 30" in n for n in f["notes"])


@pytest.mark.parametrize("kwargs, message", [
    ({"category": "Nope"}, "Unknown category"), ({"start": "2024-13"}, "start must look like"),
    ({"end": "March"}, "end must look like"), ({"start": "2024-03", "end": "2024-01"}, "must not be after"),
    ({"start": "2030-01"}, "No sales match"),
])
def test_bad_scopes_are_refused_with_a_reason(db, kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_sales_facts(db, **kwargs)


def test_user_text_is_never_run_as_sql(db):
    with pytest.raises(ValueError, match="Unknown category"):
        get_sales_facts(db, category="Food'; DROP TABLE sales_clean; --")
    assert get_sales_facts(db)["overall"]["transactions"] == 5  # the table is still there


def test_missing_database_or_table_says_how_to_fix_it(tmp_path):
    with pytest.raises(DataUnavailableError, match="run_pipeline"):
        get_sales_facts(tmp_path / "missing.db")
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(DataUnavailableError, match="sales_clean"):
        get_sales_facts(empty)


def test_the_database_is_opened_read_only(db):
    from src.llm.week9.data_facts import _connect
    con = _connect(db)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM sales_clean")
    con.close()


# ----------------------------------------------------------------------------------------------- incomplete months (found in the real data)
def partial_db(tmp_path):
    """The real dataset ends in the middle of its last month: 2025-01 had 202 transactions against ~327 in a normal month."""
    rows = [("T1", "Food", 1, 100.0, "Cash", "Online", "2024-01-01 00:00:00"), ("T2", "Food", 1, 100.0, "Cash", "Online", "2024-01-31 00:00:00"),
            ("T3", "Food", 1, 300.0, "Cash", "Online", "2024-02-01 00:00:00"), ("T4", "Food", 1, 300.0, "Cash", "Online", "2024-02-29 00:00:00"),
            ("T5", "Food", 1, 20.0, "Cash", "Online", "2024-03-01 00:00:00"), ("T6", "Food", 1, 20.0, "Cash", "Online", "2024-03-09 00:00:00")]
    return make_db(tmp_path / "partial.db", rows)


def test_an_incomplete_last_month_is_flagged_and_never_the_weakest_month(tmp_path):
    f = get_sales_facts(partial_db(tmp_path))
    march = f["by_month"][-1]
    assert march["month"] == "2024-03" and march["incomplete"] is True and march["mom_growth_pct"] is None  # not "-93%"
    assert f["worst_month"] == {"month": "2024-01", "revenue": 200.0} and f["best_month"] == {"month": "2024-02", "revenue": 600.0}
    assert f["data_covers"] == {"from": "2024-01-01", "to": "2024-03-09"}
    assert any("2024-03 is an incomplete month (the data ends on 2024-03-09)" in n for n in f["notes"])
    assert "incomplete" not in f["by_month"][0]


def test_an_incomplete_first_month_is_flagged_and_so_is_growth_out_of_it(tmp_path):
    rows = [("T1", "Food", 1, 10.0, "Cash", "Online", "2024-01-20 00:00:00"), ("T2", "Food", 1, 90.0, "Cash", "Online", "2024-02-01 00:00:00"),
            ("T3", "Food", 1, 90.0, "Cash", "Online", "2024-02-29 00:00:00"), ("T4", "Food", 1, 95.0, "Cash", "Online", "2024-03-01 00:00:00"),
            ("T5", "Food", 1, 95.0, "Cash", "Online", "2024-03-31 00:00:00")]
    f = get_sales_facts(make_db(tmp_path / "p.db", rows))
    assert [m.get("incomplete") for m in f["by_month"]] == [True, None, None]
    assert [m["mom_growth_pct"] for m in f["by_month"]] == [None, None, 5.56]  # February is not "+800%" over a partial January
    assert f["worst_month"]["month"] == "2024-02"


def test_incompleteness_is_judged_on_the_whole_dataset_not_on_the_slice(tmp_path):
    db = partial_db(tmp_path)
    con = sqlite3.connect(db)
    con.execute("INSERT INTO sales_clean VALUES ('T9', 'C', 'Furniture', 'i', 1.0, 1, 50.0, 'Card', 'Online', '2024-02-10 00:00:00', 0)")
    con.commit(); con.close()
    f = get_sales_facts(db, category="Furniture")  # a category with one sale in the middle of a month
    assert f["by_month"][0] == {"month": "2024-02", "transactions": 1, "revenue": 50.0, "mom_growth_pct": None}  # no "incomplete"
    assert f["notes"] == []


def test_a_slice_that_excludes_the_incomplete_month_has_no_warning(tmp_path):
    f = get_sales_facts(partial_db(tmp_path), start="2024-01", end="2024-02")
    assert not any("incomplete" in n for n in f["notes"]) and all("incomplete" not in m for m in f["by_month"])


def test_definitions_give_the_model_the_reference_numbers_it_needs(db, tmp_path):
    f = get_sales_facts(db)
    assert "100 means exactly on target" in f["definitions"]["target_achievement_pct"]  # the "100" the model kept writing
    assert "target_achievement_pct" not in get_sales_facts(make_db(tmp_path / "n.db", targets=False))["definitions"]
