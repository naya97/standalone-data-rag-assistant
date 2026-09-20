"""
Week 6 — Analytics Dashboard (Streamlit)

Displays the KPIs, views, and JOIN-based analysis built in
sql/week6/views.sql and sql/week6/queries.sql, against the
Week 4 pipeline output database (data/db/retail_pipeline.db).

Run with:
    streamlit run src/analytics/week6/app.py
"""

import os
import sqlite3
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# Locate the project root reliably regardless of where this file
# lives in the folder tree (marker-based, not a fixed dirname() count)
# ------------------------------------------------------------------
def find_project_root(marker: str = "requirements.txt") -> str:
    path = os.path.dirname(os.path.abspath(__file__))
    while path != os.path.dirname(path):
        if os.path.exists(os.path.join(path, marker)):
            return path
        path = os.path.dirname(path)
    raise FileNotFoundError(f"Could not find project root (looked for {marker})")


PROJECT_ROOT = find_project_root()
DB_PATH = os.path.join(PROJECT_ROOT, "data", "db", "retail_pipeline.db")
VIEWS_SQL_PATH = os.path.join(PROJECT_ROOT, "sql", "week6", "views.sql")


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    """One shared connection for the app's lifetime."""
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at: {DB_PATH}\n"
                 f"Run `python run_pipeline.py` first to generate it.")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def ensure_views_exist(conn: sqlite3.Connection) -> None:
    """
    (Re)creates the category_targets table and all views from
    views.sql. Safe to run every time the app starts, since the
    script uses CREATE TABLE/VIEW IF NOT EXISTS and INSERT OR REPLACE.
    """
    if not os.path.exists(VIEWS_SQL_PATH):
        st.warning(f"views.sql not found at {VIEWS_SQL_PATH} — "
                   f"skipping view creation (they may already exist).")
        return
    with open(VIEWS_SQL_PATH, "r", encoding="utf-8") as f:
        script = f.read()
    conn.executescript(script)
    conn.commit()


def run_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Week 6 — Retail Analytics", layout="wide")
st.title("Retail Sales Analytics Dashboard")
st.caption("Week 6 — Analytics and simple data modeling")

conn = get_connection()
ensure_views_exist(conn)

# ------------------------------------------------------------------
# Top-level KPIs
# ------------------------------------------------------------------
st.header("Key Performance Indicators")

kpi_cols = st.columns(3)

avg_order_value = run_query(conn, "SELECT AVG(total_spent) AS v FROM sales_clean")["v"][0]
total_revenue = run_query(conn, "SELECT SUM(total_spent) AS v FROM sales_clean")["v"][0]
total_transactions = run_query(conn, "SELECT COUNT(*) AS v FROM sales_clean")["v"][0]

kpi_cols[0].metric("Total Revenue", f"${total_revenue:,.2f}")
kpi_cols[1].metric("Average Order Value", f"${avg_order_value:,.2f}")
kpi_cols[2].metric("Total Transactions", f"{total_transactions:,}")

st.divider()

# ------------------------------------------------------------------
# Monthly revenue trend
# ------------------------------------------------------------------
st.header("Monthly Revenue Trend")
monthly_df = run_query(conn, "SELECT * FROM monthly_sales_summary")
st.line_chart(monthly_df.set_index("sale_month")["total_revenue"])
st.dataframe(monthly_df, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Category performance (JOIN with category_targets)
# ------------------------------------------------------------------
st.header("Category Performance vs. Target")
category_perf_df = run_query(
    conn,
    """
    SELECT category, total_revenue, revenue_target, target_achievement_pct
    FROM category_performance
    ORDER BY target_achievement_pct DESC
    """,
)
st.bar_chart(category_perf_df.set_index("category")["target_achievement_pct"])
st.dataframe(category_perf_df, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Underperforming categories (raw JOIN + calculated field)
# ------------------------------------------------------------------
st.header("Underperforming Categories (Revenue Gap)")
gap_df = run_query(
    conn,
    """
    SELECT
        s.category,
        SUM(s.total_spent) AS total_revenue,
        ct.revenue_target,
        SUM(s.total_spent) - ct.revenue_target AS revenue_gap
    FROM sales_clean s
    JOIN category_targets ct
        ON s.category = ct.category
    GROUP BY s.category, ct.revenue_target
    HAVING revenue_gap < 0
    ORDER BY revenue_gap ASC
    """,
)
if gap_df.empty:
    st.success("No categories are currently below their revenue target.")
else:
    st.dataframe(gap_df, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Payment method breakdown
# ------------------------------------------------------------------
st.header("Revenue Share by Payment Method")
payment_df = run_query(
    conn,
    """
    SELECT
        payment_method,
        total_revenue,
        ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM payment_method_summary), 2) AS revenue_share_pct
    FROM payment_method_summary
    """,
)
col1, col2 = st.columns([1, 1])
col1.bar_chart(payment_df.set_index("payment_method")["total_revenue"])
col2.dataframe(payment_df, use_container_width=True)


# streamlit run src/analytics/week6/app.py