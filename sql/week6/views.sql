-- sql/week6/views.sql

-- ============================================================
-- Additional dimension table (not present in the source dataset)
-- Represents a hypothetical business target per category, added
-- specifically to enable a real JOIN exercise, since sales_clean
-- is a single flat fact table with no natural table to join against.
--
-- NOTE: revenue_target represents a target for the FULL dataset
-- period (not literally monthly), since sales_clean spans multiple
-- months and comparing to a single "monthly" figure would be
-- mathematically misleading without a matching monthly breakdown.
-- ============================================================
CREATE TABLE IF NOT EXISTS category_targets (
    category TEXT PRIMARY KEY,
    revenue_target REAL NOT NULL
);

INSERT OR REPLACE INTO category_targets (category, revenue_target) VALUES
    ('Patisserie', 30000),
    ('Milk Products', 32000),
    -- ('Milk Products', 500000),
    ('Butchers', 32000),
    ('Beverages', 32000),
    ('Food', 32000),
    ('Furniture', 33000),
    ('Electric household essentials', 33000),
    ('Computers and electric accessories', 33000);

-- Monthly summary view
CREATE VIEW IF NOT EXISTS monthly_sales_summary AS
SELECT
    strftime('%Y-%m', transaction_date) AS sale_month,
    COUNT(*) AS transaction_count,
    SUM(total_spent) AS total_revenue,
    AVG(total_spent) AS avg_order_value
FROM sales_clean
GROUP BY sale_month
ORDER BY sale_month;

-- Category summary view
CREATE VIEW IF NOT EXISTS category_summary AS
SELECT
    category,
    COUNT(*) AS transaction_count,
    SUM(total_spent) AS total_revenue,
    SUM(quantity) AS total_units_sold
FROM sales_clean
GROUP BY category;

-- Payment method summary view
CREATE VIEW IF NOT EXISTS payment_method_summary AS
SELECT
    payment_method,
    COUNT(*) AS transaction_count,
    SUM(total_spent) AS total_revenue
FROM sales_clean
GROUP BY payment_method;

-- ============================================================
-- Category performance view (JOIN + calculated field)
-- Combines the category_summary view with the category_targets
-- dimension table to compute how much of each category's target
-- has actually been achieved.
-- ============================================================
CREATE VIEW IF NOT EXISTS category_performance AS
SELECT
    cs.category,
    cs.total_revenue,
    ct.revenue_target,
    ROUND(cs.total_revenue * 100.0 / ct.revenue_target, 2) AS target_achievement_pct
FROM category_summary cs
JOIN category_targets ct
    ON cs.category = ct.category;