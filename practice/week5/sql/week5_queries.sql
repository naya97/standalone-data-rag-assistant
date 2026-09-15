-- ============================================================
-- Week 5: BigQuery-equivalent analytical queries (local SQLite)
-- Cost awareness principle applied throughout:
-- only required columns are selected (never SELECT *), since in
-- BigQuery cost is based on bytes scanned per column, not per row.
-- ============================================================

-- 1. Total revenue by category
-- Cost note: selecting only (category, total_spent) avoids scanning
-- unrelated columns like customer_id or transaction_date.
SELECT category, SUM(total_spent) AS total_revenue
FROM sales_clean
GROUP BY category
ORDER BY total_revenue DESC;

-- 2. Top 5 items by quantity sold
-- Cost note: aggregating on (item, quantity) only, not the full row.
SELECT item, SUM(quantity) AS total_sold
FROM sales_clean
GROUP BY item
ORDER BY total_sold DESC
LIMIT 5;

-- 3. Daily transaction count trend
-- Cost note: in real BigQuery, this table would ideally be
-- partitioned by transaction_date to avoid a full-table scan
-- every time this query runs.
SELECT DATE(transaction_date) AS sale_date, COUNT(transaction_id) AS total_transactions
FROM sales_clean
GROUP BY sale_date
ORDER BY sale_date ASC;

-- 4. Average transaction value
-- Cost note: total_spent is already validated (= quantity * price_per_unit),
-- so it's reused directly instead of recomputing from two columns.
SELECT AVG(total_spent) AS avg_transaction_value
FROM sales_clean;

-- 5. Revenue by discount status
-- Cost note: expected result should show near-identical average revenue
-- regardless of discount_applied, consistent with the Week 3 finding
-- that the discount column has no measurable effect on total_spent.
SELECT discount_applied, AVG(total_spent) AS avg_revenue, COUNT(*) AS transaction_count
FROM sales_clean
GROUP BY discount_applied;

-- 6. Revenue by payment method
-- Cost note: only (payment_method, total_spent) scanned, not the full table.
SELECT payment_method, SUM(total_spent) AS total_revenue
FROM sales_clean
GROUP BY payment_method
ORDER BY total_revenue DESC;

-- 7. Revenue by location (Online vs In-store)
SELECT location, SUM(total_spent) AS total_revenue, COUNT(*) AS transaction_count
FROM sales_clean
GROUP BY location
ORDER BY total_revenue DESC;

-- 8. Top 5 customers by total spend
-- Cost note: LIMIT reduces returned rows but NOT bytes scanned in
-- BigQuery — the full column set used in GROUP BY is still scanned
-- before the top 5 are selected.
SELECT customer_id, SUM(total_spent) AS total_spent_by_customer
FROM sales_clean
GROUP BY customer_id
ORDER BY total_spent_by_customer DESC
LIMIT 5;

--  Get-Content sql/week5_queries.sql | sqlite3 data/db/retail_sample.db
