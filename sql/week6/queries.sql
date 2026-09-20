-- sql/week6/queries.sql

-- KPI 1: Overall Average Order Value
SELECT AVG(total_spent) AS overall_avg_order_value FROM sales_clean;

-- KPI 2: Best-performing month by revenue
SELECT sale_month, total_revenue
FROM monthly_sales_summary
ORDER BY total_revenue DESC
LIMIT 1;

-- KPI 3: Month-over-month revenue growth
SELECT
    sale_month,
    total_revenue,
    total_revenue - LAG(total_revenue) OVER (ORDER BY sale_month) AS revenue_change
FROM monthly_sales_summary;

-- KPI 4: Top category by revenue (reusing category_summary view)
SELECT category, total_revenue
FROM category_summary
WHERE total_revenue > 5000
ORDER BY total_revenue DESC
LIMIT 1;

-- KPI 5: Revenue share by payment method (% of total)
SELECT
    payment_method,
    total_revenue,
    ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM payment_method_summary), 2) AS revenue_share_pct
FROM payment_method_summary;

-- ============================================================
-- JOIN queries (using the new category_targets dimension table)
-- ============================================================

-- KPI 6: Category performance vs revenue target (via the pre-built view)
SELECT category, total_revenue, revenue_target, target_achievement_pct
FROM category_performance
ORDER BY target_achievement_pct DESC;

-- KPI 7: Underperforming categories — raw JOIN + GROUP BY + calculated field
-- (written directly against sales_clean, not through a view, to
-- demonstrate an explicit join rather than only reusing views)
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
ORDER BY revenue_gap ASC;
