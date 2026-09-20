# Week 6 — Analytics Summary

Analytical questions answered using the views and queries built in `sql/week6/views.sql` and `sql/week6/queries.sql`, run against `data/db/retail_pipeline.db` (Week 4 pipeline output, 11,971 cleaned transactions). Results below are taken directly from the Streamlit dashboard (`src/analytics/week6/app.py`).

---

## 1. Overall KPIs

| KPI | Value |
|---|---:|
| Total Revenue | $1,552,071.00 |
| Average Order Value | $129.65 |
| Total Transactions | 11,971 |

---

## 2. Monthly Revenue Trend

Revenue by month (`monthly_sales_summary` view) is fairly stable, fluctuating mostly between ~$38,000 and ~$48,000 per month from 2022 through late 2024, with no strong long-term upward or downward trend.

Sample of recent months:

| Month | Transactions | Total Revenue | Avg Order Value |
|---|---:|---:|---:|
| 2024-01 | 363 | 47,908.5 | 131.98 |
| 2024-02 | 304 | 37,145.0 | 122.19 |
| 2024-03 | 336 | 42,861.5 | 127.56 |
| 2024-04 | 311 | 46,271.0 | 148.78 |
| 2024-05 | 345 | 43,766.5 | 126.86 |
| 2024-06 | 334 | 44,721.0 | 133.90 |
| 2024-07 | 334 | 41,405.0 | 123.97 |
| 2024-08 | 354 | 43,362.0 | 122.49 |
| 2024-09 | 328 | 42,161.5 | 128.54 |
| 2024-10 | 338 | 42,736.5 | 126.44 |

**Note:** the final data point (2025-01, $25,548.50) is noticeably lower than every other month on the chart. This is almost certainly a **partial month** — the dataset's transaction dates simply end partway through January 2025, not a genuine drop in sales. This should be excluded (or explicitly flagged) in any "best/worst month" analysis to avoid a misleading conclusion.

**Q: Is there a clear month-over-month trend?**
**A:** No strong trend — revenue oscillates within a fairly narrow band (~$38K–$48K) for the entire observed period, with no sustained growth or decline. The apparent dip at the very end (2025-01) is a data-completeness artifact, not a real trend.

---

## 3. Category Performance vs. Target (JOIN + Calculated Field)

Using the `category_performance` view (`category_summary` JOIN `category_targets`):

| Category | Total Revenue | Revenue Target | Target Achievement % |
|---|---:|---:|---:|
| Butchers | 208,118.0 | 32,000 | 650.37% |
| Electric household essentials | 203,813.5 | 33,000 | 617.62% |
| Beverages | 197,047.5 | 32,000 | 615.77% |
| Food | 194,812.0 | 32,000 | 608.79% |
| Patisserie | 182,165.5 | 30,000 | 607.22% |
| Furniture | 195,310.0 | 33,000 | 591.85% |
| Computers and electric accessories | 190,692.5 | 33,000 | 577.86% |
| Milk Products | 180,112.0 | 32,000 | 562.85% |

**Q: Does any category generate disproportionately more revenue relative to expectations?**
**A:** Every category exceeded its target by 5.6x to 6.5x. Since `category_targets` is a hypothetical target table created specifically to demonstrate a JOIN (not a real business figure from the dataset), this result mainly shows the JOIN and calculated field working correctly rather than a genuine business insight — the targets chosen were simply set far below actual revenue. **Butchers** is the strongest performer relative to its target (650%), while **Milk Products** is the weakest (563%) — though all eight categories are within a fairly tight range of each other (562–650%), showing no single category dramatically outperforms the rest.

---

## 4. Underperforming Categories (Raw JOIN Query)

```
No categories are currently below their revenue target.
```

Consistent with the table above — since every category cleared its (hypothetical) target comfortably, the underperformance query correctly returns an empty result rather than a false positive.

---

## 5. Revenue Share by Payment Method

| Payment Method | Total Revenue | Revenue Share % |
|---|---:|---:|
| Cash | 537,710 | 34.64% |
| Digital Wallet | 507,279 | 32.68% |
| Credit Card | 507,082 | 32.67% |

**Q: Is any payment method associated with disproportionately higher-value transactions?**
**A:** No — revenue is split almost evenly across all three methods (~33% each), with Cash only marginally ahead. This mirrors the same kind of near-uniform distribution observed with `discount_applied` in Week 3 (values spread randomly with no dominant category), suggesting payment method is not a meaningful predictor of transaction value in this dataset.

---

## 6. Summary

- Revenue is stable month-to-month with no clear trend, aside from an expected partial-month artifact at the very end of the dataset.
- The category/target JOIN and calculated fields work correctly (verified by both the view and the standalone JOIN query), though the specific target values used were arbitrary and mainly serve to demonstrate the join mechanics.
- No payment method stands out — revenue is almost evenly split three ways, consistent with the dataset's broader tendency (also seen with `discount_applied` in Week 3) toward randomly distributed categorical fields rather than meaningful business patterns.