# Data Quality Report — Retail Store Sales (Week 3)

**Dataset:** [Retail Store Sales: Dirty for Data Cleaning](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning) (Kaggle)

## Before / After Summary

| Metric | Before | After |
|---|---:|---:|
| Rows | 12,575 | 11,971 |
| Duplicate rows | 0 | 0 |
| Duplicate Transaction IDs | 0 | 0 |
| Unparsable dates | 0 | 0 |
| `transaction_date` dtype | object (string) | datetime64 |
| `discount_applied` dtype | object (True/False/NaN) | nullable boolean |

**Missing values by column:**

| Column | Before | After | How resolved |
|---|---:|---:|---|
| item | 1,213 | 0 | 609 recovered via `category + price → item` mapping; 604 removed with irrecoverable rows |
| price_per_unit | 609 | 0 | Recovered via `total_spent / quantity` |
| quantity | 604 | 0 | Rows removed (also missing total_spent — no basis to recover) |
| total_spent | 604 | 0 | Rows removed (also missing quantity — no basis to recover) |
| discount_applied | 4,199 | 3,988 | Not imputed — kept as unknown (see note below) |

**Rows removed:** 604, all missing 2+ of `price_per_unit`, `quantity`, `total_spent` simultaneously — no reliable basis to recover them.

## Key Decision: Discount Applied
Testing showed `discount_applied` has **no measurable effect** on `total_spent` or `price_per_unit` anywhere in the dataset, and no reliable per-item pattern (missing-value inference tested and discarded — true rates were randomly distributed around 50% across items). It was converted to a nullable boolean, and missing values were **left as unknown rather than guessed**.

## Validation Results (all PASS)
Column naming (snake_case) · Data types · No unexpected missing values · No irrecoverable rows remaining · No duplicates · Transaction ID uniqueness · `total_spent = quantity × price_per_unit` consistency · Valid value ranges · Valid categorical values · Dates parsed · `category + price → item` mapping unambiguous.

**Outliers:** 60 statistical outliers (IQR) in `total_spent`, retained — they reflect valid maximum-value transactions (max price × max quantity), not errors. 0 outliers in `price_per_unit`.

## Output
Cleaned dataset saved to: `data/processed/week3/retail_store_sales_cleaned.csv` (11,971 rows × 11 columns)

*For the full investigation details (including the discount analysis methodology), see `Report_week3_detailed.md`.*
