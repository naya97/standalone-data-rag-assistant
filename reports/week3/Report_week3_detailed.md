# Data Cleaning & Validation Report — Week 3
### Dataset: Retail Store Sales (Dirty for Data Cleaning)
Source: [Kaggle — Retail Store Sales: Dirty for Data Cleaning](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning)

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Original rows | 12,575 |
| Original columns | 11 |
| Final (cleaned) rows | 11,971 |
| Final (cleaned) columns | 11 |
| Rows removed | 604 |

**Raw column list:** Transaction ID, Customer ID, Category, Item, Price Per Unit, Quantity, Total Spent, Payment Method, Location, Transaction Date, Discount Applied.

---

## 2. Initial Data Quality Audit

### 2.1 Missing Values (raw dataset)

| Column | Missing |
|---|---:|
| Item | 1,213 |
| Price Per Unit | 609 |
| Quantity | 604 |
| Total Spent | 604 |
| Discount Applied | 4,199 |
| All other columns | 0 |

### 2.2 Duplicates
- Exact duplicate rows: **0**
- Duplicate `Transaction ID` values: **0**

### 2.3 Categorical Consistency
- `Category`: 8 unique values, evenly distributed (1,528–1,591 rows each).
- `Payment Method`: 3 unique values (Cash, Digital Wallet, Credit Card).
- `Location`: 2 unique values (Online, In-store).
- `Item`: 200 unique values (excluding missing).
- Hidden whitespace check on `Category`, `Payment Method`, `Location`, `Item`: **0 rows affected** — text was already clean and consistently cased.

### 2.4 Value Ranges (raw dataset)
- Negative or zero `Quantity`: 0
- Negative or zero `Price Per Unit`: 0
- Negative `Total Spent`: 0
- `Price Per Unit` range: 5.0 – 41.0 (fixed price list)
- `Quantity` range: 1 – 10
- `Total Spent` range: 5.0 – 410.0

### 2.5 Dates
- `Transaction Date` parsed to `datetime` with `errors='coerce'`.
- Unparsable dates found: **0** — all 12,575 dates were valid.

### 2.6 Statistical Outliers (IQR method, raw dataset)
- `Total Spent`: **60 outliers** detected (values > upper IQR bound). All correspond to valid maximum combinations (`Price Per Unit = 41.0`, `Quantity = 10`, `Total Spent = 410.0`) — legitimate large purchases, not data errors.
- `Price Per Unit`: 0 outliers.

---

## 3. Discount Applied — Detailed Investigation

This column required deeper investigation before deciding on a cleaning rule, since a simple `NaN → False` fill was not supported by evidence.

### 3.1 Test 1 — Effect on Total Spent
For all rows with complete data, `Expected Total = Quantity × Price Per Unit` was compared against the recorded `Total Spent`:

| Discount Applied | Count | Min diff | Max diff | Mean diff | Median diff |
|---|---:|---:|---:|---:|---:|
| False | 3,778 | 0.0 | 0.0 | 0.0 | 0.0 |
| True | 3,801 | 0.0 | 0.0 | 0.0 | 0.0 |

**Result:** `Total Spent` always equals `Quantity × Price Per Unit` exactly, whether or not a discount was recorded. The discount has **zero measurable effect** on the amount charged.

### 3.2 Test 2 — Effect on Price Per Unit
Average `Price Per Unit` per `Item`, split by discount status, was compared directly (e.g. `Item_10_BEV`: 18.5 regardless of `True`/`False`). Across all sampled items, **prices were identical** regardless of discount status — confirming the discount does not affect the unit price either.

### 3.3 Test 3 — Attempted Pattern-Based Inference for Missing Values
An attempt was made to infer missing `Discount Applied` values from historical patterns per `(Item, Category, Price Per Unit)` combination, using a confidence threshold (`true_rate ≥ 0.80` → True, `≤ 0.20` → False, count ≥ 5).

Initial result: 3 rows inferred `True`, 13 rows inferred `False`, 4,183 remained `Unknown`.

**Follow-up validation:** the distribution of `true_rate` across all 197 items with sufficient sample size was examined:

| Statistic | Value |
|---|---:|
| Mean | 0.505 |
| Std | 0.103 |
| Min | 0.143 |
| Median | 0.500 |
| Max | 1.000 |

The distribution is tightly centered around **0.50** (a near-perfect coin-flip), with no bimodal structure. This confirms that discount assignment is **randomly distributed per item**, and the 16 "confident" cases from the initial threshold test were statistical noise from small sample sizes — not a real pattern. **This inference approach was discarded.**

### 3.4 Final Decision — Discount Applied
- Converted to pandas nullable `boolean` dtype.
- Missing values are **not imputed** and are preserved as `<NA>` (unknown).
- No discount percentage was calculated, since there is no original pre-discount price available in the dataset, and the column has no measurable effect on any financial field.

---

## 4. Cleaning Rules Applied

### 4.1 Column Naming
All column names converted to `snake_case` (e.g. `Transaction ID` → `transaction_id`).

### 4.2 Data Type Rules

| Column | Target type |
|---|---|
| transaction_id, customer_id, category, item, payment_method, location | string |
| price_per_unit, total_spent | float64 |
| quantity | Int64 (nullable integer) |
| transaction_date | datetime64 |
| discount_applied | boolean (nullable) |

### 4.3 Missing Value Rules
- **item**: inferred from the verified one-to-one mapping `(category, price_per_unit) → item` (confirmed unambiguous across the entire dataset — 0 conflicting mappings found).
- **price_per_unit**: inferred as `total_spent / quantity`.
- **quantity**: inferred as `total_spent / price_per_unit`, accepted only if the result is a whole number.
- **total_spent**: inferred as `quantity × price_per_unit`.
- **discount_applied**: not imputed — kept as unknown (`<NA>`), per the investigation in Section 3.
- **Irrecoverable rows rule**: rows missing 2 or more of `price_per_unit`, `quantity`, `total_spent` simultaneously were removed, since no combination of the remaining fields provides a mathematical basis for recovery.

### 4.4 Outlier Rule
Outliers detected via IQR were **retained**, since they fall within valid business ranges (max price × max quantity) and do not violate any schema or business rule.

---

## 5. Missing Value Resolution — Final Results

| Column | Missing (Before) | Recovered via Logic | Removed with Row (of the 604) | Remaining (After) |
|---|---:|---:|---:|---:|
| item | 1,213 | 609 (Category + Price mapping) | 604 | 0 |
| price_per_unit | 609 | 609 (Total Spent / Quantity) | 0 | 0 |
| quantity | 604 | 0 | 604 | 0 |
| total_spent | 604 | 0 | 604 | 0 |
| discount_applied | 4,199 | 0 | 211 | 3,988 (intentionally kept as unknown) |

**Key finding:** Only `price_per_unit` was fully recovered through calculation. `quantity` and `total_spent` were missing in the *same* 604 rows simultaneously, leaving no mathematical basis for recovery — these rows were removed entirely under the irrecoverable-rows rule, not repaired. `item` was partially recovered (609 of 1,213); the remaining 604 belonged to the same removed rows. `discount_applied` was never imputed; 211 of its missing values disappeared only because their rows were removed for unrelated reasons (missing financial data), while the remaining 3,988 were deliberately preserved as unknown.

**Verification:** `604 (rows removed)` = rows missing `item`, `quantity`, and `total_spent` simultaneously (confirmed: all three columns show exactly 604 missing values within the removed-rows subset).

**Total rows removed: 604**

---

## 6. Validation Results (Post-Cleaning)

All checks were run via a dedicated `DataValidator` class against the cleaned dataset (11,971 rows):

| Check | Result |
|---|---|
| Column names snake_case | PASS |
| Data types match schema | PASS |
| Missing values (excluding discount_applied) | PASS (0 unexpected missing) |
| Irrecoverable rows remaining | PASS (0) |
| Exact duplicate rows | PASS (0) |
| Transaction ID uniqueness | PASS |
| Total Spent consistency (`= quantity × price`) | PASS |
| Value ranges (no negative/zero prices, quantities, totals; quantity is integer) | PASS |
| Categorical values (payment_method, location) | PASS (0 invalid) |
| Date parsing (no unparsable dates) | PASS |
| Discount Applied nullable boolean type | PASS |
| Category + Price → Item mapping (no ambiguous mappings) | PASS |

### Outlier Report (informational, not a pass/fail check)
- `price_per_unit`: 0 statistical outliers (IQR).
- `total_spent`: 60 statistical outliers (IQR) — retained, as they represent valid maximum-value transactions.

---

## 7. Final Cleaned Dataset

| Property | Value |
|---|---|
| Rows | 11,971 |
| Columns | 11 |
| Missing values | 0 in all columns except `discount_applied` (3,988 preserved as unknown) |
| Duplicate rows | 0 |
| Duplicate Transaction IDs | 0 |

**Final schema:**

| Column | Dtype |
|---|---|
| transaction_id | string |
| customer_id | string |
| category | string |
| item | string |
| price_per_unit | float64 |
| quantity | Int64 |
| total_spent | float64 |
| payment_method | string |
| location | string |
| transaction_date | datetime64 |
| discount_applied | boolean (nullable) |

**Output file:** `data/processed/week3/retail_store_sales_cleaned.csv`

---

## 8. Conclusion

The raw dataset contained missing values, one unreliable/non-functional column, and no duplicates or type inconsistencies beyond what standard conversion resolved. Financial fields (`price_per_unit`, `quantity`, `total_spent`, `item`) were recovered wherever the remaining fields in the same row provided a valid mathematical or logical basis; rows without such a basis (604 rows, missing 2+ financial fields at once) were removed rather than guessed at.

The `discount_applied` column was investigated in depth and found to have **no measurable effect** on price or total spent anywhere in the dataset, and **no reliable pattern** by item that would justify inferring its missing values. It was therefore converted to a nullable boolean and its missing values were deliberately preserved as unknown, rather than imputed — a decision grounded in evidence rather than assumption.

The resulting dataset (11,971 rows) passed all validation checks: correct schema, no unexpected missing values, no duplicates, internally consistent financial calculations, valid value ranges, valid categorical values, and a confirmed one-to-one `category + price → item` mapping.
