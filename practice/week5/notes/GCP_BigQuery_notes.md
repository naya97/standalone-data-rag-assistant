# GCP & BigQuery Fundamentals — Notes & Local Equivalents

## 1. Introduction
Due to the lack of access to an active GCP Sandbox environment, the core concepts of Google Cloud Platform (GCP) and BigQuery were explored theoretically and applied practically using a local SQLite database. The retail dataset cleaned in Week 3 and loaded into SQLite via the Week 4 ETL pipeline was used to execute the Week 5 analytical queries, serving as a direct local equivalent to a BigQuery environment.

## 2. Concept Mapping: Cloud vs. Local Equivalents
The following table maps the core GCP/BigQuery concepts to their local equivalents used in this project:

| GCP / BigQuery Concept | Local Equivalent in this Project | Description |
| :--- | :--- | :--- |
| **Cloud Storage (GCS)** | Local File System (`data/raw/week4/retail_store_sales.csv`) | Raw data storage before ingestion. |
| **BigQuery (Data Warehouse)** | SQLite (`data/db/retail_pipeline.db`) | The analytical database engine. |
| **Dataset** | SQLite Database / Schema context | A container for tables and views. |
| **Table** | `sales_clean` table | The structured, cleaned data ready for querying. |
| **Public Datasets** | Week 3 cleaned retail dataset | A ready-made dataset to query, standing in for Google's free public datasets |

## 3. IAM & Service Accounts (Theoretical Concepts)
Since local environments do not have a permission management layer, **Identity and Access Management (IAM)** and **Service Accounts** were studied theoretically:

*   **IAM (Identity and Access Management):** In GCP, IAM defines *who* (users, groups) has *what* access (viewer, editor, owner) to *which* resources. It answers the question: "Who can view, edit, or run queries on this dataset?"
*   **Service Accounts:** These are special Google accounts designed for applications and virtual machines (VMs) rather than human users. For example, our `run_pipeline.py` ETL script would run under a Service Account in the cloud, requiring specific permissions to read from Cloud Storage and write to BigQuery.
*   **Why there is no local equivalent:** In a local SQLite setup, there is no concept of user roles or granular permissions. The local OS user who runs the Python script or opens the database file inherently has full administrative access (read, write, delete) to the entire database. Therefore, permission management is moot locally.

## 4. Query Cost Awareness
BigQuery operates on an **On-Demand Pricing** model. Unlike traditional databases where you pay for compute time (how long the query takes), BigQuery charges based on the **number of bytes scanned** (columns read) during the query execution. This makes column selection critical.

**Why `SELECT *` is expensive:** 
Selecting all columns forces BigQuery to scan the entire table, which dramatically increases the bytes scanned and the cost. 

**How this was applied in the Week 5 queries (`week5_queries.sql`):**
The 8 analytical queries were written with this cost-awareness principle in mind:
1.  **Column Pruning:** Instead of `SELECT *`, queries explicitly select only the required columns (e.g., `SELECT category, SUM(total_spent)`). This avoids scanning unrelated columns like `customer_id` or `transaction_date` when they aren't needed.
2.  **Avoiding Redundant Computations:** Query 4 (`AVG(total_spent)`) uses the pre-calculated `total_spent` column rather than recomputing it from `quantity * price_per_unit`. This saves the cost of scanning two separate columns.
3.  **Understanding `LIMIT`:** Query 8 uses `LIMIT 5`. It is a common misconception that `LIMIT` reduces cost. In BigQuery, `LIMIT` reduces the number of rows *returned*, but the database still has to scan the full column to determine the top 5. The `GROUP BY customer_id` and `SUM(total_spent)` still scan the entire respective columns.
4.  **Partitioning (Theoretical):** Query 3 (`Daily transaction count trend`) notes that in a real BigQuery environment, this table should be *partitioned* by `transaction_date`. Without partitioning, querying a specific date range requires a full-table scan, incurring unnecessary costs.

## 5. Public Datasets Note
BigQuery hosts a variety of **Public Datasets** (e.g., `bigquery-public-data`) that are free to query (though you still pay for the bytes scanned). If a sandbox account had been available, the exercise would have involved querying one of these public datasets and uploading a small CSV. 

Since the sandbox was unavailable, the **Week 3 local Retail Store Sales dataset** was used as the equivalent. It provided a realistic, messy, real-world dataset that required cleaning (Week 3/4) before being queried (Week 5), perfectly simulating the end-to-end workflow of a cloud data pipeline.