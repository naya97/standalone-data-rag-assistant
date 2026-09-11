# Standalone Data-RAG Assistant Engine

A simple standalone project built to practice core Python and data engineering basics: reading CSV/JSON files, storing data in SQLite, running basic SQL queries, exposing a simple REST API, cleaning messy real-world datasets with Pandas, and building a repeatable ETL pipeline.

The project is organized into two parts:
- **Core application** (Task 1): loads sample question/category data from CSV and JSON files, stores it in a local SQLite database, runs basic SQL queries, and provides a `/health` endpoint.
- **Weekly practice/exercises**: standalone or building-block exercises (Pandas, SQL, data cleaning, ETL pipelines) kept separate from the core application code, but reusing its shared infrastructure where relevant.

---

## Project Structure

```
standalone-data-rag-assistant/
├── data/
│   ├── db/
│   │   ├── local_storage.sqlite        # SQLite database (Task 1)
│   │   └── retail_pipeline.db          # SQLite database (Week 4 pipeline output)
│   ├── raw/
│   │   ├── config.json                 # App config (Task 1)
│   │   ├── sample.csv                  # Sample data (Task 1)
│   │   ├── sample.json                 # Sample data (Task 1)
│   │   ├── week3/
│   │   │   └── retail_store_sales.csv  # Raw dataset (Week 3)
│   │   └── week4/
│   │       └── retail_store_sales.csv  # Pipeline input folder (Week 4, batch-scanned)
│   └── processed/
│       └── week3/
│           └── retail_store_sales_cleaned.csv   # Cleaned dataset (Week 3)
│
├── src/
│   ├── api/
│   │   └── health_controller.py        # FastAPI app + /health endpoint (Task 1)
│   ├── infrastructure/
│   │   ├── db_manager.py               # SQLiteManager (Task 1, extended in Week 4)
│   │   └── file_reader.py              # CSV/JSON reader (Task 1)
│   ├── week3/
│   │   ├── cleaning.py                 # DataCleaner class (reusable cleaning pipeline)
│   │   └── validation.py               # DataValidator class (data-quality checks)
│   └── pipeline/
│       └── week4/
│           ├── extract.py              # Batch CSV extraction
│           ├── transform.py            # Wraps Week 3's DataCleaner + DataValidator
│           └── load.py                 # Writes cleaned data to SQLite
│
├── notebooks/
│   └── week3/
│       └── week3-retailstore.ipynb     # Data cleaning & validation notebook
│
├── reports/
│   └── week3/
│       ├── Report_week3.md             # Short data-quality report
│       └── Report_week3_detailed.md    # Full investigation details
│
├── logs/
│   └── week4/
│       └── pipeline.log                # ETL pipeline run log (Week 4)
│
├── practice/
│   └── week2/                          # Standalone Pandas/SQL practice (Superstore dataset)
│       ├── data/
│       │   ├── Superstore.csv
│       │   └── superstore.db           # generated locally, not committed (see .gitignore)
│       ├── notebooks/
│       │   └── superstore-week2.ipynb
│       └── SQL/
│           └── superstore_queries.sql
│
├── .env                                 # Environment variables
├── .gitignore
├── main.py                              # Entry point (Task 1: ETL + SQL queries)
├── run_pipeline.py                      # Entry point (Week 4: ETL pipeline)
├── requirements.txt
└── README.md
```

> **Note:** `practice/` holds standalone weekly exercises that are self-contained and unrelated to the core application (e.g. Week 2). Weeks that build on each other and feed into the core project (Week 3, and Week 4 which reuses Week 3's cleaning code) live directly under `data/`, `src/`, `notebooks/`, `reports/`, and `logs/`, each in their own `weekN` subfolder.

---

## Requirements

- Python 3.10+
- pip

Main packages (`requirements.txt`):

```
fastapi==0.141.1
uvicorn==0.52.4
python-dotenv==1.2.3
pandas==3.0.5
numpy==2.5.2
ipykernel
```

---

## Setup

```bash
git clone <repo-url>
cd standalone-data-rag-assistant

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

## Environment Variables (.env)

```dotenv
APP_NAME="Standalone Data-RAG Assistant"
APP_ENV="development"
DEBUG_MODE=True
DB_PATH="data/db/local_storage.sqlite"
```

---

## Task 1 — Core Application

### Run the ETL script
```bash
python main.py
```

### Run the API
```bash
uvicorn src.api.health_controller:app --reload
```
Visit `http://127.0.0.1:8000/health` — docs at `/docs`.

### Database
Table `rag_topics`:

| Column     | Type    | Description       |
|------------|---------|--------------------|
| `id`       | INTEGER | Primary key        |
| `question` | TEXT    | Question text      |
| `category` | TEXT    | Question category  |

---

## Week 2 — Pandas + SQL Practice (Superstore Dataset)

Located entirely under `practice/week2/` — a standalone exercise, unrelated to the core application. Explores the [Superstore dataset](https://www.kaggle.com/datasets/binib1997/superstore) using Pandas and SQL (SQLite), answering the same analytical questions with both approaches for comparison.

```bash
sqlite3 practice/week2/data/superstore.db
sqlite> .read practice/week2/SQL/superstore_queries.sql
```

---

## Week 3 — Data Cleaning & Validation (Retail Store Sales Dataset)

Explores and cleans the [Retail Store Sales: Dirty for Data Cleaning](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning) dataset.

- **Notebook:** `notebooks/week3/week3-retailstore.ipynb`
- **Reusable code:** `src/week3/cleaning.py` (`DataCleaner`), `src/week3/validation.py` (`DataValidator`)
- **Cleaned output:** `data/processed/week3/retail_store_sales_cleaned.csv`
- **Reports:** `reports/week3/Report_week3.md` (short) and `Report_week3_detailed.md` (full investigation, including the `discount_applied` analysis)

```python
from src.week3.cleaning import DataCleaner
from src.week3.validation import DataValidator

cleaner = DataCleaner(df)
df_clean = cleaner.clean_all()

validator = DataValidator(df_clean)
validator.run_all_validations()
```

---

## Week 4 — ETL Pipeline

A small, repeatable ETL pipeline that reads raw files, cleans and validates them using Week 3's logic, and loads the result into SQLite — replacing one-off notebook execution with a single runnable script.

### Architecture

```
Extract (extract.py) → Transform (transform.py) → Load (load.py)
```

- **`src/pipeline/week4/extract.py`** — scans `data/raw/week4/` for all `.csv` files and combines them into one DataFrame (batch processing: works whether the folder holds one file or many).
- **`src/pipeline/week4/transform.py`** — reuses `DataCleaner` and `DataValidator` from Week 3 directly (no cleaning logic is duplicated); raises an error if any validation check fails.
- **`src/pipeline/week4/load.py`** — writes the cleaned DataFrame to `data/db/retail_pipeline.db`, table `sales_clean`, using the shared `SQLiteManager`.

### Run the pipeline
```bash
python run_pipeline.py
```

### Logs
Every run appends to `logs/week4/pipeline.log`, recording each step (rows read, rows after cleaning, rows loaded) and, on failure, exactly which phase (Extract / Transform / Load) failed and why — instead of a generic error.

### Shared Infrastructure Change
`src/infrastructure/db_manager.py`'s `SQLiteManager` was extended for reuse across multiple databases:
- Added `insert_dataframe(table_name, df)` for bulk DataFrame writes.
- `_initialize_tables()` (which creates the Task 1 `rag_topics` table) is no longer called automatically — it now requires `initialize_default_schema=True`, so pipelines writing to other databases (like `retail_pipeline.db`) don't get an unrelated table created in them. Task 1's `main.py` and `health_controller.py` were updated to pass `initialize_default_schema=True` explicitly.

### Failure Handling — Verified Scenarios
The pipeline was tested against three failure cases to confirm each produces a clear, phase-specific log message rather than a generic error:
1. Missing input file → `Extract` fails with a clear "no CSV files found" message.
2. Missing required column → `Transform` fails with the exact missing column name.
3. Invalid database path → `Load` fails with the underlying SQLite error, clearly attributed to the Load phase.

---

## Git Tags

Each completed week is marked with an annotated Git tag once finalized, so the project history can be browsed one week at a time:

```bash
git tag
git checkout tags/week2-complete
git checkout tags/week3-complete
git checkout tags/week4-complete
```