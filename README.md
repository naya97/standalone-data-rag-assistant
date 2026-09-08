# Standalone Data-RAG Assistant Engine

A simple standalone project built to practice core Python and data engineering basics: reading CSV/JSON files, storing data in SQLite, running basic SQL queries, exposing a simple REST API, and cleaning messy real-world datasets with Pandas.

The project is organized into two parts:
- **Core application** (Task 1): loads sample question/category data from CSV and JSON files, stores it in a local SQLite database, runs basic SQL queries, and provides a `/health` endpoint.
- **Weekly practice/exercises**: standalone or building-block exercises (Pandas, SQL, data cleaning) kept separate from the core application code.

---

## Project Structure

```
standalone-data-rag-assistant/
├── data/
│   ├── db/
│   │   └── local_storage.sqlite       # SQLite database (Task 1)
│   ├── raw/
│   │   ├── config.json                # App config (Task 1)
│   │   ├── sample.csv                 # Sample data (Task 1)
│   │   ├── sample.json                # Sample data (Task 1)
│   │   └── week3/
│   │       └── retail_store_sales.csv # Raw dataset (Week 3)
│   └── processed/
│       └── week3/
│           └── retail_store_sales_cleaned.csv   # Cleaned dataset (Week 3)
│
├── src/
│   ├── api/
│   │   └── health_controller.py       # FastAPI app + /health endpoint (Task 1)
│   ├── infrastructure/
│   │   ├── db_manager.py              # SQLite manager (Task 1)
│   │   └── file_reader.py             # CSV/JSON reader (Task 1)
│   └── week3/
│       ├── cleaning.py                # DataCleaner class (reusable cleaning pipeline)
│       └── validation.py              # DataValidator class (data-quality checks)
│
├── notebooks/
│   └── week3/
│       └── week3-retailstore.ipynb    # Data cleaning & validation notebook
│
├── reports/
│   └── week3/
│       ├── Report_week3.md            # Short data-quality report
│       └── Report_week3_detailed.md   # Full investigation details
│
├── practice/
│   └── week2/                         # Standalone Pandas/SQL practice (Superstore dataset)
│       ├── data/
│       │   ├── Superstore.csv
│       │   └── superstore.db          # generated locally, not committed (see .gitignore)
│       ├── notebooks/
│       │   └── superstore-week2.ipynb
│       └── SQL/
│           └── superstore_queries.sql
│
├── .env                                # Environment variables
├── .gitignore
├── main.py                             # Entry point (Task 1: ETL + SQL queries)
├── requirements.txt
└── README.md
```

> **Note:** `practice/` holds standalone weekly exercises that are self-contained and unrelated to the core application (e.g. Week 2). Weeks that build on each other and feed into the core project (e.g. Week 3, and Week 4 which will reuse Week 3's cleaning code) live directly under `data/`, `src/`, `notebooks/`, and `reports/`, each in their own `weekN` subfolder.

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
Loads the CSV/JSON sample data into SQLite and runs sample SQL queries (SELECT, WHERE, GROUP BY).

```bash
python main.py
```

### Run the API

```bash
uvicorn src.api.health_controller:app --reload
```

Then visit:
```
http://127.0.0.1:8000/health
```
Docs: `http://127.0.0.1:8000/docs`

### Database
Table `rag_topics`:

| Column     | Type    | Description       |
|------------|---------|--------------------|
| `id`       | INTEGER | Primary key        |
| `question` | TEXT    | Question text      |
| `category` | TEXT    | Question category  |

---

## Week 2 — Pandas + SQL Practice (Superstore Dataset)

Located entirely under `practice/week2/` — a standalone exercise, unrelated to the core application.

Explores the [Superstore dataset](https://www.kaggle.com/datasets/binib1997/superstore) using Pandas and SQL (SQLite), answering the same set of analytical questions with both approaches for comparison.

### Dataset
Download from Kaggle and place the CSV at `practice/week2/data/Superstore.csv`.

### Notebook
`practice/week2/notebooks/superstore-week2.ipynb`

### SQL Practice File
`practice/week2/SQL/superstore_queries.sql` contains the standalone SQL queries used in the notebook. To run them independently:

```bash
sqlite3 practice/week2/data/superstore.db
sqlite> .read practice/week2/SQL/superstore_queries.sql
```

---

## Week 3 — Data Cleaning & Validation (Retail Store Sales Dataset)

Explores and cleans the [Retail Store Sales: Dirty for Data Cleaning](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning) dataset from Kaggle.

### Dataset
Download from Kaggle and place the raw CSV at `data/raw/week3/retail_store_sales.csv`.

### Notebook
`notebooks/week3/week3-retailstore.ipynb` — full exploration, cleaning, and validation walkthrough.

### Reusable Cleaning & Validation Code
- `src/week3/cleaning.py` — `DataCleaner` class: column name normalization, dtype conversion, text normalization, missing-value imputation (math/logic-based only), irrecoverable-row removal.
- `src/week3/validation.py` — `DataValidator` class: schema, missing values, duplicates, financial consistency, value ranges, categorical values, date parsing, and outlier reporting checks.

```python
from src.week3.cleaning import DataCleaner
from src.week3.validation import DataValidator

cleaner = DataCleaner(df)
df_clean = cleaner.clean_all()

validator = DataValidator(df_clean)
validator.run_all_validations()
```

### Cleaned Output
`data/processed/week3/retail_store_sales_cleaned.csv`

### Reports
- `reports/week3/Report_week3.md` — short data-quality report (before/after checks + validation summary).
- `reports/week3/Report_week3_detailed.md` — full investigation details, including the `discount_applied` column analysis.

---

## Git Tags

Each completed week is marked with an annotated Git tag once finalized, so the project history can be browsed one week at a time:

```bash
git tag           # list all tags
git checkout tags/week2-complete   # view the project exactly as it was at the end of Week 2
```