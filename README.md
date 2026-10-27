# Standalone Data-RAG Assistant Engine

A simple standalone project built to practice core Python and data engineering basics: reading CSV/JSON files, storing data in SQLite, running basic SQL queries, exposing a simple REST API, cleaning messy real-world datasets with Pandas, building a repeatable ETL pipeline, understanding core Google Cloud / BigQuery concepts, building a small analytics layer with an interactive dashboard, and preparing a clean, metadata-tagged document collection as the foundation for a Retrieval-Augmented Generation (RAG) assistant.

The project is organized into two parts:
- **Core application** (Task 1): loads sample question/category data from CSV and JSON files, stores it in a local SQLite database, runs basic SQL queries, and provides a `/health` endpoint.
- **Weekly practice/exercises**: standalone or building-block exercises (Pandas, SQL, data cleaning, ETL pipelines, GCP/BigQuery fundamentals, analytics/data modeling, RAG document preparation) kept separate from the core application code, but reusing its shared infrastructure where relevant.

---

## Project Structure

```
standalone-data-rag-assistant/
├── data/
│   ├── db/
│   │   ├── local_storage.sqlite        # SQLite database (Task 1 `rag_topics`; Week 7 `rag_documents`/`rag_chunks`)
│   │   ├── retail_pipeline.db          # SQLite database (Week 4 pipeline output, used by Week 6)
│   │   └── chroma/                     # (Week 8) ChromaDB vector index, derived from rag_chunks (git-ignored)
│   ├── eval/
│   │   ├── week8/
│   │   │   ├── eval_questions.json            # 49 questions (dev/test), expected sources + expected facts
│   │   │   ├── eval_questions_extended.json   # 35 questions (calib/holdout): cross-language + fresh ones
│   │   │   └── eval_results.json              # last evaluation run (every score, signal and answer)
│   │   └── week9/
│   │       ├── llm_eval_results.json        # last LLM evaluation (every outcome, token count, latency)
│   │       ├── gate_experiment.json         # lexical-gate settings compared on the saved questions
│   │       └── llm_cache.json               # cached model replies for the evaluation (git-ignored)
│   ├── raw/
│   │   ├── config.json                 # App config (Task 1)
│   │   ├── sample.csv                  # Sample data (Task 1)
│   │   ├── sample.json                 # Sample data (Task 1)
│   │   ├── week3/
│   │   │   └── retail_store_sales.csv  # Raw dataset (Week 3)
│   │   ├── week4/
│   │   │   └── retail_store_sales.csv  # Pipeline input folder (Week 4, batch-scanned)
│   │   └── week7/
│   │       ├── invoices/               # Sample + real invoice documents (.txt, .pdf)
│   │       ├── policies/               # Sample policy documents (.md)
│   │       ├── reports/                # Sample report documents (.txt)
│   │       └── misc/                   # Real external PDFs that don't match the naming
│   │                                   #   convention (job-task briefs, a JSON payload spec) -
│   │                                   #   ingested as 'unclassified', not renamed
│   └── processed/
│       └── week3/
│           └── retail_store_sales_cleaned.csv   # Cleaned dataset (Week 3)
│
├── src/
│   ├── api/
│   │   ├── health_controller.py        # FastAPI app + /health endpoint (Task 1)
│   │   ├── rag_controller.py           # (Week 8) POST /rag/ask, GET /rag/status
│   │   └── llm_controller.py            # (Week 9) POST /llm/sales-insights, POST /llm/summarize-documents
│   ├── llm/
│   │   └── week9/
│   │       ├── config.py                    # LLM settings from .env
│   │       ├── client.py                    # Gemini client, scripted fake, error types, one shared client
│   │       ├── retry.py                     # retries, time budget, request throttle
│   │       ├── schemas.py                   # structured outputs (RAG answer, sales insights, document summary)
│   │       ├── guardrails.py                # JSON/schema checks, citations, numbers must exist in the source
│   │       ├── prompts.py                   # prompts for summaries and insights
│   │       ├── runner.py                    # ask for a small JSON object: send, check, repair once
│   │       ├── data_facts.py                # sales figures computed by SQL (the model never computes)
│   │       ├── insights.py                  # sales summary and insights (+ no-LLM fallback)
│   │       └── tool_specs.py                # function-calling concept: two tools, validation, execution
│   ├── infrastructure/
│   │   ├── db_manager.py               # SQLiteManager (Task 1, extended in Week 4)
│   │   └── file_reader.py              # CSV/JSON reader (Task 1)
│   ├── week3/
│   │   ├── cleaning.py                 # DataCleaner class (reusable cleaning pipeline)
│   │   └── validation.py               # DataValidator class (data-quality checks)
│   ├── pipeline/
│   │   └── week4/
│   │       ├── extract.py              # Batch CSV extraction
│   │       ├── transform.py            # Wraps Week 3's DataCleaner + DataValidator
│   │       └── load.py                 # Writes cleaned data to SQLite
│   ├── analytics/
│   │   └── week6/
│   │       └── app.py                  # Streamlit analytics dashboard
│   └── rag/
│       ├── week7/
│       │   ├── document_loader.py      # Unified .txt/.md/.pdf reader (pypdf-based)
│       │   ├── table_extractor.py      # pdfplumber, used ONLY to detect/clean PDF tables
│       │   ├── metadata_extractor.py   # Best-effort filename classification (never blocking)
│       │   ├── text_cleaner.py         # Noise removal, whitespace/paragraph reconstruction
│       │   ├── chunking.py             # DocumentChunker - unified, safety-capped chunker
│       │   ├── db_store.py             # RagStore - rag_documents / rag_chunks persistence
│       │   └── ingest.py               # Orchestrator: Scan→Load→Metadata→Clean→Chunk→Store
│       └── week8/
│           ├── config.py               # settings from .env
│           ├── embedder.py             # sentence-transformers wrapper (model loaded once)
│           ├── vector_store.py         # ChromaDB wrapper (metadata cleaning, filters, model guard)
│           ├── indexer.py              # rag_chunks -> embeddings -> Chroma (idempotent sync)
│           ├── retriever.py            # top-k + metadata filters + stale-index detection
│           ├── answerability.py        # can the best chunk answer the question? (rule calibrated in the notebook)
│           ├── prompt_builder.py       # grounded prompt + fixed messages (EN/AR)
│           ├── answer_generator.py     # extractive answer (interface for the Week 9 LLM)
│           ├── rag_service.py          # ask(): ties everything together
│           └── evaluator.py            # system-level evaluation
│       └── week9/
│           ├── llm_generator.py             # LLM answer for /rag/ask with the extractive fallback
│           ├── doc_summary.py               # document summary (+ no-LLM fallback)
│           ├── evaluator.py                 # LLM vs extractive, several gate settings, cached replies
│           └── gate_experiment.py           # which lexical threshold? (gate only, no model)
│
├── sql/
│   └── week6/
│       ├── views.sql                   # Analytics layer: dimension table + reusable views
│       └── queries.sql                 # KPI queries + JOIN-based analysis
│
├── tests/
│   └── week9/                               # scripted fake model: no network, no key
│
├── notebooks/
│   ├── week3/
│   │   └── week3-retailstore.ipynb     # Data cleaning & validation notebook
│   ├── week7/
│   │   └── week7-rag-preparation.ipynb # Sandbox: parsing/metadata/chunking experiments
│   └── week8/
│       └── week8-retrieval-experiments.ipynb
│
├── reports/
│   ├── week3/
│   │   ├── Report_week3.md             # Short data-quality report
│   │   └── Report_week3_detailed.md    # Full investigation details
│   ├── week6/
│   │   └── Week6_Analytics_Summary.md  # Analytical questions answered with query evidence
│   ├── week7/
│   │   ├── Report_week7.md                      # Summary, results, known limitations
│   │   └── parsing_and_chunking_walkthrough.txt  # Full step-by-step design-decision log
│   ├── week8/
│   │   ├── Report_week8.md
│   │   ├── retrieval_evaluation.md         # evaluation notes (the 10 predefined questions + the 84-question run)
│   │   └── rag_design_walkthrough.txt
│   └── week9/
│       ├── Report_week9.md
│       ├── llm_evaluation.md
│       └── llm_design_walkthrough.txt
│
├── logs/
│   ├── week4/
│   │   └── pipeline.log                # ETL pipeline run log (Week 4)
│   ├── week7/
│   │   └── ingestion.log               # RAG ingestion run log (Week 7)
│   ├── week8/
│   │   └── rag.log
│   └── week9/
│       └── llm.log                          # every model request: attempt, latency, tokens, errors (never the key)
│
├── practice/
│   ├── week2/                          # Standalone Pandas/SQL practice (Superstore dataset)
│   │   ├── data/
│   │   │   ├── Superstore.csv
│   │   │   └── superstore.db           # generated locally, not committed (see .gitignore)
│   │   ├── notebooks/
│   │   │   └── superstore-week2.ipynb
│   │   └── SQL/
│   │       └── superstore_queries.sql
│   └── week5/                          # GCP + BigQuery fundamentals (local equivalents)
│       ├── data/
│       │   └── retail_sample.db        # local stand-in for a BigQuery dataset/table
│       ├── sql/
│       │   └── week5_queries.sql       # 8 analytical queries (BigQuery-style)
│       └── notes/
│           └── GCP_BigQuery_notes.md   # Concept notes + local-equivalent mapping
│   └── week9/                          # LLM concepts notes
│       └── notes/
│           └── Gemini_Vertex_notes.md  # Gemini API vs Vertex AI concepts (no cloud account: notes only)
│
├── .env                                 # Environment variables
├── .gitignore
├── main.py                              # Entry point (Task 1: ETL + SQL queries)
├── run_pipeline.py                      # Entry point (Week 4: ETL pipeline)
├── run_rag_ingestion.py                 # Entry point (Week 7: RAG document ingestion)
├── run_rag_indexing.py                  # build / sync the vector index (Week 8)
├── run_rag_query.py                     # dev tool: one question from the terminal (--ask prints the full response)
├── run_rag_eval.py                      # evaluation entry point
├── run_llm_insights.py                  # sales / docs summaries and insights from the command line
├── run_llm_tools.py                     # function-calling concept demo
├── run_llm_eval.py                      # LLM evaluation entry point
├── run_gate_experiment.py               # lexical-gate experiment entry point
├── .env.example                         # every setting, without secrets
├── requirements.txt
└── README.md
```

> **Note:** `practice/` holds standalone weekly exercises that are self-contained and unrelated to the core application (e.g. Week 2 and Week 5). Weeks that build on each other and feed into the core project (Week 3, Week 4, Week 6 — which queries Week 4's pipeline output — and Week 7, whose output is the document collection Week 8's RAG assistant will embed and retrieve from) live directly under `data/`, `src/`, `sql/`, `notebooks/`, `reports/`, and `logs/`, each in their own `weekN` subfolder. Within `src/`, only importable Python code lives (`analytics/`, `pipeline/`, `week3/`, `rag/week7/`); raw SQL files live under the top-level `sql/` folder instead, kept separate from Python source for the same reason `notebooks/` and `reports/` are separate. Week 8 is **not** a standalone exercise: it builds on Week 7's chunks and is the base of the LLM, tools and routing work of the following weeks, so it lives under `src/`, `data/`, `reports/`, `notebooks/` and `logs/` (`week8` subfolders), not under `practice/`.

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
streamlit
pypdf==5.9.0
pdfplumber
sentence-transformers==<version>
chromadb==<version>
google-genai==<version>      # pin what `pip show google-genai` prints
pytest                       # tests/week9
```

`pypdf` is the primary PDF text extractor (Week 7). `pdfplumber` is used narrowly, only to detect and cleanly extract tables on PDF pages that have one — it is not a replacement for `pypdf` (see Week 7 below for why). `fpdf2` was used once, as a dev-only tool, to generate a synthetic sample invoice PDF for testing — it is not a runtime dependency of the ingestion pipeline.

For `sentence-transformers` and `chromadb`, pin the versions printed by `pip show sentence-transformers chromadb`. The first run downloads the embedding model (~470 MB); afterwards it loads from the local cache.

---

## Environment Variables (.env)

```dotenv
APP_NAME="Standalone Data-RAG Assistant"
APP_ENV="development"
DEBUG_MODE=True
DB_PATH="data/db/local_storage.sqlite"

EMBEDDING_MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_TEXT_VARIANT="text_only"
CHROMA_PATH="data/db/chroma"
RAG_COLLECTION="rag_chunks_v1"
EMBED_BATCH_SIZE=32
RAG_TOP_K=3
RAG_MIN_SCORE=0.0
RAG_MIN_LEXICAL_COVERAGE=0.5
RAG_UNKNOWN_ENTITY_CHECK=True
RAG_MIN_SCORE_CROSS_LINGUAL=0.30
RAG_SUPPORT_RATIO=0.65

RETAIL_DB_PATH="data/db/retail_pipeline.db"
RAG_ANSWER_MODE="llm"
RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE=0.5
RAG_ARABIC_LIGHT=False

LLM_PROVIDER="gemini"
GEMINI_API_KEY=""                 # Google AI Studio key. NEVER commit .env
GEMINI_MODEL=""                   # exact model id from AI Studio
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=1024
LLM_TIMEOUT_SECONDS=45
LLM_DEADLINE_SECONDS=100
LLM_MAX_RETRIES=1
LLM_BACKOFF_BASE_SECONDS=1.0
LLM_MAX_RPM=10
LLM_THINKING_LEVEL=
```

`RAG_MIN_LEXICAL_COVERAGE` is now the gate **in front of the LLM** (0.33 after the Week 9 evaluation); verbatim answers keep
the Week 8 value through `RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE`.

`.gitignore`: add `data/db/chroma/`, `.env` and `data/eval/week9/llm_cache.json`.

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
The pipeline was tested against real failure cases (see `logs/week4/pipeline.log` history) to confirm each phase fails clearly rather than silently or with a generic error:

1. **Missing input file (Extract):**
   ```
   CRITICAL - ETL Pipeline Terminated Due To Error: No CSV files found in data/raw/week4
   ```
2. **Missing required column (Transform):**
   ```
   ERROR - Transform Phase FAILED: ['category']
   CRITICAL - ETL Pipeline Terminated Due To Error: ['category']
   ```
3. **Duplicate batch input (Transform, extra safeguard):** loading the same file twice was correctly rejected by validation instead of silently producing duplicate rows:
   ```
   ERROR - Transform Phase FAILED: Validation checks failed after cleaning.
   ```
4. **Invalid database path (Load):**
   ```
   ERROR - Database Load Error: unable to open database file
   ERROR - Load Phase FAILED: unable to open database file
   CRITICAL - ETL Pipeline Terminated Due To Error: unable to open database file
   ```

### Bug Found & Fixed During Testing
An early version of `SQLiteManager.insert_dataframe()` logged database errors but did not re-raise them, so a failed Load step could be swallowed silently and the pipeline would report "Finished Successfully" despite writing no data. This was caught by comparing consecutive test runs in `pipeline.log` and fixed by adding `raise` after the error log, ensuring Load failures always propagate and terminate the pipeline with a clear message (scenario 4 above).

---

## Week 5 — GCP + BigQuery Fundamentals

Understanding the core Google Cloud / BigQuery concepts (Cloud Storage, BigQuery, datasets/tables, IAM, service accounts, public datasets, query cost) without requiring advanced cloud administration.

**No GCP sandbox account was available**, so — per the plan's own fallback option — the exercise was reproduced locally: the retail dataset cleaned in Week 3 and loaded by the Week 4 pipeline was treated as a stand-in for a BigQuery dataset/table, and 8 analytical queries were written and run against it locally with the same cost-awareness principles that would apply on real BigQuery.

- **Notes:** `practice/week5/notes/GCP_BigQuery_notes.md` — explains each concept (Cloud Storage, BigQuery, Dataset, Table, IAM, Service Accounts, Public Datasets, Query Cost) and its local equivalent (or why no local equivalent exists, for IAM/Service Accounts).
- **Queries:** `practice/week5/sql/week5_queries.sql` — 8 analytical SQL queries against the local dataset, written with BigQuery's cost model in mind (only required columns selected, no `SELECT *`, no reliance on `LIMIT` to reduce cost, partitioning noted where relevant).
- **Local dataset:** `practice/week5/data/retail_sample.db` — a local SQLite database standing in for a BigQuery dataset/table.

### Run the queries
```bash
sqlite3 practice/week5/data/retail_sample.db
sqlite> .read practice/week5/sql/week5_queries.sql
```

---

## Week 6 — Analytics and Simple Data Modeling

Shapes the Week 4 pipeline's cleaned output (`sales_clean`) into a small, reusable analytics layer: views, calculated fields, basic fact/dimension thinking, and a set of KPIs — plus an interactive dashboard to explore them.

### Fact / Dimension Design
`sales_clean` (from `data/db/retail_pipeline.db`) is the **fact table**. Since it's a single flat table with no natural table to join against, a small **dimension table** was added specifically to enable a real JOIN exercise:

- **`category_targets`** — a hypothetical revenue target per category (business data not present in the source dataset), added in `sql/week6/views.sql`.

### Analytics Layer (`sql/week6/views.sql`)
Reusable SQL views built on top of `sales_clean`:
- `monthly_sales_summary` — transactions, revenue, and average order value per month.
- `category_summary` — revenue and units sold per category.
- `payment_method_summary` — revenue per payment method.
- `category_performance` — **JOIN** between `category_summary` and `category_targets`, with a **calculated field** (`target_achievement_pct`) showing how much of each category's target was achieved.

### KPI & Analytical Queries (`sql/week6/queries.sql`)
Reusable queries built on the views above, including:
- Overall average order value, best-performing month, month-over-month revenue growth (via `LAG()`).
- Top category by revenue, revenue share by payment method.
- **Underperforming categories** — a direct `JOIN` + `GROUP BY` + calculated field (`revenue_gap`) against `category_targets`, independent of the views, to demonstrate an explicit join.

### Example Results & Evidence
`reports/week6/Week6_Analytics_Summary.md` answers a set of practical analytical questions (e.g. *"Does any category generate disproportionately more revenue relative to its transaction count?"*, *"Is there a clear month-over-month trend?"*) using the actual query output as evidence, following the same evidence-based approach used for the `discount_applied` investigation in Week 3.

### Interactive Dashboard
`src/analytics/week6/app.py` — a Streamlit dashboard that runs `views.sql` automatically on startup and displays the KPIs, monthly trend, category performance (with the JOIN), underperforming categories, and payment method breakdown.

```bash
streamlit run src/analytics/week6/app.py
```
Opens automatically at `http://localhost:8501`. Requires `data/db/retail_pipeline.db` to already exist — run `python run_pipeline.py` first if it doesn't.

---

## Week 7 — RAG Foundations & Document Preparation

Prepares a small, clean, metadata-tagged document collection (invoices, policies, reports, plus real external PDFs) as the input Week 8's RAG assistant will embed and retrieve from. Covers the RAG foundation topics: PDF/text parsing, noise cleaning, chunking, metadata tagging, and table handling — no embedding generation or vector store yet, that's Week 8's scope.

- **Notebook (sandbox):** `notebooks/week7/week7-rag-preparation.ipynb` — where parsing, metadata extraction, and chunking were first calibrated against real sample data before any of it was moved into `src/rag/week7/`.
- **Full design-decision log:** `reports/week7/parsing_and_chunking_walkthrough.txt` — a detailed, step-by-step account of every parsing/cleaning/chunking decision, what each one does, and why, including every bug found by testing and how it was fixed.
- **Summary report:** `reports/week7/Report_week7.md`.

### Architecture
```
Scan → Load → Metadata (best-effort) → Clean → Chunk → Store
```

- **`document_loader.py`** — unified `.txt`/`.md`/`.pdf` reader. `pypdf` is the primary PDF extractor; `pdfplumber` is invoked narrowly, only when a page actually contains a table (see "Table Handling" below).
- **`metadata_extractor.py`** — classifies a document from its filename (`<type>_<YYYY-MM>_<project>_[service].<ext>`) via a single, whole-string, all-or-nothing regex. A filename that doesn't match is **never a hard failure** — it's tagged `unclassified` and still fully ingested. Classification is opt-in enrichment, not a gate: real documents (job-task briefs, a JSON spec) rarely follow a convention invented for our own invoices/policies/reports, and forcing a match either breaks traceability to the real file or silently produces wrong metadata (caught in testing: an early, lenient version of this parsed `email_payload_spec.pdf` as `project='spec'` — plausible-looking, completely wrong).
- **`text_cleaner.py`** — a `TextCleaner` class, run between loading and chunking: Unicode normalization, bullet-symbol unification, stray control-character removal, de-hyphenation, and (PDF only) reconstruction of real paragraph/sentence breaks from a pypdf text stream that otherwise preserves almost none. The PDF-only "aggressive" rules are explicitly never applied to `.md`/`.txt` — verified by testing that running them on our own clean policy file destroys its markdown headers.
- **`table_extractor.py`** — `pypdf` has no concept of a table: a PDF table extracts as flat, column-less text. `pdfplumber` is used only to detect tables and splice a clean Markdown version into the *same* position in the `pypdf` text (verified: the cell values `pdfplumber` extracts reproduce `pypdf`'s flattened text exactly, so the swap is safe and leaves zero duplication).
- **`chunking.py`** — one universal `DocumentChunker` for every document type and format (no per-type chunking strategies). Splits text into whole `\n\n`-delimited units and greedily packs them under a hard `chunk_size` safety cap (default 300 chars) — a unit is never split unless it alone exceeds the cap. "Overlap" (default 50 chars) carries at most one complete prior unit forward, never a raw character fragment. Every chunk is tagged with `section_header` (the header of the section it belongs to, propagated to *every* chunk of that section, not just the first) so retrieval-time context is available without duplicating the header into the stored chunk text itself.
- **`db_store.py`** — `RagStore` persists into `rag_documents` / `rag_chunks`, in the *same* `local_storage.sqlite` used by Task 1 but fully independent of `rag_topics`. `document_id` / `chunk_id` are app-generated UUIDs, not derived from the filename, so retrieval references stay stable even if a source file is later renamed. `original_filename` is stored verbatim for traceability.
- **`ingest.py`** — orchestrates all of the above, logging each phase per file and, like Week 4, attributing any failure to the specific phase that caused it rather than a generic error — while letting the rest of the batch continue.

### Run the pipeline
```bash
python run_rag_ingestion.py
```

### Database
Tables in `data/db/local_storage.sqlite` (independent of Task 1's `rag_topics`):

**`rag_documents`**

| Column | Type | Description |
|---|---|---|
| `document_id` | TEXT (UUID) | Primary key |
| `original_filename` | TEXT | Verbatim source filename |
| `document_type` | TEXT | `invoice` / `policy` / `report` / `NULL` if unclassified |
| `source_format` | TEXT | `pdf` / `txt` / `md` |
| `file_path`, `month`, `project`, `service` | TEXT | — |
| `classification_status` | TEXT | `classified` / `unclassified` |
| `created_at` | TEXT | — |

**`rag_chunks`**

| Column | Type | Description |
|---|---|---|
| `chunk_id` | TEXT (UUID) | Primary key |
| `document_id` | TEXT | FK → `rag_documents` |
| `chunk_index`, `chunk_text`, `char_count` | — | — |
| `section_header` | TEXT | Header of the section this chunk belongs to (may repeat across several chunks) |
| `original_filename`, `document_type`, `month`, `project`, `service` | TEXT | Denormalized for filterable retrieval without a JOIN |

### Logs
Every run appends to `logs/week7/ingestion.log`, with the same per-phase, per-file logging discipline as Week 4.

### Failure Handling — Verified Scenarios
1. **No input documents found (fatal):** `CRITICAL - Ingestion Terminated: No supported documents (.txt/.md/.pdf) found in <dir>`
2. **Filename doesn't match the naming convention (non-fatal):** logged as unclassified; the document is still loaded, cleaned, chunked and stored normally.
3. **Corrupted/unreadable PDF (non-fatal):** `ERROR - Load Phase FAILED for <file>: ...` — attributed to the correct phase; the rest of the batch still completes.
4. **Table extraction failure on an otherwise-readable PDF (non-fatal):** logged as a warning; the document still ingests using `pypdf`'s original (uncleaned) table text rather than failing the whole document.

### Latest Verified Run
```
Files found:  10 (7 authored: invoices/policies/reports, 3 real external PDFs in misc/)
Processed:    10 | Failed: 0
Classified:   7  | Unclassified: 3
Total chunks: 66
By type:   {invoice: 3, policy: 2, report: 2, unclassified: 3}
By format: {pdf: 4, txt: 4, md: 2}
```

### Known Limitations (see `Report_week7.md` for full detail)
- **PDF table trap:** solved for detected tables (via `pdfplumber` + Markdown splicing); a table `pdfplumber` fails to detect still extracts as flat, column-less text via `pypdf` alone.
- **Arabic PDF word-spacing:** `pypdf` can drop spaces between Arabic words during extraction (confirmed in the raw text, before any of our own code runs). Tested a second library (`PyMuPDF`) as a potential fix — result was mixed (fixed some occurrences, not others, introduced a different artifact elsewhere) — **not a reliably fixable issue with the current tooling**, documented as accepted technical debt.
- **Header/footer stripping:** a general-purpose stripper for repeated running headers / "Page X of Y" footers is implemented but currently a no-op — none of the real sample PDFs (up to 3 pages each) had any repeated boilerplate to strip.

---

## Week 8 — Basic RAG Assistant (Embeddings, Vector Store, Retrieval, Citations)

Turns the Week 7 document collection into a question-answering flow: local embeddings, a local vector store, retrieval
with metadata filtering, a grounded prompt, citations, handling of missing information, an evaluation set, and a
`POST /rag/ask` endpoint. The answer is extractive by default; Week 9 adds the LLM.

### Architecture
```
rag_chunks (SQLite, source of truth)
   └─ run_rag_indexing.py ─► embeddings ─► ChromaDB (cosine)
question ─► embed ─► top-k (+ filters) ─► answerability gate ─► grounded prompt ─► extractive answer + citations
```
- **SQLite stays the source of truth.** Chroma holds vectors and filter fields only; text is read back by `chunk_id`.
- **Indexing is a sync, not a rebuild:** it compares the sets of `chunk_id`s, embeds what is new, deletes what disappeared, and does not even load the model when nothing changed. The collection records the embedding model and text variant; a mismatch is refused (`--rebuild`).
- **Answerability gate.** One cosine threshold cannot separate answerable from unanswerable questions (their scores overlap), so the best chunk must share ≥ 50% of the question's key words and the question may not name an entity that appears nowhere in the documents. For questions in a different language than the best chunk (no shared words) the word check is replaced by a cosine floor of 0.30, fitted on a separate calibration split.
- **Three statuses:** `answered`, `no_relevant_context`, `no_documents_match_filter`. Refusals return their reasons and the measured signals.
- **Shared infrastructure change.** `RagStore.replace_document()` makes re-ingestion **replace** a document and its chunks in one transaction (it used to append duplicates), and removes documents whose file disappeared. `health_controller.py` mounts the RAG router and loads the model once at startup; `/health` is unchanged.

### Run
```bash
python run_rag_ingestion.py          # Week 7: documents -> rag_documents / rag_chunks
python run_rag_indexing.py           # Week 8: chunks -> vectors   (--rebuild after changing the model)
uvicorn src.api.health_controller:app --reload       # http://127.0.0.1:8000/docs
python run_rag_query.py "What is the total amount due?" --filter project=projectBeta --filter document_type=invoice --ask
python run_rag_eval.py               # evaluation
```
Run `run_rag_indexing.py` after every ingestion (chunk ids change); restart the API after changing the documents.

### API
`POST /rag/ask`
```json
{ "question": "What is the total amount due?", "top_k": 3,
  "filters": {"project": "projectBeta", "document_type": "invoice"}, "include_prompt": false }
```
Allowed filters (exact stored values): `document_type`, `month`, `project`, `service`, `original_filename`.
Response: `status`, `answer`, `citations` (file, section, chunk, score, text), `answerability` (reasons + signals),
`latency_ms`; with `include_prompt=true` also the grounded `prompt` and every retrieved chunk.
`GET /rag/status` compares the index with SQLite by chunk id.

### Failure handling — verified scenarios
1. Index not built: `/rag/ask` → 503 `run python run_rag_indexing.py`; the app and `/health` still work, and building the index needs no restart.
2. Stale index (re-ingested, not re-indexed; same count, different ids): `/rag/status` shows `in_sync: false`, `/rag/ask` → 503.
3. Embedding model cannot be loaded: clear error (hint: stale Hugging Face token), index untouched, `/health` still 200.
4. Filters match nothing: `no_documents_match_filter`.
5. Invalid request (empty question, `top_k` outside 1–10, unknown filter or field): 422.
6. Index built with another model or text variant: refused.
7. Documents with NULL metadata are indexed (`document_type="unclassified"`) because Chroma rejects `None`.

### Bugs found and fixed during Week 8
- **Re-ingestion duplicated everything** (Week 7 code): fixed with replace-in-one-transaction.
- **Stale-index check by count was wrong:** re-ingestion keeps the count and changes every id; now compared by id.
- **Model loaded inside the first request** (20–50 s): now loaded once at startup.
- **Every valid cross-language question was refused** (0/17): found by the evaluation; fixed as described above (17/17 answered, holdout 5/5).
- **The pronoun "I" counted as an unknown entity:** stop words are no longer entities.

### Latest verified run (84 questions: 56 answerable, 28 unanswerable)
```
Index:         60 chunks / 60 vectors
Retrieval:     Hit@1 94.6% | Hit@3 98.2% | Hit@5 100% | MRR 0.969
Answerable:    47 correct | 4 right source, fact missing | 2 wrong source | 3 false refusals
Unanswerable:  15 rejected | 13 leaked
Latency:       p50 17 ms | p95 25 ms
10 predefined questions: correct source retrieved 8/8 answerable; answered correctly 7/8; unanswerable refused 1/2
```
Details: `reports/week8/retrieval_evaluation.md`.

### Known limitations (see `Report_week8.md`)
- Questions about an existing topic whose specific fact is missing mostly pass the gate (12 of 17 leak); separating them needs an LLM or a cross-encoder.
- The cross-language gate only removes off-topic questions.
- The lexical gate is literal: 3 false refusals (Arabic morphology, paraphrased wording, one retrieval miss).
- Extractive answers are verbatim chunks: 4 right-source answers lack the fact.
- Small evaluation set (84 questions, 11 in the holdout).

---

## Week 9 — LLM Application Basics (Gemini): prompts, structured outputs, fallback, summaries and insights

Puts a language model on top of the Week 8 RAG and the cleaned retail data. Gemini (Google AI Studio key) writes the
answers; every function has a fallback that needs no model; the model's output is checked before it is trusted. No Google
Cloud account was available, so Vertex AI is documented in `practice/week9/notes/Gemini_Vertex_notes.md`, not executed.

### Architecture
```
question ─► retrieve ─► gate (lenient) ─► Gemini (system + user, JSON schema) ─► guardrails ─► answer + citations
                                               │  error, or output rejected twice (one repair request in between)
                                               ▼
                         extractive answer (Week 8), judged AGAIN by the stricter gate
```
- **Structured outputs.** The model answers in a fixed JSON shape (`schemas.py`); the shape is validated, then the content: cited passages must exist, every number must appear in the source (passages or computed facts).
- **System / user messages.** Rules and examples in the system message; passages and question in the user message, wrapped so a passage cannot talk as the prompt.
- **Retries and timeouts.** Transient errors (timeout, 429 per-minute, 5xx, network) are retried with backoff inside a total time budget; wrong key, unknown model, bad request and an exhausted daily quota fail at once. A throttle keeps requests under the per-minute limit; one shared client means one limiter for the whole app. Every attempt is logged to `logs/week9/llm.log` (the key is removed from any message).
- **Two-tier gate.** `RAG_MIN_LEXICAL_COVERAGE` (0.33) lets questions reach the model, which can answer "not found". A verbatim chunk (extractive mode, or the fallback when the model fails) is judged again with `RAG_EXTRACTIVE_MIN_LEXICAL_COVERAGE` (0.50). Statuses are unchanged: `answered`, `no_relevant_context` (the model's "not found" adds the reason `llm_not_found`), `no_documents_match_filter`.
- **Sales insights.** `data_facts.py` computes the figures with SQL on `sales_clean` (revenue, months and growth, categories vs targets, payment methods, locations; optional category and month range). The model words them; incomplete first/last months are marked and left out of best/worst/growth; the targets are labelled hypothetical. Fallback: a fixed template from the same figures.
- **Document summary.** Passages chosen by filters or by a question (the same search as `/rag/ask`), up to 20, numbered; the model returns a summary and key points with their source numbers. Fallback: the passages as written (a large selection: the opening passage of each document).
- **Function calling (concept).** Two tools (`search_documents`, `get_sales_summary`): the model chooses one and its arguments, `tool_specs.py` validates them and runs the tool without ever raising. Sending results back and routing are Week 10–11.

### Run
```bash
# .env: GEMINI_API_KEY, GEMINI_MODEL (+ the Week 8 setup: ingestion and indexing already done)
python -m pytest tests/week9 -v                                     # no key needed
python run_rag_query.py "What is the total amount due?" --ask --mode llm      # or --mode extractive
python run_llm_insights.py sales --focus "Which categories are behind their revenue target?"
python run_llm_insights.py sales --category Furniture --start 2024-01 --end 2024-06 --language ar
python run_llm_insights.py docs --filter project=projectBeta --filter document_type=invoice
python run_llm_insights.py --no-llm sales                            # the fallback, no request is sent
python run_llm_tools.py --demo                                       # 6 requests: which tool does the model choose?
python run_llm_eval.py --variants extractive@0.50,llm@0.33 --ids 1,2,3,4,5,6,7,8,9,10 --max-requests 15
python run_gate_experiment.py                                        # lexical threshold, no model
uvicorn src.api.health_controller:app --reload                       # /docs
```

### API
- `POST /rag/ask`: unchanged request; the response has a new `generation` block (`mode` = `llm` / `extractive` / `extractive_fallback`, `fallback_reason`, `model`, `prompt_version`, `attempts`, `repairs`, tokens incl. `thinking_tokens`, `model_latency_ms`). With `include_prompt=true`, `prompt` is the system + user text actually sent.
- `POST /llm/sales-insights` `{category?, start?, end?, focus?, language: en|ar}` → `{mode, result: {summary, insights, caveats}, facts, notes, generation}`.
- `POST /llm/summarize-documents` `{question?, filters?, top_k?, language?}` (a question or at least one filter) → `{mode, result: {summary, key_points: [{text, sources}]}, sources, generation}`.
- Both new endpoints answer even without a key (`mode: "fallback"` + the reason); they return 422 for a bad request, 404 for no matching documents, 503 for missing data.

### Failure handling — verified scenarios (real API)
1. **Wrong key:** `LLMAuthError`, one attempt, no retry, the key is not in the output, fallback returned.
2. **Wrong model:** `LLMConfigError` (404) with Google's message, no retry.
3. **Timeout:** `LLMTimeoutError`, one retry as configured, then the fallback.
4. **Daily quota exhausted:** detected as permanent; the evaluation stops, never scores unanswered questions as if the model had answered, and resumes from the cache.
5. **Model down, question accepted only by the lenient gate:** `no_relevant_context` ("stricter gate ... minimum 50%") instead of quoting a chunk.

### Bugs found and fixed during Week 9
- **Fallback loophole with a lenient gate:** a failed model call returned a chunk for a question that only the lenient gate accepted → verbatim answers are judged by the stricter gate.
- **A permanent 429 was retried three times** (15–30 s each) → quota exhaustion is not retried; the provider's retry delay is honoured.
- **20 s timeout cut valid answers** (median 9 s, 95th percentile 53 s on one model) → 45 s, and one retry.
- **Error messages cut at 300 characters** hid the quota details → 1000.
- **`include_prompt` showed the Week 8 prompt** → it shows what was sent.
- **Cached replies reported a 20 ms latency** → latency is the model's own time.
- **Sales insights:** the model needed the number 100 to say "no category is behind its target" (rejected by the numbers check) → definitions in the facts; the last month (2025-01) was incomplete and read as a decline → incomplete months flagged; the template was silent when no category was behind → it now says so.
- **Summary fallback lost the amount due** (one point per document) → small selections are shown passage by passage.
- **A cache keyed on the configuration object** failed for non-hashable configs → explicit key; one shared client.

### Latest verified runs (Gemini 3.5 Flash Lite)
```
Week 8's 10 questions     extractive: 7/8 answerable correct, 1/2 unanswerable rejected
                          LLM @ 0.33: 8/8 correct, 2/2 rejected (Q9 by the model)
Everything tried (18)     answerable 11/12 correct (Q27 false refusal); unanswerable 6/6 rejected (5 by the model)
Cost                      RAG answer ~850 in / ~45 out tokens; sales insights ~2,560 / ~350; no hidden thinking tokens
Model time                median 5.8 s, 95th percentile 11.3 s
```
Details: `reports/week9/llm_evaluation.md`.

### Known limitations (see `Report_week9.md`)
- Small evaluation (18 different questions with the model): a difference of one question is not evidence.
- The numbers check misses small numbers that always occur in the source (years, months, counts) and does not judge whether a sentence makes sense.
- Wording is not deterministic between runs; the evaluation caches replies.
- Free-tier limits depend on the model; free-tier prompts may be used by Google (only synthetic and public data are sent).
- The fallback texts are English only; the sales targets are hypothetical (Week 6).

---

## Git Tags

Each completed week is marked with an annotated Git tag once finalized, so the project history can be browsed one week at a time:

```bash
git tag
git checkout tags/week2-complete
git checkout tags/week3-complete
git checkout tags/week4-complete
git checkout tags/week5-complete
git checkout tags/week6-complete
git checkout tags/week7-complete
git checkout tags/week8-complete
git checkout tags/week9-complete
```