# Standalone Data-RAG Assistant Engine

A simple standalone project built to practice core Python and data basics: reading CSV/JSON files, storing data in SQLite, running basic SQL queries, and exposing a simple REST API.

The project loads sample question/category data from CSV and JSON files, stores it in a local SQLite database, runs a few basic SQL queries on it, and provides a `/health` endpoint to check the system status.

---

## Project Structure

```
standalone-data-rag-assistant/
├── data/
│   ├── db/
│   │   └── local_storage.sqlite      # SQLite database
│   └── raw/
│       ├── config.json               # App config
│       ├── sample.csv                # Sample data
│       └── sample.json               # Sample data
├── src/
│   ├── api/
│   │   └── health_controller.py      # FastAPI app + /health endpoint
│   └── infrastructure/
│       ├── db_manager.py             # SQLite manager
│       └── file_reader.py            # CSV/JSON reader
├── .env                               # Environment variables
├── main.py                            # Entry point (ETL + SQL queries)
├── requirements.txt
└── README.md
```

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

## Run the ETL script

Loads the CSV/JSON sample data into SQLite and runs sample SQL queries (SELECT, WHERE, GROUP BY).

```bash
python main.py
```

---

## Run the API

```bash
uvicorn src.api.health_controller:app --reload
```

Then visit:

```
http://127.0.0.1:8000/health
```

Docs: `http://127.0.0.1:8000/docs`

---

## Database

Table `rag_topics`:

| Column     | Type    | Description       |
|------------|---------|--------------------|
| `id`       | INTEGER | Primary key        |
| `question` | TEXT    | Question text      |
| `category` | TEXT    | Question category  |