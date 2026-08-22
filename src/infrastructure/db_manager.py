import sqlite3
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_tables()

    def _initialize_tables(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS rag_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            category TEXT NOT NULL
        )
        """
        self.execute_non_query(query)

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        # Execute a SQL query and return the results as a list of dictionaries.
        try:
            with sqlite3.connect(self.db_path) as conn:
                       conn.row_factory = sqlite3.Row
                       cursor = conn.cursor()
                       cursor.execute(query, params)
                       rows = cursor.fetchall()
                       return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logging.error(f"Database Query Error: {e}")
            return []

    def execute_non_query(self, query: str, params: tuple = ()) -> None:
        # Execute a SQL command that does not return any results (e.g., INSERT, UPDATE).
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database Non-Query Error: {e}")
