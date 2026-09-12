import sqlite3
import logging
from typing import List, Dict

class SQLiteManager:
    def __init__(self, db_path: str, initialize_default_schema: bool = False):
        self.db_path = db_path
        if initialize_default_schema:
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

    # Week4 
    def insert_dataframe(self, table_name: str, df) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Loaded {len(df)} rows into '{table_name}'")
        except sqlite3.Error as e:
            logging.error(f"Database Load Error: {e}")
            raise
