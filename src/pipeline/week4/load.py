import pandas as pd
import logging
from src.infrastructure.db_manager import SQLiteManager

def load_data(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    logging.info("Load: Initiating database connection.")
    try:
        db = SQLiteManager(db_path, initialize_default_schema=False)
        db.insert_dataframe(table_name, df)
        logging.info(f"Load: {len(df)} rows written to '{table_name}' successfully.")
    except Exception as e:
        logging.error(f"Load Phase FAILED: {e}")
        raise