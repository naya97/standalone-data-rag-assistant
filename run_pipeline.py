import os
import logging
from src.pipeline.week4.extract import extract_data
from src.pipeline.week4.transform import DataTransformer
from src.pipeline.week4.load import load_data

os.makedirs('logs/week4', exist_ok=True)
logging.basicConfig(
    filename='logs/week4/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_etl_pipeline():
    source_dir = "data/raw/week4"
    db_destination = "data/db/retail_pipeline.db"
    target_table = "sales_clean"

    logging.info("--- Starting ETL Pipeline ---")
    try:
        raw_data = extract_data(source_dir)
        clean_data = DataTransformer.clean_and_validate(raw_data)
        load_data(clean_data, db_destination, target_table)
        logging.info("--- ETL Pipeline Finished Successfully ---")
    except Exception as e:
        logging.critical(f"--- ETL Pipeline Terminated Due To Error: {e} ---")

if __name__ == "__main__":
    run_etl_pipeline()