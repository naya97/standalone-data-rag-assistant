import os
import pandas as pd
import logging

def extract_data(input_dir: str) -> pd.DataFrame:
    logging.info(f"Extract: Scanning directory {input_dir} for CSV files.")
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dataframes = []
    for filename in csv_files:
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Extract: Read {len(df)} rows from {filename}")
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Extract FAILED for {filename}: {e}")
            raise

    combined_df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Extract: Combined {len(combined_df)} rows from {len(csv_files)} file(s)")
    return combined_df