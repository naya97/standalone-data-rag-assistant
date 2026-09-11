import pandas as pd
import logging
from src.week3.cleaning import DataCleaner
from src.week3.validation import DataValidator

class DataTransformer:
    @staticmethod
    def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Transform: Starting data cleaning and validation.")
        try:
            cleaner = DataCleaner(df)
            df_clean = cleaner.clean_all()
            logging.info(f"Transform: Imputation stats: {cleaner.imputation_stats}")

            validator = DataValidator(df_clean)
            results = validator.run_all_validations()

            if not all(results.values()):
                raise ValueError("Validation checks failed after cleaning.")

            logging.info(f"Transform: {len(df_clean)} rows passed all validations.")
            return df_clean
        except Exception as e:
            logging.error(f"Transform Phase FAILED: {e}")
            raise