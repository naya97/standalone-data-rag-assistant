import pandas as pd
import numpy as np
import re


class DataValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_column_names(self) -> bool:
        pattern = r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$'
        is_snake_case = all(bool(re.match(pattern, col)) for col in self.df.columns)
        print(f"[Schema] Column Names Snake Case: {'PASS' if is_snake_case else 'FAIL'}")
        return is_snake_case

    def validate_data_types(self) -> bool:
        expected_types = {
            'transaction_id': 'string', 'customer_id': 'string', 'category': 'string',
            'item': 'string', 'price_per_unit': 'float64', 'quantity': 'Int64',
            'total_spent': 'float64', 'payment_method': 'string', 'location': 'string',
            'payment_method': 'string', 'location': 'string',
            'discount_applied': 'boolean'
        }
        valid = True
        for column, expected_type in expected_types.items():
            if column not in self.df.columns:
                print(f"[Schema] {column}: MISSING COLUMN")
                valid = False
                continue
            actual_type = str(self.df[column].dtype)
            if actual_type != expected_type:
                print(f"[Schema] {column}: FAIL (expected {expected_type}, got {actual_type})")
                valid = False

        # فحص التاريخ بشكل مرن (أي دقة datetime64 مقبولة)
        if 'transaction_date' not in self.df.columns:
            print("[Schema] transaction_date: MISSING COLUMN")
            valid = False
        elif not pd.api.types.is_datetime64_any_dtype(self.df['transaction_date']):
            print(f"[Schema] transaction_date: FAIL (expected datetime, got {self.df['transaction_date'].dtype})")
            valid = False

        print(f"[Schema] Data Types Validation: {'PASS' if valid else 'FAIL'}")
        return valid

    def validate_missing_values(self) -> bool:
        allowed_missing = {'discount_applied'}
        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        unexpected = missing[~missing.index.isin(allowed_missing)]

        print(f"[Missing Data] Missing Values: {'PASS' if unexpected.empty else 'FAIL'}")
        if not missing.empty:
            print(missing)
        return unexpected.empty

    def validate_irrecoverable_rows(self) -> bool:
        fin_cols = ['price_per_unit', 'quantity', 'total_spent']
        irrecoverable = (self.df[fin_cols].isna().sum(axis=1) >= 2).sum()
        status = 'PASS' if irrecoverable == 0 else f'FAIL ({irrecoverable} rows)'
        print(f"[Missing Data] Irrecoverable Rows Check: {status}")
        return irrecoverable == 0

    def validate_duplicates(self) -> bool:
        dupes = self.df.duplicated().sum()
        status = 'PASS' if dupes == 0 else f'FAIL ({dupes} duplicates)'
        print(f"[Duplicates] Exact Row Duplicates: {status}")
        return dupes == 0

    def validate_transaction_id_uniqueness(self) -> bool:
        if 'transaction_id' not in self.df.columns:
            print("[Duplicates] Transaction ID: FAIL (missing column)")
            return False
        valid = not self.df['transaction_id'].isna().any() and self.df['transaction_id'].is_unique
        print(f"[Duplicates] Transaction ID Uniqueness: {'PASS' if valid else 'FAIL'}")
        return valid

    def validate_total_spent_consistency(self, tolerance: float = 1e-6) -> bool:
        check_df = self.df.dropna(subset=['price_per_unit', 'quantity', 'total_spent'])
        if check_df.empty:
            print("[Financial] Total Spent Consistency: PASS (no complete rows to check)")
            return True
        is_consistent = np.isclose(
            check_df['total_spent'], check_df['quantity'] * check_df['price_per_unit'],
            rtol=tolerance, atol=tolerance
        ).all()
        print(f"[Financial] Total Spent Consistency: {'PASS' if is_consistent else 'FAIL'}")
        return is_consistent

    def validate_value_ranges(self) -> bool:
        invalid_price = (self.df['price_per_unit'] <= 0).sum()
        invalid_qty = (self.df['quantity'] <= 0).sum()
        invalid_total = (self.df['total_spent'] <= 0).sum()
        quantity_non_integer = (self.df['quantity'].dropna() % 1 != 0).sum()

        valid = invalid_price == 0 and invalid_qty == 0 and invalid_total == 0 and quantity_non_integer == 0
        print(f"[Ranges] Value Range Validation: {'PASS' if valid else 'FAIL'}")
        if not valid:
            print(f"  Invalid prices: {invalid_price}")
            print(f"  Invalid quantities: {invalid_qty}")
            print(f"  Invalid totals: {invalid_total}")
            print(f"  Non-integer quantities: {quantity_non_integer}")
        return valid

    def validate_categorical_values(self) -> bool:
        valid_payments = {"Digital Wallet", "Credit Card", "Cash"}
        valid_locations = {"Online", "In-store"}

        invalid_pay = self.df[~self.df['payment_method'].isin(valid_payments) & self.df['payment_method'].notna()]
        invalid_loc = self.df[~self.df['location'].isin(valid_locations) & self.df['location'].notna()]

        valid = len(invalid_pay) == 0 and len(invalid_loc) == 0
        print(f"[Categorical] Categorical Values: {'PASS' if valid else 'FAIL'}")
        print(f"  Invalid Payment Methods: {len(invalid_pay)}")
        print(f"  Invalid Locations: {len(invalid_loc)}")
        return valid

    def validate_dates(self) -> bool:
        nat_count = self.df['transaction_date'].isna().sum()
        status = 'PASS' if nat_count == 0 else f'FAIL ({nat_count} NaT)'
        print(f"[Dates] Unparsable Dates: {status}")
        return nat_count == 0

    def validate_discount_values(self) -> bool:
        valid = str(self.df['discount_applied'].dtype) == 'boolean'
        print(f"[Discount] Nullable Boolean Type: {'PASS' if valid else 'FAIL'}")
        return valid

    def validate_item_mapping(self) -> bool:
        mapping_check = (
            self.df.dropna(subset=['item', 'category', 'price_per_unit'])
            .groupby(['category', 'price_per_unit'])['item'].nunique()
        )
        ambiguous = mapping_check[mapping_check > 1]
        status = 'PASS' if ambiguous.empty else f'FAIL ({len(ambiguous)} ambiguous mappings)'
        print(f"[Logic] Category + Price -> Item Mapping: {status}")
        return ambiguous.empty

    def report_outliers_iqr(self, column: str) -> int:
        if column not in self.df.columns or self.df[column].dropna().empty:
            print(f"[Outliers] {column}: No data available.")
            return 0
        q1, q3 = self.df[column].quantile(0.25), self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]
        print(f"[Outliers] {column}: {len(outliers)} statistical outliers detected (retained based on business rules).")
        return len(outliers)

    def run_all_validations(self):
        print("=== DATA QUALITY VALIDATION REPORT ===")
        results = {
            'column_names': self.validate_column_names(),
            'data_types': self.validate_data_types(),
            'missing_values': self.validate_missing_values(),
            'irrecoverable_rows': self.validate_irrecoverable_rows(),
            'duplicates': self.validate_duplicates(),
            'transaction_id': self.validate_transaction_id_uniqueness(),
            'total_spent_consistency': self.validate_total_spent_consistency(),
            'value_ranges': self.validate_value_ranges(),
            'categorical_values': self.validate_categorical_values(),
            'dates': self.validate_dates(),
            'discount': self.validate_discount_values(),
            'item_mapping': self.validate_item_mapping(),
        }
        self.report_outliers_iqr('price_per_unit')
        self.report_outliers_iqr('total_spent')
        print("========================================")
        return results