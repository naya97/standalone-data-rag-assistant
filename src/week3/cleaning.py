import pandas as pd
import numpy as np


class DataCleaner:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.imputation_stats = {}

    def clean_column_names(self) -> 'DataCleaner':
        self.df.columns = (
            self.df.columns.str.strip().str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')
        )
        return self

    def convert_data_types(self) -> 'DataCleaner':
        text_cols = ['transaction_id', 'customer_id', 'category', 'item', 'payment_method', 'location']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('string')

        if 'price_per_unit' in self.df.columns:
            self.df['price_per_unit'] = pd.to_numeric(self.df['price_per_unit'], errors='coerce').astype('float64')

        if 'total_spent' in self.df.columns:
            self.df['total_spent'] = pd.to_numeric(self.df['total_spent'], errors='coerce').astype('float64')

        if 'quantity' in self.df.columns:
            self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors='coerce').astype('Int64')

        if 'discount_applied' in self.df.columns:
            self.df['discount_applied'] = self.df['discount_applied'].astype('boolean')

        if 'transaction_date' in self.df.columns:
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'], errors='coerce')

        return self

    def normalize_text(self) -> 'DataCleaner':
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns
        for col in text_cols:
            self.df[col] = self.df[col].str.strip()
        return self

    def impute_financial_columns(self) -> 'DataCleaner':
        # 1. Price = Total / Quantity
        mask_price_from_total = (
            self.df['price_per_unit'].isna() & self.df['quantity'].notna()
            & self.df['total_spent'].notna() & (self.df['quantity'] != 0)
        )
        self.imputation_stats['price_from_total_quantity'] = int(mask_price_from_total.sum())
        self.df.loc[mask_price_from_total, 'price_per_unit'] = (
            self.df.loc[mask_price_from_total, 'total_spent']
            / self.df.loc[mask_price_from_total, 'quantity']
        )

        # 2. Price = Price associated with Item (only unambiguous mappings)
        valid_item_prices = self.df.dropna(subset=['item', 'price_per_unit'])
        item_price_counts = valid_item_prices.groupby('item')['price_per_unit'].nunique()
        unambiguous_items = item_price_counts[item_price_counts == 1].index
        item_price_map = (
            valid_item_prices[valid_item_prices['item'].isin(unambiguous_items)]
            .drop_duplicates('item').set_index('item')['price_per_unit'].to_dict()
        )
        mask_price_from_item = (
            self.df['price_per_unit'].isna() & self.df['item'].notna()
            & self.df['item'].isin(item_price_map)
        )
        self.imputation_stats['price_from_item_mapping'] = int(mask_price_from_item.sum())
        self.df.loc[mask_price_from_item, 'price_per_unit'] = (
            self.df.loc[mask_price_from_item, 'item'].map(item_price_map)
        )

        # 3. Quantity = Total / Price (accepted only if the result is an integer)
        mask_qty_from_total_price = (
            self.df['quantity'].isna() & self.df['total_spent'].notna()
            & self.df['price_per_unit'].notna() & (self.df['price_per_unit'] != 0)
        )
        calculated_qty = (
            self.df.loc[mask_qty_from_total_price, 'total_spent']
            / self.df.loc[mask_qty_from_total_price, 'price_per_unit']
        )
        valid_qty = np.isclose(calculated_qty, calculated_qty.round())
        self.imputation_stats['quantity_from_total_price'] = int(valid_qty.sum())
        self.imputation_stats['quantity_rejected_non_integer'] = int((~valid_qty).sum())
        self.df.loc[calculated_qty.index[valid_qty], 'quantity'] = (
            calculated_qty[valid_qty].round().astype('Int64')
        )

        # 4. Total = Quantity * Price
        mask_total_from_price_qty = (
            self.df['total_spent'].isna() & self.df['price_per_unit'].notna()
            & self.df['quantity'].notna()
        )
        self.imputation_stats['total_from_quantity_price'] = int(mask_total_from_price_qty.sum())
        self.df.loc[mask_total_from_price_qty, 'total_spent'] = (
            self.df.loc[mask_total_from_price_qty, 'quantity']
            * self.df.loc[mask_total_from_price_qty, 'price_per_unit']
        )
        return self

    def remove_irrecoverable_rows(self) -> 'DataCleaner':
        financial_cols = ['price_per_unit', 'quantity', 'total_spent']
        missing_counts = self.df[financial_cols].isna().sum(axis=1)
        self.imputation_stats['rows_removed_irrecoverable'] = int((missing_counts >= 2).sum())
        self.df = self.df[missing_counts < 2].copy()
        return self

    def impute_item(self) -> 'DataCleaner':
        valid_items = self.df.dropna(subset=['item', 'category', 'price_per_unit'])

        mapping_check = valid_items.groupby(['category', 'price_per_unit'])['item'].nunique()
        valid_keys = mapping_check[mapping_check == 1].index

        unique_mapping = (
            valid_items
            .drop_duplicates(subset=['category', 'price_per_unit'])
            .set_index(['category', 'price_per_unit'])['item']
        )

        item_map = unique_mapping.loc[valid_keys].to_dict()

        missing_item_mask = self.df['item'].isna()
        recovered_count = 0
        if missing_item_mask.any():
            imputed_values = self.df.loc[missing_item_mask].apply(
                lambda row: item_map.get((row['category'], row['price_per_unit']), pd.NA),
                axis=1
            )
            recovered_count = int(imputed_values.notna().sum())
            self.df.loc[missing_item_mask, 'item'] = imputed_values

        self.imputation_stats['item_from_category_price_mapping'] = recovered_count
        return self

    def clean_all(self) -> pd.DataFrame:
        return (
            self.clean_column_names()
            .convert_data_types()
            .normalize_text()
            .impute_financial_columns()
            .remove_irrecoverable_rows()
            .impute_item()
            .df
        )