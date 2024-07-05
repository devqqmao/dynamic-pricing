import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from typing import TYPE_CHECKING

from dynamic_pricing.ml.release import utils

if TYPE_CHECKING:
    from dynamic_pricing.ml.release.hyper_parameters import HyperParameters, Shop
    from dynamic_pricing.ml.release.model.load.load_dataset import DatasetLoader


class DatasetPreprocessorOnMeta:
    def __init__(self, hyper_parameters: 'HyperParameters', dataset_loader: 'DatasetLoader') -> None:
        super().__init__()
        self.dataset_loader = dataset_loader
        self.hp = hyper_parameters
        self.shop_name = None

    def _sort_index(self, df: pd.DataFrame):
        return df.sort_index(inplace=False)

    def _choose_columns(self, df, columns):
        df = df.loc[:, columns]
        return df

    def _add_shop_id_column(self, df: pd.DataFrame):
        # TODO: Unused by now
        # Don't forget to transform
        df["shop_id"] = self.shop_name
        return df

    def _rename_columns(self, df):
        df = df.rename({"quantity": "demand",
                        "marketing_seller_price": "price",
                        "offer_id": "offer_id",
                        "shop_id": "shop_id",
                        "date": "date",
                        }, axis=1, inplace=False)

        return df

    def _choose_offer_ids(self, df: pd.DataFrame, offer_ids):
        df = df[(df['offer_id'].isin(offer_ids))]
        return df

    def _choose_dates(self, df: pd.DataFrame):
        meta: 'Shop' = self.hp.shops.meta[self.shop_name]
        return df.loc[meta.start_date:meta.end_date]

    def _fix_order(self, df, columns):
        return df.loc[:, columns]

    def _set_index(self, df: pd.DataFrame):
        return df.set_index(pd.to_datetime(df['date']))

    def _resample(self, df):
        df = df.groupby('product').resample('D').agg(
            {'demand': 'mean',
             'price': 'mean',
             }).reset_index(level=0)
        return df

    def _fill_nans(self, df: pd.DataFrame):

        df.set_index(["product", df.index], inplace=True)
        df["price"].replace(0, np.nan, inplace=True)
        df["demand"].fillna(0, inplace=True)

        df = df.groupby(level=0).bfill()
        df.reset_index(0, inplace=True)

        assert df.isna().sum().sum() == 0

        return df

    def _get_zero_product_ids(self, df, threshold_per_day=0, threshold_total=0.9):
        offer_ids = []
        for product in df['product'].unique():
            ts = df[df['product'] == product]
            n = len(ts)
            zero_demand_count = pd.Series([ts['demand'] <= threshold_per_day]).sum().sum()
            if zero_demand_count > (n * threshold_total):
                # выкидываем эти
                offer_ids.append(product)
        return offer_ids

    def _get_short_product_ids(self, df):
        days_on_sale: int = self.hp.shops.meta[self.shop_name].days_on_sale
        ts = df.groupby(["product"]).size() < days_on_sale
        return ts[ts].index

    def _choose_product_ids(self, df):
        product_ids = set(df["product"].unique())

        product_ids_0 = self._get_zero_product_ids(df)
        product_ids_1 = self._get_short_product_ids(df)

        product_ids -= set(product_ids_0)
        product_ids -= set(product_ids_1)

        product_ids = list(product_ids)
        return product_ids

    def _trim_dates(self, df):
        start_date = self.hp.shops.meta[self.shop_name].start_date
        end_date = self.hp.shops.meta[self.shop_name].end_date
        return df[(df.index >= pd.Timestamp(start_date)) & (df.index < pd.Timestamp(end_date))]

    def _trim_invalid_products(self, df):
        product_ids = self._choose_product_ids(df)
        df = df.loc[df['product'].isin(product_ids)]
        return df

    def _generate_unique_product_ids(self, df):
        df["product"] = self.shop_name + "_" + df["offer_id"].astype("string")
        return df

    def _get_unique_product_ids(self, df):
        product_ids = list(df["product"].unique())
        return product_ids

    def _get_unique_offer_ids(self, df):
        offer_ids = list(df["offer_id"].unique())
        return offer_ids

    def _step_0_set_dates(self, df):
        df = self._set_index(df)
        df = self._sort_index(df)
        if self.hp.mode.train_on_dates:
            df = self._trim_dates(df)
        return df

    def _get_dp_offer_ids(self):
        pass

    def _step_1_choose_and_set_offer_ids(self, df, shop_name):
        # all local
        if self.hp.mode.train_on_meta_offers:
            offer_ids = self.hp.shops.meta[self.shop_name].offer_ids
        elif self.hp.mode.train_on_all_offers:
            offer_ids = self._get_unique_offer_ids(df)
        elif self.hp.mode.train_on_dp_offers:
            offer_ids = self.dataset_loader.get_meta(shop_name, "dp_offer_ids")
        else:
            raise Exception

        self.hp.shops.meta[shop_name].offer_ids = offer_ids
        print("len(offer_ids)", len(offer_ids))
        df = self._choose_offer_ids(df, offer_ids)
        return df

    def _step_2_choose_columns(self, df):
        columns = ["demand", "price", "product"]
        df = self._choose_columns(df, columns)
        df = self._fix_order(df, columns)
        return df

    def _step_3_preprocess(self, df):
        df = self._resample(df)

        df = self._sort_index(df)
        df = self._fill_nans(df)
        df = self._sort_index(df)

        return df

    def preprocess_dataset(self, df, shop_name):
        self.shop_name = shop_name
        df = self._rename_columns(df)

        df = self._step_0_set_dates(df)
        df = self._step_1_choose_and_set_offer_ids(df, shop_name)

        df = self._generate_unique_product_ids(df)

        df = self._step_2_choose_columns(df)
        df = self._step_3_preprocess(df)

        df = self._trim_invalid_products(df)

        product_ids = self._get_unique_product_ids(df)
        print("unique_product_ids:", len(product_ids))
        self.hp.shops.product_ids.extend(product_ids)

        df = self._sort_index(df)

        return df


class DatasetPreprocessor:
    def __init__(self, hyper_parameters: 'HyperParameters', dataset_loader: 'DatasetLoader'):
        self.dataset_loader = dataset_loader
        self.hp = hyper_parameters
        self.loader = None
        self.init_preprocessor()

    def init_preprocessor(self):
        self.loader = DatasetPreprocessorOnMeta(self.hp, self.dataset_loader)

    def preprocess_dataset(self, df, shop_name):
        print(f"enter: preprocess_dataset: {shop_name}")
        df = self.loader.preprocess_dataset(df, shop_name)
        return df
